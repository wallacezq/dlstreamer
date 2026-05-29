/*******************************************************************************
 * Copyright (C) 2026 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "zeroshot_openclip.h"

#include "copy_blob_to_gststruct.h"
#include "inference_backend/logger.h"
#include "safe_arithmetic.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#define popen _popen
#define pclose _pclose
#else
#include <unistd.h>
#endif

namespace post_processing {
namespace {

std::string shellQuote(const std::string &arg) {
    std::string escaped;
    escaped.reserve(arg.size() + 2);
    escaped.push_back('\'');
    for (char c : arg) {
        if (c == '\'') {
            escaped += "'\\''";
        } else {
            escaped.push_back(c);
        }
    }
    escaped.push_back('\'');
    return escaped;
}

std::string createTempFilePath() {
#ifdef _WIN32
    char tmp_name[L_tmpnam] = {0};
    if (!std::tmpnam(tmp_name))
        throw std::runtime_error("Failed to create temporary filename for zero-shot embeddings.");
    return std::string(tmp_name);
#else
    char tmp_template[] = "/tmp/dls_zeroshot_embeddings_XXXXXX";
    const int fd = mkstemp(tmp_template);
    if (fd < 0)
        throw std::runtime_error("Failed to create temporary file for zero-shot embeddings.");
    close(fd);
    return std::string(tmp_template);
#endif
}

std::string runCommandAndCapture(const std::string &command) {
    FILE *pipe = popen(command.c_str(), "r");
    if (!pipe)
        throw std::runtime_error("Failed to execute command: " + command);

    std::string output;
    char buffer[512] = {0};
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }

    const int status = pclose(pipe);
    if (status != 0)
        throw std::runtime_error("Command failed with status " + std::to_string(status) + ": " + output);

    return output;
}

std::vector<float> normalizeVector(const float *src, size_t size) {
    if (size == 0)
        throw std::invalid_argument("Embedding vector size is zero.");

    double norm_sq = 0.0;
    for (size_t i = 0; i < size; ++i)
        norm_sq += static_cast<double>(src[i]) * static_cast<double>(src[i]);

    const double norm = std::sqrt(norm_sq);
    if (norm <= std::numeric_limits<double>::epsilon())
        throw std::runtime_error("Embedding vector norm is zero.");

    std::vector<float> normalized(size);
    for (size_t i = 0; i < size; ++i)
        normalized[i] = static_cast<float>(src[i] / norm);

    return normalized;
}

} // namespace

ZeroShotOpenCLIPConverter::ZeroShotOpenCLIPConverter(BlobToMetaConverter::Initializer initializer)
    : BlobToTensorConverter(std::move(initializer)) {
    topk_ = std::max<uint32_t>(1, getZeroshotTopk());

    const std::string &embeddings_path = getZeroshotEmbeddingsFile();
    if (embeddings_path.empty())
        throw std::invalid_argument("Zero-shot embeddings file is not set. Use gvaclassify zeroshot-embeddings-file.");

    loadEmbeddingsFromPth(embeddings_path);

    const auto &labels = getLabels();
    if (!labels.empty() && labels.size() != num_classes_) {
        throw std::invalid_argument("Number of labels (" + std::to_string(labels.size()) +
                                    ") does not match number of embeddings (" + std::to_string(num_classes_) + ").");
    }
}

void ZeroShotOpenCLIPConverter::loadEmbeddingsFromPth(const std::string &embeddings_path) {
    const std::string output_path = createTempFilePath();

    // Load tensor from .pth and dump as simple text matrix to avoid binary format coupling in C++ runtime.
    const std::string py_script =
        "import sys, torch\n"
        "obj = torch.load(sys.argv[1], map_location='cpu')\n"
        "if isinstance(obj, dict):\n"
        "    t = None\n"
        "    for k in ('embeddings', 'label_embeddings', 'E_text', 'text_embeddings'):\n"
        "        if k in obj:\n"
        "            t = obj[k]\n"
        "            break\n"
        "    if t is None:\n"
        "        raise RuntimeError('Unable to find embeddings tensor in dict payload')\n"
        "else:\n"
        "    t = obj\n"
        "if not torch.is_tensor(t):\n"
        "    raise RuntimeError('Loaded object is not a tensor')\n"
        "if t.ndim == 1:\n"
        "    t = t.unsqueeze(0)\n"
        "if t.ndim != 2:\n"
        "    t = t.reshape(t.shape[0], -1)\n"
        "t = t.detach().to(dtype=torch.float32, device='cpu').contiguous()\n"
        "with open(sys.argv[2], 'w', encoding='utf-8') as f:\n"
        "    f.write(f'{t.shape[0]} {t.shape[1]}\\n')\n"
        "    for row in t.tolist():\n"
        "        f.write(' '.join(str(float(v)) for v in row))\n"
        "        f.write('\\n')\n";

    const std::string command = "python3 -c " + shellQuote(py_script) + " " + shellQuote(embeddings_path) + " " +
                                shellQuote(output_path) + " 2>&1";

    try {
        runCommandAndCapture(command);
    } catch (...) {
        std::remove(output_path.c_str());
        throw;
    }

    std::ifstream file(output_path);
    std::remove(output_path.c_str());
    if (!file.is_open())
        throw std::runtime_error("Failed to open converted embeddings data from " + output_path);

    size_t rows = 0;
    size_t cols = 0;
    file >> rows >> cols;
    if (!file.good() || rows == 0 || cols == 0)
        throw std::runtime_error("Invalid embeddings dimensions in converted .pth payload.");

    num_classes_ = rows;
    embedding_dim_ = cols;
    embeddings_.resize(num_classes_ * embedding_dim_);

    for (size_t i = 0; i < embeddings_.size(); ++i) {
        file >> embeddings_[i];
        if (!file.good())
            throw std::runtime_error("Unexpected end of embeddings data while parsing converted .pth payload.");
    }

    for (size_t row = 0; row < num_classes_; ++row) {
        float *row_ptr = embeddings_.data() + row * embedding_dim_;
        const auto normalized = normalizeVector(row_ptr, embedding_dim_);
        std::copy(normalized.begin(), normalized.end(), row_ptr);
    }
}

std::vector<float> ZeroShotOpenCLIPConverter::computeLogits(const float *image_embedding, size_t image_embedding_size) const {
    if (image_embedding_size < embedding_dim_) {
        throw std::runtime_error("Image embedding dimension " + std::to_string(image_embedding_size) +
                                 " is smaller than label embedding dimension " + std::to_string(embedding_dim_) + ".");
    }

    if (image_embedding_size > embedding_dim_) {
        GVA_WARNING("Image embedding has %zu values, but embeddings file expects %zu. Using the first %zu values.",
                    image_embedding_size, embedding_dim_, embedding_dim_);
    }

    const auto normalized_image = normalizeVector(image_embedding, embedding_dim_);
    std::vector<float> logits(num_classes_, 0.0F);

    for (size_t row = 0; row < num_classes_; ++row) {
        const float *class_embedding = embeddings_.data() + row * embedding_dim_;
        float dot = 0.0F;
        for (size_t col = 0; col < embedding_dim_; ++col)
            dot += normalized_image[col] * class_embedding[col];
        logits[row] = dot;
    }

    return logits;
}

std::vector<float> ZeroShotOpenCLIPConverter::computeProbabilities(const std::vector<float> &logits) const {
    if (logits.empty())
        throw std::runtime_error("No logits produced for zero-shot classification.");

    const auto max_it = std::max_element(logits.begin(), logits.end());
    const float max_logit = *max_it;

    std::vector<float> probabilities(logits.size(), 0.0F);
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        probabilities[i] = std::exp(logits[i] - max_logit);
        sum += probabilities[i];
    }

    if (sum <= std::numeric_limits<double>::epsilon())
        throw std::runtime_error("Softmax normalization sum is zero.");

    for (float &probability : probabilities)
        probability = static_cast<float>(probability / sum);

    return probabilities;
}

TensorsTable ZeroShotOpenCLIPConverter::convert(const OutputBlobs &output_blobs) {
    if (output_blobs.empty())
        throw std::invalid_argument("Output blobs are empty.");

    if (output_blobs.size() != 1) {
        throw std::runtime_error("Zero-shot converter expects exactly one output blob, got " +
                                 std::to_string(output_blobs.size()) + ".");
    }

    const auto &blob_it = *output_blobs.begin();
    const std::string &layer_name = blob_it.first;
    InferenceBackend::OutputBlob::Ptr blob = blob_it.second;
    if (!blob)
        throw std::invalid_argument("Output blob is empty.");
    if (blob->GetData() == nullptr)
        throw std::invalid_argument("Output blob data is nullptr.");

    const size_t batch_size = getModelInputImageInfo().batch_size;
    TensorsTable tensors_table(batch_size);

    std::vector<float> blob_data_fp32;
    const float *blob_data = nullptr;

    if (blob->GetPrecision() == InferenceBackend::Blob::Precision::FP32) {
        blob_data = reinterpret_cast<const float *>(blob->GetData());
    } else if (blob->GetPrecision() == InferenceBackend::Blob::Precision::FP64) {
        const auto *blob_data_fp64 = reinterpret_cast<const double *>(blob->GetData());
        blob_data_fp32.resize(blob->GetSize());
        for (size_t i = 0; i < blob->GetSize(); ++i)
            blob_data_fp32[i] = static_cast<float>(blob_data_fp64[i]);
        blob_data = blob_data_fp32.data();
    } else {
        throw std::runtime_error("Zero-shot converter supports FP32/FP64 output only.");
    }

    const auto &labels = getLabels();
    const uint32_t actual_topk = static_cast<uint32_t>(std::min<size_t>(topk_, num_classes_));

    for (size_t frame_index = 0; frame_index < batch_size; ++frame_index) {
        const auto item = get_data_by_batch_index<float>(blob_data, blob->GetSize(), batch_size, frame_index);
        const float *item_data = item.first;
        const size_t item_data_size = item.second;

        const auto logits = computeLogits(item_data, item_data_size);
        const auto probabilities = computeProbabilities(logits);

        std::vector<size_t> indices(num_classes_);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + actual_topk, indices.end(),
                          [&](size_t lhs, size_t rhs) { return logits[lhs] > logits[rhs]; });

        std::vector<GstStructure *> tensors;
        tensors.reserve(actual_topk);

        for (uint32_t rank = 0; rank < actual_topk; ++rank) {
            const size_t class_id = indices[rank];
            GVA::Tensor classification_result = createTensor();

            if (!skipRawTensors() && rank == 0) {
                CopyOutputBlobToGstStructure(blob, classification_result.gst_structure(),
                                             BlobToMetaConverter::getModelName().c_str(), layer_name.c_str(),
                                             batch_size, frame_index);
            }

            const std::string label = labels.empty() ? std::to_string(class_id) : labels.at(class_id);
            classification_result.set_string("label", label);
            classification_result.set_int("label_id", class_id);
            classification_result.set_double("confidence", probabilities[class_id]);
            classification_result.set_int("rank", rank + 1);

            gst_structure_set(classification_result.gst_structure(), "tensor_id", G_TYPE_INT,
                              safe_convert<int>(frame_index), "type", G_TYPE_STRING, GVA::GST_ANALYTICS_CLS_2_TENSOR,
                              NULL);

            tensors.push_back(classification_result.gst_structure());
        }

        tensors_table[frame_index].push_back(tensors);
    }

    return tensors_table;
}

} // namespace post_processing
