/*******************************************************************************
 * Copyright (C) 2026 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "common/post_processor/converters/to_tensor/zeroshot_openclip.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

using namespace InferenceBackend;
using namespace post_processing;

namespace {

class TestOutputBlob : public OutputBlob {
  public:
    TestOutputBlob(std::vector<float> data, std::vector<size_t> dims) : data_(std::move(data)), dims_(std::move(dims)) {
    }

    const std::vector<size_t> &GetDims() const override {
        return dims_;
    }

    const void *GetData() const override {
        return data_.data();
    }

    Layout GetLayout() const override {
        return Layout::NC;
    }

    Precision GetPrecision() const override {
        return Precision::FP32;
    }

  private:
    std::vector<float> data_;
    std::vector<size_t> dims_;
};

bool hasPythonTorch() {
    const int status = std::system("python3 -c \"import torch\" > /dev/null 2>&1");
    return status == 0;
}

std::string shellQuote(const std::string &arg) {
    std::string escaped;
    escaped.reserve(arg.size() + 2);
    escaped.push_back('\'');
    for (char c : arg) {
        if (c == '\'')
            escaped += "'\\''";
        else
            escaped.push_back(c);
    }
    escaped.push_back('\'');
    return escaped;
}

std::filesystem::path createTempEmbeddingsPath(const std::string &test_name) {
    const auto dir = std::filesystem::temp_directory_path();
    return dir / ("dlstreamer_zeroshot_" + test_name + ".pth");
}

void writeEmbeddingsPth(const std::filesystem::path &path, const std::string &python_tensor_expr) {
    const std::string cmd =
        "python3 -c \"import torch; t=" + python_tensor_expr + "; torch.save(t, " + shellQuote(path.string()) +
        ")\" > /dev/null 2>&1";
    const int status = std::system(cmd.c_str());
    ASSERT_EQ(status, 0);
}

BlobToMetaConverter::Initializer createInitializer(const std::filesystem::path &pth_path, uint32_t topk,
                                                   std::vector<std::string> labels) {
    BlobToMetaConverter::Initializer initializer;
    initializer.model_name = "zeroshot-openclip-test";
    initializer.outputs_info = {{"output", {1, 2}}};
    initializer.input_image_info.batch_size = 1;
    initializer.input_image_info.width = 224;
    initializer.input_image_info.height = 224;
    initializer.model_proc_output_info.reset(gst_structure_new_empty("ANY"));
    initializer.labels = std::move(labels);
    initializer.zeroshot_embeddings_file = pth_path.string();
    initializer.zeroshot_topk = topk;
    return initializer;
}

} // namespace

TEST(ZeroShotOpenCLIPConverter, ThrowsIfEmbeddingsPathInvalid) {
    if (!hasPythonTorch())
        GTEST_SKIP() << "python3/torch is not available in test environment";

    auto initializer = createInitializer("/tmp/dlstreamer_nonexistent_embeddings.pth", 1, {"a", "b"});
    EXPECT_THROW(ZeroShotOpenCLIPConverter(std::move(initializer)), std::exception);
}

TEST(ZeroShotOpenCLIPConverter, ThrowsIfLabelsCountDoesNotMatchEmbeddingsRows) {
    if (!hasPythonTorch())
        GTEST_SKIP() << "python3/torch is not available in test environment";

    const auto pth_path = createTempEmbeddingsPath("label_mismatch");
    writeEmbeddingsPth(pth_path, "torch.tensor([[1.0,0.0],[0.0,1.0],[0.7,0.7]], dtype=torch.float32)");

    auto initializer = createInitializer(pth_path, 2, {"alpha", "beta"});
    EXPECT_THROW(ZeroShotOpenCLIPConverter(std::move(initializer)), std::exception);

    std::filesystem::remove(pth_path);
}

TEST(ZeroShotOpenCLIPConverter, ReturnsTopKRankedLabels) {
    if (!hasPythonTorch())
        GTEST_SKIP() << "python3/torch is not available in test environment";

    const auto pth_path = createTempEmbeddingsPath("topk");
    writeEmbeddingsPth(pth_path, "torch.tensor([[1.0,0.0],[0.0,1.0],[0.7,0.7]], dtype=torch.float32)");

    auto initializer = createInitializer(pth_path, 2, {"alpha", "beta", "gamma"});
    ZeroShotOpenCLIPConverter converter(std::move(initializer));

    auto blob = std::make_shared<TestOutputBlob>(std::vector<float>{1.0F, 0.0F}, std::vector<size_t>{1, 2});
    OutputBlobs output_blobs{{"output", blob}};

    auto tensors_table = converter.convert(output_blobs);
    ASSERT_EQ(tensors_table.size(), 1U);
    ASSERT_EQ(tensors_table[0].size(), 2U);
    ASSERT_EQ(tensors_table[0][0].size(), 1U);
    ASSERT_EQ(tensors_table[0][1].size(), 1U);

    GstStructure *rank1 = tensors_table[0][0][0];
    GstStructure *rank2 = tensors_table[0][1][0];

    ASSERT_STREQ(gst_structure_get_string(rank1, "label"), "alpha");
    ASSERT_STREQ(gst_structure_get_string(rank2, "label"), "gamma");

    int rank1_value = 0;
    int rank2_value = 0;
    ASSERT_TRUE(gst_structure_get_int(rank1, "rank", &rank1_value));
    ASSERT_TRUE(gst_structure_get_int(rank2, "rank", &rank2_value));
    EXPECT_EQ(rank1_value, 1);
    EXPECT_EQ(rank2_value, 2);

    double confidence1 = 0.0;
    double confidence2 = 0.0;
    ASSERT_TRUE(gst_structure_get_double(rank1, "confidence", &confidence1));
    ASSERT_TRUE(gst_structure_get_double(rank2, "confidence", &confidence2));
    EXPECT_GT(confidence1, confidence2);

    std::filesystem::remove(pth_path);
}
