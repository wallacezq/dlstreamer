/*******************************************************************************
 * Copyright (C) 2026 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#pragma once

#include "blob_to_tensor_converter.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace post_processing {

class ZeroShotOpenCLIPConverter : public BlobToTensorConverter {
  public:
    explicit ZeroShotOpenCLIPConverter(BlobToMetaConverter::Initializer initializer);

    TensorsTable convert(const OutputBlobs &output_blobs) override;

    static std::string getName() {
        return "zeroshot_openclip";
    }

  private:
    std::vector<float> embeddings_;
    size_t num_classes_ = 0;
    size_t embedding_dim_ = 0;
    uint32_t topk_ = 1;

    void loadEmbeddingsFromPth(const std::string &embeddings_path);
    std::vector<float> computeLogits(const float *image_embedding, size_t image_embedding_size) const;
    std::vector<float> computeProbabilities(const std::vector<float> &logits) const;
};

} // namespace post_processing
