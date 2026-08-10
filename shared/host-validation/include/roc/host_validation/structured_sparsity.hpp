// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <roc/host_validation/generation_primitives.hpp>
#include <roc/host_validation/tensor.hpp>
#include <utility>
#include <vector>

namespace roc::host_validation {
enum class StructuredSparsitySelection {
    Fixed,
    Random,
};

struct StructuredSparsityPattern {
    size_t axis = 0;
    size_t groupSize = 4;
    size_t retainedElements = 2;
    StructuredSparsitySelection selection = StructuredSparsitySelection::Fixed;
    std::vector<size_t> fixedPositions{0, 1};
    uint64_t seed = 0;
    uint64_t stream = 0;
    LogicalIndexOrder indexOrder = LogicalIndexOrder::FirstDimensionFastest;
};

struct StructuredSparsityProblem {
    StructuredSparsityProblem(TensorView inputTensor, MutableTensorView prunedTensor,
                              MutableTensorView compressedTensor,
                              StructuredSparsityPattern sparsityPattern)
        : input(std::move(inputTensor)),
          pruned(std::move(prunedTensor)),
          compressed(std::move(compressedTensor)),
          pattern(std::move(sparsityPattern)) {}

    StructuredSparsityProblem(TensorView inputTensor, MutableTensorView prunedTensor,
                              MutableTensorView compressedTensor,
                              MutableTensorView retainedIndexTensor,
                              StructuredSparsityPattern sparsityPattern)
        : StructuredSparsityProblem(std::move(inputTensor), std::move(prunedTensor),
                                    std::move(compressedTensor), std::move(sparsityPattern)) {
        retainedIndices = std::move(retainedIndexTensor);
    }

    TensorView input;
    MutableTensorView pruned;
    MutableTensorView compressed;
    std::optional<MutableTensorView> retainedIndices;
    std::optional<MutableTensorView> twoOfFourMetadata;
    StructuredSparsityPattern pattern;
};

struct StructuredSparsityRunInfo {
    size_t groupsProcessed = 0;
    size_t inputElementsVisited = 0;
    size_t prunedElementsWritten = 0;
    size_t compressedElementsWritten = 0;
    size_t retainedIndicesWritten = 0;
    size_t metadataBytesWritten = 0;
};

struct StructuredSparsitySliceRange {
    size_t firstSlice = 0;
    size_t sliceCount = std::numeric_limits<size_t>::max();
};

struct TwoOfFourMetadataProblem {
    TwoOfFourMetadataProblem(TensorView retainedIndexTensor, MutableTensorView metadataTensor,
                             size_t sparsityAxis)
        : retainedIndices(std::move(retainedIndexTensor)),
          metadata(std::move(metadataTensor)),
          axis(sparsityAxis) {}

    TensorView retainedIndices;
    MutableTensorView metadata;
    size_t axis = 0;
};

struct TwoOfFourMetadataRunInfo {
    size_t sparsityGroupsEncoded = 0;
    size_t metadataBytesWritten = 0;
};

StructuredSparsityRunInfo applyStructuredSparsity(const StructuredSparsityProblem& problem,
                                                  StructuredSparsitySliceRange sliceRange = {});
TwoOfFourMetadataRunInfo encodeTwoOfFourMetadata(const TwoOfFourMetadataProblem& problem);
}  // namespace roc::host_validation
