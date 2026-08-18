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
// Selects whether every N:M group retains fixed positions or a deterministic
// counter-random position set.
enum class StructuredSparsitySelection {
    Fixed,
    Random,
};

// Describes logical N:M pruning along one tensor axis. Each groupSize-element
// group contributes retainedElements encoded values to the compressed output;
// the remaining values are zeroed in the full-size pruned output. This
// operation is independent of MX data generation and GPU-specific layouts.
struct StructuredSparsityPattern {
    // Tensor axis partitioned into N:M groups.
    size_t axis = 0;

    // Group width and retained count. The defaults describe a 2:4 pattern.
    size_t groupSize = 4;
    size_t retainedElements = 2;

    StructuredSparsitySelection selection = StructuredSparsitySelection::Fixed;

    // Positions retained in every group when selection is Fixed.
    std::vector<size_t> fixedPositions{0, 1};

    // Counter-random inputs used when selection is Random.
    uint64_t seed = 0;
    uint64_t stream = 0;

    // Defines how groups on the non-sparsity axes receive linear random indices.
    LogicalIndexOrder indexOrder = LogicalIndexOrder::FirstDimensionFastest;
};

// Binds caller-owned tensors to a pruning/compression operation. pruned has
// the input shape, while compressed has sparsity-axis extent
// inputExtent / groupSize * retainedElements. Optional outputs describe
// retained positions or packed 2:4 metadata.
struct StructuredSparsityProblem {
    StructuredSparsityProblem(Tensor inputTensor, Tensor prunedTensor, Tensor compressedTensor,
                              StructuredSparsityPattern sparsityPattern)
        : input(std::move(inputTensor)),
          pruned(std::move(prunedTensor)),
          compressed(std::move(compressedTensor)),
          pattern(std::move(sparsityPattern)) {}

    StructuredSparsityProblem(Tensor inputTensor, Tensor prunedTensor, Tensor compressedTensor,
                              Tensor retainedIndexTensor, StructuredSparsityPattern sparsityPattern)
        : StructuredSparsityProblem(std::move(inputTensor), std::move(prunedTensor),
                                    std::move(compressedTensor), std::move(sparsityPattern)) {
        retainedIndices = std::move(retainedIndexTensor);
    }

    Tensor input;
    Tensor pruned;
    Tensor compressed;
    std::optional<Tensor> retainedIndices;
    std::optional<Tensor> twoOfFourMetadata;
    StructuredSparsityPattern pattern;
};

// Counts logical work and writes completed by applyStructuredSparsity.
struct StructuredSparsityRunInfo {
    size_t groupsProcessed = 0;
    size_t inputElementsVisited = 0;
    size_t prunedElementsWritten = 0;
    size_t compressedElementsWritten = 0;
    size_t retainedIndicesWritten = 0;
    size_t metadataBytesWritten = 0;
};

// A slice fixes every coordinate except the sparsity axis. Callers can use
// disjoint ranges to schedule independent portions of one problem.
struct StructuredSparsitySliceRange {
    size_t firstSlice = 0;

    // maximum means every remaining slice.
    size_t sliceCount = std::numeric_limits<size_t>::max();
};

// Converts two retained UInt8 positions per 2:4 group into packed metadata:
// one nibble per group and two groups per output byte.
struct TwoOfFourMetadataProblem {
    TwoOfFourMetadataProblem(Tensor retainedIndexTensor, Tensor metadataTensor, size_t sparsityAxis)
        : retainedIndices(std::move(retainedIndexTensor)),
          metadata(std::move(metadataTensor)),
          axis(sparsityAxis) {}

    Tensor retainedIndices;
    Tensor metadata;
    size_t axis = 0;
};

// Counts groups encoded and metadata bytes written.
struct TwoOfFourMetadataRunInfo {
    size_t sparsityGroupsEncoded = 0;
    size_t metadataBytesWritten = 0;
};

StructuredSparsityRunInfo applyStructuredSparsity(const StructuredSparsityProblem& problem,
                                                  StructuredSparsitySliceRange sliceRange = {});
TwoOfFourMetadataRunInfo encodeTwoOfFourMetadata(const TwoOfFourMetadataProblem& problem);
}  // namespace roc::host_validation
