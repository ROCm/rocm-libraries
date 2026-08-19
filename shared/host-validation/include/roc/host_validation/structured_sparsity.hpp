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
    Fixed,   // Uses fixedPositions for every group.
    Random,  // Counter-randomly selects one increasing retained-position set per group.
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

    // Counter-random seed used when selection is Random.
    uint64_t seed = 0;

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

    Tensor input;                           // Source values; may alias pruned with the same layout.
    Tensor pruned;                          // Input-shaped output with dropped positions zeroed.
    Tensor compressed;                      // Retained values packed along pattern.axis.
    std::optional<Tensor> retainedIndices;  // UInt8 retained positions per group.
    std::optional<Tensor> twoOfFourMetadata;  // Packed nibbles for a 2:4 pattern.
    StructuredSparsityPattern pattern;        // Grouping and retained-position policy.
};

// Counts logical work and writes completed by applyStructuredSparsity.
struct StructuredSparsityRunInfo {
    size_t groupsProcessed = 0;            // N:M groups in the requested slice range.
    size_t inputElementsVisited = 0;       // groupsProcessed * groupSize.
    size_t prunedElementsWritten = 0;      // One write per visited input position.
    size_t compressedElementsWritten = 0;  // Retained values written.
    size_t retainedIndicesWritten = 0;     // Zero when retainedIndices is absent.
    size_t metadataBytesWritten = 0;       // Zero when twoOfFourMetadata is absent.
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

    Tensor retainedIndices;  // UInt8 pairs of increasing retained positions.
    Tensor metadata;         // UInt8 output containing two four-bit groups per byte.
    size_t axis = 0;         // Dimension containing retained-position pairs.
};

// Counts groups encoded and metadata bytes written.
struct TwoOfFourMetadataRunInfo {
    size_t sparsityGroupsEncoded = 0;  // Two retained indices per group.
    size_t metadataBytesWritten = 0;   // ceil(groups per line / 2) per line.
};

StructuredSparsityRunInfo applyStructuredSparsity(const StructuredSparsityProblem& problem,
                                                  StructuredSparsitySliceRange sliceRange = {});
TwoOfFourMetadataRunInfo encodeTwoOfFourMetadata(const TwoOfFourMetadataProblem& problem);
}  // namespace roc::host_validation
