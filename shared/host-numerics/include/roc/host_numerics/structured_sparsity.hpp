// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <roc/host_numerics/index_order.hpp>
#include <roc/host_numerics/tensor.hpp>
#include <utility>
#include <vector>

namespace roc::host_numerics {
// Selects fixed positions or a reproducible pseudo-random set for each N:M
// group.
enum class StructuredSparsitySelection {
    Fixed,   // Uses fixedPositions for every group.
    Random,  // Selects a sorted set from the seed and group index.
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
    IndexOrder indexOrder = IndexOrder::FirstDimensionFastest;
};

struct StructuredSparsityOutputOptions {
    bool retainedIndices = false;
    bool twoOfFourMetadata = false;
};

// The full-size pruned tensor, compacted retained values, and optional metadata
// produced from one input tensor.
struct StructuredSparseTensor {
    Tensor pruned;                            // Input-shaped output with dropped positions zeroed.
    Tensor compressed;                        // Retained values packed along pattern.axis.
    std::optional<Tensor> retainedIndices;    // UInt8 retained positions per group.
    std::optional<Tensor> twoOfFourMetadata;  // Packed nibbles for a 2:4 pattern.
};

// A slice fixes every coordinate except the sparsity axis. Callers can use
// disjoint ranges to schedule independent portions of one problem.
struct StructuredSparsitySliceRange {
    size_t firstSlice = 0;

    // maximum means every remaining slice.
    size_t sliceCount = std::numeric_limits<size_t>::max();
};

StructuredSparseTensor applyStructuredSparsity(Tensor input, StructuredSparsityPattern pattern,
                                               StructuredSparsityOutputOptions outputOptions = {});

// Writes caller-owned destinations. pruned has input.shape(); compressed has
// inputExtent / groupSize * retainedElements along pattern.axis. Input may
// exactly alias pruned. A slice fixes every coordinate except pattern.axis.
void applyStructuredSparsityInto(Tensor input, StructuredSparseTensor outputs,
                                 StructuredSparsityPattern pattern,
                                 StructuredSparsitySliceRange sliceRange = {});

// Converts two retained UInt8 positions per 2:4 group into packed metadata:
// one nibble per group and two groups per output byte.
Tensor encodeTwoOfFourMetadata(Tensor retainedIndices, size_t axis);
void encodeTwoOfFourMetadataInto(Tensor retainedIndices, Tensor metadata, size_t axis);
}  // namespace roc::host_numerics
