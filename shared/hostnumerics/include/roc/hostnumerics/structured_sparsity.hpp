// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <roc/hostnumerics/index_order.hpp>
#include <roc/hostnumerics/tensor.hpp>
#include <utility>
#include <variant>
#include <vector>

namespace roc::hostnumerics {
// Retains the same positions, each less than the group size, in every group.
struct FixedStructuredSparsitySelection {
    std::vector<size_t> positions;
};

// Retains a pseudo-random sorted set in each group, chosen from the seed and
// the group's linear index.
struct RandomStructuredSparsitySelection {
    uint64_t seed;
};

using StructuredSparsitySelection =
    std::variant<FixedStructuredSparsitySelection, RandomStructuredSparsitySelection>;

// Describes logical N:M pruning along one tensor axis. Each groupSize-element
// group contributes retainedElements encoded values to the compressed output;
// the remaining values are zeroed in the full-size pruned output. This
// operation is independent of MX data generation and GPU-specific layouts.
// Every field is required; a 2:4 pattern sets groupSize 4 and
// retainedElements 2.
struct StructuredSparsityPattern {
    // Tensor axis partitioned into N:M groups.
    size_t axis;

    size_t groupSize;
    size_t retainedElements;

    StructuredSparsitySelection selection;

    // Numbers the lines, which fix every coordinate except axis. Slice ranges
    // and random group indices follow this numbering.
    IndexOrder indexOrder;
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

    // Without a value, the range extends through the last slice.
    std::optional<size_t> sliceCount;
};

StructuredSparseTensor applyStructuredSparsity(Tensor input, StructuredSparsityPattern pattern,
                                               StructuredSparsityOutputOptions outputOptions);

// Writes caller-owned destinations. pruned has input.shape(); compressed has
// inputExtent / groupSize * retainedElements along pattern.axis. Input may
// exactly alias pruned. A slice fixes every coordinate except pattern.axis.
void applyStructuredSparsityInto(Tensor input, StructuredSparseTensor outputs,
                                 StructuredSparsityPattern pattern,
                                 StructuredSparsitySliceRange sliceRange);

// Converts two retained UInt8 positions per 2:4 group into packed metadata:
// one nibble per group and two groups per output byte.
Tensor encodeTwoOfFourMetadata(Tensor retainedIndices, size_t axis);
void encodeTwoOfFourMetadataInto(Tensor retainedIndices, Tensor metadata, size_t axis);
}  // namespace roc::hostnumerics
