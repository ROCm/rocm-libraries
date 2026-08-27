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

struct StructuredSparsityOutputs {
    bool retainedIndices = false;
    bool twoOfFourMetadata = false;
};

// Reusable pruning/compression descriptor. Output scalar types are fixed by
// the operation: pruned and compressed match input; auxiliary outputs are
// UInt8 when requested.
struct StructuredSparsityProblem {
    StructuredSparsityProblem(Tensor inputTensor, StructuredSparsityPattern sparsityPattern,
                              StructuredSparsityOutputs requestedOutputs = {})
        : input(std::move(inputTensor)),
          pattern(std::move(sparsityPattern)),
          outputs(requestedOutputs) {}

    Tensor input;                       // Source values.
    StructuredSparsityPattern pattern;  // Grouping and retained-position policy.
    StructuredSparsityOutputs outputs;
};

// Binds a structured-sparsity problem to caller-owned destinations. pruned has
// the input shape; compressed has inputExtent / groupSize * retainedElements
// along pattern.axis. Input may alias pruned only with an identical layout.
struct StructuredSparsityRequest : StructuredSparsityProblem {
    StructuredSparsityRequest(Tensor inputTensor, Tensor prunedTensor, Tensor compressedTensor,
                              StructuredSparsityPattern sparsityPattern)
        : StructuredSparsityProblem(std::move(inputTensor), std::move(sparsityPattern)),
          pruned(std::move(prunedTensor)),
          compressed(std::move(compressedTensor)) {}

    StructuredSparsityRequest(Tensor inputTensor, Tensor prunedTensor, Tensor compressedTensor,
                              Tensor retainedIndexTensor, StructuredSparsityPattern sparsityPattern)
        : StructuredSparsityRequest(std::move(inputTensor), std::move(prunedTensor),
                                    std::move(compressedTensor),
                                    std::optional<Tensor>(std::move(retainedIndexTensor)),
                                    std::nullopt, std::move(sparsityPattern)) {}

    StructuredSparsityRequest(Tensor inputTensor, Tensor prunedTensor, Tensor compressedTensor,
                              std::optional<Tensor> retainedIndexTensor,
                              std::optional<Tensor> metadataTensor,
                              StructuredSparsityPattern sparsityPattern)
        : StructuredSparsityProblem(std::move(inputTensor), std::move(sparsityPattern),
                                    {.retainedIndices = retainedIndexTensor.has_value(),
                                     .twoOfFourMetadata = metadataTensor.has_value()}),
          pruned(std::move(prunedTensor)),
          compressed(std::move(compressedTensor)),
          retainedIndices(std::move(retainedIndexTensor)),
          twoOfFourMetadata(std::move(metadataTensor)) {}

    StructuredSparsityRequest(StructuredSparsityProblem problem, Tensor prunedTensor,
                              Tensor compressedTensor,
                              std::optional<Tensor> retainedIndexTensor = std::nullopt,
                              std::optional<Tensor> metadataTensor = std::nullopt)
        : StructuredSparsityProblem(std::move(problem)),
          pruned(std::move(prunedTensor)),
          compressed(std::move(compressedTensor)),
          retainedIndices(std::move(retainedIndexTensor)),
          twoOfFourMetadata(std::move(metadataTensor)) {}

    Tensor pruned;                            // Input-shaped output with dropped positions zeroed.
    Tensor compressed;                        // Retained values packed along pattern.axis.
    std::optional<Tensor> retainedIndices;    // UInt8 retained positions per group.
    std::optional<Tensor> twoOfFourMetadata;  // Packed nibbles for a 2:4 pattern.
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
    TwoOfFourMetadataProblem(Tensor retainedIndexTensor, size_t sparsityAxis)
        : retainedIndices(std::move(retainedIndexTensor)), axis(sparsityAxis) {}

    Tensor retainedIndices;  // UInt8 pairs of increasing retained positions.
    size_t axis = 0;         // Dimension containing retained-position pairs.
};

// Binds retained indices to caller-owned UInt8 metadata storage.
struct TwoOfFourMetadataRequest : TwoOfFourMetadataProblem {
    TwoOfFourMetadataRequest(Tensor retainedIndexTensor, Tensor metadataTensor, size_t sparsityAxis)
        : TwoOfFourMetadataProblem(std::move(retainedIndexTensor), sparsityAxis),
          metadata(std::move(metadataTensor)) {}

    TwoOfFourMetadataRequest(TwoOfFourMetadataProblem problem, Tensor metadataTensor)
        : TwoOfFourMetadataProblem(std::move(problem)), metadata(std::move(metadataTensor)) {}

    Tensor metadata;  // Two four-bit groups per byte; an unused high nibble is zero.
};

// Counts groups encoded and metadata bytes written.
struct TwoOfFourMetadataRunInfo {
    size_t sparsityGroupsEncoded = 0;  // Two retained indices per group.
    size_t metadataBytesWritten = 0;   // ceil(groups per line / 2) per line.
};

struct StructuredSparsityResult {
    Tensor pruned;
    Tensor compressed;
    std::optional<Tensor> retainedIndices;
    std::optional<Tensor> twoOfFourMetadata;
    StructuredSparsityRunInfo runInfo;
};

struct TwoOfFourMetadataResult {
    Tensor metadata;
    TwoOfFourMetadataRunInfo runInfo;
};

StructuredSparsityRunInfo applyStructuredSparsity(const StructuredSparsityRequest& request,
                                                  StructuredSparsitySliceRange sliceRange = {});
StructuredSparsityResult applyStructuredSparsity(const StructuredSparsityProblem& problem);

TwoOfFourMetadataRunInfo encodeTwoOfFourMetadata(const TwoOfFourMetadataRequest& request);
TwoOfFourMetadataResult encodeTwoOfFourMetadata(const TwoOfFourMetadataProblem& problem);
}  // namespace roc::host_validation
