// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <roc/host_numerics/structured_sparsity.hpp>
#include <span>
#include <utility>
#include <vector>

namespace roc::host_numerics::detail {
struct StructuredSparsityPlan {
    Shape compressedShape;
    std::optional<Shape> metadataShape;
    size_t groupsPerLine = 0;
    size_t lineCount = 0;
    std::vector<std::vector<size_t>> retainedPositionSets;
};

struct StructuredSparsitySpecification {
    StructuredSparsitySpecification(Tensor inputTensor, StructuredSparsityPattern sparsityPattern,
                                    StructuredSparsityOutputOptions requestedOutputs = {})
        : input(std::move(inputTensor)),
          pattern(std::move(sparsityPattern)),
          outputs(requestedOutputs) {}

    Tensor input;
    StructuredSparsityPattern pattern;
    StructuredSparsityOutputOptions outputs;
};

struct StructuredSparsityInvocation : StructuredSparsitySpecification {
    StructuredSparsityInvocation(StructuredSparsitySpecification specification,
                                 StructuredSparseTensor outputTensors)
        : StructuredSparsitySpecification(std::move(specification)),
          pruned(std::move(outputTensors.pruned)),
          compressed(std::move(outputTensors.compressed)),
          retainedIndices(std::move(outputTensors.retainedIndices)),
          twoOfFourMetadata(std::move(outputTensors.twoOfFourMetadata)) {}

    Tensor pruned;
    Tensor compressed;
    std::optional<Tensor> retainedIndices;
    std::optional<Tensor> twoOfFourMetadata;
};

struct StructuredSparsityWorkCounts {
    size_t groupsProcessed = 0;
    size_t inputElementsVisited = 0;
    size_t prunedElementsWritten = 0;
    size_t compressedElementsWritten = 0;
    size_t retainedIndicesWritten = 0;
    size_t metadataBytesWritten = 0;
};

struct TwoOfFourMetadataArguments {
    Tensor retainedIndices;
    size_t axis = 0;
};

struct TwoOfFourMetadataInvocation : TwoOfFourMetadataArguments {
    TwoOfFourMetadataInvocation(Tensor retainedIndexTensor, Tensor metadataTensor,
                                size_t sparsityAxis)
        : TwoOfFourMetadataArguments{std::move(retainedIndexTensor), sparsityAxis},
          metadata(std::move(metadataTensor)) {}

    Tensor metadata;
};

struct TwoOfFourMetadataWorkCounts {
    size_t sparsityGroupsEncoded = 0;
    size_t metadataBytesWritten = 0;
};

StructuredSparsityPlan validateStructuredSparsityProblem(
    const StructuredSparsitySpecification& problem);
bool tensorStorageOverlaps(const Tensor& left, const Tensor& right);
void rejectTensorStorageOverlap(const Tensor& left, const Tensor& right, const char* message);
StructuredSparsityPlan validateStructuredSparsityRequest(
    const StructuredSparsityInvocation& request);
void coordinatesForLine(size_t line, const Shape& shape, size_t excludedAxis, IndexOrder order,
                        std::vector<size_t>& coordinates);
bool bitAt(std::span<const std::byte> storage, uint64_t bit);
void setBit(std::span<std::byte> storage, uint64_t bit, bool value);
void copyEncodedElement(ScalarType type, std::span<const std::byte> source, ptrdiff_t sourceElement,
                        std::span<std::byte> destination, ptrdiff_t destinationElement);
void zeroEncodedElement(ScalarType type, std::span<std::byte> destination,
                        ptrdiff_t destinationElement);
size_t retainedPositionSetIndexForGroup(const StructuredSparsityInvocation& problem,
                                        const StructuredSparsityPlan& plan,
                                        size_t groupLinearIndex);
const std::vector<size_t>& retainedPositionsForGroup(const StructuredSparsityInvocation& problem,
                                                     const StructuredSparsityPlan& plan,
                                                     size_t groupLinearIndex);
std::pair<size_t, size_t> validateStructuredSparsitySliceRange(const StructuredSparsityPlan& plan,
                                                               StructuredSparsitySliceRange range);
uint8_t twoOfFourMetadataNibble(uint8_t first, uint8_t second);
}  // namespace roc::host_numerics::detail
