// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "detail/structured_sparsity_plan.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <roc/host_numerics/structured_sparsity.hpp>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "detail/generation_primitives.hpp"
#include "detail/tensor_storage.hpp"
#include "detail/threading.hpp"

namespace roc::host_numerics::detail {
namespace {
void enumerateRetainedPositionSets(size_t groupSize, size_t retainedElements, size_t nextPosition,
                                   std::vector<size_t>& current,
                                   std::vector<std::vector<size_t>>& result) {
    if (current.size() == retainedElements) {
        result.push_back(current);
        if (result.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw std::overflow_error("Structured sparsity has too many retained-position sets.");
        return;
    }

    const size_t remaining = retainedElements - current.size();
    for (size_t position = nextPosition; position + remaining <= groupSize; ++position) {
        current.push_back(position);
        enumerateRetainedPositionSets(groupSize, retainedElements, position + 1, current, result);
        current.pop_back();
    }
}
}  // namespace

StructuredSparsityPlan validateStructuredSparsityProblem(
    const StructuredSparsitySpecification& problem) {
    const Shape& inputShape = problem.input.shape();
    if (inputShape.rank() == 0)
        throw std::invalid_argument("Structured sparsity requires a rank-one or higher tensor.");
    if (problem.pattern.axis >= inputShape.rank())
        throw std::out_of_range("Structured sparsity axis exceeds tensor rank.");
    if (problem.pattern.groupSize == 0)
        throw std::invalid_argument("Structured sparsity group size must be nonzero.");
    if (problem.pattern.retainedElements == 0 ||
        problem.pattern.retainedElements > problem.pattern.groupSize)
        throw std::invalid_argument(
            "Structured sparsity retained-element count must be in [1, group size].");
    if (problem.outputs.retainedIndices && problem.pattern.groupSize > 256)
        throw std::invalid_argument(
            "Structured sparsity UInt8 retained indices require group size at most 256.");
    if (inputShape[problem.pattern.axis] % problem.pattern.groupSize != 0)
        throw std::invalid_argument(
            "Structured sparsity axis extent must be divisible by group size.");
    if (scalarTypeInfo(problem.input.type()).category == ScalarCategory::Scale)
        throw std::invalid_argument(
            "Structured sparsity does not accept scale-only scalar encodings.");
    if (problem.pattern.selection != StructuredSparsitySelection::Fixed &&
        problem.pattern.selection != StructuredSparsitySelection::Random)
        throw std::invalid_argument("Structured sparsity selection is invalid.");
    if (problem.pattern.indexOrder != IndexOrder::FirstDimensionFastest &&
        problem.pattern.indexOrder != IndexOrder::LastDimensionFastest)
        throw std::invalid_argument("Structured sparsity index order is invalid.");

    std::vector<size_t> compressedDimensions(inputShape.dimensions().begin(),
                                             inputShape.dimensions().end());
    const size_t groupsPerLine = inputShape[problem.pattern.axis] / problem.pattern.groupSize;
    if (groupsPerLine > std::numeric_limits<size_t>::max() / problem.pattern.retainedElements)
        throw std::overflow_error("Structured sparsity compressed axis extent overflows.");
    compressedDimensions[problem.pattern.axis] = groupsPerLine * problem.pattern.retainedElements;
    const Shape compressedShape(std::move(compressedDimensions));

    std::optional<Shape> metadataShape;
    if (problem.outputs.twoOfFourMetadata) {
        if (problem.pattern.groupSize != 4 || problem.pattern.retainedElements != 2)
            throw std::invalid_argument(
                "Two-of-four metadata output requires a two-of-four sparsity pattern.");
        std::vector<size_t> metadataDimensions(inputShape.dimensions().begin(),
                                               inputShape.dimensions().end());
        metadataDimensions[problem.pattern.axis] = (groupsPerLine + 1) / 2;
        metadataShape.emplace(std::move(metadataDimensions));
    }

    StructuredSparsityPlan plan;
    plan.compressedShape = compressedShape;
    plan.metadataShape = std::move(metadataShape);
    plan.groupsPerLine = groupsPerLine;
    plan.lineCount = inputShape.elementCountExcluding(problem.pattern.axis);

    if (problem.pattern.selection == StructuredSparsitySelection::Fixed) {
        if (problem.pattern.fixedPositions.size() != problem.pattern.retainedElements)
            throw std::invalid_argument(
                "Structured sparsity fixed-position count does not match retained elements.");
        std::vector<size_t> positions = problem.pattern.fixedPositions;
        std::sort(positions.begin(), positions.end());
        if (std::adjacent_find(positions.begin(), positions.end()) != positions.end())
            throw std::invalid_argument("Structured sparsity fixed positions must be unique.");
        if (positions.back() >= problem.pattern.groupSize)
            throw std::out_of_range("Structured sparsity fixed position exceeds group size.");
        plan.retainedPositionSets.push_back(std::move(positions));
    } else {
        std::vector<size_t> current;
        current.reserve(problem.pattern.retainedElements);
        enumerateRetainedPositionSets(problem.pattern.groupSize, problem.pattern.retainedElements,
                                      0, current, plan.retainedPositionSets);
        std::sort(plan.retainedPositionSets.begin(), plan.retainedPositionSets.end(),
                  [](const std::vector<size_t>& left, const std::vector<size_t>& right) {
                      return std::lexicographical_compare(left.rbegin(), left.rend(),
                                                          right.rbegin(), right.rend());
                  });
    }

    return plan;
}

bool tensorStorageOverlaps(const Tensor& left, const Tensor& right) {
    return byteRangesOverlap(left.rawEncodedBackingStorage(), right.rawEncodedBackingStorage());
}

void rejectTensorStorageOverlap(const Tensor& left, const Tensor& right, const char* message) {
    if (tensorStorageOverlaps(left, right)) throw std::invalid_argument(message);
}

StructuredSparsityPlan validateStructuredSparsityRequest(
    const StructuredSparsityInvocation& request) {
    StructuredSparsityPlan plan = validateStructuredSparsityProblem(request);
    if (request.pruned.type() != request.input.type() ||
        request.compressed.type() != request.input.type())
        throw std::invalid_argument(
            "Structured sparsity input, pruned, and compressed scalar types must match.");
    if (request.pruned.shape() != request.input.shape())
        throw std::invalid_argument("Structured sparsity pruned tensor shape mismatch.");
    if (request.compressed.shape() != plan.compressedShape)
        throw std::invalid_argument("Structured sparsity compressed tensor shape mismatch.");
    if (request.retainedIndices.has_value() != request.outputs.retainedIndices)
        throw std::invalid_argument(
            "Structured sparsity retained-index destination does not match the problem.");
    if (request.twoOfFourMetadata.has_value() != request.outputs.twoOfFourMetadata)
        throw std::invalid_argument(
            "Structured sparsity metadata destination does not match the problem.");
    if (request.retainedIndices) {
        if (request.retainedIndices->type() != ScalarType::UInt8)
            throw std::invalid_argument("Structured sparsity retained indices must use UInt8.");
        if (request.retainedIndices->shape() != plan.compressedShape)
            throw std::invalid_argument(
                "Structured sparsity retained-index tensor shape mismatch.");
    }
    if (request.twoOfFourMetadata) {
        if (request.twoOfFourMetadata->type() != ScalarType::UInt8)
            throw std::invalid_argument("Two-of-four metadata output must use UInt8.");
        if (request.twoOfFourMetadata->shape() != *plan.metadataShape)
            throw std::invalid_argument("Structured sparsity two-of-four metadata shape mismatch.");
    }
    const bool inputPrunedOverlap = tensorStorageOverlaps(request.input, request.pruned);
    const bool exactInPlace = request.input.rawEncodedBackingStorage().data() ==
                                  request.pruned.rawEncodedBackingStorage().data() &&
                              request.input.layout() == request.pruned.layout();
    if (inputPrunedOverlap && !exactInPlace)
        throw std::invalid_argument(
            "In-place structured sparsity requires identical input and pruned layouts.");
    rejectTensorStorageOverlap(request.input, request.compressed,
                               "Structured sparsity compressed output overlaps the input storage.");
    rejectTensorStorageOverlap(
        request.pruned, request.compressed,
        "Structured sparsity compressed output overlaps the pruned storage.");
    if (request.retainedIndices) {
        rejectTensorStorageOverlap(
            request.input, *request.retainedIndices,
            "Structured sparsity retained indices overlap the input storage.");
        rejectTensorStorageOverlap(
            request.pruned, *request.retainedIndices,
            "Structured sparsity retained indices overlap the pruned storage.");
        rejectTensorStorageOverlap(
            request.compressed, *request.retainedIndices,
            "Structured sparsity retained indices overlap the compressed storage.");
    }
    if (request.twoOfFourMetadata) {
        rejectTensorStorageOverlap(request.input, *request.twoOfFourMetadata,
                                   "Structured sparsity metadata overlaps the input storage.");
        rejectTensorStorageOverlap(request.pruned, *request.twoOfFourMetadata,
                                   "Structured sparsity metadata overlaps the pruned storage.");
        rejectTensorStorageOverlap(request.compressed, *request.twoOfFourMetadata,
                                   "Structured sparsity metadata overlaps the compressed storage.");
        if (request.retainedIndices)
            rejectTensorStorageOverlap(
                *request.retainedIndices, *request.twoOfFourMetadata,
                "Structured sparsity metadata overlaps the retained-index storage.");
    }
    return plan;
}

void coordinatesForLine(size_t line, const Shape& shape, size_t excludedAxis, IndexOrder order,
                        std::vector<size_t>& coordinates) {
    coordinates.assign(shape.rank(), 0);
    if (order == IndexOrder::FirstDimensionFastest) {
        for (size_t dimension = 0; dimension < shape.rank(); ++dimension) {
            if (dimension == excludedAxis) continue;
            coordinates[dimension] = line % shape[dimension];
            line /= shape[dimension];
        }
    } else {
        for (size_t dimension = shape.rank(); dimension > 0; --dimension) {
            const size_t index = dimension - 1;
            if (index == excludedAxis) continue;
            coordinates[index] = line % shape[index];
            line /= shape[index];
        }
    }
}

bool bitAt(std::span<const std::byte> storage, uint64_t bit) {
    const size_t byteIndex = static_cast<size_t>(bit / 8);
    const uint8_t byte = static_cast<uint8_t>(storage[byteIndex]);
    return ((byte >> (bit % 8)) & 1U) != 0;
}

void setBit(std::span<std::byte> storage, uint64_t bit, bool value) {
    const size_t byteIndex = static_cast<size_t>(bit / 8);
    const uint8_t mask = static_cast<uint8_t>(1U << (bit % 8));
    uint8_t& byte = reinterpret_cast<uint8_t&>(storage[byteIndex]);
    byte = value ? static_cast<uint8_t>(byte | mask)
                 : static_cast<uint8_t>(byte & static_cast<uint8_t>(~mask));
}

void copyEncodedElement(ScalarType type, std::span<const std::byte> source, ptrdiff_t sourceElement,
                        std::span<std::byte> destination, ptrdiff_t destinationElement) {
    const uint64_t bits = scalarTypeInfo(type).storageBits;
    if (sourceElement < 0 || destinationElement < 0)
        throw std::out_of_range("Structured sparsity element offset is negative.");

    const uint64_t sourceBit = static_cast<uint64_t>(sourceElement) * bits;
    const uint64_t destinationBit = static_cast<uint64_t>(destinationElement) * bits;
    if (bits % 8 == 0) {
        const size_t bytes = static_cast<size_t>(bits / 8);
        std::memcpy(destination.data() + destinationBit / 8, source.data() + sourceBit / 8, bytes);
        return;
    }

    for (uint64_t bit = 0; bit < bits; ++bit)
        setBit(destination, destinationBit + bit, bitAt(source, sourceBit + bit));
}

void zeroEncodedElement(ScalarType type, std::span<std::byte> destination,
                        ptrdiff_t destinationElement) {
    const uint64_t bits = scalarTypeInfo(type).storageBits;
    if (destinationElement < 0)
        throw std::out_of_range("Structured sparsity element offset is negative.");
    const uint64_t destinationBit = static_cast<uint64_t>(destinationElement) * bits;
    if (bits % 8 == 0) {
        std::memset(destination.data() + destinationBit / 8, 0, static_cast<size_t>(bits / 8));
        return;
    }
    for (uint64_t bit = 0; bit < bits; ++bit) setBit(destination, destinationBit + bit, false);
}

size_t retainedPositionSetIndexForGroup(const StructuredSparsityInvocation& problem,
                                        const StructuredSparsityPlan& plan,
                                        size_t groupLinearIndex) {
    if (problem.pattern.selection == StructuredSparsitySelection::Fixed) return 0;
    constexpr uint64_t selectionDomain = 0;
    const int selected =
        indexedUniformInteger(problem.pattern.seed, selectionDomain, groupLinearIndex, 0,
                              static_cast<int>(plan.retainedPositionSets.size()) - 1);
    return static_cast<size_t>(selected);
}

const std::vector<size_t>& retainedPositionsForGroup(const StructuredSparsityInvocation& problem,
                                                     const StructuredSparsityPlan& plan,
                                                     size_t groupLinearIndex) {
    return plan
        .retainedPositionSets[retainedPositionSetIndexForGroup(problem, plan, groupLinearIndex)];
}

std::pair<size_t, size_t> validateStructuredSparsitySliceRange(const StructuredSparsityPlan& plan,
                                                               StructuredSparsitySliceRange range) {
    if (range.firstSlice > plan.lineCount)
        throw std::out_of_range(
            "Structured sparsity first slice exceeds the available slice count.");
    if (range.sliceCount == std::numeric_limits<size_t>::max())
        return {range.firstSlice, plan.lineCount};
    if (range.sliceCount > plan.lineCount - range.firstSlice)
        throw std::out_of_range(
            "Structured sparsity slice range exceeds the available slice count.");
    return {range.firstSlice, range.firstSlice + range.sliceCount};
}

}  // namespace roc::host_numerics::detail
