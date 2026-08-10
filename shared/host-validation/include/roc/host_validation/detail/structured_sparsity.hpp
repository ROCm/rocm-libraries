// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <roc/host_validation/generation_primitives.hpp>
#include <span>
#include <stdexcept>
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

namespace detail {
struct StructuredSparsityPlan {
    Shape compressedShape;
    size_t groupsPerLine = 0;
    size_t lineCount = 0;
    std::vector<std::vector<size_t>> retainedPositionSets;
};

inline void enumerateRetainedPositionSets(size_t groupSize, size_t retainedElements,
                                          size_t nextPosition, std::vector<size_t>& current,
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

inline StructuredSparsityPlan validateStructuredSparsity(const StructuredSparsityProblem& problem) {
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
    if (problem.pattern.groupSize > 256)
        throw std::invalid_argument(
            "Structured sparsity UInt8 retained indices require group size at most 256.");
    if (inputShape[problem.pattern.axis] == 0)
        throw std::invalid_argument("Structured sparsity axis extent must be nonzero.");
    if (inputShape[problem.pattern.axis] % problem.pattern.groupSize != 0)
        throw std::invalid_argument(
            "Structured sparsity axis extent must be divisible by group size.");
    if (scalarTypeInfo(problem.input.type()).category == ScalarCategory::Scale)
        throw std::invalid_argument(
            "Structured sparsity does not accept scale-only scalar encodings.");

    if (problem.pruned.type() != problem.input.type() ||
        problem.compressed.type() != problem.input.type())
        throw std::invalid_argument(
            "Structured sparsity input, pruned, and compressed scalar types must match.");
    if (problem.pruned.shape() != inputShape)
        throw std::invalid_argument("Structured sparsity pruned tensor shape mismatch.");

    std::vector<size_t> compressedDimensions(inputShape.dimensions().begin(),
                                             inputShape.dimensions().end());
    compressedDimensions[problem.pattern.axis] = inputShape[problem.pattern.axis] /
                                                 problem.pattern.groupSize *
                                                 problem.pattern.retainedElements;
    const Shape compressedShape(std::move(compressedDimensions));
    if (problem.compressed.shape() != compressedShape)
        throw std::invalid_argument("Structured sparsity compressed tensor shape mismatch.");
    if (problem.retainedIndices) {
        if (problem.retainedIndices->type() != ScalarType::UInt8)
            throw std::invalid_argument("Structured sparsity retained indices must use UInt8.");
        if (problem.retainedIndices->shape() != compressedShape)
            throw std::invalid_argument(
                "Structured sparsity retained-index tensor shape mismatch.");
    }
    if (problem.twoOfFourMetadata) {
        if (problem.pattern.groupSize != 4 || problem.pattern.retainedElements != 2)
            throw std::invalid_argument(
                "Two-of-four metadata output requires a two-of-four sparsity pattern.");
        if (problem.twoOfFourMetadata->type() != ScalarType::UInt8)
            throw std::invalid_argument("Two-of-four metadata output must use UInt8.");
        std::vector<size_t> metadataDimensions(inputShape.dimensions().begin(),
                                               inputShape.dimensions().end());
        metadataDimensions[problem.pattern.axis] =
            (inputShape[problem.pattern.axis] / problem.pattern.groupSize + 1) / 2;
        if (problem.twoOfFourMetadata->shape() != Shape(std::move(metadataDimensions)))
            throw std::invalid_argument("Structured sparsity two-of-four metadata shape mismatch.");
    }

    if (problem.input.storage().data() == problem.pruned.storage().data() &&
        problem.input.layout() != problem.pruned.layout())
        throw std::invalid_argument(
            "In-place structured sparsity requires identical input and pruned layouts.");

    StructuredSparsityPlan plan;
    plan.compressedShape = compressedShape;
    plan.groupsPerLine = inputShape[problem.pattern.axis] / problem.pattern.groupSize;
    plan.lineCount = inputShape.elementCount() / inputShape[problem.pattern.axis];

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

inline void coordinatesForLine(size_t line, const Shape& shape, size_t excludedAxis,
                               LogicalIndexOrder order, std::vector<size_t>& coordinates) {
    coordinates.assign(shape.rank(), 0);
    if (order == LogicalIndexOrder::FirstDimensionFastest) {
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

inline bool bitAt(std::span<const std::byte> storage, uint64_t bit) {
    const size_t byteIndex = static_cast<size_t>(bit / 8);
    const uint8_t byte = static_cast<uint8_t>(storage[byteIndex]);
    return ((byte >> (bit % 8)) & 1U) != 0;
}

inline void setBit(std::span<std::byte> storage, uint64_t bit, bool value) {
    const size_t byteIndex = static_cast<size_t>(bit / 8);
    const uint8_t mask = static_cast<uint8_t>(1U << (bit % 8));
    uint8_t& byte = reinterpret_cast<uint8_t&>(storage[byteIndex]);
    byte = value ? static_cast<uint8_t>(byte | mask)
                 : static_cast<uint8_t>(byte & static_cast<uint8_t>(~mask));
}

inline void copyEncodedElement(ScalarType type, std::span<const std::byte> source,
                               ptrdiff_t sourceElement, std::span<std::byte> destination,
                               ptrdiff_t destinationElement) {
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

inline void zeroEncodedElement(ScalarType type, std::span<std::byte> destination,
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

inline size_t retainedPositionSetIndexForGroup(const StructuredSparsityProblem& problem,
                                               const StructuredSparsityPlan& plan,
                                               size_t groupLinearIndex) {
    if (problem.pattern.selection == StructuredSparsitySelection::Fixed) return 0;
    const int selected =
        indexedUniformInteger(problem.pattern.seed, problem.pattern.stream, groupLinearIndex, 0,
                              static_cast<int>(plan.retainedPositionSets.size()) - 1);
    return static_cast<size_t>(selected);
}

inline const std::vector<size_t>& retainedPositionsForGroup(
    const StructuredSparsityProblem& problem, const StructuredSparsityPlan& plan,
    size_t groupLinearIndex) {
    return plan
        .retainedPositionSets[retainedPositionSetIndexForGroup(problem, plan, groupLinearIndex)];
}

inline std::pair<size_t, size_t> validateStructuredSparsitySliceRange(
    const StructuredSparsityPlan& plan, StructuredSparsitySliceRange range) {
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

inline uint8_t twoOfFourMetadataNibble(uint8_t first, uint8_t second);

struct TwoOfFourPositionSet {
    std::array<size_t, 2> retained{};
    std::array<size_t, 2> dropped{};
    uint8_t metadataNibble = 0;
};

inline std::vector<TwoOfFourPositionSet> makeTwoOfFourPositionSets(
    const StructuredSparsityPlan& plan) {
    std::vector<TwoOfFourPositionSet> result;
    result.reserve(plan.retainedPositionSets.size());
    for (const std::vector<size_t>& retained : plan.retainedPositionSets) {
        TwoOfFourPositionSet positionSet;
        positionSet.retained = {retained[0], retained[1]};
        size_t droppedIndex = 0;
        for (size_t position = 0; position < 4; ++position) {
            if (position != retained[0] && position != retained[1])
                positionSet.dropped[droppedIndex++] = position;
        }
        positionSet.metadataNibble = twoOfFourMetadataNibble(static_cast<uint8_t>(retained[0]),
                                                             static_cast<uint8_t>(retained[1]));
        result.push_back(positionSet);
    }
    return result;
}

template <size_t BytesPerElement, bool RandomSelection>
inline StructuredSparsityRunInfo applyTwoOfFourByteAligned(const StructuredSparsityProblem& problem,
                                                           const StructuredSparsityPlan& plan,
                                                           size_t firstSlice, size_t endSlice) {
    const size_t axis = problem.pattern.axis;
    const std::vector<TwoOfFourPositionSet> positionSets = makeTwoOfFourPositionSets(plan);
    const TwoOfFourPositionSet& fixedPositionSet = positionSets.front();
    const uint8_t fullMetadataByte =
        static_cast<uint8_t>(fixedPositionSet.metadataNibble |
                             static_cast<uint8_t>(fixedPositionSet.metadataNibble << 4));
    const size_t metadataBytesPerLine = (plan.groupsPerLine + 1) / 2;
    const size_t inputBytesPerLine = plan.groupsPerLine * 4 * BytesPerElement;

    std::vector<size_t> coordinates(problem.input.shape().rank(), 0);
    for (size_t line = firstSlice; line < endSlice; ++line) {
        coordinatesForLine(line, problem.input.shape(), axis, problem.pattern.indexOrder,
                           coordinates);
        const ptrdiff_t inputBase = problem.input.layout().elementOffset(coordinates);
        const ptrdiff_t prunedBase = problem.pruned.layout().elementOffset(coordinates);
        const ptrdiff_t compressedBase = problem.compressed.layout().elementOffset(coordinates);
        const ptrdiff_t retainedIndexBase =
            problem.retainedIndices ? problem.retainedIndices->layout().elementOffset(coordinates)
                                    : 0;
        const ptrdiff_t metadataBase =
            problem.twoOfFourMetadata
                ? problem.twoOfFourMetadata->layout().elementOffset(coordinates)
                : 0;

        const std::byte* inputPointer =
            problem.input.storage().data() + static_cast<size_t>(inputBase) * BytesPerElement;
        std::byte* prunedPointer =
            problem.pruned.storage().data() + static_cast<size_t>(prunedBase) * BytesPerElement;
        std::byte* compressedPointer = problem.compressed.storage().data() +
                                       static_cast<size_t>(compressedBase) * BytesPerElement;

        if (inputPointer != prunedPointer)
            std::memcpy(prunedPointer, inputPointer, inputBytesPerLine);

        uint8_t* metadataPointer =
            problem.twoOfFourMetadata
                ? reinterpret_cast<uint8_t*>(problem.twoOfFourMetadata->storage().data()) +
                      static_cast<size_t>(metadataBase)
                : nullptr;
        if constexpr (!RandomSelection) {
            if (metadataPointer) {
                std::memset(metadataPointer, fullMetadataByte, plan.groupsPerLine / 2);
                if (plan.groupsPerLine % 2 != 0)
                    metadataPointer[metadataBytesPerLine - 1] = fixedPositionSet.metadataNibble;
            }
        }

        uint8_t* retainedIndexPointer =
            problem.retainedIndices
                ? reinterpret_cast<uint8_t*>(problem.retainedIndices->storage().data()) +
                      static_cast<size_t>(retainedIndexBase)
                : nullptr;
        for (size_t group = 0; group < plan.groupsPerLine; ++group) {
            const TwoOfFourPositionSet& positionSet = [&]() -> const TwoOfFourPositionSet& {
                if constexpr (RandomSelection) {
                    return positionSets[retainedPositionSetIndexForGroup(
                        problem, plan, line * plan.groupsPerLine + group)];
                } else {
                    return fixedPositionSet;
                }
            }();
            const std::byte* inputGroup = inputPointer + group * 4 * BytesPerElement;
            std::byte* prunedGroup = prunedPointer + group * 4 * BytesPerElement;
            std::byte* compressedGroup = compressedPointer + group * 2 * BytesPerElement;

            std::memcpy(compressedGroup, inputGroup + positionSet.retained[0] * BytesPerElement,
                        BytesPerElement);
            std::memcpy(compressedGroup + BytesPerElement,
                        inputGroup + positionSet.retained[1] * BytesPerElement, BytesPerElement);
            std::memset(prunedGroup + positionSet.dropped[0] * BytesPerElement, 0, BytesPerElement);
            std::memset(prunedGroup + positionSet.dropped[1] * BytesPerElement, 0, BytesPerElement);

            if (retainedIndexPointer) {
                retainedIndexPointer[group * 2] = static_cast<uint8_t>(positionSet.retained[0]);
                retainedIndexPointer[group * 2 + 1] = static_cast<uint8_t>(positionSet.retained[1]);
            }
            if constexpr (RandomSelection) {
                if (metadataPointer) {
                    uint8_t& metadataByte = metadataPointer[group / 2];
                    if (group % 2 == 0)
                        metadataByte = positionSet.metadataNibble;
                    else
                        metadataByte = static_cast<uint8_t>(
                            metadataByte | static_cast<uint8_t>(positionSet.metadataNibble << 4));
                }
            }
        }
    }

    const size_t slices = endSlice - firstSlice;
    const size_t groups = slices * plan.groupsPerLine;
    return StructuredSparsityRunInfo{
        .groupsProcessed = groups,
        .inputElementsVisited = groups * 4,
        .prunedElementsWritten = groups * 4,
        .compressedElementsWritten = groups * 2,
        .retainedIndicesWritten = problem.retainedIndices ? groups * 2 : 0,
        .metadataBytesWritten = problem.twoOfFourMetadata ? slices * metadataBytesPerLine : 0,
    };
}

template <size_t BytesPerElement, bool RandomSelection>
inline StructuredSparsityRunInfo applyTwoOfFourByteAlignedStrided(
    const StructuredSparsityProblem& problem, const StructuredSparsityPlan& plan, size_t firstSlice,
    size_t endSlice) {
    const size_t axis = problem.pattern.axis;
    const std::vector<TwoOfFourPositionSet> positionSets = makeTwoOfFourPositionSets(plan);
    const TwoOfFourPositionSet& fixedPositionSet = positionSets.front();

    const ptrdiff_t inputAxisStride = problem.input.layout().strides()[axis];
    const ptrdiff_t prunedAxisStride = problem.pruned.layout().strides()[axis];
    const ptrdiff_t compressedAxisStride = problem.compressed.layout().strides()[axis];
    const ptrdiff_t retainedIndexAxisStride =
        problem.retainedIndices ? problem.retainedIndices->layout().strides()[axis] : 0;
    const ptrdiff_t metadataAxisStride =
        problem.twoOfFourMetadata ? problem.twoOfFourMetadata->layout().strides()[axis] : 0;
    const bool inPlace = problem.input.storage().data() == problem.pruned.storage().data();

    auto inputPointer = [&](ptrdiff_t elementOffset) {
        return problem.input.storage().data() +
               static_cast<size_t>(elementOffset) * BytesPerElement;
    };
    auto prunedPointer = [&](ptrdiff_t elementOffset) {
        return problem.pruned.storage().data() +
               static_cast<size_t>(elementOffset) * BytesPerElement;
    };
    auto compressedPointer = [&](ptrdiff_t elementOffset) {
        return problem.compressed.storage().data() +
               static_cast<size_t>(elementOffset) * BytesPerElement;
    };

    std::vector<size_t> coordinates(problem.input.shape().rank(), 0);
    for (size_t line = firstSlice; line < endSlice; ++line) {
        coordinatesForLine(line, problem.input.shape(), axis, problem.pattern.indexOrder,
                           coordinates);
        const ptrdiff_t inputBase = problem.input.layout().elementOffset(coordinates);
        const ptrdiff_t prunedBase = problem.pruned.layout().elementOffset(coordinates);
        const ptrdiff_t compressedBase = problem.compressed.layout().elementOffset(coordinates);
        const ptrdiff_t retainedIndexBase =
            problem.retainedIndices ? problem.retainedIndices->layout().elementOffset(coordinates)
                                    : 0;
        const ptrdiff_t metadataBase =
            problem.twoOfFourMetadata
                ? problem.twoOfFourMetadata->layout().elementOffset(coordinates)
                : 0;

        for (size_t group = 0; group < plan.groupsPerLine; ++group) {
            const TwoOfFourPositionSet& positionSet = [&]() -> const TwoOfFourPositionSet& {
                if constexpr (RandomSelection) {
                    return positionSets[retainedPositionSetIndexForGroup(
                        problem, plan, line * plan.groupsPerLine + group)];
                } else {
                    return fixedPositionSet;
                }
            }();
            const ptrdiff_t inputGroup =
                inputBase + static_cast<ptrdiff_t>(group * 4) * inputAxisStride;
            const ptrdiff_t prunedGroup =
                prunedBase + static_cast<ptrdiff_t>(group * 4) * prunedAxisStride;
            const ptrdiff_t compressedGroup =
                compressedBase + static_cast<ptrdiff_t>(group * 2) * compressedAxisStride;

            const ptrdiff_t firstInput =
                inputGroup + static_cast<ptrdiff_t>(positionSet.retained[0]) * inputAxisStride;
            const ptrdiff_t secondInput =
                inputGroup + static_cast<ptrdiff_t>(positionSet.retained[1]) * inputAxisStride;
            std::memcpy(compressedPointer(compressedGroup), inputPointer(firstInput),
                        BytesPerElement);
            std::memcpy(compressedPointer(compressedGroup + compressedAxisStride),
                        inputPointer(secondInput), BytesPerElement);

            if (!inPlace) {
                std::memcpy(
                    prunedPointer(prunedGroup + static_cast<ptrdiff_t>(positionSet.retained[0]) *
                                                    prunedAxisStride),
                    inputPointer(firstInput), BytesPerElement);
                std::memcpy(
                    prunedPointer(prunedGroup + static_cast<ptrdiff_t>(positionSet.retained[1]) *
                                                    prunedAxisStride),
                    inputPointer(secondInput), BytesPerElement);
            }
            std::memset(prunedPointer(prunedGroup + static_cast<ptrdiff_t>(positionSet.dropped[0]) *
                                                        prunedAxisStride),
                        0, BytesPerElement);
            std::memset(prunedPointer(prunedGroup + static_cast<ptrdiff_t>(positionSet.dropped[1]) *
                                                        prunedAxisStride),
                        0, BytesPerElement);

            if (problem.retainedIndices) {
                uint8_t* retainedIndexPointer =
                    reinterpret_cast<uint8_t*>(problem.retainedIndices->storage().data());
                const ptrdiff_t retainedGroup =
                    retainedIndexBase + static_cast<ptrdiff_t>(group * 2) * retainedIndexAxisStride;
                retainedIndexPointer[static_cast<size_t>(retainedGroup)] =
                    static_cast<uint8_t>(positionSet.retained[0]);
                retainedIndexPointer[static_cast<size_t>(retainedGroup + retainedIndexAxisStride)] =
                    static_cast<uint8_t>(positionSet.retained[1]);
            }
            if (problem.twoOfFourMetadata) {
                uint8_t* metadataPointer =
                    reinterpret_cast<uint8_t*>(problem.twoOfFourMetadata->storage().data());
                const ptrdiff_t metadataOffset =
                    metadataBase + static_cast<ptrdiff_t>(group / 2) * metadataAxisStride;
                uint8_t& metadataByte = metadataPointer[static_cast<size_t>(metadataOffset)];
                if (group % 2 == 0)
                    metadataByte = positionSet.metadataNibble;
                else
                    metadataByte = static_cast<uint8_t>(
                        metadataByte | static_cast<uint8_t>(positionSet.metadataNibble << 4));
            }
        }
    }

    const size_t slices = endSlice - firstSlice;
    const size_t groups = slices * plan.groupsPerLine;
    return StructuredSparsityRunInfo{
        .groupsProcessed = groups,
        .inputElementsVisited = groups * 4,
        .prunedElementsWritten = groups * 4,
        .compressedElementsWritten = groups * 2,
        .retainedIndicesWritten = problem.retainedIndices ? groups * 2 : 0,
        .metadataBytesWritten =
            problem.twoOfFourMetadata ? slices * ((plan.groupsPerLine + 1) / 2) : 0,
    };
}

template <size_t BytesPerElement>
inline StructuredSparsityRunInfo applyTwoOfFourByteAlignedBySelection(
    const StructuredSparsityProblem& problem, const StructuredSparsityPlan& plan, size_t firstSlice,
    size_t endSlice, bool contiguousAxis) {
    if (problem.pattern.selection == StructuredSparsitySelection::Random) {
        return contiguousAxis ? applyTwoOfFourByteAligned<BytesPerElement, true>(
                                    problem, plan, firstSlice, endSlice)
                              : applyTwoOfFourByteAlignedStrided<BytesPerElement, true>(
                                    problem, plan, firstSlice, endSlice);
    }
    return contiguousAxis ? applyTwoOfFourByteAligned<BytesPerElement, false>(problem, plan,
                                                                              firstSlice, endSlice)
                          : applyTwoOfFourByteAlignedStrided<BytesPerElement, false>(
                                problem, plan, firstSlice, endSlice);
}

inline std::optional<StructuredSparsityRunInfo> tryTwoOfFourByteAligned(
    const StructuredSparsityProblem& problem, const StructuredSparsityPlan& plan, size_t firstSlice,
    size_t endSlice) {
    if (problem.pattern.groupSize != 4 || problem.pattern.retainedElements != 2 ||
        scalarTypeInfo(problem.input.type()).storageBits % 8 != 0)
        return std::nullopt;

    const size_t axis = problem.pattern.axis;
    const bool contiguousAxis =
        problem.input.layout().strides()[axis] == 1 &&
        problem.pruned.layout().strides()[axis] == 1 &&
        problem.compressed.layout().strides()[axis] == 1 &&
        (!problem.retainedIndices || problem.retainedIndices->layout().strides()[axis] == 1) &&
        (!problem.twoOfFourMetadata || problem.twoOfFourMetadata->layout().strides()[axis] == 1);

    switch (scalarTypeInfo(problem.input.type()).storageBits / 8) {
        case 1:
            return applyTwoOfFourByteAlignedBySelection<1>(problem, plan, firstSlice, endSlice,
                                                           contiguousAxis);
        case 2:
            return applyTwoOfFourByteAlignedBySelection<2>(problem, plan, firstSlice, endSlice,
                                                           contiguousAxis);
        case 4:
            return applyTwoOfFourByteAlignedBySelection<4>(problem, plan, firstSlice, endSlice,
                                                           contiguousAxis);
        case 8:
            return applyTwoOfFourByteAlignedBySelection<8>(problem, plan, firstSlice, endSlice,
                                                           contiguousAxis);
        case 16:
            return applyTwoOfFourByteAlignedBySelection<16>(problem, plan, firstSlice, endSlice,
                                                            contiguousAxis);
        default:
            return std::nullopt;
    }
}

inline Shape validateTwoOfFourMetadata(const TwoOfFourMetadataProblem& problem) {
    if (problem.retainedIndices.type() != ScalarType::UInt8 ||
        problem.metadata.type() != ScalarType::UInt8)
        throw std::invalid_argument("Two-of-four metadata input/output types must be UInt8.");
    if (problem.retainedIndices.shape().rank() == 0)
        throw std::invalid_argument("Two-of-four metadata requires a rank-one or higher tensor.");
    if (problem.axis >= problem.retainedIndices.shape().rank())
        throw std::out_of_range("Two-of-four metadata axis exceeds tensor rank.");
    if (problem.retainedIndices.shape()[problem.axis] % 2 != 0)
        throw std::invalid_argument(
            "Two-of-four retained-index axis must contain two indices per sparsity group.");

    const size_t sparsityGroups = problem.retainedIndices.shape()[problem.axis] / 2;
    std::vector<size_t> metadataDimensions(problem.retainedIndices.shape().dimensions().begin(),
                                           problem.retainedIndices.shape().dimensions().end());
    metadataDimensions[problem.axis] = (sparsityGroups + 1) / 2;
    const Shape metadataShape(std::move(metadataDimensions));
    if (problem.metadata.shape() != metadataShape)
        throw std::invalid_argument("Two-of-four metadata output shape mismatch.");
    return metadataShape;
}

inline uint8_t twoOfFourMetadataNibble(uint8_t first, uint8_t second) {
    if (first >= 4 || second >= 4 || first >= second)
        throw std::invalid_argument(
            "Two-of-four retained positions must be increasing values in [0, 3].");
    return static_cast<uint8_t>(first | static_cast<uint8_t>(second << 2));
}
}  // namespace detail

inline StructuredSparsityRunInfo applyStructuredSparsity(
    const StructuredSparsityProblem& problem, StructuredSparsitySliceRange sliceRange = {}) {
    const detail::StructuredSparsityPlan plan = detail::validateStructuredSparsity(problem);
    const auto [firstSlice, endSlice] =
        detail::validateStructuredSparsitySliceRange(plan, sliceRange);
    if (const auto fastRun = detail::tryTwoOfFourByteAligned(problem, plan, firstSlice, endSlice))
        return *fastRun;
    const size_t axis = problem.pattern.axis;
    const size_t groupSize = problem.pattern.groupSize;
    const size_t retainedElements = problem.pattern.retainedElements;

    std::vector<size_t> inputCoordinates(problem.input.shape().rank(), 0);
    std::vector<bool> retained(groupSize, false);
    const ptrdiff_t inputAxisStride = problem.input.layout().strides()[axis];
    const ptrdiff_t prunedAxisStride = problem.pruned.layout().strides()[axis];
    const ptrdiff_t compressedAxisStride = problem.compressed.layout().strides()[axis];
    const ptrdiff_t retainedIndexAxisStride =
        problem.retainedIndices ? problem.retainedIndices->layout().strides()[axis] : 0;
    const ptrdiff_t metadataAxisStride =
        problem.twoOfFourMetadata ? problem.twoOfFourMetadata->layout().strides()[axis] : 0;

    for (size_t line = firstSlice; line < endSlice; ++line) {
        detail::coordinatesForLine(line, problem.input.shape(), axis, problem.pattern.indexOrder,
                                   inputCoordinates);
        const ptrdiff_t inputBase = problem.input.layout().elementOffset(inputCoordinates);
        const ptrdiff_t prunedBase = problem.pruned.layout().elementOffset(inputCoordinates);
        const ptrdiff_t compressedBase =
            problem.compressed.layout().elementOffset(inputCoordinates);
        const ptrdiff_t retainedIndexBase =
            problem.retainedIndices
                ? problem.retainedIndices->layout().elementOffset(inputCoordinates)
                : 0;
        const ptrdiff_t metadataBase =
            problem.twoOfFourMetadata
                ? problem.twoOfFourMetadata->layout().elementOffset(inputCoordinates)
                : 0;

        for (size_t group = 0; group < plan.groupsPerLine; ++group) {
            const size_t groupLinearIndex = line * plan.groupsPerLine + group;
            const std::vector<size_t>& retainedPositions =
                detail::retainedPositionsForGroup(problem, plan, groupLinearIndex);
            std::fill(retained.begin(), retained.end(), false);
            for (const size_t position : retainedPositions) retained[position] = true;

            for (size_t retainedIndex = 0; retainedIndex < retainedElements; ++retainedIndex) {
                const size_t position = retainedPositions[retainedIndex];
                const ptrdiff_t sourceOffset =
                    inputBase +
                    static_cast<ptrdiff_t>(group * groupSize + position) * inputAxisStride;
                const ptrdiff_t compressedOffset =
                    compressedBase +
                    static_cast<ptrdiff_t>(group * retainedElements + retainedIndex) *
                        compressedAxisStride;
                detail::copyEncodedElement(problem.input.type(), problem.input.storage(),
                                           sourceOffset, problem.compressed.storage(),
                                           compressedOffset);
                if (problem.retainedIndices) {
                    const ptrdiff_t retainedIndexOffset =
                        retainedIndexBase +
                        static_cast<ptrdiff_t>(group * retainedElements + retainedIndex) *
                            retainedIndexAxisStride;
                    detail::encodeScalarKnown<ScalarType::UInt8>(problem.retainedIndices->storage(),
                                                                 retainedIndexOffset,
                                                                 static_cast<uint8_t>(position));
                }
            }

            if (problem.twoOfFourMetadata) {
                const uint8_t nibble =
                    detail::twoOfFourMetadataNibble(static_cast<uint8_t>(retainedPositions[0]),
                                                    static_cast<uint8_t>(retainedPositions[1]));
                const ptrdiff_t metadataOffset =
                    metadataBase + static_cast<ptrdiff_t>(group / 2) * metadataAxisStride;
                if (group % 2 == 0) {
                    detail::encodeScalarKnown<ScalarType::UInt8>(
                        problem.twoOfFourMetadata->storage(), metadataOffset, nibble);
                } else {
                    const uint8_t previous = detail::decodeScalarKnown<ScalarType::UInt8, uint8_t>(
                        problem.twoOfFourMetadata->storage(), metadataOffset);
                    detail::encodeScalarKnown<ScalarType::UInt8>(
                        problem.twoOfFourMetadata->storage(), metadataOffset,
                        static_cast<uint8_t>(previous | static_cast<uint8_t>(nibble << 4)));
                }
            }

            for (size_t position = 0; position < groupSize; ++position) {
                const ptrdiff_t sourceOffset =
                    inputBase +
                    static_cast<ptrdiff_t>(group * groupSize + position) * inputAxisStride;
                const ptrdiff_t prunedOffset =
                    prunedBase +
                    static_cast<ptrdiff_t>(group * groupSize + position) * prunedAxisStride;
                if (retained[position]) {
                    detail::copyEncodedElement(problem.input.type(), problem.input.storage(),
                                               sourceOffset, problem.pruned.storage(),
                                               prunedOffset);
                } else {
                    detail::zeroEncodedElement(problem.pruned.type(), problem.pruned.storage(),
                                               prunedOffset);
                }
            }
        }
    }

    const size_t slices = endSlice - firstSlice;
    const size_t groups = slices * plan.groupsPerLine;
    return {
        .groupsProcessed = groups,
        .inputElementsVisited = groups * groupSize,
        .prunedElementsWritten = groups * groupSize,
        .compressedElementsWritten = groups * retainedElements,
        .retainedIndicesWritten = problem.retainedIndices ? groups * retainedElements : 0,
        .metadataBytesWritten =
            problem.twoOfFourMetadata ? slices * ((plan.groupsPerLine + 1) / 2) : 0,
    };
}

inline TwoOfFourMetadataRunInfo encodeTwoOfFourMetadata(const TwoOfFourMetadataProblem& problem) {
    const Shape metadataShape = detail::validateTwoOfFourMetadata(problem);
    const size_t sparsityGroups = problem.retainedIndices.shape()[problem.axis] / 2;
    const size_t metadataGroups = metadataShape[problem.axis];
    const size_t lineCount = problem.retainedIndices.shape().elementCount() /
                             problem.retainedIndices.shape()[problem.axis];

    std::vector<size_t> retainedCoordinates(problem.retainedIndices.shape().rank(), 0);
    const ptrdiff_t retainedAxisStride = problem.retainedIndices.layout().strides()[problem.axis];
    const ptrdiff_t metadataAxisStride = problem.metadata.layout().strides()[problem.axis];
    for (size_t line = 0; line < lineCount; ++line) {
        detail::coordinatesForLine(line, problem.retainedIndices.shape(), problem.axis,
                                   LogicalIndexOrder::FirstDimensionFastest, retainedCoordinates);
        const ptrdiff_t retainedBase =
            problem.retainedIndices.layout().elementOffset(retainedCoordinates);
        const ptrdiff_t metadataBase = problem.metadata.layout().elementOffset(retainedCoordinates);

        for (size_t metadataGroup = 0; metadataGroup < metadataGroups; ++metadataGroup) {
            const size_t firstGroup = metadataGroup * 2;
            const ptrdiff_t firstOffset =
                retainedBase + static_cast<ptrdiff_t>(firstGroup * 2) * retainedAxisStride;
            const uint8_t first0 = detail::decodeScalarKnown<ScalarType::UInt8, uint8_t>(
                problem.retainedIndices.storage(), firstOffset);
            const uint8_t first1 = detail::decodeScalarKnown<ScalarType::UInt8, uint8_t>(
                problem.retainedIndices.storage(), firstOffset + retainedAxisStride);
            uint8_t encoded = detail::twoOfFourMetadataNibble(first0, first1);

            if (firstGroup + 1 < sparsityGroups) {
                const ptrdiff_t secondOffset = firstOffset + 2 * retainedAxisStride;
                const uint8_t second0 = detail::decodeScalarKnown<ScalarType::UInt8, uint8_t>(
                    problem.retainedIndices.storage(), secondOffset);
                const uint8_t second1 = detail::decodeScalarKnown<ScalarType::UInt8, uint8_t>(
                    problem.retainedIndices.storage(), secondOffset + retainedAxisStride);
                encoded = static_cast<uint8_t>(
                    encoded |
                    static_cast<uint8_t>(detail::twoOfFourMetadataNibble(second0, second1) << 4));
            }

            detail::encodeScalarKnown<ScalarType::UInt8>(
                problem.metadata.storage(),
                metadataBase + static_cast<ptrdiff_t>(metadataGroup) * metadataAxisStride, encoded);
        }
    }

    return {
        .sparsityGroupsEncoded = lineCount * sparsityGroups,
        .metadataBytesWritten = lineCount * metadataGroups,
    };
}
}  // namespace roc::host_validation
