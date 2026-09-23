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
#include <roc/host_numerics/structured_sparsity.hpp>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "generation_primitives.hpp"
#include "structured_sparsity_plan.hpp"
#include "threading.hpp"

namespace roc::host_numerics::detail {
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
inline StructuredSparsityWorkCounts applyTwoOfFourByteAligned(
    const StructuredSparsityInvocation& problem, const StructuredSparsityPlan& plan,
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

        const std::byte* inputPointer = problem.input.rawEncodedBackingStorage().data() +
                                        static_cast<size_t>(inputBase) * BytesPerElement;
        std::byte* prunedPointer = problem.pruned.rawEncodedBackingStorage().data() +
                                   static_cast<size_t>(prunedBase) * BytesPerElement;
        std::byte* compressedPointer = problem.compressed.rawEncodedBackingStorage().data() +
                                       static_cast<size_t>(compressedBase) * BytesPerElement;

        if (inputPointer != prunedPointer)
            std::memcpy(prunedPointer, inputPointer, inputBytesPerLine);

        uint8_t* metadataPointer =
            problem.twoOfFourMetadata
                ? reinterpret_cast<uint8_t*>(
                      problem.twoOfFourMetadata->rawEncodedBackingStorage().data()) +
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
                ? reinterpret_cast<uint8_t*>(
                      problem.retainedIndices->rawEncodedBackingStorage().data()) +
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
    return StructuredSparsityWorkCounts{
        .groupsProcessed = groups,
        .inputElementsVisited = groups * 4,
        .prunedElementsWritten = groups * 4,
        .compressedElementsWritten = groups * 2,
        .retainedIndicesWritten = problem.retainedIndices ? groups * 2 : 0,
        .metadataBytesWritten = problem.twoOfFourMetadata ? slices * metadataBytesPerLine : 0,
    };
}

template <size_t BytesPerElement, bool RandomSelection>
inline StructuredSparsityWorkCounts applyTwoOfFourByteAlignedStrided(
    const StructuredSparsityInvocation& problem, const StructuredSparsityPlan& plan,
    size_t firstSlice, size_t endSlice) {
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
    const bool inPlace = problem.input.rawEncodedBackingStorage().data() ==
                         problem.pruned.rawEncodedBackingStorage().data();

    auto inputPointer = [&](ptrdiff_t elementOffset) {
        return problem.input.rawEncodedBackingStorage().data() +
               static_cast<size_t>(elementOffset) * BytesPerElement;
    };
    auto prunedPointer = [&](ptrdiff_t elementOffset) {
        return problem.pruned.rawEncodedBackingStorage().data() +
               static_cast<size_t>(elementOffset) * BytesPerElement;
    };
    auto compressedPointer = [&](ptrdiff_t elementOffset) {
        return problem.compressed.rawEncodedBackingStorage().data() +
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
                uint8_t* retainedIndexPointer = reinterpret_cast<uint8_t*>(
                    problem.retainedIndices->rawEncodedBackingStorage().data());
                const ptrdiff_t retainedGroup =
                    retainedIndexBase + static_cast<ptrdiff_t>(group * 2) * retainedIndexAxisStride;
                retainedIndexPointer[static_cast<size_t>(retainedGroup)] =
                    static_cast<uint8_t>(positionSet.retained[0]);
                retainedIndexPointer[static_cast<size_t>(retainedGroup + retainedIndexAxisStride)] =
                    static_cast<uint8_t>(positionSet.retained[1]);
            }
            if (problem.twoOfFourMetadata) {
                uint8_t* metadataPointer = reinterpret_cast<uint8_t*>(
                    problem.twoOfFourMetadata->rawEncodedBackingStorage().data());
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
    return StructuredSparsityWorkCounts{
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
inline StructuredSparsityWorkCounts applyTwoOfFourByteAlignedBySelection(
    const StructuredSparsityInvocation& problem, const StructuredSparsityPlan& plan,
    size_t firstSlice, size_t endSlice, bool contiguousAxis) {
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

inline std::optional<StructuredSparsityWorkCounts> tryTwoOfFourByteAligned(
    const StructuredSparsityInvocation& problem, const StructuredSparsityPlan& plan,
    size_t firstSlice, size_t endSlice) {
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
}  // namespace roc::host_numerics::detail
