// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "detail/structured_sparsity_byte_aligned.hpp"
#include "detail/structured_sparsity_plan.hpp"
#include "detail/threading.hpp"
#include "detail/two_of_four_metadata.hpp"

namespace roc::host_numerics {
detail::StructuredSparsityWorkCounts applyStructuredSparsityInvocation(
    const detail::StructuredSparsityInvocation& problem, StructuredSparsitySliceRange sliceRange) {
    const detail::StructuredSparsityPlan plan = detail::validateStructuredSparsityRequest(problem);
    const auto [firstSlice, endSlice] =
        detail::validateStructuredSparsitySliceRange(plan, sliceRange);
    if (problem.input.elementCount() == 0) return {};
    const bool completeRange =
        sliceRange.firstSlice == 0 && sliceRange.sliceCount == std::numeric_limits<size_t>::max();
    const bool independentOutputs =
        detail::hasProvablyIndependentElements(problem.pruned) &&
        detail::hasProvablyIndependentElements(problem.compressed) &&
        (!problem.retainedIndices ||
         detail::hasProvablyIndependentElements(*problem.retainedIndices)) &&
        (!problem.twoOfFourMetadata ||
         detail::hasProvablyIndependentElements(*problem.twoOfFourMetadata));
    const size_t work = detail::saturatedProduct(
        detail::saturatedProduct(plan.lineCount, plan.groupsPerLine), problem.pattern.groupSize);
    const size_t chunkCount =
        completeRange && independentOutputs
            ? std::min(plan.lineCount,
                       static_cast<size_t>(detail::operationThreadCount(work, 500'000)))
            : 1;
    if (chunkCount > 1) {
        std::vector<detail::StructuredSparsityWorkCounts> partialRuns(chunkCount);
        detail::forEachParallelIndex(chunkCount, work, true, 1, [&](size_t chunk) {
            const size_t chunkFirst = plan.lineCount * chunk / chunkCount;
            const size_t chunkEnd = plan.lineCount * (chunk + 1) / chunkCount;
            partialRuns[chunk] = applyStructuredSparsityInvocation(
                problem, {.firstSlice = chunkFirst, .sliceCount = chunkEnd - chunkFirst});
        });

        detail::StructuredSparsityWorkCounts combined;
        for (const detail::StructuredSparsityWorkCounts& partial : partialRuns) {
            combined.groupsProcessed += partial.groupsProcessed;
            combined.inputElementsVisited += partial.inputElementsVisited;
            combined.prunedElementsWritten += partial.prunedElementsWritten;
            combined.compressedElementsWritten += partial.compressedElementsWritten;
            combined.retainedIndicesWritten += partial.retainedIndicesWritten;
            combined.metadataBytesWritten += partial.metadataBytesWritten;
        }
        return combined;
    }
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
                detail::copyEncodedElement(
                    problem.input.type(), problem.input.rawEncodedBackingStorage(), sourceOffset,
                    problem.compressed.rawEncodedBackingStorage(), compressedOffset);
                if (problem.retainedIndices) {
                    const ptrdiff_t retainedIndexOffset =
                        retainedIndexBase +
                        static_cast<ptrdiff_t>(group * retainedElements + retainedIndex) *
                            retainedIndexAxisStride;
                    detail::encodeScalarKnown<detail::UInt8Tag>(
                        problem.retainedIndices->rawEncodedBackingStorage(), retainedIndexOffset,
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
                    detail::encodeScalarKnown<detail::UInt8Tag>(
                        problem.twoOfFourMetadata->rawEncodedBackingStorage(), metadataOffset,
                        nibble);
                } else {
                    const uint8_t previous = detail::decodeScalarKnown<detail::UInt8Tag, uint8_t>(
                        problem.twoOfFourMetadata->rawEncodedBackingStorage(), metadataOffset);
                    detail::encodeScalarKnown<detail::UInt8Tag>(
                        problem.twoOfFourMetadata->rawEncodedBackingStorage(), metadataOffset,
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
                    detail::copyEncodedElement(
                        problem.input.type(), problem.input.rawEncodedBackingStorage(),
                        sourceOffset, problem.pruned.rawEncodedBackingStorage(), prunedOffset);
                } else {
                    detail::zeroEncodedElement(problem.pruned.type(),
                                               problem.pruned.rawEncodedBackingStorage(),
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

void applyStructuredSparsityInto(Tensor input, StructuredSparseTensor outputs,
                                 StructuredSparsityPattern pattern,
                                 StructuredSparsitySliceRange sliceRange) {
    detail::StructuredSparsitySpecification specification(
        std::move(input), std::move(pattern),
        {.retainedIndices = outputs.retainedIndices.has_value(),
         .twoOfFourMetadata = outputs.twoOfFourMetadata.has_value()});
    detail::StructuredSparsityInvocation invocation(std::move(specification), std::move(outputs));
    (void)applyStructuredSparsityInvocation(invocation, sliceRange);
}

StructuredSparseTensor applyStructuredSparsity(Tensor input, StructuredSparsityPattern pattern,
                                               StructuredSparsityOutputOptions outputOptions) {
    const detail::StructuredSparsitySpecification specification(input, pattern, outputOptions);
    const detail::StructuredSparsityPlan plan =
        detail::validateStructuredSparsityProblem(specification);
    Tensor pruned(input.type(), input.shape());
    Tensor compressed(input.type(), plan.compressedShape);
    std::optional<Tensor> retainedIndices;
    std::optional<Tensor> twoOfFourMetadata;
    if (outputOptions.retainedIndices)
        retainedIndices.emplace(ScalarType::UInt8, plan.compressedShape);
    if (outputOptions.twoOfFourMetadata)
        twoOfFourMetadata.emplace(ScalarType::UInt8, *plan.metadataShape);
    StructuredSparseTensor outputs{
        .pruned = std::move(pruned),
        .compressed = std::move(compressed),
        .retainedIndices = std::move(retainedIndices),
        .twoOfFourMetadata = std::move(twoOfFourMetadata),
    };
    applyStructuredSparsityInto(std::move(input), outputs, std::move(pattern));
    return outputs;
}

detail::TwoOfFourMetadataWorkCounts encodeTwoOfFourMetadataInvocation(
    const detail::TwoOfFourMetadataInvocation& problem) {
    const Shape metadataShape = detail::validateTwoOfFourMetadataRequest(problem);
    const size_t sparsityGroups = problem.retainedIndices.shape()[problem.axis] / 2;
    const size_t metadataGroups = metadataShape[problem.axis];
    const size_t lineCount = problem.retainedIndices.shape().elementCountExcluding(problem.axis);
    if (problem.retainedIndices.elementCount() == 0) return {};

    std::vector<size_t> retainedCoordinates(problem.retainedIndices.shape().rank(), 0);
    const ptrdiff_t retainedAxisStride = problem.retainedIndices.layout().strides()[problem.axis];
    const ptrdiff_t metadataAxisStride = problem.metadata.layout().strides()[problem.axis];
    for (size_t line = 0; line < lineCount; ++line) {
        detail::coordinatesForLine(line, problem.retainedIndices.shape(), problem.axis,
                                   IndexOrder::FirstDimensionFastest, retainedCoordinates);
        const ptrdiff_t retainedBase =
            problem.retainedIndices.layout().elementOffset(retainedCoordinates);
        const ptrdiff_t metadataBase = problem.metadata.layout().elementOffset(retainedCoordinates);

        for (size_t metadataGroup = 0; metadataGroup < metadataGroups; ++metadataGroup) {
            const size_t firstGroup = metadataGroup * 2;
            const ptrdiff_t firstOffset =
                retainedBase + static_cast<ptrdiff_t>(firstGroup * 2) * retainedAxisStride;
            const uint8_t first0 = detail::decodeScalarKnown<detail::UInt8Tag, uint8_t>(
                problem.retainedIndices.rawEncodedBackingStorage(), firstOffset);
            const uint8_t first1 = detail::decodeScalarKnown<detail::UInt8Tag, uint8_t>(
                problem.retainedIndices.rawEncodedBackingStorage(),
                firstOffset + retainedAxisStride);
            uint8_t encoded = detail::twoOfFourMetadataNibble(first0, first1);

            if (firstGroup + 1 < sparsityGroups) {
                const ptrdiff_t secondOffset = firstOffset + 2 * retainedAxisStride;
                const uint8_t second0 = detail::decodeScalarKnown<detail::UInt8Tag, uint8_t>(
                    problem.retainedIndices.rawEncodedBackingStorage(), secondOffset);
                const uint8_t second1 = detail::decodeScalarKnown<detail::UInt8Tag, uint8_t>(
                    problem.retainedIndices.rawEncodedBackingStorage(),
                    secondOffset + retainedAxisStride);
                encoded = static_cast<uint8_t>(
                    encoded |
                    static_cast<uint8_t>(detail::twoOfFourMetadataNibble(second0, second1) << 4));
            }

            detail::encodeScalarKnown<detail::UInt8Tag>(
                problem.metadata.rawEncodedBackingStorage(),
                metadataBase + static_cast<ptrdiff_t>(metadataGroup) * metadataAxisStride, encoded);
        }
    }

    return {
        .sparsityGroupsEncoded = lineCount * sparsityGroups,
        .metadataBytesWritten = lineCount * metadataGroups,
    };
}

void encodeTwoOfFourMetadataInto(Tensor retainedIndices, Tensor metadata, size_t axis) {
    const detail::TwoOfFourMetadataInvocation invocation(std::move(retainedIndices),
                                                         std::move(metadata), axis);
    (void)encodeTwoOfFourMetadataInvocation(invocation);
}

Tensor encodeTwoOfFourMetadata(Tensor retainedIndices, size_t axis) {
    const detail::TwoOfFourMetadataArguments arguments{
        .retainedIndices = retainedIndices,
        .axis = axis,
    };
    const Shape metadataShape = detail::validateTwoOfFourMetadataProblem(arguments);
    Tensor metadata(ScalarType::UInt8, metadataShape);
    encodeTwoOfFourMetadataInto(std::move(retainedIndices), metadata, axis);
    return metadata;
}

}  // namespace roc::host_numerics
