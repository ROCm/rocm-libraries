// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "detail/two_of_four_metadata.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

namespace roc::host_numerics::detail {
Shape validateTwoOfFourMetadataProblem(const TwoOfFourMetadataArguments& problem) {
    if (problem.retainedIndices.type() != ScalarType::UInt8)
        throw std::invalid_argument("Two-of-four metadata input type must be UInt8.");
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

    const size_t lineCount = problem.retainedIndices.shape().elementCountExcluding(problem.axis);
    std::vector<size_t> retainedCoordinates(problem.retainedIndices.shape().rank(), 0);
    const ptrdiff_t retainedAxisStride = problem.retainedIndices.layout().strides()[problem.axis];
    for (size_t line = 0; line < lineCount; ++line) {
        coordinatesForLine(line, problem.retainedIndices.shape(), problem.axis,
                           IndexOrder::FirstDimensionFastest, retainedCoordinates);
        const ptrdiff_t retainedBase =
            problem.retainedIndices.layout().elementOffset(retainedCoordinates);
        for (size_t group = 0; group < sparsityGroups; ++group) {
            const ptrdiff_t firstOffset =
                retainedBase + static_cast<ptrdiff_t>(group * 2) * retainedAxisStride;
            const uint8_t first = decodeScalarKnown<UInt8Tag, uint8_t>(
                problem.retainedIndices.rawEncodedBackingStorage(), firstOffset);
            const uint8_t second = decodeScalarKnown<UInt8Tag, uint8_t>(
                problem.retainedIndices.rawEncodedBackingStorage(),
                firstOffset + retainedAxisStride);
            (void)twoOfFourMetadataNibble(first, second);
        }
    }
    return metadataShape;
}

Shape validateTwoOfFourMetadataRequest(const TwoOfFourMetadataInvocation& request) {
    const Shape metadataShape = validateTwoOfFourMetadataProblem(request);
    if (request.metadata.type() != ScalarType::UInt8)
        throw std::invalid_argument("Two-of-four metadata output type must be UInt8.");
    if (request.metadata.shape() != metadataShape)
        throw std::invalid_argument("Two-of-four metadata output shape mismatch.");
    rejectTensorStorageOverlap(request.retainedIndices, request.metadata,
                               "Two-of-four metadata output overlaps the retained-index storage.");
    return metadataShape;
}

uint8_t twoOfFourMetadataNibble(uint8_t first, uint8_t second) {
    if (first >= 4 || second >= 4 || first >= second)
        throw std::invalid_argument(
            "Two-of-four retained positions must be increasing values in [0, 3].");
    return static_cast<uint8_t>(first | static_cast<uint8_t>(second << 2));
}
}  // namespace roc::host_numerics::detail
