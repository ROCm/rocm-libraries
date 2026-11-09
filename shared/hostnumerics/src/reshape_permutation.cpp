// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cassert>
#include <roc/hostnumerics/reshape_permutation.hpp>
#include <stdexcept>
#include <utility>

namespace roc::hostnumerics {

ReshapePermutation::ReshapePermutation(Shape naturalShape, Shape paddedShape, Shape reshapedShape,
                                       Permutation order)
    : m_naturalShape(std::move(naturalShape)),
      m_paddedLayout(Layout::contiguousLastDimensionFastest(paddedShape)),
      m_physicalLayout(Layout::contiguous(reshapedShape, order)),
      m_order(std::move(order)) {
    if (m_naturalShape.rank() != paddedShape.rank())
        throw std::runtime_error("Padding must preserve the natural rank.");
    for (size_t axis = 0; axis < paddedShape.rank(); ++axis)
        if (m_naturalShape[axis] > paddedShape[axis])
            throw std::runtime_error("Padding cannot reduce an extent.");
    if (paddedShape.elementCount() != reshapedShape.elementCount())
        throw std::runtime_error("Reshape must preserve the padded element count.");
}

ReshapePermutation ReshapePermutation::identity(const Shape& shape) {
    return {shape, shape, shape, Permutation::identity(shape.rank())};
}

size_t ReshapePermutation::physicalIndex(size_t naturalIndex) const noexcept {
    assert(naturalIndex < naturalElementCount());
    size_t paddedIndex = 0;
    for (size_t axis = m_naturalShape.rank(); axis-- > 0;) {
        paddedIndex +=
            naturalIndex % m_naturalShape[axis] * static_cast<size_t>(m_paddedLayout.stride(axis));
        naturalIndex /= m_naturalShape[axis];
    }
    size_t physicalIndex = 0;
    for (size_t axis = reshapedShape().rank(); axis-- > 0;) {
        physicalIndex += paddedIndex % reshapedShape()[axis] *
                         static_cast<size_t>(m_physicalLayout.stride(axis));
        paddedIndex /= reshapedShape()[axis];
    }
    return physicalIndex;
}

}  // namespace roc::hostnumerics
