// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cassert>
#include <cstring>
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

ReshapePermutation ReshapePermutation::tiled(const Shape& naturalShape, const Shape& tileShape) {
    if (naturalShape.rank() != tileShape.rank())
        throw std::runtime_error("Tile shape must have the natural rank.");
    std::vector<size_t> factors;
    std::vector<size_t> order;
    for (size_t axis = 0; axis < naturalShape.rank(); ++axis) {
        if (tileShape[axis] == 0 || naturalShape[axis] % tileShape[axis] != 0)
            throw std::runtime_error("Tile extents must be nonzero and divide the natural shape.");
        factors.push_back(naturalShape[axis] / tileShape[axis]);
        factors.push_back(tileShape[axis]);
        order.push_back(2 * axis);
    }
    for (size_t axis = 0; axis < naturalShape.rank(); ++axis) order.push_back(2 * axis + 1);
    return {naturalShape, naturalShape, Shape(std::move(factors)), Permutation(std::move(order))};
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

void detail::copyReshapePermutationBytes(const std::byte* input, std::byte* output,
                                         size_t elementSize, const ReshapePermutation& transform) {
    const size_t count = transform.naturalElementCount();
    if (count == 0) return;
    // Reshaping alone preserves flat order without padding or axis reordering.
    if (transform.naturalShape() == transform.paddedShape() &&
        std::is_sorted(transform.order().begin(), transform.order().end())) {
        std::memcpy(output, input, count * elementSize);
        return;
    }
    // Natural elements have distinct destinations; padding keeps its initial value.
    for (size_t index = 0; index < count; ++index)
        std::memcpy(output + transform.physicalIndex(index) * elementSize,
                    input + index * elementSize, elementSize);
}

}  // namespace roc::hostnumerics
