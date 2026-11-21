// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <roc/hostnumerics/reshape_permutation.hpp>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "parallel_chunks.hpp"

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

ReshapePermutation& ReshapePermutation::operator=(ReshapePermutation other) noexcept {
    m_naturalShape = std::move(other.m_naturalShape);
    m_paddedLayout = std::move(other.m_paddedLayout);
    m_physicalLayout = std::move(other.m_physicalLayout);
    m_order = std::move(other.m_order);
    return *this;
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

namespace {
// Copies natural elements [begin, end). Consecutive elements of a natural row
// are consecutive in the padded tensor, so along a row the physical offset
// advances like a mixed-radix counter whose digits are the reshaped factor
// coordinates. Divisions are needed only where the range starts or enters a row.
void copyNaturalRange(const std::byte* input, std::byte* output, size_t elementSize,
                      const ReshapePermutation& transform, std::span<const size_t> strides,
                      std::span<size_t> digits, size_t begin, size_t end) noexcept {
    const Shape& natural = transform.naturalShape();
    const Shape& padded = transform.paddedShape();
    const Shape& factors = transform.reshapedShape();
    const size_t rowLength = natural.rank() == 0 ? 1 : natural[natural.rank() - 1];
    for (size_t index = begin; index < end;) {
        size_t remaining = index;
        size_t paddedIndex = 0;
        size_t paddedStride = 1;
        for (size_t axis = natural.rank(); axis-- > 0;) {
            paddedIndex += remaining % natural[axis] * paddedStride;
            remaining /= natural[axis];
            paddedStride *= padded[axis];
        }
        size_t offset = 0;
        for (size_t axis = factors.rank(); axis-- > 0;) {
            digits[axis] = paddedIndex % factors[axis];
            paddedIndex /= factors[axis];
            offset += digits[axis] * strides[axis];
        }
        const size_t rowEnd = std::min(end, index - index % rowLength + rowLength);
        for (; index < rowEnd; ++index) {
            std::memcpy(output + offset * elementSize, input + index * elementSize, elementSize);
            for (size_t axis = factors.rank(); axis-- > 0;) {
                offset += strides[axis];
                if (++digits[axis] < factors[axis]) break;
                offset -= factors[axis] * strides[axis];
                digits[axis] = 0;
            }
        }
    }
}
}  // namespace

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
    const Shape& factors = transform.reshapedShape();
    const Layout physicalLayout = Layout::contiguous(factors, transform.order());
    std::vector<size_t> strides(factors.rank());
    for (size_t axis = 0; axis < strides.size(); ++axis)
        strides[axis] = static_cast<size_t>(physicalLayout.stride(axis));
    const size_t chunkCount = parallelChunkCount(count);
    // Each chunk updates its counter for every element, so keep the counters of
    // different chunks at least one 64-byte cache line apart.
    constexpr size_t wordsPerCacheLine = 64 / sizeof(size_t);
    const size_t digitStride = factors.rank() + wordsPerCacheLine;
    std::vector<size_t> digits(chunkCount * digitStride);
    // Natural elements have distinct destinations; padding keeps its initial value.
    parallelForChunks(count, chunkCount, [&](size_t chunk, size_t begin, size_t end) noexcept {
        copyNaturalRange(input, output, elementSize, transform, strides,
                         std::span(digits).subspan(chunk * digitStride, factors.rank()), begin,
                         end);
    });
}

}  // namespace roc::hostnumerics
