// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <roc/hostnumerics/layout.hpp>
#include <roc/hostnumerics/permutation.hpp>
#include <roc/hostnumerics/shape.hpp>

namespace roc::hostnumerics {

// Describes a contiguous copy made by padding a natural tensor, reshaping
// it into factors, and permuting those factors. All shapes use last-dimension-
// fastest order; order lists the reshaped axes from slowest to fastest.
//
// Padding preserves natural coordinates. paddedShape must have the natural
// rank and contain every natural extent. reshapedShape must have the same
// element count as paddedShape. Its rank can differ from the natural rank.
// Construction validates these contracts and size/stride overflow. As with
// Layout, extents and contiguous strides must fit ptrdiff_t. The caller chooses
// the padding value when materializing the result.
class ReshapePermutation {
   public:
    ReshapePermutation(Shape naturalShape, Shape paddedShape, Shape reshapedShape,
                       Permutation order);

    static ReshapePermutation identity(const Shape& shape);

    const Shape& naturalShape() const {
        return m_naturalShape;
    }
    const Shape& paddedShape() const {
        return m_paddedLayout.shape();
    }
    const Shape& reshapedShape() const {
        return m_physicalLayout.shape();
    }
    const Permutation& order() const {
        return m_order;
    }
    size_t naturalElementCount() const {
        return m_naturalShape.elementCount();
    }
    size_t physicalElementCount() const {
        return reshapedShape().elementCount();
    }

    // Maps a flat natural index to its position in the contiguous result.
    // Requires naturalIndex < naturalElementCount().
    size_t physicalIndex(size_t naturalIndex) const noexcept;

   private:
    Shape m_naturalShape;
    Layout m_paddedLayout;
    Layout m_physicalLayout;
    Permutation m_order;
};

}  // namespace roc::hostnumerics
