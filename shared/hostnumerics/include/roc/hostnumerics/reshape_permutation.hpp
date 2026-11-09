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
// the padding value when materializing the result. A scalar has one element;
// an empty tensor has no valid natural index.
//
// For example, pad two rows of three values to two rows of four, split each
// row into two pairs, then store the first pair from each row before the second:
//
//   const ReshapePermutation transform({2, 3}, {2, 4}, {2, 2, 2},
//                                      Permutation{1, 0, 2});
//   const std::array<int, 6> input{1, 2, 3, 4, 5, 6};
//   std::vector<int> output(transform.physicalElementCount(), 0);
//   for (size_t i = 0; i < input.size(); ++i)
//       output[transform.physicalIndex(i)] = input[i];
//   // output is {1, 2, 4, 5, 3, 0, 6, 0}.
//
// The same copy can be written with explicit factor loops. Padded coordinates
// are skipped so that their output elements retain the chosen padding value:
//
//   for (size_t pair = 0; pair < 2; ++pair)          // reshaped axis 1
//       for (size_t row = 0; row < 2; ++row)        // reshaped axis 0
//           for (size_t inPair = 0; inPair < 2; ++inPair) { // reshaped axis 2
//               const size_t column = 2 * pair + inPair;
//               if (column < 3)
//                   output[(2 * pair + row) * 2 + inPair] = input[3 * row + column];
//           }
class ReshapePermutation {
   public:
    ReshapePermutation(Shape naturalShape, Shape paddedShape, Shape reshapedShape,
                       Permutation order);

    // Keeps the shape and storage order exactly: no padding or rearrangement.
    // For example, identity({2, 3}) has six physical elements and maps i to i.
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
    // Equals paddedShape().elementCount(). Construction rejects a reshape
    // with a different element count, in both Debug and Release builds.
    size_t physicalElementCount() const {
        return reshapedShape().elementCount();
    }

    // Maps a flat natural index to its position in the contiguous result.
    // Requires naturalIndex < naturalElementCount().
    // In the example above, physicalIndex(2) == 4 and physicalIndex(3) == 2:
    // input[2] is the first row's third value, input[3] the second row's first.
    size_t physicalIndex(size_t naturalIndex) const noexcept;

   private:
    Shape m_naturalShape;
    Layout m_paddedLayout;
    Layout m_physicalLayout;
    Permutation m_order;
};

}  // namespace roc::hostnumerics
