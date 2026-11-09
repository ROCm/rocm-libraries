// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <roc/hostnumerics/layout.hpp>
#include <roc/hostnumerics/permutation.hpp>
#include <roc/hostnumerics/shape.hpp>
#include <span>
#include <stdexcept>
#include <type_traits>

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
//   natural shape: [2, 3]; padded shape: [2, 4]
//   reshaped factors: [row: 2, pair: 2, position in pair: 2]
//   factor order: [pair, row, position in pair], or [1, 0, 2]
//   input: [1, 2, 3, 4, 5, 6]
//   output with zero padding: [1, 2, 4, 5, 3, 0, 6, 0]
//
// The same copy can be written with explicit factor loops. Padded coordinates
// are skipped so that their output elements retain the chosen padding value:
//
//   for each pair, then each row, then each position in the pair:
//       column = 2 * pair + position
//       if column < 3:
//           output[(2 * pair + row) * 2 + position] = input[3 * row + column]
//
// See ReshapePermutationTutorial.PadsAndPermutesFactors for the C++ example.
class ReshapePermutation {
   public:
    ReshapePermutation(Shape naturalShape, Shape paddedShape, Shape reshapedShape,
                       Permutation order);
    ReshapePermutation(const ReshapePermutation&) = default;
    ReshapePermutation(ReshapePermutation&&) noexcept = default;

    // Copy all metadata before replacing this value so an allocation failure
    // cannot leave shapes and order inconsistent. Self-move preserves the value.
    ReshapePermutation& operator=(ReshapePermutation other) noexcept;

    // Keeps the shape and storage order exactly: no padding or rearrangement.
    // For example, shape [2, 3] has six physical elements and maps i to i.
    static ReshapePermutation identity(const Shape& shape);

    // Store tiles contiguously, in last-dimension-fastest order both among tiles
    // and within each tile. Shapes must have equal rank; every tile extent must
    // be nonzero and divide its natural extent exactly, including empty tensors.
    // No padding is added. For example, shape [4, 4] with tile shape [2, 2]
    // reshapes to [2, 2, 2, 2] and orders the factors as [0, 2, 1, 3].
    static ReshapePermutation tiled(const Shape& naturalShape, const Shape& tileShape);

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
    // In the example above, natural indices 2 and 3 map to physical offsets 4 and 2:
    // input[2] is the first row's third value, input[3] the second row's first.
    size_t physicalIndex(size_t naturalIndex) const noexcept;

   private:
    Shape m_naturalShape;
    Layout m_paddedLayout;
    Layout m_physicalLayout;
    Permutation m_order;
};

namespace detail {
void copyReshapePermutationBytes(const std::byte* input, std::byte* output, size_t elementSize,
                                 const ReshapePermutation& transform);
}  // namespace detail

// Copy natural elements to their physical offsets, preserving their object
// representations. Input and output must not overlap; T must be trivially copyable.
// Specify T explicitly to accept containers and fixed-extent or mutable spans.
// Both spans must have exactly the transform's respective element counts, or
// std::runtime_error is thrown. Padding is left untouched, so the caller can
// initialize it to any value before copying. The output has last-dimension-fastest
// storage in the permuted factor order.
//
//   fill output with the chosen padding value
//   for each natural index i:
//       copy the bytes of input[i] to output[physicalIndex(i)]
//
// See the ReshapePermutationTutorial tests for padded and tiled copies.
template <typename T>
void copyWithReshapePermutation(std::span<const T> input, std::span<T> output,
                                const ReshapePermutation& transform) {
    static_assert(std::is_trivially_copyable_v<T>,
                  "Reshape/permutation copies require trivially copyable elements.");
    if (input.size() != transform.naturalElementCount() ||
        output.size() != transform.physicalElementCount())
        throw std::runtime_error("Reshape/permutation storage does not match its shape.");
    detail::copyReshapePermutationBytes(reinterpret_cast<const std::byte*>(input.data()),
                                        reinterpret_cast<std::byte*>(output.data()), sizeof(T),
                                        transform);
}

}  // namespace roc::hostnumerics
