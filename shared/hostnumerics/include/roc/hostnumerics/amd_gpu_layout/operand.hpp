// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <optional>
#include <type_traits>
#include <vector>

namespace roc::hostnumerics::amd_gpu_layout {

// Extents of a rank-two operand stored with the reduction dimension contiguous.
// The non-reduction dimension is M for A, or N for B. Element (r, n) is at
// r + extents.reduction * n. Extents count storage elements, not bytes.
struct OperandExtents {
    size_t reduction = 0;
    size_t nonReduction = 0;
};

// Lane and VGPR order in which an AMD GPU kernel loads each
// tileReduction x tileNonReduction tile. These parameters select the factors and
// axis order of a ReshapePermutation, as MX scale presets do.
// tileNonReduction must be 32 or 64; tileReduction must be a nonzero multiple
// of 4. subTileReduction must be nonzero, and 2 or 4 when tileNonReduction is 32.
struct OperandSwizzle {
    size_t tileReduction = 0;
    size_t tileNonReduction = 0;
    size_t subTileReduction = 0;
};

// Tile extents to store contiguously, with the reduction dimension contiguous.
struct OperandTile {
    size_t reduction = 0;
    size_t nonReduction = 0;
};

namespace detail {

// Type-independent implementation of copyOperandToPhysicalLayout. `operand`
// and `output` each hold `elementCount` elements of `elementSize` bytes.
void copyOperandBytes(const std::byte* operand, std::byte* output, size_t elementCount,
                      size_t elementSize, OperandExtents extents,
                      const std::optional<OperandSwizzle>& swizzle,
                      const std::optional<OperandTile>& tile);

}  // namespace detail

/**
 * @brief Copies a rank-two host operand into the storage order an AMD GPU
 *        kernel reads.
 *
 * With `swizzle`, elements are permuted into the kernel's lane and VGPR order
 * within each swizzle tile. With `tile`, each tile is stored contiguously and
 * the reduction tile index varies fastest. A swizzle tile must divide it.
 * With neither, the result is a copy of `operand`. Logical coordinates are
 * unchanged; only their storage offsets move.
 *
 * The implementation builds a ReshapePermutation for both tiling and swizzling.
 * Its natural and padded shapes are both [nonReduction, reduction]: operands
 * receive no padding. Reshaping splits the extents into tile, lane and VGPR
 * factors; permuting those factors gives the physical order. For example, a
 * 4 x 4 operand tiled into 2 x 2 blocks uses reshaped shape [2, 2, 2, 2] and
 * order {0, 2, 1, 3}. The descriptor is constructed inside the copy operation;
 * an MX scale plan also exposes its descriptor so callers can size padded storage.
 *
 * `T` must be trivially copyable.
 *
 * @throws std::runtime_error if `operand` does not hold
 * extents.reduction * extents.nonReduction elements, a swizzle or tile violates
 * its constraints, an extent is not
 * divisible by its tile (including when the operand is empty), or a size
 * calculation overflows `size_t`.
 */
template <typename T>
std::vector<T> copyOperandToPhysicalLayout(const std::vector<T>& operand, OperandExtents extents,
                                           const std::optional<OperandSwizzle>& swizzle,
                                           const std::optional<OperandTile>& tile) {
    static_assert(std::is_trivially_copyable_v<T>,
                  "AMD GPU physical-layout conversions require trivially copyable elements");

    std::vector<T> output(operand.size());
    detail::copyOperandBytes(reinterpret_cast<const std::byte*>(operand.data()),
                             reinterpret_cast<std::byte*>(output.data()), operand.size(), sizeof(T),
                             extents, swizzle, tile);
    return output;
}

}  // namespace roc::hostnumerics::amd_gpu_layout
