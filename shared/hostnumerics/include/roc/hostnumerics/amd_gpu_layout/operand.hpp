// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <optional>
#include <type_traits>
#include <vector>

namespace roc::hostnumerics::amd_gpu_layout {

// Extents of a rank-two operand stored with K contiguous: element (k, mn) is at
// k + extents.k * mn.
struct OperandExtents {
    size_t k = 0;
    size_t mn = 0;
};

// Lane and VGPR order in which an AMD GPU kernel loads each tileK x tileMN tile.
// `tileMN` must be 32 or 64 and `tileK` a nonzero multiple of 4. `subTileK`
// must be nonzero, and 2 or 4 when `tileMN` is 32.
struct OperandSwizzle {
    size_t tileK = 0;
    size_t tileMN = 0;
    size_t subTileK = 0;
};

// Tile extents to store contiguously, each with K contiguous.
struct OperandTile {
    size_t k = 0;
    size_t mn = 0;
};

namespace detail {

// Compiled bridge for the type-independent validation and copying used by
// copyOperandToPhysicalLayout. A null output requests only validation.
void copyOperandBytes(const std::byte* operand, size_t elementCount, size_t elementSize,
                      OperandExtents extents, const std::optional<OperandSwizzle>& swizzle,
                      const std::optional<OperandTile>& tile, std::byte* output);

}  // namespace detail

/**
 * @brief Copies a rank-two host operand into the storage order an AMD GPU
 *        kernel reads.
 *
 * With `swizzle`, elements are permuted into the kernel's lane and VGPR order
 * within each swizzle tile. With `tile`, each tile is stored contiguously, tiles
 * ordered K-tile first, and a swizzle tile must divide it. With neither, the
 * result is a copy of `operand`. Logical coordinates are unchanged; only their
 * storage offsets move.
 *
 * `T` must be trivially copyable. The returned vector owns its storage.
 *
 * @throws std::runtime_error if `operand` does not hold extents.k * extents.mn
 * elements, a swizzle or tile violates its constraints, an extent is not
 * divisible by its tile (including when the operand is empty), or a size
 * calculation overflows `size_t`.
 */
template <typename T>
std::vector<T> copyOperandToPhysicalLayout(const std::vector<T>& operand, OperandExtents extents,
                                           const std::optional<OperandSwizzle>& swizzle,
                                           const std::optional<OperandTile>& tile) {
    static_assert(std::is_trivially_copyable_v<T>,
                  "AMD GPU physical-layout conversions require trivially copyable elements");

    const auto* operandBytes = reinterpret_cast<const std::byte*>(operand.data());
    detail::copyOperandBytes(operandBytes, operand.size(), sizeof(T), extents, swizzle, tile,
                             nullptr);
    std::vector<T> output(operand.size());
    detail::copyOperandBytes(operandBytes, operand.size(), sizeof(T), extents, swizzle, tile,
                             reinterpret_cast<std::byte*>(output.data()));
    return output;
}

}  // namespace roc::hostnumerics::amd_gpu_layout
