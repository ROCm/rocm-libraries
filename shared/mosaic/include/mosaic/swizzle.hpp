// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <cstddef>

#include "mosaic/config.hpp"

namespace mosaic {

// =====================================================================
// Swizzle<B, M, S>
// ---------------------------------------------------------------------
// XOR-based bit-swizzle on a flat offset:
//
//   mask     = ((offset >> (M + S)) & ((1 << B) - 1)) << M
//   swizzled = offset ^ mask
//
// Reading the parameters:
//   B  = number of "high-order" bits we XOR with (one per swizzle "row")
//   M  = base bit position the XOR mask is applied at (low bits below M
//        are untouched)
//   S  = shift between the source bits we read (high) and the target
//        bits we XOR into (low)
//
// Composed onto an LDS layout via `compose(swizzle, base_layout)`, this
// re-permutes which physical LDS bank each (m, k) coordinate lands in,
// spreading consecutive accesses across distinct banks.
//
// XOR bit-swizzle as described in Cecka / CUTLASS cute/swizzle.hpp
// (https://github.com/NVIDIA/cutlass/blob/main/include/cute/swizzle.hpp).
// This file is an independent single-direction implementation of the same
// masking formula: no negative-shift direction, no MixedBits, and no
// up/downcast machinery.
// =====================================================================

template <int B, int M, int S>
struct Swizzle {
    MOSAIC_HOST_DEVICE constexpr std::size_t operator()(std::size_t offset) const {
        constexpr std::size_t bit_msk = (std::size_t{1} << B) - 1;
        std::size_t const mask = ((offset >> (M + S)) & bit_msk) << M;
        return offset ^ mask;
    }

    // Reported codomain size — used by ComposedLayout when computing
    // total addressable LDS offsets.
    MOSAIC_HOST_DEVICE constexpr std::size_t size() const {
        return std::size_t{1} << (B + M + S);
    }
};

// =====================================================================
// RotateSwizzle<BlockBits, RowShift>
// ---------------------------------------------------------------------
// Additive rotation in a power-of-2 column block:
//
//   block_mask = (1 << BlockBits) - 1
//   col        = offset & block_mask
//   row        = offset >> BlockBits
//   rotated    = (col + (row >> RowShift) * 2) & block_mask
//   result     = (offset & ~block_mask) | rotated
//
// Concretely: within each "row" of `1 << BlockBits` LDS columns, the
// column index gets rotated by an amount that depends on which row pair
// you're in. After `1 << (BlockBits + RowShift + 1)` rows the rotation
// cycles back to zero. Picks the same "every column visits every bank
// across the row-block" property as the XOR swizzle, just via a
// different permutation.
// TODO(sean): This hasn't been validated, but can be illustrative how to represent other possible
// swizzle patterns.
// =====================================================================

template <int BlockBits, int RowShift>
struct RotateSwizzle {
    MOSAIC_HOST_DEVICE constexpr std::size_t operator()(std::size_t offset) const {
        constexpr std::size_t block_mask = (std::size_t{1} << BlockBits) - 1;
        std::size_t const col = offset & block_mask;
        std::size_t const row = offset >> BlockBits;
        std::size_t const rotated = (col + (row >> RowShift) * 2) & block_mask;
        return (offset & ~block_mask) | rotated;
    }

    // The rotation is well-defined for any offset; report a max-out
    // size as a placeholder so `compose(rotate, base)` can derive its
    // size from the inner layout instead.
    MOSAIC_HOST_DEVICE constexpr std::size_t size() const {
        return ~std::size_t{0};
    }
};

}  // namespace mosaic
