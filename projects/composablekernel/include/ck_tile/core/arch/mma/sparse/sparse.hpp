// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck_tile::core::arch::mma {

/**
 * @enum SparseCompressionIndex
 * @brief Indicates which 8 bit part should be used as compression index.
 * Note that the LE16 option can select the lowest 16 bits instead.
 * \see DefaultSparseMfmaCtrlFlags
 */
enum struct SparseCompressionIndex : int
{
    LE16   = -1, // Uses bits [15:0] (lowest 16 bits)
    FIRST  = 0,  // Uses bits [7:0] (lowest 8 bits)
    SECOND = 1,  // Uses bits [15:8] (second 8 bits)
    THIRD  = 2,  // Uses bits [23:16] (third 8 bits)
    FOURTH = 3,  // Uses bits [31:24] (highest 8 bits)
};

namespace sparse::detail {
struct BuiltinParams
{
    int Override16BitDefaultMask;
    int ByteIndexToOverride;
};

template <SparseCompressionIndex Idx>
struct get_builtin_params
{
    private:
    static constexpr BuiltinParams getBuiltinParams()
    {
        if constexpr(Idx == SparseCompressionIndex::LE16)
        {
            return {.Override16BitDefaultMask = 0, .ByteIndexToOverride = 0};
        }
        else
        {
            return {.Override16BitDefaultMask = 1, .ByteIndexToOverride = static_cast<int>(Idx)};
        }
    }

    public:
    static constexpr BuiltinParams value = getBuiltinParams();
};

} // namespace sparse::detail

} // namespace ck_tile::core::arch::mma

// Include sparse MFMA traits and architecture-specific implementations
#include "ck_tile/core/arch/mma/sparse/mfma/sparse_gfx9.hpp"
#include "ck_tile/core/arch/mma/sparse/wmma/sparse_gfx12.hpp"
#include "ck_tile/core/arch/mma/sparse/sparse_transforms.hpp"
#include "ck_tile/core/arch/mma/sparse/sparse_selector.hpp"
