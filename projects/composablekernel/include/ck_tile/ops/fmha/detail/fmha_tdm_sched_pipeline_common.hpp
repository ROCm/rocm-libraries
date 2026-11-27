// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// K/V LDS buffers of one iteration: K and V double buffers swap every iteration.
template <typename KDataType, typename VDataType>
struct FmhaTdmSchedKvBuffers
{
    KDataType* k_cur;
    KDataType* k_next;
    VDataType* v_next;
    VDataType* v_cur;
};

// Steps the 16-bit and FP8 qr_tdm sched pipelines share outside their main loops.
struct FmhaTdmSchedPipelineCommon
{
    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};

    // Applies the policy's (enable, amount, interval) LDS padding to a TDM config.
    template <typename Padding>
    CK_TILE_HOST_DEVICE static void SetTdmPadding(TDMConfig& config, Padding padding)
    {
        config.pad_enable              = padding[I0];
        config.pad_config.pad_amount   = padding[I1];
        config.pad_config.pad_interval = padding[number<2>{}];
    }

    // Natural-log LSE of a row from its log2-domain max, softmax scale and row sum.
    CK_TILE_HOST_DEVICE static float RowLse(float m, float scale, float l)
    {
        return m * scale / C_LOG2E + log(l);
    }

    // Reciprocal row sum; GuardEmpty maps an empty row (sum 0) to 0.
    template <bool GuardEmpty>
    CK_TILE_HOST_DEVICE static float RowInvSum(float l)
    {
        if constexpr(GuardEmpty)
            return l == 0.f ? 0.f : 1 / l;
        else
            return 1 / l;
    }

    // K/V buffers of an even or odd iteration.
    template <typename KDataType, typename VDataType>
    CK_TILE_HOST_DEVICE static FmhaTdmSchedKvBuffers<KDataType, VDataType>
    SelectKvBuffers(bool is_even_loop, void* k0, void* k1, void* v0, void* v1)
    {
        return {is_even_loop ? static_cast<KDataType*>(k0) : static_cast<KDataType*>(k1),
                is_even_loop ? static_cast<KDataType*>(k1) : static_cast<KDataType*>(k0),
                is_even_loop ? static_cast<VDataType*>(v1) : static_cast<VDataType*>(v0),
                is_even_loop ? static_cast<VDataType*>(v0) : static_cast<VDataType*>(v1)};
    }
};

} // namespace ck_tile
