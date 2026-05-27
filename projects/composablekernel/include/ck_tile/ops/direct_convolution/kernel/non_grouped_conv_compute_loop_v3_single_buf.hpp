// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Non-grouped convolution compute loop v3 — single-buffer LDS input variant.
//
// Mirrors the double-buffer sibling in non_grouped_conv_compute_loop_v3.hpp
// but allocates ONE input LDS buffer instead of two. This halves the input
// LDS pool (the dominant term in UNIFIED_LDS_SIZE on most instances) and
// is intended to relieve the "Insufficient CU LDS" scheduling stall.
//
// The cost is an extra __syncthreads per row: with a single buffer, the
// next-row fetch cannot start until every wave has finished reading the
// current row from LDS.
//
// Structure per input row:
//   wait_vmcnt<0>
//   __syncthreads              // make async load visible to all readers
//   for S in 0..kw:
//     each wave reads its C-section from LDS, then kh MFMAs
//   __syncthreads              // all consumers done; safe to refill
//   if y+1 < hi:
//     fetch_tile_to_lds(0)     // refill same buffer
//   if flush:
//     cross_wave_reduce + flush

#pragma once

#include "ck_tile/ops/direct_convolution/utils/common.hpp"
#include "ck_tile/ops/direct_convolution/utils/detail.hpp"
#include "ck_tile/ops/direct_convolution/utils/memory.hpp"
#include "ck_tile/ops/direct_convolution/kernel/non_grouped_conv_compute_loop_v3.hpp"
#include <hip/hip_runtime.h>

namespace ck_tile::direct_conv::conv_32c_tile::v3
{

template <typename TC,
          auto cfg,
          typename MfmaFn,
          typename BlockCoordsT,
          typename InputLoaderT,
          typename WeightLoaderT,
          typename OutputWriterT,
          typename ElementType = _Float16>
__device__ void conv_compute_loop_v3_single_buf(const ElementType* __restrict__ in,
                                                 const ElementType* __restrict__ wei,
                                                 ElementType* __restrict__ out,
                                                 int N,
                                                 int C,
                                                 int K,
                                                 int hi,
                                                 int wi,
                                                 int ho,
                                                 int wo,
                                                 int py,
                                                 int px)
{
    using AccType            = typename MfmaFn::acc_type;
    constexpr int ACC_FLOATS = sizeof(AccType) / sizeof(float);

    constexpr bool is_dgrad  = (cfg.direction == Direction::Dgrad);
    constexpr int  NUM_WAVES = cfg.waves_per_wg;

    // --- LDS layout: ONE input buffer instead of two ---
    // The reduction buffer sits immediately after the (single) input buffer.
    // Unified pool: max(weight_all_waves, input_single + reduce).
    static constexpr int INPUT_TOTAL      = TC::INPUT_LDS_BUFFER_SIZE_C8;
    static constexpr int REDUCE_LDS_UINT4 = NUM_WAVES * 64 * ACC_FLOATS / 4;
    static constexpr int IO_REDUCE_LDS    = INPUT_TOTAL + REDUCE_LDS_UINT4;

    static constexpr int WEIGHT_LDS       = TC::WEIGHT_LDS_ALL_WAVES;
    static constexpr int UNIFIED_LDS_SIZE =
        (WEIGHT_LDS > IO_REDUCE_LDS) ? WEIGHT_LDS : IO_REDUCE_LDS;

    __shared__ uint4 lds_buf[UNIFIED_LDS_SIZE];

    float* reduce_lds = reinterpret_cast<float*>(lds_buf + INPUT_TOTAL);

    // --- Coordinate setup ---
    const int C_in  = is_dgrad ? K : C;
    const int C_out = is_dgrad ? C : K;

    BlockCoordsT bc(C_in, C_out);
    if(bc.block_n >= N)
        return;

    const int weight_block_k = bc.block_k_start;
    const int wave_id        = static_cast<int>(threadIdx.x) / 64;

    constexpr int y_padding = 0;
    constexpr int stride    = 1;
    constexpr int dilation  = 1;

    // --- Weight prologue: all waves load their own C-slice in parallel ---
    WeightLoaderT wl;
    constexpr int CPG                = cfg.channels_per_group();
    constexpr int WEIGHT_SLICE_UINT4 = TC::WEIGHT_LDS_SIZE_UINT4;

    uint4* wave_weight_lds = lds_buf + wave_id * WEIGHT_SLICE_UINT4;

    if constexpr(is_dgrad)
        WeightLoaderT::load_kyxc_to_lds_dgrad_wave(
            wave_weight_lds, wei, wave_id * CPG, weight_block_k, C);
    else
        WeightLoaderT::load_kyxc_to_lds_wave(
            wave_weight_lds, wei, weight_block_k, wave_id, C);

    wait_vmcnt<0>();
    __syncthreads();

    wl.read_from_lds(wave_weight_lds);

    __syncthreads();

    // --- Construct InputLoader and OutputWriter ---
    InputLoaderT il(bc, lds_buf, in, hi, wi, px, y_padding,
                    dilation, dilation, stride, stride);
    uint4* output_staging_lds = lds_buf + INPUT_TOTAL;
    OutputWriterT ow(bc, output_staging_lds, out, ho, wo);

    // --- Prefetch first input row into the single buffer (slot 0) ---
    il.prefetch_tile_to_lds(0);

    // --- Circular accumulator buffer ---
    constexpr AccType Zero{};
    AccType acc[cfg.kh];
    static_for<cfg.kh>(
        [&]<int I>()
        { acc[I] = Zero; });

    MfmaFn mfma_fn{};

    auto reduce_and_flush = [&](AccType& slot, int p_out)
    {
        cross_wave_reduce<NUM_WAVES>(slot, reduce_lds, wave_id);
        ow.flush(slot, p_out, wave_id);
        slot = Zero;
    };

    // --- Main loop: process input rows in batches of cfg.kh ---
    for(int y_base = 0; y_base + cfg.kh <= hi; y_base += cfg.kh)
    {
        static_for<cfg.kh>(
            [&]<int Y_LOCAL>()
            {
                int y = y_base + Y_LOCAL;

                wait_vmcnt<0>();
                __syncthreads();

                // MFMA core: each wave reads its C-section from LDS slot 0.
                static_for<cfg.kw>(
                    [&]<int S>()
                    {
                        typename InputLoaderT::input_type input_reg;
                        il.read_from_lds(input_reg, S, 0);

                        static_for<cfg.kh>(
                            [&]<int R>()
                            {
                                constexpr int p_idx =
                                    (Y_LOCAL - R + cfg.kh) % cfg.kh;
                                if constexpr(is_dgrad)
                                    acc[p_idx] = mfma_fn(
                                        wl.template get_transposed<R, S>(),
                                        input_reg,
                                        acc[p_idx]);
                                else
                                    acc[p_idx] = mfma_fn(
                                        wl.template get<R, S>(),
                                        input_reg,
                                        acc[p_idx]);
                            });
                    });

                // Refill same buffer: must wait until every wave has finished
                // reading from LDS, otherwise the async load can overwrite
                // data still being consumed.
                if((y + 1) < hi)
                {
                    __syncthreads();
                    il.fetch_tile_to_lds(0);
                }

                // Flush completed output row via cross-wave LDS reduction.
                constexpr int P_IDX_FLUSH = (Y_LOCAL + 1) % cfg.kh;
                int p_out = y + py - (cfg.kh - 1);
                if(p_out >= 0 && p_out < ho)
                    reduce_and_flush(acc[P_IDX_FLUSH], p_out);
                else
                    acc[P_IDX_FLUSH] = Zero;
            });
    }

    // --- Remainder loop: hi % kh leftover rows ---
    {
        int y_rem_base = (hi / cfg.kh) * cfg.kh;
        static_for<cfg.kh>(
            [&]<int Y_LOCAL>()
            {
                if(Y_LOCAL >= hi % cfg.kh)
                    return;
                int y = y_rem_base + Y_LOCAL;

                wait_vmcnt<0>();
                __syncthreads();

                static_for<cfg.kw>(
                    [&]<int S>()
                    {
                        typename InputLoaderT::input_type input_reg;
                        il.read_from_lds(input_reg, S, 0);

                        static_for<cfg.kh>(
                            [&]<int R>()
                            {
                                constexpr int p_idx =
                                    (Y_LOCAL - R + cfg.kh) % cfg.kh;
                                if constexpr(is_dgrad)
                                    acc[p_idx] = mfma_fn(
                                        wl.template get_transposed<R, S>(),
                                        input_reg,
                                        acc[p_idx]);
                                else
                                    acc[p_idx] = mfma_fn(
                                        wl.template get<R, S>(),
                                        input_reg,
                                        acc[p_idx]);
                            });
                    });

                if((y + 1) < hi)
                {
                    __syncthreads();
                    il.fetch_tile_to_lds(0);
                }

                constexpr int P_FLUSH = (Y_LOCAL + 1) % cfg.kh;
                int p_out = y + py - (cfg.kh - 1);
                if(p_out >= 0 && p_out < ho)
                    reduce_and_flush(acc[P_FLUSH], p_out);
                else
                    acc[P_FLUSH] = Zero;
            });
    }

    // --- Tail flush: output rows not flushed by the main/remainder loops ---
    for(int p_out = hi - cfg.kh + 1 + py; p_out < ho; p_out++)
    {
        int p_idx = (p_out - py + cfg.kh) % cfg.kh;
        AccType slot;
        dispatch<cfg.kh>(p_idx,
                         [&]<int P>()
                         {
                             slot   = acc[P];
                             acc[P] = Zero;
                         });
        reduce_and_flush(slot, p_out);
    }
}

} // namespace ck_tile::direct_conv::conv_32c_tile::v3
