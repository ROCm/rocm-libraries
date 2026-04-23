// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/direct_convolution/utils/common.hpp"
#include "ck_tile/ops/direct_convolution/utils/detail.hpp"

namespace ck_tile {
namespace direct_conv {

// Shared compute loop for grouped convolution kernels (Fprop and Dgrad).
//
// This function implements the main device-side compute loop shared by both
// 4-channel and 16-channel kernel variants. The only variant-specific behavior
// is the MFMA intrinsic, abstracted via the MfmaFn callable.
//
// Template parameters:
//   TC             — TileConstants type
//   cfg            — Config value (kh, kw, direction, epilogue, etc.)
//   MfmaFn         — callable: fp32x4_t mfma(fp16x4_t weight, fp16x4_t input, fp32x4_t acc)
//   BlockCoordsT   — variant-specific BlockCoords type
//   InputLoaderT   — InputLoader type (shared or alias)
//   WeightLoaderT  — WeightLoader type (variant-specific, has read_from_lds)
//   OutputWriterT  — OutputWriter or OutputWriterLds type (shared)
template <typename TC,
          auto cfg,
          typename MfmaFn,
          typename BlockCoordsT,
          typename InputLoaderT,
          typename WeightLoaderT,
          typename OutputWriterT>
__device__ void grouped_conv_compute_loop(const _Float16* __restrict__ in,
                                          const _Float16* __restrict__ wei,
                                          _Float16* __restrict__ out,
                                          int N,
                                          int groups,
                                          int hi,
                                          int wi,
                                          int ho,
                                          int wo,
                                          int py,
                                          int px)
{
    // --- LDS buffers ---
    constexpr bool use_lds_epilogue = (cfg.epilogue == EpilogueType::RegistersToLdsToGlobalMemory);

    __shared__ uint4 input_lds[TC::NUM_INPUT_LDS_BUFFERS * TC::INPUT_LDS_BUFFER_SIZE_PADDED_C8];
    static constexpr int OUTPUT_LDS_SIZE = use_lds_epilogue
                                               ? ck_tile::max(TC::Weight::WEIGHT_LDS_PADDED_UINT4,
                                                              TC::Output::OUTPUT_LDS_BUFFER_SIZE)
                                               : TC::Weight::WEIGHT_LDS_PADDED_UINT4;
    __shared__ uint4 output_lds[OUTPUT_LDS_SIZE];

    // --- Coordinate setup ---
    BlockCoordsT bc(groups);
    if(bc.block_n >= N)
        return;

    InputLoaderT il(bc, input_lds, in, hi, wi, px);
    OutputWriterT ow(bc, output_lds, out, ho, wo);

    // --- Weight loading ---
    fp16x4_t weights_reg[cfg.kh * cfg.kw];
    WeightLoaderT::load_to_lds(bc, output_lds, wei);
    wait_vmcnt<0>();
    __syncthreads();

    WeightLoaderT::read_from_lds(weights_reg, output_lds);
    __syncthreads();

    // --- Prefetch first input row ---
    il.prefetch_tile_to_lds(0);
    wait_vmcnt<0>();
    __syncthreads();

    // --- Circular accumulator buffer ---
    constexpr auto Zero = fp32x4_t{0.f, 0.f, 0.f, 0.f};
    fp32x4_t acc[cfg.kh];
    for(int i = 0; i < cfg.kh; i++)
        acc[i] = Zero;

    int tic = 1;
    int toc = 0;

    MfmaFn mfma_fn{};

    // --- Main loop: iterate over input rows ---
    for(int y_base = 0; y_base + cfg.kh <= hi; y_base += cfg.kh)
    {
        static_for<cfg.kh>(
            [&]<int Y_LOCAL>()
            {
                wait_vmcnt<0>();
                __syncthreads();

                int y = y_base + Y_LOCAL;
                if((y + 1) < hi)
                {
                    il.fetch_tile_to_lds(tic);
                }

                // Accumulate MFMA products over filter width.
                static_for<cfg.kw>(
                    [&]<int S>()
                    {
                        ck_tile::fp16x4_t input_reg;
                        il.read_from_lds(input_reg, S, toc);

                        static_for<cfg.kh>(
                            [&]<int R>()
                            {
                                constexpr int p_idx = (Y_LOCAL - R + cfg.kh) % cfg.kh;
                                if constexpr(cfg.direction == Direction::Dgrad)
                                    acc[p_idx] = mfma_fn(
                                        weights_reg[(cfg.kh - 1 - R) * cfg.kw + (cfg.kw - 1 - S)],
                                        input_reg,
                                        acc[p_idx]);
                                else
                                    acc[p_idx] = mfma_fn(
                                        weights_reg[R * cfg.kw + S],
                                        input_reg,
                                        acc[p_idx]);
                            });
                    });

                tic ^= 1;
                toc ^= 1;

                // Flush completed output row.
                constexpr int P_FLUSH = (Y_LOCAL + 1) % cfg.kh;
                int p_out             = y + py - (cfg.kh - 1);
                if(p_out >= 0 && p_out < ho)
                    ow.flush(acc[P_FLUSH], p_out);
                acc[P_FLUSH] = Zero;
            });
    } // end of the main loop

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

                if((y + 1) < hi)
                {
                    il.fetch_tile_to_lds(tic);
                }

                static_for<cfg.kw>(
                    [&]<int S>()
                    {
                        ck_tile::fp16x4_t input_reg;
                        il.read_from_lds(input_reg, S, toc);

                        static_for<cfg.kh>(
                            [&]<int R>()
                            {
                                constexpr int p_idx = (Y_LOCAL - R + cfg.kh) % cfg.kh;
                                if constexpr(cfg.direction == Direction::Dgrad)
                                    acc[p_idx] = mfma_fn(
                                        weights_reg[(cfg.kh - 1 - R) * cfg.kw + (cfg.kw - 1 - S)],
                                        input_reg,
                                        acc[p_idx]);
                                else
                                    acc[p_idx] = mfma_fn(
                                        weights_reg[R * cfg.kw + S],
                                        input_reg,
                                        acc[p_idx]);
                            });
                    });

                tic ^= 1;
                toc ^= 1;

                constexpr int P_FLUSH = (Y_LOCAL + 1) % cfg.kh;
                int p_out             = y + py - (cfg.kh - 1);
                if(p_out >= 0 && p_out < ho)
                    ow.flush(acc[P_FLUSH], p_out);
                acc[P_FLUSH] = Zero;
            });
    }

    // --- Tail flush: output rows not flushed by the main/remainder loops ---
    for(int p_out = hi - cfg.kh + 1 + py; p_out < ho; p_out++)
    {
        int p_idx = (p_out - py + cfg.kh) % cfg.kh;
        fp32x4_t slot;
        dispatch<cfg.kh>(p_idx,
                        [&]<int P>()
                        {
                            slot   = acc[P];
                            acc[P] = Zero;
                        });
        ow.flush(slot, p_out);
    }
}

} // namespace direct_conv
} // namespace ck_tile
