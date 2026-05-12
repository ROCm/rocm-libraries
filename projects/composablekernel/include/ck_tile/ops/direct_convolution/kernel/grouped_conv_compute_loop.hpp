// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/direct_convolution/utils/common.hpp"
#include "ck_tile/ops/direct_convolution/utils/conv_params.hpp"
#include "ck_tile/ops/direct_convolution/utils/detail.hpp"
#include "ck_tile/ops/direct_convolution/utils/memory.hpp"

namespace ck_tile {
namespace direct_conv {

// Shared compute loop for grouped convolution kernels (Fprop and Dgrad).
//
// This function implements the main device-side compute loop shared by all
// kernel variants (4c, 8c, 16c, and 32c). The only variant-specific behavior is
// the MFMA intrinsic (abstracted via MfmaFn) and the inner loop width
// (INNER_KW: cfg.kw for standard kernels, 1 for Toeplitz 8c where S is
// embedded in the MFMA K dimension).
//
// Template parameters:
//   TC             — TileConstants type
//   cfg            — Config value (kh, kw, direction, epilogue, etc.)
//   MfmaFn         — callable: mfma(weight, input, acc) -> fp32x4_t
//   BlockCoordsT   — variant-specific BlockCoords type
//   InputLoaderT   — InputLoader type (shared or alias)
//   WeightLoaderT  — WeightLoader type (variant-specific, has read_from_lds)
//   OutputWriterT  — OutputWriter or OutputWriterLds type (shared)
//   INNER_KW       — inner loop width (defaults to cfg.kw; set to 1 for Toeplitz)
template <typename TC,
          auto cfg,
          bool Padded,
          typename MfmaFn,
          typename BlockCoordsT,
          typename InputLoaderT,
          typename WeightLoaderT,
          typename OutputWriterT,
          int INNER_KW = cfg.kw>
__device__ void grouped_conv_compute_loop(const _Float16* __restrict__ in,
                                          const _Float16* __restrict__ wei,
                                          _Float16* __restrict__ out,
                                          int N,
                                          int groups,
                                          int c_per_group,
                                          int k_per_group,
                                          int hi,
                                          int wi,
                                          int ho,
                                          int wo,
                                          int py,
                                          int px)
{
    // --- Unified LDS buffer ---
    // Weights are loaded first into LDS, consumed, then never accessed again.
    // After that, the same LDS memory is reused for input double-buffering
    // (and output staging for the LDS epilogue path). 
    constexpr bool use_lds_epilogue = (cfg.epilogue == EpilogueType::RegistersToLdsToGlobalMemory);

    static constexpr int INPUT_TOTAL = TC::NUM_INPUT_LDS_BUFFERS * TC::INPUT_LDS_BUFFER_SIZE_C8;
    // The weight loader's LDS write descriptor uses WEIGHT_LDS_PADDED_UINT4
    // (rounded up to block_size for multi-pass tile distribution loading).
    // But the __shared__ allocation only needs WEIGHT_LDS_SIZE_UINT4 because
    // the DRAM pad transform marks padding rows as OOB, and the hardware
    // suppresses LDS writes for OOB buffer_load_lds lanes.
    static constexpr int WEIGHT_LDS  = TC::Weight::WEIGHT_LDS_SIZE_UINT4;
    static constexpr int IO_LDS      = use_lds_epilogue
                                            ? INPUT_TOTAL + TC::Output::OUTPUT_LDS_BUFFER_SIZE
                                            : INPUT_TOTAL;
    static constexpr int UNIFIED_LDS_SIZE = (WEIGHT_LDS > IO_LDS) ? WEIGHT_LDS : IO_LDS;
    __shared__ uint4 lds_buf[UNIFIED_LDS_SIZE];
    // Weight phase:  weight_lds = lds_buf (weights loaded at start, consumed before input)
    // IO phase:      input_lds  = lds_buf (double-buffered input overwrites weight region)
    //                output_lds = lds_buf + INPUT_TOTAL (output staging after both input buffers)
    uint4* input_lds  = lds_buf;
    uint4* output_lds = lds_buf + INPUT_TOTAL;

    // --- Coordinate setup ---
    //
    // For Dgrad the pointers are already swapped by the kernel wrapper:
    //   in  = output gradient (K channels per group)
    //   out = input gradient  (C channels per group)
    // So the "input" channel count is k_per_group and the "output" channel
    // count is c_per_group — the reverse of Fprop.  The weight tensor is
    // not swapped (always GKYXC), so the weight loader keeps the original
    // c_per_group / k_per_group.
    constexpr bool is_dgrad = (cfg.direction == Direction::Dgrad);
    const int in_cpg  = is_dgrad ? k_per_group : c_per_group;
    const int out_kpg = is_dgrad ? c_per_group : k_per_group;

    BlockCoordsT bc(groups, in_cpg, out_kpg);
    if(bc.block_n >= N)
        return;

    // --- Weight loading (uses start of buffer, before input phase) ---
    // Weight tensor layout is always GKYXC with the original c/k_per_group.
    WeightLoaderT wl;
    WeightLoaderT::template load_to_lds<Padded>(bc, lds_buf, wei, c_per_group, k_per_group);
    wait_vmcnt<0>();
    __syncthreads();

    wl.read_from_lds(lds_buf);

    // Note on padding: 
    // py is not passed to InputLoader's DRAM descriptor because the
    // kernel iterates over physical input rows (0..hi-1), not padded rows.
    // The py offset is handled by the compute loop's output flush logic
    // (p_out = y + py - (kh-1)).  
    //
    // px is needed because the tile window
    // spans block_q..block_q+BLOCK_W-1 horizontally and needs OOB checking
    // for positions in the padded border.
    //
    // We must have unit stride and dilation for now.
    constexpr int y_padding = 0;
    constexpr int stride = 1;
    constexpr int dilation = 1;
    InputLoaderT il(bc, input_lds, in, hi, wi, px, y_padding, dilation, dilation, stride, stride, in_cpg);
    OutputWriterT ow(bc, output_lds, out, ho, wo, out_kpg);

    __syncthreads();

    // --- Prefetch first input row ---
    il.prefetch_tile_to_lds(0);

    // --- Circular accumulator buffer ---
    constexpr auto Zero = fp32x4_t{0.f, 0.f, 0.f, 0.f};
    fp32x4_t acc[cfg.kh];
    for(int i = 0; i < cfg.kh; i++)
        acc[i] = Zero;

    int tic = 1;
    int toc = 0;

    MfmaFn mfma_fn{};

    // --- Main loop: iterate over input rows ---
    // Each input row constributes up to cfg.kh output rows.
    // The acc[kh] array has one slot per output row (each slot holds the partial sum over input channels for that output row)
    for(int y_base = 0; y_base + cfg.kh <= hi; y_base += cfg.kh)
    {
        // The main loop iterates in steps over cfg.kh input rows.
        // This static loop unrolls the cfg.kh loops over the input rows.
        static_for<cfg.kh>(
            [&]<int Y_LOCAL>()
            {
                wait_vmcnt<0>();
                __syncthreads();

                // Input row index that maps to cfg.kh output rows, for this iteration of the main loop.
                // acc[p_idx] holds the partials sums for the output rows.
                int y = y_base + Y_LOCAL;

                // Fetch the next input slice into LDS while computing on the current slice.
                // The first slice has been pre-fetch already.
                if((y + 1) < hi)
                {
                    il.fetch_tile_to_lds(tic);
                }

                // Loop over the filter width dimension (S) and
                // accumulate MFMA products for this input row with the corresponding filter weights.
                // INNER_KW defaults to cfg.kw; for the 8c Toeplitz kernel it is 1
                // (S is embedded in the MFMA K=32 dimension).
                static_for<INNER_KW>(
                    [&]<int S>()
                    {
                        // Read one input column strip from LDS into registers.
                        // Each thread reads input[n, y, q+S, :] into input_reg, where q is the horizontal offset of the tile.
                        // All input channes are read and distributed accross the threads in the block.
                        // The register type is InputLoaderT::input_type (fp16x4_t for 4c/16c, fp16x8_t for 8c Toeplitz and 32c).
                        typename InputLoaderT::input_type input_reg;
                        il.read_from_lds(input_reg, S, toc);

                        // Accumulate the MFMA products for this input column strip
                        // with the corresponding filter weights.
                        static_for<cfg.kh>(
                            [&]<int R>()
                            {
                                // Which output row (p) this (y,R) pair contributes to.
                                // Recall that this input row contributes to cfg.kh output rows, depending on the filter row (R) being applied.
                                // p + R = y --> p = y - R = (y_local + y_base) - R
                                // Since the acc array is circular, we wrap p_idx around cfg.kh
                                // Because y_base = iy * cfg.kh, the y_local is the only thing that affects the p_idx, 
                                // so we can express it purely in terms of y_local and R.
                                // Normalizing the index on interval 0,...,cfg.kh-1 range:, we get:
                                constexpr int p_idx = (Y_LOCAL - R + cfg.kh) % cfg.kh;

                                // The final sum over the input channels for this (y,R,S) position is computed
                                // by MFMAing the input_reg with the corresponding filter weights, and accumulating into acc[p_idx].
                                if constexpr(cfg.direction == Direction::Dgrad)
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

                tic ^= 1;
                toc ^= 1;

                // Flush completed output row to global memory.
                // After processing input row y = y_base + Y_LOCAL,
                // the output row p_out = y - (kh-1) + py  (py is padding in y-direction) has received contributions from all R values
                // (since the earliest contributor to p_out was input row p_out - py, processed kh-1 steps ago).
                // That slot is done and can be flushed to global memory.
                // The slot index is a compile-time const that is computed from the accumulator index definition
                // p_idx = (y_local - R + kh) % kh, with R = kh-1 (the last contributor to p_out).
                constexpr int P_IDX_FLUSH = (Y_LOCAL + 1) % cfg.kh;
                int p_out             = y + py - (cfg.kh - 1);
                if(p_out >= 0 && p_out < ho)
                    ow.flush(acc[P_IDX_FLUSH], p_out);
                acc[P_IDX_FLUSH] = Zero;

            }); // end of loop over Y_LOCAL (contributions from one input row to multiple output rows)
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

                static_for<INNER_KW>(
                    [&]<int S>()
                    {
                        typename InputLoaderT::input_type input_reg;
                        il.read_from_lds(input_reg, S, toc);

                        static_for<cfg.kh>(
                            [&]<int R>()
                            {
                                constexpr int p_idx = (Y_LOCAL - R + cfg.kh) % cfg.kh;
                                if constexpr(cfg.direction == Direction::Dgrad)
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
    // The __syncthreads() before each flush separates the previous iteration's
    // LDS reads from this iteration's LDS writes. 
    for(int p_out = hi - cfg.kh + 1 + py; p_out < ho; p_out++)
    {
        __syncthreads();
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
