// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Non-grouped convolution compute loop with C-reduction.
//
// Uses the same circular accumulator buffer and batched row processing
// as grouped_conv_compute_loop, extended with an inner C-reduction loop
// per input row. Input rows are processed in batches of cfg.kh using
// static_for with compile-time modular accumulator indexing:
//   p_idx = (Y_LOCAL - R + kh) % kh
//
// Weight LDS double-buffering:
//   When LDS budget allows (2-wave and 4-wave configs), the weight LDS
//   is split into two buffers. This eliminates the WAR (Write-After-Read)
//   hazard sync: while MFMA uses weights from registers (loaded from
//   buf[cur]), the next weight slice loads into buf[1-cur]. Since reads
//   and writes target different LDS regions, no sync is needed between
//   the weight read and the next weight load.
//
//   For configs where 2× weight LDS exceeds the LDS budget (8-wave),
//   the single-buffer path is used with an explicit WAR sync.
//
// Structure per input row y (double-buffered):
//   for c_block in 0..num_c_blocks:
//     Co-issue: input load + weight[0] load → buf[0]
//     sync                                   (input + weight[0] visible)
//     weight[0] read from buf[0] → registers
//     for c_local in 0..c_local_count:
//       weight[c_local+1] load → buf[1-cur]  (no WAR sync needed!)
//       MFMA with weight registers + input from section c_local
//       sync                                 (weight LDS write + input reads done)
//       weight[c_local+1] read from buf[1-cur] → registers
//       swap cur
//   Flush completed output row (after full C-reduction)
//
// Structure per input row y (single-buffered fallback):
//   for c_block in 0..num_c_blocks:
//     Co-issue: input load + weight[0] load
//     sync                                   (input + weight[0] visible)
//     weight[0] read → registers
//     for c_local in 0..c_local_count:
//       sync                                 (WAR: weight read complete)
//       Prefetch weight[c_local+1] → LDS     (overlaps with MFMA below)
//       MFMA with weight registers + input from section c_local
//       sync                                 (weight LDS write + input reads done)
//       weight[c_local+1] read → registers
//   Flush completed output row (after full C-reduction)

#pragma once

#include "ck_tile/ops/direct_convolution/utils/common.hpp"
#include "ck_tile/ops/direct_convolution/utils/detail.hpp"
#include "ck_tile/ops/direct_convolution/utils/memory.hpp"
#include <hip/hip_runtime.h>

namespace ck_tile::direct_conv::conv_32c_tile::v1
{

template <typename TC,
          auto cfg,
          typename MfmaFn,
          typename BlockCoordsT,
          typename InputLoaderT,
          typename WeightLoaderT,
          typename OutputWriterT,
          typename ElementType = _Float16>
__device__ void conv_compute_loop(const ElementType* __restrict__ in,
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
    constexpr bool use_lds_epilogue = (cfg.epilogue == EpilogueType::RegistersToLdsToGlobalMemory);
    constexpr bool is_dgrad = (cfg.direction == Direction::Dgrad);

    // --- LDS layout: weight region + input/output region ---
    static constexpr int INPUT_TOTAL = TC::NUM_INPUT_LDS_BUFFERS * TC::INPUT_LDS_BUFFER_SIZE_C8;
    static constexpr int IO_LDS = use_lds_epilogue
                                       ? INPUT_TOTAL + TC::Output::OUTPUT_LDS_BUFFER_SIZE
                                       : INPUT_TOTAL;

    // Weight LDS double-buffering: allocate 2 buffers when the total
    // LDS fits within the gfx950 budget (128 KB = 8192 uint4).
    static constexpr int WEIGHT_LDS_PER_BUF = TC::WEIGHT_LDS_SIZE_UINT4;
    static constexpr int LDS_BUDGET_UINT4 = 131072 / 16;
    static constexpr bool use_weight_double_buf =
        (2 * WEIGHT_LDS_PER_BUF + IO_LDS <= LDS_BUDGET_UINT4);
    static constexpr int NUM_WEIGHT_BUFS = use_weight_double_buf ? 2 : 1;

    __shared__ uint4 weight_lds_buf[NUM_WEIGHT_BUFS * WEIGHT_LDS_PER_BUF];
    __shared__ uint4 io_lds_buf[IO_LDS];

    uint4* input_lds  = io_lds_buf;
    uint4* output_lds = io_lds_buf + INPUT_TOTAL;

    // --- Coordinate setup ---
    // For Dgrad: in = output gradient (K channels), out = input gradient (C channels).
    // The weight tensor is always KYXC.
    const int C_in  = is_dgrad ? K : C;
    const int C_out = is_dgrad ? C : K;

    BlockCoordsT bc(C_in, C_out);
    if(bc.block_n >= N)
        return;

    // The weight block_k_start indexes into the K dimension of the weight tensor.
    // For Fprop: this is the output K dimension.
    // For Dgrad: the weight K dimension still corresponds to output channels.
    // Since for Dgrad in/out are swapped, the weight K maps to C_in = K.
    const int weight_block_k = bc.block_k_start;

    OutputWriterT ow(bc, output_lds, out, ho, wo);

    // --- Circular accumulator buffer ---
    constexpr auto Zero = fp32x4_t{0.f, 0.f, 0.f, 0.f};
    fp32x4_t acc[cfg.kh];
    for(int i = 0; i < cfg.kh; i++)
        acc[i] = Zero;

    MfmaFn mfma_fn{};

    // C-reduction parameters.
    const int block_c_size    = cfg.block_groups() * 32;
    const int num_c_blocks    = C_in / block_c_size;
    const int c_local_count   = cfg.c_local_count();
    const int wave_group      = static_cast<int>(threadIdx.x / 64) / 2;

    constexpr int y_padding = 0;
    constexpr int stride = 1;
    constexpr int dilation = 1;

    // Helper: load weight for a given c_slice into the specified LDS buffer.
    auto load_weight_slice = [&](int c_slice, uint4* target_lds)
    {
        if constexpr(is_dgrad)
            WeightLoaderT::load_kyxc_to_lds_dgrad(target_lds, wei,
                                                   c_slice * 32, weight_block_k, C);
        else
            WeightLoaderT::load_kyxc_to_lds(target_lds, wei,
                                             weight_block_k, c_slice, C);
    };

    // Helper lambda: process all c_blocks for input row y, accumulating
    // MFMA products into acc[] with compile-time p_idx.
    auto process_row = [&](int y, auto Y_LOCAL_const)
    {
        for(int c_block = 0; c_block < num_c_blocks; c_block++)
        {
            bc.set_c_block(c_block);

            // Construct InputLoader for this c_block's channel range.
            InputLoaderT il(bc, input_lds, in, hi, wi, px, y_padding,
                            dilation, dilation, stride, stride);

            // Advance to row y in O(1).
            if(y > 0 && il.load_active)
                il.input_voffset += y * il.row_stride_bytes;

            // Co-issue input load and first weight load (separate LDS regions).
            const int first_c_slice = c_block * c_local_count;
            il.prefetch_tile_to_lds(0);
            load_weight_slice(first_c_slice, weight_lds_buf);

            wait_vmcnt<0>();
            __syncthreads();

            // Read first weight from LDS → registers.
            WeightLoaderT wl;
            wl.read_from_lds(weight_lds_buf);

            int cur_buf = 0;

            // Iterate over C-sections within this c_block.
            for(int c_local = 0; c_local < c_local_count; c_local++)
            {
                const bool has_next = (c_local + 1 < c_local_count);

                if constexpr(use_weight_double_buf)
                {
                    // Double-buffer: load next weight into alternate buffer.
                    // No WAR sync — reads and writes target different buffers.
                    if(has_next)
                    {
                        uint4* next_lds = weight_lds_buf + (1 - cur_buf) * WEIGHT_LDS_PER_BUF;
                        load_weight_slice(first_c_slice + c_local + 1, next_lds);
                    }
                }
                else
                {
                    // Single-buffer: WAR sync before overwriting weight LDS.
                    __syncthreads();
                    if(has_next)
                        load_weight_slice(first_c_slice + c_local + 1, weight_lds_buf);
                }

                // MFMA: accumulate over filter taps.
                int c_section_delta = (c_local - wave_group) * 32;

                static_for<cfg.kw>(
                    [&]<int S>()
                    {
                        typename InputLoaderT::input_type input_reg;
                        il.read_from_lds_at_section(input_reg, S, 0, c_section_delta);

                        static_for<cfg.kh>(
                            [&]<int R>()
                            {
                                // Compile-time circular buffer index:
                                // p_idx = (Y_LOCAL - R + kh) % kh
                                constexpr int Y_LOCAL = decltype(Y_LOCAL_const)::value;
                                constexpr int p_idx = (Y_LOCAL - R + cfg.kh) % cfg.kh;
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

                // Sync: ensures (a) weight LDS write for next iteration is complete,
                //                (b) all input LDS reads for this iteration are done.
                __syncthreads();

                // Read next weight from LDS → registers.
                if(has_next)
                {
                    if constexpr(use_weight_double_buf)
                    {
                        cur_buf = 1 - cur_buf;
                        wl.read_from_lds(weight_lds_buf + cur_buf * WEIGHT_LDS_PER_BUF);
                    }
                    else
                    {
                        wl.read_from_lds(weight_lds_buf);
                    }
                }
            }
        }
    };

    // --- Main loop: process input rows in batches of cfg.kh ---
    for(int y_base = 0; y_base + cfg.kh <= hi; y_base += cfg.kh)
    {
        static_for<cfg.kh>(
            [&]<int Y_LOCAL>()
            {
                int y = y_base + Y_LOCAL;

                process_row(y, std::integral_constant<int, Y_LOCAL>{});

                // Flush completed output row.
                constexpr int P_IDX_FLUSH = (Y_LOCAL + 1) % cfg.kh;
                int p_out = y + py - (cfg.kh - 1);
                if(p_out >= 0 && p_out < ho)
                    ow.flush(acc[P_IDX_FLUSH], p_out);
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

                process_row(y, std::integral_constant<int, Y_LOCAL>{});

                constexpr int P_FLUSH = (Y_LOCAL + 1) % cfg.kh;
                int p_out = y + py - (cfg.kh - 1);
                if(p_out >= 0 && p_out < ho)
                    ow.flush(acc[P_FLUSH], p_out);
                acc[P_FLUSH] = Zero;
            });
    }

    // --- Tail flush: output rows not flushed by the main/remainder loops ---
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

} // namespace ck_tile::direct_conv::conv_32c_tile::v1
