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
// Double-buffering strategies:
//
//   Weight LDS double-buffering (when LDS budget allows):
//     The weight LDS is split into two buffers. While MFMA uses weights
//     from registers (loaded from buf[cur]), the next weight slice loads
//     into buf[1-cur]. Since reads and writes target different LDS
//     regions, the WAR sync between weight_read and weight_load is
//     eliminated. For configs where 2x weight LDS exceeds the budget
//     (8-wave), the single-buffer path with explicit WAR sync is used.
//
//   Input LDS double-buffering (across c_blocks):
//     The input LDS already has two buffers (inherited from the grouped
//     conv design). During the last c_local of c_block N, the next
//     c_block's input is prefetched into the alternate buffer via async
//     buffer_load_dwordx4_lds. When c_block N+1 starts, only the weight
//     load is needed — the input is already in LDS (or nearly so, with
//     wait_vmcnt draining any remaining transfers).
//
// Structure per input row y (double-buffered weight + input):
//   Prefetch input c_block[0] → ibuf[toc]
//   for c_block in 0..num_c_blocks:
//     weight[0] load → wbuf[0]    (co-issued with pending input prefetch)
//     wait + sync                  (input + weight[0] visible)
//     weight[0] read from wbuf[0] → registers
//     for c_local in 0..c_local_count:
//       weight[c_local+1] load → wbuf[1-cur]  (no WAR sync needed)
//       if last c_local: prefetch input c_block[N+1] → ibuf[tic]
//       MFMA with weight registers + input from ibuf[toc] section c_local
//       sync                       (weight write + input reads done)
//       weight[c_local+1] read from wbuf[1-cur] → registers; swap cur
//     swap toc/tic
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
    const int weight_block_k = bc.block_k_start;

    OutputWriterT ow(bc, output_lds, out, ho, wo);

    // --- Circular accumulator buffer ---
    constexpr auto Zero = fp32x4_t{0.f, 0.f, 0.f, 0.f};
    fp32x4_t acc[cfg.kh];
    for(int i = 0; i < cfg.kh; i++)
        acc[i] = Zero;

    MfmaFn mfma_fn{};

    // C-reduction parameters.
    // We compute chunks of 32 channels to match the 32x32 MFMA shape.
    // Depending on the configuration, we may have multiple 32 channel blocks,
    // over which we need to reduce.
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
        // Input LDS double-buffer toggle.
        int toc = 0;  // buffer being read from (tile-on-chip)
        int tic = 1;  // buffer being written to (tile-in-coming)

        for(int c_block = 0; c_block < num_c_blocks; c_block++)
        {
            bc.set_c_block(c_block);

            // Construct InputLoader for this c_block's channel range.
            InputLoaderT il(bc, input_lds, in, hi, wi, px, y_padding,
                            dilation, dilation, stride, stride);

            // Advance to row y
            if(y > 0 && il.load_active)
                il.input_voffset += y * il.row_stride_bytes;

            // First c_block: issue initial async input prefetch.
            // Subsequent c_blocks: input was prefetched during previous
            // c_block's last c_local MFMA.
            if(c_block == 0)
                il.prefetch_tile_to_lds(toc);

            // Co-issue weight load (runs concurrently with pending input prefetch).
            const int first_c_slice = c_block * c_local_count;
            load_weight_slice(first_c_slice, weight_lds_buf);

            wait_vmcnt<0>();
            __syncthreads();

            // Read first weight from LDS → registers.
            WeightLoaderT wl;
            wl.read_from_lds(weight_lds_buf);

            int cur_wbuf = 0;

            // Iterate over C-sections within this c_block.
            for(int c_local = 0; c_local < c_local_count; c_local++)
            {
                const bool has_next_c_local = (c_local + 1 < c_local_count);

                // --- Weight handling ---
                if constexpr(use_weight_double_buf)
                {
                    // Double-buffer: load next weight into alternate buffer.
                    // No WAR sync — reads and writes target different buffers.
                    if(has_next_c_local)
                    {
                        uint4* next_wlds = weight_lds_buf +
                            (1 - cur_wbuf) * WEIGHT_LDS_PER_BUF;
                        load_weight_slice(first_c_slice + c_local + 1, next_wlds);
                    }
                }
                else
                {
                    // Single-buffer: WAR sync before overwriting weight LDS.
                    __syncthreads();
                    if(has_next_c_local)
                        load_weight_slice(first_c_slice + c_local + 1, weight_lds_buf);
                }

                // --- Prefetch next c_block's input on last c_local ---
                // The async prefetch writes to ibuf[tic] while MFMA reads
                // from ibuf[toc] — no conflict (different buffers).
                if(!has_next_c_local && c_block + 1 < num_c_blocks)
                {
                    bc.set_c_block(c_block + 1);
                    InputLoaderT il_next(bc, input_lds, in, hi, wi, px,
                                         y_padding, dilation, dilation,
                                         stride, stride);
                    if(y > 0 && il_next.load_active)
                        il_next.input_voffset += y * il_next.row_stride_bytes;
                    il_next.prefetch_tile_to_lds(tic);
                }

                // --- MFMA: accumulate over filter taps ---
                int c_section_delta = (c_local - wave_group) * 32;

                static_for<cfg.kw>(
                    [&]<int S>()
                    {
                        typename InputLoaderT::input_type input_reg;
                        il.read_from_lds_at_section(input_reg, S, toc,
                                                    c_section_delta);

                        static_for<cfg.kh>(
                            [&]<int R>()
                            {
                                constexpr int Y_LOCAL =
                                    decltype(Y_LOCAL_const)::value;
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

                // Sync: ensures (a) weight LDS write for next iteration
                //       is complete, (b) all input LDS reads are done.
                __syncthreads();

                // Read next weight from LDS → registers.
                if(has_next_c_local)
                {
                    if constexpr(use_weight_double_buf)
                    {
                        cur_wbuf = 1 - cur_wbuf;
                        wl.read_from_lds(weight_lds_buf +
                                         cur_wbuf * WEIGHT_LDS_PER_BUF);
                    }
                    else
                    {
                        wl.read_from_lds(weight_lds_buf);
                    }
                }
            }

            // Toggle input buffers for next c_block.
            toc ^= 1;
            tic ^= 1;
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

// ===================================================================
// conv_compute_loop_v4 — Batched fp32 register accumulation.
//
// Partial products accumulate in fp32 registers across all c_blocks.
// A single DRAM write per output row converts fp32 → fp16/bf16.
//
// Benefits over v3:
//   1. Perfect precision: no fp16 round-trip between c_blocks.
//   2. Single DRAM write per output row (no read-modify-write).
//   3. Weight data comes from L2 cache for subsequent batches.
//
// Structure:
//   for batch (BATCH_SIZE rows):
//     fp32x4_t output_accum[BATCH_SIZE] = {0}
//     for c_block in 0..num_c_blocks:
//       Weight prologue: load all c_local weight sets → registers
//       Input double-buffered row loop (same as v3)
//       At flush points: output_accum[row] += acc (register accum)
//     Final store: output_accum → fp16 → DRAM (single write)
// ===================================================================
template <typename TC,
          auto cfg,
          typename MfmaFn,
          typename BlockCoordsT,
          typename InputLoaderT,
          typename WeightLoaderT,
          typename OutputWriterT,
          typename ElementType = _Float16>
__device__ void conv_compute_loop_v4(const ElementType* __restrict__ in,
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

    // --- Unified LDS: weight and I/O share the same buffer ---
    static constexpr int INPUT_TOTAL = TC::NUM_INPUT_LDS_BUFFERS * TC::INPUT_LDS_BUFFER_SIZE_C8;
    static constexpr int IO_LDS = use_lds_epilogue
                                       ? INPUT_TOTAL + TC::Output::OUTPUT_LDS_BUFFER_SIZE
                                       : INPUT_TOTAL;
    static constexpr int WEIGHT_LDS_PER_BUF = TC::WEIGHT_LDS_SIZE_UINT4;
    static constexpr int UNIFIED_LDS_SIZE = (WEIGHT_LDS_PER_BUF > IO_LDS)
                                                ? WEIGHT_LDS_PER_BUF : IO_LDS;

    __shared__ uint4 lds_buf[UNIFIED_LDS_SIZE];
    uint4* input_lds  = lds_buf;
    uint4* output_lds = lds_buf + INPUT_TOTAL;

    // --- Coordinate setup ---
    const int C_in  = is_dgrad ? K : C;
    const int C_out = is_dgrad ? C : K;

    BlockCoordsT bc(C_in, C_out);
    if(bc.block_n >= N)
        return;

    const int weight_block_k = bc.block_k_start;

    OutputWriterT ow(bc, output_lds, out, ho, wo);

    // --- Circular accumulator buffer ---
    constexpr auto Zero = fp32x4_t{0.f, 0.f, 0.f, 0.f};
    fp32x4_t acc[cfg.kh];

    MfmaFn mfma_fn{};

    // C-reduction parameters.
    const int block_c_size    = cfg.block_groups() * 32;
    const int num_c_blocks    = C_in / block_c_size;
    const int wave_group      = static_cast<int>(threadIdx.x / 64) / 2;

    constexpr int C_LOCAL_COUNT = cfg.c_local_count();
    constexpr int y_padding = 0;
    constexpr int stride = 1;
    constexpr int dilation = 1;

    // Helper: load weight for a given c_slice into the unified LDS buffer.
    auto load_weight_slice = [&](int c_slice)
    {
        if constexpr(is_dgrad)
            WeightLoaderT::load_kyxc_to_lds_dgrad(lds_buf, wei,
                                                   c_slice * 32, weight_block_k, C);
        else
            WeightLoaderT::load_kyxc_to_lds(lds_buf, wei,
                                             weight_block_k, c_slice, C);
    };

    // --- Batch size for fp32 register accumulation ---
    // Conservative start: 16 rows per batch fits all configs in ~256 VGPRs
    // (16 * 4 = 64 output accum VGPRs).
    static constexpr int BATCH_SIZE = 16;

    // Output accumulator array — fp32 registers, one fp32x4_t per output row in batch.
    fp32x4_t output_accum[BATCH_SIZE];

    // === OUTERMOST LOOP: batch over output rows ===
    for(int batch_start = 0; batch_start < ho; batch_start += BATCH_SIZE)
    {
        int batch_end = (batch_start + BATCH_SIZE < ho) ? batch_start + BATCH_SIZE : ho;

        // Zero output accumulators for this batch.
        for(int i = 0; i < BATCH_SIZE; i++)
            output_accum[i] = Zero;

        // === c_block loop (flush points accumulate to registers) ===
        for(int c_block = 0; c_block < num_c_blocks; c_block++)
        {
            // Zero circular accumulators at start of each c_block.
            for(int i = 0; i < cfg.kh; i++)
                acc[i] = Zero;

            // --- Weight prologue: load all c_local weight sets into registers ---
            WeightLoaderT wl_arr[C_LOCAL_COUNT];

            for(int c_local = 0; c_local < C_LOCAL_COUNT; c_local++)
            {
                int c_slice = c_block * C_LOCAL_COUNT + c_local;
                __syncthreads();
                load_weight_slice(c_slice);
                wait_vmcnt<0>();
                __syncthreads();
                wl_arr[c_local].read_from_lds(lds_buf);
            }

            __syncthreads();

            // --- Compute input row range for this batch ---
            // For output row p_out, the input rows contributing are:
            //   y = p_out - py .. p_out - py + kh - 1
            // For the batch [batch_start, batch_end), we need input rows:
            //   y_start = max(0, batch_start - py)
            //   y_end   = min(hi, batch_end - 1 + (kh - 1) - py + 1)
            //           = min(hi, batch_end + kh - 2 - py + 1)
            int y_start = batch_start - py;
            if(y_start < 0) y_start = 0;
            int y_end = batch_end + cfg.kh - 1 - py;
            if(y_end > hi) y_end = hi;

            int num_rows = y_end - y_start;
            if(num_rows <= 0)
                continue;

            // --- Input double-buffered row loop ---
            bc.set_c_block(c_block);
            InputLoaderT il(bc, input_lds, in, hi, wi, px, y_padding,
                            dilation, dilation, stride, stride);

            // Advance to y_start.
            if(y_start > 0 && il.load_active)
                il.input_voffset += y_start * il.row_stride_bytes;
            il.prefetch_tile_to_lds(0);

            int tic = 1;
            int toc = 0;

            // --- Main loop: process input rows in batches of cfg.kh ---
            for(int y_base = y_start; y_base + cfg.kh <= y_end; y_base += cfg.kh)
            {
                static_for<cfg.kh>(
                    [&]<int Y_LOCAL>()
                    {
                        int y = y_base + Y_LOCAL;

                        wait_vmcnt<0>();
                        __syncthreads();

                        if((y + 1) < y_end)
                            il.fetch_tile_to_lds(tic);

                        // MFMA C-reduction: iterate over filter width and weight slices.
                        static_for<cfg.kw>(
                            [&]<int S>()
                            {
                                static_for<C_LOCAL_COUNT>([&]<int c_local>()
                                {
                                    int c_section_delta = (c_local - wave_group) * 32;
                                    typename InputLoaderT::input_type input_reg;
                                    il.read_from_lds_at_section(input_reg, S, toc,
                                                                c_section_delta);

                                    static_for<cfg.kh>(
                                        [&]<int R>()
                                        {
                                            constexpr int p_idx =
                                                (Y_LOCAL - R + cfg.kh) % cfg.kh;
                                            if constexpr(is_dgrad)
                                                acc[p_idx] = mfma_fn(
                                                    wl_arr[c_local].template get_transposed<R, S>(),
                                                    input_reg,
                                                    acc[p_idx]);
                                            else
                                                acc[p_idx] = mfma_fn(
                                                    wl_arr[c_local].template get<R, S>(),
                                                    input_reg,
                                                    acc[p_idx]);
                                        });
                                });
                            });

                        tic ^= 1;
                        toc ^= 1;

                        // Accumulate completed output row into fp32 registers.
                        constexpr int P_IDX_FLUSH = (Y_LOCAL + 1) % cfg.kh;
                        int p_out = y + py - (cfg.kh - 1);
                        if(p_out >= batch_start && p_out < batch_end)
                        {
                            int row_in_batch = p_out - batch_start;
                            output_accum[row_in_batch][0] += acc[P_IDX_FLUSH][0];
                            output_accum[row_in_batch][1] += acc[P_IDX_FLUSH][1];
                            output_accum[row_in_batch][2] += acc[P_IDX_FLUSH][2];
                            output_accum[row_in_batch][3] += acc[P_IDX_FLUSH][3];
                        }
                        acc[P_IDX_FLUSH] = Zero;
                    });
            }

            // --- Remainder loop: leftover rows after main loop ---
            {
                int y_rem_base = y_start + (num_rows / cfg.kh) * cfg.kh;
                static_for<cfg.kh>(
                    [&]<int Y_LOCAL>()
                    {
                        if(Y_LOCAL >= num_rows % cfg.kh)
                            return;
                        int y = y_rem_base + Y_LOCAL;

                        wait_vmcnt<0>();
                        __syncthreads();

                        if((y + 1) < y_end)
                            il.fetch_tile_to_lds(tic);

                        static_for<cfg.kw>(
                            [&]<int S>()
                            {
                                static_for<C_LOCAL_COUNT>([&]<int c_local>()
                                {
                                    int c_section_delta = (c_local - wave_group) * 32;
                                    typename InputLoaderT::input_type input_reg;
                                    il.read_from_lds_at_section(input_reg, S, toc,
                                                                c_section_delta);

                                    static_for<cfg.kh>(
                                        [&]<int R>()
                                        {
                                            constexpr int p_idx =
                                                (Y_LOCAL - R + cfg.kh) % cfg.kh;
                                            if constexpr(is_dgrad)
                                                acc[p_idx] = mfma_fn(
                                                    wl_arr[c_local].template get_transposed<R, S>(),
                                                    input_reg,
                                                    acc[p_idx]);
                                            else
                                                acc[p_idx] = mfma_fn(
                                                    wl_arr[c_local].template get<R, S>(),
                                                    input_reg,
                                                    acc[p_idx]);
                                        });
                                });
                            });

                        tic ^= 1;
                        toc ^= 1;

                        constexpr int P_FLUSH = (Y_LOCAL + 1) % cfg.kh;
                        int p_out = y + py - (cfg.kh - 1);
                        if(p_out >= batch_start && p_out < batch_end)
                        {
                            int row_in_batch = p_out - batch_start;
                            output_accum[row_in_batch][0] += acc[P_FLUSH][0];
                            output_accum[row_in_batch][1] += acc[P_FLUSH][1];
                            output_accum[row_in_batch][2] += acc[P_FLUSH][2];
                            output_accum[row_in_batch][3] += acc[P_FLUSH][3];
                        }
                        acc[P_FLUSH] = Zero;
                    });
            }

            // --- Tail flush: output rows not flushed by the main/remainder loops ---
            // Note: p_idx must account for y_start (unlike v3 which always starts at y=0).
            // Y_LOCAL = (y - y_start) % kh, so the flush slot offset shifts by y_start.
            for(int p_out = y_end - cfg.kh + 1 + py; p_out < ho; p_out++)
            {
                if(p_out < batch_start || p_out >= batch_end)
                    continue;
                __syncthreads();
                int p_idx = (p_out - py + cfg.kh - y_start) % cfg.kh;
                fp32x4_t slot;
                dispatch<cfg.kh>(p_idx,
                                 [&]<int P>()
                                 {
                                     slot   = acc[P];
                                     acc[P] = Zero;
                                 });
                int row_in_batch = p_out - batch_start;
                output_accum[row_in_batch][0] += slot[0];
                output_accum[row_in_batch][1] += slot[1];
                output_accum[row_in_batch][2] += slot[2];
                output_accum[row_in_batch][3] += slot[3];
            }
        } // end of c_block loop

        // === Final store: fp32 registers → fp16/bf16 → DRAM ===
        for(int p_out = batch_start; p_out < batch_end; p_out++)
        {
            int row_in_batch = p_out - batch_start;
            ow.flush(output_accum[row_in_batch], p_out);
        }
    } // end of batch loop
}

} // namespace ck_tile::direct_conv::conv_32c_tile::v1
