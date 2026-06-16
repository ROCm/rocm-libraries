// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Non-grouped convolution compute loop v3: Cross-wave LDS reduction.
//
// Splits the C-reduction across waves within the same workgroup. Each wave
// handles a different C-slice (channels_per_group channels), all producing
// partial sums for the same block_k_size K-channels. An LDS-based cross-wave
// reduction combines the partial sums before output.
//
// Supports both mfma_f32_16x16x32 (fp32x4_t accumulators) and
// mfma_f32_32x32x16 (fp32x16_t accumulators). The accumulator type is
// derived from the MfmaFn::acc_type typedef.
//
// Structure per input row:
//   s_waitcnt + __syncthreads
//   prefetch next input row → LDS[tic]
//   for S in 0..kw:
//     each wave reads its own C-section from LDS
//     for R in 0..kh:
//       acc[p_idx] = MFMA(weight[R,S], input, acc[p_idx])
//   swap tic/toc
//   if flush:
//     cross_wave_reduce(acc[slot], reduce_lds, wave_id, num_waves)
//     if wave_id == 0: ow.flush(acc[slot], p_out)
//     acc[slot] = Zero

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/direct_convolution/utils/common.hpp"
#include <hip/hip_runtime.h>

namespace ck_tile::direct_conv::conv_32c_tile::v3 {

// Cross-wave LDS reduction for accumulator values.
//
// Each wave writes its ACC_SIZE fp32 values to a per-wave section of reduce_lds,
// then all waves read and sum across all wave sections. After the reduction,
// only wave 0's values are meaningful for output.
//
// LDS layout: [num_waves][64 threads][ACC_SIZE floats]
// Total: num_waves * 64 * ACC_SIZE * sizeof(float)
template <int NumWaves, typename AccType>
CK_TILE_DEVICE void cross_wave_reduce(AccType& val, float* reduce_lds, int wave_id)
{
    constexpr int ACC_SIZE = sizeof(AccType) / sizeof(float);
    const int lane         = static_cast<int>(threadIdx.x) % 64;

    // Write: each thread writes ACC_SIZE floats to its wave's section.
    const int write_base = wave_id * 64 * ACC_SIZE + lane * ACC_SIZE;
    ck_tile::static_for<0, ACC_SIZE, 1>{}([&](auto i_n) {
        constexpr int I            = i_n.value;
        reduce_lds[write_base + I] = val[I];
    });

    __syncthreads();

    // Read + reduce: sum across all waves.
    AccType sum{};
    ck_tile::static_for<0, NumWaves, 1>{}([&](auto w_n) {
        constexpr int W           = w_n.value;
        constexpr int read_base_w = W * 64 * ACC_SIZE;
        const int read_base       = read_base_w + lane * ACC_SIZE;
        ck_tile::static_for<0, ACC_SIZE, 1>{}([&](auto i_n) {
            constexpr int I = i_n.value;
            sum[I] += reduce_lds[read_base + I];
        });
    });
    val = sum;

    __syncthreads();
}

template <typename TC,
          auto cfg,
          typename MfmaFn,
          typename BlockCoordsT,
          typename InputLoaderT,
          typename WeightLoaderT,
          typename OutputWriterT,
          typename ElementType = _Float16>
CK_TILE_DEVICE void conv_compute_loop_v3(const ElementType* __restrict__ in,
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

    constexpr bool is_dgrad = (cfg.direction == Direction::Dgrad);
    constexpr int NUM_WAVES = cfg.waves_per_wg;

    // --- LDS layout ---
    // Phase 1 (weight prologue): each wave loads its own c_slice into a private
    // LDS region. Layout: [NUM_WAVES][WEIGHT_LDS_SIZE_UINT4].
    // Phase 2 (compute): input double-buffer + reduction buffer (coexist).
    //
    // The reduction buffer is placed after the input double-buffer region.
    // We use a unified LDS allocation sized to the max of (weight_all_waves,
    // input+reduction).
    static constexpr int INPUT_TOTAL = TC::NUM_INPUT_LDS_BUFFERS * TC::INPUT_LDS_BUFFER_SIZE_C8;
    // Reduction buffer: NUM_WAVES * 64 * ACC_FLOATS floats
    // In uint4 units: NUM_WAVES * 64 * ACC_FLOATS * sizeof(float) / sizeof(uint4)
    static constexpr int REDUCE_LDS_UINT4 = NUM_WAVES * 64 * ACC_FLOATS / 4;
    static constexpr int IO_REDUCE_LDS    = INPUT_TOTAL + REDUCE_LDS_UINT4;

    static constexpr int WEIGHT_LDS = TC::WEIGHT_LDS_ALL_WAVES;
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

    // --- Weight prologue: all waves load their own C-slice(s) in parallel ---
    // Each wave loads into its private LDS region and reads back into registers,
    // with no inter-wave serialization. This eliminates the (NUM_WAVES - 1)
    // redundant DRAM reads of the old serial approach.
    //
    // For c_slices_per_wave > 1, the prologue loops N times: each iteration
    // (re)uses the same per-wave LDS region for a different C-chunk and
    // accumulates the chunk into a separate register slot in the WeightLoader.
    // LDS footprint stays at one chunk regardless of N.
    WeightLoaderT wl;
    constexpr int CPG                = cfg.channels_per_group();
    constexpr int WEIGHT_SLICE_UINT4 = TC::WEIGHT_LDS_SIZE_UINT4;
    constexpr int N_CSPW             = cfg.c_slices_per_wave;

    uint4* wave_weight_lds = lds_buf + wave_id * WEIGHT_SLICE_UINT4;

    ck_tile::static_for<0, N_CSPW, 1>{}([&](auto cs_n) {
        constexpr int CS = cs_n.value;
        // Wave w loading chunk CS reads C-section (CS * waves_per_wg + w)
        // from KYXC DRAM. For N=1 this collapses to the original c_slice
        // == wave_id pattern.
        constexpr int wave_stride = cfg.waves_per_wg;
        const int wave_section    = CS * wave_stride + wave_id;

        if constexpr(is_dgrad)
            WeightLoaderT::load_kyxc_to_lds_dgrad_wave(
                wave_weight_lds, wei, wave_section * CPG, weight_block_k, C, K);
        else
            WeightLoaderT::load_kyxc_to_lds_wave(
                wave_weight_lds, wei, weight_block_k, wave_section, C, K);

        ck_tile::s_waitcnt<0>();
        __syncthreads();

        wl.template read_from_lds_chunk<CS>(wave_weight_lds);

        __syncthreads();
    });

    // --- Construct InputLoader and OutputWriter ---
    InputLoaderT il(bc, lds_buf, in, hi, wi, px, y_padding, dilation, dilation, stride, stride);
    uint4* output_staging_lds = lds_buf + INPUT_TOTAL;
    OutputWriterT ow(bc, output_staging_lds, out, ho, wo);

    // --- Prefetch first input chunk of row 0 ---
    il.template prefetch_tile_to_lds<0>(0);

    // --- Circular accumulator buffer ---
    constexpr AccType Zero{};
    AccType acc[cfg.kh];
    ck_tile::static_for<0, cfg.kh, 1>{}([&](auto i_n) { acc[i_n.value] = Zero; });

    int tic = 1;
    int toc = 0;

    MfmaFn mfma_fn{};

    // Helper lambda: LDS reduction + output flush.
    // Both OutputWriterV3 and OutputWriterV3Lds accept wave_id in flush().
    // OutputWriterV3::flush guards on wave_id == 0 internally.
    // OutputWriterV3Lds::flush has all threads participate in barriers.
    auto reduce_and_flush = [&](AccType& slot, int p_out) {
        if constexpr (NUM_WAVES > 1)
        {
            cross_wave_reduce<NUM_WAVES>(slot, reduce_lds, wave_id);
        }
        ow.flush(slot, p_out, wave_id);
        slot = Zero;
    };

    // --- Main loop: process input rows in batches of cfg.kh ---
    //
    // For c_slices_per_wave = N, each input row is processed as N chunks
    // streamed sequentially through the same fixed-size LDS double-buffer.
    // Per (y, CS) iteration: wait for current chunk's prefetch, issue next
    // chunk's prefetch (same row + CS+1, or next row + CS=0), MFMA against
    // chunk CS in toc, swap buffers. Accumulators persist across the CS
    // loop; flush happens once per row.
    for(int y_base = 0; y_base + cfg.kh <= hi; y_base += cfg.kh)
    {
        ck_tile::static_for<0, cfg.kh, 1>{}([&](auto y_local_n) {
            constexpr int Y_LOCAL = y_local_n.value;
            int y                 = y_base + Y_LOCAL;

            ck_tile::static_for<0, N_CSPW, 1>{}([&](auto cs_n) {
                constexpr int CS = cs_n.value;
                ck_tile::s_waitcnt<0>();
                __syncthreads();

                // Prefetch the next chunk into tic: either chunk CS+1
                // of the same row, or chunk 0 of the next row.
                if constexpr(CS + 1 < N_CSPW)
                {
                    il.template prefetch_tile_to_lds<CS + 1>(tic);
                }
                else
                {
                    if((y + 1) < hi)
                        il.template fetch_tile_to_lds<0>(tic);
                }

                // MFMA core: chunk CS lives in lds[toc].
                ck_tile::static_for<0, cfg.kw, 1>{}([&](auto s_n) {
                    constexpr int S = s_n.value;
                    typename InputLoaderT::input_type input_reg;
                    il.read_from_lds(input_reg, S, toc);

                    ck_tile::static_for<0, cfg.kh, 1>{}([&](auto r_n) {
                        constexpr int R     = r_n.value;
                        constexpr int p_idx = (Y_LOCAL - R + cfg.kh) % cfg.kh;
                        if constexpr(is_dgrad)
                            acc[p_idx] = mfma_fn(
                                wl.template get_transposed<R, S, CS>(), input_reg, acc[p_idx]);
                        else
                            acc[p_idx] =
                                mfma_fn(wl.template get<R, S, CS>(), input_reg, acc[p_idx]);
                    });
                });

                tic ^= 1;
                toc ^= 1;
            });

            // Flush completed output row via cross-wave LDS reduction
            // — once per input row, after all N chunks contributed.
            constexpr int P_IDX_FLUSH = (Y_LOCAL + 1) % cfg.kh;
            int p_out                 = y + py - (cfg.kh - 1);
            if(p_out >= 0 && p_out < ho)
                reduce_and_flush(acc[P_IDX_FLUSH], p_out);
            else
                acc[P_IDX_FLUSH] = Zero;
        });
    }

    // --- Remainder loop: hi % kh leftover rows ---
    {
        int y_rem_base = (hi / cfg.kh) * cfg.kh;
        ck_tile::static_for<0, cfg.kh, 1>{}([&](auto y_local_n) {
            constexpr int Y_LOCAL = y_local_n.value;
            if(Y_LOCAL >= hi % cfg.kh)
                return;
            int y = y_rem_base + Y_LOCAL;

            ck_tile::static_for<0, N_CSPW, 1>{}([&](auto cs_n) {
                constexpr int CS = cs_n.value;
                ck_tile::s_waitcnt<0>();
                __syncthreads();

                if constexpr(CS + 1 < N_CSPW)
                {
                    il.template prefetch_tile_to_lds<CS + 1>(tic);
                }
                else
                {
                    if((y + 1) < hi)
                        il.template fetch_tile_to_lds<0>(tic);
                }

                ck_tile::static_for<0, cfg.kw, 1>{}([&](auto s_n) {
                    constexpr int S = s_n.value;
                    typename InputLoaderT::input_type input_reg;
                    il.read_from_lds(input_reg, S, toc);

                    ck_tile::static_for<0, cfg.kh, 1>{}([&](auto r_n) {
                        constexpr int R     = r_n.value;
                        constexpr int p_idx = (Y_LOCAL - R + cfg.kh) % cfg.kh;
                        if constexpr(is_dgrad)
                            acc[p_idx] = mfma_fn(
                                wl.template get_transposed<R, S, CS>(), input_reg, acc[p_idx]);
                        else
                            acc[p_idx] =
                                mfma_fn(wl.template get<R, S, CS>(), input_reg, acc[p_idx]);
                    });
                });

                tic ^= 1;
                toc ^= 1;
            });

            constexpr int P_FLUSH = (Y_LOCAL + 1) % cfg.kh;
            int p_out             = y + py - (cfg.kh - 1);
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
        // Select the accumulator slot matching the runtime p_idx at compile time.
        ck_tile::static_for<0, cfg.kh, 1>{}([&](auto p_n) {
            constexpr int P = p_n.value;
            if(p_idx == P)
            {
                slot   = acc[P];
                acc[P] = Zero;
            }
        });
        reduce_and_flush(slot, p_out);
    }
}

} // namespace ck_tile::direct_conv::conv_32c_tile::v3
