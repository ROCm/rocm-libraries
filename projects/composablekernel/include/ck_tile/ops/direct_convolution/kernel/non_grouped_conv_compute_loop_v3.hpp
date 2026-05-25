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
//   wait_vmcnt + __syncthreads
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

#include "ck_tile/ops/direct_convolution/utils/common.hpp"
#include "ck_tile/ops/direct_convolution/utils/detail.hpp"
#include "ck_tile/ops/direct_convolution/utils/memory.hpp"
#include <hip/hip_runtime.h>

namespace ck_tile::direct_conv::conv_32c_tile::v3
{

// Cross-wave LDS reduction for accumulator values.
//
// Each wave writes its ACC_SIZE fp32 values to a per-wave section of reduce_lds,
// then all waves read and sum across all wave sections. After the reduction,
// only wave 0's values are meaningful for output.
//
// LDS layout: [num_waves][64 threads][ACC_SIZE floats]
// Total: num_waves * 64 * ACC_SIZE * sizeof(float)
template <int NumWaves, typename AccType>
__device__ __forceinline__ void cross_wave_reduce(
    AccType& val, float* reduce_lds, int wave_id)
{
    constexpr int ACC_SIZE = sizeof(AccType) / sizeof(float);
    const int lane = static_cast<int>(threadIdx.x) % 64;

    // Write: each thread writes ACC_SIZE floats to its wave's section.
    const int write_base = wave_id * 64 * ACC_SIZE + lane * ACC_SIZE;
    static_for<ACC_SIZE>(
        [&]<int I>()
        { reduce_lds[write_base + I] = val[I]; });

    __syncthreads();

    // Read + reduce: sum across all waves.
    AccType sum{};
    static_for<NumWaves>(
        [&]<int W>()
        {
            constexpr int read_base_w = W * 64 * ACC_SIZE;
            const int read_base = read_base_w + lane * ACC_SIZE;
            static_for<ACC_SIZE>(
                [&]<int I>()
                { sum[I] += reduce_lds[read_base + I]; });
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
__device__ void conv_compute_loop_v3(const ElementType* __restrict__ in,
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
    using AccType = typename MfmaFn::acc_type;
    constexpr int ACC_FLOATS = sizeof(AccType) / sizeof(float);

    constexpr bool is_dgrad = (cfg.direction == Direction::Dgrad);
    constexpr int NUM_WAVES = cfg.waves_per_wg;

    // --- LDS layout ---
    // Phase 1 (weight prologue): weight LDS sized for one c_slice.
    // Phase 2 (compute): input double-buffer + reduction buffer (coexist).
    //
    // The reduction buffer is placed after the input double-buffer region.
    // We use a unified LDS allocation sized to the max of (weight, input+reduction).
    static constexpr int INPUT_TOTAL = TC::NUM_INPUT_LDS_BUFFERS * TC::INPUT_LDS_BUFFER_SIZE_C8;
    // Reduction buffer: NUM_WAVES * 64 * ACC_FLOATS floats
    // In uint4 units: NUM_WAVES * 64 * ACC_FLOATS * sizeof(float) / sizeof(uint4)
    static constexpr int REDUCE_LDS_UINT4 = NUM_WAVES * 64 * ACC_FLOATS / 4;
    static constexpr int IO_REDUCE_LDS = INPUT_TOTAL + REDUCE_LDS_UINT4;

    static constexpr int WEIGHT_LDS = TC::WEIGHT_LDS_SIZE_UINT4;
    static constexpr int UNIFIED_LDS_SIZE = (WEIGHT_LDS > IO_REDUCE_LDS) ? WEIGHT_LDS : IO_REDUCE_LDS;

    __shared__ uint4 lds_buf[UNIFIED_LDS_SIZE];

    float* reduce_lds = reinterpret_cast<float*>(lds_buf + INPUT_TOTAL);

    // --- Coordinate setup ---
    const int C_in  = is_dgrad ? K : C;
    const int C_out = is_dgrad ? C : K;

    BlockCoordsT bc(C_in, C_out);
    if(bc.block_n >= N)
        return;

    const int weight_block_k = bc.block_k_start;
    const int wave_id = static_cast<int>(threadIdx.x) / 64;

    constexpr int y_padding = 0;
    constexpr int stride = 1;
    constexpr int dilation = 1;

    // --- Weight prologue: each wave loads only its own C-slice weights ---
    // Load all c_local weight slices through LDS, each wave keeps only its own.
    WeightLoaderT wl;
    constexpr int C_LOCAL_COUNT = cfg.c_local_count();
    constexpr int CPG = cfg.channels_per_group();

    for(int c_local = 0; c_local < C_LOCAL_COUNT; c_local++)
    {
        __syncthreads();
        if constexpr(is_dgrad)
            WeightLoaderT::load_kyxc_to_lds_dgrad(lds_buf, wei,
                                                    c_local * CPG, weight_block_k, C);
        else
            WeightLoaderT::load_kyxc_to_lds(lds_buf, wei,
                                              weight_block_k, c_local, C);
        wait_vmcnt<0>();
        __syncthreads();

        // Each wave reads only the weights for its own C-slice.
        if(c_local == wave_id)
            wl.read_from_lds(lds_buf);
    }

    __syncthreads();

    // --- Construct InputLoader and OutputWriter ---
    InputLoaderT il(bc, lds_buf, in, hi, wi, px, y_padding,
                    dilation, dilation, stride, stride);
    OutputWriterT ow(bc, nullptr, out, ho, wo);

    // --- Prefetch first input row ---
    il.prefetch_tile_to_lds(0);

    // --- Circular accumulator buffer ---
    constexpr AccType Zero{};
    AccType acc[cfg.kh];
    static_for<cfg.kh>(
        [&]<int I>()
        { acc[I] = Zero; });

    int tic = 1;
    int toc = 0;

    MfmaFn mfma_fn{};

    // Helper lambda: LDS reduction + wave-0 output flush.
    auto reduce_and_flush = [&](AccType& slot, int p_out)
    {
        cross_wave_reduce<NUM_WAVES>(slot, reduce_lds, wave_id);
        if(wave_id == 0)
            ow.flush(slot, p_out);
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

                if((y + 1) < hi)
                    il.fetch_tile_to_lds(tic);

                // MFMA core: each wave uses its own C-section of the input.
                static_for<cfg.kw>(
                    [&]<int S>()
                    {
                        typename InputLoaderT::input_type input_reg;
                        il.read_from_lds(input_reg, S, toc);

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

                tic ^= 1;
                toc ^= 1;

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

                if((y + 1) < hi)
                    il.fetch_tile_to_lds(tic);

                static_for<cfg.kw>(
                    [&]<int S>()
                    {
                        typename InputLoaderT::input_type input_reg;
                        il.read_from_lds(input_reg, S, toc);

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

                tic ^= 1;
                toc ^= 1;

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
