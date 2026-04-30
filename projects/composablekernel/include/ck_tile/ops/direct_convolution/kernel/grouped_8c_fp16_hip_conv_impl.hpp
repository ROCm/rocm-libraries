// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/direct_convolution/utils/matrix_layout.hpp"
#include "ck_tile/ops/direct_convolution/utils/swizzle.hpp"
#include "ck_tile/ops/direct_convolution/utils/types.hpp"
#include "ck_tile/ops/direct_convolution/utils/mathutil.hpp"
#include "ck_tile/ops/direct_convolution/utils/launch_params.hpp"
#include "ck_tile/ops/direct_convolution/utils/kernel_variant.hpp"
#include "ck_tile/ops/direct_convolution/utils/memory.hpp"
#include "ck_tile/ops/direct_convolution/utils/detail.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <string>

// ============================================================================
// Toeplitz FIR transforms for 8-channel grouped convolution
// ============================================================================
//
// Maps the 1D convolution F(2,3) — a 3-tap filter producing 2 outputs — into
// the 16x16x32 MFMA instruction via a block-Toeplitz matrix structure.
//
// Each data/filter element is a vector of 8 channels, giving:
//   D ~ 16 x 32 = [16 tiles] x [4 inputs/tile x 8 channels]
//   G ~ 32 x 16 = [4 taps x 8 input channels] x [2 outputs x 8 output channels]
//   Q = D G ~ 16 x 16
//
// This achieves 75% utilization of the MFMA (1/4 of G entries are zero).
namespace ck_tile::direct_hip_conv::grouped_8c_transforms
{

// GT is the transpose of the Toeplitz filter matrix G.
//
//            t=0  t=1  t=2  t=3
//          [ g0   g1   g2   0  ]  q=0
//   GT =   [  0   g0   g1   g2 ]  q=1
//
// GT ~ 16 x 32 = [2q x 8k] x [4t x 8c]
struct GT
{
    static constexpr int rows       = 16;
    static constexpr int cols       = 32;
    static constexpr int group_size = 8;

    // The row of the block matrix.
    static constexpr int q(int row) { return row / group_size; }

    // The column of the block matrix.
    static constexpr int t(int col) { return col / group_size; }

    // One of 8 output channels.
    static constexpr int k(int row) { return row % group_size; }

    // One of 8 input channels.
    static constexpr int c(int col) { return col % group_size; }

    // One of two sets of four input channels.
    static constexpr int c4(int col4) { return col4 % (group_size / 4); }

    // One of 3 filter taps.
    static constexpr int s(int row, int col4) { return GT::t(col4 * 4) - GT::q(row); }

    static constexpr int filter_is_zero(int row, int col4)
    {
        int ss = GT::s(row, col4);
        return ss < 0 || ss > 2;
    }
};

// D matrix: each row is a tile of 4 input elements (each a vector of 8 channels).
// D ~ 16 tiles x (4 inputs/tile x 8 channels) = 16 x 32
struct D
{
    static constexpr int rows       = 16;
    static constexpr int cols       = 32;
    static constexpr int group_size = 8;

    static constexpr int p(int row) { return row; }
    static constexpr int w(int col) { return col / group_size; }
    static constexpr int c(int col) { return col % group_size; }
    static constexpr int c4(int col4) { return col4 % (group_size / 4); }
};

} // namespace ck_tile::direct_hip_conv::grouped_8c_transforms

// ============================================================================
// 8-channel grouped convolution kernel (HIP reference implementation)
// ============================================================================
namespace ck_tile::direct_hip_conv::grouped_8c
{

using namespace ck_tile::direct_conv;

// Resolve ambiguity with ck_tile:: types when included alongside CK Tile core headers
using ck_tile::direct_conv::fp16x4_t;
using ck_tile::direct_conv::fp16x8_t;
using ck_tile::direct_conv::fp32x4_t;
using ck_tile::direct_conv::static_for;

// 64 threads per wave.
constexpr int WAVE_SIZE = 64;

// Block output is 32 columns wide (16 tiles x 2 outputs per tile).
constexpr int BLOCK_Q = 32;

// Kernel configuration parameters.
struct Config
{
    int waves_per_wg;

    int kh = 3;
    int kw = 3;

    int group_size = 8;

    int n_fold = 8;

    Direction direction = Direction::Fprop;

    constexpr int block_c() const { return group_size * waves_per_wg; }

    constexpr int block_size() const { return waves_per_wg * WAVE_SIZE; }

    std::string GetName() const
    {
        return "grouped_8c_waves_per_wg_" + std::to_string(waves_per_wg);
    }
};

// All instantiated configurations.
constexpr Config configs[] = {
    {.waves_per_wg = 16, .direction = Direction::Dgrad},
    {.waves_per_wg = 8, .direction = Direction::Dgrad},
    {.waves_per_wg = 7, .direction = Direction::Dgrad},
    {.waves_per_wg = 6, .direction = Direction::Dgrad},
    {.waves_per_wg = 5, .direction = Direction::Dgrad},
    {.waves_per_wg = 4, .direction = Direction::Dgrad},
    {.waves_per_wg = 3, .direction = Direction::Dgrad},
    {.waves_per_wg = 2, .direction = Direction::Dgrad},
    {.waves_per_wg = 1, .direction = Direction::Dgrad},
    {.waves_per_wg = 16},
    {.waves_per_wg = 8},
    {.waves_per_wg = 7},
    {.waves_per_wg = 6},
    {.waves_per_wg = 5},
    {.waves_per_wg = 4},
    {.waves_per_wg = 3},
    {.waves_per_wg = 2},
    {.waves_per_wg = 1},
};

constexpr int NUM_CONFIGS = sizeof(configs) / sizeof(configs[0]);

inline bool is_valid_config(const Conv2dParams& par, const Config& cfg)
{
    if(par.direction != cfg.direction)
        return false;

    // Require that the number of groups per work-group evenly divides the total number of
    // groups.
    if((par.groups % cfg.waves_per_wg) != 0)
    {
        return false;
    }

    return true;
}

inline LaunchParams get_launch_params(int config_idx, const Conv2dParams& par)
{
    const auto& cfg = configs[config_idx];

    const int out_q    = (cfg.direction == Direction::Dgrad) ? par.w : par.q;
    auto blocks_w      = ck_tile::integer_divide_ceil(out_q, BLOCK_Q);
    auto blocks_w_n    = blocks_w * cfg.n_fold;
    auto blocks_c      = ck_tile::integer_divide_ceil(par.c_tot, cfg.block_c());
    auto blocks_n_fold = ck_tile::integer_divide_ceil(par.n, cfg.n_fold);

    LaunchParams launch;
    launch.grid       = dim3(blocks_w_n, blocks_c, blocks_n_fold);
    launch.block_size = dim3(cfg.block_size(), 1, 1);
    return launch;
}

} // namespace ck_tile::direct_hip_conv::grouped_8c

template <ck_tile::direct_hip_conv::grouped_8c::Config cfg>
__device__ void conv2d_grouped_8c_fp16_cdna4_nhwc_impl(const _Float16* __restrict__ in,
                                                        const _Float16* __restrict__ wei,
                                                        double alpha,
                                                        double beta,
                                                        _Float16* __restrict__ out,
                                                        int N,
                                                        int groups,
                                                        int c_per_group,
                                                        int k_per_group,
                                                        int hi,
                                                        int wi,
                                                        int ho,
                                                        int wo,
                                                        int fy,
                                                        int fx,
                                                        int sy,
                                                        int sx,
                                                        int dy,
                                                        int dx,
                                                        int py,
                                                        int px)
{
    using namespace ck_tile::direct_hip_conv::grouped_8c;
    using namespace ck_tile::direct_hip_conv::grouped_8c_transforms;
    using namespace ck_tile::direct_conv;
    using fp16x8_t = __attribute__((ext_vector_type(8))) _Float16;

    // MFMA 16x16x32: result is fp32x4 (16x16 tile).
    using ResultLayout  = MatrixLayout<16, 16, 1, float>;

    constexpr int GROUP_SIZE_8 = cfg.group_size / 8; // 1
    constexpr int GROUP_SIZE_4 = cfg.group_size / 4; // 2

    static_assert(sizeof(uint4) / sizeof(__half) == 8, "Expected 8 halves per uint4 vector");

    // Number of input columns loaded by each workgroup (output columns plus halo)
    constexpr int BLOCK_W = BLOCK_Q + (cfg.kw - 1);

    // uint4 vectors per channels fiber
    constexpr int BLOCK_C8 = cfg.block_c() / 8;

    // Each wave processes a single group of channels.
    constexpr int BLOCK_GROUPS = cfg.waves_per_wg;

    // Vectors to store (output only)
    constexpr int STORE_VECS = BLOCK_Q * BLOCK_C8;

    constexpr int NUM_INPUT_LDS_BUFFERS = 2;

    // Size of LDS buffer for inputs (swizzled, no padding needed)
    constexpr int INPUT_LDS_BUFFER_SIZE_C8 = BLOCK_C8 * BLOCK_W;

    // Size of LDS buffer for outputs (swizzled, no padding needed)
    constexpr int OUTPUT_LDS_BUFFER_SIZE = BLOCK_C8 * BLOCK_Q;

    // Weight LDS: all K-rows for this block, in uint4 units.
    // Each uint4 holds one (k, r, s) tap: all 8 input channels.
    constexpr int WEIGHT_LDS_SIZE_UINT4 =
        BLOCK_GROUPS * cfg.group_size * cfg.kh * cfg.kw * GROUP_SIZE_8;

    // Unified LDS: large enough for either (input double-buffer + output) or (weight prologue).
    constexpr int IO_LDS_SIZE =
        NUM_INPUT_LDS_BUFFERS * INPUT_LDS_BUFFER_SIZE_C8 + OUTPUT_LDS_BUFFER_SIZE;
    __shared__ uint4 lds_buf[maximum(WEIGHT_LDS_SIZE_UINT4, IO_LDS_SIZE)];
    uint4* input_lds  = lds_buf;
    uint4* output_lds = lds_buf + NUM_INPUT_LDS_BUFFERS * INPUT_LDS_BUFFER_SIZE_C8;

    const int tid  = threadIdx.x;
    const int wave = tid / WAVE_SIZE;
    const int lane = tid % WAVE_SIZE;

    // Workgroup coordinates.
    const int block_q_n_idx   = blockIdx.x;
    const int block_n_mod_idx = block_q_n_idx % cfg.n_fold;
    const int block_q_idx     = block_q_n_idx / cfg.n_fold;
    const int block_group_idx = blockIdx.y;
    const int block_n_div_idx = blockIdx.z;
    const int block_n_idx     = block_n_div_idx * cfg.n_fold + block_n_mod_idx;
    if(block_n_idx >= N)
        return;

    const int block_n     = block_n_idx;
    const int block_q     = block_q_idx * BLOCK_Q;
    const int block_group = block_group_idx * BLOCK_GROUPS;
    const int block_k     = block_group * cfg.group_size;
    const int block_c8    = block_group * GROUP_SIZE_8; // in uint4 units

    // Base pointer for this batch image in NHWC layout (all in uint4 units)
    const int C  = groups * cfg.group_size;
    const int C8 = C / 8;

    const size_t offset_block = (size_t)block_n * hi * wi * C8;
    const size_t wi_stride    = (size_t)wi * C8;
    const size_t wo_stride    = (size_t)wo * C8;

    // Map threads to LDS input buffer addresses linearly.
    auto store_input_lds = &input_lds[tid];

    // Create the input buffer resource.
    auto input_bytes               = static_cast<size_t>(N) * hi * wi * C * sizeof(half);
    constexpr int rsrc_data_format = 1 << 15;
    auto input_rsrc                = __builtin_amdgcn_make_buffer_rsrc(
        const_cast<_Float16*>(in), 0, input_bytes, rsrc_data_format);
    uint32_t input_voffset;

    // Map threads to global memory input buffer using a swizzle.
    using Sw               = SwizzleT<cfg.block_c()>;
    const int col          = Sw::x(tid);
    const int c8_thread    = Sw::c8(tid);
    const int global_col   = (block_q - px) + col;
    const bool load_active = (tid < (BLOCK_W)*BLOCK_C8);
    if(0 <= global_col && global_col < wi)
    {
        input_voffset =
            sizeof(uint4) * (offset_block + (size_t)global_col * C8 + block_c8 + c8_thread);
    }
    else
    {
        input_voffset = input_bytes;
    }

    // Each wave computes one group of channels.
    auto wave_group = wave;

    // Load weights from global memory through LDS into registers.
    // Weight register: one fp16x8 per filter row R (S is embedded in the Toeplitz structure).
    fp16x8_t weights_reg[cfg.kh];
    {
        auto weight_bytes = static_cast<size_t>(groups * cfg.group_size) * cfg.kh * cfg.kw *
                            cfg.group_size * sizeof(half);
        auto weight_rsrc  = __builtin_amdgcn_make_buffer_rsrc(
            const_cast<_Float16*>(wei), 0, weight_bytes, rsrc_data_format);
        uint32_t voffset_base = block_k * cfg.kh * cfg.kw * cfg.group_size * sizeof(half);

        for(int j = tid; j < WEIGHT_LDS_SIZE_UINT4; j += cfg.block_size())
        {
            __builtin_amdgcn_raw_ptr_buffer_load_lds(
                weight_rsrc, &lds_buf[j], 16, voffset_base + j * sizeof(uint4), 0, 0, 0);
        }

        wait_vmcnt<0>();
        __syncthreads();

        // Map lane to GT matrix position.
        int row = lane % 16;
        int g   = lane / 16;

        if constexpr(cfg.direction == Direction::Dgrad)
        {
            int s_dgrad = cfg.kw - 1 - g + GT::q(row);
            int c_out   = GT::k(row);

            if(GT::filter_is_zero(row, g * 2))
            {
                for(int r = 0; r < cfg.kh; r++)
                    *reinterpret_cast<uint4*>(&weights_reg[r]) = uint4{0, 0, 0, 0};
            }
            else
            {
                const __half* wei_half = reinterpret_cast<const __half*>(lds_buf);
                int base_k             = wave_group * cfg.group_size;
                for(int r = 0; r < cfg.kh; r++)
                {
                    for(int k = 0; k < cfg.group_size; k++)
                    {
                        int idx = (base_k + k) * cfg.kh * cfg.kw * cfg.group_size +
                                  r * cfg.kw * cfg.group_size + s_dgrad * cfg.group_size + c_out;
                        weights_reg[r][k] = wei_half[idx];
                    }
                }
            }
        }
        else
        {
            int k_val = GT::k(row);
            int s_val = GT::s(row, g * 2);

            if(GT::filter_is_zero(row, g * 2))
            {
                for(int r = 0; r < cfg.kh; r++)
                    *reinterpret_cast<uint4*>(&weights_reg[r]) = uint4{0, 0, 0, 0};
            }
            else
            {
                int k_within_wg = wave_group * cfg.group_size + k_val;
                for(int r = 0; r < cfg.kh; r++)
                {
                    int offset     = k_within_wg * cfg.kh * cfg.kw + r * cfg.kw + s_val;
                    weights_reg[r] = *reinterpret_cast<const fp16x8_t*>(&lds_buf[offset]);
                }
            }
        }
    }

    // Ensure all threads have finished reading weights from LDS before
    // the input buffer_load_lds overwrites the same lds_buf region.
    __syncthreads();

    // Load first chunk of inputs asynchronously.
    if(load_active)
    {
        __builtin_amdgcn_raw_ptr_buffer_load_lds(
            input_rsrc, store_input_lds, 16, input_voffset, 0, 0, 0);
    }

    // Map threads to output addresses for transferring through LDS to global memory.
    const bool store_active      = (tid < STORE_VECS);
    const uint4* load_output_lds = nullptr;
    uint4* store_output_global   = nullptr;
    if(store_active)
    {
        const int c8    = tid % BLOCK_C8;
        const int q     = block_q + col;
        load_output_lds = &output_lds[Sw::offset_uint4(col, c8)];
        if(q < wo)
        {
            const int K8        = C / 8;
            store_output_global = reinterpret_cast<uint4*>(out) + (size_t)block_n * ho * wo * K8 +
                                  (size_t)q * K8 + block_c8 + c8;
        }
    }

    // Pre-compute the input LDS offset (uint4 units) for this thread.
    // Each lane reads ONE uint4 from input LDS (no S-loop needed).
    // Input position: x = 2*(lane%16) + lane/16, channel group: wave_group.
    const int input_x          = 2 * (lane % 16) + lane / 16;
    const int input_lds_offset = Sw::offset_uint4(input_x, wave_group);

    // Pre-compute the output LDS swizzle offset (thread-constant).
    // The result matrix row encodes (q_tile, k) via the GT structure.
    const int result_n        = ResultLayout::outer(lane);
    const int result_row      = ResultLayout::inner(lane, 0);
    const int output_col      = 2 * result_n + GT::q(result_row);
    const int c4_within_group = result_row / 4 % 2;
    const int output_lds_offset =
        Sw::offset_uint2(output_col, wave_group * GROUP_SIZE_4 + c4_within_group);

    // Circular buffer of accumulators.
    constexpr auto Zero = fp32x4_t{0.f, 0.f, 0.f, 0.f};
    fp32x4_t acc[cfg.kh];
    for(int i = 0; i < cfg.kh; i++)
        acc[i] = Zero;

    int tic = 1;
    int toc = 0;

    // Main loop iterates over input rows.
    for(int y_base = 0; y_base + cfg.kh <= hi; y_base += cfg.kh)
    {
        static_for<cfg.kh>(
            [&]<int Y_LOCAL>()
            {
                wait_vmcnt<0>();
                __syncthreads();

                int y = y_base + Y_LOCAL;

                if(load_active && (y + 1) < hi)
                {
                    __builtin_amdgcn_raw_ptr_buffer_load_lds(
                        input_rsrc,
                        store_input_lds + tic * INPUT_LDS_BUFFER_SIZE_C8,
                        16,
                        input_voffset + (y + 1) * wi_stride * sizeof(uint4),
                        0,
                        0,
                        0);
                }

                // Load input operand B: ONE uint4 per lane (no S-loop).
                fp16x8_t input_reg = *reinterpret_cast<const fp16x8_t*>(
                    &input_lds[toc * INPUT_LDS_BUFFER_SIZE_C8 + input_lds_offset]);

                // Accumulate: R-loop only (S is embedded in the Toeplitz structure).
                static_for<cfg.kh>(
                    [&]<int R>()
                    {
                        constexpr int p_idx = (Y_LOCAL - R + cfg.kh) % cfg.kh;
                        if constexpr(cfg.direction == Direction::Dgrad)
                            acc[p_idx] = __builtin_amdgcn_mfma_f32_16x16x32_f16(
                                weights_reg[cfg.kh - 1 - R], input_reg, acc[p_idx], 0, 0, 0);
                        else
                            acc[p_idx] = __builtin_amdgcn_mfma_f32_16x16x32_f16(
                                weights_reg[R], input_reg, acc[p_idx], 0, 0, 0);
                    });

                tic ^= 1;
                toc ^= 1;

                constexpr int P_FLUSH = (Y_LOCAL + 1) % cfg.kh;
                int p_out             = y + py - (cfg.kh - 1);
                if(p_out >= 0 && p_out < ho)
                {
                    __half2 halves[2];
                    halves[0]    = __float22half2_rn({acc[P_FLUSH][0], acc[P_FLUSH][1]});
                    halves[1]    = __float22half2_rn({acc[P_FLUSH][2], acc[P_FLUSH][3]});
                    auto out_reg = *reinterpret_cast<const fp16x4_t*>(halves);

                    auto* store_output_lds              = reinterpret_cast<uint2*>(output_lds);
                    store_output_lds[output_lds_offset] = *reinterpret_cast<const uint2*>(&out_reg);

                    __syncthreads(); // output_lds fully written

                    if(store_output_global)
                        store_output_global[p_out * wo_stride] = *load_output_lds;
                }
                acc[P_FLUSH] = Zero;
            });
    }

    // Remainder: the hi % kh leftover rows.
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

                if(load_active && (y + 1) < hi)
                {
                    __builtin_amdgcn_raw_ptr_buffer_load_lds(
                        input_rsrc,
                        store_input_lds + tic * INPUT_LDS_BUFFER_SIZE_C8,
                        16,
                        input_voffset + (y + 1) * wi_stride * sizeof(uint4),
                        0,
                        0,
                        0);
                }

                fp16x8_t input_reg = *reinterpret_cast<const fp16x8_t*>(
                    &input_lds[toc * INPUT_LDS_BUFFER_SIZE_C8 + input_lds_offset]);

                static_for<cfg.kh>(
                    [&]<int R>()
                    {
                        constexpr int p_idx = (Y_LOCAL - R + cfg.kh) % cfg.kh;
                        if constexpr(cfg.direction == Direction::Dgrad)
                            acc[p_idx] = __builtin_amdgcn_mfma_f32_16x16x32_f16(
                                weights_reg[cfg.kh - 1 - R], input_reg, acc[p_idx], 0, 0, 0);
                        else
                            acc[p_idx] = __builtin_amdgcn_mfma_f32_16x16x32_f16(
                                weights_reg[R], input_reg, acc[p_idx], 0, 0, 0);
                    });

                tic ^= 1;
                toc ^= 1;

                constexpr int P_FLUSH = (Y_LOCAL + 1) % cfg.kh;
                int p_out             = y + py - (cfg.kh - 1);
                if(p_out >= 0 && p_out < ho)
                {
                    __half2 halves[2];
                    halves[0]    = __float22half2_rn({acc[P_FLUSH][0], acc[P_FLUSH][1]});
                    halves[1]    = __float22half2_rn({acc[P_FLUSH][2], acc[P_FLUSH][3]});
                    auto out_reg = *reinterpret_cast<const fp16x4_t*>(halves);

                    auto* store_output_lds              = reinterpret_cast<uint2*>(output_lds);
                    store_output_lds[output_lds_offset] = *reinterpret_cast<const uint2*>(&out_reg);

                    __syncthreads(); // output_lds fully written

                    if(store_output_global)
                        store_output_global[p_out * wo_stride] = *load_output_lds;
                }
                acc[P_FLUSH] = Zero;
            });
    }

    // Flush output rows whose last input contribution would land at y >= hi.
    for(int p_out = hi - cfg.kh + 1 + py; p_out < ho; p_out++)
    {
        __syncthreads(); // separate prior LDS reads from this iteration's writes
        int p_idx = (p_out - py + cfg.kh) % cfg.kh;
        fp32x4_t slot;
        dispatch<cfg.kh>(p_idx,
                         [&]<int P>()
                         {
                             slot   = acc[P];
                             acc[P] = Zero;
                         });

        __half2 halves[2];
        halves[0]    = __float22half2_rn({slot[0], slot[1]});
        halves[1]    = __float22half2_rn({slot[2], slot[3]});
        auto out_reg = *reinterpret_cast<const fp16x4_t*>(halves);

        auto* store_output_lds              = reinterpret_cast<uint2*>(output_lds);
        store_output_lds[output_lds_offset] = *reinterpret_cast<const uint2*>(&out_reg);

        __syncthreads(); // output_lds fully written

        if(store_output_global)
            store_output_global[p_out * wo_stride] = *load_output_lds;
    }
}

template <ck_tile::direct_hip_conv::grouped_8c::Config cfg>
__global__ void conv2d_grouped_8c_fp16_nhwc_cdna4(const _Float16* __restrict__ in,
                                                   const _Float16* __restrict__ wei,
                                                   double alpha,
                                                   double beta,
                                                   _Float16* __restrict__ out,
                                                   int N,
                                                   int groups,
                                                   int c_per_group,
                                                   int k_per_group,
                                                   int hi,
                                                   int wi,
                                                   int ho,
                                                   int wo,
                                                   int fy,
                                                   int fx,
                                                   int sy,
                                                   int sx,
                                                   int dy,
                                                   int dx,
                                                   int py,
                                                   int px)
{
    conv2d_grouped_8c_fp16_cdna4_nhwc_impl<cfg>(in,
                                                 wei,
                                                 alpha,
                                                 beta,
                                                 out,
                                                 N,
                                                 groups,
                                                 c_per_group,
                                                 k_per_group,
                                                 hi,
                                                 wi,
                                                 ho,
                                                 wo,
                                                 fy,
                                                 fx,
                                                 sy,
                                                 sx,
                                                 dy,
                                                 dx,
                                                 py,
                                                 px);
}

namespace ck_tile::direct_hip_conv::grouped_8c
{

template <size_t... Is>
void launch_dispatch(int config_idx,
                     std::index_sequence<Is...>,
                     const LaunchParams& lp,
                     const Conv2dParams& par,
                     const void* in,
                     const void* wei,
                     void* out,
                     hipStream_t stream)
{
    auto kernel_launch = [&]<size_t I>()
    {
        auto view = SizeView<configs[I].direction>(par);
        conv2d_grouped_8c_fp16_nhwc_cdna4<configs[I]>
            <<<lp.grid, lp.block_size, lp.dynamic_shared_bytes, stream>>>(
                static_cast<const _Float16*>(in),
                static_cast<const _Float16*>(wei),
                1.0,
                0.0,
                static_cast<_Float16*>(out),
                par.n,
                par.groups,
                par.channels_per_group(),
                par.filters_per_group(),
                view.h(),
                view.w(),
                view.p(),
                view.q(),
                par.kh,
                par.kw,
                par.stride_h,
                par.stride_w,
                par.dilation_h,
                par.dilation_w,
                view.pad_h(),
                view.pad_w());
    };
    (void)((config_idx == static_cast<int>(Is) ? (kernel_launch.template operator()<Is>(), true)
                                               : false) ||
           ...);
}

inline void launch(int config_idx,
                   const LaunchParams& lp,
                   const Conv2dParams& par,
                   const void* in,
                   const void* wei,
                   void* out,
                   void* /*workspace*/,
                   hipStream_t stream)
{
    launch_dispatch(
        config_idx, std::make_index_sequence<NUM_CONFIGS>{}, lp, par, in, wei, out, stream);
}

constexpr KernelVariant make_variant()
{
    return {
        .is_applicable =
            [](const Conv2dParams& par)
        {
            if(par.in_type != DataType::fp16)
                return false;
            if(par.wei_type != DataType::fp16)
                return false;
            if(par.out_type != DataType::fp16)
                return false;
            if(par.order != TensorOrder::NHWC)
                return false;
            if(par.direction != Direction::Fprop &&
               par.direction != Direction::Dgrad)
                return false;
            if(par.kh != 3 || par.kw != 3)
                return false;
            if(par.k_tot != par.c_tot)
                return false;
            if(par.channels_per_group() != 8)
                return false;
            if(par.c_tot % 8 != 0)
                return false;
            if(par.stride_h != 1 || par.stride_w != 1)
                return false;
            if(par.dilation_h != 1 || par.dilation_w != 1)
                return false;
            if(par.pad_h > par.kh - 1 || par.pad_w > par.kw - 1)
                return false;
            return true;
        },
        .config_is_compatible = [](const Conv2dParams& par, int idx)
        { return is_valid_config(par, configs[idx]); },
        .get_launch_params  = &get_launch_params,
        .launch             = &launch,
        .get_workspace_size = [](int, const Conv2dParams&) -> size_t { return 0; },
        .num_configs        = NUM_CONFIGS,
    };
}

} // namespace ck_tile::direct_hip_conv::grouped_8c
