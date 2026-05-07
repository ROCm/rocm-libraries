// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/direct_convolution/utils/matrix_layout.hpp"
#include "ck_tile/ops/direct_convolution/utils/swizzle.hpp"
#include "ck_tile/ops/direct_convolution/utils/types.hpp"
#include "ck_tile/ops/direct_convolution/utils/mathutil.hpp"
#include "ck_tile/ops/direct_convolution/utils/launch_params.hpp"
#include "ck_tile/ops/direct_convolution/utils/kernel_variant.hpp"
#include "ck_tile/ops/direct_convolution/utils/transpose_lds_layout.hpp"
#include "ck_tile/ops/direct_convolution/utils/memory.hpp"
#include "ck_tile/ops/direct_convolution/utils/detail.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <string>

namespace ck_tile::direct_hip_conv::grouped_32c
{

using namespace ck_tile::direct_conv;

// Resolve ambiguity with ck_tile:: types when included alongside CK Tile core headers
using ck_tile::direct_conv::fp16x4_t;
using ck_tile::direct_conv::fp16x8_t;
using ck_tile::direct_conv::fp32x4_t;
using ck_tile::direct_conv::static_for;

// 64 threads per wave.
constexpr int WAVE_SIZE = 64;

// Block output is 16 columns wide.
constexpr int BLOCK_Q = 16;

// Kernel configuration parameters.
struct Config
{
    int waves_per_wg;

    int kh = 3;
    int kw = 3;

    int group_size = 32;

    int n_fold = 8;

    Direction direction = Direction::Fprop;

    constexpr int block_c() const { return group_size * waves_per_wg / 2; }

    constexpr int block_size() const { return waves_per_wg * WAVE_SIZE; }

    std::string GetName() const
    {
        return "grouped_32c_waves_per_wg_" + std::to_string(waves_per_wg);
    }
};

// All instantiated configurations.
// waves_per_wg = groups_per_wg * 2 (since 32c uses 2 waves per group)
// groups_per_wg=2 -> waves_per_wg=4, groups_per_wg=1 -> waves_per_wg=2
constexpr Config configs[] = {
    {.waves_per_wg = 4, .direction = Direction::Dgrad},
    {.waves_per_wg = 2, .direction = Direction::Dgrad},
    {.waves_per_wg = 4},
    {.waves_per_wg = 2},
};

constexpr int NUM_CONFIGS = sizeof(configs) / sizeof(configs[0]);

inline bool is_valid_config(const Conv2dParams& par, const Config& cfg)
{
    if(par.direction != cfg.direction)
    {
        return false;
    }

    // Require that the number of groups per work-group evenly divides the total number of
    // groups. groups_per_wg = waves_per_wg / 2.
    if((par.groups % (cfg.waves_per_wg / 2)) != 0)
    {
        return false;
    }

    return true;
}

inline LaunchParams get_launch_params(int config_idx, const Conv2dParams& par)
{
    const auto& cfg = configs[config_idx];

    // Compute the grid size.
    // For Dgrad the output is the input gradient (width = par.w, not par.q).
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

} // namespace ck_tile::direct_hip_conv::grouped_32c

template <ck_tile::direct_hip_conv::grouped_32c::Config cfg>
__device__ void conv2d_grouped_32c_fp16_cdna4_nhwc_impl(const _Float16* __restrict__ in,
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
    using namespace ck_tile::direct_hip_conv::grouped_32c;
    using namespace ck_tile::direct_conv;
    // MFMA 16x16x32: operands are fp16x8, result is fp32x4.
    using OperandLayout = MatrixLayout<16, 32, 1, __half>;
    using ResultLayout  = MatrixLayout<16, 16, 1, float>;
    using fp16x8_t      = __attribute__((ext_vector_type(8))) _Float16;
    using int16x4_t     = __attribute__((ext_vector_type(4))) short;
    using int16x8_t     = __attribute__((ext_vector_type(8))) short;

    // groups_per_wg = waves_per_wg / 2
    constexpr int GROUPS_PER_WG = cfg.waves_per_wg / 2;

    constexpr int GROUP_SIZE_8 = cfg.group_size / 8; // 4
    constexpr int GROUP_SIZE_4 = cfg.group_size / 4; // 8

    static_assert(sizeof(uint4) / sizeof(__half) == 8, "Expected 8 halves per uint4 vector");

    // Number of input columns loaded by each workgroup (output columns plus halo)
    constexpr int BLOCK_W = BLOCK_Q + (cfg.kw - 1);

    // uint4 vectors per channels fiber
    constexpr int BLOCK_C8 = cfg.block_c() / 8;

    // Each pair of waves processes one group of channels.
    constexpr int BLOCK_GROUPS = GROUPS_PER_WG;

    // Vectors to store (center only)
    constexpr int STORE_VECS = BLOCK_Q * BLOCK_C8;

    constexpr int NUM_INPUT_LDS_BUFFERS = 2;

    // Size of LDS buffer for inputs (swizzled, no padding needed)
    constexpr int INPUT_LDS_BUFFER_SIZE_C8 = BLOCK_C8 * BLOCK_W;

    // Size of LDS buffer for outputs (swizzled, no padding needed)
    constexpr int OUTPUT_LDS_BUFFER_SIZE = BLOCK_C8 * BLOCK_Q;

    // Weight LDS: all K-rows for this block, in uint4 units.
    // Each K-row is kh * kw * GROUP_SIZE_8 uint4 elements; total K-rows = BLOCK_GROUPS *
    // group_size.
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

    // Each group is split across 2 waves.
    const int wave_group = wave / 2;
    const int wave_half  = wave % 2; // 0 = output channels 0-15, 1 = channels 16-31

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

    // Single-pass input loading.
    // Each row needs BLOCK_W * BLOCK_C8 = 18 * groups_per_wg*4 = 72*groups_per_wg uint4 loads.
    // Block has 128*groups_per_wg threads, so single pass always suffices (72 < 128).
    auto store_input_lds_0 = &input_lds[tid];

    // Create the input buffer resource.
    auto input_bytes               = static_cast<size_t>(N) * hi * wi * C * sizeof(half);
    constexpr int rsrc_data_format = 1 << 15;
    auto input_rsrc                = __builtin_amdgcn_make_buffer_rsrc(
        const_cast<_Float16*>(in), 0, input_bytes, rsrc_data_format);
    uint32_t input_voffset_0;

    // Map threads to global memory input buffer using a swizzle.
    using Sw = SwizzleT<cfg.block_c()>;

    // Single pass: all threads (tid < block_size)
    const int col_0          = Sw::x(tid);
    const int c8_0           = Sw::c8(tid);
    const int global_col_0   = (block_q - px) + col_0;
    const bool load_active_0 = (tid < BLOCK_W * BLOCK_C8);
    if(load_active_0 && 0 <= global_col_0 && global_col_0 < wi)
    {
        input_voffset_0 =
            sizeof(uint4) * (offset_block + (size_t)global_col_0 * C8 + block_c8 + c8_0);
    }
    else
    {
        input_voffset_0 = input_bytes;
    }

    // Load weights from global memory through LDS into registers.
    // Each wave loads only its half of output channels.
    fp16x8_t weights_reg[cfg.kh * cfg.kw];
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

        if constexpr(cfg.direction == Direction::Dgrad)
        {
            // Dgrad: load transposed weights using DS_READ_B64_TR_B16.
            // Two calls per filter position load a complete 16x32 MFMA B operand.
            //
            // LDS layout is [k_all][kh*kw][c] (straight copy from global).
            // k_stride = kh * kw * group_size (in halves) between output channels.
            const int k_stride = cfg.kh * cfg.kw * cfg.group_size;
            auto* wei_grp_base =
                reinterpret_cast<__half*>(lds_buf) + wave_group * cfg.group_size * k_stride;

            using TransposeLayout = TransposeLDSLayout<16, 32>;
            const int k_0         = TransposeLayout::row(lane, 0);
            const int k_1         = TransposeLayout::row(lane, 1);
            const int c           = wave_half * 16 + TransposeLayout::col(lane);

            for(int khw = 0; khw < cfg.kh * cfg.kw; khw++)
            {
                // Address in [k][kh*kw][c] layout: k * k_stride + khw * group_size + c
                auto* addr0 = reinterpret_cast<int16x4_t*>(wei_grp_base + k_0 * k_stride +
                                                           khw * cfg.group_size + c);
                auto* addr1 = reinterpret_cast<int16x4_t*>(wei_grp_base + k_1 * k_stride +
                                                           khw * cfg.group_size + c);

                int16x4_t r0 = __builtin_amdgcn_ds_read_tr16_b64_v4i16(addr0);
                int16x4_t r1 = __builtin_amdgcn_ds_read_tr16_b64_v4i16(addr1);

                int16x8_t combined;
                combined[0] = r0[0]; combined[1] = r0[1]; combined[2] = r0[2]; combined[3] = r0[3];
                combined[4] = r1[0]; combined[5] = r1[1]; combined[6] = r1[2]; combined[7] = r1[3];
                weights_reg[khw] = __builtin_bit_cast(fp16x8_t, combined);
            }
        }
        else
        {
            // Fprop: read weights in OperandLayout order.
            // lane_k = lane % 16 (output channel within 16-wide half)
            // lane_c8 = lane / 16 (which 8-channel slice within 32-channel group, 0..3)
            auto lane_k    = OperandLayout::outer(lane);
            auto lane_c8_w = OperandLayout::inner(lane) / 8;

            // wave_half selects which 16-channel half this wave handles.
            auto k = wave_group * cfg.group_size + wave_half * 16 + lane_k;

            auto* weights_lds = reinterpret_cast<const uint4*>(lds_buf);
            for(int khw = 0; khw < cfg.kh * cfg.kw; khw++)
            {
                weights_reg[khw] =
                    *(const fp16x8_t*)(weights_lds + k * (cfg.kh * cfg.kw * GROUP_SIZE_8) +
                                       khw * GROUP_SIZE_8 + lane_c8_w);
            }
        }
    }

    // Ensure all threads have finished reading weights from LDS before
    // the input buffer_load_lds overwrites the same lds_buf region.
    __syncthreads();

    // Load first chunk of inputs asynchronously (single pass).
    if(load_active_0)
    {
        __builtin_amdgcn_raw_ptr_buffer_load_lds(
            input_rsrc, store_input_lds_0, 16, input_voffset_0, 0, 0, 0);
    }

    // Map lanes to MFMA matrix coordinates.
    const int lane_q  = lane % 16; // input position
    const int lane_c8 = lane / 16; // which 8-channel slice within 32-channel group (0..3)
    const int lane_c4 = ResultLayout::inner(lane) / 4;

    // Map threads to output addresses for transferring through LDS to global memory.
    const bool store_active      = (tid < STORE_VECS);
    const uint4* load_output_lds = nullptr;
    uint4* store_output_global   = nullptr;
    if(store_active)
    {
        const int c8    = tid % BLOCK_C8;
        const int q     = block_q + col_0;
        load_output_lds = &output_lds[Sw::offset_uint4(col_0, c8)];
        if(q < wo)
        {
            const int K8        = C / 8;
            store_output_global = reinterpret_cast<uint4*>(out) + (size_t)block_n * ho * wo * K8 +
                                  (size_t)q * K8 + block_c8 + c8;
        }
    }

    // Pre-compute the kw input LDS offsets (uint4 units) for this thread.
    // Each lane reads one uint4 = fp16x8 per filter column S.
    int input_lds_offsets[cfg.kw];
    static_for<cfg.kw>(
        [&]<int S>()
        {
            input_lds_offsets[S] =
                Sw::offset_uint4(lane_q + S, wave_group * GROUP_SIZE_8 + lane_c8);
        });

    // Pre-compute the output LDS swizzle offset (thread-constant).
    // Single write per lane: wave_half selects which 16-channel half.
    const int output_lds_offset =
        Sw::offset_uint2(lane_q, wave_group * GROUP_SIZE_4 + wave_half * 4 + lane_c4);

    // Circular buffer of accumulators: one per filter row (single output half per wave).
    constexpr auto Zero = fp32x4_t{0.f, 0.f, 0.f, 0.f};
    fp32x4_t acc[cfg.kh];
    for(int i = 0; i < cfg.kh; i++)
    {
        acc[i] = Zero;
    }

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

                if((y + 1) < hi)
                {
                    auto row_offset = (y + 1) * wi_stride * sizeof(uint4);
                    if(load_active_0)
                    {
                        __builtin_amdgcn_raw_ptr_buffer_load_lds(input_rsrc,
                                                                 store_input_lds_0 +
                                                                     tic * INPUT_LDS_BUFFER_SIZE_C8,
                                                                 16,
                                                                 input_voffset_0 + row_offset,
                                                                 0,
                                                                 0,
                                                                 0);
                    }
                }

                static_for<cfg.kw>(
                    [&]<int S>()
                    {
                        auto input_reg =
                            *(const fp16x8_t*)(&input_lds[toc * INPUT_LDS_BUFFER_SIZE_C8 +
                                                          input_lds_offsets[S]]);

                        static_for<cfg.kh>(
                            [&]<int R>()
                            {
                                constexpr int p_idx = (Y_LOCAL - R + cfg.kh) % cfg.kh;
                                if constexpr(cfg.direction == Direction::Dgrad)
                                {
                                    int w_idx  = (cfg.kh - 1 - R) * cfg.kw + (cfg.kw - 1 - S);
                                    acc[p_idx] = __builtin_amdgcn_mfma_f32_16x16x32_f16(
                                        weights_reg[w_idx], input_reg, acc[p_idx], 0, 0, 0);
                                }
                                else
                                {
                                    int w_idx  = R * cfg.kw + S;
                                    acc[p_idx] = __builtin_amdgcn_mfma_f32_16x16x32_f16(
                                        weights_reg[w_idx], input_reg, acc[p_idx], 0, 0, 0);
                                }
                            });
                    });

                tic ^= 1;
                toc ^= 1;

                constexpr int P_FLUSH = (Y_LOCAL + 1) % cfg.kh;
                int p_out             = y + py - (cfg.kh - 1);
                if(p_out >= 0 && p_out < ho)
                {
                    // Convert and store this wave's half of output channels
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

                if((y + 1) < hi)
                {
                    auto row_offset = (y + 1) * wi_stride * sizeof(uint4);
                    if(load_active_0)
                    {
                        __builtin_amdgcn_raw_ptr_buffer_load_lds(input_rsrc,
                                                                 store_input_lds_0 +
                                                                     tic * INPUT_LDS_BUFFER_SIZE_C8,
                                                                 16,
                                                                 input_voffset_0 + row_offset,
                                                                 0,
                                                                 0,
                                                                 0);
                    }
                }

                static_for<cfg.kw>(
                    [&]<int S>()
                    {
                        auto input_reg =
                            *(const fp16x8_t*)(&input_lds[toc * INPUT_LDS_BUFFER_SIZE_C8 +
                                                          input_lds_offsets[S]]);

                        static_for<cfg.kh>(
                            [&]<int R>()
                            {
                                constexpr int p_idx = (Y_LOCAL - R + cfg.kh) % cfg.kh;
                                if constexpr(cfg.direction == Direction::Dgrad)
                                {
                                    int w_idx  = (cfg.kh - 1 - R) * cfg.kw + (cfg.kw - 1 - S);
                                    acc[p_idx] = __builtin_amdgcn_mfma_f32_16x16x32_f16(
                                        weights_reg[w_idx], input_reg, acc[p_idx], 0, 0, 0);
                                }
                                else
                                {
                                    int w_idx  = R * cfg.kw + S;
                                    acc[p_idx] = __builtin_amdgcn_mfma_f32_16x16x32_f16(
                                        weights_reg[w_idx], input_reg, acc[p_idx], 0, 0, 0);
                                }
                            });
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

    // Flush output rows whose last input contribution (at filter row kh-1) would land at
    // y >= hi (out of bounds).
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

// Compute grouped convolution with stride 1 and group size 32.
//
// This kernel assumes the number of input and output channels is the same.
//
// Input layout: N hi wi C8
// Weights layout: K kh kw GROUP_SIZE_8
// Output layout N ho wo K8
template <ck_tile::direct_hip_conv::grouped_32c::Config cfg>
__global__ void conv2d_grouped_32c_fp16_nhwc_cdna4(const _Float16* __restrict__ in,
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
    conv2d_grouped_32c_fp16_cdna4_nhwc_impl<cfg>(in,
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

namespace ck_tile::direct_hip_conv::grouped_32c
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
        conv2d_grouped_32c_fp16_nhwc_cdna4<configs[I]>
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
            if(par.channels_per_group() != 32)
                return false;
            if(par.c_tot % 32 != 0)
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

} // namespace ck_tile::direct_hip_conv::grouped_32c
