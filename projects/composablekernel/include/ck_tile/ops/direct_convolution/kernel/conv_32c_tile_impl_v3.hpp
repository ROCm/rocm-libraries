// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// CK Tile v3 implementation of non-grouped (standard) convolution with
// cross-wave LDS reduction. Supports two MFMA shapes:
//
//   mfma_f32_16x16x32: 16 spatial × 16 K-output, 32-ch C-reduction
//   mfma_f32_32x32x16: 32 spatial × 32 K-output, 16-ch C-reduction
//
// Splits the C-reduction across waves within the same workgroup. Each wave
// handles one channels_per_group C-slice, all producing partial sums for
// the same block_k_size K-channels. An LDS-based cross-wave reduction
// combines the partial sums before output.
//
// Design:
//   - waves_per_group = 1 (each wave is its own C-group)
//   - block_k_size = mfma_n (16 or 32 K-channels)
//   - channels_per_group = mfma_k (32 or 16 C-channels per wave)
//   - block_c = waves_per_wg * channels_per_group
//   - No atomics, no serial C-loop per wave
//   - Cross-wave LDS reduction at flush points
//
// Supported: fp16 and bf16, Fprop and Dgrad.

#pragma once

#include "ck_tile/ops/direct_convolution/kernel/grouped_conv_kernel_base.hpp"
#include "ck_tile/ops/direct_convolution/kernel/grouped_conv_input_loader.hpp"
#include "ck_tile/ops/direct_convolution/kernel/grouped_conv_output_writer.hpp"
#include "ck_tile/ops/direct_convolution/kernel/non_grouped_conv_compute_loop_v3.hpp"
#include "ck_tile/ops/direct_convolution/utils/mfma.hpp"
#include "ck_tile/ops/direct_convolution/utils/kernel_variant.hpp"
#include "ck_tile/ops/direct_convolution/utils/memory.hpp"
#include "ck_tile/ops/direct_convolution/utils/detail.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"
#include "ck_tile/core/tensor/load_tile.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <string>

namespace ck_tile::direct_conv::conv_32c_tile::v3
{

constexpr int WAVE_SIZE = 64;

enum class MfmaShape { M16N16K32, M32N32K16 };

// ===================================================================
// weight_load_to_lds_kyxc — load one c_slice of KYXC weights to LDS.
//
// Loads weight[block_k_start : +block_k_size, :, c_slice*cpg : +cpg]
// from KYXC DRAM layout into contiguous [block_k_size, KH_KW, cpg] LDS.
// cpg = channels_per_group (32 for M16N16K32, 16 for M32N32K16).
// ===================================================================
template <auto cfg, typename ElementType = _Float16>
__device__ void weight_load_to_lds_kyxc(
    uint4* weight_lds,
    const ElementType* __restrict__ wei,
    int block_k_start,
    int c_slice,
    int C_total)
{
    constexpr int WEIGHT_K = cfg.block_k_size();
    constexpr int KH_KW = cfg.kh * cfg.kw;
    constexpr int C_SLICE = cfg.channels_per_group();
    constexpr int TOTAL_UINT4 = WEIGHT_K * KH_KW * C_SLICE / 8;
    constexpr int NUM_PASSES = (TOTAL_UINT4 + cfg.block_size() - 1) / cfg.block_size();

    const int tid = static_cast<int>(threadIdx.x);
    const int K_stride = KH_KW * C_total;

    for(int pass = 0; pass < NUM_PASSES; pass++)
    {
        int flat_idx = pass * cfg.block_size() + tid;
        if(flat_idx < TOTAL_UINT4)
        {
            // Decompose: LDS layout is [WEIGHT_K, KH_KW, C_SLICE], C innermost.
            // Each uint4 = 8 fp16 values in the C dimension.
            constexpr int C_UINT4 = C_SLICE / 8;  // 4
            int c8     = flat_idx % C_UINT4;
            int temp   = flat_idx / C_UINT4;
            int filter = temp % KH_KW;
            int k      = temp / KH_KW;

            const ElementType* src = wei
                + static_cast<size_t>(block_k_start + k) * K_stride
                + filter * C_total
                + c_slice * C_SLICE
                + c8 * 8;

            weight_lds[flat_idx] = *reinterpret_cast<const uint4*>(src);
        }
    }
}

// ===================================================================
// weight_load_to_lds_kyxc_dgrad — load one k_slice of KYXC weights
// for Dgrad into LDS.
//
// Loads weight[k_slice_start : +cpg, :, block_c_start : +block_k_size]
// from KYXC DRAM layout into contiguous [cpg_K, KH_KW, block_k_size_C] LDS.
//
// For Dgrad: K dimension = channels_per_group (MFMA reduction),
// C dimension = block_k_size (output channels of input gradient).
// ===================================================================
template <auto cfg, typename ElementType = _Float16>
__device__ void weight_load_to_lds_kyxc_dgrad(
    uint4* weight_lds,
    const ElementType* __restrict__ wei,
    int k_slice_start,
    int block_c_start,
    int C_total)
{
    constexpr int K_SLICE = cfg.channels_per_group();
    constexpr int KH_KW = cfg.kh * cfg.kw;
    constexpr int BLOCK_C = cfg.block_k_size();
    // Total uint4s = K_SLICE * KH_KW * BLOCK_C / 8 = same as Fprop weight LDS size.
    constexpr int TOTAL_UINT4 = K_SLICE * KH_KW * BLOCK_C / 8;
    constexpr int NUM_PASSES = (TOTAL_UINT4 + cfg.block_size() - 1) / cfg.block_size();

    const int tid = static_cast<int>(threadIdx.x);
    const int K_stride = KH_KW * C_total;

    for(int pass = 0; pass < NUM_PASSES; pass++)
    {
        int flat_idx = pass * cfg.block_size() + tid;
        if(flat_idx < TOTAL_UINT4)
        {
            // Decompose: LDS layout is [K_SLICE, KH_KW, BLOCK_C], C innermost.
            constexpr int C_UINT4 = BLOCK_C / 8;
            int c8     = flat_idx % C_UINT4;
            int temp   = flat_idx / C_UINT4;
            int filter = temp % KH_KW;
            int k      = temp / KH_KW;

            const ElementType* src = wei
                + static_cast<size_t>(k_slice_start + k) * K_stride
                + filter * C_total
                + block_c_start
                + c8 * 8;

            weight_lds[flat_idx] = *reinterpret_cast<const uint4*>(src);
        }
    }
}

// ===================================================================
// Config — kernel configuration for v3 cross-wave LDS reduction.
//
// Parameters:
//   mfma_shape: M16N16K32 or M32N32K16
//   waves_per_group() = 1 (each wave is its own C-group)
//   block_groups() = waves_per_wg
//   channels_per_group() = mfma_k (32 or 16)
//   block_c() = waves_per_wg * channels_per_group
//   block_k_size() = mfma_n (16 or 32 K-channels)
//   block_q() = mfma_m (16 or 32 spatial positions)
//   c_local_count() = waves_per_wg (one cpg slice per wave)
// ===================================================================
template <DataType DT = DataType::fp16>
struct Config
{
    static constexpr DataType data_type = DT;
    MfmaShape mfma_shape = MfmaShape::M16N16K32;
    int waves_per_wg;
    int kh = 3;
    int kw = 3;
    int n_fold = 8;

    // Derived from MFMA shape:
    constexpr int mfma_m() const { return (mfma_shape == MfmaShape::M16N16K32) ? 16 : 32; }
    constexpr int mfma_n() const { return (mfma_shape == MfmaShape::M16N16K32) ? 16 : 32; }
    constexpr int mfma_k() const { return (mfma_shape == MfmaShape::M16N16K32) ? 32 : 16; }

    constexpr int channels_per_group() const { return mfma_k(); }
    constexpr int group_size() const { return channels_per_group(); }
    constexpr int waves_per_group() const { return 1; }
    constexpr int block_groups() const { return waves_per_wg; }

    constexpr int num_waves() const { return waves_per_wg; }
    constexpr int block_c() const { return channels_per_group() * block_groups(); }
    constexpr int block_q() const { return mfma_m(); }
    constexpr int block_size() const { return waves_per_wg * WAVE_SIZE; }

    // All waves share the same block_k_size K-channels.
    constexpr int block_k_size() const { return mfma_n(); }

    // C-sections per c_block = waves_per_wg (one per wave).
    constexpr int c_local_count() const { return block_groups(); }

    Direction direction = Direction::Fprop;

    // TODO: Fix the swizzle, non-trivial swizzle is not working.
    SwizzleType swizzle_type = SwizzleType::None;
    EpilogueType epilogue = EpilogueType::RegistersToGlobalMemory;
    int vector_size = 8;

    std::string GetName() const
    {
        std::string mfma_str = (mfma_shape == MfmaShape::M32N32K16) ? "32x32x16" : "16x16x32";
        std::string swizzle_type_str = "_no_swizzle";
        if (swizzle_type == SwizzleType::CyclicShift)
        {
            swizzle_type_str = "_cyclic_shift_swizzle";
        }
        else if (swizzle_type == SwizzleType::XOR)
        {
            swizzle_type_str = "_xor_swizzle";
        }

        std::string base = "mfma_" + mfma_str + "_waves_per_wg_" + std::to_string(waves_per_wg) + swizzle_type_str + "_cross_wave_lds_reduce";

        if (epilogue == EpilogueType::RegistersToGlobalMemory)
        {
            return base + "_direct_dram_epilogue";
        }
        else if (epilogue == EpilogueType::RegistersToLdsToGlobalMemory)
        {
            return base + "_lds_staged_epilogue";
        }

        return base;
    }
};

// ===================================================================
// KernelConfigurations — v3 configs.
// Direct DRAM epilogue only (no LDS epilogue needed since cross-wave
// reduction uses LDS for the reduction itself).
// ===================================================================
template <DataType DT = DataType::fp16>
struct KernelConfigurations
{
static constexpr Config<DT> configs[] = {
    // 16x16x32 configs (channels_per_group=32, block_k_size=16, block_q=16)
    // Dgrad, direct DRAM epilogue
    {.waves_per_wg = 4, .direction = Direction::Dgrad},   // 0
    {.waves_per_wg = 2, .direction = Direction::Dgrad},   // 1
    // Fprop, direct DRAM epilogue
    {.waves_per_wg = 4},                                  // 2
    {.waves_per_wg = 2},                                  // 3

    // 32x32x16 configs (channels_per_group=16, block_k_size=32, block_q=32)
    // Dgrad, direct DRAM epilogue
    {.mfma_shape = MfmaShape::M32N32K16, .waves_per_wg = 4, .direction = Direction::Dgrad}, // 4
    {.mfma_shape = MfmaShape::M32N32K16, .waves_per_wg = 2, .direction = Direction::Dgrad}, // 5
    // Fprop, direct DRAM epilogue
    {.mfma_shape = MfmaShape::M32N32K16, .waves_per_wg = 4},                                // 6
    {.mfma_shape = MfmaShape::M32N32K16, .waves_per_wg = 2},                                // 7
};
static constexpr int NUM_CONFIGS = sizeof(configs) / sizeof(configs[0]);
};

// ===================================================================
// is_valid_config — config compatibility check for v3.
// ===================================================================
template <DataType DT = DataType::fp16>
inline bool is_valid_config(const Conv2dParams& par, const Config<DT>& cfg)
{
    if(par.direction != cfg.direction)
        return false;
    // C_in must equal block_c (= waves_per_wg * channels_per_group).
    const int C_in = (cfg.direction == Direction::Dgrad) ? par.k_tot : par.c_tot;
    if(C_in != cfg.block_c())
        return false;
    // K_out must be divisible by block_k_size.
    const int K_out = (cfg.direction == Direction::Dgrad) ? par.c_tot : par.k_tot;
    if(K_out % cfg.block_k_size() != 0)
        return false;
    return true;
}

template <DataType DT = DataType::fp16>
inline LaunchParams get_launch_params(int config_idx, const Conv2dParams& par)
{
    return get_launch_params_non_grouped(KernelConfigurations<DT>::configs[config_idx], par);
}

// ===================================================================
// TileConstants — extends TileConstantsBase for v3.
//
// Weight LDS is sized for [block_k_size, KH*KW, cpg] — one c_slice.
// block_k_size × cpg = 512 for both MFMA shapes.
// ===================================================================
template <auto cfg>
struct TileConstants : direct_conv::TileConstantsBase<cfg>
{
    using Base = direct_conv::TileConstantsBase<cfg>;

    static constexpr int WAVES_PER_WG = cfg.waves_per_wg;
    static constexpr int KH_KW_       = cfg.kh * cfg.kw;

    // Weight LDS sizing for one c_slice: [block_k_size, KH_KW, cpg].
    // Both MFMA shapes: block_k_size * cpg = 512, so 512 * 9 = 4,608 elements.
    static constexpr int WEIGHT_LDS_ELEMENTS =
        cfg.block_k_size() * cfg.kh * cfg.kw * cfg.channels_per_group();
    static constexpr int WEIGHT_LDS_SIZE_UINT4 = WEIGHT_LDS_ELEMENTS / 8;

    // Mfma distribution — needed by InputLoader's static type declarations.
    // Not used at runtime (ConvInputLoader passes init_mfma_offsets=false).
    // The distribution must be well-formed for type deduction to compile.
    static constexpr int SPATIAL_LANES = cfg.mfma_m();   // 16 or 32
    static constexpr int K_GROUPS      = 64 / SPATIAL_LANES; // 4 or 2
    struct Mfma
    {
        static constexpr auto MakeAccTileDistribution()
        {
            return ck_tile::make_static_tile_distribution(
                ck_tile::tile_distribution_encoding<
                    ck_tile::sequence<>,
                    ck_tile::tuple<ck_tile::sequence<SPATIAL_LANES>,
                                   ck_tile::sequence<WAVES_PER_WG, K_GROUPS>,
                                   ck_tile::sequence<4>>,
                    ck_tile::tuple<ck_tile::sequence<2>, ck_tile::sequence<2, 1>>,
                    ck_tile::tuple<ck_tile::sequence<0>, ck_tile::sequence<1, 0>>,
                    ck_tile::sequence<3>,
                    ck_tile::sequence<0>>{});
        }
    };

    // Weight — only LDS sizing overrides needed. Weight reads use manual
    // addressing (not tile distributions).
    struct Weight : Base::Weight
    {
    };
};

// ===================================================================
// BlockCoords — reuse non-grouped BlockCoords from v1.
// ===================================================================
template <auto cfg>
using ConvBlockCoordsT = direct_conv::BlockCoordsNonGrouped<cfg>;

// ===================================================================
// ConvInputLoader — extends InputLoader for v3.
//
// Key differences from v1:
//   1. wave_group = wave (not wave / 2): each wave is its own C-group.
//   2. Supplementary overflow loads for spatial positions beyond
//      TOTAL_SPATIAL. The tile distribution covers TOTAL_SPATIAL = 16
//      positions, but BLOCK_W = BLOCK_Q + kw - 1 = 18. The extra
//      2 positions (16-17) are loaded by the first 2*BLOCK_C8 threads.
// ===================================================================
template <auto cfg>
struct ConvInputLoader : direct_conv::InputLoader<TileConstants<cfg>, cfg,
    std::conditional_t<cfg.data_type == DataType::bf16, ck_tile::bf16x8_t, ck_tile::fp16x8_t>,
    false, ToType<cfg.data_type>>
{
    using ElementType = ToType<cfg.data_type>;
    using base = direct_conv::InputLoader<TileConstants<cfg>, cfg,
        std::conditional_t<cfg.data_type == DataType::bf16, ck_tile::bf16x8_t, ck_tile::fp16x8_t>,
        false, ElementType>;
    using TC = TileConstants<cfg>;
    using input_type = typename base::input_type;

    // Number of extra tile positions beyond TOTAL_SPATIAL.
    static constexpr int OVERFLOW_COUNT =
        (TC::BLOCK_W - TC::TOTAL_SPATIAL) * TC::BLOCK_C8;

    // State for overflow loads.
    ck_tile::index_t  overflow_voffset;
    CK_TILE_LDS_ADDR ElementType* overflow_lds_dest;
    ck_tile::index_t  overflow_is_valid;
    bool              overflow_active;

    template <typename BlockCoords_>
    __device__ ConvInputLoader(const BlockCoords_& bc,
                                uint4* input_lds,
                                const ElementType* __restrict__ in,
                                int hi,
                                int wi,
                                int px,
                                int py,
                                int dx,
                                int dy,
                                int sx,
                                int sy)
        : base(bc, input_lds, in, hi, wi, px, py, dx, dy, sx, sy,
               TC::GROUP_SIZE, /*init_mfma_offsets=*/false)
    {
        const int lane = static_cast<int>(threadIdx.x) % WAVE_SIZE;
        const int wave = static_cast<int>(threadIdx.x) / WAVE_SIZE;

        constexpr int MFMA_M = cfg.mfma_m();
        const int lane_q  = lane % MFMA_M;
        const int lane_c8 = lane / MFMA_M;

        // v3: each wave is its own C-group (wave_group = wave, not wave / 2).
        const int wave_group = wave;
        const int c8_pos = wave_group * TC::GROUP_SIZE_8 + lane_c8;

        static_for<cfg.kw>(
            [&]<int S>()
            {
                int spatial_pos = lane_q + S;
                base::mfma_lds_offsets[S] = spatial_pos * TC::BLOCK_C8 * 8 + c8_pos * 8;
            });

        // --- Overflow load setup ---
        // The tile distribution covers TOTAL_SPATIAL spatial positions, but
        // we need BLOCK_W = TOTAL_SPATIAL + (kw-1). The first OVERFLOW_COUNT
        // threads each handle one extra (spatial, c8) tile position.
        const int tid = static_cast<int>(threadIdx.x);
        overflow_active = (tid < OVERFLOW_COUNT);
        if(overflow_active)
        {
            const int ov_spatial = TC::TOTAL_SPATIAL + tid / TC::BLOCK_C8;
            const int ov_c8      = tid % TC::BLOCK_C8;

            // LDS destination for overflow position.
            auto* lds_base = reinterpret_cast<CK_TILE_LDS_ADDR ElementType*>(input_lds);
            overflow_lds_dest = lds_base + ov_spatial * TC::BLOCK_C8 * 8 + ov_c8 * 8;

            // DRAM offset: input_x = block_q + ov_spatial - px (padding offset).
            const int input_x = bc.block_q + ov_spatial - px;
            overflow_is_valid = (input_x >= 0 && input_x < wi) ? 1 : 0;
            overflow_voffset = static_cast<ck_tile::index_t>(
                (input_x * bc.C + ov_c8 * 8) * static_cast<int>(sizeof(ElementType)));
        }
        else
        {
            overflow_voffset = 0;
            overflow_lds_dest = nullptr;
            overflow_is_valid = 0;
        }
    }

    // Override prefetch: base load (16 spatial positions) + overflow load (2 extra).
    __device__ __forceinline__ void prefetch_tile_to_lds(int lds_buffer_index)
    {
        if(base::load_active)
        {
            CK_TILE_LDS_ADDR ElementType* lds_dest =
                base::store_input_lds + lds_buffer_index * TC::INPUT_LDS_BUFFER_SIZE_FP16;

            ck_tile::amd_async_buffer_load<ElementType, 8,
                ck_tile::amd_buffer_coherence_enum::coherence_default, true>(
                lds_dest,
                base::input_rsrc,
                base::input_voffset,
                0,
                ck_tile::number<0>{},
                base::is_valid);
        }

        if(overflow_active)
        {
            CK_TILE_LDS_ADDR ElementType* lds_dest =
                overflow_lds_dest + lds_buffer_index * TC::INPUT_LDS_BUFFER_SIZE_FP16;

            ck_tile::amd_async_buffer_load<ElementType, 8,
                ck_tile::amd_buffer_coherence_enum::coherence_default, true>(
                lds_dest,
                base::input_rsrc,
                overflow_voffset,
                0,
                ck_tile::number<0>{},
                overflow_is_valid);
        }
    }

    // Override fetch: advance both offsets, then prefetch.
    __device__ __forceinline__ void fetch_tile_to_lds(int lds_buffer_index)
    {
        if(base::load_active)
            base::input_voffset += base::row_stride_bytes;
        if(overflow_active)
            overflow_voffset += base::row_stride_bytes;
        prefetch_tile_to_lds(lds_buffer_index);
    }
};

// ===================================================================
// WeightLoader — weight accessor for v3.
//
// Weight LDS layout for one c_slice: [16_K, KH_KW, 32_C].
// All waves read the same weight data from LDS (same 16 K-channels),
// but each wave's MFMA pairs these with a different C-section of input.
//
// The DRAM→LDS load functions are reused from v1 unchanged.
// ===================================================================
template <auto cfg>
struct WeightLoader : direct_conv::WeightAccessor8<cfg.kh, cfg.kw,
    std::conditional_t<cfg.data_type == DataType::bf16, bf16x8_t, fp16x8_t>>
{
    using TC = TileConstants<cfg>;
    using ElementType = ToType<cfg.data_type>;

    // Load one c_slice of weights from KYXC DRAM into weight LDS (Fprop).
    __device__ static void load_kyxc_to_lds(
        uint4* weight_lds,
        const ElementType* __restrict__ wei,
        int block_k_start,
        int c_slice,
        int C_total)
    {
        weight_load_to_lds_kyxc<cfg, ElementType>(
            weight_lds, wei, block_k_start, c_slice, C_total);
    }

    // Load one k_slice of weights from KYXC DRAM into weight LDS (Dgrad).
    __device__ static void load_kyxc_to_lds_dgrad(
        uint4* weight_lds,
        const ElementType* __restrict__ wei,
        int k_slice_start,
        int block_c_start,
        int C_total)
    {
        weight_load_to_lds_kyxc_dgrad<cfg, ElementType>(
            weight_lds, wei, k_slice_start, block_c_start, C_total);
    }

    // Read weights from LDS into registers (this->weights[]).
    //
    // Uses manual addressing for both Fprop and Dgrad. The tile distribution
    // approach from v1 is not applicable because v3's weight LDS only holds
    // [16_K, KH_KW, 32_C] (one c_slice), whereas the tile distribution
    // expects [block_c, KH_KW, 32] which assumes the wave dimension
    // partitions the K dimension.
    //
    // All waves read the same weight data (same 16 K-channels). Each wave
    // pairs this with a different C-section of the input during MFMA.
    __device__ void read_from_lds(uint4* weight_lds)
    {
        constexpr int KH_KW_L = cfg.kh * cfg.kw;
        const int lane = static_cast<int>(threadIdx.x) % WAVE_SIZE;
        const auto* lds_ptr = reinterpret_cast<const ElementType*>(weight_lds);
        using VecType = typename std::remove_reference_t<decltype(*this)>::value_type;

        if constexpr(cfg.direction == Direction::Dgrad)
        {
            // Dgrad: LDS layout is [cpg_K, KH_KW, block_k_size_C].
            // Each thread reads 8 K-reduction values per filter position.
            //
            // MFMA A operand mapping:
            //   k_group = lane / mfma_m → selects which 8 K-reduction values
            //   c_lane  = lane % mfma_m → C-output position
            constexpr int BLOCK_C = cfg.block_k_size();
            constexpr int MFMA_M = cfg.mfma_m();

            const int k_group = lane / MFMA_M;
            const int c_lane  = lane % MFMA_M;

            static_for<KH_KW_L>(
                [&]<int F>()
                {
                    ElementType vals[8];
                    static_for<8>(
                        [&]<int J>()
                        {
                            int k = k_group * 8 + J;
                            vals[J] = lds_ptr[k * KH_KW_L * BLOCK_C + F * BLOCK_C + c_lane];
                        });
                    __builtin_memcpy(&this->weights[F], vals, sizeof(VecType));
                });
        }
        else
        {
            // Fprop: LDS layout is [block_k_size_K, KH_KW, cpg_C], C innermost.
            //
            // MFMA A operand mapping:
            //   k_out = lane % mfma_n → K-output channel
            //   c_grp = lane / mfma_n → C-reduction group (each has 8 values)
            //
            // Each thread reads 8 C values per filter position:
            //   vals[j] = weight[k_out, f, c_grp*8 + j]
            constexpr int C_SLICE = cfg.channels_per_group();
            constexpr int MFMA_N = cfg.mfma_n();

            const int k_out = lane % MFMA_N;
            const int c_grp = lane / MFMA_N;

            static_for<KH_KW_L>(
                [&]<int F>()
                {
                    ElementType vals[8];
                    static_for<8>(
                        [&]<int J>()
                        {
                            vals[J] = lds_ptr[k_out * KH_KW_L * C_SLICE + F * C_SLICE + c_grp * 8 + J];
                        });
                    __builtin_memcpy(&this->weights[F], vals, sizeof(VecType));
                });
        }
    }
};

// ===================================================================
// OutputWriterV3 — manual offset computation for v3.
//
// In v3, all waves share the same block_k_size K-channels. Only wave 0
// writes the output after cross-wave LDS reduction.
//
// M16N16K32: lane % 16 → spatial, lane / 16 → K-group (4 groups × 4 K).
//   Single 8B DRAM write per thread.
//
// M32N32K16: lane % 32 → spatial, lane / 32 → K-block (0 or 1).
//   16 accumulator values map to 4 groups of 4 contiguous K values:
//     acc[0..3]   → K = g*8 + m_block*4 + {0..3} for g=0
//     acc[4..7]   → K = g*8 + m_block*4 + {0..3} for g=1
//     acc[8..11]  → K = g*8 + m_block*4 + {0..3} for g=2
//     acc[12..15] → K = g*8 + m_block*4 + {0..3} for g=3
//   where m_block = lane / 32. Four 8B DRAM writes per thread.
// ===================================================================
template <auto cfg>
struct OutputWriterV3
{
    using ElementType = ToType<cfg.data_type>;
    using AccType = std::conditional_t<
        cfg.mfma_shape == MfmaShape::M16N16K32, fp32x4_t, fp32x16_t>;

    ElementType*      output_base;
    ck_tile::index_t  output_spatial_offset; // q_pos * K (spatial + batch offset)
    ck_tile::index_t  row_stride_elems;
    bool              store_valid;

    // For M16N16K32: single K-offset (4 contiguous K values)
    // For M32N32K16: m_block value for computing 4 K-offsets
    int k_offset_or_m_block;

    template <typename BlockCoords_>
    __device__ OutputWriterV3(const BlockCoords_& bc,
                               uint4*, // Unused, matches OutputWriter signature.
                               ElementType* __restrict__ out,
                               int ho,
                               int wo)
    {
        output_base = out + static_cast<size_t>(bc.block_n) * ho * wo * bc.K + bc.block_k_out;
        row_stride_elems = wo * bc.K;

        const int lane = static_cast<int>(threadIdx.x) % WAVE_SIZE;

        if constexpr(cfg.mfma_shape == MfmaShape::M16N16K32)
        {
            const int q_pos = bc.block_q + lane % 16;
            k_offset_or_m_block = (lane / 16) * 4;
            output_spatial_offset = static_cast<ck_tile::index_t>(q_pos) * bc.K + k_offset_or_m_block;
            store_valid = (q_pos < wo);
        }
        else
        {
            const int q_pos = bc.block_q + lane % 32;
            k_offset_or_m_block = (lane / 32) * 4; // m_block * 4
            output_spatial_offset = static_cast<ck_tile::index_t>(q_pos) * bc.K;
            store_valid = (q_pos < wo);
        }
    }

    __device__ __forceinline__ void flush(AccType acc_val, int p_out)
    {
        if(!store_valid)
            return;

        const ck_tile::index_t row_offset =
            static_cast<ck_tile::index_t>(p_out) * row_stride_elems;

        if constexpr(cfg.mfma_shape == MfmaShape::M16N16K32)
        {
            // Single 8B write: 4 contiguous K values.
            uint32_t words[2];
            words[0] = ConvertFp32ToVec4<ElementType>::convert(acc_val[0], acc_val[1]);
            words[1] = ConvertFp32ToVec4<ElementType>::convert(acc_val[2], acc_val[3]);

            ck_tile::index_t store_offset = output_spatial_offset + row_offset;
            __builtin_memcpy(output_base + store_offset, words, sizeof(words));
        }
        else
        {
            // Four 8B writes: 4 groups of 4 contiguous K values.
            // Group g: K-offset = g*8 + m_block*4, acc values [g*4 .. g*4+3].
            const ck_tile::index_t base_offset = output_spatial_offset + row_offset;

            static_for<4>(
                [&]<int G>()
                {
                    const int k_off = G * 8 + k_offset_or_m_block;
                    uint32_t words[2];
                    words[0] = ConvertFp32ToVec4<ElementType>::convert(
                        acc_val[G * 4 + 0], acc_val[G * 4 + 1]);
                    words[1] = ConvertFp32ToVec4<ElementType>::convert(
                        acc_val[G * 4 + 2], acc_val[G * 4 + 3]);
                    __builtin_memcpy(output_base + base_offset + k_off, words, sizeof(words));
                });
        }
    }
};

// ===================================================================
// Kernel entry points.
// ===================================================================
template <auto cfg>
__device__ void ck_tile_conv2d_32c_nhwc_v3_impl(const ToType<cfg.data_type>* __restrict__ in,
                                                  const ToType<cfg.data_type>* __restrict__ wei,
                                                  double alpha,
                                                  double beta,
                                                  ToType<cfg.data_type>* __restrict__ out,
                                                  int N,
                                                  int C,
                                                  int K,
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
    using TC = TileConstants<cfg>;
    using ElementType = ToType<cfg.data_type>;

    // Select MFMA functor based on shape and data type.
    using MfmaFn = std::conditional_t<
        cfg.mfma_shape == MfmaShape::M32N32K16,
        std::conditional_t<cfg.data_type == DataType::bf16, Mfma32x32x16_bf16, Mfma32x32x16>,
        std::conditional_t<cfg.data_type == DataType::bf16, Mfma16x16x32_bf16, Mfma16x16x32>>;

    conv_compute_loop_v3<
        TC, cfg, MfmaFn,
        ConvBlockCoordsT<cfg>, ConvInputLoader<cfg>, WeightLoader<cfg>,
        OutputWriterV3<cfg>,
        ElementType>(
        in, wei, out, N, C, K, hi, wi, ho, wo, py, px);
}

template <auto cfg>
__global__ void ck_tile_conv2d_32c_nhwc_v3(const ToType<cfg.data_type>* __restrict__ in,
                                             const ToType<cfg.data_type>* __restrict__ wei,
                                             double alpha,
                                             double beta,
                                             ToType<cfg.data_type>* __restrict__ out,
                                             int N,
                                             int C,
                                             int K,
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
    ck_tile_conv2d_32c_nhwc_v3_impl<cfg>(in, wei, alpha, beta, out,
                                          N, C, K, hi, wi, ho, wo, fy, fx, sy, sx, dy, dx, py, px);
}

// ===================================================================
// Launch dispatch.
// ===================================================================
template <DataType DT = DataType::fp16, size_t... Is>
void launch_dispatch(int config_idx,
                     std::index_sequence<Is...>,
                     const LaunchParams& lp,
                     const Conv2dParams& par,
                     const void* in,
                     const void* wei,
                     void* out,
                     hipStream_t stream)
{
    using ElementType = ToType<DT>;
    using KC = KernelConfigurations<DT>;

    auto kernel_launch = [&]<size_t I>()
    {
        auto view = SizeView<KC::configs[I].direction>(par);
        ck_tile_conv2d_32c_nhwc_v3<KC::configs[I]>
            <<<lp.grid, lp.block_size, lp.dynamic_shared_bytes, stream>>>(
                static_cast<const ElementType*>(in),
                static_cast<const ElementType*>(wei),
                1.0,
                0.0,
                static_cast<ElementType*>(out),
                par.n,
                par.c_tot,
                par.k_tot,
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

template <DataType DT = DataType::fp16>
inline void launch(int config_idx,
                   const LaunchParams& lp,
                   const Conv2dParams& par,
                   const void* in,
                   const void* wei,
                   void* out,
                   void* /*workspace*/,
                   hipStream_t stream)
{
    launch_dispatch<DT>(
        config_idx, std::make_index_sequence<KernelConfigurations<DT>::NUM_CONFIGS>{},
        lp, par, in, wei, out, stream);
}

// ===================================================================
// Variant registration.
// ===================================================================
template <DataType DT = DataType::fp16>
constexpr KernelVariant make_variant()
{
    return {
        .is_applicable =
            [](const Conv2dParams& par)
        {
            if(!is_applicable_base(par))
                return false;
            if(par.in_type != DT || par.wei_type != DT || par.out_type != DT)
                return false;
            if(!par.is_non_grouped())
                return false;
            // C and K must be multiples of 16 (block_k_size).
            // The stricter requirement (C_in must be block_c = waves*32) is
            // checked per-config in is_valid_config.
            if(par.c_tot % 16 != 0 || par.k_tot % 16 != 0)
                return false;
            return true;
        },
        .config_is_compatible = [](const Conv2dParams& par, int idx)
        { return is_valid_config<DT>(par, KernelConfigurations<DT>::configs[idx]); },
        .get_launch_params  = &get_launch_params<DT>,
        .launch             = &launch<DT>,
        .get_workspace_size = [](int, const Conv2dParams&) -> size_t { return 0; },
        .num_configs        = KernelConfigurations<DT>::NUM_CONFIGS,
    };
}

} // namespace ck_tile::direct_conv::conv_32c_tile::v3
