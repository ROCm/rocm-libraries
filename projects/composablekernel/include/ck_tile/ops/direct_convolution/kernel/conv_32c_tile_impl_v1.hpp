// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// CK Tile v1 implementation of non-grouped (standard) convolution using
// mfma_f32_16x16x32_f16 with C-reduction loop.
//
// This kernel handles convolutions with G=1 and C,K > 32. Each MFMA reduces
// over 32 input channels; a C-reduction loop accumulates partial products
// across C/32 iterations per filter tap.
//
// Algorithm:
//   - Grid.y maps to K-tiles (ceil(K / block_k_size)), not conv groups
//   - Each wave independently handles 16 output channels (one K-tile)
//   - Input LDS: [BLOCK_W, BLOCK_C] loaded per c_block (reuses grouped InputLoader)
//   - Weight LDS: [block_k_size, KH*KW, 32] reloaded per c_local from KYXC DRAM
//   - Separate weight and input LDS regions (coexisting, not unified)
//   - Circular accumulator buffer for streaming output rows
//
// Supported: fp16 and bf16, Fprop and Dgrad.

#pragma once

#include "ck_tile/ops/direct_convolution/kernel/grouped_conv_kernel_base.hpp"
#include "ck_tile/ops/direct_convolution/kernel/grouped_conv_input_loader.hpp"
#include "ck_tile/ops/direct_convolution/kernel/grouped_conv_output_writer.hpp"
#include "ck_tile/ops/direct_convolution/kernel/non_grouped_conv_compute_loop.hpp"
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

namespace ck_tile::direct_conv::conv_32c_tile::v1
{

constexpr int WAVE_SIZE = 64;
constexpr int BLOCK_Q = 16;

// ===================================================================
// Config — kernel configuration for non-grouped conv.
//
// Reuses the same MFMA tile structure as grouped 32c:
//   block_groups = waves_per_wg / 2 (for weight/input LDS sizing)
//   group_size   = 32 (MFMA K = C-reduction dimension)
//   block_c()    = block_groups * 32 = waves_per_wg * 16
//
// New members:
//   block_k_size()   = waves_per_wg * 16 (total K channels per workgroup)
//   c_local_count()  = block_groups (C-sections per c_block)
// ===================================================================
template <DataType DT = DataType::fp16>
struct Config
{
    static constexpr DataType data_type = DT;
    int waves_per_wg;
    int kh = 3;
    int kw = 3;
    int n_fold = 8;
    int channels_per_group = 32;

    constexpr int group_size() const { return channels_per_group; }
    constexpr int waves_per_group() const { return 2; }
    constexpr int block_groups() const { return waves_per_wg / waves_per_group(); }

    constexpr int num_waves() const { return waves_per_wg; }
    constexpr int block_c() const { return channels_per_group * block_groups(); }
    constexpr int block_q() const { return BLOCK_Q; }
    constexpr int block_size() const { return waves_per_wg * WAVE_SIZE; }

    // K channels per workgroup = waves * 16 (each wave handles 16 K-channels).
    constexpr int block_k_size() const { return waves_per_wg * 16; }

    // C-sections per c_block: how many 32-channel MFMA reductions fit in one input load.
    constexpr int c_local_count() const { return block_groups(); }

    Direction direction = Direction::Fprop;
    SwizzleType swizzle_type = SwizzleType::None;
    EpilogueType epilogue = EpilogueType::RegistersToGlobalMemory;
    int vector_size = 8;

    std::string GetName() const
    {
        std::string base = "v1_conv_32c_waves_per_wg_" + std::to_string(waves_per_wg);
        if(epilogue == EpilogueType::RegistersToGlobalMemory)
            return base + "_skip_lds_epilogue";
        else
            return base + "_lds_epilogue";
    }
};

// ===================================================================
// KernelConfigurations — all instantiated configs.
// ===================================================================
template <DataType DT = DataType::fp16>
struct KernelConfigurations
{
static constexpr Config<DT> configs[] = {
    // Dgrad, direct DRAM epilogue
    {.waves_per_wg = 8, .direction = Direction::Dgrad},
    {.waves_per_wg = 4, .direction = Direction::Dgrad},
    {.waves_per_wg = 2, .direction = Direction::Dgrad},
    // Fprop, direct DRAM epilogue
    {.waves_per_wg = 8},
    {.waves_per_wg = 4},
    {.waves_per_wg = 2},
    // Dgrad, LDS-staged epilogue
    {.waves_per_wg = 8, .direction = Direction::Dgrad,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 4, .direction = Direction::Dgrad,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 2, .direction = Direction::Dgrad,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    // Fprop, LDS-staged epilogue
    {.waves_per_wg = 8,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 4,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 2,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
};
static constexpr int NUM_CONFIGS = sizeof(configs) / sizeof(configs[0]);
};

// ===================================================================
// is_valid_config — config compatibility check for non-grouped conv.
// ===================================================================
template <DataType DT = DataType::fp16>
inline bool is_valid_config(const Conv2dParams& par, const Config<DT>& cfg)
{
    if(par.direction != cfg.direction)
        return false;
    // C must be divisible by the c_block size (block_groups * 32).
    if(par.channels_per_group() % (cfg.block_groups() * 32) != 0)
        return false;
    // K must be divisible by block_k_size (waves * 16).
    if(par.filters_per_group() % cfg.block_k_size() != 0)
        return false;
    return true;
}

template <DataType DT = DataType::fp16>
inline LaunchParams get_launch_params(int config_idx, const Conv2dParams& par)
{
    return get_launch_params_non_grouped(KernelConfigurations<DT>::configs[config_idx], par);
}

// ===================================================================
// TileConstants — reuses grouped 32c TileConstantsBase with the same
// MFMA/Weight/Output distributions.
// ===================================================================
template <auto cfg>
struct TileConstants : direct_conv::TileConstantsBase<cfg>
{
    using Base = direct_conv::TileConstantsBase<cfg>;

    static constexpr int WAVES_PER_WG = cfg.waves_per_wg;
    static constexpr int KH_KW_       = cfg.kh * cfg.kw;

    // Weight LDS sizing for one c_slice: [block_k_size, KH_KW, 32].
    // This equals the grouped 32c weight LDS size (block_groups * 32 * KH_KW * 32 / 8).
    static constexpr int WEIGHT_LDS_ELEMENTS = cfg.block_k_size() * cfg.kh * cfg.kw * 32;
    static constexpr int WEIGHT_LDS_SIZE_UINT4 = WEIGHT_LDS_ELEMENTS / 8;

    // MFMA distribution — identical to grouped 32c.
    struct Mfma
    {
        static constexpr auto MakeAccTileDistribution()
        {
            return ck_tile::make_static_tile_distribution(
                ck_tile::tile_distribution_encoding<
                    ck_tile::sequence<>,
                    ck_tile::tuple<ck_tile::sequence<16>,
                                   ck_tile::sequence<WAVES_PER_WG, 4>,
                                   ck_tile::sequence<4>>,
                    ck_tile::tuple<ck_tile::sequence<2>, ck_tile::sequence<2, 1>>,
                    ck_tile::tuple<ck_tile::sequence<0>, ck_tile::sequence<1, 0>>,
                    ck_tile::sequence<3>,
                    ck_tile::sequence<0>>{});
        }
    };

    // Weight LDS read distribution — identical to grouped 32c.
    struct Weight : Base::Weight
    {
        // Override LDS sizing for non-grouped (same formula, but explicit).
        static constexpr int WEIGHT_LDS_SIZE_UINT2 =
            cfg.kh * cfg.kw * cfg.block_groups() * 32 * 8;
        static constexpr int WEIGHT_LDS_SIZE_UINT4_   = WEIGHT_LDS_SIZE_UINT2 / 2;
        static constexpr int NUM_WEIGHT_PASSES_ =
            (WEIGHT_LDS_SIZE_UINT4_ + cfg.block_size() - 1) / cfg.block_size();
        static constexpr int WEIGHT_LDS_PADDED_UINT4_ = NUM_WEIGHT_PASSES_ * cfg.block_size();

        static constexpr auto MakeLdsReadTileDistribution()
        {
            return ck_tile::make_static_tile_distribution(
                ck_tile::tile_distribution_encoding<
                    ck_tile::sequence<>,
                    ck_tile::tuple<ck_tile::sequence<WAVES_PER_WG, 16>,
                                   ck_tile::sequence<KH_KW_>,
                                   ck_tile::sequence<4, 8>>,
                    ck_tile::tuple<ck_tile::sequence<1>, ck_tile::sequence<3, 1>>,
                    ck_tile::tuple<ck_tile::sequence<0>, ck_tile::sequence<0, 1>>,
                    ck_tile::sequence<2, 3>,
                    ck_tile::sequence<0, 1>>{});
        }

        // Dgrad distribution — identical to grouped 32c.
        static constexpr auto MakeLdsReadTileDistributionDgrad()
        {
            using OutputEncode = ck_tile::tile_distribution_encoding<
                ck_tile::sequence<>,
                ck_tile::tuple<ck_tile::sequence<16>,
                               ck_tile::sequence<4, 2, 4>>,
                ck_tile::tuple<ck_tile::sequence<2, 1>>,
                ck_tile::tuple<ck_tile::sequence<0, 0>>,
                ck_tile::sequence<2, 2>,
                ck_tile::sequence<1, 2>>;

            using InputEncode = typename ck_tile::InputTileDistributionTraits<
                OutputEncode, ToType<cfg.data_type>>::TransposedDstrEncode;

            return ck_tile::make_static_tile_distribution(InputEncode{});
        }
    };

    // Output distribution — identical to grouped 32c.
    struct Output : direct_conv::TileConstantsBase<cfg>::Output
    {
        static constexpr auto MakeDramWriteTileDistributionNarrow()
        {
            return ck_tile::make_static_tile_distribution(
                ck_tile::tile_distribution_encoding<
                    ck_tile::sequence<>,
                    ck_tile::tuple<ck_tile::sequence<1>,
                                   ck_tile::sequence<16>,
                                   ck_tile::sequence<WAVES_PER_WG, 4>,
                                   ck_tile::sequence<4>>,
                    ck_tile::tuple<ck_tile::sequence<3>, ck_tile::sequence<3, 2>>,
                    ck_tile::tuple<ck_tile::sequence<0>, ck_tile::sequence<1, 0>>,
                    ck_tile::sequence<1, 4>,
                    ck_tile::sequence<0, 0>>{});
        }
    };
};

// ===================================================================
// BlockCoordsNonGrouped — workgroup-level coordinates for non-grouped conv.
// ===================================================================
template <auto cfg>
using ConvBlockCoordsT = direct_conv::BlockCoordsNonGrouped<cfg>;

// ===================================================================
// ConvInputLoader — extends InputLoader32c for non-grouped conv.
//
// Inherits DRAM→LDS async loads from the shared InputLoader.
// LDS→register reads use wave_group-based offsets (same as grouped 32c).
// The compute loop uses read_from_lds_at_section() with a delta to read
// from any c_local section rather than the wave's own section.
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

    int wave_group_;

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

        const int lane_q  = lane % 16;
        const int lane_c8 = lane / 16;
        wave_group_ = wave / 2;

        // LDS read offsets — same as InputLoader32c.
        // wave_group selects which 32-channel section, lane_c8 selects C8 within it.
        const int c8_pos = wave_group_ * TC::GROUP_SIZE_8 + lane_c8;

        for(int s = 0; s < cfg.kw; s++)
        {
            int spatial_pos = lane_q + s;
            base::mfma_lds_offsets[s] = spatial_pos * TC::BLOCK_C8 * 8 + c8_pos * 8;
        }
    }
};

// ===================================================================
// weight_load_to_lds_kyxc — load one c_slice of KYXC weights to LDS.
//
// Loads weight[block_k_start : +block_k_size, :, c_slice*32 : +32]
// from KYXC DRAM layout into contiguous [block_k_size, KH_KW, 32] LDS.
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
    constexpr int C_SLICE = 32;
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
// Loads weight[k_slice_start : +32, :, block_c_start : +block_c_size]
// from KYXC DRAM layout into contiguous [32_K, KH_KW, block_c_size_C] LDS.
//
// For Dgrad: K dimension = 32 (MFMA reduction), C dimension = block_k_size
// (each wave handles 16 C-channels of the input gradient).
// ===================================================================
template <auto cfg, typename ElementType = _Float16>
__device__ void weight_load_to_lds_kyxc_dgrad(
    uint4* weight_lds,
    const ElementType* __restrict__ wei,
    int k_slice_start,
    int block_c_start,
    int C_total)
{
    constexpr int K_SLICE = 32;
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
// WeightLoader — weight accessor with KYXC loading support.
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
    __device__ void read_from_lds(uint4* weight_lds)
    {
        if constexpr(cfg.direction == Direction::Dgrad)
        {
            // Non-grouped Dgrad: LDS layout is [32_K, KH_KW, block_c_size_C].
            // Each thread reads 8 K values per filter position using strided access.
            //
            // MFMA A operand mapping (mfma_f32_16x16x32_f16):
            //   k_group = lane / 16 → selects which 8 of 32 K-reduction values
            //   m_idx   = lane % 16 → C-output position within wave's 16 channels
            //   c_pos   = wave * 16 + m_idx → absolute C position in this workgroup
            constexpr int KH_KW_L = cfg.kh * cfg.kw;
            constexpr int BLOCK_C = cfg.block_k_size();

            const int lane    = static_cast<int>(threadIdx.x) % WAVE_SIZE;
            const int wave    = static_cast<int>(threadIdx.x) / WAVE_SIZE;
            const int k_group = lane / 16;
            const int c_lane  = lane % 16;
            const int c_pos   = wave * 16 + c_lane;

            const auto* lds_ptr = reinterpret_cast<const ElementType*>(weight_lds);

            using VecType = typename std::remove_reference_t<decltype(*this)>::value_type;

            for(int f = 0; f < KH_KW_L; f++)
            {
                ElementType vals[8];
                for(int j = 0; j < 8; j++)
                {
                    int k = k_group * 8 + j;
                    vals[j] = lds_ptr[k * KH_KW_L * BLOCK_C + f * BLOCK_C + c_pos];
                }
                __builtin_memcpy(&this->weights[f], vals, sizeof(VecType));
            }
        }
        else
        {
            // Fprop: read from LDS using tile distribution (same as grouped 32c).
            constexpr auto weight_lds_read_desc = TC::Weight::MakeLdsReadDescriptor();
            auto weight_lds_view =
                ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
                    reinterpret_cast<ElementType*>(weight_lds), weight_lds_read_desc);

            constexpr auto weight_lds_read_dist = TC::Weight::MakeLdsReadTileDistribution();
            auto weight_lds_read_window = ck_tile::make_tile_window(
                weight_lds_view,
                ck_tile::make_tuple(ck_tile::number<TC::Weight::WEIGHT_LDS_READ_K>{},
                                    ck_tile::number<TC::KH_KW>{},
                                    ck_tile::number<TC::GROUP_SIZE>{}),
                {0, 0, 0},
                weight_lds_read_dist);

            const auto weight_tile = ck_tile::load_tile(weight_lds_read_window);

            using VecType = typename std::remove_reference_t<decltype(*this)>::value_type;
            const auto& vec_buf = weight_tile.get_thread_buffer().template get_as<VecType>();
            static_for<TC::KH_KW>(
                [&]<int khw>()
                {
                    this->weights[khw] = vec_buf[ck_tile::number<khw>{}];
                });
        }
    }
};

// OutputWriter — direct DRAM writes.
template <auto cfg>
using OutputWriter = direct_conv::OutputWriter<TileConstants<cfg>, false, ToType<cfg.data_type>>;

// OutputWriterLds — LDS-staged writes.
template <auto cfg>
using OutputWriterLds = direct_conv::OutputWriterLds<TileConstants<cfg>, false, ToType<cfg.data_type>>;

// The compute loop is defined in a separate header for modularity.
// It is included here so that the kernel entry points can reference it.

// ===================================================================
// Kernel entry points.
// ===================================================================
template <auto cfg>
__device__ void ck_tile_conv2d_32c_nhwc_impl(const ToType<cfg.data_type>* __restrict__ in,
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
    constexpr bool use_lds_epilogue = (cfg.epilogue == EpilogueType::RegistersToLdsToGlobalMemory);
    using TC = TileConstants<cfg>;
    using ElementType = ToType<cfg.data_type>;
    using MfmaFn = std::conditional_t<cfg.data_type == DataType::bf16,
        Mfma16x16x32_bf16, Mfma16x16x32>;
    using OutputWriterType = std::conditional_t<use_lds_epilogue,
        OutputWriterLds<cfg>, OutputWriter<cfg>>;

    conv_compute_loop<
        TC, cfg, MfmaFn,
        ConvBlockCoordsT<cfg>, ConvInputLoader<cfg>, WeightLoader<cfg>, OutputWriterType,
        ElementType>(
        in, wei, out, N, C, K, hi, wi, ho, wo, py, px);
}

template <auto cfg>
__global__ void ck_tile_conv2d_32c_nhwc(const ToType<cfg.data_type>* __restrict__ in,
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
    ck_tile_conv2d_32c_nhwc_impl<cfg>(in, wei, alpha, beta, out,
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
        ck_tile_conv2d_32c_nhwc<KC::configs[I]>
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
            // C and K must be multiples of 32.
            if(par.c_tot % 32 != 0 || par.k_tot % 32 != 0)
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

} // namespace ck_tile::direct_conv::conv_32c_tile::v1
