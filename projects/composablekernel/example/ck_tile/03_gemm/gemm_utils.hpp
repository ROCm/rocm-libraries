// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/utility/json_dump.hpp"
#include "gemm_common/vector_size_fallback_dispatch.hpp"

#include <string>
#include <variant>

using ck_tile_example::GemmConfigVectorSizeFallback;

// Max. vectorized global memory access in bytes.
static constexpr ck_tile::index_t kMaxVectorBytes = 16;

// Number of elements of type T in 1 max-width access.
template <typename T>
static constexpr ck_tile::index_t kMaxVectorElems =
    static_cast<ck_tile::index_t>(kMaxVectorBytes / sizeof(T));

struct GemmConfigBase
{
    static constexpr bool kPadM = false;
    static constexpr bool kPadN = false;
    static constexpr bool kPadK = false;

    static constexpr bool PermuteA = false;
    static constexpr bool PermuteB = false;

    static constexpr bool TransposeC            = false;
    static constexpr bool UseStructuredSparsity = false;

    static constexpr int kBlockPerCu                         = 1;
    static constexpr ck_tile::index_t TileParitionerGroupNum = 8;
    static constexpr ck_tile::index_t TileParitionerM01      = 4;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Intrawave;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V3;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
    static constexpr bool Preshuffle                = false;
    static constexpr bool TiledMMAPermuteN          = false;

    static constexpr ck_tile::index_t kClusterSizeM       = 1;
    static constexpr ck_tile::index_t kClusterSizeN       = 1;
    static constexpr ck_tile::index_t BlockedXDLN_PerWarp = 1;
    static constexpr ck_tile::DataCachePrefetchKind DataCachePrefetchA =
        ck_tile::DataCachePrefetchKind::None;
    static constexpr ck_tile::DataCachePrefetchKind DataCachePrefetchB =
        ck_tile::DataCachePrefetchKind::None;
    static constexpr bool Async = false;
    // COMPUTE_TDM_V1 only: use CShuffleEpilogue instead of TdmEpilogue. Every
    // other pipeline except COMPUTE_TDM_V2 (TdmEpilogue only) already uses
    // CShuffleEpilogue.
    static constexpr bool UseCShuffleEpilogue = false;

    static constexpr bool FixedVectorSize = false;
    // If FixedVectorSize==true: use these vector sizes for A/B loads and C store
    static constexpr ck_tile::index_t VectorSizeA = 1;
    static constexpr ck_tile::index_t VectorSizeB = 1;
    static constexpr ck_tile::index_t VectorSizeC = 1;

    static constexpr bool EnableSmallerVectorLoadFallback = false;
    static constexpr bool LargeTensors                    = false;
};

// A,B vector sizes must divide K_Warp_Tile. Used directly by test_gemm_unaligned_k.cpp; the
// general run_gemm_example.inc dispatch path uses GemmConfigVectorSizeFallback instead.
template <typename GemmConfig,
          ck_tile::index_t VectorSizeA_,
          ck_tile::index_t VectorSizeB_,
          ck_tile::index_t VectorSizeC_>
struct GemmConfigFixedVectorSize : public GemmConfig
{
    // Split-K partitions are aligned to K_Warp_Tile: preserve split-K remainder
    static_assert(GemmConfig::K_Warp_Tile % VectorSizeA_ == 0 &&
                      GemmConfig::K_Warp_Tile % VectorSizeB_ == 0,
                  "A/B vector width must divide K_Warp_Tile");

    // Enable K padding
    static constexpr bool kPadK = true;

    static constexpr bool FixedVectorSize         = true;
    static constexpr ck_tile::index_t VectorSizeA = VectorSizeA_;
    static constexpr ck_tile::index_t VectorSizeB = VectorSizeB_;
    static constexpr ck_tile::index_t VectorSizeC = VectorSizeC_;
};

template <typename PrecType>
struct GemmConfigMemoryInterwave : public GemmConfigBase
{
    // Memory friendly for Interwave scheduler
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 32;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 1;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = sizeof(PrecType) == 2 ? 8 : 16;

    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::MEMORY;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Interwave;
};

template <typename PrecType>
struct GemmConfigMemoryIntrawave : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 32;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 1;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = sizeof(PrecType) == 2 ? 8 : 16;

    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::MEMORY;
};

template <typename PrecType>
struct GemmConfigComputeV3 : public GemmConfigBase
{
    // Compute V3 only support Intrawave scheduler
    static constexpr ck_tile::index_t M_Tile = 16;
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 256 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile =
        ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile>();

    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V3;
};

template <typename PrecType>
struct GemmConfigComputeV3_1 : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 256;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile =
        ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile>();

    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V3;
};

template <typename PrecType>
struct GemmConfigComputeV3_2 : public GemmConfigBase
{
    static constexpr bool kPadM = true;
    static constexpr bool kPadN = true;
    static constexpr bool kPadK = true;

    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile =
        ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile>();

    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V3;

    static constexpr int kBlockPerCu = 2;
    static constexpr bool EnableSmallerVectorLoadFallback =
        !std::is_same_v<PrecType, ck_tile::int8_t>;
    static constexpr bool LargeTensors = !std::is_same_v<PrecType, ck_tile::int8_t>;
};

template <typename PrecType>
struct GemmConfigComputeV3_3 : public GemmConfigBase
{
    static constexpr bool kPadM = true;
    static constexpr bool kPadN = true;
    static constexpr bool kPadK = true;

    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 256 / sizeof(PrecType);
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile =
        ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile>();

    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V3;

    static constexpr int kBlockPerCu = 2;
    static constexpr bool Async      = true;
};

template <typename PrecType>
struct GemmConfigComputeV3_WMMA : public GemmConfigBase
{
    static constexpr bool kPadM = true;
    static constexpr bool kPadN = true;
    static constexpr bool kPadK = true;

    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 64 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile =
        ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile>();

    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V3;

    static constexpr int kBlockPerCu = 2;

    static constexpr bool EnableSmallerVectorLoadFallback = true;
    static constexpr bool LargeTensors = !std::is_same_v<PrecType, ck_tile::int8_t>;
};

template <typename PrecType>
struct GemmConfigComputeV3_WMMA_ClusterLaunch : public GemmConfigComputeV3_WMMA<PrecType>
{
    static constexpr ck_tile::index_t kClusterSizeM = 2;
    static constexpr ck_tile::index_t kClusterSizeN = 2;

    // Vector-size fallback not validated on the cluster-launch path yet, disable for now.
    static constexpr bool EnableSmallerVectorLoadFallback = false;
};

template <typename PrecType>
struct GemmConfigComputeV4 : public GemmConfigBase
{
    // Compute V4 only support Intrawave scheduler
    // Using the ping pong reader in the lds level
    static constexpr ck_tile::index_t M_Tile = 256;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 64 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile =
        ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile>();

    static constexpr bool DoubleSmemBuffer          = true;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V4;
};

template <typename PrecType>
struct GemmConfigComputeV4_1 : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 256;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile =
        ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile>();

    static constexpr bool DoubleSmemBuffer          = true;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V4;
};

template <typename PrecType>
struct GemmConfigComputeV5 : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 64 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 1;
    static constexpr ck_tile::index_t K_Warp = 2;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile =
        ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile>();

    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V5;
    static constexpr ck_tile::index_t NumWaveGroups = 2;
};

template <typename PrecType>
struct GemmConfigComputeV6 : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 256;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 32;

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V6;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
};

template <typename PrecType>
struct GemmConfigComputeAsync : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 64;
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 256;

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 128;

    static constexpr bool DoubleSmemBuffer          = true;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_ASYNC;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
    static constexpr bool UseStructuredSparsity     = false;
};

// gfx1250 bf16 compute config for rocKE "Opt-4" class shapes: 256x128x64 tile, 4x2 waves,
// 16x16x32 WMMA, no padding (M/N/K must be multiples of the tile), CShuffle epilogue. With a
// row-major C and a square warp tile, TransposeC turns the per-lane CShuffle LDS writes from 2-byte
// into 16-byte stores (same rule as test/ck_tile/gemm/test_gemm_pipeline_util.hpp).
template <typename CLayout = ck_tile::tensor_layout::gemm::RowMajor>
struct GemmConfigGfx1250Bf16_Opt4 : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 256;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 32;

#if defined(CK_USE_GFX1250)
    static constexpr bool TransposeC =
        std::is_same_v<CLayout, ck_tile::tensor_layout::gemm::RowMajor> &&
        M_Warp_Tile == N_Warp_Tile;
#else
    static constexpr bool TransposeC = false;
#endif

    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V3;
};

template <typename PrecType>
struct GemmConfigPreshuffleDecode : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 16;
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 256 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile =
        ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile, true>();

    static constexpr int kBlockPerCu                = 1;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Default;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::PRESHUFFLE_V2;
    static constexpr bool Preshuffle                = true;
    static constexpr bool DoubleSmemBuffer          = true;
    static constexpr int N_Repeat                   = N_Tile / N_Warp_Tile / N_Warp;
    static constexpr bool TiledMMAPermuteN          = N_Repeat % 2 == 0;
};

template <typename PrecType>
struct GemmConfigPreshufflePrefill : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile =
        ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile, true>();

    static constexpr int kBlockPerCu                = 2;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Default;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::PRESHUFFLE_V2;
    static constexpr bool Preshuffle                = true;
    static constexpr bool DoubleSmemBuffer          = true;
    static constexpr int N_Repeat                   = N_Tile / N_Warp_Tile / N_Warp;
    static constexpr bool TiledMMAPermuteN          = N_Repeat % 2 == 0;

    static constexpr bool Async = false;
};

template <typename PrecType>
struct GemmConfigPreshufflePrefillAsync : public GemmConfigPreshufflePrefill<PrecType>
{
    static constexpr ck_tile::index_t N_Tile = 256;

    // N_Repeat is even in this config
    static constexpr bool TiledMMAPermuteN = true;

    static constexpr bool Async = true;
};

template <typename PrecType>
struct GemmConfigPreshufflePrefill_Wmma : public GemmConfigPreshufflePrefill<PrecType>
{
    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile =
        ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile, true>();
};

template <typename PrecType>
struct GemmConfigMixedPrec_Wmma : public GemmConfigComputeV3_WMMA<PrecType>
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 128;
};

template <typename ADataType, typename BDataType = ADataType, typename CDataType = ADataType>
struct GemmTypeConfig;

template <>
struct GemmTypeConfig<ck_tile::tf32_t, ck_tile::tf32_t, float>
{
    using ADataType   = ck_tile::tf32_t;
    using BDataType   = ck_tile::tf32_t;
    using AccDataType = float;
    using CDataType   = float;
};

template <>
struct GemmTypeConfig<ck_tile::fp32_t>
{
    using ADataType   = ck_tile::fp32_t;
    using BDataType   = ck_tile::fp32_t;
    using AccDataType = ck_tile::fp32_t;
    using CDataType   = ck_tile::fp32_t;
};

template <>
struct GemmTypeConfig<ck_tile::half_t>
{
    using ADataType   = ck_tile::half_t;
    using BDataType   = ck_tile::half_t;
    using AccDataType = float;
    using CDataType   = ck_tile::half_t;
    // ToDo: Add more bias config to support different categories of GEMM.
};

template <>
struct GemmTypeConfig<ck_tile::bf16_t, ck_tile::bf16_t, ck_tile::bf16_t>
{
    using ADataType   = ck_tile::bf16_t;
    using BDataType   = ck_tile::bf16_t;
    using AccDataType = float;
    using CDataType   = ck_tile::bf16_t;
};

template <>
struct GemmTypeConfig<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t>
{
    using ADataType   = ck_tile::fp8_t;
    using BDataType   = ck_tile::fp8_t;
    using AccDataType = float;
    using CDataType   = ck_tile::half_t;
};

template <>
struct GemmTypeConfig<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t>
{
    using ADataType   = ck_tile::bf8_t;
    using BDataType   = ck_tile::bf8_t;
    using AccDataType = float;
    using CDataType   = ck_tile::half_t;
};

template <>
struct GemmTypeConfig<ck_tile::fp8_t, ck_tile::pk_int4_t, ck_tile::half_t>
{
    using ADataType   = ck_tile::fp8_t;
    using BDataType   = ck_tile::pk_int4_t;
    using AccDataType = float;
    using CDataType   = ck_tile::half_t;
};

template <>
struct GemmTypeConfig<ck_tile::bf8_t, ck_tile::pk_int4_t, ck_tile::half_t>
{
    using ADataType   = ck_tile::bf8_t;
    using BDataType   = ck_tile::pk_int4_t;
    using AccDataType = float;
    using CDataType   = ck_tile::half_t;
};

template <>
struct GemmTypeConfig<ck_tile::half_t, ck_tile::pk_int4_t, ck_tile::half_t>
{
    using ADataType   = ck_tile::half_t;
    using BDataType   = ck_tile::pk_int4_t;
    using AccDataType = float;
    using CDataType   = ck_tile::half_t;
};

template <>
struct GemmTypeConfig<ck_tile::int8_t, ck_tile::int8_t, int32_t>
{
    using ADataType   = ck_tile::int8_t;
    using BDataType   = ck_tile::int8_t;
    using AccDataType = int32_t;
    using CDataType   = int32_t;
};

template <>
struct GemmTypeConfig<ck_tile::pk_fp4_t, ck_tile::pk_fp4_t, ck_tile::half_t>
{
    using ADataType   = ck_tile::pk_fp4_t;
    using BDataType   = ck_tile::pk_fp4_t;
    using AccDataType = float;
    using CDataType   = ck_tile::half_t;
};

template <>
struct GemmTypeConfig<ck_tile::fp8_t, ck_tile::pk_fp4_t, ck_tile::half_t>
{
    using ADataType   = ck_tile::fp8_t;
    using BDataType   = ck_tile::pk_fp4_t;
    using AccDataType = float;
    using CDataType   = ck_tile::half_t;
};

template <>
struct GemmTypeConfig<ck_tile::bf8_t, ck_tile::pk_fp4_t, ck_tile::half_t>
{
    using ADataType   = ck_tile::bf8_t;
    using BDataType   = ck_tile::pk_fp4_t;
    using AccDataType = float;
    using CDataType   = ck_tile::half_t;
};

template <ck_tile::GemmPipeline PipelineId>
struct PipelineTypeTraits;

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::MEMORY>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrMem<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::BASIC_V1>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAGmemBGmemCRegV1<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::BASIC_V2>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::COMPUTE_V3>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::COMPUTE_V4>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV4<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::COMPUTE_V5>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV5<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::COMPUTE_V6>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV6<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::COMPUTE_ASYNC>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompAsync<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::PRESHUFFLE_V2>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::WeightPreshufflePipelineAGmemBGmemCRegV2<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::COMPUTE_ASYNC_V2>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompAsyncV2<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::COMPUTE_ASYNC_PP>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompAsyncPP<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::COMPUTE_TDM_V1>
{
    template <typename PipelineProblem>
    using GemmPipeline =
        ck_tile::GemmPipelineAgBgCrCompTDMV1<PipelineProblem,
                                             ck_tile::GemmPipelineAgBgCrCompTDMDefaultPolicy<
                                                 false,
                                                 PipelineProblem::Traits::DataCachePrefetchA,
                                                 PipelineProblem::Traits::DataCachePrefetchB>>;
};

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::COMPUTE_TDM_V2>
{
    template <typename PipelineProblem>
    using GemmPipeline =
        ck_tile::GemmPipelineAgBgCrCompTDMV2<PipelineProblem,
                                             ck_tile::GemmPipelineAgBgCrCompTDMDefaultPolicy<
                                                 true,
                                                 PipelineProblem::Traits::DataCachePrefetchA,
                                                 PipelineProblem::Traits::DataCachePrefetchB>>;
};

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::COMPUTE_TDM_RING>
{
    // Default ring: depth 3, every wave issues, wait-then-fill.
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompTDMRing<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::PRESHUFFLE_TDM>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::WeightPreshufflePipelineAGmemBGmemCRegTDM<
        PipelineProblem,
        ck_tile::UniversalWeightPreshufflePipelineAgBgCrTDMPolicy<
            PipelineProblem::Traits::DataCachePrefetchA,
            PipelineProblem::Traits::DataCachePrefetchB>>;
};

template <ck_tile::GemmPipeline PipelineId, typename Problem, bool UseCShuffle = false>
struct EpilogueTypeTraits
{
    using Epilogue = ck_tile::CShuffleEpilogue<Problem>;
};

// TDM V1 defaults to TdmEpilogue; UseCShuffle selects CShuffleEpilogue, whose entry
// s_wait_tensorcnt barrier retires the pipeline's TDM loads before the shuffle reuses LDS.
template <typename Problem>
struct EpilogueTypeTraits<ck_tile::GemmPipeline::COMPUTE_TDM_V1, Problem, false>
{
    using Epilogue = ck_tile::TdmEpilogue<Problem>;
};

template <typename Problem, bool UseCShuffle>
struct EpilogueTypeTraits<ck_tile::GemmPipeline::COMPUTE_TDM_V2, Problem, UseCShuffle>
{
    static_assert(!UseCShuffle, "COMPUTE_TDM_V2 supports only TdmEpilogue");
    using Epilogue = ck_tile::TdmEpilogue<Problem>;
};

template <typename Problem>
struct EpilogueTypeTraits<ck_tile::GemmPipeline::COMPUTE_TDM_RING, Problem>
{
    using Epilogue = ck_tile::TdmEpilogue<Problem>;
};

// Grid request of a persistent launch (-persistent_ctas): ctas CTAs, or the occupancy-derived
// Kernel::MaxOccupancyGridSize when ctas is 0. The occupancy query is not reliable on every target
// (gfx1250 workgroup-processor mode), so the grid is also a tuning axis.
struct PersistentGrid
{
    ck_tile::index_t ctas = 0;
};

// Reads -persistent_ctas; negative values are rejected and a value without -persistent=1 is
// ignored with a warning.
inline PersistentGrid get_persistent_grid(const ck_tile::ArgParser& arg_parser, bool persistent)
{
    const ck_tile::index_t ctas = arg_parser.get_int("persistent_ctas");
    if(ctas < 0)
    {
        throw std::runtime_error("persistent_ctas must be >= 0");
    }
    if(ctas > 0 && !persistent)
    {
        std::cout << "WARNING: Ignoring persistent_ctas for the non-persistent kernel."
                  << std::endl;
    }
    return PersistentGrid{ctas};
}

// Grid of a GemmKernel launch. A pipeline may decline the persistent request
// (Kernel::UniversalGemmKernel::PersistentKernel is false, e.g. COMPUTE_TDM_V2 or a TDM V1 cluster
// launch); its kernel computes one tile per workgroup, so it gets the full GridSize: a persistent
// grid would leave every tile beyond it uncomputed.
template <typename Kernel, bool Persistent, typename HostArgs>
dim3 gemm_grid_size(const HostArgs& args,
                    const ck_tile::stream_config& s,
                    const PersistentGrid& persistent_grid)
{
    if constexpr(Persistent && Kernel::UniversalGemmKernel::PersistentKernel)
    {
        return persistent_grid.ctas > 0 ? dim3(persistent_grid.ctas, 1, 1)
                                        : Kernel::MaxOccupancyGridSize(s);
    }
    else
    {
        if constexpr(Persistent)
        {
            std::cout << "WARNING: The pipeline has no persistent kernel; launching one "
                         "workgroup per tile."
                      << std::endl;
        }
        ck_tile::ignore = s;
        ck_tile::ignore = persistent_grid;
        return Kernel::GridSize(args.M, args.N, args.k_batch);
    }
}

inline auto create_args(const std::string& default_prec = "fp16")
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "3840", "m dimension")
        .insert("n", "4096", "n dimension")
        .insert("k", "2048", "k dimension")
        .insert("a_layout", "R", "A tensor data layout - Row by default")
        .insert("b_layout", "C", "B tensor data layout - Column by default")
        .insert("c_layout", "R", "C tensor data layout - Row by default")
        .insert("stride_a", "0", "Tensor A stride")
        .insert("stride_b", "0", "Tensor B stride")
        .insert("stride_c", "0", "Tensor C stride")
        .insert("v", "2", "0. No validation, 1. Validation on CPU, 2. Validation on GPU")
        .insert("prec",
                default_prec,
                "data type. fp16/bf16/fp8/bf8/pk_int4_t/tf32 (tf32 only on gfx950)")
        .insert("warmup", "50", "number of iterations before benchmark the kernel")
        .insert("repeat", "100", "number of iterations to benchmark the kernel")
        .insert("timer", "gpu", "gpu:gpu timer, cpu:cpu timer")
        .insert("split_k", "1", "splitK value")
        .insert("init", "0", "0:random, 1:linear, 2:constant(1)")
        .insert("persistent", "0", "0:non-persistent, 1:persistent")
        .insert("persistent_ctas",
                "0",
                "persistent grid size (CTAs); 0: occupancy-derived (MaxOccupancyGridSize)")
        .insert("json", "0", "0: No Json, 1: Dump Results in Json format")
        .insert("jsonfile", "gemm.json", "json file name to dump results")
        .insert("flush_cache", "true", "flush cache before running the kernel, defaults to true")
        .insert("rotating_count", "1000", "rotating count, defaults to 1000")
        .insert("test_async", "0", "0: normal gemm, 1: test async input scheduler");
    return arg_parser;
}

// host API
template <typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename CLayout,
          bool Persistent = false,
          typename CDEElementWise>
float gemm(const ck_tile::GemmHostArgs& args, const ck_tile::stream_config& s);
