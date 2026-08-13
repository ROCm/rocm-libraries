// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/numeric/pk_fp4.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma_impl.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_impl.hpp"
#include "ck_tile/core/arch/mma/scale/scale_mma_pipeline.hpp"
#include "ck_tile/core/arch/mma/mma_wavewise.hpp"

#if USE_NEW_UNIFIED_FRAMEWORK
namespace ck_tile {
namespace impl {
namespace warp_gemm_dispatcher {

using namespace ck_tile::core::arch;
using namespace mma;

// This is a bit awkward but we need to be able to select the appropriate Mma Pipeline (dense,
// sparse, scale) based on some constexpr calculations in the UnificationDispatcher, without
// exposing the wrong path to the compiler, which may end up being ill-formed (if we were to use a
// simple "if constexpr" instead of TMP).
template <bool IsMx,
          typename AType,
          typename BType,
          typename AccType,
          index_t M,
          index_t N,
          index_t K,
          MmaAccumPolicy AccumPolicy,
          bool TransposeC,
          index_t SwizzleFactor,
          index_t AttrNumAccessAV,
          index_t AttrNumAccessBV,
          bool UsePackedNumAccess>
struct MmaPipelineSelector;

template <typename AType,
          typename BType,
          typename AccType,
          index_t M,
          index_t N,
          index_t K,
          MmaAccumPolicy AccumPolicy,
          bool TransposeC,
          index_t SwizzleFactor,
          index_t AttrNumAccessAV,
          index_t AttrNumAccessBV,
          bool UsePackedNumAccess>
struct MmaPipelineSelector<true,
                           AType,
                           BType,
                           AccType,
                           M,
                           N,
                           K,
                           AccumPolicy,
                           TransposeC,
                           SwizzleFactor,
                           AttrNumAccessAV,
                           AttrNumAccessBV,
                           UsePackedNumAccess>
{
    using Type = ScaleMmaPipeline<AType,
                                  BType,
                                  AccType,
                                  M,
                                  N,
                                  K,
                                  AccumPolicy,
                                  TransposeC,
                                  SwizzleFactor,
                                  AttrNumAccessAV,
                                  AttrNumAccessBV,
                                  UsePackedNumAccess>;
};

template <typename AType,
          typename BType,
          typename AccType,
          index_t M,
          index_t N,
          index_t K,
          MmaAccumPolicy AccumPolicy,
          bool TransposeC,
          index_t SwizzleFactor,
          index_t AttrNumAccessAV,
          index_t AttrNumAccessBV,
          bool UsePackedNumAccess>
struct MmaPipelineSelector<false,
                           AType,
                           BType,
                           AccType,
                           M,
                           N,
                           K,
                           AccumPolicy,
                           TransposeC,
                           SwizzleFactor,
                           AttrNumAccessAV,
                           AttrNumAccessBV,
                           UsePackedNumAccess>
{
    using Type = WaveWiseMmaPipeline<AType,
                                     BType,
                                     AccType,
                                     M,
                                     N,
                                     K,
                                     AccumPolicy,
                                     TransposeC,
                                     SwizzleFactor,
                                     AttrNumAccessAV,
                                     AttrNumAccessBV,
                                     UsePackedNumAccess>;
};

// UsePackedNumAccess is threaded through the dispatch chain explicitly from the pipeline level.
// When true, operands with NumAccess > 1 use a contiguous-K layout (packed reads) instead of the
// default strided-K layout (interleaved reads). This is required by some load/transposition
// arrangements, including, but not limited to, mixed operand types with different NumAccess values.
// TODO: normalise NumAccess and packing into 1 unambiguous dispatcher, for clarity.
// TODO: Replace UsePackedNumAccess with independent A/B controls if required (e.g. fp8_t/pk_fp4_t
// 16x16x128 on gfx1250).
template <WGAttrNumAccessEnum AttrNumAccess>
struct get_wgattr_num_access_safe_v
{
    static constexpr int32_t value = get_wgattr_num_access<AttrNumAccess>::value;
};
template <>
struct get_wgattr_num_access_safe_v<WGAttrNumAccessEnum::Default>
{
    static constexpr int32_t value = 1;
};

// Restore legacy behaviour for WGAttrNumAccessEnum::Default.
template <WGAttrNumAccessEnum AttrNumAccess>
struct LegacyWGAttrNumAccess
{
    static constexpr auto value = AttrNumAccess;
};

template <>
struct LegacyWGAttrNumAccess<WGAttrNumAccessEnum::Default>
{
    static constexpr auto value = WGAttrNumAccessEnum::Single;
};

// Restore legacy behaviour for WarpGemmIterateK when FragsK=2.
template <typename AType,
          typename BType,
          typename AccType,
          index_t M,
          index_t N,
          index_t K,
          WGAttrNumAccessEnum AttrNumAccessA,
          WGAttrNumAccessEnum AttrNumAccessB>
struct LegacyWarpGemmIterateK;

// AType=BType=half_t, AccType=float, M=N=16, K=32
template <WGAttrNumAccessEnum AttrNumAccessA, WGAttrNumAccessEnum AttrNumAccessB>
struct LegacyWarpGemmIterateK<half_t, half_t, float, 16, 16, 32, AttrNumAccessA, AttrNumAccessB>
{
    using Type = WarpGemmImpl<WarpGemmAttributeMfmaIterateK<
        WarpGemmAttributeMfmaImplF16F16F32M16N16K16<WGAttrCtlEnum::Default_>,
        2,
        AttrNumAccessA,
        AttrNumAccessB>>;
};

// AType=BType=bf16_t, AccType=float, M=N=16, K=32
template <WGAttrNumAccessEnum AttrNumAccessA, WGAttrNumAccessEnum AttrNumAccessB>
struct LegacyWarpGemmIterateK<bf16_t, bf16_t, float, 16, 16, 32, AttrNumAccessA, AttrNumAccessB>
{
    using Type = WarpGemmImpl<WarpGemmAttributeMfmaIterateK<
        WarpGemmAttributeMfmaImplBf16Bf16F32M16N16K16<WGAttrCtlEnum::Default_>,
        2,
        AttrNumAccessA,
        AttrNumAccessB>>;
};

// AType=BType=fp8_t, AccType=float, M=N=16, K=64
template <WGAttrNumAccessEnum AttrNumAccessA, WGAttrNumAccessEnum AttrNumAccessB>
struct LegacyWarpGemmIterateK<fp8_t, fp8_t, float, 16, 16, 64, AttrNumAccessA, AttrNumAccessB>
{
    using Type = WarpGemmImpl<WarpGemmAttributeMfmaIterateK<
        WarpGemmAttributeMfmaImpl_f32_16x16x32_fp8_fp8<WGAttrCtlEnum::Default_>,
        2,
        AttrNumAccessA,
        AttrNumAccessB>>;
};

// AType=BType=bf8_t, AccType=float, M=N=16, K=64
template <WGAttrNumAccessEnum AttrNumAccessA, WGAttrNumAccessEnum AttrNumAccessB>
struct LegacyWarpGemmIterateK<bf8_t, bf8_t, float, 16, 16, 64, AttrNumAccessA, AttrNumAccessB>
{
    using Type = WarpGemmImpl<WarpGemmAttributeMfmaIterateK<
        WarpGemmAttributeMfmaImpl_f32_16x16x32_bf8_bf8<WGAttrCtlEnum::Default_>,
        2,
        AttrNumAccessA,
        AttrNumAccessB>>;
};

// Restore legacy behaviour for IterateK when FragsK=2.
template <bool UseLegacy,
          typename UnifiedWarpGemm,
          typename AType,
          typename BType,
          typename AccType,
          index_t M,
          index_t N,
          index_t K,
          WGAttrNumAccessEnum AttrNumAccessA,
          WGAttrNumAccessEnum AttrNumAccessB>
struct LegacyIterateKSelector
{
    using Type = UnifiedWarpGemm;
};

template <typename UnifiedWarpGemm,
          typename AType,
          typename BType,
          typename AccType,
          index_t M,
          index_t N,
          index_t K,
          WGAttrNumAccessEnum AttrNumAccessA,
          WGAttrNumAccessEnum AttrNumAccessB>
struct LegacyIterateKSelector<true, // UseLegacy
                              UnifiedWarpGemm,
                              AType,
                              BType,
                              AccType,
                              M,
                              N,
                              K,
                              AttrNumAccessA,
                              AttrNumAccessB>
{
    using Type =
        typename LegacyWarpGemmIterateK<AType,
                                        BType,
                                        AccType,
                                        M,
                                        N,
                                        K,
                                        LegacyWGAttrNumAccess<AttrNumAccessA>::value,
                                        LegacyWGAttrNumAccess<AttrNumAccessB>::value>::Type;
};

// TODO Replace IsScale16 and UseMxScale with a single MmaScaleFamily enum.
// The new enum should distinguish dense, MX scale8 and MX scale16 selection
// while preserving the current type-based inference for unambiguous MX input types.
template <typename AType,
          typename BType,
          typename AccType,
          index_t MPerWave,
          index_t NPerWave,
          index_t KPerWave,
          bool TransposeC,
          index_t SwizzleFactor              = 1,
          bool UseStructuredSparsity         = false,
          WGAttrNumAccessEnum AttrNumAccessA = WGAttrNumAccessEnum::Single,
          WGAttrNumAccessEnum AttrNumAccessB = AttrNumAccessA,
          bool IsScale16                     = false,
          bool UsePackedNumAccess            = false,
          bool UseMxScale                    = false>
struct UnificationDispatcher
{
    static_assert(!IsScale16); // TODO: We can't deal with scale16 yet.

    // pk_fp4_t/pk_fp6x16_t/pk_bf6x16_t are unambiguous (MX, block-scale only).
    // fp8_t/bf8_t are ambiguous (check UseMxScale)
    static constexpr bool HasUnambiguousMxType =
        is_any_of<AType, pk_fp4_t, pk_fp6x16_t, pk_bf6x16_t>::value ||
        is_any_of<BType, pk_fp4_t, pk_fp6x16_t, pk_bf6x16_t>::value;
    static constexpr bool IsMx = UseMxScale || HasUnambiguousMxType;

    static_assert(!IsMx || std::is_same_v<AccType, float>,
                  "MX (block-scaled) MFMA requires a float accumulator");
    static_assert(!IsMx || !UseStructuredSparsity,
                  "MX (block-scaled) MFMA pipeline is not compatible with structured "
                  "sparsity");

    // General checks. Structured sparsity Mma pipeline not adapted to UnificationDispatcher yet
    // since we have no sparse tests or examples in CK Tile.
    static_assert(UseStructuredSparsity == false);

    // Scale checks.
    // TODO: Add the tiny types after those are merged.
    static_assert(!IsMx ||
                  (std::is_same_v<AType, fp8_t> || std::is_same_v<AType, bf8_t> ||
                   std::is_same_v<AType, pk_fp4_t>) ||
                  std::is_same_v<AType, pk_fp6x16_t> || std::is_same_v<AType, pk_bf6x16_t>);
    static_assert(!IsMx || (std::is_same_v<BType, fp8_t> || std::is_same_v<BType, bf8_t> ||
                            std::is_same_v<BType, pk_fp4_t> || std::is_same_v<BType, pk_fp6x16_t> ||
                            std::is_same_v<BType, pk_bf6x16_t>));

    // Convert WGAttrNumAccessEnums to index_t values. Default value sent to 1 for now, but needs a
    // better implementation TODO.
    static constexpr index_t AttrNumAccessAV = get_wgattr_num_access_safe_v<AttrNumAccessA>::value;
    static constexpr index_t AttrNumAccessBV = get_wgattr_num_access_safe_v<AttrNumAccessB>::value;

    using UnifiedWarpGemm = typename MmaPipelineSelector<IsMx,
                                                         AType,
                                                         BType,
                                                         AccType,
                                                         MPerWave,
                                                         NPerWave,
                                                         KPerWave,
                                                         MmaAccumPolicy::ROW_MAJOR,
                                                         TransposeC,
                                                         SwizzleFactor,
                                                         AttrNumAccessAV,
                                                         AttrNumAccessBV,
                                                         UsePackedNumAccess>::Type;

    // Use legacy IterateK for the cases that led to scheduling differences in GPU assembly.
    static constexpr bool UseLegacyIterateK =
        !IsMx && !TransposeC && SwizzleFactor == 1 && !UsePackedNumAccess && MPerWave == 16 &&
        NPerWave == 16 && std::is_same_v<AType, BType> &&
        ((KPerWave == 32 && (std::is_same_v<AType, half_t> || std::is_same_v<AType, bf16_t>)) ||
         (KPerWave == 64 && (std::is_same_v<AType, fp8_t> || std::is_same_v<AType, bf8_t>)));

    using Type = typename LegacyIterateKSelector<UseLegacyIterateK,
                                                 UnifiedWarpGemm,
                                                 AType,
                                                 BType,
                                                 AccType,
                                                 MPerWave,
                                                 NPerWave,
                                                 KPerWave,
                                                 AttrNumAccessA,
                                                 AttrNumAccessB>::Type;
};
} // namespace warp_gemm_dispatcher
} // namespace impl
} // namespace ck_tile
#endif // #if USE_NEW_UNIFIED_FRAMEWORK
