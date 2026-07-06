// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/numeric/pk_fp4.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma.hpp"
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

// UsePackedNumAccess is derived centrally in the UnificationDispatcher (see below) and threaded
// down through the selector into the pipeline. When true, operands with NumAccess > 1 use a
// contiguous-K layout (packed reads) instead of the default strided-K layout (interleaved reads).
// This is needed when A and B have different NumAccess values due to different data type sizes with
// load-transpose instructions.
//
// The helper here maps a WGAttrNumAccessEnum to its numeric value and stripts out the PackedFlag
// bits. Packedness is derived in the dispatcher.
template <WGAttrNumAccessEnum AttrNumAccess>
struct get_wgattr_num_access_safe_v
{
    private:
    static constexpr int32_t packed_flag = static_cast<int32_t>(WGAttrNumAccessEnum::PackedFlag);
    static constexpr WGAttrNumAccessEnum base =
        static_cast<WGAttrNumAccessEnum>(static_cast<int32_t>(AttrNumAccess) & ~packed_flag);

    public:
    static constexpr int32_t value = get_wgattr_num_access<base>::value;
};
template <>
struct get_wgattr_num_access_safe_v<WGAttrNumAccessEnum::Default>
{
    static constexpr int32_t value = 1;
};

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
          bool IsScale16                     = false>
struct UnificationDispatcher
{
    static_assert(!IsScale16); // TODO: We can't deal with scale16 yet.

    // TODO: The dispatcher currently determines whether microscaling intrinsics are requested based
    // on the WaveTile sizes and types. This is potentially dangerous and we should add a dedicated
    // parameter instead.
    static constexpr bool IsMxSized = (MPerWave == 16 && NPerWave == 16 && KPerWave == 128) ||
                                      (MPerWave == 32 && NPerWave == 32 && KPerWave == 64);
    static constexpr bool IsMx =
        (IsMxSized && std::is_same_v<AccType, float> && UseStructuredSparsity == false);

    // General checks. Structured sparsity Mma pipeline not adapted to UnificationDispatcher yet
    // since we have no sparse tests or examples in CK Tile.
    static_assert(UseStructuredSparsity == false);

    // Scale checks.
    // TODO: Add the tiny types after those are merged.
    static_assert(!IsMx ||
                  (std::is_same_v<AType, fp8_t> || std::is_same_v<AType, bf8_t> ||
                   std::is_same_v<AType, pk_fp4_t>) ||
                  std::is_same_v<AType, pk_fp6x16_t>);
    static_assert(!IsMx || (std::is_same_v<BType, fp8_t> || std::is_same_v<BType, bf8_t> ||
                            std::is_same_v<BType, pk_fp4_t> || std::is_same_v<BType, pk_fp6x16_t>));

    // Convert WGAttrNumAccessEnums to index_t values. Default value sent to 1 for now, but needs a
    // better implementation TODO.
    static constexpr index_t AttrNumAccessAV = get_wgattr_num_access_safe_v<AttrNumAccessA>::value;
    static constexpr index_t AttrNumAccessBV = get_wgattr_num_access_safe_v<AttrNumAccessB>::value;

    // Derivation of packedness for the layouts When true, operands with NumAccess > 1 use a
    // contiguous K layout (packed) instead of the default, strided K layout (interleaved)
    static constexpr bool HasPackedFlagA =
        (static_cast<int>(AttrNumAccessA) & static_cast<int>(WGAttrNumAccessEnum::PackedFlag)) != 0;
    static constexpr bool HasPackedFlagB =
        (static_cast<int>(AttrNumAccessB) & static_cast<int>(WGAttrNumAccessEnum::PackedFlag)) != 0;
    static constexpr bool UsePackedNumAccess =
        (AttrNumAccessAV != AttrNumAccessBV) || HasPackedFlagA || HasPackedFlagB;

    using Type =
        typename MmaPipelineSelector<IsMx,
                                     AType,
                                     BType,
                                     AccType,
                                     MPerWave,
                                     NPerWave,
                                     KPerWave,
                                     MmaAccumPolicy::ROW_MAJOR, // Always ROW_MAJOR for now, we
                                                                // don't allow MN composition.
                                     TransposeC,
                                     SwizzleFactor,
                                     AttrNumAccessAV,
                                     AttrNumAccessBV,
                                     UsePackedNumAccess>::Type;
};
} // namespace warp_gemm_dispatcher
} // namespace impl
} // namespace ck_tile
#endif // #if USE_NEW_UNIFIED_FRAMEWORK
