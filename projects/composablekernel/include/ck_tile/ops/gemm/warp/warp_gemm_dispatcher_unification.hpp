// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma.hpp"
#include "ck_tile/core/arch/mma/scale/scale_mma_pipeline.hpp"
#include "ck_tile/core/arch/mma/mma_wavewise.hpp"

// When USE_NEW_UNIFIED_FRAMEWORK is 1, we replace all WarpGemms with MmaPipelines from the new
// unified framework. This means WarpGemmDispatcher will use the UnificationDispatcher instead of
// the regular Dispatcher. Furthermore, named WarpGemms like WarpGemmMfmaF32F32F32M16N16K4 will also
// get rerouted to the UnificationDispatcher. The latter is necessary because some pipelines bypass
// the WarpGemmDispatcher in favor of directly using named WarpGemms.
#define USE_NEW_UNIFIED_FRAMEWORK 1

#if USE_NEW_UNIFIED_FRAMEWORK
namespace ck_tile {
namespace impl {
namespace warp_gemm_dispatcher {

// C++20 using enum
static constexpr auto ESingle = WGAttrNumAccessEnum::Single;
static constexpr auto EDouble = WGAttrNumAccessEnum::Double;
static constexpr auto EQuad   = WGAttrNumAccessEnum::Quad;

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
          index_t AttrNumAccessBV>
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
          index_t AttrNumAccessBV>
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
                           AttrNumAccessBV>
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
                                  AttrNumAccessBV>;
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
          index_t AttrNumAccessBV>
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
                           AttrNumAccessBV>
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
                                     AttrNumAccessBV>;
};

template <typename AType,
          typename BType,
          typename AccType,
          index_t MPerWave,
          index_t NPerWave,
          index_t KPerWave,
          bool TransposeC,
          bool SwizzleA                      = false,
          bool UseStructuredSparsity         = false,
          WGAttrNumAccessEnum AttrNumAccessA = ESingle,
          WGAttrNumAccessEnum AttrNumAccessB = AttrNumAccessA>
struct UnificationDispatcher
{
    // static_assert(0);
    // TODO: The dispatcher currently determines whether microscaling intrinsics are requested based
    // on the WaveTile sizes and types. This is potentially dangerous and we should add a dedicated
    // parameter instead.
    static constexpr bool IsMxSized = (MPerWave == 16 && NPerWave == 16 && KPerWave == 128) ||
                                      (MPerWave == 32 && NPerWave == 32 && KPerWave == 64);
    static constexpr bool IsMx =
        (IsMxSized && std::is_same_v<AccType, float> && UseStructuredSparsity == false);

    // General checks. Swizzle not supported yet. Structured sparsity Mma pipeline not adapted to
    // UnificationDispatcher yet since we have no sparse tests or examples in CK Tile.
    static_assert(SwizzleA == false);
    static_assert(UseStructuredSparsity == false);

    // Scale checks.
    // TODO: Add the tiny types after those are merged.
    static_assert(!IsMx || (std::is_same_v<AType, fp8_t> || std::is_same_v<AType, bf8_t>));
    static_assert(!IsMx || (std::is_same_v<BType, fp8_t> || std::is_same_v<BType, bf8_t>));

    // Non scale checks;
    static_assert(IsMx || AttrNumAccessA == ESingle);
    static_assert(IsMx || AttrNumAccessB == ESingle);

    // Convert SwizzleA bool to SwizzleFactor. This used to be hardcoded in a number of places in
    // the original dispatcher / warpgemms, generally using a factor of 2 if swizzling was
    // requested but not always. TODO: Check original usage for correct swizzle factors.
    static constexpr index_t SwizzleFactor = SwizzleA ? 2 : 1;

    // Convert WGAttrNumAccessEnums to index_t values.
    static constexpr index_t AttrNumAccessAV = get_wgattr_num_access<AttrNumAccessA>::value;
    static constexpr index_t AttrNumAccessBV = get_wgattr_num_access<AttrNumAccessB>::value;

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
                                     AttrNumAccessBV>::Type;
};

// clang-format on
} // namespace warp_gemm_dispatcher
} // namespace impl
} // namespace ck_tile
#endif // #if USE_NEW_UNIFIED_FRAMEWORK
