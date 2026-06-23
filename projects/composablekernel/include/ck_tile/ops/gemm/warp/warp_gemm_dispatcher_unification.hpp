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
          typename CompilerTarget>
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
          typename CompilerTarget>
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
                           CompilerTarget>
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
                                  CompilerTarget>;
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
          typename CompilerTarget>
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
                           CompilerTarget>
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
                                     CompilerTarget>;
};

template <typename FallbackPipeline, typename... Pipelines>
struct CurrentTargetPipelineSelector;

template <typename FallbackPipeline, typename Pipeline, typename... TailPipelines>
struct CurrentTargetPipelineSelector<FallbackPipeline, Pipeline, TailPipelines...>
{
    static constexpr auto currentTargetId = get_compiler_target().TARGET_ID;
    static constexpr auto pipelineTargetId =
        MmaOpTraits<typename Pipeline::MmaOp>::CompilerTarget::TARGET_ID;

    using Type = std::conditional_t<
        currentTargetId == pipelineTargetId,
        Pipeline,
        typename CurrentTargetPipelineSelector<FallbackPipeline, TailPipelines...>::Type>;
};

template <typename FallbackPipeline>
struct CurrentTargetPipelineSelector<FallbackPipeline>
{
    using Type = FallbackPipeline;
};

template <typename FirstPipeline, typename... RestPipelines>
struct RuntimeTargetMmaPipeline : FirstPipeline, RestPipelines...
{
    using ActivePipeline = typename CurrentTargetPipelineSelector<FirstPipeline,
                                                                  FirstPipeline,
                                                                  RestPipelines...>::Type;

    using MmaOp = typename ActivePipeline::MmaOp;

    using ADataType = typename ActivePipeline::ADataType;
    using BDataType = typename ActivePipeline::BDataType;
    using CDataType = typename ActivePipeline::CDataType;

    static constexpr index_t kM = ActivePipeline::kM;
    static constexpr index_t kN = ActivePipeline::kN;
    static constexpr index_t kK = ActivePipeline::kK;

    static constexpr index_t kKPerThread = ActivePipeline::kKPerThread;
    static constexpr index_t kAKPack     = ActivePipeline::kAKPack;
    static constexpr index_t kBKPack     = ActivePipeline::kBKPack;

    using WarpGemmAttribute = typename ActivePipeline::WarpGemmAttribute;

    using AWarpDstrEncoding = typename ActivePipeline::AWarpDstrEncoding;
    using BWarpDstrEncoding = typename ActivePipeline::BWarpDstrEncoding;
    using CWarpDstrEncoding = typename ActivePipeline::CWarpDstrEncoding;

    using AWarpDstr = typename ActivePipeline::AWarpDstr;
    using BWarpDstr = typename ActivePipeline::BWarpDstr;
    using CWarpDstr = typename ActivePipeline::CWarpDstr;

    using AWarpTensor = typename ActivePipeline::AWarpTensor;
    using BWarpTensor = typename ActivePipeline::BWarpTensor;
    using CWarpTensor = typename ActivePipeline::CWarpTensor;

    using ATransform = typename ActivePipeline::ATransform;
    using BTransform = typename ActivePipeline::BTransform;
    using CTransform = typename ActivePipeline::CTransform;
    using DTransform = typename ActivePipeline::DTransform;

    private:
    template <typename Pipeline, typename CTensor, typename ATensor, typename BTensor>
    static constexpr bool isPipelineCompatible()
    {
        return std::is_same_v<ck_tile::remove_cvref_t<CTensor>, typename Pipeline::CWarpTensor> &&
               std::is_same_v<ck_tile::remove_cvref_t<ATensor>, typename Pipeline::AWarpTensor> &&
               std::is_same_v<ck_tile::remove_cvref_t<BTensor>, typename Pipeline::BWarpTensor>;
    }

    template <typename Pipeline, typename CTensor, typename ATensor, typename BTensor>
    CK_TILE_DEVICE static void execIfCompatible(CTensor& c, ATensor& a, const BTensor& b)
    {
        if constexpr(isPipelineCompatible<Pipeline, CTensor, ATensor, BTensor>())
        {
            Pipeline{}(c, a, b);
        }
        else
        {
            static_assert(ck_tile::always_false_v<Pipeline>,
                          "RuntimeTargetMmaPipeline selected a target with incompatible warp "
                          "tensor layouts.");
        }
    }

    template <typename Pipeline,
              index_t opselA,
              index_t opselB,
              typename CTensor,
              typename ATensor,
              typename BTensor,
              typename ScaleADataType,
              typename ScaleBDataType>
    CK_TILE_DEVICE static void execScaleIfCompatible(CTensor& c,
                                                     const ATensor& a,
                                                     const BTensor& b,
                                                     const ScaleADataType& a_scale,
                                                     const ScaleBDataType& b_scale)
    {
        if constexpr(isPipelineCompatible<Pipeline, CTensor, ATensor, BTensor>())
        {
            Pipeline::template exec<opselA, opselB>(a, b, c, a_scale, b_scale);
        }
        else
        {
            static_assert(ck_tile::always_false_v<Pipeline>,
                          "RuntimeTargetMmaPipeline selected a target with incompatible scale "
                          "warp tensor layouts.");
        }
    }

    template <typename Pipeline,
              typename... TailPipelines,
              typename CTensor,
              typename ATensor,
              typename BTensor>
    CK_TILE_DEVICE static void dispatch(CTensor& c, ATensor& a, const BTensor& b)
    {
        constexpr auto currentTargetId = get_compiler_target().TARGET_ID;
        constexpr auto pipelineTargetId =
            MmaOpTraits<typename Pipeline::MmaOp>::CompilerTarget::TARGET_ID;

        if constexpr(currentTargetId == pipelineTargetId)
        {
            execIfCompatible<Pipeline>(c, a, b);
        }
        else if constexpr(sizeof...(TailPipelines) > 0)
        {
            dispatch<TailPipelines...>(c, a, b);
        }
        else
        {
            static_assert(ck_tile::always_false_v<Pipeline>,
                          "No RuntimeTargetMmaPipeline target matches the compiler target.");
        }
    }

    template <index_t opselA,
              index_t opselB,
              typename Pipeline,
              typename... TailPipelines,
              typename CTensor,
              typename ATensor,
              typename BTensor,
              typename ScaleADataType,
              typename ScaleBDataType>
    CK_TILE_DEVICE static void dispatchScale(CTensor& c,
                                             const ATensor& a,
                                             const BTensor& b,
                                             const ScaleADataType& a_scale,
                                             const ScaleBDataType& b_scale)
    {
        constexpr auto currentTargetId = get_compiler_target().TARGET_ID;
        constexpr auto pipelineTargetId =
            MmaOpTraits<typename Pipeline::MmaOp>::CompilerTarget::TARGET_ID;

        if constexpr(currentTargetId == pipelineTargetId)
        {
            execScaleIfCompatible<Pipeline, opselA, opselB>(c, a, b, a_scale, b_scale);
        }
        else if constexpr(sizeof...(TailPipelines) > 0)
        {
            dispatchScale<opselA, opselB, TailPipelines...>(c, a, b, a_scale, b_scale);
        }
        else
        {
            static_assert(ck_tile::always_false_v<Pipeline>,
                          "No RuntimeTargetMmaPipeline scale target matches the compiler target.");
        }
    }

    public:
    // Params are intentionally accepted to keep the dense and scale WarpGemm call interfaces
    // aligned.
    template <typename... Params, typename CTensor, typename ATensor, typename BTensor>
    CK_TILE_DEVICE void operator()(CTensor& c, ATensor& a, const BTensor& b) const
    {
        dispatch<FirstPipeline, RestPipelines...>(c, a, b);
    }

    template <typename... Params,
              typename CTensor,
              typename ATensor,
              typename BTensor,
              typename ScaleADataType,
              typename ScaleBDataType>
    CK_TILE_DEVICE void operator()(CTensor& c,
                                   const ATensor& a,
                                   const BTensor& b,
                                   const ScaleADataType& a_scale,
                                   const ScaleBDataType& b_scale) const
    {
        using P = WarpGemmParamsParser<Params...>;
        dispatchScale<P::op_sel_a, P::op_sel_b, FirstPipeline, RestPipelines...>(
            c, a, b, a_scale, b_scale);
    }
};

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
          typename... CompilerTargets>
struct RuntimeTargetMmaPipelineSelector
{
    static_assert(sizeof...(CompilerTargets) > 0, "At least one compiler target is required.");

    using Type = RuntimeTargetMmaPipeline<typename MmaPipelineSelector<IsMx,
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
                                                                       CompilerTargets>::Type...>;
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
          typename... CompilerTargets>
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
                           amdgcn_targets<CompilerTargets...>>
    : RuntimeTargetMmaPipelineSelector<true,
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
                                       CompilerTargets...>
{
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
          typename... CompilerTargets>
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
                           amdgcn_targets<CompilerTargets...>>
    : RuntimeTargetMmaPipelineSelector<false,
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
                                       CompilerTargets...>
{
};

// TODO: Figure out how to deal with the "packed" version of AttrNumAccess. In the unification
// framework there is no reason to combine packedness with AttrNumAccess but in CK Tile they did,
// alongside the refactor introducing gfx1250.
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

template <typename AType,
          typename BType,
          typename AccType,
          index_t MPerWave,
          index_t NPerWave,
          index_t KPerWave,
          bool TransposeC,
          bool SwizzleA                      = false,
          bool UseStructuredSparsity         = false,
          WGAttrNumAccessEnum AttrNumAccessA = WGAttrNumAccessEnum::Single,
          WGAttrNumAccessEnum AttrNumAccessB = AttrNumAccessA,
          bool IsScale16                     = false,
          typename CompilerTarget            = decltype(getCMakeCompilerTargets())>
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

    // General checks. Swizzle not supported yet. Structured sparsity Mma pipeline not adapted to
    // UnificationDispatcher yet since we have no sparse tests or examples in CK Tile.
    static_assert(SwizzleA == false);
    static_assert(UseStructuredSparsity == false);

    // Scale checks.
    // TODO: Add the tiny types after those are merged.
    static_assert(!IsMx || (std::is_same_v<AType, fp8_t> || std::is_same_v<AType, bf8_t> ||
                            std::is_same_v<AType, pk_fp4_t>));
    static_assert(!IsMx || (std::is_same_v<BType, fp8_t> || std::is_same_v<BType, bf8_t> ||
                            std::is_same_v<BType, pk_fp4_t>));

    // Convert SwizzleA bool to SwizzleFactor. This used to be hardcoded in a number of places in
    // the original dispatcher / warpgemms, generally using a factor of 2 if swizzling was
    // requested but not always. TODO: Check original usage for correct swizzle factors.
    static constexpr index_t SwizzleFactor = SwizzleA ? 2 : 1;

    // Convert WGAttrNumAccessEnums to index_t values. Default value sent to 1 for now, but needs a
    // better implementation TODO.
    static constexpr index_t AttrNumAccessAV = get_wgattr_num_access_safe_v<AttrNumAccessA>::value;
    static constexpr index_t AttrNumAccessBV = get_wgattr_num_access_safe_v<AttrNumAccessB>::value;

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
                                     CompilerTarget>::Type;
};
} // namespace warp_gemm_dispatcher
} // namespace impl
} // namespace ck_tile
#endif // #if USE_NEW_UNIFIED_FRAMEWORK
