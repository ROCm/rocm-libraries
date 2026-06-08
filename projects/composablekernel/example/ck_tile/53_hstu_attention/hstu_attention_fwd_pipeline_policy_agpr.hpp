// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
//
// HSTU forward pipeline policy with AGPR-pinned MFMA inputs (HipKittens
// Principle 2 — `WGAttrCtlEnum::Raw_vaa`).
//
// Hypothesis (Round 1 → Round 2): on the train_b1024_h4 deployment shape
// the existing best kernel (`hstu_bf16_causal1_maxk64_mtile64_splitkv0_std_wk16`)
// reports `MfmaUtil = 25 %` while `GPU_UTIL = 100 %` — i.e. ~75 % of
// VALU cycles are non-MFMA. HipKittens' source-of-truth doc
// (`/workspaces/.github/knowledge/hipkittens-amd-kernel-techniques.md`,
// Principle 2) attributes a chunk of that to redundant
// `v_accvgpr_read` / `v_accvgpr_write` shuffles HIPCC emits when MFMA
// inputs are kept in VGPRs by default. Pinning A/B operands into AGPRs
// via the existing CK Tile `WGAttrCtlEnum::Raw_vaa` ASM path gives
// HIPCC a fixed register-class for the two input operands and lets it
// cluster the input shuffles minimally, freeing VALU cycles for MFMA.
//
// Implementation: this policy DERIVES from the standard
// `HstuAttentionFwdPipelineQRKSVSPolicy` and overrides ONLY the two
// `GetQKBlockGemm` / `GetKVBlockGemm` factories. The override bypasses
// `WarpGemmDispatcher<...>` (which hardcodes `WGAttrCtlEnum::Default_`)
// and instantiates the same `WarpGemmImpl` wrapper hierarchy by hand
// with `Raw_vaa` injected at the leaf MFMA-impl template. All other
// behavior — LDS layout, tile distributions, alignment, smem sizing —
// inherits from the base policy unchanged.
//
// Boundary: this is HSTU-example-local. CK Tile core
// (`include/ck_tile/...`) is NOT modified — the warp-gemm + MFMA-impl
// templates already accept a `Ctrl_` parameter; the public `using`
// aliases just hardcode `Default_` so we sidestep them.

#pragma once

#include <ck_tile/core.hpp>
#include <ck_tile/ops/gemm/block/block_gemm_problem.hpp>
#include <ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp>
#include <ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma.hpp>
#include <ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma_impl.hpp>
#include <ck_tile/ops/gemm/warp/warp_gemm_impl.hpp>

#include <ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v2_custom_policy.hpp>
#include <ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_one_warp_v1.hpp>

#include "block_gemm_areg_bsmem_creg_v2_hack_0.hpp"
#include "block_gemm_areg_bsmem_creg_v2_hack_1.hpp"
#include "hstu_attention_fwd_pipeline_policy.hpp"

namespace ck_tile {

// AGPR-pinned warp gemm aliases. They mirror the public CK Tile
// `WarpGemmMfmaBf16Bf16F32M16N16K{16,32}TransposedCDistribution` aliases
// but flip the MFMA-impl `Ctrl_` template from `Default_` to `Raw_vaa`
// so the inline-asm path is taken (operands `a/a` → AGPRs, c `+v` →
// VGPR). HSTU's both gemms always pass `TransposeC=true` to the
// dispatcher (see `HstuAttentionFwdPipelineQRKSVSPolicy::GetQKBlockGemm`
// / `GetKVBlockGemm`) so we only need the TransposedCDistribution
// wrappers.
//
// On gfx942 the MFMA opcode encodes both VGPR-output and AGPR-output
// variants via the ACC bit; the assembler picks the right encoding from
// the `+v` constraint. On gfx950 the same scheme extends to the
// `v_mfma_f32_16x16x32_bf16` opcode used by the WarpK=32 path.
namespace hstu_agpr_detail {

template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmAGPR_Bf16_M16N16K16_TransposedC =
    WarpGemmImpl<WarpGemmAttributeMfmaTransposedCDistribution<
        WarpGemmAttributeMfmaImplBf16Bf16F32M16N16K16<WGAttrCtlEnum::Raw_vaa>,
        AttrNumAccess>>;

#if defined(__gfx950__)
template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmAGPR_Bf16_M16N16K32_TransposedC =
    WarpGemmImpl<WarpGemmAttributeMfmaTransposedCDistribution<
        WarpGemmAttributeMfmaImplBf16Bf16F32M16N16K32<WGAttrCtlEnum::Raw_vaa>,
        AttrNumAccess>>;
#else
// On gfx942 (the round-1 measurement target) the WarpK=32 kernel
// instantiates `WarpGemmAttributeMfmaIterateK<...K16, 2>` because
// `v_mfma_f32_16x16x32_bf16` is gfx950-only. Mirror that.
template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmAGPR_Bf16_M16N16K32_TransposedC =
    WarpGemmImpl<WarpGemmAttributeMfmaIterateKAndTransposedCDistribution<
        WarpGemmAttributeMfmaImplBf16Bf16F32M16N16K16<WGAttrCtlEnum::Raw_vaa>,
        2,
        AttrNumAccess>>;
#endif

} // namespace hstu_agpr_detail

struct HstuAttentionFwdPipelineQRKSVSPolicyAGPR
    : public HstuAttentionFwdPipelineQRKSVSPolicy
{
    // Public inheritance: every other helper (LDS layouts, alignments,
    // distribution makers, SMEM sizers, NumKVLdsBuffers, ...) is
    // inherited as-is from `HstuAttentionFwdPipelineQRKSVSPolicy`. We
    // override only the two block-gemm factories below.

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetQKBlockGemm()
    {
        using GemmProblem = BlockGemmProblem<
            typename Problem::QKVDataType,
            typename Problem::QKVDataType,
            typename Problem::GemmAccDataType,
            Problem::kNumGemm0Warps * get_warp_size(),
            TileGemmShape<sequence<Problem::HstuAttentionTileSetting::kM0,
                                   Problem::HstuAttentionTileSetting::kN0Sub,
                                   Problem::HstuAttentionTileSetting::kQKHeaddim>,
                          typename Problem::HstuAttentionTileSetting::Gemm0BlockWarps,
                          typename Problem::HstuAttentionTileSetting::Gemm0WarpTile>>;

        constexpr index_t WarpGemmM =
            Problem::HstuAttentionTileSetting::Gemm0WarpTile::at(number<0>{});
        constexpr index_t WarpGemmK =
            Problem::HstuAttentionTileSetting::Gemm0WarpTile::at(number<2>{});

        static_assert(
            (std::is_same_v<typename Problem::QKVDataType, bf16_t> ||
             std::is_same_v<typename Problem::QKVDataType, half_t>),
            "AGPR-pinned HSTU policy currently bf16/fp16 only");
        static_assert(WarpGemmM == 16,
                      "AGPR-pinned HSTU policy currently 16x16 MFMA only "
                      "(WarpGemmM=16 expected; no 32x32 variant yet)");
        static_assert(WarpGemmK == 16 || WarpGemmK == 32,
                      "AGPR-pinned HSTU policy expects WarpGemmK in {16,32}");

        // Only bf16 is wired through dispatcher today; if fp16 ever flows
        // through this policy, fall back to the default ctrl path (the
        // bf16 path with `Raw_vaa` is the actual experiment).
        using AGPR16 =
            hstu_agpr_detail::WarpGemmAGPR_Bf16_M16N16K16_TransposedC<WGAttrNumAccessEnum::Single>;
        using AGPR32 =
            hstu_agpr_detail::WarpGemmAGPR_Bf16_M16N16K32_TransposedC<WGAttrNumAccessEnum::Single>;
        using WarpGemmT = std::conditional_t<WarpGemmK == 32, AGPR32, AGPR16>;

        using BlockGemmPolicy = BlockGemmARegBSmemCRegV2CustomPolicy<
            typename Problem::QKVDataType,
            typename Problem::QKVDataType,
            typename Problem::GemmAccDataType,
            typename Problem::HstuAttentionTileSetting::Gemm0BlockWarps,
            WarpGemmT>;

        if constexpr(1 < Problem::kNumGemm0Warps)
            return BlockGemmARegBSmemCRegV2Hack_0<GemmProblem, BlockGemmPolicy>{};
        else
            return BlockGemmARegBSmemCRegOneWarpV1<GemmProblem, BlockGemmPolicy>{};
    }

    template <typename Problem, bool kUseTrLoad = false>
    CK_TILE_HOST_DEVICE static constexpr auto GetKVBlockGemm()
    {
        // The trload pipeline lives in a separate header; this AGPR
        // policy targets only the synchronous (non-trload) path. If
        // someone wires trload + agpr later, they need to author the
        // matching block-gemm Hack_1 variant.
        static_assert(!kUseTrLoad,
                      "AGPR policy does not yet cover trload pipeline");

        using GemmProblem = BlockGemmProblem<
            typename Problem::QKVDataType,
            typename Problem::QKVDataType,
            typename Problem::GemmAccDataType,
            Problem::kNumGemm1Warps * get_warp_size(),
            TileGemmShape<sequence<Problem::HstuAttentionTileSetting::kM0,
                                   Problem::HstuAttentionTileSetting::kN1,
                                   Problem::HstuAttentionTileSetting::kK1>,
                          typename Problem::HstuAttentionTileSetting::Gemm1BlockWarps,
                          typename Problem::HstuAttentionTileSetting::Gemm1WarpTile>>;

        constexpr index_t WarpGemmM =
            Problem::HstuAttentionTileSetting::Gemm1WarpTile::at(number<0>{});
        constexpr index_t WarpGemmK =
            Problem::HstuAttentionTileSetting::Gemm1WarpTile::at(number<2>{});

        static_assert(
            (std::is_same_v<typename Problem::QKVDataType, bf16_t> ||
             std::is_same_v<typename Problem::QKVDataType, half_t>),
            "AGPR-pinned HSTU policy currently bf16/fp16 only");
        static_assert(WarpGemmM == 16,
                      "AGPR-pinned HSTU policy currently 16x16 MFMA only");
        static_assert(WarpGemmK == 16 || WarpGemmK == 32,
                      "AGPR-pinned HSTU policy expects WarpGemmK in {16,32}");

        // Match the base policy: KV gemm uses the `Double` access form
        // when WarpGemmK matches the native MFMA K (gfx950 16x16x32, or
        // gfx942 16x16x16-iterated to 32). The base policy picks
        // `WGAttrNumAccessEnum::Double` for K=32 / K=16 nondivided,
        // `Single` otherwise — replicating that selection here keeps
        // the C-distribution encoding identical to the base.
        constexpr auto kAccess =
            (WarpGemmK == 32) ? WGAttrNumAccessEnum::Double : WGAttrNumAccessEnum::Single;

        using AGPR16 = hstu_agpr_detail::WarpGemmAGPR_Bf16_M16N16K16_TransposedC<kAccess>;
        using AGPR32 = hstu_agpr_detail::WarpGemmAGPR_Bf16_M16N16K32_TransposedC<kAccess>;
        using WarpGemmT = std::conditional_t<WarpGemmK == 32, AGPR32, AGPR16>;

        using BlockGemmPolicy = BlockGemmARegBSmemCRegV2CustomPolicy<
            typename Problem::QKVDataType,
            typename Problem::QKVDataType,
            typename Problem::GemmAccDataType,
            typename Problem::HstuAttentionTileSetting::Gemm1BlockWarps,
            WarpGemmT>;

        return BlockGemmARegBSmemCRegV2Hack_1<GemmProblem, BlockGemmPolicy>{};
    }
};

} // namespace ck_tile
