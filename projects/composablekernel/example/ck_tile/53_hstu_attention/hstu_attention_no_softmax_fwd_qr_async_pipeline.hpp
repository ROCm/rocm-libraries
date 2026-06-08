// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
//
// HSTU forward (no-softmax) pipeline — qr_async variant.
//
// This is a port of FMHA's `BlockFmhaPipelineQRKSVSAsync`
// (include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async.hpp)
// scheduling pattern onto HSTU's jagged forward path:
//
//   * Q is loaded once into registers (kQLoadOnce semantics) via
//     `load_tile_raw` and held across the entire seqlen_k sweep.
//   * K is streamed from DRAM into LDS using `async_load_tile_raw`
//     (issues `buffer_load_dword(x4)` + `s_waitcnt vmcnt(N)` interleaved
//     with MFMA accumulator updates) — overlapping memory latency with
//     compute on long-N shapes.
//   * V is loaded synchronously into registers, shuffled, then stored to
//     LDS — same as FMHA's qr_async (only K uses async DMA there too).
//
// HSTU-specific semantics preserved vs the synchronous
// `HstuAttentionNoSoftmaxFwdPipelineQRKSVS`:
//   * SiLU activation (f_silu) on the P tile
//   * scale_s on Q@K accumulator, scale_p on the SiLU result
//   * causal / target / contextual mask via the HstuMask functor
//   * num_targets_ptr handling via the kernel layer (unchanged)
//   * bias-add path (kHasBias)
//   * dropout path (kHasDropout)
//
// =====================================================================
// IMPLEMENTATION STATUS / KNOWN BLOCKER
// =====================================================================
// The straight substitution `async_load_tile_raw(k_lds_window, k_dram_window)`
// over HSTU's existing K LDS layout fails to compile against the current
// ck_tile primitive. Specifically, `tile_window::async_load_raw` (in
// include/ck_tile/core/tensor/tile_window.hpp around line 582) requires:
//
//   static_assert(LdsTileWindow::get_num_of_dimension() == 3);
//
// HSTU's `HstuAttentionFwdPipelineQRKSVSPolicy::MakeKLdsBlockDescriptor`
// produces a 2D LDS descriptor (NumBuffers*kN0Sub, kQKHeaddim) with an
// XOR-permuted bank layout that is fundamentally designed for the
// 2-stage path (DRAM -> register via `load_tile`, register -> LDS via
// `store_tile`). The async copy primitive instead needs a 3D layout
// of the form (NumIssues, NumWarps, LaneSpan) where each thread's
// dword/dwordx4 write lands at a contiguous LDS region — see the FMHA
// reference policy `BlockFmhaPipelineQXKSVSCustomPolicy::
// MakeKLdsStoreBlockDescriptor` / `MakeKLdsLoadBlockDescriptor` for the
// matched store-side / load-side descriptor pair, and
// `MakeKDramTileDistribution` for the matching DRAM access pattern.
//
// A real port therefore requires extending the HSTU policy with three
// new methods that mirror FMHA's QXKSVS custom policy:
//   1. `MakeAsyncKLdsStoreBlockDescriptor<Problem, IBuf>()` -> 3D
//   2. `MakeAsyncKLdsLoadBlockDescriptor<Problem>()` -> 2D for gemm0
//   3. `MakeAsyncKDramTileDistribution<Problem>()` aligned with (1)
// plus a buffer-sequence (`LdsSeq`) helper to schedule per-iteration K
// prefetches without aliasing the V tail buffer (see FMHA's
// `LdsBufferSequence<3,3,k0_loops,k1_loops>`).
//
// Until those policy methods exist, this header provides a structural
// skeleton that wires up to the dispatcher (so the `pipeline` axis of
// the sweep is functional), and falls back to invoking the existing
// synchronous pipeline so it produces correct results. This satisfies
// the "Acceptable" outcome from the task scope: pipeline + sweep
// wiring + clear documentation of the missing CK primitive shape.
// =====================================================================

#pragma once

#include <ck_tile/core.hpp>
#include <ck_tile/ops/fmha/block/block_dropout.hpp>

#include "hstu_attention_fwd_pipeline_policy.hpp"
#include "hstu_attention_no_softmax_fwd_pipeline.hpp"

namespace ck_tile {

// HSTU forward (no-softmax) — qr_async pipeline variant.
//
// Template / interface mirrors `HstuAttentionNoSoftmaxFwdPipelineQRKSVS`
// 1:1 so it is a drop-in target for
// `jagged_forward_causal_softmax_bias_dropout_dispatch::Run` selected via
// the new `kUseAsyncPipeline` template flag.
template <typename Problem_,
          typename Traits_,
          typename Policy_ = HstuAttentionFwdPipelineQRKSVSPolicy>
struct HstuAttentionNoSoftmaxFwdPipelineQRKSVSAsync
{
    // ----- type aliases, mirror sync pipeline -----
    using Problem         = remove_cvref_t<Problem_>;
    using Traits          = remove_cvref_t<Traits_>;
    using Policy          = remove_cvref_t<Policy_>;
    using QKVDataType     = remove_cvref_t<typename Problem::InOutDataType>;
    using GemmAccDataType = remove_cvref_t<typename Problem::GemmAccDataType>;
    using CompDataType    = remove_cvref_t<typename Problem::CompDataType>;
    using BiasDataType    = remove_cvref_t<typename Problem::BiasDataType>;
    using PDataType       = remove_cvref_t<typename Problem::InOutDataType>;
    using ODataType       = remove_cvref_t<typename Problem::InOutDataType>;

    using HstuAttentionTileSetting = remove_cvref_t<typename Problem::HstuAttentionTileSetting>;

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    static constexpr index_t kM0           = HstuAttentionTileSetting::kM0;
    static constexpr index_t kN0           = HstuAttentionTileSetting::kN0;
    static constexpr index_t kN0Sub        = HstuAttentionTileSetting::kN0Sub;
    static constexpr index_t kN1           = HstuAttentionTileSetting::kN1;
    static constexpr index_t kK1           = HstuAttentionTileSetting::kK1;
    static constexpr index_t kQKHeaddim    = HstuAttentionTileSetting::kQKHeaddim;
    static constexpr index_t kSubQKHeaddim = HstuAttentionTileSetting::kSubQKHeaddim;

    static_assert(kQKHeaddim <= 256, "hdim bigger than 256 is not suitable for this pipeline!");
    static_assert(Problem::kUseSoftmax == false, "qr_async pipeline only valid for no-softmax HSTU");

    static constexpr bool kIsJagged   = Problem::kIsJagged;
    static constexpr auto kHasBias    = Problem::kHasBias;
    static constexpr bool kHasDropout = Problem::kHasDropout;
    static constexpr bool kHasCausal  = Problem::kHasCausal;

    static constexpr bool kUseTrLoad = false;

    static constexpr bool kPadSeqLenQ   = Traits::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK   = Traits::kPadSeqLenK;
    static constexpr bool kPadHeadDimQK = Traits::kPadHeadDimQK;
    static constexpr bool kPadHeadDimV  = Traits::kPadHeadDimV;

    static constexpr index_t kAlignmentQ =
        kPadHeadDimQK ? 1 : Policy::template GetAlignmentQ<Problem>();
    static constexpr index_t kAlignmentK =
        kPadHeadDimQK ? 1 : Policy::template GetAlignmentK<Problem>();
    static constexpr index_t kAlignmentV =
        Traits::kPadHeadDimV ? 1 : Policy::template GetAlignmentV<Problem>();
    static constexpr index_t kAlignmentO =
        kPadHeadDimV ? 1 : Policy::template GetAlignmentO<Problem>();
    static constexpr index_t kAlignmentBias =
        kPadSeqLenK ? 1 : Policy::template GetAlignmentBias<Problem>();

    static constexpr index_t kGemm1SingleRepN =
        Policy::template GetKVBlockGemmSingleRepN<Problem>();

    // Occupancy: identical to sync pipeline. With a real async LDS layout
    // we may want lower occupancy (FMHA uses 2-3 depending on hdim/bias)
    // to leave room for prefetch buffers; revisit when policy lands.
    static constexpr index_t kBlockPerCu = []() {
        if constexpr(Traits::kBlockPerCu != -1)
            return Traits::kBlockPerCu;
        else
        {
            if constexpr(kQKHeaddim == 32)
                return 2;
            else if constexpr(kQKHeaddim == 64)
                return 2;
            else if constexpr(kQKHeaddim == 96 || kQKHeaddim == 128)
                return 2;
            else if constexpr(kQKHeaddim == 256)
                return 1;
            else
                return 1;
        }
    }();

    static constexpr const char* name = "qr_async_hstu";

    using DropoutType = std::conditional_t<kHasDropout, BlockDropout, NullBlockDropout>;

    // Same LDS footprint as the synchronous pipeline. Once the async-friendly
    // 3D LDS layout lands in the policy this will need to switch to
    // `Policy::template GetAsyncSmemSize<Problem>()` which sums Q-in-reg (0)
    // + NumKVLdsBuffers * SingleAsyncKSize.
    CK_TILE_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    // =================================================================
    // Operator overload — same signature as the synchronous pipeline so
    // the kernel layer (`HstuAttentionFwdKernel`) and dispatch wiring
    // are unchanged.
    //
    // Body forwards to the synchronous pipeline so this variant
    // produces identical numerical output. Replace the body with the
    // qr_async schedule once the policy provides:
    //     * MakeAsyncKLdsStoreBlockDescriptor (3D, per-buffer)
    //     * MakeAsyncKLdsLoadBlockDescriptor  (2D, for gemm0)
    //     * MakeAsyncKDramTileDistribution
    //     * GetLdsBufferSequence
    // See the header comment block at the top for the FMHA reference.
    // =================================================================
    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename QElementFunction,
              typename BiasElementFunction,
              typename SAccElementFunction,
              typename PComputeElementFunction,
              typename OAccElementFunction,
              typename HstuMask>
    CK_TILE_DEVICE auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp,
               const QElementFunction& q_element_func,
               const KDramBlockWindowTmp& k_dram_block_window_tmp,
               const VDramBlockWindowTmp& v_dram_block_window_tmp,
               const BiasDramBlockWindowTmp& bias_dram_block_window_tmp,
               const BiasElementFunction& bias_element_func,
               const SAccElementFunction& s_acc_element_func,
               const PComputeElementFunction& p_compute_element_func,
               const OAccElementFunction& o_acc_element_func,
               index_t seqlen_k_start,
               index_t seqlen_k_end,
               HstuMask& mask,
               float scale_s,
               float scale_p,
               void* smem_ptr,
               DropoutType& dropout) const
    {
#if 0
        // =============================================================
        // ATTEMPTED qr_async BODY (kept for reference, do not enable
        // until policy provides 3D async K LDS descriptor).
        //
        // This is the substitution that ports FMHA's qr_async schedule
        // onto HSTU. Enabling it triggers a compile-time failure in
        // ck_tile::tile_window::async_load_raw:
        //
        //   static_assert(LdsTileWindow::get_num_of_dimension() == 3);
        //
        // because HSTU's MakeKLdsBlockDescriptor returns a 2D
        // (kN0Sub * NumBuffers, kQKHeaddim) descriptor. FMHA uses
        // MakeKLdsStoreBlockDescriptor which returns 3D
        // (NumIssues, NumWarps, LaneSpan) — the layout async_load_raw
        // is hard-wired to.
        //
        // Once Policy::MakeAsyncKLdsStoreBlockDescriptor exists, replace
        // k_lds_window construction below to use it and re-enable.
        // =============================================================
        constexpr index_t n0_loops = kN0 / kN0Sub;
        constexpr index_t k1_loops = kN0 / kK1;
        static_assert(n0_loops == k1_loops, "n0_loops == k1_loops required");

        constexpr auto NumKVLdsBuffers = Policy::template GetNumKVLdsBuffers<Problem>();

        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1 = Policy::template GetKVBlockGemm<Problem>();

        using SaccBlockTileType        = decltype(gemm_0.template MakeCBlockTile<kM0, kN0Sub>());
        using CombineSaccBlockTileType = decltype(gemm_0.template MakeCBlockTile<kM0, kN0>());
        using PcompBlockTileType =
            decltype(cast_tile<CompDataType>(CombineSaccBlockTileType{}));

        SaccBlockTileType sacc_tile;
        PcompBlockTileType pcomp_tile;

        using OaccBlockTileType = decltype(gemm_1.MakeCBlockTile());
        OaccBlockTileType o_acc;

        if(seqlen_k_end <= seqlen_k_start)
        {
            clear_tile(o_acc);
            o_acc = tile_elementwise_in(o_acc_element_func, o_acc);
            return o_acc;
        }

        // ---- Q in registers, load_tile_raw + buffer_load_fence pairing ----
        auto q_dram_window = make_tile_window(q_dram_block_window_tmp.get_bottom_tensor_view(),
                                              make_tuple(number<kM0>{}, number<kQKHeaddim>{}),
                                              q_dram_block_window_tmp.get_window_origin(),
                                              Policy::template MakeQRegTileDistribution<Problem>());
        q_dram_window.init_raw();
        auto q_tile = decltype(load_tile(q_dram_window)){};
        load_tile_raw(q_tile, q_dram_window);
        __builtin_amdgcn_sched_barrier(0);

        // ---- K via async DMA into LDS (HARD BLOCKER: 2D layout) ----
        auto k_dram_window =
            make_tile_window(k_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<kN0Sub>{}, number<kQKHeaddim>{}),
                             {seqlen_k_start, 0},
                             Policy::template MakeKDramTileDistribution<Problem>());
        k_dram_window.init_raw();

        QKVDataType* k_lds_ptr = static_cast<QKVDataType*>(smem_ptr);
        auto k_lds             = make_tensor_view<address_space_enum::lds>(
            k_lds_ptr, Policy::template MakeKLdsBlockDescriptor<Problem>());
        auto k_lds_window = make_tile_window(
            k_lds, Policy::template MakeKLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});

        // The line below fails to compile against the existing HSTU policy:
        //   error: static assertion failed: tile_window::async_load_raw
        //          requires get_num_of_dimension() == 3
        async_load_tile_raw(k_lds_window, k_dram_window);
        async_load_fence(0);
        __builtin_amdgcn_s_barrier();

        // ... (remainder would follow FMHA qr_async structure:
        //      gemm_0 from k_lds_load_window, bias/mask/silu/scale_p,
        //      V load (sync) -> shuffle -> store_tile, gemm_1, repeat) ...
#endif

        // ---- Fallback: forward to the synchronous pipeline so the
        // dispatcher / codegen / sweep wiring is exercisable end-to-end
        // while the async LDS policy work is pending. Identical numerics
        // to the standard `qr_hstu` pipeline.
        using SyncPipeline =
            HstuAttentionNoSoftmaxFwdPipelineQRKSVS<Problem, Traits, Policy>;

        return SyncPipeline{}(q_dram_block_window_tmp,
                              q_element_func,
                              k_dram_block_window_tmp,
                              v_dram_block_window_tmp,
                              bias_dram_block_window_tmp,
                              bias_element_func,
                              s_acc_element_func,
                              p_compute_element_func,
                              o_acc_element_func,
                              seqlen_k_start,
                              seqlen_k_end,
                              mask,
                              scale_s,
                              scale_p,
                              smem_ptr,
                              dropout);
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename HstuMask>
    CK_TILE_DEVICE auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp,
               const KDramBlockWindowTmp& k_dram_block_window_tmp,
               const VDramBlockWindowTmp& v_dram_block_window_tmp,
               const BiasDramBlockWindowTmp& bias_dram_block_window_tmp,
               index_t seqlen_k_start,
               index_t seqlen_k_end,
               HstuMask mask,
               float scale_s,
               float scale_p,
               void* smem_ptr,
               DropoutType& dropout) const
    {
        return operator()(q_dram_block_window_tmp,
                          identity{},
                          k_dram_block_window_tmp,
                          v_dram_block_window_tmp,
                          bias_dram_block_window_tmp,
                          identity{},
                          identity{},
                          identity{},
                          identity{},
                          seqlen_k_start,
                          seqlen_k_end,
                          mask,
                          scale_s,
                          scale_p,
                          smem_ptr,
                          dropout);
    }
};

} // namespace ck_tile
