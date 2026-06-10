// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async_default_policy.hpp"
#include "ck_tile/ops/fmha/block/block_dropout.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"

namespace ck_tile {

// PV-skip mode enum. kPerWave = per-wavefront butterfly vote. kPerBlock =
// block-wide consensus vote (matches upstream SpargeAttn kPerBlock semantics).
// kNone disables the skip path entirely (AST removed). Legacy bool
// `kEnablePVSkip_=true` maps to kPerWave; `false` maps to kNone (via codegen).
enum class PVSkipMode : int
{
    kNone     = 0,
    kPerWave  = 1,
    kPerBlock = 2,
};

// Sparge variant of qr/ks/vs/async pipeline. Cloned from BlockFmhaPipelineQRKSVSAsyncVSA;
// adds PV-skip per Q-tile (SpargeAttn paper 4.4). Kept as a separate file so the original
// _vsa.hpp can remain frozen as an A/B baseline.
//
// INVARIANT: V load / V->LDS store / cp_async pipeline stay UNCONDITIONAL in
// both per-wave and per-block modes — only the gemm_1 is gated.
template <typename Problem_,
          typename Policy_        = BlockFmhaPipelineQRKSVSAsyncDefaultPolicy,
          PVSkipMode kPVSkipMode_ = PVSkipMode::kPerWave>
struct BlockFmhaPipelineQRKSVSAsyncSparge
{
    static constexpr PVSkipMode kPVSkipMode = kPVSkipMode_;
    // Legacy alias: true iff any PV-skip mode (per-wave or per-block) is active.
    // Kept so existing `if constexpr (kEnablePVSkip)` reads still compile.
    static constexpr bool kEnablePVSkip   = (kPVSkipMode_ != PVSkipMode::kNone);
    static constexpr bool kPerBlockPVSkip = (kPVSkipMode_ == PVSkipMode::kPerBlock);

    using Problem               = remove_cvref_t<Problem_>;
    using Policy                = remove_cvref_t<Policy_>;
    using QDataType             = remove_cvref_t<typename Problem::QDataType>;
    using KDataType             = remove_cvref_t<typename Problem::KDataType>;
    using VDataType             = remove_cvref_t<typename Problem::VDataType>;
    using SaccDataType          = remove_cvref_t<typename Problem::SaccDataType>;
    using SMPLComputeDataType   = remove_cvref_t<typename Problem::SMPLComputeDataType>;
    using BiasDataType          = remove_cvref_t<typename Problem::BiasDataType>;
    using RandValOutputDataType = remove_cvref_t<typename Problem::RandValOutputDataType>;
    using LSEDataType           = remove_cvref_t<typename Problem::LSEDataType>;
    using PDataType             = remove_cvref_t<typename Problem::PDataType>;
    using OaccDataType          = remove_cvref_t<typename Problem::OaccDataType>;
    using ODataType             = remove_cvref_t<typename Problem::ODataType>;
    using AttentionVariant      = remove_cvref_t<typename Problem::AttentionVariant>;
    using FmhaMask              = remove_cvref_t<typename Problem::FmhaMask>;

    using BlockFmhaShape             = remove_cvref_t<typename Problem::BlockFmhaShape>;
    using VLayout                    = remove_cvref_t<typename BlockFmhaShape::VLayout>;
    static constexpr bool kQLoadOnce = true; // if q_tile load whole block length (hdim) at once
    static_assert(kQLoadOnce == Policy::QLoadOnce);

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    static constexpr index_t kM0           = BlockFmhaShape::kM0;
    static constexpr index_t kN0           = BlockFmhaShape::kN0;
    static constexpr index_t kK0           = BlockFmhaShape::kK0;
    static constexpr index_t kN1           = BlockFmhaShape::kN1;
    static constexpr index_t kK1           = BlockFmhaShape::kK1;
    static constexpr index_t kQKHeaddim    = BlockFmhaShape::kQKHeaddim;
    static constexpr index_t kSubQKHeaddim = BlockFmhaShape::kSubQKHeaddim;

    static_assert(kSubQKHeaddim <= 256, "hdim bigger than 256 is not suitable for this pipeline!");

    static constexpr bool kIsGroupMode = Problem::kIsGroupMode;
    // TODO: seq_q always support padding, hdim_q/v support multiple of vector(like 8x)
    //       only need special care about seq_k padding (oob need set -INF of p instead of zero)
    static_assert(Problem::kPadSeqLenQ == true && Problem::kPadHeadDimQ == true &&
                  Problem::kPadHeadDimV == true);
    static constexpr bool kPadSeqLenQ       = true;
    static constexpr bool kPadSeqLenK       = Problem::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ      = true; // support multiple of vector(like 8x)
    static constexpr bool kPadHeadDimV      = true; // support multiple of vector(like 8x)
    static constexpr bool kHasLogitsSoftCap = Problem::kHasLogitsSoftCap;
    static constexpr auto BiasEnum          = Problem::BiasEnum;
    static constexpr bool kStoreLSE         = Problem::kStoreLSE;
    static constexpr bool kHasDropout       = Problem::kHasDropout;
    static constexpr auto QScaleEnum        = Problem::QScaleEnum;

    // P1 plumbing scaffold (perf-neutral). Sage-style per-block quant scale
    // granularity, kept static-only here so kargs / host can size descale
    // buffers consistently with sage 49. Only BLOCKSCALE is wired for sparge;
    // other QScaleEnum values fall back to 128 (matches sage default tile size).
    // Actual int8 GEMM path is gated by the kDoFp8StaticQuant static_assert in
    // the kernel wrapper; arithmetic itself lands in P2/P3.
    static constexpr index_t kBlockScaleSizeQ = kM0;
    static constexpr index_t kBlockScaleSizeK = kN0;

    static_assert(BiasEnum == BlockAttentionBiasEnum::NO_BIAS,
                  "VSA sparse attention does not support bias.");
    static_assert(!kHasDropout, "VSA sparse attention does not support dropout.");
    static_assert(!kStoreLSE, "VSA sparse attention does not support LSE output.");
    static_assert(!kHasLogitsSoftCap, "VSA sparse attention does not support logits soft-cap.");

    // last dimension vector length used to create tensor view(and decide buffer_load vector length)
    // ... together with tensor distribution. tensor dist should able to overwrite this
    static constexpr index_t kAlignmentQ = Policy::template GetAlignmentQ<Problem>();
    static constexpr index_t kAlignmentK = Policy::template GetAlignmentK<Problem>();
    static constexpr index_t kAlignmentV = []() {
        if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
            return Policy::template GetAlignmentV<Problem>();
        else
            return kPadSeqLenK ? 1 : Policy::template GetAlignmentV<Problem>();
    }();
    static constexpr index_t kAlignmentO = Policy::template GetAlignmentO<Problem>();

#if CK_TILE_FMHA_FWD_FAST_EXP2
    static constexpr auto R_LOG2E = 1.0 / log2e_v<SaccDataType>;
#endif

    static constexpr index_t kBlockPerCu = []() {
        if constexpr(Problem::kBlockPerCu != -1)
            return Problem::kBlockPerCu;
        else
        {
            // minimize occupancy
            if constexpr(kQKHeaddim <= 32)
            {
                if constexpr(kPadSeqLenK && FmhaMask::IsMasking)
                    return 1;
                else
                    return 2;
            }
            else if constexpr(kQKHeaddim <= 64)
            {
                if constexpr(kPadSeqLenK)
                    return 2;
                else
                    return 3;
            }
            else if constexpr(kQKHeaddim <= 128)
            {
                if constexpr(kPadSeqLenK)
                    return 1;
                else
                    return 2;
            }
            else if constexpr(kQKHeaddim <= 192)
            {
                if constexpr(kPadSeqLenK)
                    return 1;
                else
                    return 2;
            }
            else if constexpr(kQKHeaddim <= 256)
            {
                return 1;
            }
            else
            {
                return 1;
            };
        }
    }();

    static constexpr const char* name = "qr_async";

    // Per-block PV-skip needs one int32 LDS slot to broadcast the AND-vote
    // result across waves. Reserved at the TAIL of the pipeline's LDS budget
    // (after the existing K + V allocations). Always reserved (4 B vs the
    // multi-kB K/V tiles is negligible) so the smem layout stays uniform.
    static constexpr ck_tile::index_t kPerBlockVoteSlotBytes = 4;

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>() + kPerBlockVoteSlotBytes;
    }

    // Byte offset of the per-block vote flag from `smem_ptr`. Lives just past
    // the policy's K+V smem footprint.
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetPerBlockVoteSlotOffset()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename AttentionVariantParams,
              typename BlockIndices>
    CK_TILE_HOST_DEVICE auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp, // M0*K0 tile
               const KDramBlockWindowTmp& k_dram_block_window_tmp, // N0*K0 tile
               const VDramBlockWindowTmp& v_dram_block_window_tmp, // N1*K1 tile
               const int* kv_block_idx_ptr,
               int kv_blocks,
               FmhaMask mask,
               float scale_s,
               float pv_threshold, // SpargeAttn PV-skip threshold; see §2 of pv_skip plan
               const AttentionVariant& variant,
               const AttentionVariantParams& variant_params,
               const BlockIndices& block_indices,
               void* smem_ptr,
               // Descale buffers per sage 49 contract. Used only on the int8
               // BLOCKSCALE path (Q/K=int8); under fp16 NO_SCALE everything
               // is no-op (q_descale_value defaults to 1.0f and the
               // k_descale_ptr load is gated by QDataType=int8_t below).
               // V is fp16 in S3c2, so v_descale_ptr stays unused.
               const float* q_descale_ptr             = nullptr,
               const float* k_descale_ptr             = nullptr,
               const float* v_descale_ptr             = nullptr,
               [[maybe_unused]] float q_descale_value = 1.0f) const
    {
        // q_descale_ptr is consumed kernel-side (scalar q_descale_value already
        // holds the per-Q-block scalar). k_descale_ptr is read per K-loop iter
        // under int8 path below. v_descale_ptr stays unused while V is fp16.
        (void)q_descale_ptr;
        (void)v_descale_ptr;
        // k_descale_ptr only matters on the int8 path; silence unused when fp16.
        if constexpr(!(std::is_same_v<QDataType, int8_t> && std::is_same_v<KDataType, int8_t>))
        {
            (void)k_descale_ptr;
        }
        if constexpr(!kEnablePVSkip)
        {
            (void)pv_threshold; // silence unused-param when PV-skip is compiled out
        }
        // PV-skip control is a compile-time gate (kEnablePVSkip). When false
        // the AST contains no vote / scalar gate / extra LDS, and codegen
        // converges with _vsa.hpp's FmhaFwdVSAKernel.
        // Runtime fast-path: pv_threshold == +1e30 sentinel disables the skip
        // via one scalar (sgpr) branch inside the `if constexpr` block.
        static_assert(
            std::is_same_v<QDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                std::is_same_v<KDataType, remove_cvref_t<typename KDramBlockWindowTmp::DataType>> &&
                std::is_same_v<VDataType, remove_cvref_t<typename VDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kM0 == QDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == KDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kK0 == KDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
                          kN1 == VDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kK1 == VDramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
                      "wrong!");

        constexpr auto LdsSeq = Policy::template GetLdsBufferSequence<Problem>();

        // K tile in LDS
        auto k_lds_ptr   = reinterpret_cast<KDataType*>(smem_ptr);
        auto k_lds_store = generate_tuple(
            [&](auto i_buf) {
                return make_tile_window(
                    make_tensor_view<address_space_enum::lds>(
                        k_lds_ptr, Policy::template MakeKLdsStoreBlockDescriptor<Problem>(i_buf)),
                    Policy::template MakeKLdsStoreBlockDescriptor<Problem>(i_buf).get_lengths(),
                    {0, 0, 0});
            },
            number<Policy::NumKVLdsBuffers>{});

        auto k_lds_Load_view = make_tensor_view<address_space_enum::lds>(
            k_lds_ptr, Policy::template MakeKLdsLoadBlockDescriptor<Problem>());

        auto k_lds_load =
            make_tile_window(k_lds_Load_view,
                             Policy::template MakeKLdsLoadBlockDescriptor<Problem>().get_lengths(),
                             {0, 0});

        // V tile in LDS
        auto v_lds = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<VDataType*>(smem_ptr),
            Policy::template MakeVLdsBlockDescriptor<Problem>());
        auto v_lds_window = make_tile_window(
            v_lds, Policy::template MakeVLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});

        // Block GEMM
        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1 = Policy::template GetKVBlockGemm<Problem>();

        int seqlen_k_start = kv_block_idx_ptr[0] * kN0;
        // Sparge LUT-aware absolute K-block index accumulator. Seeded from the
        // first absolute block (LUT[0]) and bumped by each LUT delta after
        // every K-side move_tile_window. Used by the int8 per-block dequant
        // path instead of get_window_origin() to avoid any compiler reorder /
        // CSE concern when the window origin is read post-move across iters.
        index_t cur_kv_blk = kv_block_idx_ptr[0];
        auto q_dram_window = make_tile_window(q_dram_block_window_tmp.get_bottom_tensor_view(),
                                              q_dram_block_window_tmp.get_window_lengths(),
                                              q_dram_block_window_tmp.get_window_origin(),
                                              Policy::template MakeQRegTileDistribution<Problem>());
        q_dram_window.init_raw();

        // TODO: we use async Copy for K, which is inline asm
        // a side effect is we have to use inline asm for q as well
        auto q = decltype(load_tile(q_dram_window)){};
        // TODO: start from rocm-6.2, compiler will have problem if manually set clear of q.
        // however, q would be cleared in the constructor of static distributed tensor
        // set_tile(q, number<0>{}); // use per-dword clear to avoid scratch
        load_tile_raw(q, q_dram_window);
        __builtin_amdgcn_sched_barrier(0);

        using SaccBlockTileType = decltype(gemm_0.MakeCBlockTile());
        auto s_acc              = SaccBlockTileType{};

        // reduction function for softmax
        const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };
        const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

        // infer Sacc, S, P, M, L, Oacc type
        using SBlockTileType = decltype(cast_tile<SMPLComputeDataType>(s_acc));

        using MLBlockTileType = decltype(block_tile_reduce<SMPLComputeDataType>(
            SBlockTileType{}, sequence<1>{}, f_max, SMPLComputeDataType{0}));

        using OaccBlockTileType = decltype(gemm_1.MakeCBlockTile());

        // init Oacc, M, L
        auto o_acc = OaccBlockTileType{};
        auto m     = MLBlockTileType{};
        auto l     = MLBlockTileType{};

        clear_tile(o_acc);
        set_tile(m, -numeric<SMPLComputeDataType>::infinity());
        clear_tile(l);

        __builtin_amdgcn_sched_barrier(0);
        const auto q_origin       = q_dram_window.get_window_origin();
        const auto num_total_loop = kv_blocks;

        // check early exit if no work to do
        if constexpr(FmhaMask::IsMasking || kPadSeqLenK)
        {
            if(num_total_loop <= 0)
            {
                buffer_load_fence(0); // rocm-6.1, if whole tile is masked out, need to fence(0)
                                      // otherwise will have compute error(maybe compiler bug?)

                // Note: here occ are all cleard, return it
                return o_acc;
            }
            __builtin_amdgcn_sched_barrier(0); // make sure sched_barrier(0) for this check
        }

        auto k_dram_block_window =
            make_tile_window(k_dram_block_window_tmp.get_bottom_tensor_view(),
                             k_dram_block_window_tmp.get_window_lengths(),
                             {seqlen_k_start, 0});

        auto k_dram_window = make_tile_window(
            k_dram_block_window.get_bottom_tensor_view(),
            k_dram_block_window.get_window_lengths(),
            k_dram_block_window.get_window_origin(),
            Policy::template MakeKDramTileDistribution<Problem>()); // K DRAM tile window for
                                                                    // load
        k_dram_window.init_raw();
        constexpr auto k_oob_ck = bool_constant<true>{};
        constexpr auto k_pre_np = bool_constant<false>{};
        auto v_dram_window =
            make_tile_window(v_dram_block_window_tmp.get_bottom_tensor_view(),
                             v_dram_block_window_tmp.get_window_lengths(),
                             {0, seqlen_k_start}, // TODO: hdim split?
                             Policy::template MakeVDramTileDistribution<Problem>());

        // prefetch K tile
        async_load_tile_raw(
            k_lds_store(LdsSeq.at(number<0>{})), k_dram_window, number<-1>{}, k_oob_ck, k_pre_np);
        move_tile_window(k_dram_window, {0, kK0});
        __builtin_amdgcn_sched_barrier(0);

        // buffer_load_fence(k_dram_window.get_num_of_access(), q.get_thread_buffer());
        buffer_load_fence(k_dram_window.get_num_of_access());

        index_t i_total_loops      = 0;
        constexpr index_t k0_loops = kQKHeaddim / kK0;
        constexpr index_t k1_loops = kN0 / kK1;

        static_assert(1 <= k0_loops);
        static_assert(1 <= k1_loops);
        // main loop
        do
        {
            // STAGE 1, QK gemm
            clear_tile(s_acc); // initialize C
            if constexpr(kPerBlockPVSkip)
            {
                // Hoisted vote-slot init; published by the pre-gemm_0 s_barrier below.
                auto* vote_slot   = reinterpret_cast<uint32_t*>(static_cast<char*>(smem_ptr) +
                                                              GetPerBlockVoteSlotOffset());
                const int lane_id = threadIdx.x % warpSize;
                const int warp_id = threadIdx.x / warpSize;
                if(warp_id == 0 && lane_id == 0)
                {
                    *vote_slot = 1u;
                }
            }
            if constexpr(k0_loops > 1)
            {
                static_for<0, k0_loops - 1, 1>{}([&](auto i_k0) {
                    async_load_tile_raw(k_lds_store(number<LdsSeq.at(number<i_k0 + 1>{})>{}),
                                        k_dram_window,
                                        number<-1>{},
                                        k_oob_ck,
                                        k_pre_np);
                    if constexpr(i_k0 < k0_loops - 1)
                        move_tile_window(k_dram_window, {0, kK0});

                    async_load_fence(k_dram_window.get_num_of_access());
                    __builtin_amdgcn_s_barrier();
                    __builtin_amdgcn_sched_barrier(0);
                    gemm_0(s_acc,
                           get_slice_tile(
                               q, sequence<0, i_k0 * kK0>{}, sequence<kM0, (i_k0 + 1) * kK0>{}),
                           get_slice_tile(k_lds_load,
                                          sequence<(LdsSeq.at(number<i_k0>{})) * kN0, 0>{},
                                          sequence<(LdsSeq.at(number<i_k0>{}) + 1) * kN0, kK0>{}));
                });
            }

            // TODO: this to fix a bug when loop smaller than 2,
            // the following fence/barrier will be scheduled inside 1st loop
            if constexpr(k0_loops <= 2)
                __builtin_amdgcn_sched_barrier(0);

            async_load_fence();
            __builtin_amdgcn_s_barrier();

            int block_idx = kv_block_idx_ptr[i_total_loops + 1];
            auto v_buf    = load_tile(v_dram_window, number<-1>{}, bool_constant<false>{});
            __builtin_amdgcn_sched_barrier(0);
            { // tail
                gemm_0(
                    s_acc,
                    get_slice_tile(
                        q, sequence<0, (k0_loops - 1) * kK0>{}, sequence<kM0, k0_loops * kK0>{}),
                    get_slice_tile(k_lds_load,
                                   sequence<(LdsSeq.at(number<k0_loops - 1>{})) * kN0, 0>{},
                                   sequence<(LdsSeq.at(number<k0_loops - 1>{}) + 1) * kN0, kK0>{}));
            }
            __builtin_amdgcn_sched_barrier(1);

            // STAGE 2, scale_s, mask, softmax (no bias/soft-cap)
            // Under int8 GEMM0 the s_acc tile is int32; cast to fp32 so
            // downstream softmax / mask / reduce sees fp32. Under fp16 GEMM0
            // (NO_SCALE / sparge baseline) SaccBlockTileType is already fp32
            // and cast_tile is a no-op type conversion.
            //
            // Per-block dequant (sage Option A): one fp32 scalar
            //   combined = q_descale_value * k_descale
            // multiplies the entire s_acc tile. q_descale_value is loaded
            // ONCE kernel-side (Q-block id is constant per workgroup);
            // k_descale is loaded per K-loop iter from k_descale_ptr at the
            // current K-block boundary. scale_s is kept separate (applied via
            // the FAST_EXP2 fold below or the !FAST_EXP2 explicit multiply).
            auto s_acc_fp32 = cast_tile<SMPLComputeDataType>(s_acc);
            if constexpr(std::is_same_v<QDataType, int8_t> && std::is_same_v<KDataType, int8_t>)
            {
                // Sparge K-blocks are picked sparsely (LUT-driven). Use the
                // explicit cur_kv_blk register accumulator (seeded from
                // LUT[0], bumped by each LUT delta after K move_tile_window)
                // rather than reading k_dram_block_window.get_window_origin()
                // here — defensive against any compiler reorder / CSE of the
                // window-origin read across iters. cur_kv_blk already holds
                // the absolute K-block index (kBlockScaleSizeK == kN0, so
                // the legacy origin/kBlockScaleSizeK divide collapsed to the
                // block id — no further divide needed here).
                const index_t kv_idx  = cur_kv_blk;
                const float k_descale = k_descale_ptr[kv_idx];
                const float combined  = q_descale_value * k_descale;
                tile_elementwise_inout([combined](auto& x) { x = x * combined; }, s_acc_fp32);
            }
#if !CK_TILE_FMHA_FWD_FAST_EXP2
            tile_elementwise_inout([scale_s](auto& x) { x = x * scale_s; }, s_acc_fp32);
#endif
            if constexpr(kPadSeqLenK || FmhaMask::IsMasking)
            {
                const auto k_origin      = k_dram_block_window.get_window_origin();
                bool need_perpixel_check = mask.IsEdgeTile(q_origin.at(number<0>{}),
                                                           k_origin.at(number<0>{}),
                                                           number<kM0>{},
                                                           number<kN0>{});

                if(need_perpixel_check)
                {
                    set_tile_if(
                        s_acc_fp32, -numeric<SMPLComputeDataType>::infinity(), [&](auto tile_idx) {
                            const auto row = q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
                            const auto col = k_origin.at(number<0>{}) + tile_idx.at(number<1>{});
                            return !variant.LogitsMask(variant_params,
                                                       block_indices.batch_idx,
                                                       row,
                                                       col,
                                                       block_indices.qo_head_idx,
                                                       block_indices.kv_head_idx);
                        });
                }
            }

            const auto& s = s_acc_fp32; // S{j}
            auto m_local  = block_tile_reduce<SMPLComputeDataType>(
                s,
                sequence<1>{},
                f_max,
                -numeric<SMPLComputeDataType>::infinity()); // m_local = rowmax(S{j})
            block_tile_reduce_sync(m_local, f_max, bool_constant<false>{});

            const auto m_old = m; // m{j-1}
            tile_elementwise_inout(
                [](auto& e0, auto e1, auto e2) { e0 = max(e1, e2); }, m, m_old, m_local); // m{j}

            auto p_compute = make_static_distributed_tensor<SMPLComputeDataType>(
                s.get_tile_distribution()); // Pcompute{j}

            __builtin_amdgcn_sched_barrier(0x7F);
            // Ensure gemm_0's LDS reads (K tile) from all threads are completed before V store
            // Only needed when K tail and V use the same LDS buffer
            if constexpr(LdsSeq.at(number<k0_loops - 1>{}) == LdsSeq.at(number<k0_loops>{}))
            {
                __builtin_amdgcn_s_barrier();
            }
            // store & prefetch next v, after the max reduction.
            // INVARIANT: V->LDS store and the next-V DRAM load are
            // UNCONDITIONAL — per-warp PV-skip cannot gate them (cross-warp
            // shared LDS state).
            if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
            {
                auto v_shuffle_tmp = make_static_distributed_tensor<VDataType>(
                    Policy::template MakeShuffledVRegBlockDescriptor<Problem>());
                shuffle_tile(v_shuffle_tmp, v_buf);

                auto v_lds_window_tmp =
                    get_slice_tile(v_lds_window,
                                   sequence<(LdsSeq.at(number<k0_loops>{})) * kN1, 0>{},
                                   sequence<(LdsSeq.at(number<k0_loops>{}) + 1) * kN1, kK1>{});

                store_tile(v_lds_window_tmp, v_shuffle_tmp);
            }
            else
            {
                auto v_lds_window_tmp =
                    get_slice_tile(v_lds_window,
                                   sequence<(LdsSeq.at(number<k0_loops>{})) * kN1, 0>{},
                                   sequence<(LdsSeq.at(number<k0_loops>{}) + 1) * kN1, kK1>{});
                store_tile(v_lds_window_tmp, v_buf);
            }

            if constexpr(k1_loops > 1)
            {
                move_tile_window(
                    v_dram_window,
                    {0, kK1}); // will have scratch if move this right after load_tile(v_dram)...
                v_buf = load_tile(
                    v_dram_window, number<-1>{}, bool_constant<false>{}); // load next v_buf
            }
            __builtin_amdgcn_sched_barrier(0);

            // PV-SKIP per Q-tile (SpargeAttn §4.4).
            // Per-warp predicate gates ONLY the per-row, VGPR-private work
            // (exp2 -> p_compute, rowsum, l += rowsum_p). INVARIANT: V load /
            // V->LDS store / gemm_1 / every s_barrier / block_sync_lds stay
            // UNCONDITIONAL (cross-warp LDS dep). On warp_skip we zero this
            // warp's owned rows of p_compute so the unconditional gemm_1
            // contributes 0 to o_acc. alpha-rescale (l *= tmp, o *= tmp) still
            // applies.
            // Skip iff: scale_s * (m_local - m_old) + pv_threshold <= 0
            // (m_local / m_old are warp-uniform after block_tile_reduce_sync).
            // Per-warp PV-skip predicate. Only declared when kEnablePVSkip;
            // wrapped in a lambda so the false instantiation contains nothing.
            auto compute_warp_skip = [&]() {
                if constexpr(kEnablePVSkip)
                {
                    // C3-lite scalar fast-path: pv_threshold == +1e30 sentinel
                    // disables skip; runtime cost is a single sgpr branch.
                    if(pv_threshold >= 1e29f)
                        return false;
                    // Per-row predicate: warp-AND over rows this warp owns.
                    int warp_skip_int      = 1;
                    constexpr auto m_spans = decltype(m_local)::get_distributed_spans();
                    sweep_tile_span(m_spans[number<0>{}], [&](auto idx0) {
                        constexpr auto i_idx = make_tuple(idx0);
                        const float diff     = scale_s * (static_cast<float>(m_local[i_idx]) -
                                                      static_cast<float>(m_old[i_idx]));
                        if(!(diff + pv_threshold <= 0.0f))
                            warp_skip_int = 0;
                    });
                    // Warp-level AND reduce (wave=64 on gfx942; xor butterfly).
                    // No LDS, no s_barrier, no cross-warp dependency.
                    warp_skip_int &= __shfl_xor(warp_skip_int, 32);
                    warp_skip_int &= __shfl_xor(warp_skip_int, 16);
                    warp_skip_int &= __shfl_xor(warp_skip_int, 8);
                    warp_skip_int &= __shfl_xor(warp_skip_int, 4);
                    warp_skip_int &= __shfl_xor(warp_skip_int, 2);
                    warp_skip_int &= __shfl_xor(warp_skip_int, 1);
                    return warp_skip_int != 0;
                }
                else
                {
                    return false;
                }
            };
            const bool warp_skip = compute_warp_skip();

            // Per-block PV-skip — block-wide AND vote over warp_skip.
            // Sentinel init is hoisted to top-of-loop (hidden by pre-gemm_0 s_barrier);
            // here we only atomicAnd + consensus barrier + broadcast.
            bool block_skip = false;
            if constexpr(kPerBlockPVSkip)
            {
                auto* vote_slot   = reinterpret_cast<uint32_t*>(static_cast<char*>(smem_ptr) +
                                                              GetPerBlockVoteSlotOffset());
                const int lane_id = threadIdx.x % warpSize;

                if(lane_id == 0)
                {
                    atomicAnd(vote_slot, warp_skip ? 1u : 0u);
                }
                block_sync_lds();

                const uint32_t consensus = *vote_slot;
                block_skip               = (consensus != 0u);
            }

            static const auto get_validated_m = [](SMPLComputeDataType raw_m) {
                if constexpr(FmhaMask::IsMasking)
                {
                    return raw_m == -numeric<SMPLComputeDataType>::infinity()
                               ? type_convert<SMPLComputeDataType>(0.f)
                               : raw_m;
                }
                else
                {
                    return raw_m;
                }
            };

            // exp2 -> p_compute and rowsum_p.
            // kEnablePVSkip + warp_skip: zero this warp's owned rows of
            // p_compute so the unconditional gemm_1 contributes zero to o_acc,
            // and skip the rowsum.
            // Per-block mode uses block_skip (uniform across waves) and also
            // skips gemm_1 itself (guard at the gemm_1 site below). The
            // p_compute zeroing remains so rowsum_p -> 0 and `l += rowsum_p`
            // is a no-op.
            constexpr auto p_spans = decltype(p_compute)::get_distributed_spans();
            sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
#if CK_TILE_FMHA_FWD_FAST_EXP2
                auto row_max = scale_s * get_validated_m(m[i_idx]);
#endif
                sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    if constexpr(kPerBlockPVSkip)
                    {
                        if(block_skip)
                        {
                            p_compute(i_j_idx) = SMPLComputeDataType{0};
                            return;
                        }
                    }
                    else if constexpr(kEnablePVSkip)
                    {
                        if(warp_skip)
                        {
                            p_compute(i_j_idx) = SMPLComputeDataType{0};
                            return;
                        }
                    }
#if CK_TILE_FMHA_FWD_FAST_EXP2
                    p_compute(i_j_idx) = exp2(scale_s * s[i_j_idx] - row_max);
#else
                    p_compute(i_j_idx) = exp(s[i_j_idx] - get_validated_m(m[i_idx]));
#endif
                });
            });

            auto rowsum_p = block_tile_reduce<SMPLComputeDataType>(
                p_compute, sequence<1>{}, f_sum, SMPLComputeDataType{0}); // rowsum(Pcompute{j})

            block_tile_reduce_sync(rowsum_p, f_sum, bool_constant<false>{});

            // l{j}, Oacc{j}: alpha rescale of l / o always runs.
            //                When warp_skip, rowsum_p is already 0 for this
            //                warp's owned rows (p_compute zeroed above), so
            //                `l += rowsum_p` is a no-op — no extra branch needed.
            constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();
            sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
#if CK_TILE_FMHA_FWD_FAST_EXP2
                const auto tmp = [&]() {
                    auto row_max = scale_s * get_validated_m(m[i_idx]);
                    return exp2(scale_s * m_old[i_idx] - row_max);
                }();
#else
                const auto tmp = exp(m_old[i_idx] - get_validated_m(m[i_idx]));
#endif
                l(i_idx) = tmp * l[i_idx] + rowsum_p[i_idx];
                sweep_tile_span(o_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    // FIXME: this use different equation from FA v2 paper,
                    // but produce correc result.
                    // Is the equation wrong?
                    o_acc(i_j_idx) *= tmp;
                });
            });

            const auto p = [&]() {
                if constexpr(std::is_same_v<PDataType, fp16_t>)
                    return impl::cast_tile_pkrtz_fp16_fp32<PDataType>(p_compute);
                else
                    return cast_tile<PDataType>(p_compute);
            }();

            // STAGE 3, KV gemm — always runs (block-wide LDS dep; per-warp
            // skipping has been absorbed by zeroing p_compute rows above).
            {
                if constexpr(k1_loops > 1)
                {
                    static_for<0, k1_loops - 1, 1>{}([&](auto i_k1) {
                        if constexpr(i_k1 != 0 && i_k1 < k1_loops - 1)
                        {
                            v_buf = load_tile(v_dram_window,
                                              number<-1>{},
                                              bool_constant<false>{}); // load next v_buf
                        }
                        // block_sync_lds() stays UNCONDITIONAL — it is the
                        // workgroup barrier the V->LDS rotation chain requires
                        // (idiom catalog §3.1 / §4.1). Only the gemm_1 MFMA is
                        // gated on block_skip when in per-block mode.
                        block_sync_lds();
                        if constexpr(kPerBlockPVSkip)
                        {
                            if(!block_skip)
                            {
                                gemm_1(
                                    o_acc,
                                    get_slice_tile(p,
                                                   sequence<0, i_k1 * kK1>{},
                                                   sequence<kM0, (i_k1 + 1) * kK1>{}),
                                    get_slice_tile(
                                        v_lds_window,
                                        sequence<(LdsSeq.at(number<k0_loops + i_k1>{})) * kN1, 0>{},
                                        sequence<(LdsSeq.at(number<k0_loops + i_k1>{}) + 1) * kN1,
                                                 kK1>{}));
                            }
                        }
                        else
                        {
                            gemm_1(o_acc,
                                   get_slice_tile(p,
                                                  sequence<0, i_k1 * kK1>{},
                                                  sequence<kM0, (i_k1 + 1) * kK1>{}),
                                   get_slice_tile(
                                       v_lds_window,
                                       sequence<(LdsSeq.at(number<k0_loops + i_k1>{})) * kN1, 0>{},
                                       sequence<(LdsSeq.at(number<k0_loops + i_k1>{}) + 1) * kN1,
                                                kK1>{}));
                        }

                        if constexpr(std::is_same_v<VLayout,
                                                    ck_tile::tensor_layout::gemm::RowMajor>)
                        {
                            auto v_shuffle_tmp = make_static_distributed_tensor<VDataType>(
                                Policy::template MakeShuffledVRegBlockDescriptor<Problem>());
                            shuffle_tile(v_shuffle_tmp, v_buf);
                            auto v_lds_window_tmp = get_slice_tile(
                                v_lds_window,
                                sequence<(LdsSeq.at(number<k0_loops + i_k1 + 1>{})) * kN1, 0>{},
                                sequence<(LdsSeq.at(number<k0_loops + i_k1 + 1>{}) + 1) * kN1,
                                         kK1>{});
                            store_tile(v_lds_window_tmp, v_shuffle_tmp);
                        }
                        else
                        {
                            auto v_lds_window_tmp = get_slice_tile(
                                v_lds_window,
                                sequence<(LdsSeq.at(number<k0_loops + i_k1 + 1>{})) * kN1, 0>{},
                                sequence<(LdsSeq.at(number<k0_loops + i_k1 + 1>{}) + 1) * kN1,
                                         kK1>{});
                            store_tile(v_lds_window_tmp, v_buf);
                        }
                        if constexpr(i_k1 < k1_loops - 1)
                            move_tile_window(v_dram_window, {0, kK1});
                    });
                }
            }
            i_total_loops++;
            if(i_total_loops < num_total_loop)
            {
                // V load runs unconditionally under redesign D, so no skip
                // compensation needed (same offset arithmetic as _vsa.hpp).
                move_tile_window(v_dram_window, {0, kN0 * (block_idx - 1)});
                move_tile_window(k_dram_block_window, {kN0 * block_idx, 0});
                // Mirror the LUT delta advance on the per-block-scale index
                // accumulator so the next iter's dequant sees the correct
                // absolute K-block id (see comment at decl of cur_kv_blk).
                cur_kv_blk += block_idx;
                k_dram_window.set_window_origin(k_dram_block_window.get_window_origin());

                if constexpr(k1_loops >= 2 &&
                             LdsSeq.at(number<0>{}) == LdsSeq.at(number<k0_loops + k1_loops - 2>{}))
                    __builtin_amdgcn_s_barrier();
                async_load_tile_raw(k_lds_store(LdsSeq.at(number<0>{})),
                                    k_dram_window,
                                    number<-1>{},
                                    k_oob_ck,
                                    k_pre_np);
                move_tile_window(k_dram_window, {0, kK0});
            }
            // tail — gemm_1 runs unconditionally under redesign D (per-wave).
            // Per-block mode gates the MFMA on block_skip; block_sync_lds
            // still runs UNCONDITIONALLY (workgroup barrier for LDS rotation).
            {
                block_sync_lds();
                if constexpr(kPerBlockPVSkip)
                {
                    if(!block_skip)
                    {
                        gemm_1(
                            o_acc,
                            get_slice_tile(
                                p, sequence<0, (k1_loops - 1) * kK1>{}, sequence<kM0, kN0>{}),
                            get_slice_tile(
                                v_lds_window,
                                sequence<(LdsSeq.at(number<k0_loops + k1_loops - 1>{})) * kN1, 0>{},
                                sequence<(LdsSeq.at(number<k0_loops + k1_loops - 1>{}) + 1) * kN1,
                                         kK1>{}));
                    }
                }
                else
                {
                    gemm_1(o_acc,
                           get_slice_tile(
                               p, sequence<0, (k1_loops - 1) * kK1>{}, sequence<kM0, kN0>{}),
                           get_slice_tile(
                               v_lds_window,
                               sequence<(LdsSeq.at(number<k0_loops + k1_loops - 1>{})) * kN1, 0>{},
                               sequence<(LdsSeq.at(number<k0_loops + k1_loops - 1>{}) + 1) * kN1,
                                        kK1>{}));
                }
            }
        } while(i_total_loops < num_total_loop);

        // finally, O
        constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();

        sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            const auto tmp       = [&]() {
                if constexpr(FmhaMask::IsMasking)
                {
                    return l[i_idx] == 0.f ? 0.f : 1 / l[i_idx];
                }
                else
                    return 1 / l[i_idx];
            }();
            sweep_tile_span(o_spans[number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                o_acc(i_j_idx) *= tmp;
            });
        });

        return o_acc;
    }
};

} // namespace ck_tile
