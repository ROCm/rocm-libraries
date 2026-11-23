// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/block_attention_quant_scale_enum.hpp"
#include "ck_tile/ops/fmha/block/block_masking.hpp"
#include "ck_tile/ops/fmha/detail/fmha_batch_prefill_page_tile.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_executor.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_output.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_softmax.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_batch_prefill_pipeline_qr_ks_vs_tdm_sched_page_tile_policy.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"

namespace ck_tile {

// Batch-prefill pipeline of the gfx1250 page tile: the scheduled-TDM QR loop (M128 or M64 query
// tiles, N128 KV tiles) over a paged KV cache whose physical page is independent of the KV tile.
// K and V of a logical N128 tile share its page IDs (P>=128 a slice of one page, P64 two pages,
// one per pair of waves, P16 eight page-local fragments); the next IDs are fetched before their TDM
// descriptors are built. The per-head sink logit (virtual sink) is merged at normalization in sink
// instances only.
template <typename Problem_>
struct BlockFmhaBatchPrefillPipelineQRKSVSTdmSchedPageTile
{
    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};

    using Problem               = remove_cvref_t<Problem_>;
    using Policy                = FmhaBatchPrefillPageTilePolicyFor<Problem>;
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
    using SplitSoftmax =
        FmhaTdmSchedSplitSoftmaxFor<Policy::Geometry::kN / Policy::kScoreGroupColumns,
                                    Policy::Geometry::kQkMIter>;
    using Schedule = typename Policy::Schedule;
    using Mode     = FmhaTdmSchedMode<Problem, Policy>;
    static_assert(Mode::kIsLegal,
                  "qr_tdm_sched schedule geometry or output-rescale placement is incompatible");

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
    static constexpr index_t kNWarp        = BlockFmhaShape::Gemm0BlockWarps::at(I1);

    static constexpr bool kIsGroupMode       = Problem::kIsGroupMode;
    static constexpr bool kPadSeqLenQ        = Problem::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK        = Problem::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ       = Problem::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV       = Problem::kPadHeadDimV;
    static constexpr bool kHasLogitsSoftCap  = Problem::kHasLogitsSoftCap;
    static constexpr bool kHasDropout        = Problem::kHasDropout;
    static constexpr auto BiasEnum           = Problem::BiasEnum;
    static constexpr bool kStoreLSE          = Problem::kStoreLSE;
    static constexpr bool kHasUnevenSplits   = true;
    static constexpr bool kHasSink           = Problem::kHasSink;
    static constexpr bool kEnableVirtualSink = kHasSink;
    static constexpr auto QScaleEnum         = Problem::QScaleEnum;

    static_assert(CK_TILE_FMHA_FWD_FAST_EXP2,
                  "qr_tdm_sched: every softmax site calls exp2, so log2(e) must be folded into "
                  "scale_s");
    static_assert(
        CK_TILE_FMHA_PAGE_TILE_EXPERT_SCHEDULING,
        "the page tile is built with -mllvm -amdgpu-expert-scheduling-mode: add it and "
        "-DCK_TILE_FMHA_PAGE_TILE_EXPERT_SCHEDULING=1 (fmha_batch_prefill_page_tile.hpp)");
    // Bias and ALiBi scale the scores into the exp2 domain before the softmax (process_score), so
    // the softmax then runs with a unit scale.
    static constexpr bool kScoresInLog2Domain = BiasEnum != BlockAttentionBiasEnum::NO_BIAS;
    static_assert(!kHasLogitsSoftCap || BiasEnum == BlockAttentionBiasEnum::NO_BIAS,
                  "the logits soft cap applies to unbiased scores");
    static_assert(!kHasDropout, "qr_tdm_sched does not support dropout");
    static_assert(QScaleEnum == BlockAttentionQuantScaleEnum::NO_SCALE,
                  "qr_tdm_sched does not implement quantized attention yet");

    // last dimension vector length used to create tensor view(and decide buffer_load vector length)
    // ... together with tensor distribution. tensor dist should able to overwrite this
    static constexpr index_t kAlignmentQ = Policy::template GetAlignmentQ<Problem>();
    static constexpr index_t kAlignmentK = Policy::template GetAlignmentK<Problem>();
    static constexpr index_t kAlignmentV = Policy::template GetAlignmentV<Problem>();
    static constexpr index_t kAlignmentO = Policy::template GetAlignmentO<Problem>();
    static constexpr index_t kAlignmentBias =
        kPadSeqLenK ? 1 : Policy::template GetAlignmentBias<Problem>();
    static constexpr index_t kAlignmentRandVal =
        kPadSeqLenK ? 1 : Policy::template GetAlignmentRandVal<Problem>();

    static_assert(Policy::template IsSupportedProblem<Problem>(),
                  "qr_tdm_sched requires an exact supported FP16/BF16 geometry");

    // IsSupportedProblem above pins Geometry == FmhaTdmSchedGeometryFor<Problem>.
    using Geometry = typename Policy::Geometry;
    // Partial-count barriers with the workgroup LDS fences (fmha_page_tile_fenced_barriers_v).
    static constexpr bool kFencedBarriers =
        detail::fmha_page_tile_fenced_barriers_v<Problem, Geometry>;
    static_assert(Geometry::kM == kM0 && Geometry::kN == kN0 && Geometry::kDv == kN1);

    // Page transfer budget. The physical page size selects how many native tensor
    // issues one wave makes per K/V load group: one for P>=64, two for P16.
    // This budget is page-specific; the shared policy's 0/1 tensor counts do not apply to page
    // transfers.
    using PageTileTraits = FmhaBatchPrefillPageTileTraits<Problem::kPageBlockSize, kQKHeaddim>;
    using PageIds        = FmhaBatchPrefillPageTileIds<Problem::kPageBlockSize>;
    struct PagedTransfer
    {
        static constexpr index_t kKPrefetchTensorCount = PageTileTraits::kKPrefetchTensorCount;
        static constexpr index_t kVPrefetchTensorCount = PageTileTraits::kVPrefetchTensorCount;
        static constexpr index_t kFragmentsPerWave     = PageTileTraits::kFragmentsPerWave;
    };
    static_assert(PagedTransfer::kFragmentsPerWave == 1 || PagedTransfer::kFragmentsPerWave == 2);
    // The page tile writer is a workgroup collective over its own wave count.
    static_assert(PageTileTraits::kWaves == Geometry::kWaves,
                  "page tile writer waves must equal the pipeline's waves");
    // PagedTransfer is the only tensor-issue budget this core waits on. The shared policy's
    // 0/1 counts and tail-drain flag are still what enables the schedule's epilogue tensor
    // drain (fmha_tdm_sched_schedule.hpp RunEpilogue: drain iff (Policy K or V count != 0)
    // and kPrefetchTailDrain). Whenever page transfers can be in flight, that drain must
    // therefore be enabled.
    static_assert((PagedTransfer::kKPrefetchTensorCount == 0 &&
                   PagedTransfer::kVPrefetchTensorCount == 0) ||
                      ((Policy::kKPrefetchTensorCount != 0 || Policy::kVPrefetchTensorCount != 0) &&
                       Policy::kPrefetchTailDrain),
                  "page transfers in flight but the shared policy disables the epilogue drain");

    static constexpr index_t kBlockPerCu = Policy::kBlockPerCu;

    // Scope of the page tile.
    static_assert(Problem::kIsGroupMode);
    // Linear pages, or the vectorized layout with 8 x 16-bit vectors (the window issue lands both
    // in the same LDS image).
    static_assert(Problem::kKVMemoryLayout ==
                      BlockAttentionKVCacheMemoryLayoutEnum::LINEAR_LAYOUT ||
                  (Problem::kKVMemoryLayout ==
                       BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT &&
                   Problem::kVectorSize == 8));
    static_assert(Problem::kKVLookupTable ==
                      BlockAttentionKVCacheLookupTableEnum::VLLM_BLOCK_TABLE_2D ||
                  Problem::kKVLookupTable ==
                      BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D);
    static_assert(Problem::kPadSeqLenQ && Problem::kPadSeqLenK);
    static_assert(!Problem::kPadHeadDimQ && !Problem::kPadHeadDimV);
    // BF16 or FP16, the same type for Q, K and V.
    static_assert((std::is_same_v<QDataType, bf16_t> || std::is_same_v<QDataType, half_t>) &&
                  std::is_same_v<KDataType, QDataType> && std::is_same_v<VDataType, QDataType>);
    static_assert((kM0 == 128 || kM0 == 64) && kN0 == 128);
    // M64 tiles (small grids, and the KV split, which also stores the LSE) are validated for the
    // vectorized layout with P64/P128 pages, without sink, bias or logits soft cap; the other cases
    // use M128 tiles.
    static_assert(kM0 == 128 ||
                      (Problem::kKVMemoryLayout ==
                           BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT &&
                       (Problem::kPageBlockSize == 64 || Problem::kPageBlockSize == 128) &&
                       !Problem::kHasSink && Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS &&
                       !Problem::kHasLogitsSoftCap),
                  "M64 page tiles: vectorized KV, P64/P128, no sink, bias or soft cap");
    // D64/V64 and D128/V128 are the supported head dimensions; the physical WMMA shape stays
    // 16x16x32 and is not the head width.
    static_assert((kQKHeaddim == 64 && kN1 == 64) || (kQKHeaddim == 128 && kN1 == 128),
                  "the batch-prefill page tile supports D64/V64 or D128/V128");
    static_assert(kBlockPerCu == 1);
    // K and V reach LDS by asynchronous (TDM) copies only.
    static_assert(Policy::AsyncCopy);

    // Read by the batch-prefill kernels.
    static constexpr index_t kPageBlockSize = Problem::kPageBlockSize;
    static constexpr index_t kVectorSize    = Problem::kVectorSize;
    static constexpr auto kKVMemoryLayout   = Problem::kKVMemoryLayout;
    static constexpr const char* name       = "qr_tdm_sched_page_tile";

    // Lean TDM issue: see Policy::UseLeanTdmIssue.
    static constexpr bool kLeanIssue = Policy::template UseLeanTdmIssue<Problem>();

    // Prologue order. Q is staged in the V1 buffer so the K0/V0 loads (which do not depend on Q)
    // are issued right after it, and the Q and K/V load latencies overlap instead of adding up.
    // Without it Q is staged in K0 and must be read out before K0 is loaded. Whole-tile pages
    // only: for P16 the extra live state raises SGPR spills.
    static constexpr bool kEarlyKvPrologue = PagedTransfer::kFragmentsPerWave == 1;

    // TDM/LDS ordering workaround (Policy::kDrainLdsBeforeTdmIssue): the barrier after
    // the K prefetch issue, which publishes the V tile (and the prologue's K0/V0), also waits
    // for the wave's own LDS reads.
    static constexpr index_t kKPrefetchBarrierDsCount =
        Policy::kDrainLdsBeforeTdmIssue ? 0 : waitcnt_arg::kMaxLgkmCnt;

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
    using QKBlockGemm = remove_cvref_t<decltype(Policy::template GetQKBlockGemm<Problem>())>;
    using PVBlockGemm = remove_cvref_t<decltype(Policy::template GetPVBlockGemm<Problem>())>;
    using ScoreTile   = decltype(QKBlockGemm::MakeCBlockTile());
    static_assert(ScoreTile::get_thread_buffer_size() == SplitSoftmax::Mapping::kThreadBufferSize);
    // Both WarpGemm implementations expose the physical instruction shape here.
    using QkMmaShape = typename QKBlockGemm::WarpGemm::WarpGemmAttribute::Impl;
    using PvMmaShape = typename PVBlockGemm::WarpGemm::WarpGemmAttribute::Impl;
    static_assert(QkMmaShape::kM == Geometry::QkWmma::kM &&
                      QkMmaShape::kN == Geometry::QkWmma::kN &&
                      QkMmaShape::kK == Geometry::QkWmma::kK,
                  "gemm_0 dispatched a different physical WMMA shape");
    static_assert(PvMmaShape::kM == Geometry::PvWmma::kM &&
                      PvMmaShape::kN == Geometry::PvWmma::kN &&
                      PvMmaShape::kK == Geometry::PvWmma::kK,
                  "gemm_1 dispatched a different physical WMMA shape");
#endif

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        static_assert(Policy::template GetSmemSize<Problem>() == Policy::GetLdsArenaSize());
        // A separate aligned slot after the arena stores the pending page-table ID metadata:
        // 32 bytes for P16 (up to eight owned ids), 16 bytes otherwise. The kernel places it and
        // passes its address (page_id_slot).
        return Policy::GetLdsArenaSize() + PageTileTraits::kMetadataBytes;
    }

    // Re-pack gemm_0 C into gemm_1 A: C is M-outer (MIter,KIter), A is K-outer.
    // Lanes already align (NWarp==1), so this is an in-thread block reorder;
    // identity when MIterPerWarp==1.
    template <typename Gemm1, typename PComputeTensor>
    CK_TILE_DEVICE static auto MakePForGemm1(const PComputeTensor& p_compute)
    {
        static_assert(Geometry::PRepackLayout::kInThread &&
                          Gemm1::MIterPerWarp == Geometry::kPvMIter,
                      "P re-pack requires gemm_0 C^T and gemm_1 A to share lanes");
        auto p_tile = make_static_distributed_tensor<PDataType>(
            Policy::template MakePRegTileDistribution<Problem>());
        const auto p_src        = cast_tile<PDataType>(p_compute);
        constexpr index_t kPBuf = decltype(p_src)::get_thread_buffer_size();
        constexpr index_t kPMI  = Gemm1::MIterPerWarp;
        constexpr index_t kPKI  = kN0 / Gemm1::WarpGemm::kK;
        constexpr index_t kPSub = kPBuf / (kPMI * kPKI);
        using p_bulk_t          = array<PDataType, kPSub>;
        static_for<0, kPMI, 1>{}([&](auto mi) {
            static_for<0, kPKI, 1>{}([&](auto ki) {
                p_tile.get_thread_buffer().template set_as<p_bulk_t>(
                    number<ki * kPMI + mi>{},
                    p_src.get_thread_buffer().template get_as<p_bulk_t>(number<mi * kPKI + ki>{}));
            });
        });
        return p_tile;
    }

    // Prefill, double lds
    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename LSEaccDramBlockWindowTmp,
              typename PositionEncoding>
    CK_TILE_HOST_DEVICE auto
    run(const QDramBlockWindowTmp& __restrict__ q_dram_block_window_tmp,       // M0*K0 tile
        const KDramBlockWindowTmp& __restrict__ k_dram_block_window_tmp,       // N0*K0 tile
        const VDramBlockWindowTmp& __restrict__ v_dram_block_window_tmp,       // N1*K1 tile
        const BiasDramBlockWindowTmp& __restrict__ bias_dram_block_window_tmp, // M0*N0 tile
        LSEaccDramBlockWindowTmp& __restrict__ lse_acc_dram_window_tmp,        // M0*1 tile
        FmhaMask mask,
        PositionEncoding position_encoding,
        float scale_s,
        void* __restrict__ smem_ptrk0,
        void* __restrict__ smem_ptrk1,
        void* __restrict__ smem_ptrv0,
        void* __restrict__ smem_ptrv1,
        void* __restrict__ page_id_slot,
        float sink_v,
        float logits_soft_cap) const
    {
        static_assert(
            std::is_same_v<QDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                std::is_same_v<KDataType, remove_cvref_t<typename KDramBlockWindowTmp::DataType>> &&
                std::is_same_v<VDataType, remove_cvref_t<typename VDramBlockWindowTmp::DataType>>,
            "wrong!");

        // Q, K, and V use TDM global-to-LDS transfers; V is consumed through
        // transposed LDS loads.
        static_assert(kM0 == QDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                          kSubQKHeaddim == QDramBlockWindowTmp{}.get_window_lengths()[I1] &&
                          kN0 == KDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                          kK0 == KDramBlockWindowTmp{}.get_window_lengths()[I1] &&
                          kN1 == VDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                          kK1 == VDramBlockWindowTmp{}.get_window_lengths()[I1] &&
                          kM0 == BiasDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                          kN0 == BiasDramBlockWindowTmp{}.get_window_lengths()[I1],
                      "wrong!");
        // Block GEMM
        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1 = Policy::template GetPVBlockGemm<Problem>();

        using SaccBlockTileType = decltype(gemm_0.MakeCBlockTile());
        auto s_acc              = SaccBlockTileType{};

        // reduction function for softmax
        const auto f_max        = [](auto e0, auto e1) { return max(e0, e1); };
        using OaccBlockTileType = decltype(gemm_1.MakeCBlockTile());

        using PvBlockGemm   = remove_cvref_t<decltype(gemm_1)>;
        using PvWarpGemm    = typename PvBlockGemm::WarpGemm;
        using PvCWarpTensor = typename PvWarpGemm::CWarpTensor;
        static_assert(PvBlockGemm::KIterPerWarp == Geometry::kPvStageKIter &&
                      PvBlockGemm::MIterPerWarp == Geometry::kPvMIter &&
                      PvBlockGemm::NIterPerWarp == Geometry::kPvNIter);
        static_assert(std::is_same_v<OaccDataType, float>);
        static_assert(std::is_same_v<remove_cvref_t<typename PvCWarpTensor::DataType>, float>);
        static_assert(std::is_same_v<typename Policy::OutputFragments::Fragment, fp32x8_t>);
        static_assert(PvCWarpTensor::get_thread_buffer_size() ==
                      Policy::OutputFragments::kElementsPerFragment);
        static_assert(Policy::OutputFragments::kNumDmsb == 2 * PvBlockGemm::MIterPerWarp);
        static_assert(Policy::OutputFragments::kNumN == PvBlockGemm::NIterPerWarp / 2);
        static_assert(OaccBlockTileType::get_thread_buffer_size() ==
                      Policy::OutputFragments::Mapping::kThreadBufferSize);
        if constexpr(Geometry::kN == 128 && Geometry::kDv == 128)
            static_assert(std::is_same_v<typename SaccBlockTileType::StaticTileDistribution,
                                         typename OaccBlockTileType::StaticTileDistribution>,
                          "N128 score/output distributions must preserve the fragment "
                          "reconstruction map");

        auto o_acc = Policy::OutputFragments::MakeZero();

        // infer Sacc, S, P, M, L, Oacc type
        using SBlockTileType = decltype(cast_tile<SMPLComputeDataType>(OaccBlockTileType{}));

        using MLBlockTileType = decltype(block_tile_reduce<SMPLComputeDataType>(
            SBlockTileType{}, sequence<1>{}, f_max, SMPLComputeDataType{0}));

        // The shared kernel can supply a sink logit independently of kHasSink,
        // which controls sink tokens in the mask. Preserve this runtime input.
        auto m = MLBlockTileType{};
        auto l = MLBlockTileType{};

        if constexpr(BiasEnum == BlockAttentionBiasEnum::NO_BIAS)
        {
            // A virtual sink has V=0. Merge it only at normalization so its
            // logit cannot change the BF16 rounding of the real-token P operand.
            set_tile(m, -numeric<SMPLComputeDataType>::infinity());
            clear_tile(l);
        }
        else if(__builtin_isinf_sign(sink_v) >= 0)
        {
            set_tile(m, sink_v * C_LOG2E);
            set_tile(l, SMPLComputeDataType{1.0f});
        }
        else
        {
            set_tile(m, -numeric<SMPLComputeDataType>::infinity());
            clear_tile(l);
        }

        const auto q_origin = q_dram_block_window_tmp.get_window_origin();

        // Masked tiles of Policy::UseMaskLoopTrims: the key tiles of this query tile that need no
        // per-pixel mask, as {first, width} (GetInteriorTileRangeForValidRows), formed once for the
        // loop.
        constexpr bool kMaskLoopTrims =
            FmhaMask::IsMasking && Policy::template UseMaskLoopTrims<Problem>();
        [[maybe_unused]] index_t mask_interior_first = 0;
        [[maybe_unused]] index_t mask_interior_width = 0;
        if constexpr(kMaskLoopTrims)
        {
            const auto interior = mask.GetInteriorTileRangeForValidRows(
                q_origin.at(I0), number<kM0>{}, number<kN0>{});
            mask_interior_first = interior.at(I0);
            mask_interior_width = interior.at(I1);
        }

        const auto tile_range_result = [&]() {
            if constexpr(kHasSink)
                return mask.GetSinkTileRangeAlongX(q_origin.at(I0), number<kM0>{}, number<kN0>{});
            else
            {
                auto [start, end] =
                    mask.GetTileRangeAlongX(q_origin.at(I0), number<kM0>{}, number<kN0>{});
                return ck_tile::make_tuple(0, start, end);
            }
        }();
        const auto sink_seq_end           = tile_range_result.get(ck_tile::number<0>{});
        const auto logical_seqlen_k_start = tile_range_result.get(ck_tile::number<1>{});
        const auto logical_seqlen_k_end   = tile_range_result.get(ck_tile::number<2>{});

        const auto num_sink_loop = integer_divide_ceil(sink_seq_end, kN0);

        // check early exit if no work to do
        if constexpr(FmhaMask::IsMasking || kPadSeqLenK || kHasUnevenSplits)
        {
            const index_t logical_num_total_loop =
                integer_divide_ceil(logical_seqlen_k_end - logical_seqlen_k_start, kN0) +
                num_sink_loop;
            if(logical_num_total_loop <= 0)
            {
                if constexpr(kStoreLSE)
                {
                    auto lse_acc =
                        make_static_distributed_tensor<LSEDataType>(m.get_tile_distribution());

                    if(__builtin_isinf_sign(sink_v) >= 0)
                    {
                        // With bias sink_v is the natural sink logit itself.
                        set_tile(
                            lse_acc,
                            SMPLComputeDataType{kScoresInLog2Domain ? sink_v : sink_v * scale_s});
                    }
                    else
                    {
                        set_tile(lse_acc, -numeric<SMPLComputeDataType>::infinity());
                    }

                    store_tile(lse_acc_dram_window_tmp, lse_acc);
                }

                // Note: here occ are all cleard, return it
                // Note: q loaded but no fence, ignore it.
                return Policy::OutputFragments::template Reconstruct<OaccBlockTileType>(o_acc);
            }
        }

        // ---------------------------------------------------------------------
        // TDM configs for Q / K / V
        // pad_enable + pad_amount + pad_interval are compile-time, sourced from
        // the policy. workgroup_mask defaults to 0 (no cluster multicast).
        // V uses load_tile_tdm (single-box plain layout) -- same TDM machinery
        // as Q / K, with V dram dist switched to trivial tile-major and the
        // V LDS read view kept plain row-major (matches the write view).
        // ---------------------------------------------------------------------
        TDMConfig tdm_config_q;
        TDMConfig tdm_config_k;
        TDMConfig tdm_config_v;
        {
            constexpr auto LdsPaddingConfigQ     = Policy::template GetLdsPaddingConfigQ<Problem>();
            tdm_config_q.pad_enable              = LdsPaddingConfigQ[I0];
            tdm_config_q.pad_config.pad_amount   = LdsPaddingConfigQ[I1];
            tdm_config_q.pad_config.pad_interval = LdsPaddingConfigQ[number<2>{}];

            constexpr auto LdsPaddingConfigK     = Policy::template GetLdsPaddingConfigK<Problem>();
            tdm_config_k.pad_enable              = LdsPaddingConfigK[I0];
            tdm_config_k.pad_config.pad_amount   = LdsPaddingConfigK[I1];
            tdm_config_k.pad_config.pad_interval = LdsPaddingConfigK[number<2>{}];

            constexpr auto LdsPaddingConfigV     = Policy::template GetLdsPaddingConfigV<Problem>();
            tdm_config_v.pad_enable              = LdsPaddingConfigV[I0];
            tdm_config_v.pad_config.pad_amount   = LdsPaddingConfigV[I1];
            tdm_config_v.pad_config.pad_interval = LdsPaddingConfigV[number<2>{}];
        }

        // Q tile in LDS. With kEarlyKvPrologue Q lands in the V1 buffer, which is first
        // written by the V prefetch of iteration 0, so the K0/V0 loads can be issued while Q
        // is still in flight.
        auto* q_lds_ptr = static_cast<QDataType*>(kEarlyKvPrologue ? smem_ptrv1 : smem_ptrk0);
        static_assert(!kEarlyKvPrologue ||
                          Policy::template GetSmemSizeQ<Problem>() <= Policy::kVFootprintBytes,
                      "Q does not fit in the V1 buffer");

        auto q_dram_window = make_tile_window(
            q_dram_block_window_tmp, Policy::template MakeQDramTileDistribution<Problem>());

        auto q_lds_write_view = make_tensor_view<address_space_enum::lds>(
            q_lds_ptr, Policy::template MakeQLdsBlockDescriptor<Problem>());

        auto q_lds_read_view = make_tensor_view<address_space_enum::lds>(
            q_lds_ptr, Policy::template MakeQLdsBlockDescriptor<Problem>());

        auto q_lds_store_window =
            make_tile_window(q_lds_write_view,
                             Policy::template MakeQLdsBlockDescriptor<Problem>().get_lengths(),
                             {0, 0});

        auto q_lds_read_window =
            make_tile_window(q_lds_read_view,
                             Policy::template MakeQLdsBlockDescriptor<Problem>().get_lengths(),
                             {0, 0},
                             Policy::template MakeQRegTileDistribution<Problem>());

        load_tile_tdm(tdm_config_q, q_lds_store_window, q_dram_window);
        using QTile  = decltype(load_tile(q_lds_read_window));
        QTile q_tile = [&]() {
            if constexpr(kEarlyKvPrologue)
                return QTile{}; // read after the K/V prologue loads are issued
            else
            {
                s_wait_tensorcnt_barrier<0>();
                return load_tile(q_lds_read_window);
            }
        }();

        // K tile in LDS (sink-aware start)
        const auto kv_load_start =
            (sink_seq_end == 0 && logical_seqlen_k_start > 0) ? logical_seqlen_k_start : 0;
        const index_t physical_seqlen_k_start = logical_seqlen_k_start;
        const index_t physical_seqlen_k_end   = logical_seqlen_k_end;
        // Bias tile window (prefill path)
        const auto bias_origin = bias_dram_block_window_tmp.get_window_origin();
        auto bias_dram_window =
            make_tile_window(bias_dram_block_window_tmp.get_bottom_tensor_view(),
                             bias_dram_block_window_tmp.get_window_lengths(),
                             {bias_origin.at(number<0>{}), kv_load_start},
                             gemm_0.MakeCBlockTile().get_tile_distribution());

        auto k_dram_window =
            make_tile_window(k_dram_block_window_tmp,
                             {kv_load_start, 0},
                             Policy::template MakeKDramTileDistribution<Problem, true>());

        auto k_lds_write_view = make_tensor_view<address_space_enum::lds>(
            static_cast<KDataType* __restrict__>(smem_ptrk0),
            Policy::template MakeKLdsBlockDescriptor<Problem, true>());

        auto k_lds_read_view = make_tensor_view<address_space_enum::lds>(
            static_cast<KDataType* __restrict__>(smem_ptrk0),
            Policy::template MakeKLdsBlockDescriptor<Problem, true>());

        auto k_lds_write_window = make_tile_window(
            k_lds_write_view,
            Policy::template MakeKLdsBlockDescriptor<Problem, true>().get_lengths(),
            {0, 0});

        auto k_lds_read_window =
            make_tile_window(k_lds_read_view,
                             make_tuple(number<kN0>{}, number<kK0>{}),
                             {0, 0},
                             Policy::template MakeKRegTileDistribution<Problem>());

        // V tile in LDS (sink-aware start)
        auto v_dram_window =
            make_tile_window(v_dram_block_window_tmp,
                             {kv_load_start, 0},
                             Policy::template MakeVDramTileDistribution<Problem>());

        auto v_lds_write_view = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<VDataType* __restrict__>(static_cast<char*>(smem_ptrv0)),
            Policy::template MakeVLdsBlockDescriptor<Problem>());

        auto v_lds_read_view = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<VDataType* __restrict__>(static_cast<char*>(smem_ptrv0)),
            Policy::template MakeVLdsBlockDescriptor<Problem>());

        auto v_lds_write_window =
            make_tile_window(v_lds_write_view,
                             Policy::template MakeVLdsBlockDescriptor<Problem>().get_lengths(),
                             {0, 0});

        auto v_lds_read_window =
            make_tile_window(v_lds_read_view,
                             make_tuple(number<kK1>{}, number<kN1>{}),
                             {0, 0},
                             Policy::template MakeVRegTileDistribution<Problem>());

        index_t num_total_loop =
            integer_divide_ceil(physical_seqlen_k_end - physical_seqlen_k_start, kN0) +
            num_sink_loop;

        constexpr bool kUseCountdownLoop = Policy::template UseCountdownLoop<Problem>();
        bool even_buffer                 = true;
        index_t i_total_loops            = 0;
        index_t remaining_loops          = num_total_loop;
        constexpr index_t k1_loops       = kN0 / kK1;
        index_t page_tile_i              = 0;
        auto* id_lds_slot = reinterpret_cast<CK_TILE_LDS_ADDR int32_t*>(page_id_slot);

        // Read each valid table entry once. Invalid lookahead tiles use page
        // zero with zero token rows, but retain the original TDM issue count.
        // Each value owns exactly this wave's fragments: one shared id for
        // P>=128, the id of the wave's page for P64, two wave-local ids for P16.
        // With sink tokens the tiles are not contiguous: the sink tiles from origin 0, then the
        // window from physical_seqlen_k_start (the windows below jump the same way). The page
        // lookups follow that order, and read no table entry of the tokens in between.
        [[maybe_unused]] const FmhaPageTileSinkTiles page_tile_order{num_sink_loop,
                                                                     physical_seqlen_k_start};
        PageIds page_id0, page_id1, page_id2;
        if constexpr(kHasSink)
        {
            page_id0 =
                k_dram_window.prefetch_page_id(0, num_total_loop, kv_load_start, page_tile_order);
            page_id1 =
                k_dram_window.prefetch_page_id(1, num_total_loop, kv_load_start, page_tile_order);
            page_id2 =
                k_dram_window.prefetch_page_id(2, num_total_loop, kv_load_start, page_tile_order);
        }
        else
        {
            page_id0 = k_dram_window.prefetch_page_id(0, num_total_loop, kv_load_start);
            page_id1 = k_dram_window.prefetch_page_id(1, num_total_loop, kv_load_start);
            page_id2 = k_dram_window.prefetch_page_id(2, num_total_loop, kv_load_start);
        }
        PageIds page_id3{};
        // Raw ids read before PV, completed into the queue value after PV.
        PageIds page_id3_raw{};
        // For P>=64 the async j+3 queue keeps SCALAR semantics: owned[0] carries one
        // physical id (shared by the tile for P>=128, the wave's page for P64). P16 uses the
        // two-fragment struct.
        index_t page_id3_valid_tokens = 0;
        index_t page_id3_raw_scalar   = 0;
        // Policy::UsePageIdBatch: the j+3 ids come from VGPR batches of 32 tiles
        // (load_page_id_batch), the next batch loaded one batch ahead. The other P>=64 tiles
        // publish each id through LDS (the scalar queue).
        constexpr bool kPageIdBatch             = Policy::template UsePageIdBatch<Problem>();
        [[maybe_unused]] int32_t id_batch       = 0;
        [[maybe_unused]] int32_t id_batch_next  = 0;
        [[maybe_unused]] index_t id_batch_tile0 = 0;
        if constexpr(kPageIdBatch)
        {
            id_batch      = k_dram_window.load_page_id_batch(0, num_total_loop, kv_load_start);
            id_batch_next = k_dram_window.load_page_id_batch(32, num_total_loop, kv_load_start);
        }

        static_assert(1 <= k1_loops);
        // kLeanIssue: per-wave TDM descriptor inputs, derived once (see tdm_issue_lean).
        [[maybe_unused]] const auto k_lean_issue = [&]() {
            if constexpr(kLeanIssue)
                return k_dram_window.make_lean_issue_state(k_lds_write_window);
            else
                return 0;
        }();
        [[maybe_unused]] const auto v_lean_issue = [&]() {
            if constexpr(kLeanIssue)
                return v_dram_window.make_lean_issue_state(v_lds_write_window);
            else
                return 0;
        }();
        // Without kEarlyKvPrologue, Q occupies K0: all Q reads must end before K0 is written.
        if constexpr(!kEarlyKvPrologue)
            block_sync_lds<0>();
        // Fixed TDM prologue: current K/V and the first look-ahead K buffer.
        k_dram_window.set_page_id(page_id0);
        v_dram_window.set_page_id(page_id0);
        load_tile_tdm(tdm_config_k, k_lds_write_window, k_dram_window);
        load_tile_tdm(tdm_config_v, v_lds_write_window, v_dram_window);
        move_tile_window(k_dram_window, {kN0, 0});
        // K1 is consumed at iteration 1. A one-tile prefix must jump before K1 loads.
        if constexpr(kHasSink)
        {
            if(num_sink_loop == 1 && num_total_loop > 1)
                move_tile_window(k_dram_window, {physical_seqlen_k_start - sink_seq_end, 0});
        }
        if constexpr(kEarlyKvPrologue)
        {
            // Tensor loads complete in issue order (the assumption of every partial TENSORcnt wait,
            // here and in the scheduled-TDM loop): once at most the K0 and V0 issues remain, Q has
            // landed. K1 is issued only after this wait, so no wave has more tensor loads
            // in flight than in the Q-in-K0 order.
            fmha_page_tile_tensorcnt_barrier<kFencedBarriers,
                                             PagedTransfer::kKPrefetchTensorCount +
                                                 PagedTransfer::kVPrefetchTensorCount>();
            q_tile = load_tile(q_lds_read_window);
        }
        k_lds_write_window.set_bottom_tensor_view_data_ptr(
            static_cast<KDataType* __restrict__>(smem_ptrk1));
        k_dram_window.set_page_id(page_id1);
        load_tile_tdm(tdm_config_k, k_lds_write_window, k_dram_window);

        constexpr index_t k_lds_insts = Geometry::kHeadDimQK == 64 && Geometry::kN == 128
                                            ? Geometry::kKSuLoadCount
                                            : k_lds_read_window.get_num_of_access();
        constexpr index_t v_lds_insts = v_lds_read_window.get_num_of_access();
        static_assert(k_lds_insts <= Geometry::kKSuLoadCount &&
                      v_lds_insts == Geometry::kVStageLoadCount);

        if constexpr(kEarlyKvPrologue)
        {
            // K0/V0 landed, and (dscnt 0 before the barrier) every wave has finished reading Q,
            // so iteration 0 may overwrite V1.
            s_wait_tensorcnt_barrier<PagedTransfer::kKPrefetchTensorCount, 0>();
        }
        else
        {
            fmha_page_tile_tensorcnt_barrier<kFencedBarriers,
                                             PagedTransfer::kKPrefetchTensorCount,
                                             kKPrefetchBarrierDsCount>();
        }
        auto initial_k_lds_su_read_window =
            make_tile_window(k_lds_read_view,
                             make_tuple(number<Geometry::kQkSuColumns>{}, number<kQKHeaddim>{}),
                             {0, 0},
                             Policy::template MakeKSuRegTileDistribution<Problem>());
        initial_k_lds_su_read_window.set_bottom_tensor_view_data_ptr(
            static_cast<KDataType* __restrict__>(smem_ptrk0));
        static_assert(initial_k_lds_su_read_window.get_num_of_access() == Geometry::kKSuLoadCount);
        auto k_su0_tile = load_tile(initial_k_lds_su_read_window);

        __builtin_amdgcn_sched_barrier(0);

        auto mainloop = [&](KDataType* __restrict__ k_lds_write_ptr,
                            KDataType* __restrict__ k_lds_read_ptr,
                            KDataType* __restrict__ v_lds_write_ptr,
                            KDataType* __restrict__ v_lds_read_ptr) {
            // LDS read bases of the tile, formed in begin_iteration (Policy::PrepareTileBases).
            typename Policy::TileBases bases{};

            decltype(load_tile_transpose(v_lds_read_window)) v_tile;
            [[maybe_unused]] float pv_output_scale_m0 = 1.0f;
            [[maybe_unused]] float pv_output_scale_m1 = 1.0f;
            v_lds_read_window.set_bottom_tensor_view_data_ptr(v_lds_read_ptr);

            auto run_fragment_pv_stage = [&](auto stage,
                                             const auto& a_block_tensor,
                                             const auto& b_block_tensor,
                                             auto& next_b_block_tensor,
                                             const auto& b_lds_window) {
                using BlockGemm   = remove_cvref_t<decltype(gemm_1)>;
                using WarpGemm    = typename BlockGemm::WarpGemm;
                using AWarpDstr   = typename WarpGemm::AWarpDstr;
                using BWarpDstr   = typename WarpGemm::BWarpDstr;
                using AWarpTensor = typename WarpGemm::AWarpTensor;
                using BWarpTensor = typename WarpGemm::BWarpTensor;
                using CWarpTensor = typename WarpGemm::CWarpTensor;
                constexpr auto a_warp_y_lengths =
                    to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
                constexpr auto b_warp_y_lengths =
                    to_sequence(BWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
                constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
                constexpr auto b_warp_y_index_zeros = uniform_sequence_gen_t<BWarpDstr::NDimY, 0>{};

                auto rescale_fragment = [&](auto ordinal) {
                    Policy::template RescaleOutputFragment<decltype(ordinal)::value>(
                        o_acc, pv_output_scale_m0, pv_output_scale_m1);
                };
                Schedule::template RunPvStagePrelude<Mode, decltype(stage)::value>(
                    [&](auto ordinal) { rescale_fragment(ordinal); });

                auto emit_wmma = [&](auto, auto wmma) {
                    // Row ordinal = k_iter * output fragments + output fragment: a K64 PV stage
                    // (D64 M64) runs the output fragments once per WMMA K step.
                    constexpr index_t ordinal = decltype(wmma)::value;
                    constexpr index_t kOutputOrdinal =
                        ordinal % Policy::OutputFragments::kNumFragments;
                    constexpr index_t kKIter = ordinal / Policy::OutputFragments::kNumFragments;
                    static_assert(Geometry::kPvStageKIter == 1 || Geometry::kPvMIter == 1);
                    constexpr index_t d_msb       = kOutputOrdinal / Policy::OutputFragments::kNumN;
                    constexpr index_t n           = kOutputOrdinal % Policy::OutputFragments::kNumN;
                    constexpr index_t m_iter      = d_msb / 2;
                    constexpr index_t v_msb       = d_msb % 2;
                    constexpr index_t full_n_iter = n * 2 + v_msb;

                    AWarpTensor a_warp_tensor;
                    a_warp_tensor.get_thread_buffer() = a_block_tensor.get_y_sliced_thread_data(
                        merge_sequences(sequence<kKIter, m_iter>{}, a_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

                    BWarpTensor b_warp_tensor;
                    b_warp_tensor.get_thread_buffer() = b_block_tensor.get_y_sliced_thread_data(
                        merge_sequences(sequence<kKIter, full_n_iter>{}, b_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));

                    // WMMA operand guard of the row (Policy::PvKeptRows, PvRawFence): keep_v keeps
                    // the operands of row o allocated.
                    auto keep_v = [&](auto o) {
                        constexpr index_t kO      = decltype(o)::value;
                        constexpr index_t kOutput = kO % Policy::OutputFragments::kNumFragments;
                        constexpr index_t kK      = kO / Policy::OutputFragments::kNumFragments;
                        constexpr index_t kN      = kOutput % Policy::OutputFragments::kNumN;
                        constexpr index_t kMsb    = kOutput / Policy::OutputFragments::kNumN % 2;
                        BWarpTensor kept;
                        kept.get_thread_buffer() = b_block_tensor.get_y_sliced_thread_data(
                            merge_sequences(sequence<kK, kN * 2 + kMsb>{}, b_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));
                        Policy::KeepOperandLive(kept);
                        // K64 stage (one M iteration): the P chunk (A operand) of a K step dies
                        // at that step's last row, inside the stage, and the deferred reads of
                        // the next rows may be allocated onto it while its readers are in
                        // flight, so it stays allocated until the drain too. With one K step
                        // a chunk lives up to its M iteration's last row; Policy::PvKeptPIters
                        // names the chunks kept after the stage's last-row fence.
                        if constexpr(Geometry::kPvStageKIter > 1)
                        {
                            constexpr index_t kM = kOutput / Policy::OutputFragments::kNumN / 2;
                            AWarpTensor kept_p;
                            kept_p.get_thread_buffer() = a_block_tensor.get_y_sliced_thread_data(
                                merge_sequences(sequence<kK, kM>{}, a_warp_y_index_zeros),
                                merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));
                            Policy::KeepOperandLive(kept_p);
                        }
                    };
                    // The rows that the wait of row ordinal - 1 drained stay allocated until here.
                    constexpr auto kKeptBefore = Policy::template PvKeptRows<Mode>(ordinal, false);
                    static_for<kKeptBefore.begin, kKeptBefore.end, 1>{}([&](auto q) { keep_v(q); });
                    CWarpTensor c_warp_tensor;
                    c_warp_tensor.get_thread_buffer().template set_as<fp32x8_t>(
                        number<0>{}, o_acc.at(number<kOutputOrdinal>{}));
                    WarpGemm{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);
                    // The last M iteration (rows from Policy::kPvLastMRow) is the last reader of
                    // the V operand registers that the next V read refills (see
                    // WaitWmmaOperandsRead).
                    static_assert(Policy::OutputFragments::kNumDmsb == 2 * Geometry::kPvMIter);
                    static_assert((m_iter == Geometry::kPvMIter - 1) ==
                                  (ordinal >= Policy::kPvLastMRow));
                    if constexpr(Policy::template PvRawFence<Mode>(ordinal))
                        Policy::WaitWmmaOperandsRead(c_warp_tensor);
                    // The rows pending at the RAW fence stay allocated until here.
                    constexpr auto kKeptAfter = Policy::template PvKeptRows<Mode>(ordinal, true);
                    static_for<kKeptAfter.begin, kKeptAfter.end, 1>{}([&](auto q) { keep_v(q); });
                    // The P chunks of the earlier M iterations stay allocated until here too.
                    static_for<0, Policy::template PvKeptPIters<Mode>(ordinal), 1>{}([&](auto i_m) {
                        AWarpTensor kept_p;
                        kept_p.get_thread_buffer() = a_block_tensor.get_y_sliced_thread_data(
                            merge_sequences(sequence<0, decltype(i_m)::value>{},
                                            a_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));
                        Policy::KeepOperandLive(kept_p);
                    });
                    o_acc.at(number<kOutputOrdinal>{}) =
                        c_warp_tensor.get_thread_buffer().template get_as<fp32x8_t>(number<0>{});
                };

                auto emit_rescale = [&](auto ordinal) {
                    __builtin_amdgcn_sched_barrier(0);
                    rescale_fragment(ordinal);
                };

                Policy::template RunPvStage<decltype(stage)::value, Mode>(
                    next_b_block_tensor, b_lds_window, emit_wmma, emit_rescale, bases);
            };

            auto run_pv = [&](const auto& p_tile) {
                if constexpr(1 < k1_loops)
                {
                    static_for<0, k1_loops - 1, 1>{}([&](auto i_k1) {
                        move_tile_window(v_lds_read_window, {kK1, 0});
                        decltype(v_tile) v_tile_switch;
                        auto p_tile_stage = get_slice_tile(
                            p_tile, sequence<0, i_k1 * kK1>{}, sequence<kM0, (i_k1 + 1) * kK1>{});
                        run_fragment_pv_stage(
                            i_k1, p_tile_stage, v_tile, v_tile_switch, v_lds_read_window);
                        v_tile = v_tile_switch;
                    });
                    move_tile_window(v_lds_read_window, {-kK1 * (k1_loops - 1), 0});
                }

                auto p_tile_last = get_slice_tile(
                    p_tile, sequence<0, (k1_loops - 1) * kK1>{}, sequence<kM0, k1_loops * kK1>{});
                auto k_lds_su0_read_window = make_tile_window(
                    k_lds_read_view,
                    make_tuple(number<Geometry::kQkSuColumns>{}, number<kQKHeaddim>{}),
                    {0, 0},
                    Policy::template MakeKSuRegTileDistribution<Problem>());
                k_lds_su0_read_window.set_bottom_tensor_view_data_ptr(k_lds_read_ptr);
                run_fragment_pv_stage(number<Geometry::kPvStages - 1>{},
                                      p_tile_last,
                                      v_tile,
                                      k_su0_tile,
                                      k_lds_su0_read_window);
            };

            auto begin_iteration = [&]() {
                // Fixed boundary: start the current V TDM transfer.
                auto next_su_window = make_tile_window(
                    k_lds_read_view,
                    make_tuple(number<Geometry::kQkSuColumns>{}, number<kQKHeaddim>{}),
                    {Geometry::kQkSuColumns, 0},
                    Policy::template MakeKSuRegTileDistribution<Problem>());
                next_su_window.set_bottom_tensor_view_data_ptr(k_lds_write_ptr);
                bases.qk_stage0 = FmhaTdmPreparedKRead::Prepare<0>(next_su_window);
                // The policy's other bases of the tile, ahead of every WMMA.
                Policy::template PrepareTileBases<Problem>(
                    bases, k_lds_read_view, k_lds_read_ptr, v_lds_read_window);
                __builtin_amdgcn_sched_barrier(0);

                // Reuse the V LDS buffer only after the preceding reads have completed.
                fmha_page_tile_block_sync_lds<kFencedBarriers, k_lds_insts>();
                // The barrier also ends all reads of the previous page-ID slot.
                // Issue the next ID before QK/PV without exposing its result.
                if constexpr(Problem::kPageBlockSize >= 64)
                {
                    if constexpr(kHasSink)
                        page_id3_valid_tokens =
                            k_dram_window.prefetch_page_id_to_lds_scalar(page_tile_i + 3,
                                                                         num_total_loop,
                                                                         kv_load_start,
                                                                         id_lds_slot,
                                                                         page_tile_order);
                    else if constexpr(!kPageIdBatch)
                        page_id3_valid_tokens = k_dram_window.prefetch_page_id_to_lds_scalar(
                            page_tile_i + 3, num_total_loop, kv_load_start, id_lds_slot);
                }
                else
                {
                    if constexpr(kHasSink)
                        page_id3 = k_dram_window.prefetch_page_id_to_lds(page_tile_i + 3,
                                                                         num_total_loop,
                                                                         kv_load_start,
                                                                         id_lds_slot,
                                                                         page_tile_order);
                    else
                        page_id3 = k_dram_window.prefetch_page_id_to_lds(
                            page_tile_i + 3, num_total_loop, kv_load_start, id_lds_slot);
                }
                // V is prefetched one iteration before its consumer.
                if constexpr(kHasSink)
                {
                    if(i_total_loops == num_sink_loop - 1 && i_total_loops + 1 < num_total_loop)
                        move_tile_window(v_dram_window,
                                         {physical_seqlen_k_start - sink_seq_end, 0});
                }
                move_tile_window(v_dram_window, {kN0, 0});
                v_lds_write_window.set_bottom_tensor_view_data_ptr(v_lds_write_ptr);
                v_dram_window.set_page_id(page_id1);
                // Policy::kDrainLdsBeforeTdmIssue: own LDS reads complete before the V TDM issue.
                if constexpr(Policy::kDrainLdsBeforeTdmIssue)
                    s_wait_dscnt<0>();
                if constexpr(kLeanIssue)
                {
                    v_dram_window.tdm_issue_lean(tdm_config_v, v_lds_write_window, v_lean_issue);
                }
                else if constexpr(kUseCountdownLoop && kN0 == 128 && kN1 == 128 &&
                                  std::is_same_v<VDataType, bf16_t>)
                {
                    v_dram_window.tdm_load_to_lds(
                        tdm_config_v,
                        v_lds_write_window,
                        make_null_tile_window(v_dram_window.get_window_lengths()),
                        number<-1>{},
                        bool_constant<false>{},
                        bool_constant<true>{});
                }
                else
                {
                    load_tile_tdm(tdm_config_v, v_lds_write_window, v_dram_window);
                }
            };

            auto run_qk = [&] {
                // STAGE 1, QK gemm
                clear_tile(s_acc); // initialize C
                constexpr auto gemm_0_su  = Policy::template GetQKBlockGemmSu<Problem>();
                auto k_lds_su_read_window = make_tile_window(
                    k_lds_read_view,
                    make_tuple(number<Geometry::kQkSuColumns>{}, number<kQKHeaddim>{}),
                    {0, 0},
                    Policy::template MakeKSuRegTileDistribution<Problem>());
                k_lds_su_read_window.set_bottom_tensor_view_data_ptr(k_lds_write_ptr);
                auto s_acc_su  = gemm_0_su.MakeCBlockTile();
                auto k_su_tile = k_su0_tile;
                static_for<0, Geometry::kQkStages, 1>{}([&](auto i_su) {
                    clear_tile(s_acc_su);
                    if constexpr(decltype(i_su)::value < Geometry::kQkStages - 1)
                    {
                        move_tile_window(k_lds_su_read_window, {Geometry::kQkSuColumns, 0});
                        decltype(k_su_tile) k_su_tile_next;

                        Policy::template RunQkStage<decltype(i_su)::value, Mode>(
                            gemm_0_su,
                            s_acc_su,
                            q_tile,
                            k_su_tile,
                            k_su_tile_next,
                            k_lds_su_read_window,
                            bases);

                        k_su_tile = k_su_tile_next;
                    }
                    else
                    {
                        Policy::template RunQkStage<Geometry::kQkStages - 1, Mode>(
                            gemm_0_su,
                            s_acc_su,
                            q_tile,
                            k_su_tile,
                            v_tile,
                            v_lds_read_window,
                            bases);
                    }
                    set_slice_tile(
                        s_acc,
                        s_acc_su,
                        sequence<0, decltype(i_su)::value * Geometry::kQkSuColumns>{},
                        sequence<kM0, (decltype(i_su)::value + 1) * Geometry::kQkSuColumns>{});
                });
            };

            auto process_score = [&]() {
                if constexpr(kHasLogitsSoftCap)
                {
                    // Tanh soft cap of the existing kernels. scale_s already holds
                    // scale * log2(e), so s <- (cap * log2(e) / scale_s) * tanh(s * scale / cap)
                    // gives log2(e) * cap * tanh(scale * s / cap) after the softmax's scale_s.
                    const float cap_in  = scale_s * log2e_rcp_v<float> / logits_soft_cap;
                    const float cap_out = logits_soft_cap * log2e_v<float> / scale_s;
                    tile_elementwise_inout(
                        [&](auto& x) { x = cap_out * tanh_fast<float>(x * cap_in); }, s_acc);
                }
                // QK scores are ready for bias, masking, and softmax.
                // STAGE 2: scale_s, add bias (as in the scheduled TDM pipeline)
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
                {
                    tile_elementwise_inout([&scale_s](auto& x) { x = x * scale_s; }, s_acc);
                    const auto bias_tile = load_tile(bias_dram_window);
                    tile_elementwise_inout(
                        [](auto& x, const auto& y) {
                            x += log2e_v<SaccDataType> * type_convert<SaccDataType>(y);
                        },
                        s_acc,
                        bias_tile);
                }
                else if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
                {
                    const auto current_k_origin = [&]() {
                        const bool in_sink = (num_sink_loop > i_total_loops);
                        if(in_sink)
                            return make_tuple(kN0 * i_total_loops + kv_load_start, 0);
                        else
                            return make_tuple(
                                kN0 * (i_total_loops - num_sink_loop) + physical_seqlen_k_start, 0);
                    }();
                    constexpr auto s_spans = decltype(s_acc)::get_distributed_spans();
                    sweep_tile_span(s_spans[number<0>{}], [&](auto idx0) {
                        sweep_tile_span(s_spans[number<1>{}], [&](auto idx1) {
                            const auto tile_idx = get_x_indices_from_distributed_indices(
                                s_acc.get_tile_distribution(), make_tuple(idx0, idx1));
                            const auto row = q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
                            const auto col = current_k_origin.at(I0) + tile_idx.at(number<1>{});
                            constexpr auto i_j_idx = make_tuple(idx0, idx1);
                            s_acc(i_j_idx) *= scale_s;
                            position_encoding.update(s_acc(i_j_idx), row, col);
                        });
                    });
                }

                constexpr bool kDeferTensorReady = Mode::kDeferTensorReady;
                if constexpr(!kDeferTensorReady &&
                             !Policy::template WaitTensorReadyAtLoadNextK<Problem>())
                    fmha_page_tile_tensorcnt_barrier<kFencedBarriers,
                                                     PagedTransfer::kVPrefetchTensorCount>();

                // Sink-aware k_origin (prefill path)
                const auto k_origin = [&]() {
                    if constexpr(kUseCountdownLoop)
                    {
                        // V's window already points one prefetched tile ahead.
                        return make_tuple(v_dram_window.get_window_origin().at(I0) - kN0, 0);
                    }
                    const bool in_sink_phase = (num_sink_loop > i_total_loops);
                    if(in_sink_phase)
                        return make_tuple(kN0 * i_total_loops + kv_load_start, 0);
                    else
                        return make_tuple(
                            kN0 * (i_total_loops - num_sink_loop) + physical_seqlen_k_start, 0);
                }();
                // The split tail predicate (the columns of the last tile past
                // physical_seqlen_k_end). In the tiles of kMaskLoopTrims the key range ends at
                // seqlen_k: the loop's last tile is an edge tile, and the mask predicate masks its
                // columns past seqlen_k, so the split predicate is redundant there. A KV-split
                // partial's range ends at its split's end, which the mask does not cover.
                constexpr bool kSplitTailCheck = kHasUnevenSplits && !kMaskLoopTrims;
                if constexpr(kUseCountdownLoop)
                {
                    [[maybe_unused]] bool need_split_check = false;
                    if constexpr(kSplitTailCheck)
                    {
                        const bool needs_tail_predicate =
                            k_origin.at(I0) + kN0 > physical_seqlen_k_end;
                        need_split_check = remaining_loops == 1 && needs_tail_predicate;
                    }
                    bool need_mask_check = false;
                    if constexpr(kMaskLoopTrims)
                    {
                        need_mask_check =
                            static_cast<uint32_t>(k_origin.at(I0) - mask_interior_first) >=
                            static_cast<uint32_t>(mask_interior_width);
                    }
                    else if constexpr(kPadSeqLenK || FmhaMask::IsMasking)
                    {
                        need_mask_check = mask.IsEdgeTileForValidRows(
                            q_origin.at(I0), k_origin.at(I0), number<kM0>{}, number<kN0>{});
                    }

                    if constexpr(FmhaMask::IsMasking)
                    {
                        if constexpr(kSplitTailCheck)
                        {
                            if(__builtin_expect(need_split_check, false))
                            {
                                set_tile_if(s_acc,
                                            -numeric<SMPLComputeDataType>::infinity(),
                                            [&](auto tile_idx) {
                                                const auto col = k_origin.at(I0) + tile_idx.at(I1);
                                                return physical_seqlen_k_end <= col;
                                            });
                            }
                        }
                        if(__builtin_expect(need_mask_check, false))
                        {
                            set_tile_if(s_acc,
                                        -numeric<SMPLComputeDataType>::infinity(),
                                        [&](auto tile_idx) {
                                            const auto row = q_origin.at(I0) + tile_idx.at(I0);
                                            const auto col = k_origin.at(I0) + tile_idx.at(I1);
                                            return Policy::template IsMaskedOut<Problem>(
                                                mask, row, col);
                                        });
                        }
                    }
                    else if(__builtin_expect(need_split_check || need_mask_check, false))
                    {
                        set_tile_if(
                            s_acc, -numeric<SMPLComputeDataType>::infinity(), [&](auto tile_idx) {
                                const auto row = q_origin.at(I0) + tile_idx.at(I0);
                                const auto col = k_origin.at(I0) + tile_idx.at(I1);
                                return (need_split_check && physical_seqlen_k_end <= col) ||
                                       (need_mask_check && mask.IsOutOfBound(row, col));
                            });
                    }
                }
                else
                {
                    if constexpr(kSplitTailCheck)
                    {
                        const bool needs_tail_predicate =
                            k_origin.at(I0) + kN0 > physical_seqlen_k_end;
                        const bool is_last_loop = kUseCountdownLoop
                                                      ? remaining_loops == 1
                                                      : i_total_loops == (num_total_loop - 1);
                        if(is_last_loop && needs_tail_predicate)
                        {
                            set_tile_if(
                                s_acc,
                                -numeric<SMPLComputeDataType>::infinity(),
                                [&, physical_seqlen_k_end_ = physical_seqlen_k_end](auto tile_idx) {
                                    const auto col = k_origin.at(I0) + tile_idx.at(I1);

                                    {
                                        return physical_seqlen_k_end_ <= col;
                                    }
                                });
                        }
                    }

                    if constexpr(kPadSeqLenK || FmhaMask::IsMasking)
                    {
                        // Rows past seqlen_q are never stored (the kernel's O view is bounded by
                        // seqlen_q), so they do not force the per-pixel check on every key tile of
                        // a partial last query tile.
                        bool need_perpixel_check;
                        if constexpr(kMaskLoopTrims)
                            need_perpixel_check =
                                static_cast<uint32_t>(k_origin.at(I0) - mask_interior_first) >=
                                static_cast<uint32_t>(mask_interior_width);
                        else
                            need_perpixel_check = mask.IsEdgeTileForValidRows(
                                q_origin.at(I0), k_origin.at(I0), number<kM0>{}, number<kN0>{});
                        if(need_perpixel_check)
                        {
                            set_tile_if(s_acc,
                                        -numeric<SMPLComputeDataType>::infinity(),
                                        [&](auto tile_idx) {
                                            const auto row = q_origin.at(I0) + tile_idx.at(I0);
                                            const auto col = k_origin.at(I0) + tile_idx.at(I1);
                                            return Policy::template IsMaskedOut<Problem>(
                                                mask, row, col);
                                        });
                        }
                    }
                }

                // Sink->normal window jump (prefill path)
                if constexpr(kHasSink)
                {
                    if(i_total_loops == num_sink_loop - 1)
                    {
                        move_tile_window(bias_dram_window,
                                         {0, physical_seqlen_k_start - sink_seq_end});
                    }
                }

                // move bias window (prefill path)
                move_tile_window(bias_dram_window, {0, kN0});
            };

            auto run_softmax = [&] {
                // Gemm1
                auto s_new = cast_tile<SMPLComputeDataType>(s_acc); // S{j}

                auto p_tile = [&]() {
                    static_assert(kNWarp == 1);
                    static_assert(std::is_same_v<SaccDataType, float> &&
                                  std::is_same_v<SMPLComputeDataType, float>);
                    float delta_m0;
                    float delta_m1;
                    Schedule::template RunSplitSoftmax<Mode>(
                        [&]() {
                            Policy::template RunSplitSoftmaxPart01<Problem>(
                                s_new, m, kScoresInLog2Domain ? 1.0f : scale_s, delta_m0, delta_m1);
                        },
                        [&]() {
                            fmha_page_tile_tensorcnt_barrier<
                                kFencedBarriers,
                                PagedTransfer::kVPrefetchTensorCount>();
                        },
                        [&](auto ordinal) {
                            const float scale_m0 = Mode::kEarlyOutputRescale
                                                       ? SplitSoftmax::Exp2(delta_m0)
                                                       : pv_output_scale_m0;
                            const float scale_m1 = Mode::kEarlyOutputRescale
                                                       ? SplitSoftmax::Exp2(delta_m1)
                                                       : pv_output_scale_m1;
                            Policy::template RescaleOutputFragment<decltype(ordinal)::value>(
                                o_acc, scale_m0, scale_m1);
                        },
                        [&](auto) {
                            if constexpr(Policy::kReadsV0InSoftmax)
                                Policy::template LoadV0InSoftmax<Problem>(
                                    v_tile, v_lds_read_window, bases);
                            Policy::template RunSplitSoftmaxPart2AndGetScale<Problem>(
                                s_new,
                                m,
                                l,
                                kScoresInLog2Domain ? 1.0f : scale_s,
                                delta_m0,
                                delta_m1,
                                pv_output_scale_m0,
                                pv_output_scale_m1);
                        });
                    return MakePForGemm1<decltype(gemm_1)>(s_new);
                }();
                return p_tile;
            };

            auto load_next_k = [&]() {
                // Fixed boundary: the next K TDM transfer follows softmax.
                // The existing V-read barrier also publishes the async page ID.
                // Read it here so PV can cover the LDS-to-VGPR latency. The page-id DMA carries
                // no asyncmark: wait on the hardware counter (fmha_page_tile_wait_page_id_dma).
                fmha_page_tile_wait_page_id_dma();
                if constexpr(Policy::template WaitTensorReadyAtLoadNextK<Problem>())
                {
                    // The K tile issued in the previous iteration must be complete; only the V
                    // tile issued after it may stay in flight. The barrier below then publishes
                    // that K tile to every wave.
                    s_wait_tensorcnt<PagedTransfer::kVPrefetchTensorCount>();
                }
                fmha_page_tile_block_sync_lds<kFencedBarriers, v_lds_insts>();
                if constexpr(Problem::kPageBlockSize >= 64)
                {
                    if constexpr(!kPageIdBatch)
                        page_id3_raw_scalar = k_dram_window.load_page_id_from_lds_scalar(
                            page_id3_valid_tokens, id_lds_slot);
                }
                else
                {
                    page_id3_raw = k_dram_window.load_page_id_from_lds(page_id3, id_lds_slot);
                }
                // The K transfer here feeds iteration i+2, beyond the already prefetched K1.
                if constexpr(kHasSink)
                {
                    if(i_total_loops == num_sink_loop - 2 && i_total_loops + 2 < num_total_loop)
                        move_tile_window(k_dram_window,
                                         {physical_seqlen_k_start - sink_seq_end, 0});
                }
                move_tile_window(k_dram_window, {kN0, 0});
                k_lds_write_window.set_bottom_tensor_view_data_ptr(k_lds_write_ptr);
                k_dram_window.set_page_id(page_id2);
                // Policy::kDrainLdsBeforeTdmIssue: own LDS reads complete before the K TDM issue.
                if constexpr(Policy::kDrainLdsBeforeTdmIssue)
                    s_wait_dscnt<0>();
                if constexpr(kLeanIssue)
                    k_dram_window.tdm_issue_lean(tdm_config_k, k_lds_write_window, k_lean_issue);
                else
                    load_tile_tdm(tdm_config_k, k_lds_write_window, k_dram_window);
            };

            auto advance_iteration = [&]() {
                // PV separates the page-ID LDS load from its first use.
                fmha_page_tile_tensorcnt_barrier<kFencedBarriers,
                                                 PagedTransfer::kKPrefetchTensorCount,
                                                 kKPrefetchBarrierDsCount>();
                if constexpr(kPageIdBatch)
                {
                    const index_t tile = page_tile_i + 3;
                    if(tile - id_batch_tile0 >= 32)
                    {
                        id_batch = id_batch_next;
                        id_batch_tile0 += 32;
                        id_batch_next = k_dram_window.load_page_id_batch(
                            id_batch_tile0 + 32, num_total_loop, kv_load_start);
                    }
                    page_id3 = k_dram_window.page_id_from_batch(
                        id_batch,
                        tile - id_batch_tile0,
                        k_dram_window.tile_valid_tokens(tile, num_total_loop, kv_load_start));
                }
                else if constexpr(Problem::kPageBlockSize >= 64)
                {
                    page_id3 = k_dram_window.complete_page_id_from_value_scalar(
                        page_id3_valid_tokens, page_id3_raw_scalar);
                }
                else
                    page_id3 = k_dram_window.complete_page_id_from_value(page_id3, page_id3_raw);
                page_id0 = page_id1;
                page_id1 = page_id2;
                page_id2 = page_id3;
                ++page_tile_i;
            };

            auto tile_ops = MakeFmhaTdmSchedTileOps(begin_iteration,
                                                    run_qk,
                                                    process_score,
                                                    run_softmax,
                                                    load_next_k,
                                                    run_pv,
                                                    advance_iteration);
            Schedule::Run(tile_ops);
        }; // mainloop

        const index_t num_pipeline_loop = num_total_loop;
        do
        {
            bool is_even_loop    = kUseCountdownLoop ? even_buffer : i_total_loops % 2 == 0;
            auto k_lds_write_ptr = is_even_loop ? static_cast<KDataType* __restrict__>(smem_ptrk0)
                                                : static_cast<KDataType* __restrict__>(smem_ptrk1);
            auto k_lds_read_ptr  = is_even_loop ? static_cast<KDataType* __restrict__>(smem_ptrk1)
                                                : static_cast<KDataType* __restrict__>(smem_ptrk0);
            auto v_lds_write_ptr = is_even_loop ? static_cast<VDataType* __restrict__>(smem_ptrv1)
                                                : static_cast<VDataType* __restrict__>(smem_ptrv0);
            auto v_lds_read_ptr  = is_even_loop ? static_cast<VDataType* __restrict__>(smem_ptrv0)
                                                : static_cast<VDataType* __restrict__>(smem_ptrv1);
            mainloop(k_lds_write_ptr, k_lds_read_ptr, v_lds_write_ptr, v_lds_read_ptr);
            if constexpr(kUseCountdownLoop)
            {
                even_buffer = !even_buffer;
                --remaining_loops;
            }
            else
            {
                i_total_loops++;
            }
        } while(kUseCountdownLoop ? remaining_loops > 0 : i_total_loops < num_pipeline_loop);

        // Reconstruct the CK tensor only at the final normalization boundary.
        auto output = Policy::OutputFragments::template Reconstruct<OaccBlockTileType>(o_acc);
        constexpr auto o_spans = decltype(output)::get_distributed_spans();

        auto finalize_real_tokens = [&]() {
            if constexpr(kStoreLSE)
            {
                // store lse acc
                auto lse_acc =
                    make_static_distributed_tensor<LSEDataType>(m.get_tile_distribution());

                constexpr auto lse_acc_spans = decltype(lse_acc)::get_distributed_spans();
                sweep_tile_span(lse_acc_spans[I0], [&, m_ = m, l_ = l](auto idx0) {
                    constexpr auto i_idx = make_tuple(idx0);
                    if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                                 BiasEnum == BlockAttentionBiasEnum::ALIBI)
                    {
                        lse_acc(i_idx) = m_[i_idx] / C_LOG2E + log(l_[i_idx]);
                    }
                    else
                    {
                        lse_acc(i_idx) = m_[i_idx] * scale_s / C_LOG2E + log(l_[i_idx]);
                    }
                });

                store_tile(lse_acc_dram_window_tmp, lse_acc);
            }

            sweep_tile_span(o_spans[I0], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                const auto tmp       = [&]() {
                    if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                                 FmhaMask::IsMasking)
                    {
                        return l[i_idx] == 0.f ? 0.f : 1 / l[i_idx];
                    }
                    else
                        return 1 / l[i_idx];
                }();
                sweep_tile_span(o_spans[I1], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    output(i_j_idx) *= tmp;
                });
            });
        };

        if constexpr(kEnableVirtualSink && BiasEnum == BlockAttentionBiasEnum::NO_BIAS)
        {
            if(__builtin_isinf_sign(sink_v) != -1)
            {
                auto lse_acc =
                    make_static_distributed_tensor<LSEDataType>(m.get_tile_distribution());
                const float sink_natural = sink_v * scale_s;
                const float sink_log2    = sink_natural * C_LOG2E;
                sweep_tile_span(o_spans[I0], [&](auto idx0) {
                    constexpr auto i_idx = make_tuple(idx0);
                    float output_scale;
                    float merged_lse;
                    if(__builtin_isnan(sink_natural))
                    {
                        output_scale = sink_natural;
                        merged_lse   = sink_natural;
                    }
                    else if(l[i_idx] == 0.f || __builtin_isinf_sign(sink_natural) > 0)
                    {
                        output_scale = 0.f;
                        merged_lse   = sink_natural;
                    }
                    else
                    {
                        const float real_log2   = m[i_idx] * scale_s;
                        const float merged_max  = max(real_log2, sink_log2);
                        const float real_weight = SplitSoftmax::Exp2(real_log2 - merged_max);
                        const float sink_weight = SplitSoftmax::Exp2(sink_log2 - merged_max);
                        const float merged_sum = __builtin_fmaf(l[i_idx], real_weight, sink_weight);
                        output_scale           = real_weight / merged_sum;
                        merged_lse             = merged_max / C_LOG2E + log(merged_sum);
                    }
                    if constexpr(kStoreLSE)
                        lse_acc(i_idx) = merged_lse;
                    sweep_tile_span(o_spans[I1], [&](auto idx1) {
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        // Explicit zeros also cover empty rows and an infinite sink.
                        output(i_j_idx) =
                            output_scale == 0.f ? 0.f : output(i_j_idx) * output_scale;
                    });
                });
                if constexpr(kStoreLSE)
                    store_tile(lse_acc_dram_window_tmp, lse_acc);
            }
            else
            {
                finalize_real_tokens();
            }
        }
        else
        {
            finalize_real_tokens();
        }

        Schedule::template RunEpilogue<Mode>(
            [&]() {
                // A partial prefetch wait can leave the terminal look-ahead TDM in flight.
                s_wait_tensorcnt<0>();
            },
            [&]() {
                // The final PV stage's next-K reads have no consumer in the last iteration.
                s_wait_dscnt<0>();
            });

        return output;
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename LSEaccDramBlockWindowTmp,
              typename PositionEncoding>
    CK_TILE_HOST_DEVICE auto operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp,
                                        const KDramBlockWindowTmp& k_dram_block_window_tmp,
                                        const VDramBlockWindowTmp& v_dram_block_window_tmp,
                                        const BiasDramBlockWindowTmp& bias_dram_block_window_tmp,
                                        LSEaccDramBlockWindowTmp& lse_acc_dram_window_tmp,
                                        FmhaMask mask,
                                        PositionEncoding position_encoding,
                                        float scale_s,
                                        float sink_v,
                                        float logits_soft_cap,
                                        void* smem_ptrk0,
                                        void* smem_ptrk1,
                                        void* smem_ptrv0,
                                        void* smem_ptrv1,
                                        void* page_id_slot) const
    {
        return run(q_dram_block_window_tmp,
                   k_dram_block_window_tmp,
                   v_dram_block_window_tmp,
                   bias_dram_block_window_tmp,
                   lse_acc_dram_window_tmp,
                   mask,
                   position_encoding,
                   scale_s,
                   smem_ptrk0,
                   smem_ptrk1,
                   smem_ptrv0,
                   smem_ptrv1,
                   page_id_slot,
                   sink_v,
                   logits_soft_cap);
    }
};

} // namespace ck_tile
