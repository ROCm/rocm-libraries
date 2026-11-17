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
#include "ck_tile/ops/fmha/pipeline/block_fmha_batch_prefill_pipeline_qr_ks_vs_tdm_sched_fp8_page_tile_policy.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"

namespace ck_tile {

// FP8 batch-prefill pipeline of the gfx1250 page tile (fp8bf16, per-tensor descales): FP8 Q / K / V
// and P with the FP8 WMMAs, BF16 O. The loop is the lean D64 loop of the BF16 page tile
// (BlockFmhaBatchPrefillPipelineQRKSVSTdmSchedPageTile): M128 query tiles, N128 KV tiles of two
// P64 pages (one per pair of waves) from the vectorized x = 16 pool, the page ids of tile i + 3
// from VGPR batches, three barriers per tile. P is handled as in the forward FP8 pipeline
// (BlockFmhaPipelineQRKSVSTdmSchedFp8): it is cast to FP8 as kPScale * P, the row sum is the mass
// of that FP8 P, and the output is O / l. The kernel folds q_descale * k_descale into scale_s and
// scales the output by v_descale.
template <typename Problem_, typename Policy_ = FmhaBatchPrefillFp8PageTilePolicyFor<Problem_>>
struct BlockFmhaBatchPrefillPipelineQRKSVSTdmSchedFp8PageTile
{
    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};

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
    using SoftmaxPolicy         = typename Policy::SoftmaxPolicy;
    using SplitSoftmax          = typename Policy::ScoreSoftmax;
    using Schedule              = typename Policy::Schedule;
    using Mode                  = FmhaTdmSchedMode<Problem, Policy>;
    static_assert(Mode::kIsLegal, "qr_tdm_sched FP8 page tile: schedule geometry is incompatible");

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

    static constexpr bool kIsGroupMode      = Problem::kIsGroupMode;
    static constexpr bool kPadSeqLenQ       = Problem::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK       = Problem::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ      = Problem::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV      = Problem::kPadHeadDimV;
    static constexpr bool kHasLogitsSoftCap = Problem::kHasLogitsSoftCap;
    static constexpr bool kHasDropout       = Problem::kHasDropout;
    static constexpr auto BiasEnum          = Problem::BiasEnum;
    static constexpr bool kStoreLSE         = Problem::kStoreLSE;
    static constexpr bool kHasUnevenSplits  = true;
    static constexpr bool kHasSink          = Problem::kHasSink;
    static constexpr auto QScaleEnum        = Problem::QScaleEnum;

    static_assert(CK_TILE_FMHA_FWD_FAST_EXP2,
                  "qr_tdm_sched: every softmax site calls exp2, so log2(e) must be folded into "
                  "scale_s");

    // Scope of the FP8 page tile: per-tensor FP8 Q / K / V / P with BF16 O; D64/V64 M128 N128
    // tiles of P64 pages from the vectorized x = 16 pool through a vLLM block table; no sink
    // tokens, bias, logits soft cap, LSE or dropout.
    static_assert(std::is_same_v<QDataType, fp8_t> && std::is_same_v<KDataType, fp8_t> &&
                      std::is_same_v<VDataType, fp8_t> && std::is_same_v<PDataType, fp8_t> &&
                      std::is_same_v<ODataType, bf16_t>,
                  "the FP8 page tile takes FP8 Q, K, V and P and stores BF16 O");
    static_assert(QScaleEnum == BlockAttentionQuantScaleEnum::PERTENSOR,
                  "the FP8 page tile takes per-tensor descales");
    static_assert(kM0 == 128 && kN0 == 128 && kQKHeaddim == 64 && kN1 == 64,
                  "the FP8 page tile supports D64/V64 M128 N128 tiles");
    static_assert(Problem::kIsGroupMode && Problem::kPageBlockSize == 64);
    static_assert(Problem::kKVMemoryLayout ==
                          BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT &&
                      Problem::kVectorSize == 16,
                  "the FP8 page tile reads the vectorized pool with x = 16");
    static_assert(Problem::kKVLookupTable ==
                  BlockAttentionKVCacheLookupTableEnum::VLLM_BLOCK_TABLE_2D);
    static_assert(Problem::kPadSeqLenQ && Problem::kPadSeqLenK);
    static_assert(!Problem::kPadHeadDimQ && !Problem::kPadHeadDimV);
    static_assert(!kHasSink && BiasEnum == BlockAttentionBiasEnum::NO_BIAS && !kHasLogitsSoftCap &&
                      !kStoreLSE && !kHasDropout,
                  "the FP8 page tile has no sink, bias, logits soft cap, LSE or dropout");
    // One workgroup per CU: the policy decides it, and the kernel launches with the pipeline's
    // kBlockPerCu, so a problem asking for another occupancy would be ignored.
    static_assert(Policy::kBlockPerCu == 1 &&
                      (Problem::kBlockPerCu == -1 || Problem::kBlockPerCu == 1),
                  "the FP8 page tile runs one workgroup per CU");
    // K and V reach LDS by asynchronous (TDM) copies only.
    static_assert(Policy::AsyncCopy);
    // The loop below is the lean D64 loop with the page-id batch, the lean TDM issue and the
    // K(i + 1) wait at NextK.
    static_assert(Policy::template UseLeanD64Loop<Problem>() &&
                  Policy::template UseCountdownLoop<Problem>() &&
                  Policy::template WaitTensorReadyAtLoadNextK<Problem>() &&
                  Policy::template UseLeanTdmIssue<Problem>() &&
                  Policy::template UsePageIdBatch<Problem>() && !Mode::kDeferTensorReady &&
                  Mode::kEarlyOutputRescale && !Mode::kStreamOutputRescale);

    // P is cast to FP8 against this fixed scale (kPScale * P keeps the small P out of the e4m3
    // subnormals), as in the forward FP8 pipeline; the accuracy contract takes it as an input.
    static constexpr float kPScale = 256.0f;
    // Without LSE or sink nothing needs the unquantized row sum: l sums the FP8 P that gemm_1
    // consumes (AccumulateQuantizedMass), so a common rounding gain cancels in O / l.
    static constexpr bool kQuantizedRowSum = true;

    static constexpr index_t kAlignmentQ = Policy::template GetAlignmentQ<Problem>();
    static constexpr index_t kAlignmentK = Policy::template GetAlignmentK<Problem>();
    static constexpr index_t kAlignmentV = Policy::template GetAlignmentV<Problem>();
    static constexpr index_t kAlignmentO = Policy::template GetAlignmentO<Problem>();
    static constexpr index_t kAlignmentBias =
        kPadSeqLenK ? 1 : Policy::template GetAlignmentBias<Problem>();
    static constexpr index_t kAlignmentRandVal =
        kPadSeqLenK ? 1 : Policy::template GetAlignmentRandVal<Problem>();

    static_assert(Policy::template IsSupportedProblem<Problem>(),
                  "the FP8 page-tile policy does not take this problem");

    using Geometry = typename Policy::Geometry;
    static_assert(Geometry::kM == kM0 && Geometry::kN == kN0 && Geometry::kDv == kN1);

    // Page transfer budget: one native tensor issue per wave per K or V load group at P64.
    using PageTileTraits = FmhaBatchPrefillPageTileTraits<Problem::kPageBlockSize, kQKHeaddim>;
    using PageIds        = FmhaBatchPrefillPageTileIds<Problem::kPageBlockSize>;
    struct PagedTransfer
    {
        static constexpr index_t kKPrefetchTensorCount = PageTileTraits::kKPrefetchTensorCount;
        static constexpr index_t kVPrefetchTensorCount = PageTileTraits::kVPrefetchTensorCount;
        static constexpr index_t kFragmentsPerWave     = PageTileTraits::kFragmentsPerWave;
    };
    static_assert(PagedTransfer::kFragmentsPerWave == 1);
    // The page tile writer is a workgroup collective over its own wave count.
    static_assert(PageTileTraits::kWaves == Geometry::kWaves,
                  "page tile writer waves must equal the pipeline's waves");
    // The schedule's epilogue tensor drain (RunEpilogue) is enabled by the policy's counts; page
    // transfers can be in flight there.
    static_assert((PagedTransfer::kKPrefetchTensorCount == 0 &&
                   PagedTransfer::kVPrefetchTensorCount == 0) ||
                      ((Policy::kKPrefetchTensorCount != 0 || Policy::kVPrefetchTensorCount != 0) &&
                       Policy::kPrefetchTailDrain),
                  "page transfers in flight but the policy disables the epilogue drain");

    static constexpr index_t kBlockPerCu = Policy::kBlockPerCu;

    // Read by the batch-prefill kernels.
    static constexpr index_t kPageBlockSize = Problem::kPageBlockSize;
    static constexpr index_t kVectorSize    = Problem::kVectorSize;
    static constexpr auto kKVMemoryLayout   = Problem::kKVMemoryLayout;
    static constexpr const char* name       = "qr_tdm_sched_fp8_page_tile";

    // Without the stale-fragment workaround (Policy::kDrainLdsBeforeTdmIssue) the barrier after
    // the K prefetch issue does not wait for the wave's LDS reads.
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
        // The page-id slot after the arena, as in the BF16 page tile: the kernel places it and
        // passes its address. The P64 page-id batch does not use it.
        return Policy::GetLdsArenaSize() + PageTileTraits::kMetadataBytes;
    }

    // fp8x8 as fp32. On gfx1250 (not strict, not gfx1251) one literal-scale builtin, the same for
    // every lane, converts the whole wave (fp8x8_to_fp32x8 converts each half-wave under its own
    // scale select there). The other targets take fp8x8_to_fp32x8: float8_ext.hpp allows the
    // builtin only on that target. The forward FP8 pipeline's Dequantize guards it with
    // __gfx125__ only.
    CK_TILE_DEVICE static fp32x8_t Dequantize(const fp8x8_t& q)
    {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1250__) && !defined(__gfx1250_strict__) && \
    !defined(__gfx1251__)
        return __builtin_amdgcn_cvt_scale_pk8_f32_fp8(bit_cast<uint32x2_t>(q), 0x7f7f7f7fu, 0);
#else
        return fp8x8_to_fp32x8(q, 1.0f);
#endif
    }

    // Rescales row_sum like l and adds the lane rows' sum of the FP8 P that gemm_1 consumes (the
    // forward FP8 pipeline's AccumulateQuantizedMass).
    template <typename Gemm1, typename PTile, typename RowTile>
    CK_TILE_DEVICE static void
    AccumulateQuantizedMass(const PTile& p_tile, RowTile& row_sum, float scale_m0, float scale_m1)
    {
        constexpr index_t kVec  = 8;
        constexpr index_t kPMI  = Geometry::kPvMIter;
        constexpr index_t kBuf  = PTile::get_thread_buffer_size();
        constexpr index_t kPKI  = kN0 / Gemm1::WarpGemm::kK;
        constexpr index_t kPSub = kBuf / (kPMI * kPKI);
        static_assert(kPMI == 2 && kPSub % kVec == 0);
        array<fp32x8_t, 4> acc{};
        static_for<0, kBuf / kVec, 1>{}([&](auto i) {
            constexpr index_t row = (decltype(i)::value * kVec / kPSub) % kPMI;
            constexpr index_t a   = row * 2 + decltype(i)::value % 2;
            acc(number<a>{}) += Dequantize(p_tile.get_thread_buffer().template get_as<fp8x8_t>(i));
        });
        auto& lq = row_sum.get_thread_buffer();
        static_for<0, 2, 1>{}([&](auto row) {
            const fp32x8_t v = acc[number<row * 2>{}] + acc[number<row * 2 + 1>{}];
            float s          = ((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7]));
            s += SplitSoftmax::PermuteLaneX16(s);
            lq(row) = __builtin_fmaf(decltype(row)::value == 0 ? scale_m0 : scale_m1, lq[row], s);
        });
    }

    // P for gemm_1: the scaled convert divides by its power-of-two scale, so this stores
    // kPScale * P (the forward FP8 pipeline's MakePForGemm1), then the policy's in-lane repack.
    // The rounding is round-to-nearest-even whatever CK_TILE_FLOAT_TO_FP8_DEFAULT a build sets:
    // the accuracy contract's u_P assumes it.
    template <typename Gemm1, typename PComputeTensor>
    CK_TILE_DEVICE static auto MakePForGemm1(const PComputeTensor& p_compute)
    {
        auto p_src = make_static_distributed_tensor<PDataType>(p_compute.get_tile_distribution());
        constexpr index_t kVec = 8;
        static_assert(decltype(p_src)::get_thread_buffer_size() % kVec == 0);
        static_for<0, decltype(p_src)::get_thread_buffer_size() / kVec, 1>{}([&](auto i) {
            fp32x8_t v;
            static_for<0, kVec, 1>{}(
                [&](auto e) { v[e()] = p_compute.get_thread_buffer()[number<i * kVec + e>{}]; });
            p_src.get_thread_buffer().template set_as<fp8x8_t>(
                i, fp32x8_to_fp8x8<fp8_rounding_mode::standard>(v, 1.0f / kPScale));
        });
        return Policy::template RepackPForGemm1<Problem, Gemm1>(p_src);
    }

    // Q, K and V reach the fixed LDS arena by TDM; K and V are double buffered. The arguments are
    // those of the BF16 page tile; the sink logit, the logits soft cap, the bias and LSE windows
    // and the position encoding have no use here (asserted off above).
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
        ignore = bias_dram_block_window_tmp;
        ignore = lse_acc_dram_window_tmp;
        ignore = position_encoding;
        ignore = sink_v;
        ignore = logits_soft_cap;
        ignore = page_id_slot;
        static_assert(
            std::is_same_v<QDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                std::is_same_v<KDataType, remove_cvref_t<typename KDramBlockWindowTmp::DataType>> &&
                std::is_same_v<VDataType, remove_cvref_t<typename VDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kM0 == QDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                          kSubQKHeaddim == QDramBlockWindowTmp{}.get_window_lengths()[I1] &&
                          kN0 == KDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                          kK0 == KDramBlockWindowTmp{}.get_window_lengths()[I1] &&
                          kN1 == VDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                          kK1 == VDramBlockWindowTmp{}.get_window_lengths()[I1],
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

        auto o_acc = Policy::OutputFragments::MakeZero();

        // infer Sacc, S, P, M, L, Oacc type
        using SBlockTileType = decltype(cast_tile<SMPLComputeDataType>(OaccBlockTileType{}));

        using MLBlockTileType = decltype(block_tile_reduce<SMPLComputeDataType>(
            SBlockTileType{}, sequence<1>{}, f_max, SMPLComputeDataType{0}));

        auto m = MLBlockTileType{};
        auto l = MLBlockTileType{};
        set_tile(m, -numeric<SMPLComputeDataType>::infinity());
        clear_tile(l);

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

        const auto tile_range =
            mask.GetTileRangeAlongX(q_origin.at(I0), number<kM0>{}, number<kN0>{});
        const index_t logical_seqlen_k_start = tile_range.get(I0);
        const index_t logical_seqlen_k_end   = tile_range.get(I1);

        // No key tile to visit (nothing has been issued yet): the output stays zero.
        const index_t num_total_loop =
            integer_divide_ceil(logical_seqlen_k_end - logical_seqlen_k_start, kN0);
        if(num_total_loop <= 0)
            return Policy::OutputFragments::template Reconstruct<OaccBlockTileType>(o_acc);

        // ---------------------------------------------------------------------
        // TDM configs for Q / K / V, from the policy.
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

        // Q tile in LDS: staged in the V1 buffer, which is first written by the V prefetch of
        // iteration 0, so the K0/V0 loads are issued while Q is still in flight.
        auto* q_lds_ptr = static_cast<QDataType*>(smem_ptrv1);

        auto q_dram_window = make_tile_window(
            q_dram_block_window_tmp, Policy::template MakeQDramTileDistribution<Problem>());

        auto q_lds_view = make_tensor_view<address_space_enum::lds>(
            q_lds_ptr, Policy::template MakeQLdsBlockDescriptor<Problem>());

        auto q_lds_store_window = make_tile_window(
            q_lds_view, Policy::template MakeQLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});

        auto q_lds_read_window =
            make_tile_window(q_lds_view,
                             Policy::template MakeQLdsBlockDescriptor<Problem>().get_lengths(),
                             {0, 0},
                             Policy::template MakeQRegTileDistribution<Problem>());

        load_tile_tdm(tdm_config_q, q_lds_store_window, q_dram_window);
        // read after the K/V prologue loads are issued
        decltype(load_tile(q_lds_read_window)) q_tile;

        // K and V tiles in LDS
        const index_t kv_load_start = logical_seqlen_k_start > 0 ? logical_seqlen_k_start : 0;
        const index_t physical_seqlen_k_end = logical_seqlen_k_end;

        auto k_dram_window =
            make_tile_window(k_dram_block_window_tmp,
                             {kv_load_start, 0},
                             Policy::template MakeKDramTileDistribution<Problem, true>());

        auto k_lds_view = make_tensor_view<address_space_enum::lds>(
            static_cast<KDataType* __restrict__>(smem_ptrk0),
            Policy::template MakeKLdsBlockDescriptor<Problem, true>());

        auto k_lds_write_window = make_tile_window(
            k_lds_view,
            Policy::template MakeKLdsBlockDescriptor<Problem, true>().get_lengths(),
            {0, 0});

        // The K operand of one QK stage: row group 0 of a K tile (the stage reads use constant
        // offsets from its base, Policy::RunQkStage).
        auto make_k_su_window = [&](KDataType* __restrict__ ptr) {
            auto window =
                make_tile_window(k_lds_view,
                                 make_tuple(number<Geometry::kQkSuColumns>{}, number<kQKHeaddim>{}),
                                 {0, 0},
                                 Policy::template MakeKSuRegTileDistribution<Problem>());
            window.set_bottom_tensor_view_data_ptr(ptr);
            return window;
        };
        using KSuWindow = decltype(make_k_su_window(nullptr));
        using KRead     = typename Policy::template KReadFor<KSuWindow>;
        using KSuTile   = decltype(make_static_distributed_tensor<KDataType>(
            Policy::template MakeKSuRegTileDistribution<Problem>()));

        auto v_dram_window =
            make_tile_window(v_dram_block_window_tmp,
                             {kv_load_start, 0},
                             Policy::template MakeVDramTileDistribution<Problem>());

        auto v_lds_view = make_tensor_view<address_space_enum::lds>(
            static_cast<VDataType* __restrict__>(smem_ptrv0),
            Policy::template MakeVLdsBlockDescriptor<Problem>());

        auto v_lds_write_window = make_tile_window(
            v_lds_view, Policy::template MakeVLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});

        auto v_lds_read_window =
            make_tile_window(v_lds_view,
                             make_tuple(number<kK1>{}, number<kN1>{}),
                             {0, 0},
                             Policy::template MakeVRegTileDistribution<Problem>());
        using VRead = typename Policy::template VReadFor<decltype(v_lds_read_window)>;
        using VTile = decltype(make_static_distributed_tensor<VDataType>(
            typename VRead::OwnerDistribution{}));
        static_assert(
            std::is_same_v<typename VTile::StaticTileDistribution::DstrEncode,
                           remove_cvref_t<decltype(PvBlockGemm::MakeBBlockDistributionEncode())>>,
            "the V reads leave V in gemm_1's B layout");

        constexpr index_t k_lds_insts = Geometry::kKSuLoadCount;
        constexpr index_t v_lds_insts = Geometry::kVStageLoadCount;
        static_assert(KRead::kNumAccess == k_lds_insts && VRead::kNumAccess == v_lds_insts);
        static_assert(kN0 == kK1, "one PV stage covers the KV tile");

        bool even_buffer        = true;
        index_t remaining_loops = num_total_loop;
        index_t page_tile_i     = 0;

        // Read each valid table entry once. Invalid lookahead tiles use page zero with zero token
        // rows, but retain the original TDM issue count. Each value owns the id of the wave's
        // page (P64).
        PageIds page_id0 = k_dram_window.prefetch_page_id(0, num_total_loop, kv_load_start);
        PageIds page_id1 = k_dram_window.prefetch_page_id(1, num_total_loop, kv_load_start);
        PageIds page_id2 = k_dram_window.prefetch_page_id(2, num_total_loop, kv_load_start);
        PageIds page_id3{};
        // Policy::UsePageIdBatch: the j+3 ids come from VGPR batches of 32 tiles
        // (load_page_id_batch), the next batch loaded one batch ahead.
        int32_t id_batch      = k_dram_window.load_page_id_batch(0, num_total_loop, kv_load_start);
        int32_t id_batch_next = k_dram_window.load_page_id_batch(32, num_total_loop, kv_load_start);
        index_t id_batch_tile0 = 0;

        // Lean TDM issue: per-wave TDM descriptor inputs, derived once (see tdm_issue_lean).
        const auto k_lean_issue = k_dram_window.make_lean_issue_state(k_lds_write_window);
        const auto v_lean_issue = v_dram_window.make_lean_issue_state(v_lds_write_window);
        // Fixed TDM prologue: current K/V and the first look-ahead K buffer.
        k_dram_window.set_page_id(page_id0);
        v_dram_window.set_page_id(page_id0);
        load_tile_tdm(tdm_config_k, k_lds_write_window, k_dram_window);
        load_tile_tdm(tdm_config_v, v_lds_write_window, v_dram_window);
        move_tile_window(k_dram_window, {kN0, 0});
        // Tensor loads complete in issue order (the assumption of every partial TENSORcnt wait,
        // here and in the loop): once at most the K0 and V0 issues remain, Q has landed. K1 is
        // issued only after this wait.
        s_wait_tensorcnt_barrier<PagedTransfer::kKPrefetchTensorCount +
                                 PagedTransfer::kVPrefetchTensorCount>();
        q_tile = load_tile(q_lds_read_window);
        k_lds_write_window.set_bottom_tensor_view_data_ptr(
            static_cast<KDataType* __restrict__>(smem_ptrk1));
        k_dram_window.set_page_id(page_id1);
        load_tile_tdm(tdm_config_k, k_lds_write_window, k_dram_window);

        // K0/V0 landed, and (dscnt 0 before the barrier) every wave has finished reading Q, so
        // iteration 0 may overwrite V1.
        s_wait_tensorcnt_barrier<PagedTransfer::kKPrefetchTensorCount, 0>();
        // K row group 0 of tile 0, the operand of QK stage 0.
        KSuTile k_su0_tile;
        {
            const uint32_t k0_base =
                KRead::LaneBase(make_k_su_window(static_cast<KDataType* __restrict__>(smem_ptrk0)));
            static_for<0, k_lds_insts, 1>{}([&](auto access) {
                KRead::template Load<decltype(access)::value>(k_su0_tile, k0_base);
            });
        }

        __builtin_amdgcn_sched_barrier(0);

        auto mainloop = [&](KDataType* __restrict__ k_lds_write_ptr,
                            KDataType* __restrict__ k_lds_read_ptr,
                            VDataType* __restrict__ v_lds_write_ptr,
                            VDataType* __restrict__ v_lds_read_ptr) {
            // LDS read bases of the tile, formed in begin_iteration (Policy::MakeTileBases).
            typename Policy::TileBases bases{};

            VTile v_tile;
            float pv_output_scale_m0 = 1.0f;
            float pv_output_scale_m1 = 1.0f;
            v_lds_read_window.set_bottom_tensor_view_data_ptr(v_lds_read_ptr);

            auto run_pv = [&](const auto& p_tile) {
                using WarpGemm    = typename PvBlockGemm::WarpGemm;
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
                constexpr index_t kNumN             = Policy::OutputFragments::kNumN;

                auto emit_wmma = [&](auto, auto wmma) {
                    // Row ordinal = output fragment (one PV stage).
                    constexpr index_t ordinal     = decltype(wmma)::value;
                    constexpr index_t d_msb       = ordinal / kNumN;
                    constexpr index_t n           = ordinal % kNumN;
                    constexpr index_t m_iter      = d_msb / 2;
                    constexpr index_t v_msb       = d_msb % 2;
                    constexpr index_t full_n_iter = n * 2 + v_msb;

                    AWarpTensor a_warp_tensor;
                    a_warp_tensor.get_thread_buffer() = p_tile.get_y_sliced_thread_data(
                        merge_sequences(sequence<0, m_iter>{}, a_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

                    BWarpTensor b_warp_tensor;
                    b_warp_tensor.get_thread_buffer() = v_tile.get_y_sliced_thread_data(
                        merge_sequences(sequence<0, full_n_iter>{}, b_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));

                    // WMMA operand guard of the row (Policy::PvKeptRows, PvRawFence): keep_v keeps
                    // the V operand of row o allocated.
                    auto keep_v = [&](auto o) {
                        constexpr index_t kO   = decltype(o)::value;
                        constexpr index_t kN   = kO % kNumN;
                        constexpr index_t kMsb = kO / kNumN % 2;
                        BWarpTensor kept;
                        kept.get_thread_buffer() = v_tile.get_y_sliced_thread_data(
                            merge_sequences(sequence<0, kN * 2 + kMsb>{}, b_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));
                        Policy::KeepOperandLive(kept);
                    };
                    // The rows that the wait of row ordinal - 1 drained stay allocated until here.
                    constexpr auto kKeptBefore = Policy::template PvKeptRows<Mode>(ordinal, false);
                    static_for<kKeptBefore.begin, kKeptBefore.end, 1>{}([&](auto q) { keep_v(q); });
                    CWarpTensor c_warp_tensor;
                    c_warp_tensor.get_thread_buffer().template set_as<fp32x8_t>(
                        number<0>{}, o_acc.at(number<ordinal>{}));
                    WarpGemm{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);
                    // The last M iteration (rows from Policy::kPvLastMRow) is the last reader of
                    // the V operand registers.
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
                        kept_p.get_thread_buffer() = p_tile.get_y_sliced_thread_data(
                            merge_sequences(sequence<0, decltype(i_m)::value>{},
                                            a_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));
                        Policy::KeepOperandLive(kept_p);
                    });
                    o_acc.at(number<ordinal>{}) =
                        c_warp_tensor.get_thread_buffer().template get_as<fp32x8_t>(number<0>{});
                };

                // Between the WMMAs: K row group 0 of the next tile.
                Policy::template RunPvStage<Geometry::kPvStages - 1, Mode, KRead>(
                    k_su0_tile, bases, emit_wmma);
                AccumulateQuantizedMass<PvBlockGemm>(
                    p_tile, l, pv_output_scale_m0, pv_output_scale_m1);
            };

            auto begin_iteration = [&]() {
                // The read bases of the tile, ahead of every WMMA: K(i) (QK), V(i) (the QK tail
                // stage) and K(i + 1) (PV).
                bases = Policy::MakeTileBases(make_k_su_window(k_lds_write_ptr),
                                              v_lds_read_window,
                                              make_k_su_window(k_lds_read_ptr));
                __builtin_amdgcn_sched_barrier(0);

                // Reuse of the V LDS buffer: every V(i - 1) read fed a PV(i - 1) WMMA issued before
                // this point, so none is pending. The barrier's count leaves the next K tile's
                // reads (issued in the PV stage) in flight; it counts reads, while s_wait_dscnt
                // counts DS instructions and the compiler pairs the reads, so the reuse rests on
                // the consuming WMMAs, not on the count.
                block_sync_lds<k_lds_insts>();
                // V is prefetched one iteration before its consumer.
                move_tile_window(v_dram_window, {kN0, 0});
                v_lds_write_window.set_bottom_tensor_view_data_ptr(v_lds_write_ptr);
                v_dram_window.set_page_id(page_id1);
                // Policy::kDrainLdsBeforeTdmIssue: own LDS reads complete before the V TDM issue.
                if constexpr(Policy::kDrainLdsBeforeTdmIssue)
                    s_wait_dscnt<0>();
                v_dram_window.tdm_issue_lean(tdm_config_v, v_lds_write_window, v_lean_issue);
            };

            auto run_qk = [&] {
                // STAGE 1, QK gemm
                clear_tile(s_acc); // initialize C
                constexpr auto gemm_0_su = Policy::template GetQKBlockGemmSu<Problem>();
                auto s_acc_su            = gemm_0_su.MakeCBlockTile();
                auto k_su_tile           = k_su0_tile;
                static_for<0, Geometry::kQkStages, 1>{}([&](auto i_su) {
                    clear_tile(s_acc_su);
                    if constexpr(decltype(i_su)::value < Geometry::kQkStages - 1)
                    {
                        KSuTile k_su_tile_next;
                        Policy::template RunQkStage<decltype(i_su)::value, Mode, KRead>(
                            gemm_0_su, s_acc_su, q_tile, k_su_tile, k_su_tile_next, bases);
                        k_su_tile = k_su_tile_next;
                    }
                    else
                    {
                        Policy::template RunQkStage<Geometry::kQkStages - 1, Mode, VRead>(
                            gemm_0_su, s_acc_su, q_tile, k_su_tile, v_tile, bases);
                    }
                    set_slice_tile(
                        s_acc,
                        s_acc_su,
                        sequence<0, decltype(i_su)::value * Geometry::kQkSuColumns>{},
                        sequence<kM0, (decltype(i_su)::value + 1) * Geometry::kQkSuColumns>{});
                });
            };

            auto process_score = [&]() {
                // The split tail predicate (the columns of the last tile past
                // physical_seqlen_k_end). In the tiles of kMaskLoopTrims the key range ends at
                // seqlen_k: the loop's last tile is an edge tile, and the mask predicate masks its
                // columns past seqlen_k, so the split predicate is redundant there.
                constexpr bool kSplitTailCheck = kHasUnevenSplits && !kMaskLoopTrims;
                // V's window already points one prefetched tile ahead.
                const auto k_origin = make_tuple(v_dram_window.get_window_origin().at(I0) - kN0, 0);
                [[maybe_unused]] bool need_split_check = false;
                if constexpr(kSplitTailCheck)
                {
                    const bool needs_tail_predicate = k_origin.at(I0) + kN0 > physical_seqlen_k_end;
                    need_split_check                = remaining_loops == 1 && needs_tail_predicate;
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
                        set_tile_if(
                            s_acc, -numeric<SMPLComputeDataType>::infinity(), [&](auto tile_idx) {
                                const auto row = q_origin.at(I0) + tile_idx.at(I0);
                                const auto col = k_origin.at(I0) + tile_idx.at(I1);
                                return Policy::template IsMaskedOut<Problem>(mask, row, col);
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
                            SoftmaxPolicy::template RunSplitSoftmaxPart01<Problem>(
                                s_new, m, scale_s, delta_m0, delta_m1);
                        },
                        // Policy::DeferTensorReady is false: K(i + 1) is waited at NextK.
                        []() {},
                        // Part 2 below rescales the output inside its exp stream.
                        [](auto) {},
                        [&](auto) {
                            SoftmaxPolicy::template RunSplitSoftmaxPart2AndGetScale<
                                Problem,
                                !kQuantizedRowSum>(s_new,
                                                   m,
                                                   l,
                                                   scale_s,
                                                   delta_m0,
                                                   delta_m1,
                                                   pv_output_scale_m0,
                                                   pv_output_scale_m1,
                                                   o_acc);
                        });
                    return MakePForGemm1<PvBlockGemm>(s_new);
                }();
                return p_tile;
            };

            auto load_next_k = [&]() {
                // Fixed boundary: the next K TDM transfer follows softmax. The page-id slot read of
                // the BF16 page tile compiles out with the page-id batch; its async wait stays.
                s_wait_asynccnt<0>();
                // The K tile issued in the previous iteration must be complete; only the V tile
                // issued after it may stay in flight. The barrier below then publishes that K
                // tile to every wave. The K buffer the transfer below overwrites has no read
                // pending: every K(i) read fed a QK WMMA issued before this point. The barrier's
                // count leaves the V tile's reads (issued in the last QK stage) in flight; it
                // counts reads, not the paired DS instructions, so the reuse rests on those WMMAs.
                s_wait_tensorcnt<PagedTransfer::kVPrefetchTensorCount>();
                block_sync_lds<v_lds_insts>();
                // The K transfer here feeds iteration i+2, beyond the already prefetched K1.
                move_tile_window(k_dram_window, {kN0, 0});
                k_lds_write_window.set_bottom_tensor_view_data_ptr(k_lds_write_ptr);
                k_dram_window.set_page_id(page_id2);
                // Policy::kDrainLdsBeforeTdmIssue: own LDS reads complete before the K TDM issue.
                if constexpr(Policy::kDrainLdsBeforeTdmIssue)
                    s_wait_dscnt<0>();
                k_dram_window.tdm_issue_lean(tdm_config_k, k_lds_write_window, k_lean_issue);
            };

            auto advance_iteration = [&]() {
                // Publishes V(i + 1) for the QK tail stage of the next iteration.
                s_wait_tensorcnt_barrier<PagedTransfer::kKPrefetchTensorCount,
                                         kKPrefetchBarrierDsCount>();
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

        do
        {
            auto k_lds_write_ptr = even_buffer ? static_cast<KDataType* __restrict__>(smem_ptrk0)
                                               : static_cast<KDataType* __restrict__>(smem_ptrk1);
            auto k_lds_read_ptr  = even_buffer ? static_cast<KDataType* __restrict__>(smem_ptrk1)
                                               : static_cast<KDataType* __restrict__>(smem_ptrk0);
            auto v_lds_write_ptr = even_buffer ? static_cast<VDataType* __restrict__>(smem_ptrv1)
                                               : static_cast<VDataType* __restrict__>(smem_ptrv0);
            auto v_lds_read_ptr  = even_buffer ? static_cast<VDataType* __restrict__>(smem_ptrv0)
                                               : static_cast<VDataType* __restrict__>(smem_ptrv1);
            mainloop(k_lds_write_ptr, k_lds_read_ptr, v_lds_write_ptr, v_lds_read_ptr);
            even_buffer = !even_buffer;
            --remaining_loops;
        } while(remaining_loops > 0);

        // Reconstruct the CK tensor only at the final normalization boundary.
        auto output = Policy::OutputFragments::template Reconstruct<OaccBlockTileType>(o_acc);
        constexpr auto o_spans = decltype(output)::get_distributed_spans();

        // O / l: l is the mass of the FP8 P that gemm_1 consumed, so kPScale cancels. A fully
        // masked row has l = 0 and stores zeros.
        sweep_tile_span(o_spans[I0], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            const float scale    = [&]() {
                if constexpr(FmhaMask::IsMasking)
                    return l[i_idx] == 0.f ? 0.f : 1 / l[i_idx];
                else
                    return 1 / l[i_idx];
            }();
            sweep_tile_span(o_spans[I1],
                            [&](auto idx1) { output(make_tuple(idx0, idx1)) *= scale; });
        });

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
};

} // namespace ck_tile
