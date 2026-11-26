// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/block_attention_quant_scale_enum.hpp"
#include "ck_tile/ops/fmha/block/block_masking.hpp"
#include "ck_tile/ops/fmha/detail/fmha_fwd_kernel_path_markers.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_sched_fp8_policy.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_output.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_executor.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_pipeline_common.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_fp8_schedule.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_softmax.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_fp8_features.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_fp8_merged_rows.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_fp8_output_store.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_fp8_quant_scale.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_fp8_tile_index.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"

namespace ck_tile {

// Hand-scheduled gfx125 TDM pipeline for FP8 D64/D128/D192/D256 (K64 QK WMMA at D64/D192).
template <typename Problem_,
          typename Policy_        = FmhaTdmSchedFp8PolicyFor<Problem_>,
          bool EnableVirtualSink_ = true>
struct BlockFmhaPipelineQRKSVSTdmSchedFp8
    : FmhaTdmSchedFp8OutputStore<typename remove_cvref_t<Policy_>::Geometry,
                                 remove_cvref_t<typename Problem_::ODataType>>
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
    using SplitSoftmax          = typename Policy::ScoreSoftmax;
    using Softmax               = typename Policy::Softmax;
    using Schedule              = typename Policy::Schedule;
    using Mode                  = FmhaTdmSchedMode<Problem, Policy>;
    static_assert(Mode::kIsLegal, "FP8 qr_tdm_sched schedule geometry is incompatible");

    using BlockFmhaShape             = remove_cvref_t<typename Problem::BlockFmhaShape>;
    using VLayout                    = remove_cvref_t<typename BlockFmhaShape::VLayout>;
    static constexpr bool kQLoadOnce = true; // if q_tile load whole block length (hdim) at once
    static_assert(kQLoadOnce == Policy::QLoadOnce);
    // K TDM tiles always span the whole head dim.
    static constexpr bool kKLoadOnce = true;

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
    static constexpr bool kHasSink           = Problem::kHasSink;
    static constexpr bool kEnableVirtualSink = EnableVirtualSink_;
    static constexpr auto QScaleEnum         = Problem::QScaleEnum;
    // Query rows fold the heads of one KV head: row = g * seqlen_q + s.
    static constexpr bool kMergeHeads = Problem::kMergeNumHeadGroupsSeqLenQ;

    static_assert(CK_TILE_FMHA_FWD_FAST_EXP2,
                  "FP8 qr_tdm_sched: every softmax site calls exp2, so log2(e) must be folded "
                  "into scale_s");
    static_assert(BiasEnum == BlockAttentionBiasEnum::NO_BIAS ||
                      BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS,
                  "FP8 qr_tdm_sched implements elementwise bias only, not alibi");
    static_assert(!kHasLogitsSoftCap, "FP8 qr_tdm_sched does not implement logits soft cap");
    static_assert(!kHasDropout, "FP8 qr_tdm_sched does not support dropout");
    static_assert(QScaleEnum == BlockAttentionQuantScaleEnum::NO_SCALE ||
                      QScaleEnum == BlockAttentionQuantScaleEnum::PERTENSOR ||
                      QScaleEnum == BlockAttentionQuantScaleEnum::PERHEAD ||
                      QScaleEnum == BlockAttentionQuantScaleEnum::BLOCKSCALE ||
                      QScaleEnum == BlockAttentionQuantScaleEnum::MX,
                  "FP8 qr_tdm_sched: unsupported quantization granularity");
    static_assert(!kMergeHeads || (!kIsGroupMode && BiasEnum == BlockAttentionBiasEnum::NO_BIAS),
                  "merged GQA rows need batch mode and no bias");
    // Derived switches, each described in the features table.
    using Features                   = FmhaTdmSchedFp8Features<Problem, Policy, kEnableVirtualSink>;
    static constexpr bool kQuantized = Features::kQuantized;
    static constexpr bool kBlockScale          = Features::kBlockScale;
    static constexpr bool kMx                  = Features::kMx;
    static constexpr bool kD256                = Features::kD256;
    static constexpr bool kRowScale            = Features::kRowScale;
    static constexpr float kPScale             = Features::kPScale;
    static constexpr bool kNormalizeQuantizedP = Features::kNormalizeQuantizedP;
    static constexpr bool kQuantizedRowSum     = Features::kQuantizedRowSum;
    static constexpr bool kStochasticP         = Features::kStochasticP;
    static constexpr bool kLdsSegSplit         = Features::kLdsSegSplit;
    static constexpr bool kUnpinnedMaskWidth   = Features::kUnpinnedMaskWidth;
    static constexpr bool kBiasViaTdm          = Features::kBiasViaTdm;

    static constexpr index_t kAlignmentQ    = Policy::template GetAlignmentQ<Problem>();
    static constexpr index_t kAlignmentK    = Policy::template GetAlignmentK<Problem>();
    static constexpr index_t kAlignmentV    = Policy::template GetAlignmentV<Problem>();
    static constexpr index_t kAlignmentO    = Policy::template GetAlignmentO<Problem>();
    static constexpr index_t kAlignmentOacc = Policy::template GetAlignmentO<Problem>();
    static constexpr index_t kAlignmentBias =
        kPadSeqLenK ? 1 : Policy::template GetAlignmentBias<Problem>();
    static constexpr index_t kAlignmentRandVal =
        kPadSeqLenK ? 1 : Policy::template GetAlignmentRandVal<Problem>();

    static_assert(Policy::template IsSupportedProblem<Problem>(),
                  "FP8 qr_tdm_sched requires an exact supported FP8 geometry");

    // IsSupportedProblem above pins Geometry == FmhaTdmSchedGeometryFor<Problem>.
    using Geometry = typename Policy::Geometry;
    static_assert(Geometry::kM == kM0 && Geometry::kN == kN0 && Geometry::kDv == kN1);
    static_assert(!kLdsSegSplit || kFmhaTdmSchedFp8LdsSegLayout<Problem>,
                  "the segmented LDS layout supports dense D256 problems only");

    // D256 streams V in PV: row r pairs V chunk r / 2 with P block r % 2; others run fragment r.
    CK_TILE_HOST_DEVICE static constexpr index_t PvOrdinal(index_t row)
    {
        if constexpr(!kD256)
            return row;
        constexpr index_t kNumN = Policy::OutputFragments::kNumN;
        const index_t chunk     = row / 2;
        return ((row % 2) * 2 + chunk % 2) * kNumN + chunk / 2;
    }

    static constexpr const char* name = "qr_tdm_sched";
    using Common                      = FmhaTdmSchedPipelineCommon;

    static constexpr bool kUsesUntransposedVKernelPath = true;
    static constexpr bool kUsesTdmAffineDramPath       = true;
    static constexpr bool kUsesFixedSegmentedLdsArena  = true;
    static constexpr bool kTakesQuantDescales          = true;
    static constexpr bool kTakesMxDescales = QScaleEnum == BlockAttentionQuantScaleEnum::MX;
    static constexpr bool kSupportsMergedQueryHeads = true;
    static constexpr bool kRequiresExactHeadDims    = true;
    static constexpr auto kCausalTileOrder = detail::FmhaFwdCausalTileOrder::kGlobalLongestFirst;
    static constexpr index_t kBlockPerCu   = Policy::kBlockPerCu;

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
        return Policy::GetLdsArenaSize();
    }

    using TileIndex  = FmhaTdmSchedFp8TileIndex;
    using Pins       = FmhaTdmSchedFp8Pins;
    using MergedRows = FmhaTdmSchedFp8MergedRows;
    using QuantScale = FmhaTdmSchedFp8QuantScale<Geometry, kBlockSize, kK1, kQKHeaddim>;
    // The kernel stores O through the inherited StoreOutput and kOLdsBytes.
    using OutputStore = FmhaTdmSchedFp8OutputStore<Geometry, ODataType>;
    using BiasLoad    = FmhaTdmSchedFp8BiasLoad<Geometry>;
    static_assert(!kBiasViaTdm || std::is_same_v<BiasDataType, float>);
    // Bias TDM transfers per wavefront and tile, counted into every later tensorcnt wait.
    static constexpr index_t kBiasTdmOps = kBiasViaTdm ? BiasLoad::kTdmOps : 0;
    // fp32 bias elements per 16-byte LDS read.
    static constexpr index_t kBiasElemsPerRead = 16 / static_cast<index_t>(sizeof(float));
    // The bias tile shares the O staging LDS, which only the epilogue writes.
    static constexpr bool kOwnsOutputLds = kBiasViaTdm;
    static constexpr index_t kOutputLdsBytes =
        kBiasViaTdm ? max(OutputStore::kOLdsBytes, BiasLoad::kLdsBytes) : OutputStore::kOLdsBytes;
    CK_TILE_DEVICE static void* OutputLds()
    {
        alignas(16) __shared__ char buf[kOutputLdsBytes];
        return buf;
    }

    static_assert(!kBlockScale || (QuantScale::kPScaleGranularity == QuantScale::kKVScaleAlign &&
                                   kN0 % QuantScale::kKVScaleAlign == 0 &&
                                   Geometry::kQkStages == QuantScale::kKVScaleGroups));
    static_assert(!kMx || ((QuantScale::kMxStepScaleBytes == 2 ||
                            QuantScale::kMxStepScaleBytes == QuantScale::kScaleBytes) &&
                           QuantScale::kMxQkScaleBytes ==
                               QuantScale::kMxStepScaleBytes * QuantScale::kMxQkScaleDwords &&
                           QuantScale::kPScaleGranularity == QuantScale::kMxGranularity),
                  "MX needs whole K64/K128 QK steps and a 32-element P/V scale block");
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
    static_assert(
        !kMx || QuantScale::kMxPairedK64 ==
                    Policy::template kPairsScaledK64<
                        remove_cvref_t<decltype(Policy::template GetQKBlockGemmSu<Problem>())>>,
        "MX Q/K scale pairing must match the paired K128 QK WMMA");
#endif
    // Read by the kernel's host-side argument checks.
    static constexpr index_t kKVScaleAlign   = QuantScale::kKVScaleAlign;
    static constexpr index_t kMxQkScaleAlign = QuantScale::kMxQkScaleAlign;
    using KVDescales                         = typename QuantScale::KVDescales;
    using KVScaleCursor                      = typename QuantScale::KVScaleCursor;
    using KVScaleOffsets                     = typename QuantScale::KVScaleOffsets;
    using MxCopyOffsets                      = typename QuantScale::MxCopyOffsets;
    using MxDescales                         = FmhaTdmSchedFp8MxDescales;
    using MergedHeads                        = FmhaTdmSchedFp8MergedHeads;

    // Host-side argument checks beyond the kernel's generic ones.
    template <typename Kargs>
    CK_TILE_HOST static bool IsSupportedKargs(const Kargs& kargs)
    {
        const bool log    = ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING));
        const auto reject = [&](const char* reason) {
            if(log)
                CK_TILE_ERROR("FMHA fwd FP8 qr_tdm_sched: ", reason);
            return false;
        };
        if constexpr(!kEnableVirtualSink)
        {
            if(kargs.sink_ptr != nullptr)
                return reject("sink_ptr is not supported.");
        }
        if constexpr(kMergeHeads)
        {
            // Merged heads must form one row stride: a single query, or seqlen_q-row heads.
            const auto s  = kargs.seqlen_q;
            bool foldable = s == 1 || (kargs.nhead_stride_q == s * kargs.stride_q &&
                                       kargs.nhead_stride_o == s * kargs.stride_o);
            if constexpr(kStoreLSE)
                foldable = foldable && (s == 1 || kargs.nhead_stride_lse == s);
            if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::MX)
                foldable = foldable &&
                           (s == 1 || kargs.nhead_stride_q_descale == s * kargs.stride_q_descale);
            if(!foldable)
                return reject("merged query heads need foldable Q/O/LSE/scale strides.");
            if(s * kargs.nhead_ratio_qk > kM0)
                return reject("merged query heads must fit one tile of query rows.");
            if(kargs.cu_seqlen_q_ptr != nullptr || kargs.cu_seqlen_k_ptr != nullptr)
                return reject("merged query heads do not support cu_seqlen_q/k_ptr.");
        }
        if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::MX)
        {
            // Q/K E8M0 rows are read per lane as int32s (D128/D256) or an int16 (D64).
            const auto bits  = [](auto v) { return static_cast<uint64_t>(v); };
            uint64_t offsets = bits(reinterpret_cast<uintptr_t>(kargs.q_descale_ptr)) |
                               bits(reinterpret_cast<uintptr_t>(kargs.k_descale_ptr)) |
                               bits(kargs.stride_q_descale) | bits(kargs.stride_k_descale) |
                               bits(kargs.nhead_stride_q_descale) |
                               bits(kargs.nhead_stride_k_descale);
            if constexpr(!kIsGroupMode)
                offsets |= bits(kargs.batch_stride_q_descale) | bits(kargs.batch_stride_k_descale);
            if(offsets % kMxQkScaleAlign != 0)
                return reject("MX Q/K descale pointers and strides must align to a scale row.");
            if constexpr(kIsGroupMode)
            {
                if(kargs.seqstart_v_scale_ptr == nullptr)
                    return reject("group mode MX needs seqstart_v_scale_ptr.");
            }
        }
        return true;
    }

    // MX descales of one workgroup; offsets are the kernel's per-head/batch scale offsets.
    template <typename Kargs>
    CK_TILE_DEVICE static MxDescales MakeMxDescales(const Kargs& kargs,
                                                    long_index_t offset_q,
                                                    long_index_t offset_k,
                                                    long_index_t offset_v,
                                                    bool rows_by_head,
                                                    index_t seqlen_q)
    {
        MxDescales mx;
        mx.q_ptr    = reinterpret_cast<const e8m0_t*>(kargs.q_descale_ptr) + offset_q;
        mx.k_ptr    = reinterpret_cast<const e8m0_t*>(kargs.k_descale_ptr) + offset_k;
        mx.v_ptr    = reinterpret_cast<const e8m0_t*>(kargs.v_descale_ptr) + offset_v;
        mx.stride_q = rows_by_head ? kargs.nhead_stride_q_descale : kargs.stride_q_descale;
        mx.stride_k = kargs.stride_k_descale;
        mx.stride_v = kargs.stride_v_descale;
        mx.seqlen_q = seqlen_q;
        return mx;
    }

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
    // Static LDS of one workgroup: arena, O staging, and MX or row scales.
    static constexpr index_t kStaticLdsBytes =
        Policy::GetLdsArenaSize() + kOutputLdsBytes +
        (kMx ? 2 * QuantScale::kMxLdsTileDwords * static_cast<index_t>(sizeof(int32_t)) : 0) +
        (kRowScale ? 2 * kBlockSize * static_cast<index_t>(sizeof(float)) : 0);
    static constexpr index_t kLdsWorkgroupsPerCu =
        static_cast<index_t>(get_lds_size()) / kStaticLdsBytes;
    // Four wavefronts: kBlockPerCu workgroups fit. One: LDS caps residency below the VGPR bound.
    static_assert(Geometry::kWaves == 1
                      ? kLdsWorkgroupsPerCu >= 1 && kLdsWorkgroupsPerCu <= 4 * kBlockPerCu
                      : kLdsWorkgroupsPerCu >= kBlockPerCu);
#endif
    CK_TILE_DEVICE static int32_t* MxScaleLds(index_t tile)
    {
        __shared__ int32_t buf[2 * QuantScale::kMxLdsTileDwords];
        return buf + (tile & 1) * QuantScale::kMxLdsTileDwords;
    }

    // Merged row softmax scales live in LDS, one pair per lane, so they hold no VGPRs in the loop.
    using RowScalePair = float __attribute__((ext_vector_type(2)));
    CK_TILE_DEVICE static RowScalePair* RowScaleLds()
    {
        __shared__ RowScalePair buf[kBlockSize];
        return buf + get_thread_id();
    }

    // The fence keeps each read where it is used instead of hoisted out of the loop.
    CK_TILE_DEVICE static RowScalePair LoadRowScales()
    {
        Pins::CompilerMemoryFence();
        return *RowScaleLds();
    }

    // TDM instructions one load_tile_tdm issues per wavefront: one per precomputed coordinate.
    template <typename Window>
    static constexpr index_t kTdmInstsPerLoad = -1;
    template <typename B, typename L, typename D, index_t NumCoord>
    static constexpr index_t
        kTdmInstsPerLoad<tile_window_with_static_distribution<B, L, D, NumCoord>> = NumCoord;

    // Recomputes the clamped row stride in SGPRs at each issue instead of a VGPR v_readfirstlane.
    template <typename LdsWindow, typename DramWindow>
    CK_TILE_DEVICE static void
    LoadTileTdmInLoop(const TDMConfig& config, LdsWindow& lds_window, const DramWindow& dram)
    {
        dram.tdm_load_to_lds(config,
                             lds_window,
                             make_null_tile_window(dram.get_window_lengths()),
                             number<-1>{},
                             bool_constant<false>{},
                             bool_constant<true>{});
    }

    // Stochastic-rounding seeds of the two query rows a lane holds.
    using PSeeds = array<uint32_t, 2>;

    // Re-pack gemm_0 C (M-outer) into gemm_1 A (K-outer) in-thread; lanes already align.
    template <typename Gemm1, typename PComputeTensor>
    CK_TILE_DEVICE static auto MakePForGemm1(const PComputeTensor& p_compute,
                                             [[maybe_unused]] const PSeeds& p_seeds)
    {
        static_assert(Geometry::PRepackLayout::kInThread &&
                          Gemm1::MIterPerWarp == Geometry::kPvMIter,
                      "P re-pack requires gemm_0 C^T and gemm_1 A to share lanes");
        auto p_tile = make_static_distributed_tensor<PDataType>(
            Policy::template MakePRegTileDistribution<Problem>());
        const auto p_src = [&] {
            if constexpr(kQuantized)
            {
                // The scaled convert divides by its power-of-two scale, so this stores 256 * P.
                auto t =
                    make_static_distributed_tensor<PDataType>(p_compute.get_tile_distribution());
                constexpr index_t kVec = 8;
                static_assert(decltype(t)::get_thread_buffer_size() % kVec == 0);
                static_for<0, decltype(t)::get_thread_buffer_size() / kVec, 1>{}([&](auto i) {
                    fp32x8_t v;
                    static_for<0, kVec, 1>{}([&](auto e) {
                        v[e()] = p_compute.get_thread_buffer()[number<i * kVec + e>{}];
                    });
                    // The first half of the converts holds the lane's first row (M-outer).
                    constexpr index_t kRow =
                        Features::kScalarPSeed
                            ? 0
                            : decltype(i)::value * kVec * 2 / decltype(t)::get_thread_buffer_size();
                    if constexpr(kStochasticP)
                        t.get_thread_buffer().template set_as<fp8x8_t>(
                            i, QuantScale::QuantizePStochastic(v, p_seeds[kRow], 1.0f / kPScale));
                    else
                        t.get_thread_buffer().template set_as<fp8x8_t>(
                            i, fp32x8_to_fp8x8(v, 1.0f / kPScale));
                });
                return t;
            }
            else // NO_SCALE skips kPScale, so small P lose precision to e4m3 subnormals.
                return cast_tile<PDataType>(p_compute);
        }();
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

    // Q, K and V reach the fixed LDS arena by TDM; K and V are double buffered.
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
        [[maybe_unused]] PositionEncoding position_encoding,
        float scale_s,
        void* __restrict__ smem_ptrk0,
        void* __restrict__ smem_ptrk1,
        void* __restrict__ smem_ptrv0,
        void* __restrict__ smem_ptrv1,
        float sink_v,
        [[maybe_unused]] const float* k_descale_ptr,
        [[maybe_unused]] const float* v_descale_ptr,
        [[maybe_unused]] index_t block_scale_size_kv,
        [[maybe_unused]] float v_descale,
        [[maybe_unused]] const MxDescales& mx,
        [[maybe_unused]] const MergedHeads& merged) const
    {
        static_assert(
            std::is_same_v<QDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                std::is_same_v<KDataType, remove_cvref_t<typename KDramBlockWindowTmp::DataType>> &&
                std::is_same_v<VDataType, remove_cvref_t<typename VDramBlockWindowTmp::DataType>>,
            "wrong!");

        // Q, K, and V use TDM global-to-LDS transfers; V is read with transposed LDS loads.
        static_assert(kM0 == QDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                          kSubQKHeaddim == QDramBlockWindowTmp{}.get_window_lengths()[I1] &&
                          kN0 == KDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                          kK0 == KDramBlockWindowTmp{}.get_window_lengths()[I1] &&
                          kN1 == VDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                          kK1 == VDramBlockWindowTmp{}.get_window_lengths()[I1] &&
                          kM0 == BiasDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                          kN0 == BiasDramBlockWindowTmp{}.get_window_lengths()[I1],
                      "wrong!");
        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1 = Policy::template GetPVBlockGemm<Problem>();

        using SaccBlockTileType                   = decltype(gemm_0.MakeCBlockTile());
        auto s_acc                                = SaccBlockTileType{};
        constexpr bool kBandMask                  = Features::kBandMask;
        [[maybe_unused]] const auto s_lane_origin = [] {
            if constexpr(kBandMask)
                return TileIndex::TileLaneOrigin<SaccBlockTileType>();
            else
                return 0;
        }();

        const auto f_max        = [](auto e0, auto e1) { return max(e0, e1); };
        using OaccBlockTileType = decltype(gemm_1.MakeCBlockTile());

        using PvBlockGemm   = remove_cvref_t<decltype(gemm_1)>;
        using PvWarpGemm    = typename PvBlockGemm::WarpGemm;
        using PvCWarpTensor = typename PvWarpGemm::CWarpTensor;
        static_assert(PvBlockGemm::KIterPerWarp == 1 && PvBlockGemm::MIterPerWarp == 2 &&
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
        if constexpr(Geometry::kN == Geometry::kDv)
            static_assert(std::is_same_v<typename SaccBlockTileType::StaticTileDistribution,
                                         typename OaccBlockTileType::StaticTileDistribution>,
                          "equal-width score/output distributions must preserve the fragment "
                          "reconstruction map");

        auto o_acc = Policy::OutputFragments::MakeZero();

        using SBlockTileType = decltype(cast_tile<SMPLComputeDataType>(OaccBlockTileType{}));

        using MLBlockTileType = decltype(block_tile_reduce<SMPLComputeDataType>(
            SBlockTileType{}, sequence<1>{}, f_max, SMPLComputeDataType{0}));

        auto m = MLBlockTileType{};
        auto l = MLBlockTileType{};
        // Row sum of kPScale * q(P) beside the exact l, for instances that also need the exact l.
        [[maybe_unused]] auto l_quant = MLBlockTileType{};
        if constexpr(kNormalizeQuantizedP)
            clear_tile(l_quant);

        // A virtual sink has V=0; merge it only at normalization so it cannot shift P.
        set_tile(m, -numeric<SMPLComputeDataType>::infinity());
        clear_tile(l);

        const auto q_origin = q_dram_block_window_tmp.get_window_origin();

        // Softmax scale of each lane row: scale_s, or the row head's own for merged descales.
        [[maybe_unused]] auto row_scale =
            make_static_distributed_tensor<SMPLComputeDataType>(m.get_tile_distribution());
        if constexpr(kRowScale)
        {
            sweep_tile_span(decltype(row_scale)::get_distributed_spans()[I0], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                row_scale(i_idx)     = MergedRows::MergedRowScale<kBlockScale>(
                    merged, q_origin.at(I0) + TileIndex::LaneRowOf<decltype(m)>(i_idx));
            });
            *RowScaleLds() =
                RowScalePair{row_scale.get_thread_buffer()[I0], row_scale.get_thread_buffer()[I1]};
        }
        const auto scale_of = [&](auto i_idx) {
            if constexpr(kRowScale)
                return row_scale[i_idx];
            else
                return scale_s;
        };

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

        // Seeded by query position and KV tile, so merged GQA rows match their unmerged heads.
        [[maybe_unused]] PSeeds p_seeds{};
        if constexpr(kStochasticP)
        {
            static_assert(MLBlockTileType::get_thread_buffer_size() == PSeeds::size());
            const index_t first_tile = num_sink_loop > 0 ? 0 : logical_seqlen_k_start / kN0;
            index_t r                = 0;
            sweep_tile_span(decltype(m)::get_distributed_spans()[I0], [&](auto idx0) {
                index_t row = q_origin.at(I0) + TileIndex::LaneRowOf<decltype(m)>(make_tuple(idx0));
                if constexpr(kMergeHeads)
                    row = merged.seqlen_q == 1 ? 0 : row % merged.seqlen_q;
                if constexpr(Features::kScalarPSeed)
                    row = __builtin_amdgcn_readfirstlane(row);
                p_seeds(r++) = QuantScale::PSeed(row, first_tile);
            });
        }

        // No KV tile to visit: write the empty-row LSE and a zero O.
        if(const index_t logical_num_total_loop =
               integer_divide_ceil(logical_seqlen_k_end - logical_seqlen_k_start, kN0) +
               num_sink_loop;
           logical_num_total_loop <= 0)
        {
            if constexpr(kStoreLSE)
            {
                auto lse_acc =
                    make_static_distributed_tensor<LSEDataType>(m.get_tile_distribution());

                if(__builtin_isinf_sign(sink_v) >= 0)
                {
                    if constexpr(kMergeHeads)
                        sweep_tile_span(decltype(lse_acc)::get_distributed_spans()[I0],
                                        [&](auto idx0) {
                                            constexpr auto i_idx = make_tuple(idx0);
                                            lse_acc(i_idx)       = MergedRows::MergedSinkLogit(
                                                merged,
                                                TileIndex::LaneRowOf<decltype(m)>(i_idx),
                                                scale_of(i_idx));
                                        });
                    else
                        set_tile(lse_acc, SMPLComputeDataType{sink_v * scale_s});
                }
                else
                {
                    set_tile(lse_acc, -numeric<SMPLComputeDataType>::infinity());
                }

                store_tile(lse_acc_dram_window_tmp, lse_acc);
            }

            // O is still zero here and no TDM transfer has been issued yet.
            return Policy::OutputFragments::template Reconstruct<OaccBlockTileType>(o_acc);
        }

        // TDM configs for Q/K/V; LDS padding comes from the policy, no cluster multicast.
        TDMConfig tdm_config_q;
        TDMConfig tdm_config_k;
        TDMConfig tdm_config_v;
        Common::SetTdmPadding(tdm_config_q, Policy::template GetLdsPaddingConfigQ<Problem>());
        Common::SetTdmPadding(tdm_config_k, Policy::template GetLdsPaddingConfigK<Problem>());
        Common::SetTdmPadding(tdm_config_v, Policy::template GetLdsPaddingConfigV<Problem>());

        // Q tile in LDS: staged in V1, which the iteration-0 V prefetch overwrites first.
        auto q_dram_window = make_tile_window(
            q_dram_block_window_tmp, Policy::template MakeQDramTileDistribution<Problem>());

        auto q_lds_view = make_tensor_view<address_space_enum::lds>(
            static_cast<QDataType*>(smem_ptrv1),
            Policy::template MakeQLdsBlockDescriptor<Problem>());

        auto q_lds_store_window = make_tile_window(
            q_lds_view, Policy::template MakeQLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});

        auto q_lds_read_window =
            make_tile_window(q_lds_view,
                             Policy::template MakeQLdsBlockDescriptor<Problem>().get_lengths(),
                             {0, 0},
                             Policy::template MakeQRegTileDistribution<Problem>());

        // Lane l carries the scales of WMMA row/column l % 16 (both K halves agree).
        [[maybe_unused]] const index_t mx_lane = get_lane_id() % Geometry::QkWmma::kM;

        // K tile in LDS (sink-aware start)
        const auto kv_load_start =
            (sink_seq_end == 0 && logical_seqlen_k_start > 0) ? logical_seqlen_k_start : 0;

        // Without bias this is a null window; its no-op moves keep those instances unchanged.
        const auto bias_origin = bias_dram_block_window_tmp.get_window_origin();
        auto bias_dram_window =
            make_tile_window(bias_dram_block_window_tmp.get_bottom_tensor_view(),
                             bias_dram_block_window_tmp.get_window_lengths(),
                             {bias_origin.at(number<0>{}), kv_load_start},
                             gemm_0.MakeCBlockTile().get_tile_distribution());
        // Bias by TDM: tensor base, row stride, real extents, and the column of the next tile.
        [[maybe_unused]] const float* bias_base = nullptr;
        [[maybe_unused]] index_t bias_stride    = 0;
        [[maybe_unused]] index_t bias_rows      = 0;
        [[maybe_unused]] index_t bias_cols      = 0;
        [[maybe_unused]] index_t bias_row       = 0;
        [[maybe_unused]] index_t bias_col       = kv_load_start;
        [[maybe_unused]] float* bias_lds        = nullptr;
        if constexpr(kBiasViaTdm)
        {
            static_assert(TileIndex::IsWaveInterleavedBands<
                              typename SaccBlockTileType::StaticTileDistribution,
                              Geometry::kQkMIter,
                              Geometry::QkWmma::kM,
                              Geometry::kWaves>(),
                          "each wavefront reads back only the bias rows it loaded");
            const auto& bias_view = bias_dram_block_window_tmp.get_bottom_tensor_view();
            const auto& bias_desc = bias_view.get_tensor_descriptor();
            const auto lengths    = detail::tdm_real_lengths(bias_desc);
            bias_base             = bias_view.get_buffer_view().p_data_;
            bias_stride           = bias_desc.calculate_offset(make_multi_index(1, 0));
            bias_rows             = lengths.at(I0);
            bias_cols             = lengths.at(I1);
            bias_row              = bias_dram_block_window_tmp.get_window_origin().at(I0);
            bias_lds              = static_cast<float*>(OutputLds());
        }
        [[maybe_unused]] auto bias_lds_window = [&] {
            if constexpr(kBiasViaTdm)
                return make_tile_window(
                    make_tensor_view<address_space_enum::lds>(
                        bias_lds,
                        make_naive_tensor_descriptor(
                            make_tuple(number<kM0>{}, number<kN0>{}),
                            make_tuple(number<BiasLoad::kRowPitch>{}, number<1>{}),
                            number<kBiasElemsPerRead>{},
                            number<1>{})),
                    make_tuple(number<kM0>{}, number<kN0>{}),
                    {0, 0},
                    gemm_0.MakeCBlockTile().get_tile_distribution());
            else
                return 0;
        }();
        // LDS byte address of this lane's first bias read; every other read is an immediate offset.
        [[maybe_unused]] int32_t bias_lds_base = 0;
        if constexpr(kBiasViaTdm)
        {
            FmhaTdmSchedFp8PinnedWindow<remove_cvref_t<decltype(bias_lds_window)>, 1>::ComputeBases(
                bias_lds_window, &bias_lds_base);
            bias_lds_base = Pins::PinToVgpr(bias_lds_base);
        }
        [[maybe_unused]] auto issue_bias = [&]() {
            BiasLoad::Issue(
                bias_base, bias_stride, bias_rows, bias_cols, bias_row, bias_col, bias_lds);
        };

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

        // Read window of one QK stage's K group over the K buffer at k_lds_ptr.
        const auto make_k_su_read_window = [&](KDataType* __restrict__ k_lds_ptr) {
            auto window =
                make_tile_window(k_lds_view,
                                 make_tuple(number<Geometry::kQkSuColumns>{}, number<kQKHeaddim>{}),
                                 {0, 0},
                                 Policy::template MakeKSuRegTileDistribution<Problem>());
            window.set_bottom_tensor_view_data_ptr(k_lds_ptr);
            return window;
        };

        // V tile in LDS (sink-aware start)
        auto v_dram_window =
            make_tile_window(v_dram_block_window_tmp,
                             {kv_load_start, 0},
                             Policy::template MakeVDramTileDistribution<Problem>());

        auto v_lds_view = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<VDataType* __restrict__>(static_cast<char*>(smem_ptrv0)),
            Policy::template MakeVLdsBlockDescriptor<Problem>());

        auto v_lds_write_window = make_tile_window(
            v_lds_view, Policy::template MakeVLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});

        auto v_lds_read_window =
            make_tile_window(v_lds_view,
                             make_tuple(number<kK1>{}, number<kN1>{}),
                             {0, 0},
                             Policy::template MakeVRegTileDistribution<Problem>());

        const index_t num_total_loop =
            integer_divide_ceil(logical_seqlen_k_end - logical_seqlen_k_start, kN0) + num_sink_loop;
        // The tile range rounds its end up to kN0, so bound the scale index by seqlen_k.
        [[maybe_unused]] const index_t kv_last = mask.GetXTotal() - 1;

        constexpr bool kUseCountdownLoop = Features::kUseCountdownLoop;
        bool even_buffer                 = true;
        index_t i_total_loops            = 0;
        index_t remaining_loops          = num_total_loop;
        constexpr index_t k1_loops       = kN0 / kK1;

        constexpr bool kProgressiveMax = Features::kProgressiveMax;

        static_assert(1 <= k1_loops);
        constexpr index_t kKHalfBytes = Geometry::kN / 2 * Policy::kKPhysicalStride;
        constexpr index_t kVHalfBytes = Geometry::kN / 2 * Policy::kVPhysicalStride;
        // Per-wavefront segment byte offsets; only the segmented layout reads them.
        [[maybe_unused]] index_t seg_write = 0;
        [[maybe_unused]] index_t seg_k_lo  = 0;
        [[maybe_unused]] index_t seg_k_hi  = 0;
        [[maybe_unused]] index_t seg_v_lo  = 0;
        [[maybe_unused]] index_t seg_v_hi  = 0;
        [[maybe_unused]] index_t seg_col   = 0;
        [[maybe_unused]] const auto k_seg  = [](void* p, index_t d) {
            return reinterpret_cast<KDataType* __restrict__>(static_cast<char*>(p) + d);
        };
        [[maybe_unused]] const auto v_seg = [](void* p, index_t d) {
            return reinterpret_cast<VDataType* __restrict__>(static_cast<char*>(p) + d);
        };
        if constexpr(kLdsSegSplit)
        {
            constexpr index_t kSegShift = Policy::kLdsSegShift;
            const index_t seg_wave      = get_warp_id();
            const bool seg_odd          = seg_wave & 1;
            seg_write                   = seg_wave >= 2 ? kSegShift : 0;
            seg_k_lo                    = seg_odd ? kKHalfBytes + kSegShift : 0;
            seg_k_hi                    = seg_odd ? -kKHalfBytes : kSegShift;
            seg_v_lo                    = seg_odd ? kVHalfBytes + kSegShift : 0;
            seg_v_hi                    = seg_odd ? -kVHalfBytes : kSegShift;
            seg_col                     = seg_odd ? kN0 / 2 : 0;
            k_lds_write_window.set_bottom_tensor_view_data_ptr(k_seg(smem_ptrk0, seg_write));
            v_lds_write_window.set_bottom_tensor_view_data_ptr(v_seg(smem_ptrv0, seg_write));
        }
        // Fixed TDM prologue, issued back to back: Q, current K/V, and the look-ahead K1.
        load_tile_tdm(tdm_config_q, q_lds_store_window, q_dram_window);
        load_tile_tdm(tdm_config_k, k_lds_write_window, k_dram_window);
        load_tile_tdm(tdm_config_v, v_lds_write_window, v_dram_window);
        if constexpr(kBiasViaTdm)
            issue_bias();
        move_tile_window(k_dram_window, {kN0, 0});
        // K1 is consumed at iteration 1. A one-tile prefix must jump before K1 loads.
        if constexpr(kHasSink)
        {
            if(num_sink_loop == 1 && num_total_loop > 1)
                move_tile_window(k_dram_window, {logical_seqlen_k_start - sink_seq_end, 0});
        }
        if constexpr(kLdsSegSplit)
            k_lds_write_window.set_bottom_tensor_view_data_ptr(k_seg(smem_ptrk1, seg_write));
        else
            k_lds_write_window.set_bottom_tensor_view_data_ptr(
                static_cast<KDataType* __restrict__>(smem_ptrk1));
        load_tile_tdm(tdm_config_k, k_lds_write_window, k_dram_window);

        // Reads still in flight at the K and V buffer-reuse barriers.
        constexpr index_t k_lds_insts = kFmhaTdmSchedFp8DsWait<Geometry::kKSuLoadCount>;
        constexpr index_t v_lds_insts = kFmhaTdmSchedFp8DsWait<Geometry::kVStageLoadCount>;
        static_assert(v_lds_read_window.get_num_of_access() == Geometry::kVStageLoadCount);

        // The counts below assume one TDM instruction per wavefront per load_tile_tdm.
        static_assert(kTdmInstsPerLoad<decltype(q_dram_window)> == 1 &&
                          kTdmInstsPerLoad<decltype(k_dram_window)> == 1 &&
                          kTdmInstsPerLoad<decltype(v_dram_window)> == 1,
                      "prologue tensorcnt counts assume a single TDM coordinate per window");
        // V0 is waited in Begin, which the sequential schedule runs before Qk reads it.
        static_assert(
            std::is_same_v<Schedule,
                           FmhaTdmSchedSequentialTileSchedule<typename Schedule::LocalPlacement>>,
            "the iteration-0 V0 wait in Begin assumes the sequential tile schedule");
        // Prologue TDM transfers issued after Q (K0, V0, bias, K1) and after K0 (V0, bias, K1).
        constexpr index_t kTdmIssuedAfterQ  = 3 + kBiasTdmOps;
        constexpr index_t kTdmIssuedAfterK0 = 2 + kBiasTdmOps;
        // Begin waits for its V with that tile's bias and the next K still in flight.
        constexpr index_t kTdmIssuedAfterV0 = 1 + kBiasTdmOps;
        // A bias tile is read with the next K and V transfers still in flight.
        [[maybe_unused]] constexpr index_t kTdmIssuedAfterBias = 2;

        [[maybe_unused]] array<int32_t, Geometry::kQkMIter * QuantScale::kMxQkScaleDwords> q_scales;
        if constexpr(kMx)
        {
            static_assert(Geometry::QkWmma::kM == Geometry::QkWmma::kN &&
                          Geometry::PvWmma::kN == Geometry::QkWmma::kN);
            static_assert(TileIndex::IsWaveInterleavedBands<
                              typename SaccBlockTileType::StaticTileDistribution,
                              Geometry::kQkMIter,
                              Geometry::QkWmma::kM,
                              Geometry::kWaves>(),
                          "MX Q scale rows assume wave-interleaved score row bands");
            static_for<0, Geometry::kQkMIter, 1>{}([&](auto m_iter) {
                const index_t row =
                    q_origin.at(I0) +
                    (m_iter * Geometry::kWaves + get_warp_id()) * Geometry::QkWmma::kM + mx_lane;
                static_for<0, QuantScale::kMxQkScaleDwords, 1>{}([&](auto j) {
                    q_scales(number<j * Geometry::kQkMIter + m_iter>{}) =
                        QuantScale::LoadMxRowScale(mx.q_ptr + j * QuantScale::kMxStepScaleBytes,
                                                   min(row, mx.seqlen_q - 1),
                                                   mx.stride_q);
                });
            });
        }
        // MX scales of tile j + 1 land in LDS during iteration j; K is read one iteration early.
        [[maybe_unused]] array<int32_t,
                               QuantScale::kMxKScalesPerPlane * QuantScale::kMxQkScaleDwords>
            k_scales;
        [[maybe_unused]] array<int32_t, Geometry::kPvNIter> v_scales;
        [[maybe_unused]] MxCopyOffsets mx_copy;
        if constexpr(kMx)
        {
            mx_copy = QuantScale::MakeMxCopyOffsets(mx, kv_load_start, kv_last);
            QuantScale::IssueMxScalesToLds(mx, mx_copy, MxScaleLds(0), kv_load_start, kv_last);
            if constexpr(QuantScale::kMxPairedK64)
                QuantScale::PairMxQkScales(q_scales);
            Pins::PinEachToVgpr(q_scales);
        }
        // MX scale loads above do not touch Q's LDS, so they overlap the Q transfer.
        s_wait_tensorcnt_barrier<kTdmIssuedAfterQ>();
        auto q_tile = load_tile(q_lds_read_window);

        [[maybe_unused]] int32x4_t k_descale_res;
        [[maybe_unused]] int32x4_t v_descale_res;
        // Descales are loaded one iteration ahead; kv_scale_origin is the tile they belong to.
        [[maybe_unused]] KVDescales k_descales;
        [[maybe_unused]] KVDescales v_descales;
        [[maybe_unused]] index_t kv_scale_origin = kv_load_start;
        // Without sink tokens the tiles are contiguous, so a cursor replaces the division.
        [[maybe_unused]] KVScaleCursor kv_scale_cursor;
        if constexpr(kBlockScale && !kHasSink)
            kv_scale_cursor = KVScaleCursor(kv_load_start, block_scale_size_kv);
        [[maybe_unused]] KVScaleOffsets kv_scale_offsets;
        if constexpr(kBlockScale)
        {
            k_descale_res =
                QuantScale::MakeKVDescaleResource(k_descale_ptr, kv_last, block_scale_size_kv);
            v_descale_res =
                QuantScale::MakeKVDescaleResource(v_descale_ptr, kv_last, block_scale_size_kv);
            if constexpr(kHasSink)
            {
                QuantScale::LoadKVDescalesAt(
                    k_descale_res, kv_scale_origin, block_scale_size_kv, k_descales);
                QuantScale::LoadKVDescalesAt(
                    v_descale_res, kv_scale_origin, block_scale_size_kv, v_descales);
            }
            else
            {
                QuantScale::AdvanceKVScaleCursor(kv_scale_cursor, kv_scale_offsets);
                QuantScale::LoadKVDescalesFromOffsets(
                    k_descale_res, v_descale_res, kv_scale_offsets, k_descales, v_descales);
            }
        }

        // dscnt 0 + barrier retires every wavefront's Q reads before the iteration-0 V1 write.
        __builtin_amdgcn_sched_barrier(0);
        // The MX scale copies issue no asyncmark, so wait on ASYNCcnt itself.
        if constexpr(kMx)
            __builtin_amdgcn_s_wait_asynccnt(0);
        s_wait_tensorcnt_barrier<kTdmIssuedAfterK0, 0>();
        if constexpr(kMx)
        {
            Pins::CompilerMemoryFence();
            QuantScale::ReadMxScalesFromLds(MxScaleLds(0) + mx_lane, k_scales);
            QuantScale::PadMxKScales(k_scales);
        }
        __builtin_amdgcn_sched_barrier(0);
        auto initial_k_lds_su_read_window =
            make_tile_window(k_lds_view,
                             make_tuple(number<Geometry::kQkSuColumns>{}, number<kQKHeaddim>{}),
                             {0, 0},
                             Policy::template MakeKSuRegTileDistribution<Problem>());
        if constexpr(kLdsSegSplit)
            initial_k_lds_su_read_window.set_bottom_tensor_view_data_ptr(
                k_seg(smem_ptrk0, seg_k_lo));
        else
            initial_k_lds_su_read_window.set_bottom_tensor_view_data_ptr(
                static_cast<KDataType* __restrict__>(smem_ptrk0));
        auto k_su0_tile = load_tile(initial_k_lds_su_read_window);

        __builtin_amdgcn_sched_barrier(0);

        auto mainloop = [&](KDataType* __restrict__ k_lds_cur_ptr,
                            KDataType* __restrict__ k_lds_next_ptr,
                            VDataType* __restrict__ v_lds_next_ptr,
                            VDataType* __restrict__ v_lds_cur_ptr,
                            auto mx_all_fast,
                            auto sink_escape) {
            decltype(load_tile_transpose(v_lds_read_window)) v_tile;
            [[maybe_unused]] array<float, 2> qk_lane_max;
            [[maybe_unused]] float pv_output_scale_m0 = 1.0f;
            [[maybe_unused]] float pv_output_scale_m1 = 1.0f;
            [[maybe_unused]] int32_t v_scale          = QuantScale::kE8m0One;
            [[maybe_unused]] index_t kv_tile_start    = 0;
            [[maybe_unused]] const index_t mx_loop =
                kUseCountdownLoop ? num_total_loop - remaining_loops : i_total_loops;
            [[maybe_unused]] array<int32_t, k1_loops> v_scales_bs;
            // LDS byte addresses of this lane's next-tile K scales and this tile's V scales.
            [[maybe_unused]] int32_t mx_k_read = 0;
            [[maybe_unused]] int32_t mx_v_read = 0;
            v_lds_read_window.set_bottom_tensor_view_data_ptr(v_lds_cur_ptr);
            // LDS read bases of this iteration: K groups 1.., the V tile, then next K group 0.
            constexpr index_t kKBases    = Geometry::kQkSuNIter;
            constexpr index_t kVBase     = (Geometry::kQkStages - 1) * kKBases;
            constexpr index_t kVBases    = kLdsSegSplit ? 2 : 1;
            constexpr index_t kVSplit    = kLdsSegSplit ? kVHalfBytes : 0;
            constexpr index_t kNextKBase = kVBase + kVBases;
            array<int32_t, kNextKBase + 1> lds_bases;
            auto pinned = [&](const auto& window, auto first, auto num_bases) {
                return FmhaTdmSchedFp8PinnedWindow<remove_cvref_t<decltype(window)>,
                                                   decltype(num_bases)::value>{
                    window, &lds_bases[decltype(first)::value]};
            };
            // V reads: one base, or one per K half in the segmented layout.
            auto pinned_v = [&] {
                return FmhaTdmSchedFp8PinnedWindow<remove_cvref_t<decltype(v_lds_read_window)>,
                                                   kVBases,
                                                   kVSplit>{v_lds_read_window, &lds_bases[kVBase]};
            };

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

                if constexpr(kBlockScale)
                    v_scale = v_scales_bs[stage];
                static_assert(!kMx || Geometry::kPvStages == 1, "MX V scales cover one PV stage");

                auto emit_wmma = [&](auto, auto wmma) {
                    constexpr index_t ordinal     = PvOrdinal(decltype(wmma)::value);
                    constexpr index_t d_msb       = ordinal / Policy::OutputFragments::kNumN;
                    constexpr index_t n           = ordinal % Policy::OutputFragments::kNumN;
                    constexpr index_t m_iter      = d_msb / 2;
                    constexpr index_t v_msb       = d_msb % 2;
                    constexpr index_t full_n_iter = n * 2 + v_msb;

                    AWarpTensor a_warp_tensor;
                    a_warp_tensor.get_thread_buffer() = a_block_tensor.get_y_sliced_thread_data(
                        merge_sequences(sequence<0, m_iter>{}, a_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

                    BWarpTensor b_warp_tensor;
                    b_warp_tensor.get_thread_buffer() = b_block_tensor.get_y_sliced_thread_data(
                        merge_sequences(sequence<0, full_n_iter>{}, b_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));

                    CWarpTensor c_warp_tensor;
                    c_warp_tensor.get_thread_buffer().template set_as<fp32x8_t>(
                        number<0>{}, o_acc.at(number<ordinal>{}));
                    // LDS-read MX scales are already VGPRs; a pin would add a wait per scale.
                    if constexpr(kMx)
                        WarpGemm{}(c_warp_tensor,
                                   a_warp_tensor,
                                   b_warp_tensor,
                                   QuantScale::kE8m0One,
                                   v_scales[full_n_iter]);
                    else if constexpr(kBlockScale)
                        WarpGemm{}(c_warp_tensor,
                                   a_warp_tensor,
                                   b_warp_tensor,
                                   QuantScale::kE8m0One,
                                   v_scale);
                    else
                        WarpGemm{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);
                    o_acc.at(number<ordinal>{}) =
                        c_warp_tensor.get_thread_buffer().template get_as<fp32x8_t>(number<0>{});
                };

                // Dword next-K stores drop QK byte permutes; MX and blockscale keep bytes for size.
                if constexpr(kD256)
                    Policy::template RunPvStageWithWmma<decltype(stage)::value,
                                                        Mode,
                                                        !kMx && !kBlockScale>(
                        next_b_block_tensor, b_lds_window, v_tile, pinned_v(), emit_wmma);
                else
                {
                    typename Policy::NoStreamedV no_v;
                    Policy::template RunPvStageWithWmma<decltype(stage)::value,
                                                        Mode,
                                                        !kMx && !kBlockScale>(
                        next_b_block_tensor, b_lds_window, no_v, no_v, emit_wmma);
                }
            };

            // Wave priority by phase: PV > QK > softmax exps > softmax max.
            constexpr int16_t kQkPrio = 2, kSoftmaxMaxPrio = 0, kSoftmaxExpPrio = 1, kPvPrio = 3;
            auto run_pv = [&](const auto& p_tile) {
                __builtin_amdgcn_s_setprio(kPvPrio);
                if constexpr(kBlockScale)
                {
                    static_for<0, k1_loops, 1>{}([&](auto i_k1) {
                        v_scales_bs(i_k1) = QuantScale::PackVScale(v_descales, i_k1);
                    });
                    // Both descale sets are consumed; fetch the next tile's.
                    if constexpr(kHasSink)
                    {
                        const index_t next = i_total_loops + 1;
                        kv_scale_origin    = next < num_sink_loop ? kN0 * next + kv_load_start
                                                                  : kN0 * (next - num_sink_loop) +
                                                                     logical_seqlen_k_start;
                        QuantScale::LoadKVDescalesAt(
                            k_descale_res, kv_scale_origin, block_scale_size_kv, k_descales);
                        QuantScale::LoadKVDescalesAt(
                            v_descale_res, kv_scale_origin, block_scale_size_kv, v_descales);
                    }
                    else
                        QuantScale::LoadKVDescalesFromOffsets(
                            k_descale_res, v_descale_res, kv_scale_offsets, k_descales, v_descales);
                }
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
                auto k_lds_su0_read_window = make_k_su_read_window(k_lds_next_ptr);
                run_fragment_pv_stage(
                    number<Geometry::kPvStages - 1>{},
                    p_tile_last,
                    v_tile,
                    k_su0_tile,
                    pinned(k_lds_su0_read_window, number<kNextKBase>{}, number<1>{}));
                if constexpr(kNormalizeQuantizedP)
                    QuantScale::template AccumulateQuantizedMass<decltype(gemm_1)>(
                        p_tile,
                        kQuantizedRowSum ? l : l_quant,
                        pv_output_scale_m0,
                        pv_output_scale_m1);
            };

            auto begin_iteration = [&]() {
                // Fixed boundary: start the V TDM transfer once preceding reads drain.
                s_wait_tensorcnt<kTdmIssuedAfterV0>();
                block_sync_lds<k_lds_insts>();
                if constexpr(kMx)
                {
                    // DS reads retire in order: the wait above retired every read of this buffer.
                    Pins::CompilerMemoryFence();
                    // Without sink tokens the KV tiles are contiguous from kv_load_start.
                    kv_tile_start           = kv_load_start + mx_loop * kN0;
                    index_t next_tile_start = kv_tile_start + kN0;
                    if constexpr(kHasSink)
                    {
                        // Tiles past the sink prefix jump to the window start.
                        const index_t window_base = logical_seqlen_k_start - sink_seq_end;
                        if(mx_loop >= num_sink_loop)
                            kv_tile_start = window_base + mx_loop * kN0;
                        if(mx_loop + 1 >= num_sink_loop)
                            next_tile_start = window_base + (mx_loop + 1) * kN0;
                    }
                    const bool has_next = kUseCountdownLoop ? remaining_loops > 1
                                                            : i_total_loops + 1 < num_total_loop;
                    // The last tile reloads its own scales so the copy stays in bounds.
                    QuantScale::template IssueMxScalesToLds<decltype(mx_all_fast)::value>(
                        mx,
                        mx_copy,
                        MxScaleLds(mx_loop + 1),
                        has_next ? next_tile_start : kv_tile_start,
                        kv_last);
                    // Address the post-softmax scale reads now so they need no fresh VALU there.
                    mx_k_read = Pins::PinToVgpr(
                        QuantScale::LdsByteAddress(MxScaleLds(mx_loop + 1) + mx_lane));
                    mx_v_read = Pins::PinToVgpr(QuantScale::LdsByteAddress(
                        MxScaleLds(mx_loop) + QuantScale::kMxKDwords + mx_lane));
                }
                // V is prefetched one iteration before its consumer.
                if constexpr(kHasSink)
                {
                    if(i_total_loops == num_sink_loop - 1 && i_total_loops + 1 < num_total_loop)
                        move_tile_window(v_dram_window, {logical_seqlen_k_start - sink_seq_end, 0});
                }
                move_tile_window(v_dram_window, {kN0, 0});
                if constexpr(kLdsSegSplit)
                    v_lds_write_window.set_bottom_tensor_view_data_ptr(
                        v_seg(v_lds_next_ptr, seg_write));
                else
                    v_lds_write_window.set_bottom_tensor_view_data_ptr(v_lds_next_ptr);
                LoadTileTdmInLoop(tdm_config_v, v_lds_write_window, v_dram_window);
            };

            auto run_qk = [&] {
                __builtin_amdgcn_s_setprio(kQkPrio);
                clear_tile(s_acc);
                constexpr auto gemm_0_su  = Policy::template GetQKBlockGemmSu<Problem>();
                auto k_lds_su_read_window = make_k_su_read_window(k_lds_cur_ptr);
                // One address wait here, after the barrier, instead of one per K group.
                static_for<0, Geometry::kQkStages - 1, 1>{}([&](auto i_su) {
                    auto window = k_lds_su_read_window;
                    move_tile_window(window, {(i_su + 1) * Geometry::kQkSuColumns, 0});
                    FmhaTdmSchedFp8PinnedWindow<decltype(window), kKBases>::ComputeBases(
                        window, &lds_bases(number<i_su * kKBases>{}));
                    if constexpr(kLdsSegSplit)
                        static_for<0, kKBases, 1>{}([&](auto j) {
                            lds_bases(number<i_su * kKBases + j>{}) +=
                                decltype(i_su)::value + 1 < Geometry::kQkStages / 2 ? seg_k_lo
                                                                                    : seg_k_hi;
                        });
                });
                FmhaTdmSchedFp8PinnedWindow<decltype(v_lds_read_window), kVBases, kVSplit>::
                    ComputeBases(v_lds_read_window, &lds_bases(number<kVBase>{}));
                auto k_su0_window = k_lds_su_read_window;
                k_su0_window.set_bottom_tensor_view_data_ptr(k_lds_next_ptr);
                FmhaTdmSchedFp8PinnedWindow<decltype(k_su0_window), 1>::ComputeBases(
                    k_su0_window, &lds_bases(number<kNextKBase>{}));
                if constexpr(kLdsSegSplit)
                {
                    lds_bases(number<kVBase>{}) += seg_v_lo;
                    lds_bases(number<kVBase + 1>{}) += seg_v_hi;
                    lds_bases(number<kNextKBase>{}) += seg_k_lo;
                }
                Pins::PinEachToVgpr(lds_bases);
                auto s_acc_su  = gemm_0_su.MakeCBlockTile();
                auto k_su_tile = k_su0_tile;
                decltype(k_su_tile) k_su_tile_next;
                constexpr bool kQkLead = Features::kQkLead;
                // Group 1's first half goes out before the first WMMA, a full stage ahead of use.
                if constexpr(kQkLead)
                {
                    static_for<0, Geometry::kKSuLoadCount / 2, 1>{}([&](auto a) {
                        Policy::KLoad::template LoadInstruction<decltype(a)::value>(
                            k_su_tile_next,
                            pinned(k_lds_su_read_window, number<0>{}, number<kKBases>{}));
                    });
                    __builtin_amdgcn_sched_barrier(0);
                }
                // WMMA scale operands of QK stage `stage`, used by MX only.
                auto qk_a_scale = [&](auto) {
                    if constexpr(kMx)
                        return [&](auto m_iter, auto k_iter) {
                            return q_scales[number<k_iter * Geometry::kQkMIter + m_iter>{}];
                        };
                    else
                        return typename Policy::NoScale{};
                };
                auto qk_b_scale = [&](auto stage) {
                    if constexpr(kMx)
                        return [&](auto n_iter, auto k_iter) {
                            return k_scales[number<k_iter * QuantScale::kMxKScalesPerPlane +
                                                   decltype(stage)::value * Geometry::kQkSuNIter +
                                                   decltype(n_iter)::value>{}];
                        };
                    else
                        return typename Policy::NoScale{};
                };
                auto run_qk_lead_stage = [&](auto stage,
                                             auto& next_tile,
                                             const auto& window,
                                             auto& lead_tile,
                                             const auto& lead_window) {
                    Policy::template RunQkScheduledLeadStage<decltype(stage)::value, Mode>(
                        gemm_0_su,
                        s_acc_su,
                        q_tile,
                        k_su_tile,
                        next_tile,
                        window,
                        lead_tile,
                        lead_window,
                        qk_a_scale(stage),
                        qk_b_scale(stage));
                };
                auto run_qk_stage = [&](auto stage, auto& next_tile, const auto& window) {
                    Policy::template RunQkScheduledStage<decltype(stage)::value, Mode>(
                        gemm_0_su,
                        s_acc_su,
                        q_tile,
                        k_su_tile,
                        next_tile,
                        window,
                        qk_a_scale(stage),
                        qk_b_scale(stage));
                };
                static_for<0, Geometry::kQkStages, 1>{}([&](auto i_su) {
                    // Re-issued mid-QK so the extra setprio also splits the QK schedule.
                    if constexpr(decltype(i_su)::value == Geometry::kQkStages / 2)
                        __builtin_amdgcn_s_setprio(kQkPrio);
                    clear_tile(s_acc_su);
                    if constexpr(!kQkLead && decltype(i_su)::value < Geometry::kQkStages - 1)
                    {
                        run_qk_stage(i_su,
                                     k_su_tile_next,
                                     pinned(k_lds_su_read_window,
                                            number<i_su * kKBases>{},
                                            number<kKBases>{}));
                        k_su_tile = k_su_tile_next;
                    }
                    else if constexpr(decltype(i_su)::value < Geometry::kQkStages - 2)
                    {
                        decltype(k_su_tile) k_su_tile_lead;
                        run_qk_lead_stage(i_su,
                                          k_su_tile_next,
                                          pinned(k_lds_su_read_window,
                                                 number<i_su * kKBases>{},
                                                 number<kKBases>{}),
                                          k_su_tile_lead,
                                          pinned(k_lds_su_read_window,
                                                 number<(i_su + 1) * kKBases>{},
                                                 number<kKBases>{}));
                        k_su_tile      = k_su_tile_next;
                        k_su_tile_next = k_su_tile_lead;
                    }
                    else if constexpr(decltype(i_su)::value == Geometry::kQkStages - 2)
                    {
                        typename Policy::NoLeadGroup no_lead;
                        run_qk_lead_stage(i_su,
                                          k_su_tile_next,
                                          pinned(k_lds_su_read_window,
                                                 number<i_su * kKBases>{},
                                                 number<kKBases>{}),
                                          no_lead,
                                          no_lead);
                        k_su_tile = k_su_tile_next;
                    }
                    else
                    {
                        run_qk_stage(i_su, v_tile, pinned_v());
                    }
                    if constexpr(Features::kKDescale == FmhaTdmSchedFp8KDescale::WithScaleS)
                        tile_elementwise_inout(
                            [kd = k_descales[i_su] * scale_s](auto& x) { x *= kd; }, s_acc_su);
                    else if constexpr(Features::kKDescale == FmhaTdmSchedFp8KDescale::PerStage)
                        tile_elementwise_inout([kd = k_descales[i_su]](auto& x) { x *= kd; },
                                               s_acc_su);
                    set_slice_tile(
                        s_acc,
                        s_acc_su,
                        sequence<0, decltype(i_su)::value * Geometry::kQkSuColumns>{},
                        sequence<kM0, (decltype(i_su)::value + 1) * Geometry::kQkSuColumns>{});
                    if constexpr(kProgressiveMax && decltype(i_su)::value < Geometry::kQkStages - 1)
                        Softmax::template StageLaneMax<decltype(i_su)::value,
                                                       (decltype(i_su)::value > 0)>(s_acc,
                                                                                    qk_lane_max);
                });
            };

            auto process_score = [&]() {
                // QK scores are ready for bias, masking, and softmax.
                if constexpr(kBiasViaTdm)
                {
                    if constexpr(Features::kKDescale != FmhaTdmSchedFp8KDescale::WithScaleS)
                        tile_elementwise_inout([&scale_s](auto& x) { x = x * scale_s; }, s_acc);
                    // This wavefront's rows only, so its own tensorcnt is the whole RAW wait.
                    s_wait_tensorcnt<kTdmIssuedAfterBias>();
                    Pins::CompilerMemoryFence();
                    using BiasWindow = remove_cvref_t<decltype(bias_lds_window)>;
                    using BiasRead   = FmhaTdmSchedFp8Load<BiasWindow::NumAccessPerCoord, 16>;
                    constexpr index_t kChunk  = 2;
                    constexpr index_t kChunks = BiasWindow::NumAccessPerCoord / kChunk;
                    const FmhaTdmSchedFp8PinnedWindow<BiasWindow, 1> pinned_bias{bias_lds_window,
                                                                                 &bias_lds_base};
                    SaccBlockTileType bias_tile;
                    auto load_chunk = [&](auto c) {
                        static_for<0, kChunk, 1>{}([&](auto i) {
                            BiasRead::template LoadInstruction<c * kChunk + i>(bias_tile,
                                                                               pinned_bias);
                        });
                    };
                    // Two chunks in flight bound the bias VGPRs; each add retires one chunk.
                    load_chunk(number<0>{});
                    static_for<0, kChunks, 1>{}([&](auto c) {
                        __builtin_amdgcn_sched_barrier(0);
                        if constexpr(c + 1 < kChunks)
                            load_chunk(number<c + 1>{});
                        static_for<0, kChunk, 1>{}([&](auto i) {
                            static_for<0, kBiasElemsPerRead, 1>{}([&](auto j) {
                                constexpr index_t e =
                                    BiasRead::template DestinationOffset<typename BiasWindow::Base,
                                                                         c * kChunk + i>(j);
                                s_acc.get_thread_buffer()(number<e>{}) +=
                                    log2e_v<SaccDataType> *
                                    bias_tile.get_thread_buffer()[number<e>{}];
                            });
                        });
                    });
                    __builtin_amdgcn_sched_barrier(0);
                }

                constexpr bool kDeferTensorReady = Mode::kDeferTensorReady;
                if constexpr(!kDeferTensorReady)
                    s_wait_tensorcnt_barrier<Policy::kVPrefetchTensorCount>();

                const auto is_masked_out = [&](index_t row, index_t col) {
                    if constexpr(kHasSink)
                        return mask.IsOutOfSinkBound(row, col);
                    else if constexpr(std::is_same_v<FmhaMask,
                                                     SimplifiedGenericAttentionMask<true>>)
                    {
                        if constexpr(kUnpinnedMaskWidth)
                        {
                            const index_t x_start = -mask.GetWindowY() + row + 1;
                            const index_t x_end   = min(row + mask.GetWindowX(), mask.GetXTotal());
                            const uint32_t width  = row < mask.GetYTotal() && x_start < x_end
                                                        ? uint32_t(x_end) - uint32_t(x_start)
                                                        : 0u;
                            return uint32_t(col) - uint32_t(x_start) >= width;
                        }
                        else
                            return mask.IsOutOfBoundUnsigned(row, col);
                    }
                    else
                        return mask.IsOutOfBound(row, col);
                };
                constexpr bool kFusedEdgePredicate = Features::kFusedEdgePredicate;

                // Sink-aware k_origin
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
                            kN0 * (i_total_loops - num_sink_loop) + logical_seqlen_k_start, 0);
                }();
                if constexpr(kUseCountdownLoop)
                {
                    const bool needs_tail_predicate = k_origin.at(I0) + kN0 > logical_seqlen_k_end;
                    const bool need_split_check     = remaining_loops == 1 && needs_tail_predicate;
                    bool need_mask_check            = false;
                    if constexpr(kPadSeqLenK)
                    {
                        need_mask_check = mask.IsEdgeTile(
                            q_origin.at(I0), k_origin.at(I0), number<kM0>{}, number<kN0>{});
                    }

                    if(__builtin_expect(need_split_check || need_mask_check, false))
                    {
                        const auto is_edge_out = [&](index_t row, index_t col) {
                            return (need_split_check && logical_seqlen_k_end <= col) ||
                                   (need_mask_check && mask.IsOutOfBound(row, col));
                        };
                        if constexpr(kMergeHeads)
                            TileIndex::SetTileIfRowPos(s_acc,
                                                       -numeric<SMPLComputeDataType>::infinity(),
                                                       MergedRows::MergedRowPositions<decltype(m)>(
                                                           merged, q_origin.at(I0)),
                                                       k_origin.at(I0),
                                                       is_edge_out);
                        else if constexpr(!kLdsSegSplit)
                            TileIndex::SetTileIfLaneBase(s_acc,
                                                         -numeric<SMPLComputeDataType>::infinity(),
                                                         [&](auto tile_idx) {
                                                             return is_edge_out(
                                                                 q_origin.at(I0) + tile_idx.at(I0),
                                                                 k_origin.at(I0) + tile_idx.at(I1));
                                                         });
                        else
                            TileIndex::SetTileIfLaneBase(
                                s_acc,
                                -numeric<SMPLComputeDataType>::infinity(),
                                [&](auto tile_idx, auto const_idx) {
                                    // The constant column part alone picks the K half.
                                    return is_edge_out(
                                        q_origin.at(I0) + tile_idx.at(I0),
                                        k_origin.at(I0) + tile_idx.at(I1) +
                                            (const_idx.at(I1) < kN0 / 2 ? seg_col : -seg_col));
                                });
                        if constexpr(kProgressiveMax)
                            static_for<0, Geometry::kQkStages - 1, 1>{}([&](auto su) {
                                Softmax::template StageLaneMax<decltype(su)::value,
                                                               (decltype(su)::value > 0)>(
                                    s_acc, qk_lane_max);
                            });
                    }
                }
                else if constexpr(kFusedEdgePredicate)
                {
                    const bool needs_tail_predicate = k_origin.at(I0) + kN0 > logical_seqlen_k_end;
                    const bool need_split_check =
                        i_total_loops == (num_total_loop - 1) && needs_tail_predicate;
                    bool need_mask_check = false;
                    if constexpr(kPadSeqLenK || FmhaMask::IsMasking)
                    {
                        need_mask_check = mask.IsEdgeTile(
                            q_origin.at(I0), k_origin.at(I0), number<kM0>{}, number<kN0>{});
                    }

                    if(__builtin_expect(need_split_check || need_mask_check, false))
                    {
                        TileIndex::SetTileIfLaneBase(
                            s_acc, -numeric<SMPLComputeDataType>::infinity(), [&](auto tile_idx) {
                                const auto row = q_origin.at(I0) + tile_idx.at(I0);
                                const auto col = k_origin.at(I0) + tile_idx.at(I1);
                                return (need_split_check && logical_seqlen_k_end <= col) ||
                                       (need_mask_check && is_masked_out(row, col));
                            });
                    }
                }
                else if constexpr(kBandMask)
                {
                    const auto ninf                     = -numeric<SMPLComputeDataType>::infinity();
                    const index_t q0                    = q_origin.at(I0);
                    const index_t k0                    = k_origin.at(I0);
                    const index_t lr                    = s_lane_origin.at(I0);
                    const index_t lc                    = s_lane_origin.at(I1);
                    [[maybe_unused]] const index_t diag = k0 - q0;
                    // Sink keys of in-range rows escape every bound but the right one.
                    [[maybe_unused]] const index_t ts =
                        (mask.GetWindowY() < mask.GetYTotal() ? mask.GetSink() : 0) - 1 - k0 - lc;
                    [[maybe_unused]] const index_t tr = mask.GetXTotal() - 1 - q0 - lr;
                    auto mask_if                      = [&](auto out) {
                        if constexpr(decltype(sink_escape)::value)
                            TileIndex::SetTileIfConstPart(s_acc, ninf, [&](index_t r, index_t c) {
                                return out(r, c) && !(c <= ts && r <= tr);
                            });
                        else
                            TileIndex::SetTileIfConstPart(s_acc, ninf, out);
                    };
                    // Also covers the tile-aligned split end, which only adds out-of-window keys.
                    if(k0 + kN0 > mask.GetXTotal())
                    {
                        const index_t t = mask.GetXTotal() - 1 - k0 - lc;
                        mask_if([&](index_t, index_t c) { return c > t; });
                    }
                    if constexpr(FmhaMask::IsMasking)
                    {
                        if(diag + (kN0 - 1) >= mask.GetWindowX())
                        {
                            const index_t t = mask.GetWindowX() - 1 - diag - (lc - lr);
                            TileIndex::SetTileIfConstPart(
                                s_acc, ninf, [&](index_t r, index_t c) { return c - r > t; });
                        }
                        if(diag - (kM0 - 1) <= -mask.GetWindowY())
                        {
                            const index_t t = 1 - mask.GetWindowY() - diag - (lc - lr);
                            mask_if([&](index_t r, index_t c) { return c - r < t; });
                        }
                        if(q0 + kM0 > mask.GetYTotal())
                        {
                            const index_t t = mask.GetYTotal() - 1 - q0 - lr;
                            mask_if([&](index_t r, index_t) { return r > t; });
                        }
                    }
                }
                else
                {
                    const bool needs_tail_predicate = k_origin.at(I0) + kN0 > logical_seqlen_k_end;
                    const bool is_last_loop         = i_total_loops == (num_total_loop - 1);
                    if(is_last_loop && needs_tail_predicate)
                    {
                        set_tile_if(
                            s_acc,
                            -numeric<SMPLComputeDataType>::infinity(),
                            [&, logical_seqlen_k_end_ = logical_seqlen_k_end](auto tile_idx) {
                                const auto col = k_origin.at(I0) + tile_idx.at(I1);
                                return logical_seqlen_k_end_ <= col;
                            });
                    }

                    if constexpr(kPadSeqLenK || FmhaMask::IsMasking)
                    {
                        bool need_perpixel_check;
                        if constexpr(kMergeHeads &&
                                     std::is_same_v<FmhaMask, SimplifiedGenericAttentionMask<true>>)
                        {
                            // Merged rows are query positions below seqlen_q, so no row is padding.
                            const index_t k0 = k_origin.at(I0);
                            need_perpixel_check =
                                k0 + kN0 > min(mask.GetWindowX(), mask.GetXTotal()) ||
                                k0 + mask.GetWindowY() < mask.GetYTotal();
                        }
                        else
                            need_perpixel_check = mask.IsEdgeTile(
                                q_origin.at(I0), k_origin.at(I0), number<kM0>{}, number<kN0>{});
                        if(need_perpixel_check)
                        {
                            if constexpr(kMergeHeads)
                                TileIndex::SetTileIfRowPos(
                                    s_acc,
                                    -numeric<SMPLComputeDataType>::infinity(),
                                    MergedRows::MergedRowPositions<decltype(m)>(merged,
                                                                                q_origin.at(I0)),
                                    k_origin.at(I0),
                                    is_masked_out);
                            else
                                set_tile_if(s_acc,
                                            -numeric<SMPLComputeDataType>::infinity(),
                                            [&](auto tile_idx) {
                                                const auto row = q_origin.at(I0) + tile_idx.at(I0);
                                                const auto col = k_origin.at(I0) + tile_idx.at(I1);
                                                return is_masked_out(row, col);
                                            });
                        }
                    }
                }

                if constexpr(kBiasViaTdm)
                {
                    if constexpr(kHasSink)
                    {
                        if(i_total_loops == num_sink_loop - 1)
                            bias_col += logical_seqlen_k_start - sink_seq_end;
                    }
                    bias_col += kN0;
                    // The adds consumed this tile; its reads retire before the TDM overwrites it.
                    Pins::CompilerMemoryFence();
                    s_wait_dscnt<0>();
                    issue_bias();
                }
                else
                {
                    if constexpr(kHasSink)
                    {
                        if(i_total_loops == num_sink_loop - 1)
                        {
                            move_tile_window(bias_dram_window,
                                             {0, logical_seqlen_k_start - sink_seq_end});
                        }
                    }
                    move_tile_window(bias_dram_window, {0, kN0});
                }
            };

            auto run_softmax = [&] {
                __builtin_amdgcn_s_setprio(kSoftmaxMaxPrio);
                if constexpr(kBlockScale && !kHasSink)
                    QuantScale::AdvanceKVScaleCursor(kv_scale_cursor, kv_scale_offsets);
                auto s_new = cast_tile<SMPLComputeDataType>(s_acc);

                auto p_tile = [&]() {
                    static_assert(kNWarp == 1);
                    static_assert(std::is_same_v<SaccDataType, float> &&
                                  std::is_same_v<SMPLComputeDataType, float>);
                    float delta_m0;
                    float delta_m1;
                    // The bias path already scaled S into the log2 domain.
                    const auto softmax_scale = [&]() {
                        if constexpr(kRowScale)
                        {
                            const RowScalePair scales = LoadRowScales();
                            return array<float, 2>{scales[0], scales[1]};
                        }
                        else if constexpr(BiasEnum == BlockAttentionBiasEnum::NO_BIAS)
                            return scale_s;
                        else
                            return 1.0f;
                    }();
                    const auto col_scale = [&]() {
                        if constexpr(Features::kKDescale == FmhaTdmSchedFp8KDescale::InSoftmax)
                        {
                            // Floor: out-of-range blocks read 0, and -inf * 0 is NaN.
                            KVDescales scales;
                            static_for<0, QuantScale::kKVScaleGroups, 1>{}(
                                [&](auto g) { scales(g) = max(k_descales[g], 0x1p-60f); });
                            return scales;
                        }
                        else
                            return typename Softmax::NoColumnScale{};
                    }();
                    Schedule::template RunSplitSoftmax<Mode>(
                        [&]() {
                            if constexpr(kProgressiveMax)
                                Softmax::template RunSplitSoftmaxPart01FromPartial<Problem>(
                                    s_new, m, softmax_scale, delta_m0, delta_m1, qk_lane_max);
                            else
                                Softmax::template RunSplitSoftmaxPart01<Problem>(
                                    s_new, m, softmax_scale, delta_m0, delta_m1, col_scale);
                        },
                        [&]() {
                            s_wait_tensorcnt_barrier<Policy::kVPrefetchTensorCount + kBiasTdmOps>();
                        },
                        // Part 2 below rescales the output inside its exp stream.
                        [](auto) {},
                        [&](auto) {
                            __builtin_amdgcn_s_setprio(kSoftmaxExpPrio);
                            Softmax::template RunSplitSoftmaxPart2AndGetScale<Problem,
                                                                              !kQuantizedRowSum>(
                                s_new,
                                m,
                                l,
                                softmax_scale,
                                delta_m0,
                                delta_m1,
                                pv_output_scale_m0,
                                pv_output_scale_m1,
                                o_acc,
                                col_scale);
                        });
                    if constexpr(kStochasticP)
                    {
                        const PSeeds tile_seeds = p_seeds;
                        static_for<0, Features::kScalarPSeed ? 1 : 2, 1>{}(
                            [&](auto k) { p_seeds(k) += QuantScale::kPSeedStep; });
                        return MakePForGemm1<decltype(gemm_1)>(s_new, tile_seeds);
                    }
                    else
                        return MakePForGemm1<decltype(gemm_1)>(s_new, PSeeds{});
                }();
                return p_tile;
            };

            auto load_next_k = [&]() {
                // Fixed boundary: the next K TDM transfer follows softmax.
                if constexpr(kMx)
                    __builtin_amdgcn_s_wait_asynccnt(0); // no asyncmark, see the prologue
                block_sync_lds<v_lds_insts>();
                if constexpr(kMx)
                {
                    Pins::CompilerMemoryFence();
                    QuantScale::ReadMxScalesFromLds(QuantScale::MxScaleLanePtr(mx_k_read),
                                                    k_scales);
                    QuantScale::PadMxKScales(k_scales);
                    QuantScale::ReadMxScalesFromLds(QuantScale::MxScaleLanePtr(mx_v_read),
                                                    v_scales);
                }
                // The K transfer here feeds iteration i+2, beyond the already prefetched K1.
                if constexpr(kHasSink)
                {
                    if(i_total_loops == num_sink_loop - 2 && i_total_loops + 2 < num_total_loop)
                        move_tile_window(k_dram_window, {logical_seqlen_k_start - sink_seq_end, 0});
                }
                move_tile_window(k_dram_window, {kN0, 0});
                if constexpr(kLdsSegSplit)
                    k_lds_write_window.set_bottom_tensor_view_data_ptr(
                        k_seg(k_lds_cur_ptr, seg_write));
                else
                    k_lds_write_window.set_bottom_tensor_view_data_ptr(k_lds_cur_ptr);
                LoadTileTdmInLoop(tdm_config_k, k_lds_write_window, k_dram_window);
            };

            auto advance_iteration = [&]() {
                // Fixed boundary: wait for the next K tile; without K prefetch drain all, bias too.
                s_wait_tensorcnt_barrier<Policy::kKPrefetchTensorCount == 0
                                             ? 0
                                             : Policy::kKPrefetchTensorCount + kBiasTdmOps>();
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

        // Dense D256 MX runs a loop copy without the edge-tile scale path when every tile is full.
        if constexpr(kMx && kUseCountdownLoop && kD256)
        {
            auto run_loop = [&](auto mx_all_fast) {
                do
                {
                    bool is_even_loop  = kUseCountdownLoop ? even_buffer : i_total_loops % 2 == 0;
                    const auto buffers = Common::SelectKvBuffers<KDataType, VDataType>(
                        is_even_loop, smem_ptrk0, smem_ptrk1, smem_ptrv0, smem_ptrv1);
                    mainloop(buffers.k_cur,
                             buffers.k_next,
                             buffers.v_next,
                             buffers.v_cur,
                             mx_all_fast,
                             std::false_type{});
                    if constexpr(kUseCountdownLoop)
                    {
                        even_buffer = !even_buffer;
                        --remaining_loops;
                    }
                    else
                    {
                        i_total_loops++;
                    }
                } while(kUseCountdownLoop ? remaining_loops > 0 : i_total_loops < num_total_loop);
            };
            if(kv_load_start + (num_total_loop - 1) * kN0 < mx_copy.fast_end)
                run_loop(std::true_type{});
            else
                run_loop(std::false_type{});
        }
        else if constexpr(Features::kSinkEscapeLoop)
        {
            static_assert(!kUseCountdownLoop, "the sink-escape loop picks buffers by tile parity");
            // Only the leading tiles can hold sink keys, so only they run the per-lane escape.
            const index_t sink_end     = mask.GetWindowY() < mask.GetYTotal() ? mask.GetSink() : 0;
            const index_t escape_loops = min(integer_divide_ceil(sink_end, kN0), num_total_loop);
            auto run_tiles             = [&](auto sink_escape, index_t end) {
                while(i_total_loops < end)
                {
                    const auto buffers = Common::SelectKvBuffers<KDataType, VDataType>(
                        i_total_loops % 2 == 0, smem_ptrk0, smem_ptrk1, smem_ptrv0, smem_ptrv1);
                    mainloop(buffers.k_cur,
                             buffers.k_next,
                             buffers.v_next,
                             buffers.v_cur,
                             std::false_type{},
                             sink_escape);
                    i_total_loops++;
                }
            };
            run_tiles(std::true_type{}, escape_loops);
            run_tiles(std::false_type{}, num_total_loop);
        }
        else
        {
            do
            {
                bool is_even_loop  = kUseCountdownLoop ? even_buffer : i_total_loops % 2 == 0;
                const auto buffers = Common::SelectKvBuffers<KDataType, VDataType>(
                    is_even_loop, smem_ptrk0, smem_ptrk1, smem_ptrv0, smem_ptrv1);
                mainloop(buffers.k_cur,
                         buffers.k_next,
                         buffers.v_next,
                         buffers.v_cur,
                         std::false_type{},
                         bool_constant<Features::kSinkEscapeEveryTile>{});
                if constexpr(kUseCountdownLoop)
                {
                    even_buffer = !even_buffer;
                    --remaining_loops;
                }
                else
                {
                    i_total_loops++;
                }
            } while(kUseCountdownLoop ? remaining_loops > 0 : i_total_loops < num_total_loop);
        }
        // Wave priority stays at kPvPrio; only the epilogue runs from here.

        // Reconstruct the CK tensor only at the final normalization boundary.
        auto output = Policy::OutputFragments::template Reconstruct<OaccBlockTileType>(o_acc);
        constexpr auto o_spans = decltype(output)::get_distributed_spans();

        if constexpr(kQuantized)
        {
            // Undo the fixed P scale; v_descale is the scalar descale, or 1 for blockscale and MX.
            sweep_tile_span(o_spans[I0], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                const float o_scale  = [&]() {
                    if constexpr(kQuantizedRowSum)
                        return v_descale;
                    else if constexpr(kNormalizeQuantizedP)
                        return l_quant[i_idx] == 0.f ? 0.f : v_descale * l[i_idx] / l_quant[i_idx];
                    else
                        return v_descale * (1.0f / kPScale);
                }();
                sweep_tile_span(o_spans[I1],
                                [&](auto idx1) { output(make_tuple(idx0, idx1)) *= o_scale; });
            });
        }

        if constexpr(kRowScale)
        {
            const RowScalePair scales         = LoadRowScales();
            row_scale.get_thread_buffer()(I0) = scales[0];
            row_scale.get_thread_buffer()(I1) = scales[1];
        }

        auto finalize_real_tokens = [&]() {
            if constexpr(kStoreLSE)
            {
                auto lse_acc =
                    make_static_distributed_tensor<LSEDataType>(m.get_tile_distribution());

                constexpr auto lse_acc_spans = decltype(lse_acc)::get_distributed_spans();
                sweep_tile_span(lse_acc_spans[I0], [&, m_ = m, l_ = l](auto idx0) {
                    constexpr auto i_idx = make_tuple(idx0);
                    if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
                    {
                        lse_acc(i_idx) = m_[i_idx] / C_LOG2E + log(l_[i_idx]);
                    }
                    else
                    {
                        lse_acc(i_idx) = Common::RowLse(m_[i_idx], scale_of(i_idx), l_[i_idx]);
                    }
                });

                store_tile(lse_acc_dram_window_tmp, lse_acc);
            }

            sweep_tile_span(o_spans[I0], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                const auto tmp =
                    Common::RowInvSum<(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                                       FmhaMask::IsMasking)>(l[i_idx]);
                sweep_tile_span(o_spans[I1], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    output(i_j_idx) *= tmp;
                });
            });
        };

        if constexpr(kEnableVirtualSink)
        {
            if(__builtin_isinf_sign(sink_v) != -1)
            {
                auto lse_acc =
                    make_static_distributed_tensor<LSEDataType>(m.get_tile_distribution());
                const float sink_natural = sink_v * scale_s;
                const float sink_log2    = sink_natural * C_LOG2E;
                sweep_tile_span(o_spans[I0], [&](auto idx0) {
                    constexpr auto i_idx         = make_tuple(idx0);
                    const float row_sink_natural = [&] {
                        if constexpr(kMergeHeads)
                            return MergedRows::MergedSinkLogit(
                                merged, TileIndex::LaneRowOf<decltype(m)>(i_idx), scale_of(i_idx));
                        else
                            return sink_natural;
                    }();
                    const float row_sink_log2 =
                        kMergeHeads ? row_sink_natural * C_LOG2E : sink_log2;
                    float output_scale;
                    float merged_lse;
                    if(__builtin_isnan(row_sink_natural))
                    {
                        output_scale = row_sink_natural;
                        merged_lse   = row_sink_natural;
                    }
                    else if(l[i_idx] == 0.f || __builtin_isinf_sign(row_sink_natural) > 0)
                    {
                        output_scale = 0.f;
                        merged_lse   = row_sink_natural;
                    }
                    else
                    {
                        // The bias path keeps scale_s inside m.
                        const float real_log2 = BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS
                                                    ? m[i_idx]
                                                    : m[i_idx] * scale_of(i_idx);
                        const float merged_max  = max(real_log2, row_sink_log2);
                        const float real_weight = SplitSoftmax::Exp2(real_log2 - merged_max);
                        const float sink_weight = SplitSoftmax::Exp2(row_sink_log2 - merged_max);
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
        // Every wavefront's terminal bias TDM lands before any O staging write reuses the LDS.
        if constexpr(kBiasViaTdm)
            s_wait_tensorcnt_barrier<0>();

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
                                        void* smem_ptrk0,
                                        void* smem_ptrk1,
                                        void* smem_ptrv0,
                                        void* smem_ptrv1,
                                        const float* k_descale_ptr  = nullptr,
                                        const float* v_descale_ptr  = nullptr,
                                        index_t block_scale_size_kv = 1,
                                        float v_descale             = 1.0f,
                                        const MxDescales& mx        = {},
                                        const MergedHeads& merged   = {}) const
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
                   sink_v,
                   k_descale_ptr,
                   v_descale_ptr,
                   block_scale_size_kv,
                   v_descale,
                   mx,
                   merged);
    }
};

} // namespace ck_tile
