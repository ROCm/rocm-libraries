// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/block_attention_kvcache_layout_enum.hpp"
#include "ck_tile/ops/fmha/block/block_attention_quant_scale_enum.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_load.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_output.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_softmax.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_config.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_schedule.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_executor.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_prepared_k_read.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_prepared_v_read.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_affine_k_read.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_affine_v_read.hpp"

namespace ck_tile {

namespace detail {
// Vectorized KV cache problems (batch prefill VECTORIZED_LAYOUT); other problems have no layout.
template <typename Problem, typename = void>
struct fmha_tdm_is_vectorized_kv : std::false_type
{
};
template <typename Problem>
struct fmha_tdm_is_vectorized_kv<Problem, std::void_t<decltype(Problem::kKVMemoryLayout)>>
    : bool_constant<Problem::kKVMemoryLayout ==
                    BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT>
{
};
} // namespace detail

// gfx1250 WMMA operand write-after-read (WAR) hazard of the scheduled-TDM loop.
//
// Problem. A gfx1250 WMMA can read its A/B source registers after later instructions of the
// wave have issued, for example while the WMMA unit is still busy with earlier WMMAs or with
// another wave of the SIMD. The hardware does not hold back a later LDS load that overwrites
// such a register, and the compiler (with -amdgpu-expert-scheduling-mode) does not model the
// hazard either. The QK and PV stages refill their K/V operand registers from LDS while the
// WMMAs of the previous rows may still be in flight, so without a guard the refill can land
// first and a WMMA multiplies the next stage's data: rare, launch-to-launch different output
// (about one corrupted launch in a thousand on large grids before the guard).
//
// Not a fix: a counted wait. `s_wait_alu depctr_va_vdst(k)` with k >= 1 only says that all but
// the k newest VALU/WMMA writes have completed. WMMAs do not retire in issue order, so it does
// not prove that a particular older WMMA has read its operands. Only va_vdst(0), a full drain,
// does.
//
// Guard (Options::kWmmaOperandGuard; see WaitWmmaOperandsRead for the RAW fence). With the
// grouped-drain placement (FmhaTdmSchedGroupedDrainPlacement, which the page tile selects for its
// guarded WMMA-first problems), the reads that follow a last reader are issued after the next WMMA,
// and one full drain, `s_wait_alu depctr_va_vdst(0)`, covers four m0/m1 pairs or four
// last-iteration PV rows. The operand registers of the WMMAs that are still in flight before a
// drain stay allocated (KeepOperandLive, an empty asm use) until that drain, so the register
// allocator cannot place a refill onto them. Each stage's last row keeps the RAW fence. The other
// placements keep a RAW fence after every last-M-iteration WMMA (the last reader of a refilled
// operand), which drains the WMMA pipe once per last reader.
// Measured on gfx1250 (BF16, P64/P128 x D64/D128, dense and causal): no refill of a pending WMMA
// operand left in the ISA, no output difference over 4 x 5000 relaunches, and 12-19 % less kernel
// time than a RAW fence after every last reader (6-12 % on the largest grids), because four rows
// share one drain.
//
// Scope. The hazard does not depend on the KV layout: every loop of this policy refills K/V
// operand registers while earlier WMMAs may still read them. The guard is applied to the
// vectorized-KV page tile, which requires it. The linear page tile and the forward qr_tdm_sched
// kernels keep unguarded refills: in relaunch race screens of the linear page tile (P64/P128,
// D64/D128, dense BF16) no launch showed the scattered rows of this hazard, and its whole-tile
// events (see kDrainLdsBeforeTdmIssue) came at the rate of the guarded vectorized tile. After a
// change of their schedule or register allocation, re-run the relaunch race screen.
//
// Once the compiler tracks this hazard, the explicit drains and keep-alive uses can go.

template <typename Geometry_,
          typename Tuning_,
          typename TileSchedule_ = FmhaTdmSchedDefaultScheduleFor<Geometry_>>
struct BlockFmhaPipelineQRKSVSTdmSchedPolicy : BlockFmhaPipelineQRKSVSTdmDefaultPolicy
{
    using BasePolicy = BlockFmhaPipelineQRKSVSTdmDefaultPolicy;
    using Geometry   = Geometry_;
    using Tuning     = Tuning_;
    using Options    = typename FmhaTdmSchedTuningOptionsOf<Tuning>::type;
    using Schedule   = TileSchedule_;
    using Layout     = FmhaTdmSchedLayout<Geometry>;
    using KLoad      = FmhaTdmSchedLoad<Geometry::kKSuLoadCount>;
    using VLoad      = FmhaTdmSchedTransposeLoad<Geometry::kVStageLoadCount>;
    static_assert(Schedule::kNumQkStages == Geometry::kQkStages &&
                  Schedule::kNumPvStages == Geometry::kPvStages &&
                  Schedule::kQkWmmasPerStage == Geometry::kQkWmmasPerStage &&
                  Schedule::kPvWmmasPerStage == Geometry::kPvWmmasPerStage);
    static constexpr index_t kBlockPerCu = Tuning::kBlockPerCu;

    static constexpr index_t kLdsRows            = Geometry::kN;
    static constexpr index_t kKValidWidth        = Geometry::kHeadDimQK;
    static constexpr index_t kKPhysicalStride    = Layout::kKPhysicalStride;
    static constexpr index_t kVLogicalWidth      = Geometry::kDv;
    static constexpr index_t kVPhysicalStride    = Layout::kVPhysicalStride;
    static constexpr index_t kVPadAmount         = Layout::kVPadAmount;
    static constexpr index_t kVPadInterval       = Layout::kVPadInterval;
    static constexpr index_t kKFootprintElements = kLdsRows * kKPhysicalStride;
    static constexpr index_t kVFootprintElements = kLdsRows * kVPhysicalStride;
    static constexpr index_t kKFootprintBytes =
        kKFootprintElements * sizeof(typename Geometry::KDataType);
    static constexpr index_t kVFootprintBytes =
        kVFootprintElements * sizeof(typename Geometry::VDataType);
    // Named geometry knobs. Values frozen to the current D128/D192 formulas.
    static constexpr bool kEarlyOutputRescale = Geometry::kHeadDimQK == 128;
    static constexpr bool kLeadFirstLoad      = Geometry::kHeadDimQK == 128;
    static constexpr bool kUseUnsignedCausalMask =
        Geometry::kHeadDimQK == 128 || Geometry::kHeadDimQK == 192;
    static constexpr index_t kKPrefetchTensorCount = Tuning::kKWait;
    static constexpr index_t kVPrefetchTensorCount = Tuning::kVWait;
    static constexpr bool kPrefetchTailDrain       = Tuning::kTailDrain;
    static_assert(kKPrefetchTensorCount == 0 || kKPrefetchTensorCount == 1);
    static_assert(kVPrefetchTensorCount == 0 || kVPrefetchTensorCount == 1);

    // gfx1250 stale-fragment class. On some gfx1250 boards a K or V fragment that one wave writes
    // to LDS with tensor_load_to_lds is occasionally read by the other waves with the previous
    // contents of that LDS buffer, and one whole query tile of a rare launch is wrong. It occurs
    // with M64 and M128 tiles and with both KV layouts, and its rate depends on the issue
    // schedule: on large dense BF16 grids about 1-3e-5 per launch with M128 D64 tiles, none seen
    // on boards with newer firmware. The hardware cause is not known. Workaround: before every K/V
    // TDM issue the wave waits for its own LDS reads (s_wait_dscnt 0, in the pipeline and in the
    // page-tile window), and the barrier that publishes the next V tile waits for them too. Taken
    // by D64 M64 tiles, where it was most frequent (about 5e-5 per launch without it), and by the
    // problems whose options ask for it (Options::kDrainLdsBeforeTdmIssue: the linear page tile,
    // where P64 D128 softcap tiles showed it in most relaunches of a device-filling grid). With
    // P128 pages no event in 150000 relaunches followed; P64 pages keep about 3e-6 with or
    // without it.
    static constexpr bool kDrainLdsBeforeTdmIssue =
        Options::kDrainLdsBeforeTdmIssue || (Geometry::kM == 64 && Geometry::kHeadDimQK == 64);

    template <index_t Stage>
    CK_TILE_HOST_DEVICE static constexpr index_t GetQkTailDsCount()
    {
        return Tuning::template GetQkTailDsCount<Stage>();
    }

    template <index_t Stage>
    CK_TILE_HOST_DEVICE static constexpr index_t GetPvTailDsCount()
    {
        return Tuning::template GetPvTailDsCount<Stage>();
    }

    template <index_t Stage>
    CK_TILE_DEVICE static void WaitQkStageTail()
    {
        static_assert(Stage >= 0 && Stage < Geometry::kQkStages);
        s_wait_dscnt<GetQkTailDsCount<Stage>()>();
    }

    template <index_t Stage>
    CK_TILE_DEVICE static void WaitPvStageTail()
    {
        static_assert(Stage >= 0 && Stage < Geometry::kPvStages);
        s_wait_dscnt<GetPvTailDsCount<Stage>()>();
    }

    static constexpr index_t kLdsOffsetK0  = 0;
    static constexpr index_t kLdsOffsetK1  = kLdsOffsetK0 + kKFootprintBytes;
    static constexpr index_t kLdsOffsetV0  = kLdsOffsetK1 + kKFootprintBytes;
    static constexpr index_t kLdsOffsetV1  = kLdsOffsetV0 + kVFootprintBytes;
    static constexpr index_t kLdsArenaSize = kLdsOffsetV1 + kVFootprintBytes;

    using OutputFragments =
        FmhaTdmSchedOutputFragmentsFor<Geometry::kPvNIter / 2, Geometry::kPvMIter>;

    // The split-softmax score mapping and the PV output fragments are hand-written for one
    // decomposition; tie them to the derived geometry. The softmax works on 32-column score
    // groups whatever the QK stage width (64 columns for D64 M64).
    static_assert(FmhaTdmSchedSupportedGeometry<Geometry>::value);
    static constexpr index_t kScoreGroupColumns = 32;
    using ScoreSoftmax = FmhaTdmSchedSplitSoftmaxFor<Geometry::kN / kScoreGroupColumns,
                                                     Geometry::kQkMIter,
                                                     Options::kRowMaxMaximum>;
    using ScoreMapping = typename ScoreSoftmax::Mapping;
    static_assert(ScoreMapping::kNumMIter == Geometry::kQkMIter &&
                      ScoreMapping::kNumNIter == Geometry::kN / Geometry::kQkWmmaN &&
                      ScoreMapping::kNumSu == Geometry::kN / kScoreGroupColumns &&
                      ScoreMapping::kNIterPerSu == kScoreGroupColumns / Geometry::kQkWmmaN &&
                      ScoreMapping::kNumMsb ==
                          Geometry::kQkMIter * (kScoreGroupColumns / Geometry::kQkWmmaN) &&
                      ScoreMapping::kElementsPerWmma ==
                          Geometry::QkWmma::kCM0PerLane * Geometry::QkWmma::kCM1PerLane,
                  "split-softmax score mapping does not match the QK geometry");
    static_assert(OutputFragments::kNumFragments == Geometry::kPvMIter * Geometry::kPvNIter &&
                      OutputFragments::kNumDmsb % Geometry::kPvMIter == 0 &&
                      OutputFragments::kElementsPerFragment ==
                          Geometry::PvWmma::kCM0PerLane * Geometry::PvWmma::kCM1PerLane,
                  "output fragments do not match the PV geometry");

    // Q is used only during the prologue. Its descriptor may span both K buffers
    // when the KV tile is 64 rows; the LDS barrier precedes the K loads that overwrite it.
    static_assert(Geometry::kM * Geometry::kHeadDimQK * sizeof(typename Geometry::QDataType) <=
                  2 * kKFootprintBytes);
    static_assert(kLdsOffsetK0 % 16 == 0 && kLdsOffsetK1 % 16 == 0 && kLdsOffsetV0 % 16 == 0 &&
                  kLdsOffsetV1 % 16 == 0);

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetQKBlockGemmSu()
    {
        using GemmProblem =
            BlockGemmProblem<typename Problem::QDataType,
                             typename Problem::KDataType,
                             typename Problem::SaccDataType,
                             Problem::kBlockSize,
                             TileGemmShape<sequence<Problem::BlockFmhaShape::kM0,
                                                    Geometry::kQkSuColumns,
                                                    Problem::BlockFmhaShape::kQKHeaddim>,
                                           typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                                           typename Problem::BlockFmhaShape::Gemm0WarpTile>>;

        using WarpGemm = WarpGemmDispatcher<typename Problem::QDataType,
                                            typename Problem::KDataType,
                                            typename Problem::SaccDataType,
                                            Problem::BlockFmhaShape::Gemm0WarpTile::at(number<0>{}),
                                            Problem::BlockFmhaShape::Gemm0WarpTile::at(number<1>{}),
                                            Problem::BlockFmhaShape::Gemm0WarpTile::at(number<2>{}),
                                            true>;

        using BlockGemmPolicy =
            BlockGemmARegBRegCRegV2CustomPolicy<typename Problem::QDataType,
                                                typename Problem::KDataType,
                                                typename Problem::SaccDataType,
                                                typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                                                WarpGemm,
                                                GemmLoopOrder::MNK>;

        return BlockGemmARegBRegCRegV2<GemmProblem, BlockGemmPolicy>{};
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeKSuRegTileDistribution()
    {
        using BlockGemm = remove_cvref_t<decltype(GetQKBlockGemmSu<Problem>())>;
        static_assert(BlockGemm::NIterPerWarp == Geometry::kQkSuNIter &&
                      BlockGemm::KIterPerWarp == Geometry::kQkKIter);
        return make_static_tile_distribution(BlockGemm::MakeBBlockDistributionEncode());
    }

    // RAW fence: waits until the WMMA that produced c has completed, and so has read its A/B
    // operands (the WAR hazard is described at the top of this file). Reading the
    // result makes the compiler insert the WMMA result wait before the instruction that follows.
    template <typename CWarpTensor>
    CK_TILE_DEVICE static void WaitWmmaOperandsRead(CWarpTensor& c)
    {
        auto x = c.get_thread_buffer()[number<0>{}];
        asm volatile("v_mov_b32 %0, %0" : "+v"(x)::"memory");
        c.get_thread_buffer()(number<0>{}) = x;
    }

    // Access index of the J-th K read of a stage: in order, or k-major with Options::kKReadKMajor
    // (n = (J / 2) % 2, k = J / 4, h = J % 2: the slices the k-outer QK order needs first).
    template <index_t J>
    CK_TILE_HOST_DEVICE static constexpr index_t KReadAccess()
    {
        constexpr index_t kNumAccess = Geometry::kKSuLoadCount;
        if constexpr(Options::kKReadKMajor)
        {
            static_assert(kNumAccess % 4 == 0 && J >= 0 && J < kNumAccess);
            constexpr index_t kAccess = ((J / 2) % 2) * (kNumAccess / 2) + (J / 4) * 2 + (J % 2);
            static_assert(J != 0 || kAccess == 0, "access 0 must stay first (kLeadFirstLoad)");
            return kAccess;
        }
        else
            return J;
    }

    // Iterations of the QK WMMA with this ordinal: order (n, k, m), or (k, n, m) with
    // Options::kQkKOuter.
    struct QkWmmaCoords
    {
        index_t n;
        index_t k;
        index_t m;
    };
    template <typename BlockGemm>
    CK_TILE_HOST_DEVICE static constexpr QkWmmaCoords GetQkWmmaCoords(index_t ordinal)
    {
        constexpr index_t kN = BlockGemm::NIterPerWarp;
        constexpr index_t kK = BlockGemm::KIterPerWarp;
        constexpr index_t kM = BlockGemm::MIterPerWarp;
        if constexpr(Options::kQkKOuter)
            return {(ordinal / kM) % kN, ordinal / (kN * kM), ordinal % kM};
        else
            return {ordinal / (kK * kM), (ordinal / kM) % kK, ordinal % kM};
    }

    // FenceLastMIter: wait for the last M iteration's WMMA of each (n, k), the last reader of
    // the shared K operand registers that the next K read refills (see WaitWmmaOperandsRead).
    template <index_t WmmaOrdinal,
              typename BlockGemm,
              typename CBlockTensor,
              typename ABlockTensor,
              typename BBlockTensor,
              bool FenceLastMIter = false>
    CK_TILE_DEVICE static void RunQkSuWmma(CBlockTensor& c_block_tensor,
                                           const ABlockTensor& a_block_tensor,
                                           const BBlockTensor& b_block_tensor,
                                           bool_constant<FenceLastMIter> = {})
    {
        using WarpGemm    = typename BlockGemm::WarpGemm;
        using AWarpDstr   = typename WarpGemm::AWarpDstr;
        using BWarpDstr   = typename WarpGemm::BWarpDstr;
        using CWarpDstr   = typename WarpGemm::CWarpDstr;
        using AWarpTensor = typename WarpGemm::AWarpTensor;
        using BWarpTensor = typename WarpGemm::BWarpTensor;
        using CWarpTensor = typename WarpGemm::CWarpTensor;

        constexpr auto a_warp_y_lengths =
            to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto b_warp_y_lengths =
            to_sequence(BWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
        constexpr auto b_warp_y_index_zeros = uniform_sequence_gen_t<BWarpDstr::NDimY, 0>{};
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        static_assert(WmmaOrdinal >= 0 && WmmaOrdinal < BlockGemm::NIterPerWarp *
                                                            BlockGemm::KIterPerWarp *
                                                            BlockGemm::MIterPerWarp);
        constexpr auto kCoords   = GetQkWmmaCoords<BlockGemm>(WmmaOrdinal);
        constexpr index_t n_iter = kCoords.n;
        constexpr index_t k_iter = kCoords.k;
        constexpr index_t m_iter = kCoords.m;

        AWarpTensor a_warp_tensor;
        a_warp_tensor.get_thread_buffer() = a_block_tensor.get_y_sliced_thread_data(
            merge_sequences(sequence<m_iter, k_iter>{}, a_warp_y_index_zeros),
            merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

        BWarpTensor b_warp_tensor;
        b_warp_tensor.get_thread_buffer() = b_block_tensor.get_y_sliced_thread_data(
            merge_sequences(sequence<n_iter, k_iter>{}, b_warp_y_index_zeros),
            merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));

        CWarpTensor c_warp_tensor;
        c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
            merge_sequences(sequence<m_iter, n_iter>{}, c_warp_y_index_zeros),
            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

        WarpGemm{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);
        if constexpr(FenceLastMIter && m_iter == BlockGemm::MIterPerWarp - 1)
            WaitWmmaOperandsRead(c_warp_tensor);
        c_block_tensor.set_y_sliced_thread_data(
            merge_sequences(sequence<m_iter, n_iter>{}, c_warp_y_index_zeros),
            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
            c_warp_tensor.get_thread_buffer());
    }

    template <index_t Stage,
              typename Mode,
              typename BlockGemm,
              typename CBlockTensor,
              typename ABlockTensor,
              typename BBlockTensor,
              typename NextBBlockTensor,
              typename BTileWindow,
              bool UsePreparedTailBase = false>
    CK_TILE_DEVICE static void
    RunQkScheduledStage(const BlockGemm&,
                        CBlockTensor& c_block_tensor,
                        const ABlockTensor& a_block_tensor,
                        const BBlockTensor& b_block_tensor,
                        NextBBlockTensor& next_b_block_tensor,
                        const BTileWindow& b_lds_window,
                        const FmhaTdmPreparedKRead::Address* prepared_stage0 = nullptr,
                        bool_constant<UsePreparedTailBase>                   = {})
    {
        // UsePreparedTailBase: the caller formed the tail stage's V access-0 address before the
        // WMMA stream and passes it in prepared_stage0 (see the paged D64/N128 pipeline).
        static_assert(!UsePreparedTailBase || Stage == Geometry::kQkStages - 1);
        using SelectedSchedule = typename Mode::SelectedPolicy::Schedule;
        using Executor         = FmhaTdmSchedPlacementExecutor<SelectedSchedule, Mode>;
        using Kind             = FmhaTdmSchedLoadKind;
        static_assert(Stage >= 0 && Stage < SelectedSchedule::kNumQkStages);
        static_assert((Stage < SelectedSchedule::kNumQkStages - 1 &&
                       BTileWindow::NumAccessPerCoord == Geometry::kKSuLoadCount) ||
                      (Stage == SelectedSchedule::kNumQkStages - 1 &&
                       BTileWindow::NumAccessPerCoord == Geometry::kVStageLoadCount));
        constexpr bool kUseAffineKRead = Stage < Geometry::kQkStages - 1;

        const auto affine_k0 = [&]() {
            if constexpr(kUseAffineKRead)
            {
                if constexpr(Stage == 0)
                {
                    if(prepared_stage0 != nullptr)
                        return *prepared_stage0;
                }
                return FmhaTdmPreparedKRead::Prepare<0>(b_lds_window);
            }
            else
                return FmhaTdmPreparedKRead::Address{};
        }();
        const auto affine_v0 = [&]() {
            if constexpr(UsePreparedTailBase)
                return *prepared_stage0;
            else if constexpr(Stage == Geometry::kQkStages - 1)
                return FmhaTdmPreparedKRead::Prepare<0>(b_lds_window);
            else
                return FmhaTdmPreparedKRead::Address{};
        }();
        constexpr bool kLeadFirstLoadThisStage = Mode::SelectedPolicy::kLeadFirstLoad &&
                                                 (Stage == 0 || Stage == Geometry::kQkStages - 1);
        if constexpr(kLeadFirstLoadThisStage)
        {
            SelectedSchedule::template RunQkReadPrelude<Mode, Stage>([&](auto event) {
                static_assert(decltype(event)::value == 0);
                if constexpr(Stage == 0)
                    FmhaTdmAffineKRead::Load<Stage, 0, kKPhysicalStride>(
                        next_b_block_tensor, b_lds_window, affine_k0);
                else
                    FmhaTdmAffineVRead::Load<0, kVPhysicalStride>(
                        next_b_block_tensor, b_lds_window, affine_v0);
            });
            __builtin_amdgcn_sched_barrier(0);
        }
        auto emit_token = [&](auto kind, auto access) {
            if constexpr(decltype(kind)::value == Kind::KRead)
            {
                static_assert(Stage < Geometry::kQkStages - 1);
                if constexpr(kLeadFirstLoadThisStage && decltype(access)::value == 0)
                {
                    // Access0 was issued before WMMA0; do not issue it twice.
                }
                else
                    FmhaTdmAffineKRead::
                        Load<Stage, KReadAccess<decltype(access)::value>(), kKPhysicalStride>(
                            next_b_block_tensor, b_lds_window, affine_k0);
            }
            else if constexpr(decltype(kind)::value == Kind::VRead)
            {
                static_assert(Stage == Geometry::kQkStages - 1);
                if constexpr(kLeadFirstLoadThisStage && decltype(access)::value == 0)
                {
                    // Access0 was issued before WMMA0; do not issue it twice.
                }
                else
                    FmhaTdmAffineVRead::Load<decltype(access)::value, kVPhysicalStride>(
                        next_b_block_tensor, b_lds_window, affine_v0);
            }
        };
        // Grouped drains (FmhaTdmSchedGroupedDrainPlacement): the K slice of a pair's last-M WMMA
        // stays allocated (an empty asm use) until after the drain that covers it, so the reads
        // issued in between cannot be allocated onto it while that WMMA is in flight.
        auto keep_k_of_pair = [&](auto pair) {
            constexpr index_t kMIter = BlockGemm::MIterPerWarp;
            using KWarpDstr          = typename BlockGemm::WarpGemm::BWarpDstr;
            constexpr auto k_lengths =
                to_sequence(KWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
            constexpr auto k_zeros = uniform_sequence_gen_t<KWarpDstr::NDimY, 0>{};
            constexpr index_t kO   = kMIter * decltype(pair)::value + kMIter - 1; // last M reader
            constexpr auto kCoords = GetQkWmmaCoords<BlockGemm>(kO);
            typename BlockGemm::WarpGemm::BWarpTensor kept;
            kept.get_thread_buffer() = b_block_tensor.get_y_sliced_thread_data(
                merge_sequences(sequence<kCoords.n, kCoords.k>{}, k_zeros),
                merge_sequences(sequence<1, 1>{}, k_lengths));
            KeepOperandLive(kept);
        };
        auto emit_point = [&](auto, auto row, auto point) {
            using Point            = FmhaTdmSchedSchedulePoint;
            constexpr Point kPoint = decltype(point)::value;
            if constexpr(kPoint == Point::AfterTokens)
                __builtin_amdgcn_sched_barrier(0);
            else if constexpr(kPoint == Point::Drain || kPoint == Point::KeepPending)
                EmitQkGroupedDrain<typename SelectedSchedule::LocalPlacement,
                                   BlockGemm,
                                   decltype(row)::value,
                                   kPoint == Point::Drain>(keep_k_of_pair);
            else
                __builtin_amdgcn_sched_barrier(0x0002 | 0x0400);
        };

        // With the WMMA operand guard, the last M iteration's WMMAs keep the RAW fence: with
        // grouped waits only on the stage's last row, otherwise on every m1.
        auto emit_wmma = [&](auto, auto wmma) {
            constexpr index_t kOrd  = decltype(wmma)::value;
            constexpr bool kLastRow = kOrd == Geometry::kQkWmmasPerStage - 1;
            RunQkSuWmma<kOrd, BlockGemm>(c_block_tensor,
                                         a_block_tensor,
                                         b_block_tensor,
                                         bool_constant < kWmmaOperandGuard &&
                                             (kLastRow || !GroupedWait<Mode>()) > {});
        };
        Executor::template ExecuteQkStage<Stage>(emit_wmma, emit_token, emit_point);
        WaitQkStageTail<Stage>();
    }

    template <index_t Stage,
              typename Mode,
              typename NextBBlockTensor,
              typename BTileWindow,
              typename WmmaEmitter,
              typename RescaleEmitter,
              bool UsePreparedBase = false>
    CK_TILE_DEVICE static void
    RunPvScheduledStageWithWmma(NextBBlockTensor& next_b_block_tensor,
                                const BTileWindow& b_lds_window,
                                WmmaEmitter& emit_wmma,
                                RescaleEmitter& emit_rescale,
                                const FmhaTdmPreparedKRead::Address* prepared = nullptr,
                                bool_constant<UsePreparedBase>                = {})
    {
        // UsePreparedBase: *prepared is this stage's access-0 LDS address, formed by the
        // caller before the WMMA stream (V for the V stages, the next K tile for the last
        // stage). Every read of the stage is then a constant offset of that one register, so
        // no fresh VALU address write sits between a WMMA and the reads that follow it; with
        // kLeadFirstLoad the prelude's access 0 reads from it too.
        using SelectedSchedule = typename Mode::SelectedPolicy::Schedule;
        using Executor         = FmhaTdmSchedPlacementExecutor<SelectedSchedule, Mode>;
        using Kind             = FmhaTdmSchedLoadKind;
        static_assert(Stage >= 0 && Stage < SelectedSchedule::kNumPvStages);
        static_assert((Stage < SelectedSchedule::kNumPvStages - 1 &&
                       BTileWindow::NumAccessPerCoord == Geometry::kVStageLoadCount) ||
                      (Stage == SelectedSchedule::kNumPvStages - 1 &&
                       BTileWindow::NumAccessPerCoord == Geometry::kKSuLoadCount));

        FmhaTdmPreparedKRead::Address affine_k0{};
        FmhaTdmPreparedKRead::Address prepared_v1{};
        constexpr bool kLeadFirstLoadThisStage = Mode::SelectedPolicy::kLeadFirstLoad &&
                                                 (Stage == 0 || Stage == Geometry::kPvStages - 1);
        if constexpr(kLeadFirstLoadThisStage)
        {
            SelectedSchedule::template RunPvReadPrelude<Mode, Stage>([&](auto event) {
                static_assert(decltype(event)::value == 0);
                if constexpr(Stage == 0 && UsePreparedBase)
                    FmhaTdmAffineVRead::Load<0, kVPhysicalStride>(
                        next_b_block_tensor, b_lds_window, *prepared);
                else if constexpr(Stage == 0)
                {
                    prepared_v1 = FmhaTdmPreparedKRead::Prepare<1>(b_lds_window);
                    __builtin_amdgcn_sched_barrier(0);
                    VLoad::template LoadAccess<0>(next_b_block_tensor, b_lds_window);
                }
                else
                {
                    if constexpr(UsePreparedBase)
                        affine_k0 = *prepared;
                    else
                        affine_k0 = FmhaTdmPreparedKRead::Prepare<0>(b_lds_window);
                    FmhaTdmAffineKRead::Load<Stage, 0, kKPhysicalStride>(
                        next_b_block_tensor, b_lds_window, affine_k0);
                }
            });
            __builtin_amdgcn_sched_barrier(0);
        }
        auto emit_prepared_wmma = [&](auto stage, auto wmma) {
            if constexpr(!UsePreparedBase && !kLeadFirstLoadThisStage && Stage == 0 &&
                         decltype(wmma)::value == 0)
            {
                prepared_v1 = FmhaTdmPreparedKRead::Prepare<1>(b_lds_window);
                __builtin_amdgcn_sched_barrier(0);
            }
            if constexpr(!kLeadFirstLoadThisStage && Stage == Geometry::kPvStages - 1 &&
                         decltype(wmma)::value == 0)
            {
                if constexpr(UsePreparedBase)
                    affine_k0 = *prepared;
                else
                    affine_k0 = FmhaTdmPreparedKRead::Prepare<0>(b_lds_window);
            }
            emit_wmma(stage, wmma);
        };
        auto emit_token = [&](auto kind, auto access) {
            if constexpr(decltype(kind)::value == Kind::KRead)
            {
                static_assert(Stage == Geometry::kPvStages - 1);
                if constexpr(Geometry::kN == 64)
                    KLoad::template LoadInstruction<decltype(access)::value>(next_b_block_tensor,
                                                                             b_lds_window);
                else
                    FmhaTdmAffineKRead::
                        Load<Stage, KReadAccess<decltype(access)::value>(), kKPhysicalStride>(
                            next_b_block_tensor, b_lds_window, affine_k0);
            }
            else if constexpr(decltype(kind)::value == Kind::VRead)
            {
                static_assert(Stage < Geometry::kPvStages - 1);
                if constexpr(UsePreparedBase)
                    FmhaTdmAffineVRead::Load<decltype(access)::value, kVPhysicalStride>(
                        next_b_block_tensor, b_lds_window, *prepared);
                else
                    VLoad::template LoadAccess<decltype(access)::value>(next_b_block_tensor,
                                                                        b_lds_window);
            }
        };
        auto emit_point = [](auto, auto, auto point) {
            using Point            = FmhaTdmSchedSchedulePoint;
            constexpr Point kPoint = decltype(point)::value;
            if constexpr(kPoint == Point::AfterTokens)
                __builtin_amdgcn_sched_barrier(0);
            else if constexpr(kPoint == Point::Drain)
            {
                // Grouped drains (FmhaTdmSchedGroupedDrainPlacement): the WMMA emitter keeps the
                // operands of the drained rows allocated until the next row (PvKeptRows).
                __builtin_amdgcn_sched_barrier(0);
                // Full drain only: a counted va_vdst(k >= 1) does not cover the WAR.
                asm volatile("s_wait_alu depctr_va_vdst(0)" ::: "memory");
                __builtin_amdgcn_sched_barrier(0);
            }
            else
                __builtin_amdgcn_sched_barrier(0x0002 | 0x0400);
        };

        auto emit_prepared_token = [&](auto kind, auto access) {
            if constexpr(kLeadFirstLoadThisStage && decltype(access)::value == 0)
            {
                // Access0 was issued before WMMA0; do not issue it twice.
            }
            else if constexpr(!UsePreparedBase && Stage == 0 &&
                              decltype(kind)::value == Kind::VRead && decltype(access)::value == 1)
                FmhaTdmPreparedVRead::Load<1>(next_b_block_tensor, b_lds_window, prepared_v1);
            else
                emit_token(kind, access);
        };

        Executor::template ExecutePvStage<Stage>(
            emit_prepared_wmma, emit_prepared_token, emit_point, emit_rescale);
        WaitPvStageTail<Stage>();
    }

    // WMMA operand guard (the hazard note at the top of this file). The page-tile tuning sets it
    // for the vectorized-KV page tile only; the linear page tile and the forward kernels keep their
    // unguarded refills (see the scope paragraph there).
    static constexpr bool kWmmaOperandGuard = Options::kWmmaOperandGuard;

    // Empty use: keeps an operand's registers live so no later load is allocated onto them.
    template <typename Tensor>
    CK_TILE_DEVICE static void KeepOperandLive(const Tensor& t)
    {
        using Buffer            = remove_cvref_t<decltype(t.get_thread_buffer())>;
        constexpr index_t kSize = sizeof(Buffer);
        static_assert(kSize % 16 == 0);
        const auto chunks = t.get_thread_buffer().template get_as<int32x4_t>();
#pragma unroll
        for(index_t i = 0; i < kSize / 16; ++i)
            asm volatile("" ::"v"(chunks[i]) : "memory");
    }

    // The Drain or KeepPending point of QK row Row of a grouped-drain schedule: the full drain
    // (Drain only), then the keep-alive of the last-M K slices of the pairs it covers. A function
    // of its own, so only grouped-drain schedules instantiate it.
    template <typename Placement, typename BlockGemm, index_t Row, bool Drain, typename KeepPair>
    CK_TILE_DEVICE static void EmitQkGroupedDrain(KeepPair& keep_k_of_pair)
    {
        static_assert(Placement::kQkMIter == BlockGemm::MIterPerWarp);
        constexpr auto kPairs = Placement::QkKeptPairs(Row, Drain);
        __builtin_amdgcn_sched_barrier(0);
        if constexpr(Drain)
            // Full drain only: a counted va_vdst(k >= 1) does not cover the WAR.
            asm volatile("s_wait_alu depctr_va_vdst(0)" ::: "memory");
        static_for<kPairs.begin, kPairs.end, 1>{}([&](auto q) { keep_k_of_pair(q); });
        __builtin_amdgcn_sched_barrier(0);
    }

    // Grouped waits: the schedule runs FmhaTdmSchedGroupedDrainPlacement, which the page tile
    // selects for its guarded WMMA-first problems. The other placements keep a RAW fence after
    // every last reader.
    template <typename Mode>
    CK_TILE_HOST_DEVICE static constexpr bool GroupedWait()
    {
        constexpr bool kGrouped =
            kIsFmhaTdmSchedGroupedDrain<typename Mode::SelectedPolicy::Schedule::LocalPlacement>;
        static_assert(!kGrouped || kWmmaOperandGuard,
                      "the grouped drains belong to the WMMA operand guard");
        return kGrouped;
    }

    // PV rows [kPvLastMRow, kPvWmmasPerStage) of a stage are its last M iteration.
    static constexpr index_t kPvLastMRow = kFmhaTdmSchedPvLastMRow<Geometry>;

    // The PV WMMAs are issued by the pipeline's WMMA emitter, which guards PV row r as follows.
    // RAW fence after its WMMA: the last M iteration's rows, with grouped waits only the stage's
    // last row.
    template <typename Mode>
    CK_TILE_HOST_DEVICE static constexpr bool PvRawFence(index_t r)
    {
        return kWmmaOperandGuard && r >= kPvLastMRow &&
               (!GroupedWait<Mode>() || r == Geometry::kPvWmmasPerStage - 1);
    }
    // Rows [begin, end) whose operands stay allocated (KeepOperandLive) at row r: with grouped
    // waits, the placement's PvKeptRows; otherwise none.
    template <typename Mode>
    CK_TILE_HOST_DEVICE static constexpr FmhaTdmSchedRowRange PvKeptRows(index_t r,
                                                                         bool after_fence)
    {
        if constexpr(GroupedWait<Mode>())
        {
            using Placement = typename Mode::SelectedPolicy::Schedule::LocalPlacement;
            return Placement::PvKeptRows(r, after_fence);
        }
        else
            return {r, r};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool UseCountdownLoop()
    {
        return Geometry::kHeadDimQK == 128 && !Problem::FmhaMask::IsMasking && !Problem::kHasSink &&
               Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool DeferTensorReady()
    {
        // Defer the V tensor wait only for the supported geometry and problem cases.
        return (Geometry::kHeadDimQK == 128 ||
                (std::is_same_v<Geometry, FmhaTdmSchedBf16D192N64Geometry> &&
                 !Problem::FmhaMask::IsMasking)) &&
               !Problem::kHasSink && Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS &&
               Problem::BlockFmhaShape::Gemm0BlockWarps::at(number<1>{}) == 1;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool UseCompilerMax()
    {
        // Let the allocator place score tuples above the inline-asm low-VGPR range.
        // D64/D128 and D192 dense batch use compiler-scheduled maximum chains.
        return Geometry::kHeadDimQK <= 128 ||
               (Geometry::kHeadDimQK == 192 && !Problem::kIsGroupMode &&
                !Problem::FmhaMask::IsMasking);
    }

    // A row's scores can all be -inf within a tile: masked, or with an elementwise bias that
    // carries a materialized mask (as in BlockFmhaPipelineQRKSVS). Such a row keeps a finite
    // exponent base, so the online softmax does not form -inf - (-inf).
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool ValidateRowMax()
    {
        return Problem::FmhaMask::IsMasking ||
               Problem::BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS;
    }

    template <typename Problem, typename ScoreTensor, typename RowTensor>
    CK_TILE_DEVICE static void RunSplitSoftmaxPart01(
        ScoreTensor& score, RowTensor& row_max, float log2e_scale, float& delta_m0, float& delta_m1)
    {
        if constexpr(ScoreMapping::kNumMIter == 1)
        {
            using Softmax = ScoreSoftmax;
            using Mapping = typename Softmax::Mapping;
            static_assert(ScoreTensor::get_thread_buffer_size() == Mapping::kThreadBufferSize);
            static_assert(RowTensor::get_thread_buffer_size() == 1);
            const float old_max_m0         = row_max.get_thread_buffer()[number<0>{}];
            constexpr bool kUseCompilerMax = UseCompilerMax<Problem>();
            const auto p0_m0 = Softmax::template RunPart0<0, ScoreTensor, kUseCompilerMax>(
                score, old_max_m0, log2e_scale);
            const auto p0_m1 = Softmax::template RunPart0<1, ScoreTensor, kUseCompilerMax>(
                score, old_max_m0, log2e_scale);
            constexpr bool kValidateMax = ValidateRowMax<Problem>();
            const auto p1_m0            = Softmax::template RunPart1<kValidateMax>(
                p0_m0.local_max, p0_m1.local_max, p0_m0.old_max_log2e, log2e_scale);
            row_max.get_thread_buffer()[number<0>{}] = p1_m0.row_max;
            delta_m0                                 = p1_m0.delta;
            delta_m1                                 = 0.0f;
        }
        else
        {
            using Softmax = ScoreSoftmax;
            using Mapping = typename Softmax::Mapping;

            // Bias, ALiBi and a logits soft cap are applied to the scores before the softmax (paged
            // core only; the forward core rejects them); with bias the caller passes a unit scale.
            static_assert(ScoreTensor::get_thread_buffer_size() == Mapping::kThreadBufferSize);
            static_assert(RowTensor::get_thread_buffer_size() == 2);

            const float old_max_m0 = row_max.get_thread_buffer()[number<0>{}];
            const float old_max_m1 = row_max.get_thread_buffer()[number<1>{}];

            constexpr bool kUseCompilerMax = UseCompilerMax<Problem>();
            const auto p0_m0 = Softmax::template RunPart0<0, ScoreTensor, kUseCompilerMax>(
                score, old_max_m0, log2e_scale);
            const auto p0_m1 = Softmax::template RunPart0<1, ScoreTensor, kUseCompilerMax>(
                score, old_max_m0, log2e_scale);
            const auto p0_m2 = Softmax::template RunPart0<2, ScoreTensor, kUseCompilerMax>(
                score, old_max_m1, log2e_scale);
            const auto p0_m3 = Softmax::template RunPart0<3, ScoreTensor, kUseCompilerMax>(
                score, old_max_m1, log2e_scale);

            constexpr bool kValidateMax = ValidateRowMax<Problem>();
            const auto p1_m0            = Softmax::template RunPart1<kValidateMax>(
                p0_m0.local_max, p0_m1.local_max, p0_m0.old_max_log2e, log2e_scale);
            const auto p1_m1 = Softmax::template RunPart1<kValidateMax>(
                p0_m2.local_max, p0_m3.local_max, p0_m2.old_max_log2e, log2e_scale);

            row_max.get_thread_buffer()[number<0>{}] = p1_m0.row_max;
            row_max.get_thread_buffer()[number<1>{}] = p1_m1.row_max;
            delta_m0                                 = p1_m0.delta;
            delta_m1                                 = p1_m1.delta;
        }
    }

    template <typename Problem, typename ScoreTensor, typename RowTensor>
    CK_TILE_DEVICE static void RunSplitSoftmaxPart2AndGetScale(ScoreTensor& score,
                                                               const RowTensor& row_max,
                                                               RowTensor& row_sum,
                                                               float log2e_scale,
                                                               float delta_m0,
                                                               float delta_m1,
                                                               float& output_scale_m0,
                                                               float& output_scale_m1)
    {
        if constexpr(ScoreMapping::kNumMIter == 1)
        {
            using Softmax = ScoreSoftmax;
            using Mapping = typename Softmax::Mapping;
            static_assert(ScoreTensor::get_thread_buffer_size() == Mapping::kThreadBufferSize);
            static_assert(RowTensor::get_thread_buffer_size() == 1);
            constexpr bool kValidateMax = ValidateRowMax<Problem>();
            const float raw_max_m0      = row_max.get_thread_buffer()[number<0>{}];
            const float row_max_m0 =
                kValidateMax && raw_max_m0 <= -numeric<float>::infinity() ? 0.0f : raw_max_m0;
            const float old_sum_m0 = row_sum.get_thread_buffer()[number<0>{}];
            const float sum_m0 =
                Softmax::template RunPart2LocalSum<0>(score, row_max_m0, log2e_scale);
            const float sum_m1 =
                Softmax::template RunPart2LocalSum<1>(score, row_max_m0, log2e_scale);
            row_sum.get_thread_buffer()[number<0>{}] =
                Softmax::UpdateRowSum(old_sum_m0, Softmax::MergeRowSum(sum_m0, sum_m1), delta_m0);
            output_scale_m0 = Softmax::Exp2(delta_m0);
            output_scale_m1 = 1.0f;
        }
        else
        {
            using Softmax = ScoreSoftmax;
            using Mapping = typename Softmax::Mapping;

            // Bias, ALiBi and a logits soft cap are applied to the scores before the softmax (paged
            // core only; the forward core rejects them); with bias the caller passes a unit scale.
            static_assert(ScoreTensor::get_thread_buffer_size() == Mapping::kThreadBufferSize);
            static_assert(RowTensor::get_thread_buffer_size() == 2);

            constexpr bool kValidateMax = ValidateRowMax<Problem>();

            const auto validated_max = [](float value) {
                if constexpr(kValidateMax)
                {
                    return value <= -numeric<float>::infinity() ? 0.0f : value;
                }
                else
                {
                    return value;
                }
            };

            const float row_max_m0 = row_max.get_thread_buffer()[number<0>{}];
            const float row_max_m1 = row_max.get_thread_buffer()[number<1>{}];
            const float old_sum_m0 = row_sum.get_thread_buffer()[number<0>{}];
            const float old_sum_m1 = row_sum.get_thread_buffer()[number<1>{}];

            const float sum_m0 = Softmax::template RunPart2LocalSum<0>(
                score, validated_max(row_max_m0), log2e_scale);
            const float sum_m1 = Softmax::template RunPart2LocalSum<1>(
                score, validated_max(row_max_m0), log2e_scale);
            const float sum_m2 = Softmax::template RunPart2LocalSum<2>(
                score, validated_max(row_max_m1), log2e_scale);
            const float sum_m3 = Softmax::template RunPart2LocalSum<3>(
                score, validated_max(row_max_m1), log2e_scale);

            row_sum.get_thread_buffer()[number<0>{}] =
                Softmax::UpdateRowSum(old_sum_m0, Softmax::MergeRowSum(sum_m0, sum_m1), delta_m0);
            row_sum.get_thread_buffer()[number<1>{}] =
                Softmax::UpdateRowSum(old_sum_m1, Softmax::MergeRowSum(sum_m2, sum_m3), delta_m1);

            output_scale_m0 = Softmax::Exp2(delta_m0);
            output_scale_m1 = Softmax::Exp2(delta_m1);
        }
    }

    template <index_t Ordinal, typename Fragments>
    CK_TILE_DEVICE static void
    RescaleOutputFragment(Fragments& output, float scale_m0, float scale_m1)
    {
        static_assert(Ordinal >= 0 && Ordinal < OutputFragments::kNumFragments);
        constexpr index_t d_msb = Ordinal / OutputFragments::kNumN;
        const float scale       = d_msb < 2 ? scale_m0 : scale_m1;
        auto& fragment          = output.at(number<Ordinal>{});
        static_for<0, OutputFragments::kElementsPerFragment, 1>{}(
            [&](auto element) { fragment[decltype(element)::value] *= scale; });
    }

    template <typename Problem,
              bool WaitTensorAfterMax = false,
              typename ScoreTensor,
              typename RowTensor,
              typename Fragments>
    CK_TILE_DEVICE static void RunSplitSoftmaxFragments(ScoreTensor& score,
                                                        RowTensor& row_max,
                                                        RowTensor& row_sum,
                                                        Fragments& output,
                                                        float log2e_scale)
    {
        float delta_m0;
        float delta_m1;
        RunSplitSoftmaxPart01<Problem>(score, row_max, log2e_scale, delta_m0, delta_m1);

        if constexpr(WaitTensorAfterMax)
            s_wait_tensorcnt_barrier<kVPrefetchTensorCount>();

        const auto rescale_output = [&](float scale_m0, float scale_m1) {
            static_for<0, OutputFragments::kNumFragments, 1>{}([&](auto ordinal) {
                constexpr index_t d_msb = decltype(ordinal)::value / OutputFragments::kNumN;
                const float scale       = d_msb < 2 ? scale_m0 : scale_m1;
                auto& fragment          = output.at(ordinal);
                static_for<0, OutputFragments::kElementsPerFragment, 1>{}(
                    [&](auto element) { fragment[decltype(element)::value] *= scale; });
            });
        };
        if constexpr(kEarlyOutputRescale)
        {
            using Softmax = ScoreSoftmax;
            rescale_output(Softmax::Exp2(delta_m0), Softmax::Exp2(delta_m1));
        }

        float output_scale_m0;
        float output_scale_m1;
        RunSplitSoftmaxPart2AndGetScale<Problem>(score,
                                                 row_max,
                                                 row_sum,
                                                 log2e_scale,
                                                 delta_m0,
                                                 delta_m1,
                                                 output_scale_m0,
                                                 output_scale_m1);

        if constexpr(!kEarlyOutputRescale)
        {
            rescale_output(output_scale_m0, output_scale_m1);
        }
    }

    // Problems of the forward pipeline: this policy's geometry, without a bias (the forward
    // pipeline applies none). A policy derived for another pipeline widens the feature set.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool IsSupportedProblem()
    {
        return Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS &&
               IsSupportedGeometryProblem<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool IsSupportedGeometryProblem()
    {
        using Shape = remove_cvref_t<typename Problem::BlockFmhaShape>;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
        constexpr bool is_wave32 = get_warp_size() == Geometry::kWaveSize;
#else
        constexpr bool is_wave32 = true;
#endif
        // The problem must describe exactly this policy's geometry (dtypes, block tile, wave
        // layout, WMMA tiles). Both GEMMs share one wave layout: P stays in the lanes that
        // produced S.
        return std::is_same_v<FmhaTdmSchedGeometryFor<Problem, Geometry::kQkSuColumns>, Geometry> &&
               Shape::Gemm1BlockWarps::at(number<0>{}) == Shape::Gemm0BlockWarps::at(number<0>{}) &&
               Shape::Gemm1BlockWarps::at(number<1>{}) == Shape::Gemm0BlockWarps::at(number<1>{}) &&
               Shape::Gemm1BlockWarps::at(number<2>{}) == Shape::Gemm0BlockWarps::at(number<2>{}) &&
               std::is_same_v<remove_cvref_t<typename Problem::SaccDataType>, float> &&
               std::is_same_v<remove_cvref_t<typename Problem::OaccDataType>, float> &&
               Shape::kK0 == Geometry::kQkWmmaK && Shape::kSubQKHeaddim == Geometry::kHeadDimQK &&
               Shape::IsVLayoutRowMajor && Shape::NumWarps == Geometry::kWaves && is_wave32;
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetLdsOffsetK0() { return kLdsOffsetK0; }
    CK_TILE_HOST_DEVICE static constexpr index_t GetLdsOffsetK1() { return kLdsOffsetK1; }
    CK_TILE_HOST_DEVICE static constexpr index_t GetLdsOffsetV0() { return kLdsOffsetV0; }
    CK_TILE_HOST_DEVICE static constexpr index_t GetLdsOffsetV1() { return kLdsOffsetV1; }
    CK_TILE_HOST_DEVICE static constexpr index_t GetLdsArenaSize() { return kLdsArenaSize; }

    template <typename Problem, bool LoadOnce = true>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKDramTileDistribution()
    {
        static_assert(IsSupportedGeometryProblem<Problem>(),
                      "qr_tdm_sched policy geometry does not match the problem");
        static_assert(LoadOnce, "qr_tdm_sched requires a full-head K TDM load");
        constexpr index_t warp_num = Problem::kBlockSize / get_warp_size();
        static_assert(kLdsRows % warp_num == 0);

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<warp_num, kLdsRows / warp_num>, sequence<kKPhysicalStride>>,
                tuple<sequence<1>>,
                tuple<sequence<0>>,
                sequence<1, 2>,
                sequence<1, 0>>{},
            bool_constant<true>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsWriteBlockDescriptor()
    {
        static_assert(IsSupportedGeometryProblem<Problem>(),
                      "qr_tdm_sched policy geometry does not match the problem");
        if constexpr(detail::fmha_tdm_is_vectorized_kv<Problem>::value)
        {
            // Vectorized KV: each TDM box (G rows: the P16 fragment or the 32-row wave slice)
            // lands as the page stores it, [D/8][G][8]: 16-byte column vectors of G consecutive
            // tokens one after another. A K read (8 columns of one token) is one aligned b128.
            constexpr index_t kGroupRows = Problem::kPageBlockSize == 16 ? 16 : 32;
            static_assert(kLdsRows % kGroupRows == 0 && kKValidWidth % 8 == 0);
            constexpr auto grouped =
                make_naive_tensor_descriptor(make_tuple(number<kLdsRows / kGroupRows>{},
                                                        number<kKValidWidth / 8>{},
                                                        number<kGroupRows>{},
                                                        number<8>{}),
                                             make_tuple(number<kGroupRows * kKValidWidth>{},
                                                        number<kGroupRows * 8>{},
                                                        number<8>{},
                                                        number<1>{}),
                                             number<8>{},
                                             number<1>{});
            return transform_tensor_descriptor(
                grouped,
                make_tuple(
                    make_merge_transform(
                        make_tuple(number<kLdsRows / kGroupRows>{}, number<kGroupRows>{})),
                    make_merge_transform(make_tuple(number<kKValidWidth / 8>{}, number<8>{}))),
                make_tuple(sequence<0, 2>{}, sequence<1, 3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
        else
        {
            constexpr index_t kKPack = BasePolicy::template GetSmemKPackK<Problem>();
            return make_naive_tensor_descriptor(
                make_tuple(number<kLdsRows>{}, number<kKPhysicalStride>{}),
                make_tuple(number<kKPhysicalStride>{}, number<1>{}),
                number<kKPack>{},
                number<1>{});
        }
    }

    template <typename Problem, bool LoadOnce = true>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsBlockDescriptor()
    {
        static_assert(LoadOnce, "qr_tdm_sched requires a full-head K LDS descriptor");
        return MakeKLdsWriteBlockDescriptor<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsWriteBlockDescriptor()
    {
        static_assert(IsSupportedGeometryProblem<Problem>(),
                      "qr_tdm_sched policy geometry does not match the problem");
        if constexpr(detail::fmha_tdm_is_vectorized_kv<Problem>::value)
        {
            // Vectorized KV: [rows/8][D][8], the 8 tokens of a column adjacent (the page
            // layout; the TDM copies whole groups of 8 rows), with 16 padding elements after
            // every 512 (the TDM pads each 1 KB it writes), so that the two half-waves of a read,
            // which read neighbouring groups, start on different banks. Every read of this tile
            // goes through the TDM-schedule V read helpers, which address it themselves with
            // plain b128 loads (FmhaTdmVecVRead); the declared column vector of 8 is the
            // transposed-read geometry they keep (the access count and the element groups of
            // each access), not a property of this layout.
            static_assert(kLdsRows % 8 == 0 && kVLogicalWidth % 64 == 0);
            constexpr index_t kBlock       = 512 + 16; // one padded 1 KB block: 64 columns x 8
            constexpr index_t kGroupStride = kVLogicalWidth / 64 * kBlock;
            constexpr auto grouped         = make_naive_tensor_descriptor(
                make_tuple(number<kLdsRows / 8>{},
                           number<8>{},
                           number<kVLogicalWidth / 64>{},
                           number<64>{}),
                make_tuple(number<kGroupStride>{}, number<1>{}, number<kBlock>{}, number<8>{}),
                number<8>{},
                number<1>{});
            return transform_tensor_descriptor(
                grouped,
                make_tuple(
                    make_merge_transform(make_tuple(number<kLdsRows / 8>{}, number<8>{})),
                    make_merge_transform(make_tuple(number<kVLogicalWidth / 64>{}, number<64>{}))),
                make_tuple(sequence<0, 1>{}, sequence<2, 3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
        else
        {
            constexpr index_t kKPack = BasePolicy::template GetSmemKPackV<Problem>();
            return make_naive_tensor_descriptor(
                make_tuple(number<kLdsRows>{}, number<kVLogicalWidth>{}),
                make_tuple(number<kVPhysicalStride>{}, number<1>{}),
                number<kKPack>{},
                number<1>{});
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsReadBlockDescriptor()
    {
        return MakeVLdsWriteBlockDescriptor<Problem>();
    }

    template <typename Problem, bool Xor = false>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsBlockDescriptor()
    {
        static_assert(!Xor, "D192/V128 TDM writer requires an affine V LDS descriptor");
        return MakeVLdsWriteBlockDescriptor<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVRegTileDistribution()
    {
        static_assert(IsSupportedGeometryProblem<Problem>(),
                      "qr_tdm_sched policy geometry does not match the problem");
        using BlockGemm = remove_cvref_t<decltype(BasePolicy::template GetPVBlockGemm<Problem>())>;
        using WarpGemm  = typename BlockGemm::WarpGemm;

        constexpr index_t kMWarp        = Geometry::kWaves;
        constexpr index_t kNWarp        = 1;
        constexpr index_t kNIterPerWarp = Geometry::kPvNIter;
        constexpr index_t kKIterPerWarp = Geometry::kPvStageKIter;
        static_assert(kNIterPerWarp == kVLogicalWidth / (kNWarp * WarpGemm::kN) &&
                      kKIterPerWarp == Geometry::kPvStageK / WarpGemm::kK);

        constexpr auto outer_encoding = tile_distribution_encoding<
            sequence<kMWarp>,
            tuple<sequence<kNIterPerWarp, kNWarp>, sequence<kKIterPerWarp>>,
            tuple<sequence<0, 1>>,
            tuple<sequence<0, 1>>,
            sequence<2, 1>,
            sequence<0, 0>>{};
        constexpr auto block_encoding = detail::make_embed_tile_distribution_encoding(
            outer_encoding, typename WarpGemm::BWarpDstrEncoding{});
        using ReadEncoding =
            typename InputTileDistributionTraits<decltype(block_encoding),
                                                 typename Problem::VDataType>::TransposedDstrEncode;

        return make_static_tile_distribution(ReadEncoding{});
    }

    template <typename Problem, bool LoadOnce = true>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeK()
    {
        static_assert(IsSupportedGeometryProblem<Problem>(),
                      "qr_tdm_sched policy geometry does not match the problem");
        static_assert(LoadOnce, "qr_tdm_sched requires a full-head K LDS allocation");
        return kKFootprintBytes;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeV()
    {
        static_assert(IsSupportedGeometryProblem<Problem>(),
                      "qr_tdm_sched policy geometry does not match the problem");
        return kVFootprintBytes;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        static_assert(IsSupportedGeometryProblem<Problem>(),
                      "qr_tdm_sched policy geometry does not match the problem");
        return GetLdsArenaSize();
    }

    template <typename Problem>
    static constexpr bool kUsePaddedQ =
        Geometry::kHeadDimQK <= 128 && (Geometry::kM == 128 || Geometry::kM == 64);

    template <typename Problem>
    using LdsPaddingConfigQ = std::conditional_t<
        kUsePaddedQ<Problem>,
        detail::
            LdsPaddingConfig<true, Geometry::kHeadDimQK * sizeof(typename Geometry::QDataType), 16>,
        typename BasePolicy::template LdsPaddingConfigQ<Problem>>;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQLdsBlockDescriptor()
    {
        static_assert(IsSupportedGeometryProblem<Problem>(),
                      "V128 policy received an invalid problem");
        if constexpr(kUsePaddedQ<Problem>)
        {
            using Padding = LdsPaddingConfigQ<Problem>;
            constexpr auto desc =
                detail::make_qr_tdm_row_major_lds_descriptor<typename Problem::QDataType,
                                                             Geometry::kM,
                                                             Geometry::kHeadDimQK,
                                                             Padding,
                                                             detail::kQrTdmLdsAccessBytes>();
            static_assert(desc.get_element_space_size() * sizeof(typename Problem::QDataType) <=
                          2 * kKFootprintBytes);
            return desc;
        }
        else
        {
            constexpr auto desc = BasePolicy::template MakeQLdsBlockDescriptor<Problem>();
            static_assert(desc.get_element_space_size() * sizeof(typename Problem::QDataType) <=
                              2 * kKFootprintBytes,
                          "Q LDS descriptor overlaps the V buffer");
            return desc;
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeQ()
    {
        return MakeQLdsBlockDescriptor<Problem>().get_element_space_size() *
               sizeof(typename Problem::QDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsPaddingConfigQ()
    {
        static_assert(IsSupportedGeometryProblem<Problem>(),
                      "V128 policy received an invalid problem");
        using Padding = detail::EncodedTdmPadding<LdsPaddingConfigQ<Problem>>;
        return make_tuple(number<Padding::kEnabled>{},
                          number<Padding::kPadAmount>{},
                          number<Padding::kPadInterval>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsPaddingConfigK()
    {
        static_assert(IsSupportedGeometryProblem<Problem>(),
                      "qr_tdm_sched policy geometry does not match the problem");
        return make_tuple(number<false>{}, number<0>{}, number<0>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsPaddingConfigV()
    {
        static_assert(IsSupportedGeometryProblem<Problem>(),
                      "qr_tdm_sched policy geometry does not match the problem");
        if constexpr(detail::fmha_tdm_is_vectorized_kv<Problem>::value)
            return make_tuple(number<true>{}, number<7>{}, number<7>{}); // 32 B every 1 KB
        else
            return make_tuple(number<true>{}, number<kVPadAmount>{}, number<kVPadInterval>{});
    }
};

template <typename Geometry>
struct FmhaTdmSchedTuningSelector
{
    static constexpr index_t kBlockPerCu =
        Geometry::kHeadDimQK == 64 ? (Geometry::kN == 128 ? 1 : 3)
                                   : (Geometry::kHeadDimQK == 192 && Geometry::kN == 64 ? 2 : 1);
    using type = FmhaTdmSchedDerivedTuning<Geometry, kBlockPerCu>;
};

template <typename Geometry>
using FmhaTdmSchedDefaultTuning = typename FmhaTdmSchedTuningSelector<Geometry>::type;

template <typename Problem>
using FmhaTdmSchedPolicyFor = BlockFmhaPipelineQRKSVSTdmSchedPolicy<
    FmhaTdmSchedGeometryFor<Problem>,
    FmhaTdmSchedDefaultTuning<FmhaTdmSchedGeometryFor<Problem>>>;

} // namespace ck_tile
