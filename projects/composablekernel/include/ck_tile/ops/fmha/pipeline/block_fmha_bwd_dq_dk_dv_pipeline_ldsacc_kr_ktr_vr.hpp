// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/block_dropout.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_default_policy.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_ldsacc_policy.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"

namespace ck_tile {

#ifndef CK_TILE_FMHA_BWD_V_NONRESIDENT
#define CK_TILE_FMHA_BWD_V_NONRESIDENT 1
#endif

// kM0 floor for the V-in-LDS choice above. The kM0 >= 64 gate it defaults to is
// inherited from kDVInReg, whose comment justifies it for the dV accumulator --
// V itself has never been measured below kM0 64. Set to 0 to put V in LDS on
// the kM0 16/32 tiles too.
#ifndef CK_TILE_FMHA_BWD_V_NONRESIDENT_MIN_M0
#define CK_TILE_FMHA_BWD_V_NONRESIDENT_MIN_M0 64
#endif

// headdim floors for the same choice. V in registers costs
// kN0 * kVHeaddim / kBlockSize VGPRs, so the kM0 gate alone cannot separate
// d=128 (128 regs, ~16 of 1024 free) from d=32/64 (32/64 regs, 614/444 free).
// Measured on gfx1250: d=32 wants V resident under both masks (+1.5..3.4%),
// d=64 only under a mask (+3.6..6.0% causal, -2.2..2.9% nomask), d=128 neither.
#ifndef CK_TILE_FMHA_BWD_V_NONRESIDENT_MIN_HDIM
#define CK_TILE_FMHA_BWD_V_NONRESIDENT_MIN_HDIM 128
#endif
#ifndef CK_TILE_FMHA_BWD_V_NONRESIDENT_MIN_HDIM_NOMASK
#define CK_TILE_FMHA_BWD_V_NONRESIDENT_MIN_HDIM_NOMASK 64
#endif

// Sink the next iteration's Q/LSE loads past gemm_4, so their live ranges do
// not span it. dO/D are already loaded after gemm_4; Q/LSE were the odd ones
// out. Safe: gemm_4 touches neither, and nothing between gemm_4 and the dO/D
// loads writes LDS or barriers, so they sit in the same validity window the
// dO/D loads already rely on.
#ifndef CK_TILE_FMHA_BWD_SINK_TDM_WAIT
#define CK_TILE_FMHA_BWD_SINK_TDM_WAIT 1
#endif

// 0 = keep the hand-written scheduler prescriptions, 1 = drop them where the
// loop body holds a single Q tile, 2 = drop them always. See schedgate.py.
//
// Mode 1 existed because a doubled (mirror-tile-paired) body measured 1.4-3.2%
// worse without the prescriptions. That no longer reproduces: on this base the
// paired bodies are 9.4-14.5% FASTER dropped, with the unpaired instances flat
// either way as an internal control. Mode 1 is kept so the comparison can be
// re-run, but it is no longer the default.
#ifndef CK_TILE_FMHA_BWD_SCHED_DROP_MODE
#define CK_TILE_FMHA_BWD_SCHED_DROP_MODE 2
#endif

// Bitmask of GemmStagedScheduler prescriptions to KEEP even when
// CK_TILE_FMHA_BWD_SCHED_DROP_MODE would drop them; bit N selects scheduler N.
// Dropping all of them is right for the shipped V-in-LDS config, but with V
// register-resident the <4> prescription is what stops the 64 dQ atomic
// addresses collapsing into a serial recurrence through one register (60
// add->lshl pairs, s_clause 0, +135 s_wait_alu). This lets that one come back
// without paying for the rest.
#ifndef CK_TILE_FMHA_BWD_SCHED_KEEP_MASK
#define CK_TILE_FMHA_BWD_SCHED_KEEP_MASK 0
#endif

// Keep only dV in registers, leave dK in LDS. Halves the extra register cost
// versus the fully register-resident pipeline (128 VGPR instead of 256).
#ifndef CK_TILE_FMHA_BWD_DV_IN_REG
#define CK_TILE_FMHA_BWD_DV_IN_REG 1
#endif

// kM0 floor for the dV-in-registers choice above. The kM0 >= 64 gate it
// defaults to ties dV-in-reg to evicting V to LDS as a package, and the package
// was only measured as a package. Set to 0 to keep dV in registers on the
// kM0 16/32 tiles as well, so the kM0 and the V-placement variables can be
// separated.
#ifndef CK_TILE_FMHA_BWD_DV_IN_REG_MIN_M0
#define CK_TILE_FMHA_BWD_DV_IN_REG_MIN_M0 64
#endif

// ABLATION ONLY -- PRODUCES WRONG RESULTS. Drops the D (row-sum of dO*O) TDM
// transfer from every tile, taking the per-wave issue count from 4 to 3. D is
// still read out of whatever stale LDS the slot happens to hold, so the numbers
// are garbage; this exists purely to put an upper bound on what removing one of
// the four transfers is worth before building the real thing.
//
// The real change is not a deletion: aiter covers all four streams with three
// transfers per wave by specialising the scalar load across waves (waves 0/1
// carry LSE, waves 2/3 carry D -- see the s_bfe of ttmp8 feeding the
// s_cmp_gt_i32 s2, 1 that selects ptr_lse vs ptr_d in the disassembly). This
// ablation measures the ceiling of that idea, not the idea itself.
#ifndef CK_TILE_FMHA_BWD_ABLATE_DROP_D
#define CK_TILE_FMHA_BWD_ABLATE_DROP_D 0
#endif

// ABLATION ONLY -- PRODUCES WRONG RESULTS FOR CAUSAL. Forces the per-pixel mask
// check off, so the edge-tile `set_tile_if` never runs and the compiler can drop
// the masked path out of the body entirely.
//
// The per-pixel mask is already gated on mask.IsEdgeTile(), so a correct tile
// specialisation (report item P3) could at best make the non-edge body look like
// this one: no predicate evaluation, and no masked code competing for registers
// or schedule slots. This measures that ceiling. It is not itself a candidate
// implementation -- it simply computes the wrong answer on the diagonal.
#ifndef CK_TILE_FMHA_BWD_ABLATE_NO_MASK
#define CK_TILE_FMHA_BWD_ABLATE_NO_MASK 0
#endif

// ABLATION ONLY. The opposite probe: run the per-pixel mask on *every* tile
// instead of just the diagonal ones. Results stay correct -- masking a fully
// valid tile is a no-op -- so this one can be validated.
//
// Together with the control and ABLATE_NO_MASK this separates the three things
// the mask costs: (always - control) is the marginal cost of executing
// set_tile_if on a tile; the control already pays that on the ~6% of tiles that
// are edge tiles; whatever of (control - no_mask) remains is the price of merely
// having the masked code sitting in the body. Only that last part is
// recoverable by a correct tile specialisation.
#ifndef CK_TILE_FMHA_BWD_ABLATE_ALL_MASK
#define CK_TILE_FMHA_BWD_ABLATE_ALL_MASK 0
#endif

// Split the Q-tile loop into an edge-tile prefix and a mask-free remainder,
// instead of asking mask.IsEdgeTile() on every tile of a single shared body.
//
// For a non-local mask the edge tiles really are a prefix: IsEdgeTile reduces to
// (k_origin + kN0) > min(seqlen_q_step + x, x_total), and the right-hand side
// only grows as the loop walks Q down, so the predicate flips true->false once
// and never back. Local/band masks can have edge tiles at both ends, so they
// keep the old single-loop form.
//
// The point is not the saved predicate -- that is a couple of scalar ops. It is
// that the mask-free instantiation of the body no longer carries the
// set_tile_if block, so it stops competing for registers and schedule slots on
// the ~94% of tiles that never needed it.
// MEASURED A LOSS -- default off. Emitting the body twice costs far more than
// the mask code it removes: mask1 0.446 -> 0.712 ms (59% worse), and even mask0,
// whose predicate is loop-invariant so only the mask-free loop ever runs, lost
// 8% (0.699 -> 0.756 ms). Both arms validate. Kept because the measurement is
// the useful part: this kernel is bound by code size / register lifetime far
// more tightly than by the ~5.5% the mask work itself is worth.
#ifndef CK_TILE_FMHA_BWD_SPLIT_EDGE_TILES
#define CK_TILE_FMHA_BWD_SPLIT_EDGE_TILES 0
#endif

// Keep one body, but stop asking the mask which tiles are edge tiles. The edge
// tiles form a leading run and a trailing run (IsEdgeTile is
// top_right_edge || bottom_left_edge; the first only goes true->false down the Q
// loop, the second only false->true), so both run lengths can be found once per
// workgroup with two short scalar scans, and the per-tile question becomes two
// integer compares on the loop counter.
//
// Unlike SPLIT_EDGE_TILES this emits no extra copy of the body, so it isolates
// what the predicate evaluation itself costs, with none of the code-size
// penalty that sank the split.
#ifndef CK_TILE_FMHA_BWD_HOIST_EDGE_TEST
#define CK_TILE_FMHA_BWD_HOIST_EDGE_TEST 0
#endif

// Skip the -inf guard on the row LSE. A fully masked-out row carries
// LSE = -inf, and feeding that into exp2(scale*s - row_lse) would give NaN, so
// the guard maps it to 0. But a row only reaches -inf if it has no valid key at
// all, which cannot happen for bottom-right causal with seqlen_q == seqlen_k --
// every query row sees at least itself. Where that holds the guard is pure
// overhead. Validate before trusting this on any other shape.
#ifndef CK_TILE_FMHA_BWD_ABLATE_NO_LSE_VALIDATE
#define CK_TILE_FMHA_BWD_ABLATE_NO_LSE_VALIDATE 0
#endif

// PROBE: drop the remaining hand-written GemmStagedScheduler prescriptions
// (<0>, <3>, <4>) and/or the sched_barrier that follows each, the same way A2
// did for <1>/<2>.  Bit N of the mask selects scheduler N.

// PROBE: fence the softmax/dropout stage from the operand prefetch that follows
// it. The hot-loop VGPR peak is a ~1000-line band where the scheduler has
// hoisted the next gemm's ds_load_tr fragments into the dropout stage; this
// measures what that overlap costs in registers.

// A2: merge the gemm_1 / gemm_2 scheduling regions -- drop the sched_barrier
// between them AND both hand-written GemmStagedScheduler prescriptions.
// Both halves are required; doing either alone is a loss (see HANDOFF_A2).

// PROBE: keep ONLY dV in registers, leave dK in LDS. Halves the extra register
// cost versus the fully register-resident pipeline (128 VGPR instead of 256).
// PROBE: sink the next iteration's Q/LSE loads past gemm_4, so their live
// ranges do not span it.  dO/D are already loaded after gemm_4; Q/LSE were the
// odd ones out.  Safe: gemm_4 touches neither, and nothing between gemm_4 and
// the dO/D loads writes LDS or barriers, so they sit in the same validity
// window the dO/D loads already rely on.

// Same algorithm as BlockFmhaBwdDQDKDVPipelineKRKTRVRIGLP, with the dK and dV
// accumulators held in LDS instead of registers.
//
// The two fp32 accumulators are kN0*headdim floats each. Kept in registers they
// are live across the whole Q loop, which on gfx1250 (wave32) costs
// kN0*headdim/kBlockSize VGPRs apiece -- 64 each at kN0=64, headdim=128. That is
// what takes the kernel from 2 waves/SIMD to 1: measured 377 VGPRs / occupancy 2
// at headdim 64 (226 TFLOPS) versus 597 VGPRs / occupancy 1 at headdim 128
// (120 TFLOPS), against a forward pipeline reaching 369 TFLOPS on the same part.
//
// Here the running sums live in LDS, which is otherwise 89% idle (36 KiB of the
// 320 KiB a gfx1250 workgroup may take), and each accumulation becomes
// load -> gemm -> store so the register tile is live only around its own gemm.
// The trade is extra LDS traffic once per Q tile; whether that pays has to be
// measured on hardware.
//
// No LDS atomics are needed: gemm_1/gemm_3 distribute C with MWarp warps
// splitting M (=kN0) disjointly and NWarp=1, so every LDS element has exactly
// one owning thread and a plain read-modify-write is race free.
template <typename Problem, typename Policy = BlockFmhaBwdPipelineLdsAccPolicy>
struct BlockFmhaBwdDQDKDVPipelineLdsAccKRKTRVR
{
    using QDataType             = remove_cvref_t<typename Problem::QDataType>;
    using KDataType             = remove_cvref_t<typename Problem::KDataType>;
    using VDataType             = remove_cvref_t<typename Problem::VDataType>;
    using GemmDataType          = remove_cvref_t<typename Problem::GemmDataType>;
    using BiasDataType          = remove_cvref_t<typename Problem::BiasDataType>;
    using LSEDataType           = remove_cvref_t<typename Problem::LSEDataType>;
    using AccDataType           = remove_cvref_t<typename Problem::AccDataType>;
    using DDataType             = remove_cvref_t<typename Problem::DDataType>;
    using RandValOutputDataType = remove_cvref_t<typename Problem::RandValOutputDataType>;
    using ODataType             = remove_cvref_t<typename Problem::ODataType>;
    using OGradDataType         = remove_cvref_t<typename Problem::OGradDataType>;
    using QGradDataType         = remove_cvref_t<typename Problem::QGradDataType>;
    using KGradDataType         = remove_cvref_t<typename Problem::KGradDataType>;
    using VGradDataType         = remove_cvref_t<typename Problem::VGradDataType>;
    using BiasGradDataType      = remove_cvref_t<typename Problem::BiasGradDataType>;
    using FmhaMask              = remove_cvref_t<typename Problem::FmhaMask>;
    using FmhaDropout           = remove_cvref_t<typename Problem::FmhaDropout>;
    using HotLoopScheduler      = typename Policy::template HotLoopScheduler<Problem>;

    using BlockFmhaShape = remove_cvref_t<typename Problem::BlockFmhaShape>;

    static constexpr index_t kBlockPerCu = Problem::kBlockPerCu;
    static constexpr index_t kBlockSize  = Problem::kBlockSize;

    static constexpr index_t kM0 = BlockFmhaShape::kM0;
    // Resolved Q/dO ring depth, exposed so the kernel can keep the dQ static
    // stride paired with it -- the fold only pays alongside the deep ring.
    static constexpr index_t kQDOSlotsResolved = Policy::template GetQDOSlots<Problem>();

    // Holding dV in registers, and evicting V to pay for it, only wins at
    // kM0 = 64. The dV accumulator is kN0 x headdim and does not shrink with
    // kM0, so at kM0 = 32 the same work does twice the dV LDS round trips --
    // measured -8.2% on d=256 causal, whose tile is kM0=32.
    static constexpr bool kDVInReg =
        CK_TILE_FMHA_BWD_DV_IN_REG && (kM0 >= CK_TILE_FMHA_BWD_DV_IN_REG_MIN_M0);
    static constexpr bool kVNonResident =
        CK_TILE_FMHA_BWD_V_NONRESIDENT && (kM0 >= CK_TILE_FMHA_BWD_V_NONRESIDENT_MIN_M0) &&
        (BlockFmhaShape::kVHeaddim >= CK_TILE_FMHA_BWD_V_NONRESIDENT_MIN_HDIM ||
         (!FmhaMask::IsMasking &&
          BlockFmhaShape::kVHeaddim >= CK_TILE_FMHA_BWD_V_NONRESIDENT_MIN_HDIM_NOMASK));
    static constexpr index_t kN0        = BlockFmhaShape::kN0;
    static constexpr index_t kK0        = BlockFmhaShape::kK0;
    static constexpr index_t kK1        = BlockFmhaShape::kK1;
    static constexpr index_t kK2        = BlockFmhaShape::kK2;
    static constexpr index_t kK3        = BlockFmhaShape::kK3;
    static constexpr index_t kK4        = BlockFmhaShape::kK4;
    static constexpr index_t kQKHeaddim = BlockFmhaShape::kQKHeaddim;
    static constexpr index_t kVHeaddim  = BlockFmhaShape::kVHeaddim;

    static constexpr bool kIsGroupMode     = Problem::kIsGroupMode;
    static constexpr index_t kPadHeadDimQ  = Problem::kPadHeadDimQ;
    static constexpr index_t kPadHeadDimV  = Problem::kPadHeadDimV;
    static constexpr auto BiasEnum         = Problem::BiasEnum;
    static constexpr bool kHasBiasGrad     = Problem::kHasBiasGrad;
    static constexpr bool kIsDeterministic = Problem::kIsDeterministic;
    static constexpr bool kUseTrLoad       = Problem::kUseTrLoad;

    // Mirror tile pairing inlines two bodies into the function, and the machine
    // scheduler's clustering heuristics go the wrong way in a doubled region:
    // dropping the hand-written prescriptions is +14.9..+28.2% wherever the
    // body holds one Q tile and -1.4..-3.2% where it holds two. So the
    // discriminator is whether pairing fires, not whether the instance is
    // masked -- group mode never pairs, and its causal instances measure
    // +15.65% with the drop.
    //
    // Mirrors kMaskTilePairing in fmha_bwd_kernel.hpp. Two of its terms
    // simplify here: kUseQrQtrDorPipeline is false by construction in this
    // pipeline, and with it false kUsePersistent reduces to kIsDeterministic.
    static constexpr bool kBodyIsPaired = CK_TILE_FMHA_BWD_MASK_TILE_PAIRING &&
                                          FmhaMask::IsMasking && !kIsGroupMode && !kIsDeterministic;
    static constexpr bool kDropStagedSched =
        (CK_TILE_FMHA_BWD_SCHED_DROP_MODE == 2) ||
        (CK_TILE_FMHA_BWD_SCHED_DROP_MODE == 1 && !kBodyIsPaired);
    static_assert(!kUseTrLoad, "This pipeline does not use trload!");

    // last dimension vector length used to create tensor view(and decide buffer_load vector length)
    // ... together with tensor distribution. tensor dist should able to overwrite this
    static constexpr index_t kAlignmentQ =
        kPadHeadDimQ ? kPadHeadDimQ : Policy::template GetAlignmentQ<Problem>();
    static constexpr index_t kAlignmentK =
        kPadHeadDimQ ? kPadHeadDimQ : Policy::template GetAlignmentK<Problem>();
    static constexpr index_t kAlignmentV =
        kPadHeadDimV ? kPadHeadDimV : Policy::template GetAlignmentV<Problem>();
    static constexpr index_t kAlignmentOGrad =
        kPadHeadDimV ? kPadHeadDimV : Policy::template GetAlignmentOGrad<Problem>();
    static constexpr index_t kAlignmentQGrad = 1;
    static constexpr index_t kAlignmentKGrad =
        kPadHeadDimQ ? kPadHeadDimQ : Policy::template GetAlignmentKGrad<Problem>();
    static constexpr index_t kAlignmentVGrad =
        kPadHeadDimV ? kPadHeadDimV : Policy::template GetAlignmentVGrad<Problem>();
    static constexpr index_t kAlignmentBias = 1;

    static constexpr const char* name = "ldsacc_kr_ktr_vr";

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

#if CK_TILE_FMHA_BWD_DQ_ATOMIC_FOLD
    // Fold the gemm_4 accumulator so one dQ atomic covers one whole cache line.
    //
    // The wmma C fragment hands a wave two rows 8 apart, 16 columns each, so a
    // buffer_atomic_add_f32 built from it straddles two 128 B lines. With
    // PackMNIter on, the two registers of an N-adjacent pair hold the same two
    // rows over adjacent 16-column blocks:
    //
    //   A : lanes 0-15 = row r   cols c..c+15  | lanes 16-31 = row r+8 cols c..c+15
    //   B : lanes 0-15 = row r   cols c+16..   | lanes 16-31 = row r+8 cols c+16..
    //
    // v_permlane16_swap_b32 exchanges A's odd 16-lane row with B's even one,
    // which is precisely the transpose of that 2x2 block:
    //
    //   A': lanes 0-31 = row r   cols c..c+31   -> 128 B, one line
    //   B': lanes 0-31 = row r+8 cols c..c+31   -> 128 B, one line
    //
    // The thread-buffer index of every value is unchanged -- only which (M, N)
    // it stands for -- so this is a relabelling plus one VALU op per register
    // pair, and MakeQGradStoreBlockDistribution is the new label.
    template <typename QGradAccTensor>
    CK_TILE_DEVICE static auto FoldQGradForAtomic(const QGradAccTensor& dq_acc)
    {
        auto dq_out = make_static_distributed_tensor<AccDataType>(
            Policy::template MakeQGradStoreBlockDistribution<Problem>());

        static_assert(QGradAccTensor::get_thread_buffer_size() ==
                          decltype(dq_out)::get_thread_buffer_size(),
                      "the fold must not change how many values a lane holds");

        constexpr index_t kWarpSize = get_warp_size();
        constexpr index_t kMPerLane = BlockFmhaShape::Gemm4WarpTile::at(number<0>{}) *
                                      BlockFmhaShape::Gemm4WarpTile::at(number<1>{}) / kWarpSize;
        // One pair-group is two consecutive N iterations, i.e. two runs of
        // kMPerLane values in the thread buffer.
        constexpr index_t kPairGroups = QGradAccTensor::get_thread_buffer_size() / (2 * kMPerLane);
        static_assert(kPairGroups * 2 * kMPerLane == QGradAccTensor::get_thread_buffer_size(),
                      "accumulator does not split into N-adjacent pairs");

        const auto& src = dq_acc.get_thread_buffer();
        auto& dst       = dq_out.get_thread_buffer();

        static_for<0, kPairGroups, 1>{}([&](auto i_pair) {
            static_for<0, kMPerLane, 1>{}([&](auto i_e) {
                constexpr auto i_lo = number<i_pair * 2 * kMPerLane + i_e>{};
                constexpr auto i_hi = number<i_pair * 2 * kMPerLane + kMPerLane + i_e>{};

                const int32x2_t s = __builtin_amdgcn_permlane16_swap(
                    bit_cast<int32_t>(src[i_lo]), bit_cast<int32_t>(src[i_hi]), false, false);

                dst(i_lo) = bit_cast<AccDataType>(s[0]);
                dst(i_hi) = bit_cast<AccDataType>(s[1]);
            });
        });

        return dq_out;
    }
#endif

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename RandValDramBlockWindowTmp,
              typename OGradDramBlockWindowTmp,
              typename LSEDramBlockWindowTmp,
              typename DDramBlockWindowTmp,
              typename QGradDramBlockWindowTmp,
              typename BiasGradDramBlockWindowTmp,
              typename PositionEncoding>
    CK_TILE_HOST_DEVICE auto
    operator()(void* smem_ptr,
               const QDramBlockWindowTmp& q_dram_block_window_tmp,
               const KDramBlockWindowTmp& k_dram_block_window_tmp,
               const VDramBlockWindowTmp& v_dram_block_window_tmp,
               const BiasDramBlockWindowTmp& bias_dram_block_window_tmp,
               const RandValDramBlockWindowTmp& randval_dram_block_window_tmp,
               const OGradDramBlockWindowTmp& do_dram_block_window_tmp,
               const LSEDramBlockWindowTmp& lse_dram_block_window_tmp,
               const DDramBlockWindowTmp& d_dram_block_window_tmp,
               const QGradDramBlockWindowTmp& dq_dram_block_window_tmp,
               const BiasGradDramBlockWindowTmp& dbias_dram_block_window_tmp,
               FmhaMask mask,
               PositionEncoding position_encoding,
               float raw_scale,
               float scale,
               float rp_undrop,
               float scale_rp_undrop,
               FmhaDropout& dropout) const
    {
        static_assert(
            std::is_same_v<QDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                std::is_same_v<KDataType, remove_cvref_t<typename KDramBlockWindowTmp::DataType>> &&
                std::is_same_v<VDataType, remove_cvref_t<typename VDramBlockWindowTmp::DataType>> &&
                std::is_same_v<OGradDataType,
                               remove_cvref_t<typename OGradDramBlockWindowTmp::DataType>> &&
                std::is_same_v<LSEDataType,
                               remove_cvref_t<typename LSEDramBlockWindowTmp::DataType>> &&
                std::is_same_v<DDataType, remove_cvref_t<typename DDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kM0 == QDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == KDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == VDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kM0 == BiasDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == BiasDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
                          kM0 == OGradDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kM0 == LSEDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kM0 == DDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kM0 == QGradDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kM0 == BiasGradDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == BiasGradDramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
                      "wrong!");

        // Block GEMM
        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1 = Policy::template GetPTOGradTBlockGemm<Problem>();
        constexpr auto gemm_2 = Policy::template GetOGradVBlockGemm<Problem>();
        constexpr auto gemm_3 = Policy::template GetSGradTQTBlockGemm<Problem>();
        constexpr auto gemm_4 = Policy::template GetSGradKTBlockGemm<Problem>();

        // VGrad & KGrad accumulators, in LDS.
        //
        // These sit after every staged region rather than inside the max() over
        // phases, because they are live for the whole Q loop while K/V/Q/dO/dS
        // each die at the end of their phase. Putting them last leaves all the
        // existing staged offsets untouched.
        //
        // Both windows carry the gemm's own C distribution, so the tiles loaded
        // from them can be fed straight back into gemm_1 / gemm_3 (which assert
        // on that distribution) and handed to the epilogue unchanged.
        auto dk_acc_lds = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<AccDataType*>(static_cast<char*>(smem_ptr) +
                                           Policy::template GetKGradAccSmemOffset<Problem>()),
            Policy::template MakeKGradAccLdsBlockDescriptor<Problem>());
        auto dv_acc_lds = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<AccDataType*>(static_cast<char*>(smem_ptr) +
                                           Policy::template GetVGradAccSmemOffset<Problem>()),
            Policy::template MakeVGradAccLdsBlockDescriptor<Problem>());

        [[maybe_unused]] auto dk_acc_lds_window =
            make_tile_window(dk_acc_lds,
                             make_tuple(number<kN0>{}, number<kQKHeaddim>{}),
                             {0, 0},
                             decltype(gemm_3.MakeCBlockTile())::get_tile_distribution());
        [[maybe_unused]] auto dv_acc_lds_window =
            make_tile_window(dv_acc_lds,
                             make_tuple(number<kN0>{}, number<kVHeaddim>{}),
                             {0, 0},
                             decltype(gemm_1.MakeCBlockTile())::get_tile_distribution());

        // K, HBM ->LDS ->Reg
        auto k_dram_window =
            make_tile_window(k_dram_block_window_tmp.get_bottom_tensor_view(),
                             k_dram_block_window_tmp.get_window_lengths(),
                             k_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeKDramTileDistribution<Problem>());

        const auto k_origin = k_dram_window.get_window_origin();
        // Early termination
        const auto [seqlen_q_start, seqlen_q_end] =
            mask.GetTileRangeAlongY(k_origin.at(number<0>{}), number<kM0>{}, number<kN0>{});

        const auto num_total_loop =
            amd_wave_read_first_lane(integer_divide_ceil(seqlen_q_end - seqlen_q_start, kM0));

        // Leading and trailing edge-tile run lengths, for HOIST_EDGE_TEST. Both
        // scans stop at the first tile that is not an edge tile, so they are a
        // couple of scalar iterations in the shapes that matter.
#if CK_TILE_FMHA_BWD_HOIST_EDGE_TEST
        index_t n_edge_head       = 0;
        index_t n_edge_tail_start = num_total_loop;
        {
            auto tile_is_edge = [&](index_t i) {
                return mask.IsEdgeTile(seqlen_q_start + i * kM0,
                                       k_origin.at(number<0>{}),
                                       number<kM0>{},
                                       number<kN0>{});
            };
            while(n_edge_head < num_total_loop && tile_is_edge(n_edge_head))
            {
                n_edge_head += 1;
            }
            while(n_edge_tail_start > n_edge_head && tile_is_edge(n_edge_tail_start - 1))
            {
                n_edge_tail_start -= 1;
            }
        }
#endif

        // check early exit if no work to do.
        // __builtin_expect is load-bearing: omitting it causes incorrect AGPR allocation in
        // the dK/dV accumulation loop on some compiler versions, leading to wrong results.
        if(__builtin_expect(num_total_loop <= 0, 0))
        {
            // Nothing was accumulated, so hand back zeroed register tiles rather
            // than reading the LDS accumulators (which were never initialised).
            auto dk_zero = decltype(gemm_3.MakeCBlockTile()){};
            auto dv_zero = decltype(gemm_1.MakeCBlockTile()){};
            clear_tile(dk_zero);
            clear_tile(dv_zero);
            return make_tuple(dk_zero, dv_zero);
        }
        KDataType* k_lds_ptr =
            static_cast<KDataType*>(static_cast<void*>(static_cast<char*>(smem_ptr)));
        auto k_lds = make_tensor_view<address_space_enum::lds>(
            k_lds_ptr, Policy::template MakeKLdsWriteBlockDescriptor<Problem>());

        auto k_lds_write_window =
            make_tile_window(k_lds, make_tuple(number<kN0>{}, number<kQKHeaddim>{}), {0, 0});

        auto k_lds_read_window =
            make_tile_window(k_lds_write_window.get_bottom_tensor_view(),
                             make_tuple(number<kN0>{}, number<kQKHeaddim>{}),
                             k_lds_write_window.get_window_origin(),
                             Policy::template MakeKRegBlockDescriptor<Problem>());

        auto k_reg_tensor = make_static_distributed_tensor<KDataType>(
            Policy::template MakeKRegBlockDescriptor<Problem>());

        //------------------------------------------------------------------
        // V, HBM ->LDS ->Reg
        auto v_dram_window =
            make_tile_window(v_dram_block_window_tmp.get_bottom_tensor_view(),
                             v_dram_block_window_tmp.get_window_lengths(),
                             v_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeVDramTileDistribution<Problem>());

        // V has a dedicated region past the staged/accumulator blocks -- see
        // GetVSmemOffset. It used to alias K/KT at offset 0.
        VDataType* v_lds_ptr = static_cast<VDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetVSmemOffset<Problem>()));

        auto v_lds = make_tensor_view<address_space_enum::lds>(
            v_lds_ptr, Policy::template MakeVLdsWriteBlockDescriptor<Problem>());

        auto v_lds_write_window =
            make_tile_window(v_lds, make_tuple(number<kN0>{}, number<kVHeaddim>{}), {0, 0});

        auto v_lds_read_window =
            make_tile_window(v_lds_write_window.get_bottom_tensor_view(),
                             make_tuple(number<kN0>{}, number<kVHeaddim>{}),
                             v_lds_write_window.get_window_origin(),
                             Policy::template MakeVRegBlockDescriptor<Problem>());

        //------------------------------------------------------------------
        // KT, read transposed straight out of the single K box
        //
        // There used to be a second LDS copy of K here, produced by shuffling
        // k_block_tile in registers, purely so gemm_4 could read K^T with a
        // plain load_tile. ds_load_tr16_b128 does that in hardware, so the
        // shuffle, the staging tile and the copy are all gone; the window below
        // points at the same box k_lds_write_window fills.
        auto kt_lds_read_window =
            make_tile_window(k_lds_write_window.get_bottom_tensor_view(),
                             make_tuple(number<kN0>{}, number<kQKHeaddim>{}),
                             k_lds_write_window.get_window_origin(),
                             Policy::template MakeKTRegBlockDescriptor<Problem>());

        // V is moved global->LDS by TDM, so it never lands in registers. pad is
        // disabled by the policy; workgroup_mask stays 0 (no cluster multicast).
        TDMConfig tdm_config_v;
        TDMConfig tdm_config_k;
        TDMConfig tdm_config_q;
        TDMConfig tdm_config_do;
        TDMConfig tdm_config_lse;
        TDMConfig tdm_config_d;
        {
            constexpr auto LdsPaddingConfigV     = Policy::template GetLdsPaddingConfigV<Problem>();
            tdm_config_v.pad_enable              = LdsPaddingConfigV[number<0>{}];
            tdm_config_v.pad_config.pad_amount   = LdsPaddingConfigV[number<1>{}];
            tdm_config_v.pad_config.pad_interval = LdsPaddingConfigV[number<2>{}];

            constexpr auto LdsPaddingConfigK     = Policy::template GetLdsPaddingConfigK<Problem>();
            tdm_config_k.pad_enable              = LdsPaddingConfigK[number<0>{}];
            tdm_config_k.pad_config.pad_amount   = LdsPaddingConfigK[number<1>{}];
            tdm_config_k.pad_config.pad_interval = LdsPaddingConfigK[number<2>{}];

            constexpr auto LdsPaddingConfigQ     = Policy::template GetLdsPaddingConfigQ<Problem>();
            tdm_config_q.pad_enable              = LdsPaddingConfigQ[number<0>{}];
            tdm_config_q.pad_config.pad_amount   = LdsPaddingConfigQ[number<1>{}];
            tdm_config_q.pad_config.pad_interval = LdsPaddingConfigQ[number<2>{}];

            constexpr auto LdsPaddingConfigDO =
                Policy::template GetLdsPaddingConfigOGrad<Problem>();
            tdm_config_do.pad_enable              = LdsPaddingConfigDO[number<0>{}];
            tdm_config_do.pad_config.pad_amount   = LdsPaddingConfigDO[number<1>{}];
            tdm_config_do.pad_config.pad_interval = LdsPaddingConfigDO[number<2>{}];

            // LSE/D are rank 1: a single contiguous run with no rows to
            // pad between, so the padding machinery stays off.
            tdm_config_lse.pad_enable = false;
            tdm_config_d.pad_enable   = false;
        }

        //------------------------------------------------------------------
        // Pre-Load KV: both go global->LDS by TDM, so neither lands in registers
        // on the way. K is issued first and V second, and TENSORcnt retires in
        // issue order, so waiting for "at most one outstanding" below releases K
        // while V is still in flight -- V keeps the overlap it needs, and K does
        // not have to wait behind it.
        load_tile_tdm(tdm_config_k, k_lds_write_window, k_dram_window);
        load_tile_tdm(tdm_config_v, v_lds_write_window, v_dram_window);

        // K only: V may still be transferring.
        s_wait_tensorcnt_barrier<1>();
        k_reg_tensor = load_tile(k_lds_read_window);
        block_sync_lds();

        auto kt_reg_tensor = load_tile_transpose(kt_lds_read_window);

        // Now V as well. TDM commits on TENSORcnt, not on the LDS or
        // vector-memory counters, so block_sync_lds alone would not fence it.
        s_wait_tensorcnt_barrier<0>();

        // Unconditional: on the non-resident path every use below overwrites
        // this before reading it, so it is dead and the compiler drops both the
        // load and the live range.
        auto v_reg_tensor = load_tile(v_lds_read_window);
        //---------------------------- Loop Load in ----------------------------//
        // Q: HBM ->Reg ->LDS
        auto q_dram_window =
            make_tile_window(q_dram_block_window_tmp.get_bottom_tensor_view(),
                             q_dram_block_window_tmp.get_window_lengths(),
                             {seqlen_q_start, 0},
                             Policy::template MakeQDramTileDistribution<Problem>());

        // ---- the Q/dO/LSE/D slot ring -----------------------------------
        //
        // kQDOSlots tiles are resident in LDS at once. Slot 0 is the box the
        // staged region already held; slots 1.. are appended past V, slot 1
        // landing exactly where the old second Q/dO pair sat -- so kQDOSlots==2
        // is the previous layout byte for byte.
        //
        // Slot 0's four boxes are laid out dO, Q, LSE, D (that is the order the
        // staged offsets were built in); the appended slots use Q, dO, LSE, D.
        // Nothing depends on the order, only on the four offsets, so each box
        // gets its own accessor rather than a single base plus a stride.
        constexpr index_t kQDOSlots = Policy::template GetQDOSlots<Problem>();
        static_assert(kQDOSlots >= 1 && kQDOSlots <= 4,
                      "the hot loop is unrolled by kQDOSlots; keep it small");

        auto q_slot_off = [&](auto j) -> index_t {
            constexpr index_t jj = j;
            if constexpr(jj == 0)
            {
                return Policy::template GetSmemSizeQT<Problem>() +
                       Policy::template GetSmemSizeOGrad<Problem>() +
                       Policy::template GetSmemSizeOGradT<Problem>();
            }
            else
            {
                return Policy::template GetQDOSlotBase<Problem>(jj);
            }
        };
        auto do_slot_off = [&](auto j) -> index_t {
            constexpr index_t jj = j;
            if constexpr(jj == 0)
            {
                return Policy::template GetSmemSizeQT<Problem>();
            }
            else
            {
                return Policy::template GetQDOSlotBase<Problem>(jj) +
                       Policy::template GetSmemSizeQ<Problem>();
            }
        };
        auto lse_slot_off = [&](auto j) -> index_t {
            constexpr index_t jj = j;
            if constexpr(jj == 0)
            {
                return Policy::template GetSmemSizeQT<Problem>() +
                       Policy::template GetSmemSizeOGrad<Problem>() +
                       Policy::template GetSmemSizeOGradT<Problem>() +
                       Policy::template GetSmemSizeQ<Problem>();
            }
            else
            {
                return Policy::template GetQDOSlotBase<Problem>(jj) +
                       Policy::template GetSmemSizeQ<Problem>() +
                       Policy::template GetSmemSizeOGrad<Problem>();
            }
        };
        auto d_slot_off = [&](auto j) -> index_t {
            constexpr index_t jj = j;
            if constexpr(jj == 0)
            {
                return Policy::template GetSmemSizeQT<Problem>() +
                       Policy::template GetSmemSizeOGrad<Problem>() +
                       Policy::template GetSmemSizeOGradT<Problem>() +
                       Policy::template GetSmemSizeQ<Problem>() +
                       Policy::template GetSmemSizeLSE<Problem>();
            }
            else
            {
                return Policy::template GetQDOSlotBase<Problem>(jj) +
                       Policy::template GetSmemSizeQ<Problem>() +
                       Policy::template GetSmemSizeOGrad<Problem>() +
                       Policy::template GetSmemSizeLSE<Problem>();
            }
        };

        auto q_lds_windows = generate_tuple(
            [&](auto j) {
                auto tv = make_tensor_view<address_space_enum::lds>(
                    static_cast<QDataType*>(
                        static_cast<void*>(static_cast<char*>(smem_ptr) + q_slot_off(j))),
                    Policy::template MakeQLdsBlockDescriptor<Problem>());
                return make_tile_window(
                    tv, make_tuple(number<kM0>{}, number<kQKHeaddim>{}), {0, 0});
            },
            number<kQDOSlots>{});

        auto q_lds_read_windows = generate_tuple(
            [&](auto j) {
                return make_tile_window(q_lds_windows.at(j).get_bottom_tensor_view(),
                                        make_tuple(number<kM0>{}, number<kK0>{}),
                                        q_lds_windows.at(j).get_window_origin(),
                                        Policy::template MakeQRegSliceBlockDescriptor<Problem>());
            },
            number<kQDOSlots>{});

        auto pt_reg_tensor = make_static_distributed_tensor<GemmDataType>(
            Policy::template MakePTRegSliceBlockDescriptor<Problem>());
        // Q^T: read transposed out of the single Q box. The shuffle and the
        // second LDS copy it fed are gone -- ds_load_tr16_b128 does the
        // transpose in hardware. Q's shuffle ran once per Q-loop iteration, so
        // this removes hot-loop work, unlike K's which was once per block.
        auto qt_lds_read_windows = generate_tuple(
            [&](auto j) {
                return make_tile_window(q_lds_windows.at(j).get_bottom_tensor_view(),
                                        make_tuple(number<kM0>{}, number<kQKHeaddim>{}),
                                        q_lds_windows.at(j).get_window_origin(),
                                        Policy::template MakeQTRegSliceBlockDescriptor<Problem>());
            },
            number<kQDOSlots>{});

        // dO: HBM ->Reg ->LDS
        auto do_dram_window =
            make_tile_window(do_dram_block_window_tmp.get_bottom_tensor_view(),
                             do_dram_block_window_tmp.get_window_lengths(),
                             {seqlen_q_start, 0},
                             Policy::template MakeOGradDramTileDistribution<Problem>());

        auto do_lds_windows = generate_tuple(
            [&](auto j) {
                auto tv = make_tensor_view<address_space_enum::lds>(
                    static_cast<OGradDataType*>(
                        static_cast<void*>(static_cast<char*>(smem_ptr) + do_slot_off(j))),
                    Policy::template MakeOGradLdsBlockDescriptor<Problem>());
                return make_tile_window(tv, make_tuple(number<kM0>{}, number<kVHeaddim>{}), {0, 0});
            },
            number<kQDOSlots>{});

        auto do_lds_read_windows = generate_tuple(
            [&](auto j) {
                return make_tile_window(
                    do_lds_windows.at(j).get_bottom_tensor_view(),
                    make_tuple(number<kM0>{}, number<kK2>{}),
                    do_lds_windows.at(j).get_window_origin(),
                    Policy::template MakeOGradRegSliceBlockDescriptor<Problem>());
            },
            number<kQDOSlots>{});
        // dO^T: read transposed straight out of the single dO box.
        //
        // There used to be a second LDS copy here, produced by shuffling
        // do_block_tile in registers, so gemm_1 could read dO^T with a plain
        // load_tile. ds_load_tr16_b128 does that in hardware. Unlike K, dO is
        // reloaded every Q iteration, so this shuffle was in the hot loop.
        auto dot_lds_read_windows = generate_tuple(
            [&](auto j) {
                return make_tile_window(
                    do_lds_windows.at(j).get_bottom_tensor_view(),
                    make_tuple(number<kM0>{}, number<kVHeaddim>{}),
                    do_lds_windows.at(j).get_window_origin(),
                    Policy::template MakeOGradTRegSliceBlockDescriptor<Problem>());
            },
            number<kQDOSlots>{});

        // dS: Reg -> Reg -> LDS
        GemmDataType* ds_lds_ptr = static_cast<GemmDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeQT<Problem>() +
            Policy::template GetSmemSizeOGrad<Problem>() +
            Policy::template GetSmemSizeOGradT<Problem>() +
            Policy::template GetSmemSizeQ<Problem>() + Policy::template GetSmemSizeLSE<Problem>() +
            Policy::template GetSmemSizeD<Problem>()));

        auto ds_lds = make_tensor_view<address_space_enum::lds>(
            ds_lds_ptr, Policy::template MakeSGradLdsBlockDescriptor<Problem>());

        // Box is [kN0][kM0]; the gemm_4 slice is kK4 rows of it, and the window
        // therefore walks dim 0 instead of dim 1.
        auto ds_lds_window =
            make_tile_window(ds_lds, make_tuple(number<kN0>{}, number<kM0>{}), {0, 0});

        auto ds_lds_read_window =
            make_tile_window(ds_lds_window.get_bottom_tensor_view(),
                             make_tuple(number<kN0>{}, number<kM0>{}),
                             ds_lds_window.get_window_origin(),
                             Policy::template MakeSGradRegSliceBlockDescriptor<Problem>());

        auto dst_reg_tensor = make_static_distributed_tensor<GemmDataType>(
            Policy::template MakeSGradTRegSliceBlockDescriptor<Problem>());
        // Bias: HBM ->Reg ->Reg ->LDS
        const auto bias_origin = bias_dram_block_window_tmp.get_window_origin();

        auto bias_dram_window =
            make_tile_window(bias_dram_block_window_tmp.get_bottom_tensor_view(),
                             bias_dram_block_window_tmp.get_window_lengths(),
                             {seqlen_q_start, bias_origin.at(number<1>{})},
                             Policy::template MakeBiasTileDistribution<Problem>());

        BiasDataType* bias_lds_ptr = static_cast<BiasDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeQT<Problem>() +
            Policy::template GetSmemSizeOGrad<Problem>() +
            Policy::template GetSmemSizeOGradT<Problem>() +
            Policy::template GetSmemSizeQ<Problem>() + Policy::template GetSmemSizeLSE<Problem>() +
            Policy::template GetSmemSizeD<Problem>()));

        auto bias_lds = make_tensor_view<address_space_enum::lds>(
            bias_lds_ptr, Policy::template MakeBiasLdsBlockDescriptor<Problem>());

        auto bias_lds_write_window =
            make_tile_window(bias_lds, make_tuple(number<kM0>{}, number<kN0>{}), {0, 0});

        auto bias_s_lds_read_window =
            make_tile_window(bias_lds_write_window.get_bottom_tensor_view(),
                             bias_lds_write_window.get_window_lengths(),
                             bias_lds_write_window.get_window_origin(),
                             Policy::template MakeBiasSTileDistribution<decltype(gemm_0)>());

        static_assert(std::is_same_v<BiasDataType, BiasGradDataType>,
                      "BiasDataType and BiasGradDataType should be the same!");

        // LSE: HBM -> LDS ->Reg
        auto lse_dram_window =
            make_tile_window(lse_dram_block_window_tmp.get_bottom_tensor_view(),
                             lse_dram_block_window_tmp.get_window_lengths(),
                             {seqlen_q_start},
                             Policy::template MakeLSEDDramTdmDistribution<Problem>());

        auto lse_lds_views = generate_tuple(
            [&](auto j) {
                return make_tensor_view<address_space_enum::lds>(
                    static_cast<LSEDataType*>(
                        static_cast<void*>(static_cast<char*>(smem_ptr) + lse_slot_off(j))),
                    Policy::template MakeLSEDLdsWriteBlockDescriptor<Problem>());
            },
            number<kQDOSlots>{});

        auto lse_lds_write_windows = generate_tuple(
            [&](auto j) {
                return make_tile_window(lse_lds_views.at(j), make_tuple(number<kM0>{}), {0});
            },
            number<kQDOSlots>{});

        auto lse_lds_read_windows = generate_tuple(
            [&](auto j) {
                return make_tile_window(
                    lse_lds_views.at(j),
                    make_tuple(number<kM0>{}),
                    {0},
                    Policy::template MakeLSEDLdsReadBlockDescriptor<Problem, decltype(gemm_0)>());
            },
            number<kQDOSlots>{});

        // D: HBM ->Reg
        auto d_dram_window =
            make_tile_window(d_dram_block_window_tmp.get_bottom_tensor_view(),
                             d_dram_block_window_tmp.get_window_lengths(),
                             {seqlen_q_start},
                             Policy::template MakeLSEDDramTdmDistribution<Problem>());

        auto d_lds_views = generate_tuple(
            [&](auto j) {
                return make_tensor_view<address_space_enum::lds>(
                    static_cast<DDataType*>(
                        static_cast<void*>(static_cast<char*>(smem_ptr) + d_slot_off(j))),
                    Policy::template MakeLSEDLdsWriteBlockDescriptor<Problem>());
            },
            number<kQDOSlots>{});

        auto d_lds_write_windows = generate_tuple(
            [&](auto j) {
                return make_tile_window(d_lds_views.at(j), make_tuple(number<kM0>{}), {0});
            },
            number<kQDOSlots>{});
#if CK_TILE_FMHA_BWD_ABLATE_DROP_D
        // The ablation drops every use of these; keep the definition so the
        // rest of the slot plumbing is untouched between the two arms.
        (void)d_lds_write_windows;
#endif

        auto d_lds_read_windows = generate_tuple(
            [&](auto j) {
                return make_tile_window(
                    d_lds_views.at(j),
                    make_tuple(number<kM0>{}),
                    {0},
                    Policy::template MakeLSEDLdsReadBlockDescriptor<Problem, decltype(gemm_0)>());
            },
            number<kQDOSlots>{});

        // RandVal: HBM ->Reg
        auto randval_dram_window = dropout.template MakeRandvalDramWindow<decltype(gemm_0), false>(
            randval_dram_block_window_tmp, seqlen_q_start);

        // BiasGrad
        // Reg ->LDS ->Reg ->HBM
        const auto dbias_origin = dbias_dram_block_window_tmp.get_window_origin();

        auto dbias_dram_window =
            make_tile_window(dbias_dram_block_window_tmp.get_bottom_tensor_view(),
                             dbias_dram_block_window_tmp.get_window_lengths(),
                             {seqlen_q_start, dbias_origin.at(number<1>{})}); // M/N

        auto dbias_lds_read_window =
            make_tile_window(bias_lds,
                             make_tuple(number<kM0>{}, number<kN0>{}),
                             {0, 0},
                             Policy::template MakeShuffledBiasTileDistribution<Problem>());

        // ----------------------------Loop write out------------------------------//
        auto dq_dram_window = make_tile_window(dq_dram_block_window_tmp.get_bottom_tensor_view(),
                                               dq_dram_block_window_tmp.get_window_lengths(),
                                               {seqlen_q_start, 0});

        using SPBlockTileType     = decltype(gemm_0.MakeCBlockTile());
        using SPGradBlockTileType = decltype(gemm_2.MakeCBlockTile());
        using QGradBlockTileType  = decltype(gemm_4.MakeCBlockTile());

        index_t i_total_loops = 0;
        index_t seqlen_q_step = seqlen_q_start;
        static_assert(kQKHeaddim >= kK0, "kQKHeaddim should be equal or greater than kK0");
        static_assert(kM0 == kK1, "kM0 should equal to kK1");
        static_assert(kVHeaddim >= kK2, "kVHeaddim should be equal or greater than kK2");
        static_assert(kM0 == kK3, "kM0 should equal to kK3");
        constexpr index_t k4_loops = kN0 / kK4;

        /*
         * Prefetch Q, LSE, dO, D
         */
        // Q and dO go global -> LDS by TDM, so nothing is prefetched into
        // registers here. Their DRAM windows are advanced only after the
        // transfer has been issued, because TDM reads at issue time whereas the
        // old load_tile read before the advance.
        /*
         * Store prefetched data into LDS
         */
        // ---- pipeline depth ---------------------------------------------
        //
        // Iteration i holds tile i in registers, reads tile i's Q^T/dO^T out of
        // slot i % kQDOSlots, and issues tile i + kIssueAhead into slot
        // (i + kIssueAhead) % kQDOSlots.
        //
        // kIssueAhead is kQDOSlots - 1, which is what makes the write safe with
        // no new barrier: (i + kQDOSlots - 1) % kQDOSlots is slot i - 1, last
        // read in iteration i-1 before that iteration's block_sync_lds. Issuing
        // any further ahead would land on the slot this iteration is still
        // reading.
        //
        // At the wait, tiles up to i + kIssueAhead have been issued and tile
        // i + 1 must have landed, so kQDOSlots - 2 tiles may still be in flight
        // -- 4 transfers each, because CK splits LSE and D where aiter shares a
        // descriptor. kQDOSlots == 2 gives a wait of 0, i.e. the full drain this
        // loop used to do; kQDOSlots == 4 gives 8 and never drains.
        constexpr index_t kIssueAhead = kQDOSlots == 1 ? 1 : kQDOSlots - 1;
        constexpr index_t kTdmPerTile = CK_TILE_FMHA_BWD_ABLATE_DROP_D ? 3 : 4;
        constexpr index_t kTdmWaitCnt = kQDOSlots >= 2 ? kTdmPerTile * (kQDOSlots - 2) : 0;

        // Advance the DRAM windows only while a real tile remains. Past the end
        // the issue re-reads the last tile into a slot nothing will look at:
        // in bounds, warm in L2, and it keeps the outstanding count at exactly
        // kTdmPerTile * kIssueAhead so one compile-time wait is correct for
        // every iteration including the last two.
        auto advance_qdo_windows = [&](index_t next_tile) {
            if(next_tile <= num_total_loop - 1)
            {
                move_tile_window(q_dram_window, {kM0, 0});
                move_tile_window(lse_dram_window, {kM0});
                move_tile_window(do_dram_window, {kM0, 0});
                move_tile_window(d_dram_window, {kM0});
            }
        };

        // Issue tile `tile` into slot `j`. The issue order Q, LSE, dO, D is
        // fixed: TENSORcnt retires in order, so reordering these four changes
        // what a given wait count means.
        auto issue_qdo_tile = [&](auto j, index_t tile) {
            load_tile_tdm(tdm_config_q, q_lds_windows.at(j), q_dram_window);
            load_tile_tdm(tdm_config_lse, lse_lds_write_windows.at(j), lse_dram_window);
            load_tile_tdm(tdm_config_do, do_lds_windows.at(j), do_dram_window);
#if !CK_TILE_FMHA_BWD_ABLATE_DROP_D
            load_tile_tdm(tdm_config_d, d_lds_write_windows.at(j), d_dram_window);
#endif
            advance_qdo_windows(tile + 1);
        };

        block_sync_lds();
        // Fill: tiles 0 .. kIssueAhead-1, one per slot. No compute to overlap
        // them with, which is why the wait below is the only one in the kernel
        // that has to be met before any work at all can start.
        static_for<0, kIssueAhead, 1>{}([&](auto j) { issue_qdo_tile(j, j); });

        // All four operands now arrive by TDM and commit on TENSORcnt; nothing
        // in this region goes through dscnt any more. Waiting to kTdmWaitCnt
        // rather than 0 releases tile 0 while tiles 1.. keep transferring.
        s_wait_tensorcnt_barrier<kTdmWaitCnt>();
        block_sync_lds();

        /*
         * Prefetch LDS data into Reg to Asynchronous Data Movement and MFMA pipeline
         */

        auto q_reg_tensor  = load_tile(q_lds_read_windows.at(number<0>{}));
        auto lse           = load_tile(lse_lds_read_windows.at(number<0>{}));
        auto do_reg_tensor = load_tile(do_lds_read_windows.at(number<0>{}));
        auto d             = load_tile(d_lds_read_windows.at(number<0>{}));

        // Zero the LDS accumulators. No barrier: each element is written and
        // later read-modify-written by the same thread, so there is no sharing
        // to synchronise.
        auto dk_acc = decltype(gemm_3.MakeCBlockTile()){};
        clear_tile(dk_acc);
        // Declared either way so both branches below can name it; when it
        // lives in LDS nothing reads it here and it costs nothing.
        auto dv_acc = decltype(gemm_1.MakeCBlockTile()){};
        if constexpr(kDVInReg)
        {
            clear_tile(dv_acc);
        }
        // Each accumulator is zeroed in LDS only if it actually lives there.
        // (These two were previously joined under an `#if 0 / #else`, which made
        // the pair unconditional -- so a register-resident arm still paid for a
        // dead zero-store of the accumulator it had just hoisted out.)
        if constexpr(!kDVInReg)
        {
            auto dv_zero = decltype(gemm_1.MakeCBlockTile()){};
            clear_tile(dv_zero);
            store_tile(dv_acc_lds_window, dv_zero);
        }

        // Every slot's window has the same type -- they differ only in the LDS
        // base held by the tensor view -- so picking one is a select, not a
        // switch over code.
        //
        // Written as a conditional chain rather than a pointer that walks the
        // tuple: taking the address of a tuple element makes the windows
        // addressable, which stops them being scrubbed into registers and puts
        // every descriptor back in scratch. Measured 13-18% on causal, which is
        // the configuration that leans on this path.
        auto pick_slot = [&](auto& tup, index_t j) -> auto& {
            if constexpr(kQDOSlots == 1)
            {
                ignore = j;
                return tup.at(number<0>{});
            }
            else if constexpr(kQDOSlots == 2)
            {
                // Two slots is what the loop always had, and a single ternary
                // on two named windows is what it compiled to. Anything
                // cleverer here costs: a chain over three or four windows makes
                // all of them addressable and puts their descriptors back in
                // scratch, which measured -14 to -37% across the sweep.
                return j == 0 ? tup.at(number<0>{}) : tup.at(number<1>{});
            }
            else
            {
                auto* p = &tup.at(number<0>{});
                static_for<1, kQDOSlots, 1>{}([&](auto k) {
                    if(j == k)
                    {
                        p = &tup.at(k);
                    }
                });
                return *p;
            }
        };

        __builtin_amdgcn_sched_barrier(0);
        // Hot loop.
        //
        // The body takes its eight LDS windows as parameters so that the two
        // drivers below can bind them differently without a second copy of the
        // body in the source:
        //
        //   depth >= 3 -- unrolled kQDOSlots times, each copy binding a
        //     compile-time slot. This is the point of the unroll: a runtime
        //     rotation costs address arithmetic per operand, and at 1024 VGPR
        //     with spills already there is nowhere to put it (an earlier
        //     three-deep Q experiment did exactly that and went 7 -> 384
        //     spills).
        //   depth <= 2 -- one copy, slot chosen by a running index, which is
        //     what the ping-pong always did. Two slots do not need the unroll
        //     and instances that keep depth 2 should not pay for it: mirror
        //     tile pairing already doubles this body, and unrolling a doubled
        //     body costs those instances 15-26%.
        auto hot_loop_body = [&](auto kEdge,
                                 auto& q_rd_cur,
                                 auto& qt_rd_cur,
                                 auto& do_rd_cur,
                                 auto& dot_rd_cur,
                                 auto& q_rd_dst,
                                 auto& do_rd_dst,
                                 auto& lse_rd_dst,
                                 auto& d_rd_dst,
                                 auto issue_next_tile) {
            // gemm_0/gemm_2 consume the register copies loaded at the end of the
            // previous iteration, so these two selections have no direct user --
            // they exist to keep the slot mapping symmetric and readable.
            ignore = q_rd_cur;
            ignore = do_rd_cur;
            // STAGE 1, Q@K Gemm0
            auto s_acc = SPBlockTileType{};

            s_acc = gemm_0(q_reg_tensor, k_reg_tensor);

            auto dot_reg_tensor = load_tile_transpose(dot_rd_cur);

            if constexpr(!kDropStagedSched || (CK_TILE_FMHA_BWD_SCHED_KEEP_MASK & (1 << 0)))
            {
                HotLoopScheduler::template GemmStagedScheduler<0>();
                __builtin_amdgcn_sched_barrier(0);
            }
            // STAGE 2, Scale, Add bias, Mask, Softmax, Dropout
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                const auto bias_tile    = load_tile(bias_dram_window);
                auto shuffled_bias_tile = make_static_distributed_tensor<BiasDataType>(
                    Policy::template MakeShuffledBiasTileDistribution<Problem>());
                shuffle_tile(shuffled_bias_tile, bias_tile);
                // SGrad and Bias use the same address in LDS, finish loading ds on the previous
                // iteration to reuse LDS.
                block_sync_lds();
                store_tile(bias_lds_write_window, shuffled_bias_tile);
                block_sync_lds();
                auto bias_s_tile = load_tile(bias_s_lds_read_window);
                tile_elementwise_inout(
                    [&](auto& x, const auto& y) {
                        x = scale * x + log2e_v<AccDataType> * type_convert<AccDataType>(y);
                    },
                    s_acc,
                    bias_s_tile);
                move_tile_window(bias_dram_window, {kM0, 0});
                __builtin_amdgcn_sched_barrier(0);
            }
            else if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
            {
                constexpr auto s_spans = decltype(s_acc)::get_distributed_spans();
                sweep_tile_span(s_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(s_spans[number<1>{}], [&](auto idx1) {
                        const auto tile_idx = get_x_indices_from_distributed_indices(
                            s_acc.get_tile_distribution(), make_tuple(idx0, idx1));

                        const auto row = seqlen_q_step + tile_idx.at(number<0>{});
                        const auto col = k_origin.at(number<0>{}) + tile_idx.at(number<1>{});
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);

                        s_acc(i_j_idx) *= scale;
                        position_encoding.update(s_acc(i_j_idx), row, col);
                    });
                });
            }

            if constexpr(decltype(kEdge)::value)
            {
#if CK_TILE_FMHA_BWD_HOIST_EDGE_TEST
                bool need_perpixel_check =
                    (i_total_loops < n_edge_head) || (i_total_loops >= n_edge_tail_start);
#else
                bool need_perpixel_check = mask.IsEdgeTile(
                    seqlen_q_step, k_origin.at(number<0>{}), number<kM0>{}, number<kN0>{});
#endif
#if CK_TILE_FMHA_BWD_ABLATE_NO_MASK
                need_perpixel_check = false;
#elif CK_TILE_FMHA_BWD_ABLATE_ALL_MASK
                need_perpixel_check = FmhaMask::IsMasking;
#endif
                if(need_perpixel_check)
                {
                    set_tile_if(s_acc, -numeric<AccDataType>::infinity(), [&](auto tile_idx) {
                        const auto row = seqlen_q_step + tile_idx.at(number<0>{});
                        const auto col = k_origin.at(number<0>{}) + tile_idx.at(number<1>{});
                        return mask.IsOutOfBound(row, col);
                    });
                }
            }

            static const auto get_validated_lse = [](LSEDataType raw_lse) {
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                             FmhaMask::IsMasking)
                {
#if CK_TILE_FMHA_BWD_ABLATE_NO_LSE_VALIDATE
                    return raw_lse;
#else
                    // A fully masked row has raw_lse == -inf; only finiteness
                    // matters, not the value. Every s_acc in such a row is -inf,
                    // so any finite row_lse gives exp2(-inf) == 0. The sentinel
                    // must stay finite after the log2e scaling below, which
                    // rules out -FLT_MAX. One v_max replaces a compare+select.
                    return max(raw_lse, type_convert<LSEDataType>(-1e30f));
#endif
                }
                else
                {
                    return raw_lse;
                }
            };

            auto p                 = SPBlockTileType{};
            constexpr auto p_spans = decltype(p)::get_distributed_spans();
            sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                auto row_lse         = log2e_v<LSEDataType> * get_validated_lse(lse[i_idx]);

                sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);

                    if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                                 BiasEnum == BlockAttentionBiasEnum::ALIBI)
                    {
                        p(i_j_idx) = exp2(s_acc[i_j_idx] - row_lse);
                    }
                    else
                    {
                        p(i_j_idx) = exp2(scale * s_acc[i_j_idx] - row_lse);
                    }
                });
            });

            if constexpr(FmhaDropout::IsDropout)
            {
                dropout.template Run<decltype(gemm_0), RandValOutputDataType>(
                    seqlen_q_step, k_origin.at(number<0>{}), p, randval_dram_window);
            }
            const auto p_gemm = [&]() {
                if constexpr(FmhaDropout::IsDropout)
                {
                    return tile_elementwise_in(
                        [](const auto& x) { return type_convert<GemmDataType>(x > 0.f ? x : 0.f); },
                        p);
                }
                else
                {
                    return cast_tile<GemmDataType>(p);
                }
            }();

            // STAGE 3, P^T@OGrad^T Gemm1
            Policy::template PTFromGemm0CToGemm1A<Problem>(pt_reg_tensor, p_gemm);
            {
                if constexpr(kDVInReg)
                {
                    gemm_1(dv_acc, pt_reg_tensor, dot_reg_tensor);
                }
                else
                {
                    auto dv_lds = load_tile(dv_acc_lds_window);
                    gemm_1(dv_lds, pt_reg_tensor, dot_reg_tensor);
                    store_tile(dv_acc_lds_window, dv_lds);
                }
            }

            auto qt_reg_tensor = load_tile_transpose(qt_rd_cur);

            // STAGE 4, OGrad@V Gemm2
            auto dp_acc = SPGradBlockTileType{};

            if constexpr(kVNonResident)
            {
                v_reg_tensor = load_tile(v_lds_read_window);
            }
            dp_acc = gemm_2(do_reg_tensor, v_reg_tensor);

            // This barrier existed so TDM could not overwrite an LDS box that
            // some wave was still reading. With two or more slots the issue
            // below targets slot kWr, which is slot kCur - 1: last read in the
            // previous iteration, behind that iteration's block_sync_lds. Only
            // the single-slot fallback still overwrites what this iteration is
            // reading and still needs the drain here.
            if constexpr(kQDOSlots == 1)
            {
                block_sync_lds();
            }

            issue_next_tile();
#if !CK_TILE_FMHA_BWD_SINK_TDM_WAIT
            // same as the prologue: Q/dO are on TENSORcnt now
            s_wait_tensorcnt_barrier<kTdmWaitCnt>();
#endif

            __builtin_amdgcn_sched_barrier(0);
            // STAGE 5, P^T(PGrad^T - D)
            auto ds                 = SPGradBlockTileType{};
            constexpr auto ds_spans = decltype(ds)::get_distributed_spans();
            sweep_tile_span(ds_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                sweep_tile_span(ds_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    bool undrop_flag       = p[i_j_idx] >= 0;
                    ds(i_j_idx)            = p[i_j_idx] * (!FmhaDropout::IsDropout || undrop_flag
                                                               ? (dp_acc[i_j_idx] - d[i_idx])
                                                               : d[i_idx]);
                });
            });

            if constexpr(kHasBiasGrad)
            {
                const auto dbias = [&]() {
                    if constexpr(FmhaDropout::IsDropout)
                    {
                        return tile_elementwise_in(
                            [&rp_undrop](const auto& x) {
                                return type_convert<BiasGradDataType>(x * rp_undrop);
                            },
                            ds);
                    }
                    else
                    {
                        return cast_tile<BiasGradDataType>(ds);
                    }
                }();
                store_tile(bias_lds_write_window, dbias);
                block_sync_lds();
                auto shuffled_dbias_tile = load_tile(dbias_lds_read_window);
                auto dbias_tile          = make_static_distributed_tensor<BiasGradDataType>(
                    Policy::template MakeBiasTileDistribution<Problem>());
                shuffle_tile(dbias_tile, shuffled_dbias_tile);
                store_tile(dbias_dram_window, dbias_tile);
                move_tile_window(dbias_dram_window, {kM0, 0});
                __builtin_amdgcn_sched_barrier(0);
            }

            // STAGE 6, SGrad^T@Q^T Gemm3
            const auto ds_gemm = cast_tile<GemmDataType>(ds);

            Policy::template SGradTFromGemm2CToGemm3A<Problem>(dst_reg_tensor, ds_gemm);

            gemm_3(dk_acc, dst_reg_tensor, qt_reg_tensor);

            if constexpr(kHasBiasGrad)
            {
                // SGrad and BiasGrad use the same address in LDS.
                block_sync_lds();
            }
            store_tile(ds_lds_window, dst_reg_tensor);

            block_sync_lds();

#if CK_TILE_FMHA_BWD_SINK_TDM_WAIT
            // Release tile i+1 only. kQDOSlots - 2 tiles stay in flight, which
            // is the whole point of the depth: at kQDOSlots == 2 this is a full
            // drain and the transfer has had one iteration's compute to hide
            // behind; at 4 it has had three.
            s_wait_tensorcnt_barrier<kTdmWaitCnt>();
#endif
            auto ds_reg_tensor = load_tile_transpose(ds_lds_read_window);
            q_reg_tensor       = load_tile(q_rd_dst);
            lse                = load_tile(lse_rd_dst);

            if constexpr(!kDropStagedSched || (CK_TILE_FMHA_BWD_SCHED_KEEP_MASK & (1 << 3)))
            {
                HotLoopScheduler::template GemmStagedScheduler<3>();
                __builtin_amdgcn_sched_barrier(0);
            }
            // STAGE7 SGrad@K^T Gemm4
            auto dq_acc = QGradBlockTileType{};
            clear_tile(dq_acc);

            static_for<0, k4_loops, 1>{}([&](auto i_k4) {
                auto kt_reg_tensor_slice = get_slice_tile(kt_reg_tensor,
                                                          sequence<0, i_k4 * kK4>{},
                                                          sequence<kQKHeaddim, (i_k4 + 1) * kK4>{});
                gemm_4(dq_acc,
                       get_slice_tile(ds_reg_tensor,
                                      sequence<0, i_k4 * kK4>{},
                                      sequence<kM0, (i_k4 + 1) * kK4>{}),
                       kt_reg_tensor_slice);
            });

            do_reg_tensor = load_tile(do_rd_dst);
            d             = load_tile(d_rd_dst);

            if constexpr(!kDropStagedSched || (CK_TILE_FMHA_BWD_SCHED_KEEP_MASK & (1 << 4)))
            {
                HotLoopScheduler::template GemmStagedScheduler<4>();
            }

            // QGrad Scale
            if constexpr(FmhaDropout::IsDropout)
            {
                tile_elementwise_inout([&scale_rp_undrop](auto& x) { x = x * scale_rp_undrop; },
                                       dq_acc);
            }
            else
            {
                tile_elementwise_inout([&raw_scale](auto& x) { x = x * raw_scale; }, dq_acc);
            }
#if CK_TILE_FMHA_BWD_DQ_ATOMIC_FOLD
            const auto dq_out = FoldQGradForAtomic(dq_acc);
#else
            const auto& dq_out = dq_acc;
#endif
            if constexpr(decltype(dq_dram_window)::BottomTensorView::DstInMemOp ==
                         memory_operation_enum::set)
            {
                store_tile(dq_dram_window, dq_out);
            }
            else
            {
                update_tile(dq_dram_window, dq_out);
            }
            move_tile_window(dq_dram_window, {kM0, 0});

            i_total_loops += 1;
            seqlen_q_step += kM0;
        };

        if constexpr(kQDOSlots >= 3)
        {
            // kQDOSlots copies, one per slot, with an exit test between them.
            // The trip count is arbitrary, so a `main loop xN + peeled
            // remainder` shape would need kQDOSlots - 1 extra copies of the body
            // on top; breaking out of the middle costs a few scalar branches per
            // round instead and emits the body exactly kQDOSlots times.
            while(i_total_loops < (num_total_loop - 1))
            {
                static_for<0, kQDOSlots, 1>{}([&](auto j) {
                    if(i_total_loops < (num_total_loop - 1))
                    {
                        constexpr index_t kCur = j;
                        constexpr index_t kNxt = (kCur + 1) % kQDOSlots;
                        constexpr index_t kWr  = (kCur + kIssueAhead) % kQDOSlots;
                        hot_loop_body(
                            bool_constant<true>{},
                            q_lds_read_windows.at(number<kCur>{}),
                            qt_lds_read_windows.at(number<kCur>{}),
                            do_lds_read_windows.at(number<kCur>{}),
                            dot_lds_read_windows.at(number<kCur>{}),
                            q_lds_read_windows.at(number<kNxt>{}),
                            do_lds_read_windows.at(number<kNxt>{}),
                            lse_lds_read_windows.at(number<kNxt>{}),
                            d_lds_read_windows.at(number<kNxt>{}),
                            [&] { issue_qdo_tile(number<kWr>{}, i_total_loops + kIssueAhead); });
                    }
                });
            }
        }
        else
        {
            // One copy, and the slot is a single bool select on two named
            // windows -- character for character the binding the ping-pong
            // always compiled to. kB is the other slot: 1 at depth 2, and 0 at
            // depth 1, where both arms of every select name the same window and
            // the select folds away.
            constexpr index_t kB   = kQDOSlots - 1;
            constexpr bool kTwoBox = (kQDOSlots == 2);
            // false: read slot 0 and refill slot kB; true: the reverse.
            bool phase = false;

            // How many leading tiles can be edge tiles. Walking IsEdgeTile until
            // it turns false is exact for a non-local mask (it is monotone in
            // seqlen_q_step) and costs a handful of scalar iterations once per
            // workgroup -- typically 2 at kN0/kM0 = 2, and the walk stops at the
            // first false. A local mask, or a K block hanging off the end of the
            // key sequence, yields num_total_loop and the whole thing degrades
            // to exactly the old single masked loop.
            index_t n_edge_tiles = num_total_loop;
#if CK_TILE_FMHA_BWD_SPLIT_EDGE_TILES
            {
                auto tile_is_edge = [&](index_t i) {
                    return mask.IsEdgeTile(seqlen_q_start + i * kM0,
                                           k_origin.at(number<0>{}),
                                           number<kM0>{},
                                           number<kN0>{});
                };
                // Edge tiles are a leading run plus a trailing run, never a hole
                // in the middle: IsEdgeTile is top_right_edge || bottom_left_edge,
                // top_right_edge only ever goes true->false as the loop walks Q
                // down, and bottom_left_edge only ever goes false->true. So the
                // front scan finds the whole leading run, and because the
                // trailing term is monotone, testing the last tile alone decides
                // whether a trailing run exists at all.
                index_t probe = 0;
                while(probe < num_total_loop && tile_is_edge(probe))
                {
                    probe += 1;
                }
                const bool has_trailing_edge =
                    (num_total_loop > 0) && tile_is_edge(num_total_loop - 1);
                // A trailing run would need a third loop to stay correct; not
                // worth a third copy of the body, so those shapes keep the old
                // fully-masked loop and the remainder below runs zero times.
                n_edge_tiles = has_trailing_edge ? num_total_loop : probe;
            }
#endif
            const index_t n_edge_body =
                min(n_edge_tiles, num_total_loop > 0 ? num_total_loop - 1 : 0);

            while(i_total_loops < n_edge_body)
            {
                const bool sec = kTwoBox && phase;
                hot_loop_body(
                    bool_constant<true>{},
                    sec ? q_lds_read_windows.at(number<kB>{}) : q_lds_read_windows.at(number<0>{}),
                    sec ? qt_lds_read_windows.at(number<kB>{})
                        : qt_lds_read_windows.at(number<0>{}),
                    sec ? do_lds_read_windows.at(number<kB>{})
                        : do_lds_read_windows.at(number<0>{}),
                    sec ? dot_lds_read_windows.at(number<kB>{})
                        : dot_lds_read_windows.at(number<0>{}),
                    sec ? q_lds_read_windows.at(number<0>{}) : q_lds_read_windows.at(number<kB>{}),
                    sec ? do_lds_read_windows.at(number<0>{})
                        : do_lds_read_windows.at(number<kB>{}),
                    sec ? lse_lds_read_windows.at(number<0>{})
                        : lse_lds_read_windows.at(number<kB>{}),
                    sec ? d_lds_read_windows.at(number<0>{}) : d_lds_read_windows.at(number<kB>{}),
                    [&] {
                        load_tile_tdm(tdm_config_q,
                                      sec ? q_lds_windows.at(number<0>{})
                                          : q_lds_windows.at(number<kB>{}),
                                      q_dram_window);
                        load_tile_tdm(tdm_config_lse,
                                      sec ? lse_lds_write_windows.at(number<0>{})
                                          : lse_lds_write_windows.at(number<kB>{}),
                                      lse_dram_window);
                        load_tile_tdm(tdm_config_do,
                                      sec ? do_lds_windows.at(number<0>{})
                                          : do_lds_windows.at(number<kB>{}),
                                      do_dram_window);
#if !CK_TILE_FMHA_BWD_ABLATE_DROP_D
                        load_tile_tdm(tdm_config_d,
                                      sec ? d_lds_write_windows.at(number<0>{})
                                          : d_lds_write_windows.at(number<kB>{}),
                                      d_dram_window);
#endif
                        advance_qdo_windows(i_total_loops + kIssueAhead + 1);
                    });
                phase = !phase;
            }
#if CK_TILE_FMHA_BWD_SPLIT_EDGE_TILES
            // Remainder: every tile from here down is fully inside the mask,
            // so this instantiation of the body carries no set_tile_if at all.
            // Guarded so that with the split off the second body is not emitted
            // at all and the code stays byte-for-byte the old single-loop form,
            // which is what makes the A/B honest.
            while(i_total_loops < (num_total_loop - 1))
            {
                const bool sec = kTwoBox && phase;
                hot_loop_body(
                    bool_constant<false>{},
                    sec ? q_lds_read_windows.at(number<kB>{}) : q_lds_read_windows.at(number<0>{}),
                    sec ? qt_lds_read_windows.at(number<kB>{})
                        : qt_lds_read_windows.at(number<0>{}),
                    sec ? do_lds_read_windows.at(number<kB>{})
                        : do_lds_read_windows.at(number<0>{}),
                    sec ? dot_lds_read_windows.at(number<kB>{})
                        : dot_lds_read_windows.at(number<0>{}),
                    sec ? q_lds_read_windows.at(number<0>{}) : q_lds_read_windows.at(number<kB>{}),
                    sec ? do_lds_read_windows.at(number<0>{})
                        : do_lds_read_windows.at(number<kB>{}),
                    sec ? lse_lds_read_windows.at(number<0>{})
                        : lse_lds_read_windows.at(number<kB>{}),
                    sec ? d_lds_read_windows.at(number<0>{}) : d_lds_read_windows.at(number<kB>{}),
                    [&] {
                        load_tile_tdm(tdm_config_q,
                                      sec ? q_lds_windows.at(number<0>{})
                                          : q_lds_windows.at(number<kB>{}),
                                      q_dram_window);
                        load_tile_tdm(tdm_config_lse,
                                      sec ? lse_lds_write_windows.at(number<0>{})
                                          : lse_lds_write_windows.at(number<kB>{}),
                                      lse_dram_window);
                        load_tile_tdm(tdm_config_do,
                                      sec ? do_lds_windows.at(number<0>{})
                                          : do_lds_windows.at(number<kB>{}),
                                      do_dram_window);
#if !CK_TILE_FMHA_BWD_ABLATE_DROP_D
                        load_tile_tdm(tdm_config_d,
                                      sec ? d_lds_write_windows.at(number<0>{})
                                          : d_lds_write_windows.at(number<kB>{}),
                                      d_dram_window);
#endif
                        advance_qdo_windows(i_total_loops + kIssueAhead + 1);
                    });
                phase = !phase;
            }
#endif
        }
        __builtin_amdgcn_sched_barrier(0);

        // The tail runs on tile num_total_loop - 1, whose slot is only known at
        // run time. It is outside the loop, so this is one select on an LDS base
        // address rather than anything the schedule depends on.
        const index_t i_tail_slot = (num_total_loop - 1) % kQDOSlots;

        // Tail
        auto s_acc = SPBlockTileType{};

        // STAGE 1, Q@K Gemm0
        s_acc = gemm_0(q_reg_tensor, k_reg_tensor);

        // STAGE 2, Scale, Add bias, Mask, Softmax, Dropout
        if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
        {
            const auto bias_tile    = load_tile(bias_dram_window);
            auto shuffled_bias_tile = make_static_distributed_tensor<BiasDataType>(
                Policy::template MakeShuffledBiasTileDistribution<Problem>());
            shuffle_tile(shuffled_bias_tile, bias_tile);
            // SGrad and Bias use the same address in LDS, finish loading ds in the hot loop to
            // reuse LDS.
            block_sync_lds();
            store_tile(bias_lds_write_window, shuffled_bias_tile);
            block_sync_lds();
            auto bias_s_tile = load_tile(bias_s_lds_read_window);
            tile_elementwise_inout(
                [&](auto& x, const auto& y) {
                    x = scale * x + log2e_v<AccDataType> * type_convert<AccDataType>(y);
                },
                s_acc,
                bias_s_tile);
            __builtin_amdgcn_sched_barrier(0);
        }
        else if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
        {
            constexpr auto s_spans = decltype(s_acc)::get_distributed_spans();
            sweep_tile_span(s_spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(s_spans[number<1>{}], [&](auto idx1) {
                    const auto tile_idx = get_x_indices_from_distributed_indices(
                        s_acc.get_tile_distribution(), make_tuple(idx0, idx1));

                    const auto row         = seqlen_q_step + tile_idx.at(number<0>{});
                    const auto col         = k_origin.at(number<0>{}) + tile_idx.at(number<1>{});
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);

                    s_acc(i_j_idx) *= scale;
                    position_encoding.update(s_acc(i_j_idx), row, col);
                });
            });
        }

        {
            bool need_perpixel_check = mask.IsEdgeTile(
                seqlen_q_step, k_origin.at(number<0>{}), number<kM0>{}, number<kN0>{});
#if CK_TILE_FMHA_BWD_ABLATE_NO_MASK
            need_perpixel_check = false;
#elif CK_TILE_FMHA_BWD_ABLATE_ALL_MASK
            need_perpixel_check = FmhaMask::IsMasking;
#endif
            if(need_perpixel_check)
            {
                set_tile_if(s_acc, -numeric<AccDataType>::infinity(), [&](auto tile_idx) {
                    const auto row = seqlen_q_step + tile_idx.at(number<0>{});
                    const auto col = k_origin.at(number<0>{}) + tile_idx.at(number<1>{});
                    return mask.IsOutOfBound(row, col);
                });
            }
        }

        static const auto get_validated_lse = [](LSEDataType raw_lse) {
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                         FmhaMask::IsMasking)
            {
#if CK_TILE_FMHA_BWD_ABLATE_NO_LSE_VALIDATE
                return raw_lse;
#else
                // See the hot-loop copy: the sentinel only has to be finite.
                return max(raw_lse, type_convert<LSEDataType>(-1e30f));
#endif
            }
            else
            {
                return raw_lse;
            }
        };

        auto p                 = SPBlockTileType{};
        constexpr auto p_spans = decltype(p)::get_distributed_spans();
        sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            auto row_lse         = log2e_v<LSEDataType> * get_validated_lse(lse[i_idx]);

            sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                             BiasEnum == BlockAttentionBiasEnum::ALIBI)
                {
                    p(i_j_idx) = exp2(s_acc[i_j_idx] - row_lse);
                }
                else
                {
                    p(i_j_idx) = exp2(scale * s_acc[i_j_idx] - row_lse);
                }
            });
        });

        if constexpr(FmhaDropout::IsDropout)
        {
            dropout.template Run<decltype(gemm_0), RandValOutputDataType>(
                seqlen_q_step, k_origin.at(number<0>{}), p, randval_dram_window);
        }

        // STAGE 3, P^T@OGrad^T Gemm1
        const auto p_gemm = [&]() {
            if constexpr(FmhaDropout::IsDropout)
            {
                return tile_elementwise_in(
                    [](const auto& x) { return type_convert<GemmDataType>(x > 0.f ? x : 0.f); }, p);
            }
            else
            {
                return cast_tile<GemmDataType>(p);
            }
        }();

        Policy::template PTFromGemm0CToGemm1A<Problem, decltype(pt_reg_tensor), decltype(p_gemm)>(
            pt_reg_tensor, p_gemm);
        auto& dot_rd_tail = pick_slot(dot_lds_read_windows, i_tail_slot);
        auto& qt_rd_tail  = pick_slot(qt_lds_read_windows, i_tail_slot);

        auto dot_reg_tensor = load_tile_transpose(dot_rd_tail);
        {
            if constexpr(kDVInReg)
            {
                gemm_1(dv_acc, pt_reg_tensor, dot_reg_tensor);
            }
            else
            {
                auto dv_lds = load_tile(dv_acc_lds_window);
                gemm_1(dv_lds, pt_reg_tensor, dot_reg_tensor);
                store_tile(dv_acc_lds_window, dv_lds);
            }
        }

        // STAGE 4, OGrad@V Gemm2
        auto dp_acc = SPGradBlockTileType{};

        auto qt_reg_tensor = load_tile_transpose(qt_rd_tail);

        if constexpr(kVNonResident)
        {
            v_reg_tensor = load_tile(v_lds_read_window);
        }
        dp_acc = gemm_2(do_reg_tensor, v_reg_tensor);

        HotLoopScheduler::template GemmStagedScheduler<2>();
        __builtin_amdgcn_sched_barrier(0);

        // STAGE 5, P^T(PGrad^T - D)
        auto ds                 = SPGradBlockTileType{};
        constexpr auto ds_spans = decltype(ds)::get_distributed_spans();
        sweep_tile_span(ds_spans[number<0>{}], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            sweep_tile_span(ds_spans[number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                bool undrop_flag       = p[i_j_idx] >= 0;
                ds(i_j_idx)            = p[i_j_idx] * (!FmhaDropout::IsDropout || undrop_flag
                                                           ? (dp_acc[i_j_idx] - d[i_idx])
                                                           : d[i_idx]);
            });
        });

        if constexpr(kHasBiasGrad)
        {
            const auto dbias = [&]() {
                if constexpr(FmhaDropout::IsDropout)
                {
                    return tile_elementwise_in(
                        [&rp_undrop](const auto& x) {
                            return type_convert<BiasGradDataType>(x * rp_undrop);
                        },
                        ds);
                }
                else
                {
                    return cast_tile<BiasGradDataType>(ds);
                }
            }();
            // Finish loading bias_s to reuse LDS.
            block_sync_lds();
            store_tile(bias_lds_write_window, dbias);
            block_sync_lds();
            auto shuffled_dbias_tile = load_tile(dbias_lds_read_window);
            auto dbias_tile          = make_static_distributed_tensor<BiasGradDataType>(
                Policy::template MakeBiasTileDistribution<Problem>());
            shuffle_tile(dbias_tile, shuffled_dbias_tile);
            store_tile(dbias_dram_window, dbias_tile);
            __builtin_amdgcn_sched_barrier(0);
        }

        // STAGE 6, SGrad^T@Q^T Gemm3
        const auto ds_gemm = cast_tile<GemmDataType>(ds);

        Policy::template SGradTFromGemm2CToGemm3A<Problem,
                                                  decltype(dst_reg_tensor),
                                                  decltype(ds_gemm)>(dst_reg_tensor, ds_gemm);

        gemm_3(dk_acc, dst_reg_tensor, qt_reg_tensor);

        // SGrad and Bias/BiasGrad use the same address in LDS, finish loading bias/dbias or, when
        // bias is not used, loading ds in the hot loop to reuse LDS.
        block_sync_lds();
        store_tile(ds_lds_window, dst_reg_tensor);

        block_sync_lds();

        auto ds_reg_tensor = load_tile_transpose(ds_lds_read_window);

        HotLoopScheduler::template GemmStagedScheduler<3>();
        __builtin_amdgcn_sched_barrier(0);
        // STAGE 7, SGrad@K^T Gemm4
        auto dq_acc = QGradBlockTileType{};
        clear_tile(dq_acc);

        static_for<0, k4_loops, 1>{}([&](auto i_k4) {
            auto kt_reg_tensor_slice = get_slice_tile(
                kt_reg_tensor, sequence<0, i_k4 * kK4>{}, sequence<kQKHeaddim, (i_k4 + 1) * kK4>{});

            gemm_4(dq_acc,
                   get_slice_tile(
                       ds_reg_tensor, sequence<0, i_k4 * kK4>{}, sequence<kM0, (i_k4 + 1) * kK4>{}),
                   kt_reg_tensor_slice);
        });

        HotLoopScheduler::template GemmStagedScheduler<4>();
        __builtin_amdgcn_sched_barrier(0);

        // Pull the finished accumulators out of LDS. They come back with the
        // gemm C distribution, which is what the epilogue already expects, so
        // the return type is unchanged from the register-resident pipeline.
        if constexpr(!kDVInReg)
        {
            dv_acc = load_tile(dv_acc_lds_window);
        }

        // Results Scale
        if constexpr(FmhaDropout::IsDropout)
        {
            tile_elementwise_inout([&scale_rp_undrop](auto& x) { x = x * scale_rp_undrop; },
                                   dq_acc);
            tile_elementwise_inout([&scale_rp_undrop](auto& x) { x = x * scale_rp_undrop; },
                                   dk_acc);
            tile_elementwise_inout([&rp_undrop](auto& x) { x = x * rp_undrop; }, dv_acc);
        }
        else
        {
            tile_elementwise_inout([&raw_scale](auto& x) { x = x * raw_scale; }, dq_acc);
            tile_elementwise_inout([&raw_scale](auto& x) { x = x * raw_scale; }, dk_acc);
        }

#if CK_TILE_FMHA_BWD_DQ_ATOMIC_FOLD
        const auto dq_out = FoldQGradForAtomic(dq_acc);
#else
        const auto& dq_out = dq_acc;
#endif
        if constexpr(decltype(dq_dram_window)::BottomTensorView::DstInMemOp ==
                     memory_operation_enum::set)
        {
            store_tile(dq_dram_window, dq_out);
        }
        else
        {
            update_tile(dq_dram_window, dq_out);
        }

        return make_tuple(dk_acc, dv_acc);
    }
};

} // namespace ck_tile
