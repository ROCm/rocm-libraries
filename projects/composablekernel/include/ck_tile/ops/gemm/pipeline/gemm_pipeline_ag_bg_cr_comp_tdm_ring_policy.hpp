// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_tdm_default_policy.hpp"

namespace ck_tile {

// Which waves issue the TDM (tensor_load_to_lds) transfers of one ring stage.
enum struct TdmIssueMode
{
    // Every wave issues one slab of A and one slab of B (the comp_tdm V1 split): two TDM ops per
    // wave per stage.
    AllWaves,
    // One wave issues the whole A tile and another the whole B tile: one TDM op per issuing wave
    // per stage, none on the other waves.
    PerOperandWave
};

// Order of the per-tile TENSORcnt wait and the refill of a ring slot.
enum struct TdmOrder
{
    // Wait for tile i+1, barrier, then refill the slot tile i-1 used: one barrier per tile, up to
    // NumStages-1 stages in flight.
    WaitThenFill,
    // Barrier, refill the slot tile i just used, then wait for tile i+1: two barriers per tile,
    // up to NumStages stages in flight. NumStages == 2 is the comp_tdm V1 schedule.
    FillThenWait
};

/**
 * @brief Policy of the TDM ring pipeline (GemmPipelineAgBgCrCompTDMRing).
 *
 * Tile distributions, LDS descriptors (Auto LDS padding) and the block GEMM are the comp_tdm
 * ones; PerOperandWave uses the single-wave (wave-specialized) DRAM distributions so one TDM op
 * moves a whole A or B tile.
 *
 * @tparam NumStages_    ring depth D: LDS slots, each holding one A and one B K-tile (2..4).
 * @tparam IssueMode_    which waves issue the TDM transfers.
 * @tparam Order_        wait/refill order.
 * @tparam SplitBarrier_ WaitThenFill only: arrive (s_barrier_signal) before the last block-GEMM
 *                       of a tile and wait (s_barrier_wait) at the top of the next one.
 *
 * The default (D=3, AllWaves, WaitThenFill, no split barrier) is spill-free at 256x128x64 on 8
 * waves (4x2). At 256x256x64 on 8 waves (2x4) D=3 and D=4 without the split barrier reach 512
 * VGPRs and spill; use D=2 or SplitBarrier there until the register footprint is reduced.
 */
template <index_t NumStages_      = 3,
          TdmIssueMode IssueMode_ = TdmIssueMode::AllWaves,
          TdmOrder Order_         = TdmOrder::WaitThenFill,
          bool SplitBarrier_      = false>
struct GemmPipelineAgBgCrCompTDMRingPolicy
    : public GemmPipelineAgBgCrCompTDMDefaultPolicy<IssueMode_ == TdmIssueMode::PerOperandWave>
{
    static constexpr index_t NumStages      = NumStages_;
    static constexpr TdmIssueMode IssueMode = IssueMode_;
    static constexpr TdmOrder Order         = Order_;
    static constexpr bool SplitBarrier      = SplitBarrier_;

    static_assert(NumStages >= 2 && NumStages <= 4, "TDM ring depth must be 2, 3 or 4");
    static_assert(!SplitBarrier || Order == TdmOrder::WaitThenFill,
                  "the split barrier is only defined for the WaitThenFill order");

    // PerOperandWave issuers. Wave pairs {0,2} and {1,3} are reported to share one tile-DMA
    // engine, so the A and B issuers are kept on waves of different parity.
    static constexpr index_t IssuerWaveA = 0;
    static constexpr index_t IssuerWaveB = 1;
    static_assert(IssuerWaveA % 2 != IssuerWaveB % 2,
                  "A and B issuer waves must differ in parity (SIMD pair rule)");

    // TDM ops one issuing wave adds to its TENSORcnt per stage.
    static constexpr index_t TdmOpsPerStage = IssueMode == TdmIssueMode::AllWaves ? 2 : 1;

    // Stages issued ahead of the one being consumed.
    static constexpr index_t IssueAhead =
        Order == TdmOrder::WaitThenFill ? NumStages - 1 : NumStages;

    // The wait releasing tile i+1 leaves the later issued stages in flight. TENSORcnt retires in
    // issue order, so this is a single compile-time immediate for every tile.
    // AllWaves also assumes that a zero-extent TDM box (a wave's slab past an M/N tail, which
    // tile_window clamps to an extent of 0 but still issues) increments and retires TENSORcnt like
    // any other op. If it did not, a wave with one counted op per stage would pass a non-zero
    // wait with its tile i+1 slab still in flight. Only TdmWaitCnt == 0 (WaitThenFill, D=2) is
    // independent of this; V1's wait of 2 already relies on it. Unverified on hardware.
    static constexpr index_t TdmWaitCnt = (IssueAhead - 1) * TdmOpsPerStage;
};

} // namespace ck_tile
