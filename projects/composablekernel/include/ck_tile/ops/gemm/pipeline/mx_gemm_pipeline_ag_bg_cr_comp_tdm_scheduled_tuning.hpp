// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core/numeric/integer.hpp"

namespace ck_tile {

/// How the K loop of MxGemmPipelineAgBgCrCompTDMScheduled reads B from LDS. B: in two halves, two
/// s_wait_dscnt per K step, B-outer WMMAs (needs local prefetch and an even N repeat). None: whole.
enum class MxTdmScheduledHalfPlr : int
{
    B    = 0,
    None = 1
};

/// Where the half-B reads of the next K step go (HalfPlr::B only). Ahead: everything but B's second
/// half before the first wait; three B half-sets live. Late: scales and A after the first wait, B's
/// halves one half later; two B half-sets live.
enum class MxTdmScheduledHalfBOrder : int
{
    Ahead = 0,
    Late  = 1
};

/**
 * @brief Tuning of MxGemmPipelineAgBgCrCompTDMScheduled: the scheduling choices that are not part
 * of the tile shape, with the names and defaults of rocKE's gfx1250 MxGemmSpec.
 *
 * The tile (tile_m/n/k), wave group (warp_m/n), WMMA atom (warp tile) and data types come from the
 * pipeline Problem. Derive from this type and override what changes, or use MxTdmScheduledTuningT.
 */
struct MxTdmScheduledDefaultTuning
{
    /// Workgroups per cluster along grid x (N tiles), y (M tiles) and z. 1x1x1 is no cluster
    /// launch. A tiles are multicast along x, B tiles along y.
    static constexpr index_t kClusterDimX = 4;
    static constexpr index_t kClusterDimY = 4;
    static constexpr index_t kClusterDimZ = 1;
    /// GL2 prefetch of the tile this many K tiles ahead; 0 is off, else 2..8.
    static constexpr index_t kGl2PrefetchDistance = 4;
    /// WMMAs of the second-to-last K step before wave 0 signals the cluster; -1 derives it (6,
    /// clamped to the K step's WMMAs, or half of them with HalfPlr::B).
    static constexpr index_t kClusterSignalWmmas = -1;
    /// 1: K step s + 1 is read from LDS during K step s. 0: each K step reads, drains, computes.
    static constexpr index_t kLocalPrefetch               = 1;
    static constexpr MxTdmScheduledHalfPlr kHalfPlr       = MxTdmScheduledHalfPlr::B;
    static constexpr MxTdmScheduledHalfBOrder kHalfBOrder = MxTdmScheduledHalfBOrder::Ahead;
    /// TensileLite's LDSSegmentInterleave 1 (true) or 0 (the baseline region order).
    static constexpr bool kLdsSegmentInterleave = true;
};

/// MxTdmScheduledDefaultTuning with every value given, in a fixed order (for generated code).
template <index_t ClusterDimX,
          index_t ClusterDimY,
          index_t Gl2PrefetchDistance,
          index_t ClusterSignalWmmas,
          index_t LocalPrefetch,
          MxTdmScheduledHalfPlr HalfPlr,
          MxTdmScheduledHalfBOrder HalfBOrder,
          bool LdsSegmentInterleave>
struct MxTdmScheduledTuningT : MxTdmScheduledDefaultTuning
{
    static constexpr index_t kClusterDimX                 = ClusterDimX;
    static constexpr index_t kClusterDimY                 = ClusterDimY;
    static constexpr index_t kGl2PrefetchDistance         = Gl2PrefetchDistance;
    static constexpr index_t kClusterSignalWmmas          = ClusterSignalWmmas;
    static constexpr index_t kLocalPrefetch               = LocalPrefetch;
    static constexpr MxTdmScheduledHalfPlr kHalfPlr       = HalfPlr;
    static constexpr MxTdmScheduledHalfBOrder kHalfBOrder = HalfBOrder;
    static constexpr bool kLdsSegmentInterleave           = LdsSegmentInterleave;
};

} // namespace ck_tile
