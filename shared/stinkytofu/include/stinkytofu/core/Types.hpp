/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#pragma once

#include <array>
#include <climits>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace stinkytofu {
// Error codes for StinkyIRConverter operations
enum class StinkyErrorCode : int {
    SUCCESS = 0,
    PASSCTX_EMPTY = 1,
    PARSE_ERROR = 2,
};

/// GEMM-specific tile configuration
/// This configuration is specific to GEMM kernels and their tiling strategy
/// Note: WavefrontSize is NOT included here as it's derived from architecture,
///       not a user-configurable parameter. Use getWaveFrontSize(arch) to query
///       it.
struct GemmTileConfig {
    // Every member carries a default initializer. Only `arch` used to, so a
    // default-initialized GemmTileConfig left the rest holding whatever was on
    // the stack. Passes read these (NumWaves in RemoveDscntPass and in the CDNA5
    // ds issue-cost model), so scheduling could depend on leftover memory --
    // quiet, and different between runs. Sanitizers flag it as a read of an
    // uninitialized value.
    //
    // NumWaves defaults to 1 rather than 0 because 1 is a real occupancy: a
    // reader gets single-wave behaviour, not a sentinel it has to special-case.
    // The tile sizes default to 0, which is NOT a valid tile, so 0 there means
    // "nobody configured this" and is worth complaining about.
    std::array<int, 3> arch{0, 0, 0};  ///< GPU architecture [gfx, major, minor]
    uint32_t TileA0 = 0;               ///< Tile size for A dimension 0; 0 = unset
    uint32_t TileB0 = 0;               ///< Tile size for B dimension 0; 0 = unset
    uint32_t TileM0 = 0;               ///< Tile size for M dimension 0; 0 = unset
    uint32_t NumGRA = 0;               ///< Number of global read A
    uint32_t NumGRB = 0;               ///< Number of global read B
    uint32_t NumGRM = 0;               ///< Number of global read M
    uint32_t NumWaves = 1;             ///< Number of waves; 1 = single wave
};

/// Pass-specific feature configuration
/// Categorizes optimization behaviors into semantics, properties, and features
struct PassFeatureConfig {
    /// Loop structure and unrolling properties
    /// These are code structure PROPERTIES (not optional features)
    struct LoopConfig {
        bool unrollGemm = false;  ///< Whether GEMM loops are unrolled
    };

    /// DAG scheduler switches.
    /// DS read reorder strategy for WMMA operand scheduling.
    enum class DsReadOrder {
        ProgramOrder,    ///< No reorder (AABB)
        Ascending,       ///< Pair by WMMA affinity: A0 B0 A1 B1
        AscendingCache,  ///< Zigzag for cache reuse: A0 B0 B1 A1
    };

    enum class WmmaRefillMode : int {
        Fixed = 0,    ///< A refill issues wmmaBatchSize WMMAs
        Dynamic = 1,  ///< A refill issues WMMAs until the queued cover reaches wmmaRefillToCycles
    };

    struct DagFeatures {
        bool distributeGlobalRead = false;                 ///< Enable global read distribution
        DsReadOrder dsReadOrder = DsReadOrder::Ascending;  ///< DS read reorder strategy
        /// Max in-flight tensor_load_to_lds credits (HW queue depth, to connect to
        /// sw math cycles). 0 disables the throttle (current behavior).
        int globalReadQueueDepth = 0;
        /// Modeled cycles until one tensor_load_to_lds credit frees. Fed from the
        /// cost/cycle model; varies with layout and problem size.
        int globalReadDrainLatency = 0;
        int dsReadQueueDepth = 0;
        int dsReadDrainLatency = 0;
        int tensorLoadWmmaSpace = 0;
        /// WMMA issue queue: max WMMAs outstanding in the matrix pipe (the pipe buffers
        /// ~8 on gfx1250). A WMMA is appended whenever fewer are outstanding, instead
        /// of waiting for the previous one to finish. 1 = the single-window model.
        int wmmaQueueDepth = 1;
        /// Cycles of queued WMMA work that must remain before a non-WMMA pick (ds_load,
        /// filler, tensor_load) may issue; below it, and with room in the queue, the next
        /// ready WMMA goes first so the pipe never runs dry. Picks the scheduler is forced
        /// to make (a promoted barrier) are not held. 0 = off; ignored at depth 1.
        int wmmaQueueCoverCycles = 0;
        /// How many WMMAs one refill issues once the queue drops below the refill line
        /// (wmmaQueueCoverCycles). Fixed: wmmaBatchSize of them. Dynamic: as many as it takes
        /// for the queued cover to reach wmmaRefillToCycles, counted from the live timeline.
        WmmaRefillMode wmmaRefillMode = WmmaRefillMode::Fixed;
        /// Fixed mode: WMMAs per refill. <= 0 is treated as 1.
        int wmmaBatchSize = 1;
        /// Dynamic mode: queued cover (cycles) a refill fills up to. 0 = a full queue
        /// (wmmaQueueDepth * the WMMA latency).
        int wmmaRefillToCycles = 0;
        /// Spread ds_loads over the region's WMMA batches: each batch takes the region's even share
        /// (see DsSpreadPacer), and a load issues only while the ds queue has a free credit. Off =
        /// ds_loads issue as soon as they are ready and a credit is free.
        bool dsSpread = true;
        /// Queued WMMA cover (cycles) to have in the pipe when a barrier issues, so the WMMAs hide
        /// its stall. The WMMAs before the barrier's threshold are reserved for it: the queue is
        /// refilled to this cover ahead of other work, and ds_loads the barrier waits on are
        /// released that many WMMAs earlier. Applies with dsSpread (or when set > 0) and the
        /// queue model; < 0 = automatic (72 cycles, at most a full queue), 0 = off.
        int wmmaBarrierCoverCycles = -1;
        /// Cycles the LDS needs to return one ds_load (its return rate, the old throttle latency /
        /// queue depth). The windows the ds_loads in front of a barrier need are at least
        /// loads * cycles / WMMA window. 0 = off (the even-spread density only).
        int dsReadCyclesPerLoad = 0;
        /// ds_loads a WMMA batch (dsBatchWmmas WMMAs) may take above the region's even share when
        /// the batch began behind the even line (a dependency delayed its loads), so the region
        /// closes the gap gradually. 0 = no catching up.
        int dsCatchUpPace = 2;
        /// WMMAs per ds_load pacing batch (independent of the queue refill batch wmmaBatchSize).
        int dsBatchWmmas = 4;
        /// Extra cycles kept between an after-barrier and the before-side
        /// ds_loads when exclusive overlap uses gap placement. Converted to
        /// WMMA windows by the region's matrix latency. 0 disables the extra
        /// gap. Mirrors ModuleOptions::TensorLoadDsLoadGapCycles.
        int tensorLoadDsLoadGapCycles = 64;
        /// Max cycle-distance between two adjacent barrier groups for
        /// StinkyMergeBarrierPass to merge them into a single multi-token
        /// barrier group. 0 = use the CDNA5 default (kCdna5MergeBarrierThreshold).
        /// Internal tuning knob only — deliberately not surfaced as a module
        /// option, so TensileLite cannot set it.
        int mergeBarrierThreshold = 0;
        /// Mirrors ModuleOptions::ClusterBarrier: InsertClusterBarrierPass will run
        /// after the scheduler and plant SCC-clobbering handshakes around workgroup
        /// barriers. Enables the scheduler's cluster-barrier SCC rule and the
        /// CDNA5ReadyQueue paths that enforce it (see
        /// ReadyQueue::clusterBarrierEnabled).
        bool clusterBarrier = false;
        /// Mirrors ModuleOptions::LockDsReadOrder. Defaults on: ds_loads that
        /// share a PSEUDO memory token are chained into dsReadPriority order.
        /// Loads on different tokens are not ordered against each other. Set
        /// false to leave a ready lower-priority ds_load free to issue first.
        bool lockDsReadOrder = true;
        /// Mirrors moduleOptions.EnableESM2 && EnableESM2TrackValuVsrc. The mode2 WAR
        /// gate only recovers waits va_vsrc tracking creates, so it is inert when false.
        bool enableESM2TrackValuVsrc = false;
        /// Spread SALU/VALU fillers evenly across WMMA windows: each window is
        /// owed ceil(fillers / WMMAs) of its region and closes once that quota
        /// is met, instead of being padded to its full co-issue length. ds_load
        /// selection and coexec hazard padding are unaffected. See
        /// CDNA5ReadyQueue::fillQuotaPerWindow_ for the full mechanism.
        bool evenSpreadFillers = false;
        /// Mirrors ModuleOptions::WaitAluHoldStrictCount. A VALU/other filler that
        /// InsertWaitAlu would put an s_wait_alu of count <= this before (WaitAluTracker
        /// query) is held until just before the next s_barrier_wait; < 0 = off.
        int waitAluHoldStrictCount = -1;
        /// Mirrors ModuleOptions::PrefetchLeadWmmas. A global prefetch is held until
        /// this many WMMA windows before the tensor_load it precedes, staggered over its
        /// group; below 8 marks a single-stage loop (whole group at the load's window,
        /// ahead of the stage barrier); 0 = off.
        int prefetchLeadWmmas = 0;
        /// Mirrors ModuleOptions::PrefetchLeadMinStageWmmas. A basic block whose stages
        /// (WMMAs / tensor_load groups) are shorter than this runs with no multi-stage
        /// prefetch lead (8 or more); a single-stage lead (below 8) is kept.
        int prefetchLeadMinStageWmmas = 64;
        /// Mirrors ModuleOptions::WarGateWmmas. WMMAs a ds_load waits before
        /// overwriting a vgpr a WMMA read (WmmaVgprSrcToDsWrite); <= 0 = derived.
        int warGateWmmas = 0;
    };

    LoopConfig loopConfig;
    DagFeatures dagFeatures;
};

/// VGPR MSB encoding mode supported by the toolchain.
enum class VgprMsbMode : uint8_t {
    None,   ///< Toolchain does not support `s_set_vgpr_msb`
    Msb8,   ///< 8-bit form only (`s_set_vgpr_msb 0`)
    Msb16,  ///< 16-bit form (`s_set_vgpr_msb 0x0101`) — packs prev + curr MSB
};

/// Capabilities forwarded from rocisa (asmCaps and archCaps) by the conversion
/// layer, or discovered by ToolchainCaps::probe() for the standalone path.
struct AsmCapsConfig {
    VgprMsbMode vgprMsbMode = VgprMsbMode::None;

    /// rocisa archCaps `RequiresXCntForVolatileVMEM`. False on the standalone
    /// path, which has no rocisa to ask.
    /// When set alone (without `enableXnackReplay`), Gfx1250HazardPass only
    /// inserts atomic drains (Rule 4a). See Gfx1250HazardPass for the full
    /// rule set (Rules 1–4).
    bool requiresXCntForVolatileVMEM = false;

    /// Enable full XNACK replay protection in Gfx1250HazardPass:
    ///   - Source-clobber checks: SMEM Rule 3, FLAT Rule 2
    ///   - Boundary drains: ForeverSleep, ScalarPrefetch, VgprMsb
    ///   - Atomic drains: Rule 4a (implies `requiresXCntForVolatileVMEM`)
    /// When false, all of the above are skipped; only Rule 4a remains
    /// active if `requiresXCntForVolatileVMEM` is set independently.
    /// See Gfx1250HazardPass for the complete rule definitions.
    bool enableXnackReplay = false;
};
}  // namespace stinkytofu
