// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Per-arch physical hardware facts, in one place, reachable from any pass via
// PassContext::getHWModel().
//
// SCOPE — facts, not policy. A value belongs here only if it describes what the
// silicon does: a queue depth, a fixed latency, a scoreboard size. Scheduling
// heuristics and tunable knobs do NOT belong here; they live in
// PassFeatureConfig (user-overridable, plumbed to both the Python bindings and
// stinkytofu-opt) or stay local to the pass that owns the policy. Examples
// deliberately kept out include InsertClusterBarrierPass's configurable Rule 3
// signal lead and the dsReadPerCap / globalReadPerWmma scheduling ratios in
// CDNA5Config.
//
// Per-opcode LDS drain caps / throughputs live on HwInstDesc (filled from each
// arch's *.Instructions.def via .dsMaxDrain / .dsThroughput). This header only
// keeps arch-level LDS queue facts and fallbacks for opcodes that omit those
// fields. It is deliberately include-light: HazardRule is forward-declared and
// referenced by pointer; only HWModel.cpp includes the rule table itself.

#include <array>
#include <span>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {

struct HazardRule;  // stinkytofu/transforms/asm/dag/HazardRules.hpp

/// Physical hardware facts for one architecture.
///
/// Grouped into per-unit sub-structs so a future arch family can describe the
/// units it actually has. A unit an arch lacks is left zero-valued, which the
/// consuming passes already treat as "inert" (e.g. a zero queue depth disables
/// the corresponding throttle).
struct HWModel {
    /// LDS (ds_read) return-queue model.
    struct Lds {
        int readQueueDepth;
        int readDrainLatency;
        int readThrottleLatency;
        /// Fallback overflow issue throughput (per WGP) when an opcode's
        /// HwInstDesc::dsThroughput is 0.
        int dsLoadDefaultThroughput;
        /// Fallback experimental max drain latency when an opcode's
        /// HwInstDesc::dsMaxDrain is 0.
        int dsLoadDefaultMaxDrain;
        /// Waves that share one ds issue pipeline.
        ///
        /// A ds_load's ISA issue cost is quoted for a single wave. The issue
        /// pipe is shared, so when several resident waves contend for it they
        /// round-robin and one wave sees its own issues spaced out by the
        /// number of waves sharing its pipe. The ISA number is therefore only
        /// correct at one wave; at four waves on a 2-wave-per-pipe part the
        /// effective cost is doubled.
        ///
        /// 0 or 1 = unmodelled, issue cost is taken from the ISA as-is.
        ///
        /// NOTE: this is NOT a substitute for dagFeatures.dsReadPerCap. That
        /// knob is a separate manual ceiling the hardware team tunes; this
        /// field only makes the modelled issue cost match the machine.
        int wavesPerDsIssuePipe;
    };

    /// s_barrier_signal / s_barrier_wait timing, and branch overhead.
    struct Barrier {
        /// Cycles from an s_barrier_signal until a paired s_barrier_wait can
        /// retire.
        int signalToWaitLatency;
        /// Fixed cycle cost charged to a taken branch.
        int jumpOverheadCycles;
    };

    /// Co-execution hazard spacing. The per-producer V_NOP counts come from each
    /// instruction's HwInstDesc::coIssueWindow bitmask at runtime; only the
    /// arch-level rules live here.
    struct Coexec {
        /// TRANS -> TRANS and TRANS -> XDL WMMA spacing. Only the non-core-side
        /// direction is modeled: on every arch here the hardware interlocks the
        /// core-side one. An arch that needs software spacing there has to add
        /// both the field and the code in InsertCoexecHazardPass that reads it -
        /// carrying a flag no pass consults would only look like coverage.
        int transToNonCoreSide;
        /// Bounds the backward scan for co-exec hazards.
        int maxSlotBudget;
    };

    /// Producer->consumer hazard gap rules. Points at the arch's static rule
    /// table (see HazardRules.hpp); this is a reference to that table, not a
    /// copy.
    struct Hazards {
        const HazardRule* rules;
        int numRules;
    };

    /// s_delay_alu SW scoreboard depths plus 1
    struct DelayAlu {
        unsigned valuDepth;
        unsigned transDepth;
        unsigned saluCycleMax;
    };

    /// VMEM completion-counter shape.
    struct Counters {
        /// The legacy vmcnt is split into separate loadcnt/storecnt. When true a
        /// buffer_store bumps STOREcnt only, so it may legally sink across an
        /// s_wait_loadcnt (which tests LOADcnt) without perturbing that wait.
        bool hasSplitLoadStoreCnt;
        /// storecnt and asynccnt are independent.
        bool hasSplitStoreCntAsyncCnt;
    };

    Lds lds;
    Barrier barrier;
    Coexec coexec;
    Hazards hazards;
    DelayAlu delayAlu;
    Counters counters;
};

/// Collapse a {major, minor, stepping} arch triple to a switchable key.
///
/// Keyed on the triple rather than GfxArchID because the triple covers archs
/// that are tuned separately but not registered in Config/Archs.def
/// (gfx1250v0); getGfxArchID() cannot round-trip those.
///
/// This helper and the kArchKey* constants below are the single definition of
/// the encoding. CDNA5.hpp's cdna5ConfigForArch() selects per-arch scheduling
/// *policy* off the same keys that hwModelForArch() selects hardware *facts*
/// off, and the two must stay paired: both fall back to gfx1250 for an unlisted
/// arch, so a mismatch would silently combine one arch's policy with another's
/// facts rather than failing. Adding or restepping an arch is therefore a
/// one-line change here.
constexpr int archKey(const std::array<int, 3>& arch) {
    return arch[0] * 10000 + arch[1] * 100 + arch[2];
}

constexpr int kArchKeyGfx1250 = archKey({12, 5, 0});
// TODO: stepping 1 is a placeholder pending
// https://github.com/ROCm/rocm-libraries/pull/10273 landing the real gfx1250v0
// ArchInfo. Changing it here retargets both the HWModel and the CDNA5 policy
// table.
constexpr int kArchKeyGfx1250v0 = archKey({12, 5, 1});

/// One LDS read in an ordered burst for mixed-type drain estimation.
/// Callers resolve per-opcode throughput / max-drain from HwInstDesc (with
/// HWModel.lds defaults when the desc fields are 0) before pushing an entry.
struct DsLoadDrainEntry {
    int latency = 0;
    int throughput = 0;
    int maxDrain = 0;
};

/// Resolve a drain-model entry from an instruction's latency and optional
/// HwInstDesc overrides. \p dsThroughput / \p dsMaxDrain of 0 select the
/// arch defaults on \p hw.
inline DsLoadDrainEntry makeDsLoadDrainEntry(const HWModel& hw, int latency, int dsThroughput,
                                             int dsMaxDrain) {
    return {
        .latency = latency > 0 ? latency : hw.lds.readDrainLatency,
        .throughput = dsThroughput > 0 ? dsThroughput : hw.lds.dsLoadDefaultThroughput,
        .maxDrain = dsMaxDrain > 0 ? dsMaxDrain : hw.lds.dsLoadDefaultMaxDrain,
    };
}

/// Homogeneous-burst drain estimate. Throughput and max-drain are already
/// resolved (typically via makeDsLoadDrainEntry / HwInstDesc).
int computeDynamicDrainLatency(const HWModel& hw, int matchingDsLoadCount, int targetDSLoadLatency,
                               int dsLoadThroughput, int maxDrainLatency, int numWaves);

/// Mixed-type burst drain estimate.
///
/// Order of non-final loads does not matter. Uses:
/// - latency from the last load
/// - max-drain cap = max over every entry's maxDrain
/// - total load count
/// - issue throughput as the count-weighted average of per-load throughputs
int computeDynamicDrainLatencyForLoads(const HWModel& hw, std::span<const DsLoadDrainEntry> loads,
                                       int numWaves);

/// Effective ds issue cost in cycles for one wave, given how many waves are
/// resident. \p issueCycles is the ISA cost (single-wave); \p numWaves is
/// GemmTileConfig::NumWaves.
///
/// Waves are assumed to pair onto a pipe as soon as there are enough of them to
/// fill one, so two waves on a 2-wave-per-pipe part contend rather than landing
/// on separate pipes. That is the conservative reading and it is UNVERIFIED for
/// exactly NumWaves == 2 -- the ends (1 wave, 4 waves) are known, the middle is
/// not. If hardware turns out to spread instead, this becomes
/// ceilDiv(numWaves, pipeCount) and NumWaves == 2 drops back to the ISA cost.
int dsIssueCyclesForWaves(const HWModel& hw, int issueCycles, int numWaves);

/// Look up the hardware model for \p arch (the {major, minor, stepping} triple
/// from GemmTileConfig). gfx1250 is the fallback for any unlisted arch.
STINKYTOFU_EXPORT const HWModel& hwModelForArch(const std::array<int, 3>& arch);

}  // namespace stinkytofu
