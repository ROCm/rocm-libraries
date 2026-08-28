// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Per-arch physical hardware facts, in one place, reachable from any pass via
// PassContext::getHWModel().
//
// SCOPE — facts, not policy. A value belongs here only if it describes what the
// silicon does: a queue depth, a fixed latency, a scoreboard size. Scheduling
// heuristics and tunable knobs do NOT belong here; they live in PassFeatureConfig
// (user-overridable, plumbed to both the Python bindings and stinkytofu-opt) or
// stay local to the pass that owns the policy. Two concrete examples of things
// deliberately kept out: InsertClusterBarrierPass's kRule3SignalLeadCycles ("set
// to 0 to co-locate the signal with the wait" — a placement policy) and the
// dsReadPerWmma / globalReadPerWmma scheduling ratios in CDNA5Config.
//
// This header is deliberately include-light: it is reachable from core headers,
// so it must not drag in the asm IR. HazardRule is therefore forward-declared and
// referenced by pointer; only HWModel.cpp includes the rule table itself.

#include <array>

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
    };

    /// s_barrier_signal / s_barrier_wait timing, and branch overhead.
    struct Barrier {
        /// Cycles from an s_barrier_signal until a paired s_barrier_wait can retire.
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
    /// table (see HazardRules.hpp); this is a reference to that table, not a copy.
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
/// Keyed on the triple rather than GfxArchID because the triple covers archs that
/// are tuned separately but not registered in Config/Archs.def (gfx1250v0);
/// getGfxArchID() cannot round-trip those.
///
/// This helper and the kArchKey* constants below are the single definition of the
/// encoding. CDNA5.hpp's cdna5ConfigForArch() selects per-arch scheduling *policy*
/// off the same keys that hwModelForArch() selects hardware *facts* off, and the
/// two must stay paired: both fall back to gfx1250 for an unlisted arch, so a
/// mismatch would silently combine one arch's policy with another's facts rather
/// than failing. Adding or restepping an arch is therefore a one-line change here.
constexpr int archKey(const std::array<int, 3>& arch) {
    return arch[0] * 10000 + arch[1] * 100 + arch[2];
}

constexpr int kArchKeyGfx1250 = archKey({12, 5, 0});
// TODO: stepping 1 is a placeholder pending
// https://github.com/ROCm/rocm-libraries/pull/10273 landing the real gfx1250v0
// ArchInfo. Changing it here retargets both the HWModel and the CDNA5 policy table.
constexpr int kArchKeyGfx1250v0 = archKey({12, 5, 1});

// Internal helper used by stinkytofu passes to model dynamic LDS drain latency
// from per-arch HWModel facts. Not part of the exported API surface.
int computeDynamicDrainLatency(const HWModel& hw, int matchingDsLoadCount, int targetDSLoadLatency,
                               int numWaves);

/// Look up the hardware model for \p arch (the {major, minor, stepping} triple
/// from GemmTileConfig). gfx1250 is the fallback for any unlisted arch.
STINKYTOFU_EXPORT const HWModel& hwModelForArch(const std::array<int, 3>& arch);

}  // namespace stinkytofu
