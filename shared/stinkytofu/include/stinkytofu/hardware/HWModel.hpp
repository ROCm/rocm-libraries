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

    /// s_delay_alu scoreboard depths. Named after the HW register fields these
    /// bound: DEP_1..4 (VALU), DEP_1..3 (TRANS), SALU_CYCLE_1..3.
    ///
    /// `unsigned` matches the types these values feed in InsertDelayAluPass —
    /// they initialize uint8_t members and are compared against unsigned cycle
    /// counts. Do not narrow to int; it changes the signedness of those comparisons.
    struct AluScoreboard {
        unsigned valuMax;
        unsigned transMax;
        unsigned saluCyclesMax;
    };

    /// Co-execution hazard spacing. The per-producer V_NOP counts come from each
    /// instruction's HwInstDesc::coIssueWindow bitmask at runtime; only the
    /// arch-level rules live here.
    struct Coexec {
        /// TRANS -> TRANS and TRANS -> XDL WMMA spacing.
        int transToNonCoreSide;
        bool hwHandlesTransToCoreSide;
        /// Bounds the backward scan for co-exec hazards.
        int maxSlotBudget;
    };

    /// Producer->consumer hazard gap rules. Points at the arch's static rule
    /// table (see HazardRules.hpp); this is a reference to that table, not a copy.
    struct Hazards {
        const HazardRule* rules;
        int numRules;
    };

    Lds lds;
    Barrier barrier;
    AluScoreboard aluScoreboard;
    Coexec coexec;
    Hazards hazards;
};

/// Look up the hardware model for \p arch (the {major, minor, stepping} triple
/// from GemmTileConfig).
///
/// Keyed on the arch triple rather than GfxArchID because the triple covers archs
/// that are tuned separately but not registered in Config/Archs.def (gfx1250v0,
/// {12,5,1}); getGfxArchID() cannot round-trip those. gfx1250 is the fallback for
/// any unlisted arch.
STINKYTOFU_EXPORT const HWModel& hwModelForArch(const std::array<int, 3>& arch);

}  // namespace stinkytofu
