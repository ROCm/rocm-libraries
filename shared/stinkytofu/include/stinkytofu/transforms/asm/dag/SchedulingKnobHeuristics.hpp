// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Per-kernel scheduling-knob policy: when ModuleOptions leave a knob unset,
// derive a concrete value from main-loop IR stats (+ shape/arch facts) instead
// of silently falling through to HWModel / CDNA5Config static defaults.
//
// Precedence (independent per knob):
//   1. user-explicit ModuleOptions value
//   2. SchedulingKnobPolicy::propose() when main-loop wmma and ds_load counts
//      are both > 0
//   3. today's static defaults (HW / CDNA5 / Rule3=100) when the main loop is
//      missing or degenerate (either count == 0)
//
// HeuristicSchedulingKnobPolicy is the current owner-tunable formula. A future
// neural policy can implement the same interface without touching the DAG
// scheduler or InsertClusterBarrierPass.

#include <array>
#include <cstdint>
#include <iosfwd>
#include <optional>
#include <string_view>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/core/Types.hpp"
#include "stinkytofu/hardware/HWModel.hpp"

namespace stinkytofu {

/// Named region used as the "main loop" feature source.
inline constexpr std::string_view kMainLoopGroupName = "loopWithPrefetch";

/// CDNA5 scheduling-policy default / heuristic cap for dsReadPerCap.
/// Keep aligned with kGfx1250Config.dsReadPerCap in CDNA5.hpp (not HWModel).
inline constexpr int kStaticDefaultDsReadPerCap = 3;
/// Historical ModuleOptions / InsertClusterBarrierPass default.
inline constexpr int kStaticDefaultClusterBarrierRule3SignalLeadCycles = 100;
/// Policy default TDM-anchored ds cap for kernels with tensor_load in the main loop: while the
/// age of the latest tensor_load issue is in the TDM landing window, ds_loads are capped to one
/// per span, keeping them out of the TDM LDS-write window.
/// Landing model (sched_test, gfx1250): the TDM engine moves ~256 B/clk shared by the issuing
/// waves after ~330 cycles of fixed latency, so a stage's last byte lands at
///   tEnd = kTdmLatencyCycles + stageBytes / kTdmBytesPerCycle
/// and ds->ds stalls cluster in [tEnd - kTdmWindowPreCycles, tEnd + kTdmWindowPostCycles).
/// henry fp8 (128 KB per stage): tEnd = 842, window [602, 1002).
inline constexpr int kTdmLatencyCycles = 330;
inline constexpr int kTdmBytesPerCycle = 256;
inline constexpr int kTdmWindowPreCycles = 240;
inline constexpr int kTdmWindowPostCycles = 160;
/// Window used when the stage size is unknown (ModuleOptions::TdmStageBytes < 0).
inline constexpr int kPolicyDsCapTdmBeginCycles = 576;
inline constexpr int kPolicyDsCapTdmEndCycles = 840;
inline constexpr int kPolicyDsCapTdmPerCap = 1;
/// Kernels that also TDM MX scale tensors (a second TDM group per stage) lose ~1.4% with the
/// tight cap (fp4 / MX) and gain nothing: keep the stall-free ds rate, 2 per 8-cycle span.
inline constexpr int kPolicyDsCapTdmPerCapScaled = 2;
/// The policy is only proposed when the main loop has at least this many WMMAs.
inline constexpr int kPolicyDsCapTdmMinWmmas = 128;

enum class SchedulingKnobSource : uint8_t {
    User,           ///< Explicit ModuleOptions override
    Policy,         ///< SchedulingKnobPolicy::propose()
    StaticDefault,  ///< Degenerate / missing main loop
};

struct SchedulingIRStats {
    int wmmaCount = 0;
    int dsLoadCount = 0;
    /// tensor_load (TDM) instructions in the main loop; 0 for non-TDM kernels.
    int tensorLoadCount = 0;
    /// Sum of `latencyCycles` over main-loop matrix instructions.
    int sumWmmaLatencyCycles = 0;
    /// `latencyCycles` of the first main-loop matrix instruction (0 if none).
    int firstWmmaLatencyCycles = 0;
    /// `latencyCycles` of the first main-loop ds_load (0 if none).
    int firstDsLoadLatencyCycles = 0;

    bool degenerate() const {
        return wmmaCount <= 0 || dsLoadCount <= 0;
    }
};

/// Stable feature schema for heuristic / future NN policies (bump version when
/// the layout of fields that models train on changes).
struct SchedulingFeatures {
    int featureVersion = 6;
    std::array<int, 3> arch{};
    SchedulingIRStats stats{};
    /// Optional tile/wave shape from ModuleOptions (0 = unknown / unset).
    int tileA0 = 0;
    int tileB0 = 0;
    int waveGroup0 = 0;
    int waveGroup1 = 0;
    int prefetchGlobalRead = 0;
    int prefetchLocalRead = 0;
    /// Tensile `KernelWriter.states.unrollLoopCopies`: how many unrolled loop
    /// bodies are emitted. HalfPLR sets this to 3. 0 = not provided.
    int unrollLoopCopies = 0;
    /// LDS bytes the TDM writes per stage (A + B tiles); <= 0 = unknown.
    int tdmStageBytes = 0;
    /// The kernel TDMs MX scale tensors besides the data tiles.
    bool tdmScaleTensors = false;
};

struct ResolvedSchedulingKnobs {
    int dsReadThrottleLatency = 0;
    int dsReadPerCap = kStaticDefaultDsReadPerCap;
    int clusterBarrierRule3SignalLeadCycles = kStaticDefaultClusterBarrierRule3SignalLeadCycles;
    /// TDM-anchored ds cap: tensor_load age window [Begin, End) in cycles; PerCap <= 0 = off.
    int dsCapTdmBeginCycles = -1;
    int dsCapTdmEndCycles = -1;
    int dsCapTdmPerCap = 0;

    SchedulingKnobSource dsReadThrottleLatencySource = SchedulingKnobSource::StaticDefault;
    SchedulingKnobSource dsReadPerCapSource = SchedulingKnobSource::StaticDefault;
    SchedulingKnobSource clusterBarrierRule3SignalLeadCyclesSource =
        SchedulingKnobSource::StaticDefault;
    SchedulingKnobSource dsCapTdmSource = SchedulingKnobSource::StaticDefault;

    /// Latency-budget throttle estimate. Diagnostic only: `applyResolvedSchedulingKnobs`
    /// does not copy it, and it is not combined with `dsReadThrottleLatency`.
    /// -1 = not computed (degenerate main loop).
    int optimisticDsReadThrottleLatency = -1;
};

/// Per-knob user overrides. nullopt = unset (eligible for policy / static).
struct SchedulingKnobOverrides {
    std::optional<int> dsReadThrottleLatency;
    std::optional<int> dsReadPerCap;
    std::optional<int> clusterBarrierRule3SignalLeadCycles;
    /// DsCapTdm* options; each is independent (nullopt = unset). PerCap 0 = explicit off.
    std::optional<int> dsCapTdmBeginCycles;
    std::optional<int> dsCapTdmEndCycles;
    std::optional<int> dsCapTdmPerCap;
};

struct SchedulingKnobPolicy {
    virtual ~SchedulingKnobPolicy() = default;
    /// Propose values for every knob. Caller applies overrides independently.
    /// Only invoked when stats are non-degenerate.
    virtual ResolvedSchedulingKnobs propose(const SchedulingFeatures& features,
                                            const HWModel& hw) const = 0;
};

/// Owner-tunable closed-form policy. Replace with a neural policy later.
struct STINKYTOFU_EXPORT HeuristicSchedulingKnobPolicy : SchedulingKnobPolicy {
    ResolvedSchedulingKnobs propose(const SchedulingFeatures& features,
                                    const HWModel& hw) const override;
};

/// Static defaults that degenerate IR falls back to (and that policy may use as
/// a floor). Not used on the normal unset→policy path.
STINKYTOFU_EXPORT ResolvedSchedulingKnobs
staticSchedulingKnobDefaults(const std::array<int, 3>& arch);

/// Count WMMA / ds_load in the module's main-loop group (`loopWithPrefetch`).
STINKYTOFU_EXPORT SchedulingIRStats countMainLoopSchedulingIRStats(const StinkyAsmModule& module);

/// Build overrides from ModuleOptions sentinels:
///   DsReadThrottleLatency <= 0                    → unset
///   DsReadPerCap < 0                             → unset
///   ClusterBarrierRule3SignalLeadCycles < 0       → unset
///   DsCapTdmBeginCycles / DsCapTdmEndCycles / DsCapTdmPerCap < 0 → unset (policy: TDM kernels
///   get tensor_load age [576, 840) cycles x1); DsCapTdmPerCap == 0 is an explicit "off"
STINKYTOFU_EXPORT SchedulingKnobOverrides
schedulingKnobOverridesFromModuleOptions(const StinkyAsmModule::ModuleOptions& opts);

STINKYTOFU_EXPORT SchedulingFeatures schedulingFeaturesFromModule(const StinkyAsmModule& module);

/// Resolve each knob independently under the contract above.
STINKYTOFU_EXPORT ResolvedSchedulingKnobs
resolveSchedulingKnobs(const SchedulingFeatures& features, const SchedulingKnobOverrides& overrides,
                       const SchedulingKnobPolicy& policy);

/// Convenience: count main loop + read ModuleOptions + resolve.
STINKYTOFU_EXPORT ResolvedSchedulingKnobs
resolveSchedulingKnobsForModule(const StinkyAsmModule& module, const SchedulingKnobPolicy& policy);

STINKYTOFU_EXPORT ResolvedSchedulingKnobs
resolveSchedulingKnobsForModule(const StinkyAsmModule& module);

/// Write resolved DAG knobs into PassFeatureConfig (always concrete values).
STINKYTOFU_EXPORT void applyResolvedSchedulingKnobs(PassFeatureConfig& config,
                                                    const ResolvedSchedulingKnobs& resolved);

STINKYTOFU_EXPORT const char* schedulingKnobSourceName(SchedulingKnobSource source);

/// One-line stderr-friendly dump of resolved knobs + main-loop stats.
STINKYTOFU_EXPORT void logResolvedSchedulingKnobs(std::ostream& os, std::string_view moduleName,
                                                  const SchedulingFeatures& features,
                                                  const ResolvedSchedulingKnobs& resolved);

/// Same dump, gated by PASS_DEBUG / DEBUG_TYPE `"SchedulingKnobHeuristics"`
/// (`StinkyTofuDebugPass: "SchedulingKnobHeuristics"`).
STINKYTOFU_EXPORT void logResolvedSchedulingKnobsIfDebug(std::string_view moduleName,
                                                         const SchedulingFeatures& features,
                                                         const ResolvedSchedulingKnobs& resolved);

}  // namespace stinkytofu
