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
/// Policy default ds cap window (region WMMA index, half-open) for TDM kernels: ds_loads are
/// capped to one per span inside [8, 40), keeping them out of the TDM LDS-write window.
inline constexpr int kPolicyDsCapWindowBegin = 8;
inline constexpr int kPolicyDsCapWindowEnd = 40;
inline constexpr int kPolicyDsCapWindowPerCap = 1;
/// The policy window is only proposed when the main loop has at least this many WMMAs.
inline constexpr int kPolicyDsCapWindowMinWmmas = 128;

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
};

struct ResolvedSchedulingKnobs {
    int dsReadThrottleLatency = 0;
    int dsReadPerCap = kStaticDefaultDsReadPerCap;
    int clusterBarrierRule3SignalLeadCycles = kStaticDefaultClusterBarrierRule3SignalLeadCycles;
    /// ds cap window [Begin, End); Begin < 0 or PerCap <= 0 = off.
    int dsCapWindow1Begin = -1;
    int dsCapWindow1End = -1;
    int dsCapWindowPerCap = 0;

    SchedulingKnobSource dsReadThrottleLatencySource = SchedulingKnobSource::StaticDefault;
    SchedulingKnobSource dsReadPerCapSource = SchedulingKnobSource::StaticDefault;
    SchedulingKnobSource clusterBarrierRule3SignalLeadCyclesSource =
        SchedulingKnobSource::StaticDefault;
    SchedulingKnobSource dsCapWindowSource = SchedulingKnobSource::StaticDefault;

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
    /// Set when the user touched any DsCapWindow1*/PerCap option; carries the values as given
    /// (PerCap 0 = explicit off).
    struct DsCapWindow {
        int begin = -1;
        int end = -1;
        int perCap = 0;
    };
    std::optional<DsCapWindow> dsCapWindow;
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
///   DsCapWindowPerCap < 0 and DsCapWindow1Begin < 0 → unset (policy: TDM kernels get [8,40) x1);
///   DsCapWindowPerCap == 0 is an explicit "off"
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
