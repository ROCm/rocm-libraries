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

/// CDNA5 scheduling-policy default / heuristic cap for dsReadPerWmma.
/// Keep aligned with kGfx1250Config.dsReadPerWmma in CDNA5.hpp (not HWModel).
inline constexpr int kStaticDefaultDsReadPerWmma = 3;
/// Historical ModuleOptions / InsertClusterBarrierPass default.
inline constexpr int kStaticDefaultClusterBarrierRule3SignalLeadCycles = 100;

enum class SchedulingKnobSource : uint8_t {
    User,           ///< Explicit ModuleOptions override
    Policy,         ///< SchedulingKnobPolicy::propose()
    StaticDefault,  ///< Degenerate / missing main loop
};

struct SchedulingIRStats {
    int wmmaCount = 0;
    int dsLoadCount = 0;
    /// Sum of `latencyCycles` over main-loop matrix instructions.
    int sumWmmaLatencyCycles = 0;
    /// `latencyCycles` of the first main-loop matrix instruction (0 if none).
    int firstWmmaLatencyCycles = 0;

    bool degenerate() const {
        return wmmaCount <= 0 || dsLoadCount <= 0;
    }
};

/// Stable feature schema for heuristic / future NN policies (bump version when
/// the layout of fields that models train on changes).
struct SchedulingFeatures {
    int featureVersion = 3;
    std::array<int, 3> arch{};
    SchedulingIRStats stats{};
    /// Optional tile/wave shape from ModuleOptions (0 = unknown / unset).
    int tileA0 = 0;
    int tileB0 = 0;
    int waveGroup0 = 0;
    int waveGroup1 = 0;
    int prefetchGlobalRead = 0;
    int prefetchLocalRead = 0;
};

struct ResolvedSchedulingKnobs {
    int dsReadThrottleLatency = 0;
    int dsReadPerWmma = kStaticDefaultDsReadPerWmma;
    int clusterBarrierRule3SignalLeadCycles = kStaticDefaultClusterBarrierRule3SignalLeadCycles;

    SchedulingKnobSource dsReadThrottleLatencySource = SchedulingKnobSource::StaticDefault;
    SchedulingKnobSource dsReadPerWmmaSource = SchedulingKnobSource::StaticDefault;
    SchedulingKnobSource clusterBarrierRule3SignalLeadCyclesSource =
        SchedulingKnobSource::StaticDefault;
};

/// Per-knob user overrides. nullopt = unset (eligible for policy / static).
struct SchedulingKnobOverrides {
    std::optional<int> dsReadThrottleLatency;
    std::optional<int> dsReadPerWmma;
    std::optional<int> clusterBarrierRule3SignalLeadCycles;
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
///   DsReadPerWmma < 0                             → unset
///   ClusterBarrierRule3SignalLeadCycles < 0       → unset
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
