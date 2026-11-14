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
//   3. today's static defaults (HW / CDNA5 / Rule3=100 /
//      ReduceClusterBarrierPerLoop=off / TensorLoadWmmaSpace=0 /
//      TensorLoadDsLoadGapCycles=64) when the main loop is missing or
//      degenerate (either count == 0)
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

/// Historical ModuleOptions / InsertClusterBarrierPass default.
inline constexpr int kStaticDefaultClusterBarrierRule3SignalLeadCycles = 100;

/// Historical tensor-load spacing when the main loop is missing or degenerate.
/// Space 0 leaves the exclusive-barrier thresholds unshifted. Gap 64 is the
/// long-standing extra gap between an after-barrier and the before-side ds_loads.
inline constexpr int kStaticDefaultTensorLoadWmmaSpace = 0;
inline constexpr int kStaticDefaultTensorLoadDsLoadGapCycles = 64;

/// Policy values when both main-loop counts are positive. Chosen on the
/// K>=1024 OOB winners: space 6 with gap 64 beat space 0 by 1.66% geomean,
/// and no case was more than 5% slower. Kernel features did not hold a
/// stabler per-kernel split.
inline constexpr int kPolicyTensorLoadWmmaSpace = 6;
inline constexpr int kPolicyTensorLoadDsLoadGapCycles = 64;

/// Static default when the main loop is missing or degenerate. The policy turns
/// this on when sumWmmaLatencyCycles > 2500 and that sum covers at least 5
/// cycles per ds_load.
inline constexpr bool kStaticDefaultReduceClusterBarrierPerLoop = false;

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

    bool degenerate() const {
        return wmmaCount <= 0 || dsLoadCount <= 0;
    }
};

/// Stable feature schema for heuristic / future NN policies (bump version when
/// the layout of fields that models train on changes).
struct SchedulingFeatures {
    int featureVersion = 7;
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
    int clusterBarrierRule3SignalLeadCycles = kStaticDefaultClusterBarrierRule3SignalLeadCycles;
    SchedulingKnobSource clusterBarrierRule3SignalLeadCyclesSource =
        SchedulingKnobSource::StaticDefault;
    /// First trigger of each enclosing loop only. False handshakes every trigger.
    bool reduceClusterBarrierPerLoop = kStaticDefaultReduceClusterBarrierPerLoop;
    SchedulingKnobSource reduceClusterBarrierPerLoopSource = SchedulingKnobSource::StaticDefault;
    /// Extra cycle slack after the critical ds_load returns and before its first matrix consumer.
    int queueFillSafetyMarginCycles = 0;
    SchedulingKnobSource queueFillSafetyMarginCyclesSource = SchedulingKnobSource::StaticDefault;
    /// Exclusive after/before barrier separation, in WMMA windows. 0 leaves the
    /// thresholds unshifted.
    int tensorLoadWmmaSpace = kStaticDefaultTensorLoadWmmaSpace;
    SchedulingKnobSource tensorLoadWmmaSpaceSource = SchedulingKnobSource::StaticDefault;
    /// Extra cycles between an after-barrier and the before-side ds_loads. 0 disables.
    int tensorLoadDsLoadGapCycles = kStaticDefaultTensorLoadDsLoadGapCycles;
    SchedulingKnobSource tensorLoadDsLoadGapCyclesSource = SchedulingKnobSource::StaticDefault;
};

/// Per-knob user overrides. nullopt = unset (eligible for policy / static).
struct SchedulingKnobOverrides {
    std::optional<int> clusterBarrierRule3SignalLeadCycles;
    std::optional<bool> reduceClusterBarrierPerLoop;
    std::optional<int> tensorLoadWmmaSpace;
    std::optional<int> tensorLoadDsLoadGapCycles;
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
///   ClusterBarrierRule3SignalLeadCycles < 0       → unset
///   ReduceClusterBarrierPerLoop < 0                  → unset (0 off, >0 on)
///   TensorLoadWmmaSpace < 0                          → unset (0 off, >0 space)
///   TensorLoadDsLoadGapCycles < 0                    → unset (0 off, >0 cycles)
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
