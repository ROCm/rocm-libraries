// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stinkytofu/transforms/asm/dag/SchedulingKnobHeuristics.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <ostream>
#include <string>

#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

// Enable via PassManagerDebugConfig::addDebugOnly("SchedulingKnobHeuristics")
// or `StinkyTofuDebugPass: "SchedulingKnobHeuristics"` in YAML.
#define DEBUG_TYPE "SchedulingKnobHeuristics"

namespace stinkytofu {

ResolvedSchedulingKnobs staticSchedulingKnobDefaults(const std::array<int, 3>&) {
    ResolvedSchedulingKnobs out;
    out.clusterBarrierRule3SignalLeadCycles = kStaticDefaultClusterBarrierRule3SignalLeadCycles;
    out.clusterBarrierRule3SignalLeadCyclesSource = SchedulingKnobSource::StaticDefault;
    out.reduceClusterBarrierPerLoop = kStaticDefaultReduceClusterBarrierPerLoop;
    out.reduceClusterBarrierPerLoopSource = SchedulingKnobSource::StaticDefault;
    return out;
}

SchedulingIRStats countMainLoopSchedulingIRStats(const StinkyAsmModule& module) {
    SchedulingIRStats stats;
    auto range = module.findGroupRange(std::string(kMainLoopGroupName));
    if (!range) return stats;

    auto [begin, end] = *range;
    for (auto it = begin; it != end; ++it) {
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (!inst) continue;
        if (isMatrixInstruction(*inst)) {
            ++stats.wmmaCount;
            stats.sumWmmaLatencyCycles += inst->latencyCycles;
        } else if (isDSRead(*inst)) {
            ++stats.dsLoadCount;
        }
    }
    return stats;
}

SchedulingKnobOverrides schedulingKnobOverridesFromModuleOptions(
    const StinkyAsmModule::ModuleOptions& opts) {
    SchedulingKnobOverrides out;
    // Rule3 lead: <0 unset; 0 means co-locate signal and wait.
    if (opts.ClusterBarrierRule3SignalLeadCycles >= 0)
        out.clusterBarrierRule3SignalLeadCycles = opts.ClusterBarrierRule3SignalLeadCycles;
    // One handshake per loop: <0 unset; 0 off; >0 on.
    if (opts.ReduceClusterBarrierPerLoop >= 0)
        out.reduceClusterBarrierPerLoop = opts.ReduceClusterBarrierPerLoop != 0;
    return out;
}

SchedulingFeatures schedulingFeaturesFromModule(const StinkyAsmModule& module) {
    const auto& opts = module.getModuleOptions();
    SchedulingFeatures features;
    features.arch = module.getArch();
    features.stats = countMainLoopSchedulingIRStats(module);
    features.tileA0 = opts.TileA0;
    features.tileB0 = opts.TileB0;
    features.waveGroup0 = opts.WaveGroup0;
    features.waveGroup1 = opts.WaveGroup1;
    features.prefetchGlobalRead = opts.PrefetchGlobalRead;
    features.prefetchLocalRead = opts.PrefetchLocalRead;
    return features;
}

ResolvedSchedulingKnobs HeuristicSchedulingKnobPolicy::propose(const SchedulingFeatures& features,
                                                               const HWModel&) const {
    assert(!features.stats.degenerate());
    ResolvedSchedulingKnobs out;

    // Longer main-loop WMMA latency budgets get a larger Rule3 signal lead.
    out.clusterBarrierRule3SignalLeadCycles = features.stats.sumWmmaLatencyCycles > 500 ? 200 : 100;
    out.clusterBarrierRule3SignalLeadCyclesSource = SchedulingKnobSource::Policy;

    // One handshake cadence when the main-loop WMMA latency budget is long and
    // each ds_load has at least 5 cycles of that budget.
    const int sumWmmaLatency = features.stats.sumWmmaLatencyCycles;
    const int dsLoads = features.stats.dsLoadCount;
    out.reduceClusterBarrierPerLoop =
        sumWmmaLatency > 2500 && dsLoads > 0 && sumWmmaLatency / dsLoads >= 5;
    out.reduceClusterBarrierPerLoopSource = SchedulingKnobSource::Policy;

    return out;
}

ResolvedSchedulingKnobs resolveSchedulingKnobs(const SchedulingFeatures& features,
                                               const SchedulingKnobOverrides& overrides,
                                               const SchedulingKnobPolicy& policy) {
    const ResolvedSchedulingKnobs defaults = staticSchedulingKnobDefaults(features.arch);
    const HWModel& hw = hwModelForArch(features.arch);

    ResolvedSchedulingKnobs proposed = defaults;
    if (!features.stats.degenerate()) {
        proposed = policy.propose(features, hw);
    }

    ResolvedSchedulingKnobs out = defaults;

    if (overrides.clusterBarrierRule3SignalLeadCycles.has_value()) {
        out.clusterBarrierRule3SignalLeadCycles = *overrides.clusterBarrierRule3SignalLeadCycles;
        out.clusterBarrierRule3SignalLeadCyclesSource = SchedulingKnobSource::User;
    } else if (!features.stats.degenerate()) {
        out.clusterBarrierRule3SignalLeadCycles = proposed.clusterBarrierRule3SignalLeadCycles;
        out.clusterBarrierRule3SignalLeadCyclesSource = SchedulingKnobSource::Policy;
    } else {
        out.clusterBarrierRule3SignalLeadCycles = defaults.clusterBarrierRule3SignalLeadCycles;
        out.clusterBarrierRule3SignalLeadCyclesSource = SchedulingKnobSource::StaticDefault;
    }

    if (overrides.reduceClusterBarrierPerLoop.has_value()) {
        out.reduceClusterBarrierPerLoop = *overrides.reduceClusterBarrierPerLoop;
        out.reduceClusterBarrierPerLoopSource = SchedulingKnobSource::User;
    } else if (!features.stats.degenerate()) {
        out.reduceClusterBarrierPerLoop = proposed.reduceClusterBarrierPerLoop;
        out.reduceClusterBarrierPerLoopSource = SchedulingKnobSource::Policy;
    } else {
        out.reduceClusterBarrierPerLoop = defaults.reduceClusterBarrierPerLoop;
        out.reduceClusterBarrierPerLoopSource = SchedulingKnobSource::StaticDefault;
    }

    return out;
}

ResolvedSchedulingKnobs resolveSchedulingKnobsForModule(const StinkyAsmModule& module,
                                                        const SchedulingKnobPolicy& policy) {
    return resolveSchedulingKnobs(
        schedulingFeaturesFromModule(module),
        schedulingKnobOverridesFromModuleOptions(module.getModuleOptions()), policy);
}

ResolvedSchedulingKnobs resolveSchedulingKnobsForModule(const StinkyAsmModule& module) {
    HeuristicSchedulingKnobPolicy policy;
    return resolveSchedulingKnobsForModule(module, policy);
}

const char* schedulingKnobSourceName(SchedulingKnobSource source) {
    switch (source) {
        case SchedulingKnobSource::User:
            return "user";
        case SchedulingKnobSource::Policy:
            return "policy";
        case SchedulingKnobSource::StaticDefault:
            return "static";
    }
    return "unknown";
}

void logResolvedSchedulingKnobs(std::ostream& os, std::string_view moduleName,
                                const SchedulingFeatures& features,
                                const ResolvedSchedulingKnobs& resolved) {
    os << "[SchedulingKnobs] module=" << moduleName << " wmma=" << features.stats.wmmaCount
       << " dsLoad=" << features.stats.dsLoadCount
       << " sumWmmaLat=" << features.stats.sumWmmaLatencyCycles
       << " rule3SignalLeadCycles=" << resolved.clusterBarrierRule3SignalLeadCycles << "("
       << schedulingKnobSourceName(resolved.clusterBarrierRule3SignalLeadCyclesSource) << ")"
       << " reduceClusterBarrierPerLoop=" << resolved.reduceClusterBarrierPerLoop << "("
       << schedulingKnobSourceName(resolved.reduceClusterBarrierPerLoopSource) << ")";
    os << "\n";
}

void logResolvedSchedulingKnobsIfDebug(std::string_view moduleName,
                                       const SchedulingFeatures& features,
                                       const ResolvedSchedulingKnobs& resolved) {
    PASS_DEBUG(logResolvedSchedulingKnobs(std::cerr, moduleName, features, resolved));
}

}  // namespace stinkytofu
