// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stinkytofu/transforms/asm/dag/SchedulingKnobHeuristics.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <ostream>
#include <string>

#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

// Enable via PassManagerDebugConfig::addDebugOnly("SchedulingKnobHeuristics")
// or `StinkyTofuDebugPass: "SchedulingKnobHeuristics"` in YAML.
#define DEBUG_TYPE "SchedulingKnobHeuristics"

namespace stinkytofu {
namespace {

int ceilDivPositive(int num, int den) {
    assert(den > 0);
    return (num + den - 1) / den;
}

}  // namespace

ResolvedSchedulingKnobs staticSchedulingKnobDefaults(const std::array<int, 3>& arch) {
    const HWModel& hw = hwModelForArch(arch);
    ResolvedSchedulingKnobs out;
    out.dsReadThrottleLatency = hw.lds.readThrottleLatency > 0
                                    ? hw.lds.readThrottleLatency
                                    : 4 * std::max(1, hw.lds.readQueueDepth);
    out.dsReadPerWmma = kStaticDefaultDsReadPerWmma;
    out.clusterBarrierRule3SignalLeadCycles = kStaticDefaultClusterBarrierRule3SignalLeadCycles;
    out.dsReadThrottleLatencySource = SchedulingKnobSource::StaticDefault;
    out.dsReadPerWmmaSource = SchedulingKnobSource::StaticDefault;
    out.clusterBarrierRule3SignalLeadCyclesSource = SchedulingKnobSource::StaticDefault;
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
            if (stats.wmmaCount == 0) stats.firstWmmaLatencyCycles = inst->latencyCycles;
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
    // Throttle: <=0 unset. 0 is not a useful explicit value (accessors need >0);
    // treat <=0 as unset so legacy zero-init still goes through resolve.
    if (opts.DsReadThrottleLatency > 0) out.dsReadThrottleLatency = opts.DsReadThrottleLatency;
    // Per-WMMA: <0 unset; 0 is a valid (extreme) override.
    if (opts.DsReadPerWmma >= 0) out.dsReadPerWmma = opts.DsReadPerWmma;
    // Rule3 lead: <0 unset; 0 means co-locate signal and wait.
    if (opts.ClusterBarrierRule3SignalLeadCycles >= 0)
        out.clusterBarrierRule3SignalLeadCycles = opts.ClusterBarrierRule3SignalLeadCycles;
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
                                                               const HWModel& hw) const {
    // Owner-tunable v0 formulas. Each knob is derived independently from
    // features/hw — never from a sibling knob's proposed value.
    assert(!features.stats.degenerate());
    const int wmma = features.stats.wmmaCount;
    const int ds = features.stats.dsLoadCount;

    ResolvedSchedulingKnobs out;

    // Prefer the CDNA5 policy default when WMMA count is small; otherwise
    // ceil(dsLoadCount / wmmaCount), capped at that same default
    // (kStaticDefaultDsReadPerWmma == kGfx1250Config.dsReadPerWmma). Not an
    // HWModel fact — scheduling ratios live in CDNA5Config.
    const int perWmmaCap = kStaticDefaultDsReadPerWmma;
    out.dsReadPerWmma = wmma <= 128 ? perWmmaCap : std::min(perWmmaCap, ceilDivPositive(ds, wmma));
    out.dsReadPerWmmaSource = SchedulingKnobSource::Policy;

    // Independent of the dsReadPerWmma knob above: recompute the same capped
    // ceil ratio, then (firstWmmaLatency / perWmma) * queueDepth, floored at
    // the arch's static readThrottleLatency (72 on gfx1250; queueDepth is 16).
    const int perWmmaForThrottle = std::min(perWmmaCap, ceilDivPositive(ds, wmma));
    const int queueDepth = std::max(1, hw.lds.readQueueDepth);
    const int throttleFloor =
        hw.lds.readThrottleLatency > 0 ? hw.lds.readThrottleLatency : 4 * queueDepth;
    const int firstWmmaLatency = std::max(0, features.stats.firstWmmaLatencyCycles);
    const int computedThrottle = (firstWmmaLatency / perWmmaForThrottle) * queueDepth;
    out.dsReadThrottleLatency = std::max(throttleFloor, computedThrottle);
    out.dsReadThrottleLatencySource = SchedulingKnobSource::Policy;

    // Longer main-loop WMMA latency budgets get a larger Rule3 signal lead.
    out.clusterBarrierRule3SignalLeadCycles = features.stats.sumWmmaLatencyCycles > 500 ? 200 : 100;
    out.clusterBarrierRule3SignalLeadCyclesSource = SchedulingKnobSource::Policy;

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

    if (overrides.dsReadThrottleLatency.has_value()) {
        out.dsReadThrottleLatency = *overrides.dsReadThrottleLatency;
        out.dsReadThrottleLatencySource = SchedulingKnobSource::User;
    } else if (!features.stats.degenerate()) {
        out.dsReadThrottleLatency = proposed.dsReadThrottleLatency;
        out.dsReadThrottleLatencySource = SchedulingKnobSource::Policy;
    } else {
        out.dsReadThrottleLatency = defaults.dsReadThrottleLatency;
        out.dsReadThrottleLatencySource = SchedulingKnobSource::StaticDefault;
    }

    if (overrides.dsReadPerWmma.has_value()) {
        out.dsReadPerWmma = *overrides.dsReadPerWmma;
        out.dsReadPerWmmaSource = SchedulingKnobSource::User;
    } else if (!features.stats.degenerate()) {
        out.dsReadPerWmma = proposed.dsReadPerWmma;
        out.dsReadPerWmmaSource = SchedulingKnobSource::Policy;
    } else {
        out.dsReadPerWmma = defaults.dsReadPerWmma;
        out.dsReadPerWmmaSource = SchedulingKnobSource::StaticDefault;
    }

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

void applyResolvedSchedulingKnobs(PassFeatureConfig& config,
                                  const ResolvedSchedulingKnobs& resolved) {
    config.dagFeatures.dsReadThrottleLatency = resolved.dsReadThrottleLatency;
    config.dagFeatures.dsReadPerWmma = resolved.dsReadPerWmma;
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
       << " firstWmmaLat=" << features.stats.firstWmmaLatencyCycles
       << " sumWmmaLat=" << features.stats.sumWmmaLatencyCycles
       << " dsReadThrottleLatency=" << resolved.dsReadThrottleLatency << "("
       << schedulingKnobSourceName(resolved.dsReadThrottleLatencySource) << ")"
       << " dsReadPerWmma=" << resolved.dsReadPerWmma << "("
       << schedulingKnobSourceName(resolved.dsReadPerWmmaSource) << ")"
       << " rule3SignalLeadCycles=" << resolved.clusterBarrierRule3SignalLeadCycles << "("
       << schedulingKnobSourceName(resolved.clusterBarrierRule3SignalLeadCyclesSource) << ")\n";
}

void logResolvedSchedulingKnobsIfDebug(std::string_view moduleName,
                                       const SchedulingFeatures& features,
                                       const ResolvedSchedulingKnobs& resolved) {
    PASS_DEBUG(logResolvedSchedulingKnobs(std::cerr, moduleName, features, resolved));
}

}  // namespace stinkytofu
