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
namespace {

int ceilDivPositive(int num, int den) {
    assert(den > 0);
    return (num + den - 1) / den;
}

int perCapForThrottleOf(int dsLoadCount, int wmmaCount) {
    return std::min(kStaticDefaultDsReadPerCap, ceilDivPositive(dsLoadCount, wmmaCount));
}

// Latency-budget throttle. Not combined with dsReadThrottleLatency.
//
//   dsIssueSpace   = sumWmmaLatency - unrollLoopCopies * firstDsLoadLatency
//   basicWmmaUsage = ceil(queueDepth / perCapForThrottle)
//   throttleSpace  = dsIssueSpace - basicWmmaUsage * firstWmmaLatency
//   remainingDs    = dsLoadCount - queueDepth
//   cyclePerDs     = float(throttleSpace) / remainingDs   (only when remainingDs > 0)
//   latency        = lround(cyclePerDs * queueDepth)
struct OptimisticDsReadThrottle {
    int unrollLoopCopies = 0;
    int dsIssueSpace = 0;
    int basicWmmaUsage = 0;
    int throttleSpace = 0;
    int remainingDs = 0;
    float cyclePerDs = 0;
    int latency = 0;
    bool defined = false;
};

OptimisticDsReadThrottle estimateOptimisticDsReadThrottle(int sumWmmaLatencyCycles,
                                                          int unrollLoopCopies,
                                                          int firstWmmaLatency,
                                                          int firstDsLoadLatency, int queueDepth,
                                                          int perCapForThrottle, int dsLoadCount) {
    OptimisticDsReadThrottle out;
    out.unrollLoopCopies = std::max(0, unrollLoopCopies);
    const int first = std::max(0, firstWmmaLatency);
    const int firstDs = std::max(0, firstDsLoadLatency);
    const int depth = std::max(1, queueDepth);
    const int perCap = std::max(1, perCapForThrottle);
    out.dsIssueSpace = sumWmmaLatencyCycles - out.unrollLoopCopies * firstDs;
    out.basicWmmaUsage = ceilDivPositive(depth, perCap);
    out.throttleSpace = out.dsIssueSpace - out.basicWmmaUsage * first;
    out.remainingDs = dsLoadCount - depth;
    out.defined = out.remainingDs > 0;
    if (out.defined) {
        out.cyclePerDs =
            static_cast<float>(out.throttleSpace) / static_cast<float>(out.remainingDs);
        out.latency = static_cast<int>(std::lround(out.cyclePerDs * static_cast<float>(depth)));
    }
    return out;
}

OptimisticDsReadThrottle estimateOptimisticDsReadThrottle(const SchedulingFeatures& features,
                                                          const HWModel& hw) {
    if (features.stats.degenerate()) return {};
    const int queueDepth = std::max(1, hw.lds.readQueueDepth);
    const int perCap = perCapForThrottleOf(features.stats.dsLoadCount, features.stats.wmmaCount);
    return estimateOptimisticDsReadThrottle(
        features.stats.sumWmmaLatencyCycles, features.unrollLoopCopies,
        features.stats.firstWmmaLatencyCycles, features.stats.firstDsLoadLatencyCycles, queueDepth,
        perCap, features.stats.dsLoadCount);
}

}  // namespace

ResolvedSchedulingKnobs staticSchedulingKnobDefaults(const std::array<int, 3>& arch) {
    const HWModel& hw = hwModelForArch(arch);
    ResolvedSchedulingKnobs out;
    out.dsReadThrottleLatency = hw.lds.readThrottleLatency > 0
                                    ? hw.lds.readThrottleLatency
                                    : 4 * std::max(1, hw.lds.readQueueDepth);
    out.dsReadPerCap = kStaticDefaultDsReadPerCap;
    out.clusterBarrierRule3SignalLeadCycles = kStaticDefaultClusterBarrierRule3SignalLeadCycles;
    out.dsReadThrottleLatencySource = SchedulingKnobSource::StaticDefault;
    out.dsReadPerCapSource = SchedulingKnobSource::StaticDefault;
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
            if (stats.dsLoadCount == 0) stats.firstDsLoadLatencyCycles = inst->latencyCycles;
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
    // Cap: <0 unset; 0 is a valid (extreme) override. DsReadPerWmma is the
    // pre-rename spelling -- options arrive by string key, so a caller still
    // using it would otherwise silently get the default. New name wins.
    if (opts.DsReadPerCap >= 0) {
        out.dsReadPerCap = opts.DsReadPerCap;
    } else if (opts.DsReadPerWmma >= 0) {
        out.dsReadPerCap = opts.DsReadPerWmma;
    }
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
    features.unrollLoopCopies = opts.UnrollLoopCopies;
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
    // (kStaticDefaultDsReadPerCap == kGfx1250Config.dsReadPerCap). Not an
    // HWModel fact — scheduling ratios live in CDNA5Config.
    const int perCapCeiling = kStaticDefaultDsReadPerCap;
    out.dsReadPerCap =
        wmma <= 128 ? perCapCeiling : std::min(perCapCeiling, ceilDivPositive(ds, wmma));
    out.dsReadPerCapSource = SchedulingKnobSource::Policy;

    // Independent of the dsReadPerCap knob above: recompute the same capped
    // ceil ratio, then (firstWmmaLatency / perCap) * queueDepth, floored at
    // the arch's static readThrottleLatency (72 on gfx1250; queueDepth is 16).
    const int perCapForThrottle = perCapForThrottleOf(ds, wmma);
    const int queueDepth = std::max(1, hw.lds.readQueueDepth);
    const int throttleFloor =
        hw.lds.readThrottleLatency > 0 ? hw.lds.readThrottleLatency : 4 * queueDepth;
    const int firstWmmaLatency = std::max(0, features.stats.firstWmmaLatencyCycles);
    const int computedThrottle = (firstWmmaLatency / perCapForThrottle) * queueDepth;
    out.dsReadThrottleLatency = std::max(throttleFloor, computedThrottle);
    out.dsReadThrottleLatencySource = SchedulingKnobSource::Policy;

    // Diagnostic only. Logged by logResolvedSchedulingKnobs; not folded into
    // dsReadThrottleLatency above.
    out.optimisticDsReadThrottleLatency =
        estimateOptimisticDsReadThrottle(
            features.stats.sumWmmaLatencyCycles, features.unrollLoopCopies, firstWmmaLatency,
            features.stats.firstDsLoadLatencyCycles, queueDepth, perCapForThrottle, ds)
            .latency;

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

    if (overrides.dsReadPerCap.has_value()) {
        out.dsReadPerCap = *overrides.dsReadPerCap;
        out.dsReadPerCapSource = SchedulingKnobSource::User;
    } else if (!features.stats.degenerate()) {
        out.dsReadPerCap = proposed.dsReadPerCap;
        out.dsReadPerCapSource = SchedulingKnobSource::Policy;
    } else {
        out.dsReadPerCap = defaults.dsReadPerCap;
        out.dsReadPerCapSource = SchedulingKnobSource::StaticDefault;
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

    // Diagnostic. Stays -1 when propose() did not run.
    if (!features.stats.degenerate()) {
        out.optimisticDsReadThrottleLatency = proposed.optimisticDsReadThrottleLatency;
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
    config.dagFeatures.dsReadPerCap = resolved.dsReadPerCap;
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
       << " firstDsLat=" << features.stats.firstDsLoadLatencyCycles
       << " sumWmmaLat=" << features.stats.sumWmmaLatencyCycles
       << " dsReadThrottleLatency=" << resolved.dsReadThrottleLatency << "("
       << schedulingKnobSourceName(resolved.dsReadThrottleLatencySource) << ")"
       << " dsReadPerCap=" << resolved.dsReadPerCap << "("
       << schedulingKnobSourceName(resolved.dsReadPerCapSource) << ")"
       << " rule3SignalLeadCycles=" << resolved.clusterBarrierRule3SignalLeadCycles << "("
       << schedulingKnobSourceName(resolved.clusterBarrierRule3SignalLeadCyclesSource) << ")";
    // optimisticDsReadThrottleLatency is this logger's heuristic recompute.
    // policyOptimisticDsReadThrottleLatency is whatever propose() stored.
    // A custom policy can disagree; applyResolvedSchedulingKnobs ignores both.
    const OptimisticDsReadThrottle optimistic =
        estimateOptimisticDsReadThrottle(features, hwModelForArch(features.arch));
    const int policyOptimistic = resolved.optimisticDsReadThrottleLatency;
    if (!optimistic.defined && features.stats.degenerate()) {
        os << " optimisticDsReadThrottleLatency=n/a"
           << " policyOptimisticDsReadThrottleLatency=" << policyOptimistic
           << " optimisticThrottleMatchesPolicy=n/a";
    } else {
        const bool matches = policyOptimistic == optimistic.latency;
        os << " optimisticDsReadThrottleLatency=" << optimistic.latency
           << " policyOptimisticDsReadThrottleLatency=" << policyOptimistic
           << " optimisticThrottleMatchesPolicy=" << (matches ? 1 : 0)
           << " unrollLoopCopies=" << optimistic.unrollLoopCopies
           << " dsIssueSpace=" << optimistic.dsIssueSpace
           << " basicWmmaUsage=" << optimistic.basicWmmaUsage
           << " throttleSpace=" << optimistic.throttleSpace
           << " remainingDs=" << optimistic.remainingDs << " cyclePerDs=" << optimistic.cyclePerDs;
    }
    os << "\n";
}

void logResolvedSchedulingKnobsIfDebug(std::string_view moduleName,
                                       const SchedulingFeatures& features,
                                       const ResolvedSchedulingKnobs& resolved) {
    PASS_DEBUG(logResolvedSchedulingKnobs(std::cerr, moduleName, features, resolved));
}

}  // namespace stinkytofu
