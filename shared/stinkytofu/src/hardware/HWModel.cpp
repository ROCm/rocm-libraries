// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stinkytofu/hardware/HWModel.hpp"

#include <algorithm>

#include "stinkytofu/transforms/asm/dag/HazardRules.hpp"

namespace stinkytofu {
namespace {

// The models are defined here, out of line, rather than as inline objects in
// the header. STINKYTOFU_EXPORT is empty for consumers on Linux (see
// Export.hpp), so a header-inline object would get a distinct address in
// libstinkytofu.so and in each consumer (stinkytofu-opt, the Python module).
// PassContext caches a *pointer* to the model, which makes address identity
// load-bearing. One definition in one TU, reached through an exported function,
// keeps that sound.

constexpr HWModel kGfx1250Model = {
    .lds =
        {
            .readQueueDepth = 16,
            // 0 => derive barrier-timing drain latency dynamically from
            // matching ds_read count and the latest ds_read's own latency.
            .readDrainLatency = 0,
            .readThrottleLatency = 72,
            // DS load overflow throughput, per WGP. B128 / Tr16B128 are half rate.
            .dsLoadThroughput =
                {
                    .defaultValue = 4,
                    .b128 = 2,  // half of default => larger overflow drain
                },
            // Experimentally measured maximum drain latency, in cycles.
            .dsLoadMaxDrainLatency =
                {
                    .b32 = 120,
                    .b64 = 131,
                    .b128 = 255,
                    .tr8B64 = 135,
                    .tr16B128 = 255,
                },
        },
    .barrier =
        {
            .signalToWaitLatency = 11,
            .jumpOverheadCycles = 6,
        },
    .coexec =
        {
            .transToNonCoreSide = 1,
            .maxSlotBudget = 18,
        },
    .hazards =
        {
            .rules = kCdna5HazardRules,
            .numRules = kNumCdna5HazardRules,
        },
    .delayAlu =
        {
            .valuDepth = 5,
            .transDepth = 4,
            .saluCycleMax = 4,
        },
    .counters =
        {
            .hasSplitLoadStoreCnt = true,
            .hasSplitStoreCntAsyncCnt = true,  // only async stores on this arch
        },
};

// gfx1250v0: starts from the gfx1250 values. Kept as its own object so those
// numbers can diverge without touching gfx1250.
// TODO(tuning): fill in gfx1250v0's real queue depths / latencies, and point
// hazards at a gfx1250v0 rule table if its cycles or rule set diverge.
constexpr HWModel kGfx1250v0Model = kGfx1250Model;

constexpr int kMinModeledWaves = 1;
constexpr int kMaxModeledWaves = 4;

bool isB128ClassDsRead(DsReadKind kind) {
    return kind == DsReadKind::B128 || kind == DsReadKind::Tr16B128;
}

int dsLoadThroughputForKind(const HWModel& hw, DsReadKind kind) {
    return isB128ClassDsRead(kind) ? hw.lds.dsLoadThroughput.b128
                                   : hw.lds.dsLoadThroughput.defaultValue;
}

int maxDrainLatencyForKind(const HWModel& hw, DsReadKind kind) {
    const auto& maxDrain = hw.lds.dsLoadMaxDrainLatency;
    switch (kind) {
        case DsReadKind::B32:
            return maxDrain.b32;
        case DsReadKind::B64:
            return maxDrain.b64;
        case DsReadKind::B128:
            return maxDrain.b128;
        case DsReadKind::Tr8B64:
            return maxDrain.tr8B64;
        case DsReadKind::Tr16B128:
            return maxDrain.tr16B128;
        case DsReadKind::Unknown:
            return maxDrain.b32;
    }
    return maxDrain.b32;
}

int capDrainLatency(int latency, int maxDrainLatency) {
    return maxDrainLatency > 0 ? std::min(latency, maxDrainLatency) : latency;
}

}  // namespace

int computeDynamicDrainLatency(const HWModel& hw, DsReadKind kind, int matchingDsLoadCount,
                               int targetDSLoadLatency, int rawNumWaves) {
    const int numWaves = std::clamp(rawNumWaves, kMinModeledWaves, kMaxModeledWaves);
    const int queueDepth = hw.lds.readQueueDepth;
    const int maxDrainLatency = maxDrainLatencyForKind(hw, kind);

    // A zero queue depth means the arch has no modeled LDS return queue (the
    // other consumers of lds.* already treat it as inert), and a lone load has
    // nothing queued behind it. Either way only the load's own latency applies.
    if (queueDepth <= 0 || matchingDsLoadCount <= 1)
        return capDrainLatency(targetDSLoadLatency, maxDrainLatency);

    // Up to the queue depth every load is in flight at once, so the burst costs
    // one load's latency plus the per-wave issue spacing of the loads ahead of
    // it.
    if (matchingDsLoadCount <= queueDepth)
        return capDrainLatency(targetDSLoadLatency + (matchingDsLoadCount - 1) * numWaves,
                               maxDrainLatency);

    const int dsLoadThroughput = std::max(1, dsLoadThroughputForKind(hw, kind));
    // Past the depth the queue is full. Divide by throughput so B128 /
    // Tr16B128 (half of the default rate) pay a larger overflow term than
    // other DS read kinds.
    return capDrainLatency(targetDSLoadLatency + (queueDepth - 1) * numWaves +
                               (matchingDsLoadCount - queueDepth) * numWaves / dsLoadThroughput,
                           maxDrainLatency);
}

int computeDynamicDrainLatencyForLoads(const HWModel& hw, std::span<const DsLoadDrainEntry> loads,
                                       int rawNumWaves) {
    if (loads.empty()) return 0;

    const int numWaves = std::clamp(rawNumWaves, kMinModeledWaves, kMaxModeledWaves);
    const int queueDepth = hw.lds.readQueueDepth;
    const int count = static_cast<int>(loads.size());
    const int targetLatency = loads.back().latency;

    // Cap with the largest per-kind max among the whole burst, not just the
    // last load — a mixed burst that includes B128 should still be allowed up
    // to the B128 experimental ceiling.
    int maxDrainLatency = 0;
    long long throughputSum = 0;
    for (const DsLoadDrainEntry& load : loads) {
        maxDrainLatency = std::max(maxDrainLatency, maxDrainLatencyForKind(hw, load.kind));
        throughputSum += std::max(1, dsLoadThroughputForKind(hw, load.kind));
    }

    if (queueDepth <= 0 || count <= 1) return capDrainLatency(targetLatency, maxDrainLatency);

    if (count <= queueDepth)
        return capDrainLatency(targetLatency + (count - 1) * numWaves, maxDrainLatency);

    // Count-weighted average of per-load issue rates. Homogeneous bursts reduce
    // to the same throughput computeDynamicDrainLatency() would pick for that
    // kind.
    const int dsLoadThroughput =
        static_cast<int>(std::max<long long>(1, throughputSum / std::max(1, count)));
    return capDrainLatency(targetLatency + (queueDepth - 1) * numWaves +
                               (count - queueDepth) * numWaves / dsLoadThroughput,
                           maxDrainLatency);
}

const HWModel& hwModelForArch(const std::array<int, 3>& arch) {
    switch (archKey(arch)) {
        case kArchKeyGfx1250v0:
            return kGfx1250v0Model;
        case kArchKeyGfx1250:
        default:
            return kGfx1250Model;
    }
}

}  // namespace stinkytofu
