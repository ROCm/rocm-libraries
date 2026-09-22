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
            // Fallback when HwInstDesc::dsThroughput / dsMaxDrain are 0.
            .dsLoadDefaultThroughput = 4,
            .dsLoadDefaultMaxDrain = 120,
            // One ds issue pipe per 2 waves: the ISA's 1-cycle ds issue holds
            // at 1 wave, but 4 waves run as 2-2 pairs and each wave's issues
            // cost 2. Independent of dagFeatures.dsReadPerWmma, which stays a
            // separately tuned ceiling.
            .wavesPerDsIssuePipe = 2,
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

int capDrainLatency(int latency, int maxDrainLatency) {
    return maxDrainLatency > 0 ? std::min(latency, maxDrainLatency) : latency;
}

}  // namespace

int computeDynamicDrainLatency(const HWModel& hw, int matchingDsLoadCount, int targetDSLoadLatency,
                               int dsLoadThroughput, int maxDrainLatency, int rawNumWaves) {
    const int numWaves = std::clamp(rawNumWaves, kMinModeledWaves, kMaxModeledWaves);
    const int queueDepth = hw.lds.readQueueDepth;
    const int throughput = std::max(1, dsLoadThroughput);

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

    // Past the depth the queue is full. Divide by throughput so half-rate DS
    // loads (smaller throughput) pay a larger overflow term.
    return capDrainLatency(targetDSLoadLatency + (queueDepth - 1) * numWaves +
                               (matchingDsLoadCount - queueDepth) * numWaves / throughput,
                           maxDrainLatency);
}

int dsIssueCyclesForWaves(const HWModel& hw, int issueCycles, int numWaves) {
    const int share = hw.lds.wavesPerDsIssuePipe;
    // GemmTileConfig::NumWaves defaults to 1, so an unconfigured caller lands on
    // single-wave behaviour naturally. The <= 0 guard is for a caller that
    // explicitly passes a nonsense count.
    if (issueCycles <= 0 || share <= 1 || numWaves <= 0) return issueCycles;
    // Waves pair onto a pipe as soon as there are enough to fill one, so the
    // contending count saturates at the share. See the header for the
    // unverified numWaves == 2 case.
    return issueCycles * std::min(numWaves, share);
}

int computeDynamicDrainLatencyForLoads(const HWModel& hw, std::span<const DsLoadDrainEntry> loads,
                                       int rawNumWaves) {
    if (loads.empty()) return 0;

    const int numWaves = std::clamp(rawNumWaves, kMinModeledWaves, kMaxModeledWaves);
    const int queueDepth = hw.lds.readQueueDepth;
    const int count = static_cast<int>(loads.size());
    const int targetLatency = loads.back().latency;

    // Cap with the largest maxDrain among the whole burst, not just the last
    // load.
    int maxDrainLatency = 0;
    long long throughputSum = 0;
    for (const DsLoadDrainEntry& load : loads) {
        maxDrainLatency = std::max(maxDrainLatency, load.maxDrain);
        throughputSum += std::max(1, load.throughput);
    }

    if (queueDepth <= 0 || count <= 1) return capDrainLatency(targetLatency, maxDrainLatency);

    if (count <= queueDepth)
        return capDrainLatency(targetLatency + (count - 1) * numWaves, maxDrainLatency);

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
