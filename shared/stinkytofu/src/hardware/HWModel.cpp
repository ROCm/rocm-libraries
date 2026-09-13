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
            // DS load overflow throughput, per WGP. B128 is half the default.
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

}  // namespace

int computeDynamicDrainLatency(const HWModel& hw, DsReadKind kind, int matchingDsLoadCount,
                               int targetDSLoadLatency, int rawNumWaves) {
    // Keep these local: they only define this function's modeled input range.
    constexpr int kMinModeledWaves = 1;
    constexpr int kMaxModeledWaves = 4;
    const int numWaves = std::clamp(rawNumWaves, kMinModeledWaves, kMaxModeledWaves);
    const int queueDepth = hw.lds.readQueueDepth;
    const auto& maxDrain = hw.lds.dsLoadMaxDrainLatency;
    // Unclassified DS read types use the B32 timing model.
    int maxDrainLatency = maxDrain.b32;
    switch (kind) {
        case DsReadKind::B32:
            maxDrainLatency = maxDrain.b32;
            break;
        case DsReadKind::B64:
            maxDrainLatency = maxDrain.b64;
            break;
        case DsReadKind::B128:
            maxDrainLatency = maxDrain.b128;
            break;
        case DsReadKind::Tr8B64:
            maxDrainLatency = maxDrain.tr8B64;
            break;
        case DsReadKind::Tr16B128:
            maxDrainLatency = maxDrain.tr16B128;
            break;
        case DsReadKind::Unknown:
            break;
    }
    const auto capDrainLatency = [maxDrainLatency](int latency) {
        return maxDrainLatency > 0 ? std::min(latency, maxDrainLatency) : latency;
    };

    // A zero queue depth means the arch has no modeled LDS return queue (the
    // other consumers of lds.* already treat it as inert), and a lone load has
    // nothing queued behind it. Either way only the load's own latency applies.
    if (queueDepth <= 0 || matchingDsLoadCount <= 1) return capDrainLatency(targetDSLoadLatency);

    // Up to the queue depth every load is in flight at once, so the burst costs
    // one load's latency plus the per-wave issue spacing of the loads ahead of
    // it.
    if (matchingDsLoadCount <= queueDepth)
        return capDrainLatency(targetDSLoadLatency + (matchingDsLoadCount - 1) * numWaves);

    const int dsLoadThroughput = kind == DsReadKind::B128 ? hw.lds.dsLoadThroughput.b128
                                                          : hw.lds.dsLoadThroughput.defaultValue;
    // Past the depth the queue is full. Divide by throughput so B128 (half of
    // the default rate) pays a larger overflow term than other DS read kinds.
    return capDrainLatency(targetDSLoadLatency + (queueDepth - 1) * numWaves +
                           (matchingDsLoadCount - queueDepth) * numWaves /
                               std::max(1, dsLoadThroughput));
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
