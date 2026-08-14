// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stinkytofu/hardware/HWModel.hpp"

#include <algorithm>

#include "stinkytofu/transforms/asm/dag/HazardRules.hpp"

namespace stinkytofu {
namespace {

// The models are defined here, out of line, rather than as inline objects in the
// header. STINKYTOFU_EXPORT is empty for consumers on Linux (see Export.hpp), so a
// header-inline object would get a distinct address in libstinkytofu.so and in each
// consumer (stinkytofu-opt, the Python module). PassContext caches a *pointer* to
// the model, which makes address identity load-bearing. One definition in one TU,
// reached through an exported function, keeps that sound.

constexpr HWModel kGfx1250Model = {
    .lds =
        {
            .readQueueDepth = 16,
            .readDrainLatency = 72,
            .readThrottleLatency = 72,
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
};

// gfx1250v0: starts from the gfx1250 values. Kept as its own object so those
// numbers can diverge without touching gfx1250.
// TODO(tuning): fill in gfx1250v0's real queue depths / latencies, and point
// hazards at a gfx1250v0 rule table if its cycles or rule set diverge.
constexpr HWModel kGfx1250v0Model = kGfx1250Model;

}  // namespace

int computeDynamicDrainLatency(const HWModel& hw, int matchingDsLoadCount, int targetDSLoadLatency,
                               int rawNumWaves) {
    // Keep these local: they only define this function's modeled input range.
    constexpr int kMinModeledWaves = 1;
    constexpr int kMaxModeledWaves = 4;
    const int numWaves = std::clamp(rawNumWaves, kMinModeledWaves, kMaxModeledWaves);
    const int queueDepth = hw.lds.readQueueDepth;
    // A zero queue depth means the arch has no modeled LDS return queue (the other
    // consumers of lds.* already treat it as inert), and a lone load has nothing
    // queued behind it. Either way only the load's own latency applies.
    if (queueDepth <= 0 || matchingDsLoadCount <= 1) return targetDSLoadLatency;

    // Up to the queue depth every load is in flight at once, so the burst costs one
    // load's latency plus the per-wave issue spacing of the loads ahead of it.
    if (matchingDsLoadCount <= queueDepth)
        return targetDSLoadLatency + (matchingDsLoadCount - 1) * numWaves;

    // Past the depth the queue is full, so the overflow issues at half rate.
    return targetDSLoadLatency + (queueDepth - 1) * numWaves +
           (matchingDsLoadCount - queueDepth) * numWaves / 2;
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
