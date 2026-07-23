// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

// Shared wait-count dataflow helpers used by WaitDataflow (insertion) and
// WaitCntValidator (validation). Keeps classification, queue maintenance, and
// per-consumer required-wait computation in one place.

#include <array>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "stinkytofu/transforms/asm/waitcnt/WaitDataflow.hpp"

namespace stinkytofu {
class BasicBlock;
struct StinkyInstruction;

namespace waitcnt {
namespace utils {

constexpr size_t kMaxInFlight = 64;

bool isPhi(const StinkyInstruction& inst);
bool isTensorAnchorInst(const StinkyInstruction& inst);
bool isLdsWriterAnchorInst(const StinkyInstruction& inst);
bool isOnSamePipeline(const StinkyInstruction& a, const StinkyInstruction& b);
bool hasTokenOverlap(const std::vector<int>& a, const std::vector<int>& b);

void trimQueues(std::vector<PerPredQueue>& qs, int keep);

/// Append a local in-block memop to every per-pred queue. Returns true if the
/// queue cap dropped an op (issued past the hardware in-flight window).
bool appendToAllPaths(std::vector<PerPredQueue>& qs, StinkyInstruction* op);

int phiCurrentQueueWait(StinkyInstruction* phi, CounterKind c, const DataflowState& state,
                        std::unordered_set<StinkyInstruction*>& seen);

void computeRequiredWaits(StinkyInstruction* inst, const DataflowState& state,
                          const std::array<RawWaitPredicate, CK_Count>& rawNeedsWait,
                          int required[CK_Count]);

/// Seed `entry.queues` from each predecessor's exit queues (one PerPredQueue
/// per pred, deduplicating identical (pred, ops) queues). Does not touch
/// phiSummaries.
void seedQueuesFromPredecessors(
    BasicBlock& bb, const std::unordered_map<const BasicBlock*, DataflowState>& exitState,
    DataflowState& entry);

}  // namespace utils
}  // namespace waitcnt
}  // namespace stinkytofu
