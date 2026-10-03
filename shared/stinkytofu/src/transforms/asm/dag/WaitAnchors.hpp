/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ReadyQueue.hpp"
#include "RegionDAG.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/transforms/asm/ExecMaskGrouping.hpp"
#include "stinkytofu/transforms/asm/waitcnt/WaitDataflow.hpp"
#include "stinkytofu/transforms/asm/waitcnt/WaitPlan.hpp"

namespace stinkytofu {
namespace dag {

// ---------------------------------------------------------------------------
// Wait anchors, segment boundaries, and the ordering a repair pass has to add
// to a register DAG itself.
//
// These belong to the problem of re-scheduling IR that already has its final
// waits, not to any one pick policy.
// ---------------------------------------------------------------------------

struct WaitAnchorInfo {
    StinkyInstruction* anchor = nullptr;
    std::vector<StinkyInstruction*> waits;
    waitcnt::WaitCountSpec spec;
};

using WaitAnchorMap = std::unordered_map<StinkyInstruction*, WaitAnchorInfo>;

inline bool counterApplies(waitcnt::CounterKind kind, const waitcnt::WaitCountSpec& spec) {
    using waitcnt::WaitCountSpec;
    switch (kind) {
        case waitcnt::CK_DS:
            return spec.dsCount != waitcnt::WaitCountSpec::kUnused;
        case waitcnt::CK_Load:
            return spec.loadCount != waitcnt::WaitCountSpec::kUnused;
        case waitcnt::CK_KM:
            return spec.kmCount != waitcnt::WaitCountSpec::kUnused;
        case waitcnt::CK_Tensor:
            return spec.tensorCount != waitcnt::WaitCountSpec::kUnused;
        case waitcnt::CK_Async:
            return spec.asyncCount != waitcnt::WaitCountSpec::kUnused;
        default:
            return false;
    }
}

/// For each instruction \p shouldPin accepts, look forward for the next
/// instruction \p isBarrier accepts and add an edge between them. The edge
/// forces the first instruction to be scheduled before the second.
///
/// Edges only run forward, from a lower index to a higher one, so they can never
/// form a cycle. When both predicates accept the same instructions, each scan
/// stops at the next accepted one and the total work is linear.
///
/// The predicates take a pointer because one of them looks the instruction up in
/// a \ref WaitAnchorMap, which is keyed on pointers.
template <typename PinPred, typename BarrierPred>
inline void addEdgesToFirstFollowing(RegionDAG& dag,
                                     const std::vector<StinkyInstruction*>& instructions,
                                     PinPred shouldPin, BarrierPred isBarrier) {
    for (unsigned i = 0; i < instructions.size(); ++i) {
        if (!shouldPin(instructions[i])) continue;
        for (unsigned j = i + 1; j < instructions.size(); ++j) {
            if (!isBarrier(instructions[j])) continue;
            addEdgeById(&dag.nodes[i], &dag.nodes[j], dag.graph);
            break;
        }
    }
}

/// Preserve the meaning of final wait immediates by ordering each counter's
/// producers and wait anchors in their original sequence.
///
/// Every event of a counter is both pinned and a barrier for that counter, so
/// the edges chain each event to the next one and the whole chain stays in input
/// order. The skeleton chain below orders these events too; this one states the
/// correctness requirement on its own, whatever the skeleton covers.
inline void addCounterOrderEdges(RegionDAG& dag,
                                 const std::vector<StinkyInstruction*>& instructions,
                                 const WaitAnchorMap& anchors) {
    using waitcnt::CK_Count;
    using waitcnt::CounterKind;

    for (int ck = 0; ck < CK_Count; ++ck) {
        const auto kind = static_cast<CounterKind>(ck);
        auto isCounterEvent = [&](StinkyInstruction* inst) {
            if (waitcnt::classifyMemOp(*inst) == kind) return true;
            auto anchor = anchors.find(inst);
            return anchor != anchors.end() && counterApplies(kind, anchor->second.spec);
        };
        addEdgesToFirstFollowing(dag, instructions, isCounterEvent, isCounterEvent);
    }
}

/// Hints whose only value is the lead time they get: no waitcnt counter and no
/// destination register, so nothing else in the DAG orders them.
///
/// The null check matters: StinkyInstruction::is() dereferences its descriptor
/// unchecked, which is why hasSideEffect() tests the pointer first too.
inline bool isPrefetchHint(const StinkyInstruction& inst) {
    if (inst.getHwInstDesc() == nullptr) return false;
    return isGlobalPrefetch(inst);
}

/// Matrix ops, memory ops and prefetch hints: the instructions a repair keeps
/// where the scheduler put them.
inline bool isMatrixMemorySkeleton(const StinkyInstruction& inst) {
    return isMatrixInstruction(inst) || isPrefetchHint(inst) ||
           waitcnt::classifyMemOp(inst) != waitcnt::CK_Count;
}

/// Keep the skeleton in the order the scheduler left it, so only ALU work moves.
///
/// The scheduler places memory ops against the matrix ops that consume them
/// with a model a repair does not reproduce: queue depth, drain latency and
/// per-matrix-op affinity, decided with the whole region in view. Prefetch hints
/// join the chain because their value is the lead time the scheduler gave them,
/// which moving them either way throws away.
///
/// Every member is both pinned and a barrier, so consecutive members are linked
/// and the whole sequence holds its input order.
inline void addMatrixMemoryOrderEdges(RegionDAG& dag,
                                      const std::vector<StinkyInstruction*>& instructions) {
    auto isSkeleton = [](StinkyInstruction* inst) { return isMatrixMemorySkeleton(*inst); };
    addEdgesToFirstFollowing(dag, instructions, isSkeleton, isSkeleton);
}

/// All ordering a repair adds that register dependencies do not give it: counter
/// order, which keeps every wait immediate correct, and skeleton order, which
/// keeps the scheduler's memory and matrix placement.
inline void addSyntheticOrderEdges(RegionDAG& dag,
                                   const std::vector<StinkyInstruction*>& instructions,
                                   const WaitAnchorMap& anchors) {
    addCounterOrderEdges(dag, instructions, anchors);
    addMatrixMemoryOrderEdges(dag, instructions);
}

inline bool isAnyWaitCnt(const StinkyInstruction& inst) {
    return isWaitCnt(inst) || inst.is(InstFlag::IF_WaitTensorCnt);
}

inline waitcnt::WaitCountSpec decodeWaitSpec(const StinkyInstruction& wait) {
    waitcnt::WaitCountSpec spec;
    if (const auto* data = wait.getModifier<SWaitCntData>()) {
        if (data->dlcnt >= 0) spec.dsCount = data->dlcnt;
        if (data->vlcnt >= 0) spec.loadCount = data->vlcnt;
        if (data->kmcnt >= 0) spec.kmCount = data->kmcnt;
    }
    if (const auto* tdata = wait.getModifier<SWaitTensorCntData>()) {
        if (tdata->tlcnt >= 0) spec.tensorCount = static_cast<unsigned char>(tdata->tlcnt);
    }
    if (const auto* adata = wait.getModifier<SWaitAsyncCntData>()) {
        if (adata->asynccnt >= 0) spec.asyncCount = static_cast<unsigned char>(adata->asynccnt);
    }
    return spec;
}

inline waitcnt::WaitCountSpec mergeWaitSpecs(const waitcnt::WaitCountSpec& a,
                                             const waitcnt::WaitCountSpec& b) {
    waitcnt::WaitCountSpec out = a;
    if (b.dsCount != waitcnt::WaitCountSpec::kUnused) out.dsCount = b.dsCount;
    if (b.loadCount != waitcnt::WaitCountSpec::kUnused) out.loadCount = b.loadCount;
    if (b.kmCount != waitcnt::WaitCountSpec::kUnused) out.kmCount = b.kmCount;
    if (b.tensorCount != waitcnt::WaitCountSpec::kUnused) out.tensorCount = b.tensorCount;
    if (b.asyncCount != waitcnt::WaitCountSpec::kUnused) out.asyncCount = b.asyncCount;
    return out;
}

inline void discoverWaitAnchorsInRun(const std::vector<StinkyInstruction*>& seq,
                                     WaitAnchorMap& anchors) {
    for (size_t i = 0; i < seq.size(); ++i) {
        if (!isAnyWaitCnt(*seq[i])) continue;

        size_t waitStart = i;
        size_t waitEnd = waitStart + 1;
        while (waitEnd < seq.size() && isAnyWaitCnt(*seq[waitEnd])) ++waitEnd;

        if (waitEnd >= seq.size() || !isMatrixInstruction(*seq[waitEnd])) {
            i = waitEnd;
            continue;
        }

        WaitAnchorInfo info;
        info.anchor = seq[waitEnd];
        waitcnt::WaitCountSpec combined;
        for (size_t w = waitStart; w < waitEnd; ++w) {
            info.waits.push_back(seq[w]);
            combined = mergeWaitSpecs(combined, decodeWaitSpec(*seq[w]));
        }
        info.spec = combined;
        anchors[info.anchor] = std::move(info);
        i = waitEnd;
    }
}

/// Map each matrix op preceded directly by a group of waits to that group. Only
/// consecutive StinkyTofu instructions are considered, so a group never spans
/// other IR.
inline WaitAnchorMap discoverWaitAnchors(BasicBlock& bb) {
    WaitAnchorMap anchors;
    std::vector<StinkyInstruction*> run;
    run.reserve(bb.size());

    for (IRBase& ir : bb) {
        if (ir.getType() != IRBase::IRType::StinkyTofu) {
            discoverWaitAnchorsInRun(run, anchors);
            run.clear();
            continue;
        }
        run.push_back(cast<StinkyInstruction>(&ir));
    }
    discoverWaitAnchorsInRun(run, anchors);
    return anchors;
}

inline std::unordered_set<StinkyInstruction*> collectAttachedWaits(const WaitAnchorMap& anchors) {
    std::unordered_set<StinkyInstruction*> attached;
    for (const auto& [anchor, info] : anchors) {
        (void)anchor;
        for (StinkyInstruction* wait : info.waits) attached.insert(wait);
    }
    return attached;
}

inline bool isHardBoundary(const StinkyInstruction& inst,
                           const std::unordered_set<StinkyInstruction*>& attachedWaits) {
    if (isLabel(inst)) return true;
    if (isAnyWaitCnt(inst) && attachedWaits.count(const_cast<StinkyInstruction*>(&inst)) == 0)
        return true;
    if (hasSideEffect(inst)) return true;
    if (isExecMaskGroup(inst)) return true;
    return false;
}

}  // namespace dag
}  // namespace stinkytofu
