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

// ----------------------------------------------------------------------------
// StinkyWmmaVgprReorderPass
//
// Analysis pass that determines the optimal wmma reordering to reduce VGPR
// usage in N-buffered GEMM kernels.
//
// Problem: when the pool-varying operand (A or B depending on the kernel) is
// the outer loop dimension, its registers stay live across all iterations of
// the inner dimension, preventing cross-pool register aliasing.
//
// Solution: reorder each pool so the pool-varying operand's groups are
// contiguous. Its per-pool liveness intervals then no longer overlap across
// pools, allowing later pools' registers to be aliased onto the first pool's.
//
// detectABIndices identifies which src operand is pool-varying (labelled A)
// vs pool-shared (labelled B) from register group intersection across pools.
//
// Output (WmmaReorderAnalysisResult):
//   desiredWmmaOrder  — permutation of wmma pointers for a downstream reorder pass
//   replacements      — per-operand rewrite map for a downstream renaming pass
//   totalVgprSaved    — summary VGPR count
//
// Read-only: never mutates any instruction or register operand.
// ----------------------------------------------------------------------------

#include "stinkytofu/transforms/asm/StinkyWmmaVgprReorderPass.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <map>
#include <set>
#include <unordered_map>
#include <utility>

#define DEBUG_TYPE "StinkyWmmaVgprReorderPass"

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

namespace stinkytofu {
namespace {

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

RegGroup toRegGroup(const StinkyRegister& r) {
    assert(r.isRegister());
    return RegGroup{r.reg.idx, r.reg.num};
}

bool regOverlapsGroup(const StinkyRegister& r, const RegGroup& g) {
    if (!r.isRegister() || r.reg.type != RegType::V) return false;
    const uint32_t rEnd = r.reg.idx + r.reg.num;
    const uint32_t gEnd = g.base + g.size;
    return r.reg.idx < gEnd && g.base < rEnd;
}

/// Group all XDL-WMMA instructions in @p bb by their WmmaPoolData pool index.
/// Returns one inner vector per pool, in ascending pool-index order.
/// Returns empty if there are no wmma instructions OR if any wmma is missing a
/// WmmaPoolData modifier — a partial tag is a sign of a mis-configured pipeline
/// and the pass must not proceed on incomplete information.
std::vector<std::vector<WmmaNode>> groupWmmaByPool(const BasicBlock& bb) {
    std::map<uint32_t, std::vector<WmmaNode>> byPool;
    std::vector<const StinkyInstruction*> untagged;

    for (auto it = bb.begin(); it != bb.end(); ++it) {
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (!inst || !isXDLWMMA(*inst)) continue;
        const auto* tag = inst->getModifier<WmmaPoolData>();
        if (!tag) {
            untagged.push_back(inst);
            continue;
        }
        byPool[tag->poolIndex].push_back(WmmaNode{
            const_cast<StinkyInstruction*>(inst),
            toRegGroup(inst->getSrcReg(0)),
            toRegGroup(inst->getSrcReg(1)),
            toRegGroup(inst->getDestReg(0)),
        });
    }

    if (!untagged.empty()) {
        PASS_DEBUG({
            std::cerr << "[WmmaVgprReorderPass] " << untagged.size()
                      << " wmma instruction(s) missing WmmaPoolData modifier"
                         " — skipping block:\n";
            for (const auto* inst : untagged)
                std::cerr << "  " << inst->getHwInstDesc()->mnemonic << "\n";
        });
        return {};
    }

    std::vector<std::vector<WmmaNode>> pools;
    pools.reserve(byPool.size());
    for (auto& [idx, nodes] : byPool) pools.push_back(std::move(nodes));
    return pools;
}

/// Determine which src operand index is A (pool-varying) vs B (pool-shared).
/// The A operand has different register groups in different pools; B is shared.
/// Returns {aIdx, bIdx}.
std::pair<unsigned, unsigned> detectABIndices(const std::vector<std::vector<WmmaNode>>& pools) {
    std::set<RegGroup> pool0Src0;
    for (const WmmaNode& n : pools[0]) pool0Src0.insert(n.aGroup);  // aGroup = src0 tentatively
    for (size_t p = 1; p < pools.size(); ++p)
        for (const WmmaNode& n : pools[p])
            if (pool0Src0.count(n.aGroup)) return {1u, 0u};  // src0 shared → src0=B, src1=A
    return {0u, 1u};
}

/// Reorder @p pool so the chosen primary operand's groups are contiguous, minimizing
/// its liveness. @p primaryIsA=true groups by aGroup first; false groups by bGroup first.
/// Returns empty if (aGroup, bGroup) pairs don't form a complete grid.
std::vector<WmmaNode> reorderContiguously(const std::vector<WmmaNode>& pool, bool primaryIsA) {
    auto primary = [&](const WmmaNode& n) { return primaryIsA ? n.aGroup : n.bGroup; };
    auto secondary = [&](const WmmaNode& n) { return primaryIsA ? n.bGroup : n.aGroup; };

    std::set<RegGroup> pSet, sSet;
    std::map<std::pair<RegGroup, RegGroup>, WmmaNode> lookup;
    for (const WmmaNode& n : pool) {
        pSet.insert(primary(n));
        sSet.insert(secondary(n));
        lookup[{primary(n), secondary(n)}] = n;
    }

    std::vector<WmmaNode> result;
    result.reserve(pool.size());
    for (const RegGroup& pg : pSet)
        for (const RegGroup& sg : sSet)
            if (auto it = lookup.find({pg, sg}); it != lookup.end()) result.push_back(it->second);

    return (result.size() == pool.size()) ? result : std::vector<WmmaNode>{};
}

/// Verify that reordering @p originalPool's wmma instructions as @p reorderedPool
/// does not violate data dependencies created by non-wmma instructions in @p bb.
///
/// Two hazards checked:
///  - RAW: a non-wmma instruction writes a register that a pool wmma reads as src.
///    The wmma must not be moved before that writer.
///  - WAR: a non-wmma instruction reads the C-tile that a pool wmma writes as dest.
///    The wmma must not be moved after that reader.
///  - Hard barriers (s_barrier, s_waitcnt): no wmma may cross them; the original
///    relative ordering of wmma instructions around each barrier is preserved.
///
/// Wmma-to-wmma C-tile hazards are not checked here — the caller guarantees that
/// each wmma in a pool has a unique C tile (standard GEMM layout).
bool isPoolReorderSafe(const BasicBlock& bb, const std::vector<WmmaNode>& originalPool,
                       const std::vector<WmmaNode>& reorderedPool) {
    // Build reordered rank map and original rank map.
    std::map<const StinkyInstruction*, unsigned> reorderedRank, originalRank;
    for (unsigned i = 0; i < reorderedPool.size(); ++i) reorderedRank[reorderedPool[i].inst] = i;
    for (unsigned i = 0; i < originalPool.size(); ++i) originalRank[originalPool[i].inst] = i;

    std::set<const StinkyInstruction*> poolInsts;
    for (const WmmaNode& n : originalPool) poolInsts.insert(n.inst);

    // passedCount = number of pool wmma seen so far in BB order.
    // Non-wmma instructions at passedCount=k sit between original wmma[k-1] and wmma[k].
    // In any safe reordering, wmma that appear after such an instruction in original
    // order must still appear after it — i.e., at reordered rank >= k.
    unsigned passedCount = 0;

    for (auto it = bb.begin(); it != bb.end(); ++it) {
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (!inst) continue;

        if (poolInsts.count(inst)) {
            ++passedCount;
            continue;
        }
        if (isXDLWMMA(*inst)) continue;  // different pool — handled by its own check

        // Hard barrier: s_barrier or s_waitcnt. No pool wmma may cross it.
        // Every wmma whose original rank is < passedCount must also have reordered
        // rank < passedCount, and vice versa.
        if (isBarrier(*inst) || inst->is(InstFlag::IF_WaitCnt)) {
            for (const WmmaNode& n : originalPool) {
                bool origBefore = originalRank.at(n.inst) < passedCount;
                bool reordBefore = reorderedRank.at(n.inst) < passedCount;
                if (origBefore != reordBefore) return false;
            }
            continue;
        }

        // RAW: inst writes R → any pool wmma reading R must stay at reordered rank >= passedCount.
        for (unsigned d = 0; d < inst->getNumDestRegs(); ++d) {
            const StinkyRegister& destReg = inst->getDestReg(d);
            for (const WmmaNode& n : originalPool) {
                if (regOverlapsGroup(destReg, n.aGroup) || regOverlapsGroup(destReg, n.bGroup)) {
                    if (reorderedRank.at(n.inst) < passedCount) return false;
                }
            }
        }

        // WAR: inst reads R → any pool wmma writing R (as C dest) must stay at rank < passedCount.
        for (unsigned s = 0; s < inst->getNumSrcRegs(); ++s) {
            const StinkyRegister& srcReg = inst->getSrcReg(s);
            for (const WmmaNode& n : originalPool) {
                if (regOverlapsGroup(srcReg, n.cGroup)) {
                    if (reorderedRank.at(n.inst) >= passedCount) return false;
                }
            }
        }
    }
    return true;
}

/// Build the flat per-operand replacement list: walk every instruction in @p bb
/// and emit an entry for each operand whose VGPR range overlaps an aliasable group.
std::vector<RegReplacement> buildReplacements(const BasicBlock& bb,
                                              const std::vector<AliasCandidate>& aliases) {
    std::vector<RegReplacement> out;
    for (auto it = bb.begin(); it != bb.end(); ++it) {
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (!inst) continue;

        for (const AliasCandidate& alias : aliases) {
            auto patch = [&](const StinkyRegister& r, unsigned idx, bool isSrc) {
                if (!regOverlapsGroup(r, alias.aliasable)) return;
                StinkyRegister newReg = r;
                newReg.reg.idx = alias.canonical.base + (r.reg.idx - alias.aliasable.base);
                out.push_back({const_cast<StinkyInstruction*>(inst), idx, isSrc, r, newReg});
            };
            for (unsigned i = 0; i < inst->getNumSrcRegs(); ++i) patch(inst->getSrcReg(i), i, true);
            for (unsigned i = 0; i < inst->getNumDestRegs(); ++i)
                patch(inst->getDestReg(i), i, false);
        }
    }
    return out;
}

// Per-BB results stored for retrieval via getWmmaReorderResult().
// NOTE: global state; not thread-safe. Acceptable for a draft pass.
std::unordered_map<const BasicBlock*, WmmaReorderAnalysisResult> gResults;

// ─────────────────────────────────────────────────────────────────────────────
// Pass class
// ─────────────────────────────────────────────────────────────────────────────

class StinkyWmmaVgprReorderPassImpl : public StinkyInstPass {
   public:
    StinkyWmmaVgprReorderPassImpl(std::unique_ptr<IRegLivenessAnalysis> liveness,
                                  std::unique_ptr<IWmmaReorderAlgorithm> algorithm)
        : liveness_(std::move(liveness)), algorithm_(std::move(algorithm)) {}

    static char ID;
    const char* getName() const override {
        return "StinkyWmmaVgprReorderPass";
    }
    PassID getPassID() const override {
        return &ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager&) override {
        gResults.clear();
        for (BasicBlock& bb : func) {
            if (!passCtx.shouldProcessBasicBlock(bb)) continue;
            WmmaReorderAnalysisResult res = analyzeBlock(bb);
            PASS_DEBUG(if (res.applicable) std::cerr
                           << "[WmmaVgprReorderPass] " << res.totalVgprSaved << " VGPRs saveable, "
                           << res.replacements.size() << " operand replacements\n";);
            gResults[&bb] = std::move(res);
        }
        return preserveCFGAnalyses();
    }

   private:
    std::unique_ptr<IRegLivenessAnalysis> liveness_;
    std::unique_ptr<IWmmaReorderAlgorithm> algorithm_;

    WmmaReorderAnalysisResult analyzeBlock(const BasicBlock& bb) {
        auto pools = groupWmmaByPool(bb);
        if (pools.size() < 2) return {};

        // Identify which src operand is A (pool-varying) vs B (pool-shared).
        auto [aIdx, bIdx] = detectABIndices(pools);

        // Rebuild WmmaNode A/B groups with the correct src assignment.
        for (auto& pool : pools)
            for (auto& node : pool) {
                node.aGroup = toRegGroup(node.inst->getSrcReg(aIdx));
                node.bGroup = toRegGroup(node.inst->getSrcReg(bIdx));
            }

        // Flatten into a single sequence for liveness computation.
        std::vector<WmmaNode> wmmaSeq;
        for (const auto& pool : pools)
            for (const auto& n : pool) wmmaSeq.push_back(n);

        const auto intervals = liveness_->computeLiveness(bb, wmmaSeq);
        auto [desiredOrder, aliases] = algorithm_->solve(pools, intervals);
        if (aliases.empty()) return {};

        // Validate that the proposed reordering is safe for each pool given
        // non-wmma instructions (ds_loads, barriers, waitcnts) in the block.
        size_t offset = 0;
        for (const auto& pool : pools) {
            std::vector<WmmaNode> reorderedPool(desiredOrder.begin() + offset,
                                                desiredOrder.begin() + offset + pool.size());
            if (!isPoolReorderSafe(bb, pool, reorderedPool)) {
                PASS_DEBUG(std::cerr << "[WmmaVgprReorderPass] reorder unsafe due to"
                                        " non-wmma dependency; skipping block\n");
                return {};
            }
            offset += pool.size();
        }

        WmmaReorderAnalysisResult out;
        out.applicable = true;
        out.replacements = buildReplacements(bb, aliases);
        for (const AliasCandidate& a : aliases) out.totalVgprSaved += a.vgprSaved;
        out.desiredWmmaOrder.reserve(desiredOrder.size());
        for (const WmmaNode& n : desiredOrder) out.desiredWmmaOrder.push_back(n.inst);
        return out;
    }
};

char StinkyWmmaVgprReorderPassImpl::ID = 0;

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// ABI 1 — WmmaIntervalLiveness (out-of-line)
// ─────────────────────────────────────────────────────────────────────────────

std::map<RegGroup, RegInterval> WmmaIntervalLiveness::computeLiveness(
    const BasicBlock& /*bb*/, const std::vector<WmmaNode>& wmmaSeq) const {
    std::map<RegGroup, RegInterval> intervals;
    for (unsigned i = 0; i < wmmaSeq.size(); ++i) {
        for (const RegGroup* g : {&wmmaSeq[i].aGroup, &wmmaSeq[i].bGroup}) {
            auto [it, inserted] = intervals.emplace(*g, RegInterval{i, i});
            if (!inserted) {
                it->second.first = std::min(it->second.first, i);
                it->second.last = std::max(it->second.last, i);
            }
        }
    }
    return intervals;
}

// ─────────────────────────────────────────────────────────────────────────────
// ABI 2 — PoolVaryingReorderAlgorithm (out-of-line)
// ─────────────────────────────────────────────────────────────────────────────

IWmmaReorderAlgorithm::Result PoolVaryingReorderAlgorithm::solve(
    const std::vector<std::vector<WmmaNode>>& pools,
    const std::map<RegGroup, RegInterval>& liveness) const {
    Result result;
    if (pools.size() < 2) return result;

    // detectABIndices (called in analyzeBlock before solve) relabels WmmaNode so
    // aGroup is always the pool-varying operand and bGroup is always pool-shared.
    // Confirm that A liveness is inflated — interval wider than nB means A is
    // currently the outer loop dimension and making it contiguous will save VGPRs.
    std::set<RegGroup> aGroups0, bGroups0;
    for (const WmmaNode& n : pools[0]) {
        aGroups0.insert(n.aGroup);
        bGroups0.insert(n.bGroup);
    }
    const unsigned nB = static_cast<unsigned>(bGroups0.size());

    unsigned aMaxWidth = 0;
    for (const RegGroup& g : aGroups0) {
        auto it = liveness.find(g);
        if (it != liveness.end())
            aMaxWidth = std::max(aMaxWidth, it->second.last - it->second.first + 1);
    }
    if (aMaxWidth <= nB) return result;  // A already tight — no saving

    // A is pool-varying by construction; always make it the primary (contiguous) operand.
    constexpr bool primaryIsA = true;

    // Reorder every pool so the chosen primary operand's groups are contiguous,
    // then collect the primary groups per pool for alias pairing.
    size_t totalWmma = 0;
    for (const auto& pool : pools) totalWmma += pool.size();
    result.desiredOrder.reserve(totalWmma);
    std::vector<std::vector<RegGroup>> poolPrimaryGroups;
    poolPrimaryGroups.reserve(pools.size());

    for (const auto& pool : pools) {
        auto reordered = reorderContiguously(pool, primaryIsA);
        if (reordered.empty()) return Result{};
        for (const WmmaNode& n : reordered) result.desiredOrder.push_back(n);

        std::set<RegGroup> pSet;
        for (const WmmaNode& n : pool) pSet.insert(primaryIsA ? n.aGroup : n.bGroup);
        poolPrimaryGroups.emplace_back(pSet.begin(), pSet.end());  // sorted by base
    }

    // Build alias candidates: pools 1..N-1 aliased onto pool 0.
    // Making the primary operand contiguous ensures its per-pool intervals don't overlap.
    const auto& canonical = poolPrimaryGroups[0];
    for (size_t p = 1; p < poolPrimaryGroups.size(); ++p) {
        const auto& aliasable = poolPrimaryGroups[p];
        for (size_t i = 0; i < std::min(canonical.size(), aliasable.size()); ++i) {
            if (canonical[i].size != aliasable[i].size) continue;
            result.aliases.push_back({canonical[i], aliasable[i], canonical[i].size});
        }
    }
    if (result.aliases.empty()) return Result{};
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

std::unique_ptr<Pass> createStinkyWmmaVgprReorderPass(
    std::unique_ptr<IRegLivenessAnalysis> liveness,
    std::unique_ptr<IWmmaReorderAlgorithm> algorithm) {
    if (!liveness) liveness = std::make_unique<WmmaIntervalLiveness>();
    if (!algorithm) algorithm = std::make_unique<PoolVaryingReorderAlgorithm>();
    return std::make_unique<StinkyWmmaVgprReorderPassImpl>(std::move(liveness),
                                                           std::move(algorithm));
}

const WmmaReorderAnalysisResult* getWmmaReorderResult(const BasicBlock& bb) {
    auto it = gResults.find(&bb);
    return (it != gResults.end()) ? &it->second : nullptr;
}

}  // namespace stinkytofu
