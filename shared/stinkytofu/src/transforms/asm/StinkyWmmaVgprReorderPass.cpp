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
// usage in double-buffered GEMM kernels.
//
// Problem: the 2x-unrolled loop body keeps A_X0 live across all B groups
// (B-major outer order), forcing A_X1 into separate physical registers.
//
// Solution: simulate A-major outer ordering. Under that order, A_X0[i] dies
// after its last B group and its interval no longer overlaps A_X1[i]. The
// two groups can therefore be aliased to the same physical registers.
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
#include <utility>
#include <unordered_map>

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
    for (auto it = bb.begin(); it != bb.end(); ++it) {
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (!inst || !isXDLWMMA(*inst)) continue;
        const auto* tag = inst->getModifier<WmmaPoolData>();
        if (!tag) {
            // Any untagged wmma means the pass was not properly prepared — skip.
            std::cerr << "[WmmaVgprReorderPass] wmma instruction missing WmmaPoolData"
                         " modifier; skipping block\n";
            return {};
        }
        byPool[tag->poolIndex].push_back(WmmaNode{
            const_cast<StinkyInstruction*>(inst),
            toRegGroup(inst->getSrcReg(0)),
            toRegGroup(inst->getSrcReg(1)),
            toRegGroup(inst->getDestReg(0)),
        });
    }
    std::vector<std::vector<WmmaNode>> pools;
    pools.reserve(byPool.size());
    for (auto& [idx, nodes] : byPool) pools.push_back(std::move(nodes));
    return pools;
}

/// Determine which src operand index is A (pool-varying) vs B (pool-shared).
/// The A operand has different register groups in different pools; B is shared.
/// Returns {aIdx, bIdx}.
std::pair<unsigned, unsigned> detectABIndices(
    const std::vector<std::vector<WmmaNode>>& pools)
{
    std::set<RegGroup> pool0Src0;
    for (const WmmaNode& n : pools[0]) pool0Src0.insert(n.aGroup); // aGroup = src0 tentatively
    for (size_t p = 1; p < pools.size(); ++p)
        for (const WmmaNode& n : pools[p])
            if (pool0Src0.count(n.aGroup)) return {1u, 0u}; // src0 shared → src0=B, src1=A
    return {0u, 1u};
}

/// Produce the A-major outer permutation of @p half.
/// Returns empty if the (aGroup, bGroup) pairs don't form a complete grid.
std::vector<WmmaNode> aMajorOuter(const std::vector<WmmaNode>& half) {
    std::set<RegGroup> aSet, bSet;
    std::map<std::pair<RegGroup, RegGroup>, WmmaNode> lookup;
    for (const WmmaNode& n : half) {
        aSet.insert(n.aGroup);
        bSet.insert(n.bGroup);
        lookup[{n.aGroup, n.bGroup}] = n;
    }

    std::vector<WmmaNode> result;
    result.reserve(half.size());
    for (const RegGroup& ag : aSet)       // sorted by base
        for (const RegGroup& bg : bSet)
            if (auto it = lookup.find({ag, bg}); it != lookup.end())
                result.push_back(it->second);

    return (result.size() == half.size()) ? result : std::vector<WmmaNode>{};
}

/// Build the flat per-operand replacement list: walk every instruction in @p bb
/// and emit an entry for each operand whose VGPR range overlaps an aliasable group.
std::vector<RegReplacement> buildReplacements(
    const BasicBlock&                  bb,
    const std::vector<AliasCandidate>& aliases)
{
    std::vector<RegReplacement> out;
    for (auto it = bb.begin(); it != bb.end(); ++it) {
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (!inst) continue;

        for (const AliasCandidate& alias : aliases) {
            auto patch = [&](const StinkyRegister& r, unsigned idx, bool isSrc) {
                if (!regOverlapsGroup(r, alias.aliasable)) return;
                StinkyRegister newReg    = r;
                newReg.reg.idx = alias.canonical.base + (r.reg.idx - alias.aliasable.base);
                out.push_back({const_cast<StinkyInstruction*>(inst), idx, isSrc, r, newReg});
            };
            for (unsigned i = 0; i < inst->getNumSrcRegs();  ++i) patch(inst->getSrcReg(i),  i, true);
            for (unsigned i = 0; i < inst->getNumDestRegs(); ++i) patch(inst->getDestReg(i), i, false);
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
    StinkyWmmaVgprReorderPassImpl(
        std::unique_ptr<IRegLivenessAnalysis>  liveness,
        std::unique_ptr<IWmmaReorderAlgorithm> algorithm)
        : liveness_(std::move(liveness)), algorithm_(std::move(algorithm)) {}

    static char ID;
    const char* getName() const override { return "StinkyWmmaVgprReorderPass"; }
    PassID      getPassID() const override { return &ID; }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager&) override {
        gResults.clear();
        for (BasicBlock& bb : func) {
            if (!passCtx.shouldProcessBasicBlock(bb)) continue;
            WmmaReorderAnalysisResult res = analyzeBlock(bb);
            if (res.applicable)
                std::cerr << "[WmmaVgprReorderPass] " << res.totalVgprSaved
                          << " VGPRs saveable, " << res.replacements.size()
                          << " operand replacements\n";
            gResults[&bb] = std::move(res);
        }
        return preserveCFGAnalyses();
    }

   private:
    std::unique_ptr<IRegLivenessAnalysis>  liveness_;
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

        WmmaReorderAnalysisResult out;
        out.applicable   = true;
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
    const BasicBlock& /*bb*/,
    const std::vector<WmmaNode>& wmmaSeq) const
{
    std::map<RegGroup, RegInterval> intervals;
    for (unsigned i = 0; i < wmmaSeq.size(); ++i) {
        for (const RegGroup* g : {&wmmaSeq[i].aGroup, &wmmaSeq[i].bGroup}) {
            auto [it, inserted] = intervals.emplace(*g, RegInterval{i, i});
            if (!inserted) {
                it->second.first = std::min(it->second.first, i);
                it->second.last  = std::max(it->second.last,  i);
            }
        }
    }
    return intervals;
}

// ─────────────────────────────────────────────────────────────────────────────
// ABI 2 — AMajorOuterAlgorithm (out-of-line)
// ─────────────────────────────────────────────────────────────────────────────

IWmmaReorderAlgorithm::Result AMajorOuterAlgorithm::solve(
    const std::vector<std::vector<WmmaNode>>& pools,
    const std::map<RegGroup, RegInterval>&    liveness) const
{
    Result result;
    if (pools.size() < 2) return result;

    // Check that B-major outer is actually in effect in pool 0 (saving exists).
    std::set<RegGroup> bGroups;
    for (const WmmaNode& n : pools[0]) bGroups.insert(n.bGroup);
    const unsigned nB = static_cast<unsigned>(bGroups.size());

    bool savingExists = false;
    for (const WmmaNode& n : pools[0]) {
        auto it = liveness.find(n.aGroup);
        if (it != liveness.end() && (it->second.last - it->second.first + 1) > nB) {
            savingExists = true;
            break;
        }
    }
    if (!savingExists) return result;

    // Reorder every pool to A-major outer and collect each pool's sorted A groups.
    result.desiredOrder.reserve(wmmaSeq.size());
    std::vector<std::vector<RegGroup>> poolAGroups;
    poolAGroups.reserve(pools.size());

    for (const auto& pool : pools) {
        auto reordered = aMajorOuter(pool);
        if (reordered.empty()) return Result{};
        for (const WmmaNode& n : reordered) result.desiredOrder.push_back(n);

        std::set<RegGroup> aSet;
        for (const WmmaNode& n : pool) aSet.insert(n.aGroup);
        poolAGroups.emplace_back(aSet.begin(), aSet.end()); // sorted by base
    }

    // Build alias candidates: pools 1..N-1 are each aliased onto pool 0.
    // Under A-major outer each pool's A groups are fully live only within that
    // pool's wmma range, so inter-pool intervals never overlap.
    const auto& canonical = poolAGroups[0];
    for (size_t p = 1; p < poolAGroups.size(); ++p) {
        const auto& aliasable = poolAGroups[p];
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
    std::unique_ptr<IRegLivenessAnalysis>  liveness,
    std::unique_ptr<IWmmaReorderAlgorithm> algorithm)
{
    if (!liveness)  liveness  = std::make_unique<WmmaIntervalLiveness>();
    if (!algorithm) algorithm = std::make_unique<AMajorOuterAlgorithm>();
    return std::make_unique<StinkyWmmaVgprReorderPassImpl>(
        std::move(liveness), std::move(algorithm));
}

const WmmaReorderAnalysisResult* getWmmaReorderResult(const BasicBlock& bb) {
    auto it = gResults.find(&bb);
    return (it != gResults.end()) ? &it->second : nullptr;
}

}  // namespace stinkytofu
