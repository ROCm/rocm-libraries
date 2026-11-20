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
#include "stinkytofu/transforms/asm/EstimateAsmCyclesPass.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <deque>
#include <iostream>
#include <memory>
#include <optional>
#include <queue>
#include <set>
#include <span>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "dag/ReadyQueue.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/HWModel.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/support/LoopDetection.hpp"

#define DEBUG_TYPE "EstimateAsmCyclesPass"

namespace {
using namespace stinkytofu;
constexpr const char* kEstimateAsmTotalCyclesMetadataKey = "EstimateAsmCyclesPass.totalCycles";
constexpr const char* kEstimateAsmPreLoopCyclesMetadataKey = "EstimateAsmCyclesPass.preLoopCycles";
constexpr const char* kEstimateAsmMainLoopCyclesMetadataKey =
    "EstimateAsmCyclesPass.mainLoopCycles";
constexpr const char* kEstimateAsmPostLoopCyclesMetadataKey =
    "EstimateAsmCyclesPass.postLoopCycles";

// The unrolled K loop, not a peeled InitC clone or the tail loop. Those labels
// also contain "LoopBeginL", so a substring match picks the wrong boundary.
bool isPrimaryLoopHeader(stinkytofu::BasicBlock const& bb) {
    auto const& label = bb.getLabel();
    return label == "label_LoopBeginL" || label == "LoopBeginL";
}

// InsertClusterBarrierPass keeps the dispatch block named label_LoopBeginL and
// retargets the wave-0 latch to label_LoopBeginL_CBWave0. That branch still
// closes the primary loop. A peeled InitC clone does not use this exact name.
bool isPrimaryLoopBackTarget(std::string const& target) {
    return target == "label_LoopBeginL" || target == "LoopBeginL" ||
           target == "label_LoopBeginL_CBWave0" || target == "LoopBeginL_CBWave0";
}

// The non-zero-wave copy is the same iteration as wave 0. Counting it again in
// the epilogue would add a second copy of the body. The name alone is not
// enough: label_skipCBPreSignal_LCL_* is the LoopCounterL == 0 join, and
// CFGBuilder names the straight-line piece after it with the same prefix.
// Only a branch at or after the block, back to that block, marks the copy.
bool isAlternateWaveHeader(std::string const& label) {
    return label.starts_with("label_skipCBPreSignal_");
}

void markAlternateWaveLoopBlocks(std::vector<BasicBlock*> const& ordered,
                                 std::vector<char>& excluded) {
    for (size_t headerIdx = 0; headerIdx < ordered.size(); ++headerIdx) {
        std::string const& header = ordered[headerIdx]->getLabel();
        if (!isAlternateWaveHeader(header)) continue;
        std::optional<size_t> latch;
        for (size_t i = headerIdx; i < ordered.size(); ++i) {
            for (IRBase& irNode : *ordered[i]) {
                StinkyInstruction* inst = dyn_cast<StinkyInstruction>(&irNode);
                if (inst == nullptr || !isBranch(*inst)) continue;
                if (getBranchTarget(*inst) == header) latch = i;
            }
        }
        if (!latch.has_value()) continue;
        for (size_t i = headerIdx; i <= *latch; ++i) excluded[i] = 1;
    }
}

// Global-write edge fork. NonEdge is the full tile. Then/Else are the
// partial-tile stores, reached by a short cbranch or by an annotated
// s_setpc_b64. The estimate always keeps NonEdge.
bool isGlobalWriteEdgeTarget(std::string const& target) {
    bool const globalWrite = target.find("GW_") != std::string::npos;
    auto endsWith = [](std::string const& s, std::string const& suffix) {
        return s.size() >= suffix.size() &&
               s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
    };
    return globalWrite && (endsWith(target, "_Then") || endsWith(target, "_Else"));
}

// Longest path in one phase. Back-edges are dropped, so a loop body counts once.
// Exclusive edges contribute only the heavier side.
uint32_t longestPathCycles(std::vector<BasicBlock*> const& nodes,
                           std::unordered_map<BasicBlock*, std::vector<BasicBlock*>> const& succ,
                           std::unordered_map<BasicBlock*, uint32_t> const& cost,
                           std::unordered_set<BasicBlock*> const& ignoredRoots) {
    std::unordered_set<BasicBlock*> inSet(nodes.begin(), nodes.end());
    std::unordered_map<BasicBlock*, int> color;
    std::set<std::pair<BasicBlock*, BasicBlock*>> backEdges;
    auto dfs = [&](auto const& self, BasicBlock* bb) -> void {
        color[bb] = 1;
        auto const it = succ.find(bb);
        if (it != succ.end()) {
            for (BasicBlock* next : it->second) {
                if (inSet.count(next) == 0) continue;
                if (color[next] == 1) {
                    backEdges.insert({bb, next});
                } else if (color[next] == 0) {
                    self(self, next);
                }
            }
        }
        color[bb] = 2;
    };
    for (BasicBlock* bb : nodes) {
        if (color[bb] == 0) dfs(dfs, bb);
    }

    std::unordered_map<BasicBlock*, int> indegree;
    std::unordered_map<BasicBlock*, std::vector<BasicBlock*>> forward;
    for (BasicBlock* bb : nodes) indegree[bb] = 0;
    for (BasicBlock* bb : nodes) {
        auto const it = succ.find(bb);
        if (it == succ.end()) continue;
        for (BasicBlock* next : it->second) {
            if (inSet.count(next) == 0 || backEdges.count({bb, next}) != 0) continue;
            forward[bb].push_back(next);
            indegree[next] += 1;
        }
    }

    // A skipped global-write edge leaves its target with no predecessor. That
    // block is not an alternate entry; only blocks reachable from a real entry
    // contribute. Other indegree-0 blocks stay entries, so an unmodeled fork
    // still keeps its heavier side.
    // Membership stays in a set. Iteration uses a vector so the order does not
    // depend on pointer values (bugprone-nondeterministic-pointer-iteration-order).
    std::unordered_set<BasicBlock*> reachableSet;
    std::vector<BasicBlock*> reachable;
    std::queue<BasicBlock*> walk;
    for (BasicBlock* bb : nodes) {
        if (indegree[bb] == 0 && ignoredRoots.count(bb) == 0) {
            reachableSet.insert(bb);
            reachable.push_back(bb);
            walk.push(bb);
        }
    }
    while (!walk.empty()) {
        BasicBlock* bb = walk.front();
        walk.pop();
        for (BasicBlock* next : forward[bb]) {
            if (reachableSet.insert(next).second) {
                reachable.push_back(next);
                walk.push(next);
            }
        }
    }

    std::unordered_map<BasicBlock*, int> reachIndegree;
    for (BasicBlock* bb : reachable) reachIndegree[bb] = 0;
    for (BasicBlock* bb : reachable) {
        for (BasicBlock* next : forward[bb]) {
            if (reachableSet.count(next) != 0) reachIndegree[next] += 1;
        }
    }

    std::queue<BasicBlock*> ready;
    std::unordered_map<BasicBlock*, uint32_t> dist;
    for (BasicBlock* bb : reachable) {
        auto const costIt = cost.find(bb);
        uint32_t const blockCost = costIt == cost.end() ? 0 : costIt->second;
        dist[bb] = reachIndegree[bb] == 0 ? blockCost : 0;
        if (reachIndegree[bb] == 0) ready.push(bb);
    }
    while (!ready.empty()) {
        BasicBlock* bb = ready.front();
        ready.pop();
        for (BasicBlock* next : forward[bb]) {
            if (reachableSet.count(next) == 0) continue;
            uint32_t const nextCost = cost.count(next) ? cost.at(next) : 0;
            dist[next] = std::max(dist[next], dist[bb] + nextCost);
            reachIndegree[next] -= 1;
            if (reachIndegree[next] == 0) ready.push(next);
        }
    }
    uint32_t best = 0;
    for (BasicBlock* bb : reachable) best = std::max(best, dist[bb]);
    return best;
}

// Legacy keeps the per-instruction query on the old loop-body slice.
// LoopBody counts one iteration of a block in the primary-loop range and stops
// at the back-edge. EntireBlock counts every instruction, which is what
// prologue and epilogue blocks contribute.
enum class BlockCycleScope { Legacy, LoopBody, EntireBlock };

// The DAG issue model charges s_set_vgpr_msb its own issue cycle. On gfx1250 it can
// issue in the same cycle as the instruction immediately before it when that
// instruction is a SALU other than a barrier, a VALU, or a v_wmma (including the
// scaled v_wmma forms).
bool isSetVgprMsb(const StinkyInstruction& inst) {
    return inst.getUnifiedOpcode() == GFX::s_set_vgpr_msb;
}

bool msbCoIssuesWith(const StinkyInstruction& prev) {
    if (isBarrier(prev)) return false;
    return isScalarALU(prev) || isVectorALU(prev) || isWMMA(prev) || isMXWMMA(prev);
}

// One LDS op the DS counter still has to retire. issueClock is the replay clock at which the
// op starts issuing. The drain entry is that one op: computeDynamicDrainLatencyForLoads on a
// single entry returns its own latency, capped by maxDrain.
struct IssuedDsOp {
    int issueClock = 0;
    DsLoadDrainEntry entry;
};

// Issue state for one basic block. It is not carried to the next block: a
// predecessor may be on another side of a branch, and longest-path adds the
// block costs after the fact.
struct CycleReplay {
    std::unique_ptr<dag::ReadyQueue> queue;
    std::deque<IssuedDsOp> inflightDs;
    const StinkyInstruction* prevIssued = nullptr;
    int cycles = 0;
};

// s_wait_dscnt(N) / the ds half of s_wait_loadcnt_dscnt. N is how many LDS ops may still be
// outstanding when the wait completes. Absent when this instruction does not wait on dscnt.
std::optional<int> dsWaitKeepCount(const StinkyInstruction& inst) {
    if (!isWaitCnt(inst)) return std::nullopt;
    const uint16_t op = inst.getUnifiedOpcode();
    if (op != GFX::s_wait_dscnt && op != GFX::s_wait_loadcnt_dscnt) return std::nullopt;

    if (const SWaitCntData* w = inst.getModifier<SWaitCntData>()) {
        const int dl = w->dlcnt;
        const int ds = w->dscnt;
        if (dl >= 0 && ds >= 0) return std::min(dl, ds);
        if (dl >= 0) return dl;
        if (ds >= 0) return ds;
    }
    const auto& srcs = inst.getSrcRegs();
    if (!srcs.empty() && srcs[0].dataType == StinkyRegister::Type::LiteralInt) {
        const int imm = srcs[0].getLiteralInt();
        if (op == GFX::s_wait_loadcnt_dscnt) return unpackDsWaitCnt(imm);
        return imm;
    }
    return std::nullopt;
}

int dsOpRemaining(const HWModel& hw, const IssuedDsOp& op, int now, int numWaves) {
    const int drain = computeDynamicDrainLatencyForLoads(
        hw, std::span<const DsLoadDrainEntry>(&op.entry, 1), numWaves);
    return drain - (now - op.issueClock);
}

// Cycles until the oldest outstanding ops have returned and only \p keep remain.
// ds_load, ds_load, s_wait_dscnt 1 waits for the first load: its drain minus the cycles
// already spent since that load started issuing.
int dsWaitStallCycles(const HWModel& hw, const std::deque<IssuedDsOp>& inflight, int keep, int now,
                      int numWaves) {
    std::vector<int> outstanding;
    outstanding.reserve(inflight.size());
    for (const IssuedDsOp& op : inflight) {
        const int left = dsOpRemaining(hw, op, now, numWaves);
        if (left > 0) outstanding.push_back(left);
    }
    if (keep < 0) keep = 0;
    if (static_cast<int>(outstanding.size()) <= keep) return 0;
    const size_t mustFinish = outstanding.size() - static_cast<size_t>(keep);
    int stall = 0;
    for (size_t i = 0; i < mustFinish; ++i) stall = std::max(stall, outstanding[i]);
    return stall;
}

void retireReturnedDsOps(const HWModel& hw, std::deque<IssuedDsOp>& inflight, int now,
                         int numWaves) {
    while (!inflight.empty() && dsOpRemaining(hw, inflight.front(), now, numWaves) <= 0)
        inflight.pop_front();
}

/// Implementation of the EstimateAsmCycles pass
///
/// This pass estimates cycle counts for asm-related operations
/// by analyzing instruction sequences and their characteristics.
class EstimateAsmCyclesPassImpl : public Pass {
   public:
    static constexpr const char* PassName = "EstimateAsmCyclesPass";
    static char ID;

    PassID getPassID() const override {
        return &ID;
    }

    const char* getName() const override {
        return PassName;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& AM) override {
        (void)AM;
        // Reset total and phase cycles before processing.
        totalCycles_ = 0;
        phaseCycles_ = {};

        // The per-instruction query is a read-only slice of the unrolled loop.
        // Phase totals are for the publishing pass, which sees the whole kernel.
        if (perInstructionCycleQuery_) {
            for (BasicBlock& bb : func) {
                if (!passCtx.shouldProcessBasicBlock(bb)) continue;
                calculateMathClocksInUnrolledLoop(bb, passCtx, BlockCycleScope::Legacy);
            }
            PASS_DEBUG(std::cerr << "[EstimateAsmCyclesPass] totalCycles=" << totalCycles_ << "\n");
            return PreservedAnalyses::all();
        }

        // detectLoops walks CFG successor edges. The K-loop back-edge is the
        // branch to label_LoopBeginL, or to label_LoopBeginL_CBWave0 after
        // InsertClusterBarrierPass retargets the wave-0 latch. That edge is often
        // missing after later passes split the body. Close the range on the branch
        // instruction itself: every block from the header through the block that
        // branches back is the loop. detectLoops still contributes when its edges
        // are present.
        std::vector<BasicBlock*> ordered;
        ordered.reserve(func.size());
        for (BasicBlock& bb : func) ordered.push_back(&bb);

        std::unordered_set<BasicBlock*> loopBlocks;
        for (const Loop& loop : detectLoops(func)) {
            if (loop.headerBB != nullptr && isPrimaryLoopHeader(*loop.headerBB)) {
                loopBlocks.insert(loop.bodyBBs.begin(), loop.bodyBBs.end());
            }
        }

        auto branchesToPrimaryLoop = [](BasicBlock& bb) {
            for (IRBase& irNode : bb) {
                StinkyInstruction* inst = dyn_cast<StinkyInstruction>(&irNode);
                if (inst == nullptr || !isBranch(*inst)) continue;
                std::string const target = getBranchTarget(*inst);
                if (isPrimaryLoopBackTarget(target)) return true;
            }
            return false;
        };

        std::optional<size_t> headerIdx;
        std::optional<size_t> latchIdx;
        for (size_t i = 0; i < ordered.size(); ++i) {
            if (!headerIdx.has_value() && isPrimaryLoopHeader(*ordered[i])) headerIdx = i;
            if (branchesToPrimaryLoop(*ordered[i])) latchIdx = i;
        }
        if (headerIdx.has_value() && latchIdx.has_value() && *latchIdx >= *headerIdx) {
            for (size_t i = *headerIdx; i <= *latchIdx; ++i) loopBlocks.insert(ordered[i]);
        } else if (loopBlocks.empty() && headerIdx.has_value()) {
            loopBlocks.insert(ordered[*headerIdx]);
        }

        std::optional<size_t> rangeBegin;
        std::optional<size_t> rangeEnd;
        for (size_t i = 0; i < ordered.size(); ++i) {
            if (loopBlocks.count(ordered[i]) == 0) continue;
            if (!rangeBegin.has_value()) rangeBegin = i;
            rangeEnd = i;
        }

        std::unordered_map<std::string, BasicBlock*> labelMap;
        for (BasicBlock* bb : ordered) {
            if (!bb->getLabel().empty()) labelMap.emplace(bb->getLabel(), bb);
        }
        std::unordered_map<BasicBlock*, std::vector<BasicBlock*>> succ;
        std::unordered_set<BasicBlock*> ignoredEdgeRoots;
        for (BasicBlock* bb : ordered) {
            if (isGlobalWriteEdgeTarget(bb->getLabel())) ignoredEdgeRoots.insert(bb);
        }
        for (size_t i = 0; i < ordered.size(); ++i) {
            BasicBlock* bb = ordered[i];
            StinkyInstruction* term = nullptr;
            for (auto it = bb->rbegin(); it != bb->rend(); ++it) {
                if (StinkyInstruction* inst = dyn_cast<StinkyInstruction>(it.getNodePtr())) {
                    term = inst;
                    break;
                }
            }
            bool fallsThrough = true;
            if (term != nullptr && isBranch(*term)) {
                std::string const target = getBranchTarget(*term);
                // Branch to the partial-tile store, either a short cbranch or a
                // long s_setpc_b64. Leave it untaken so the path stays on NonEdge.
                bool const skipEdgeStore = isGlobalWriteEdgeTarget(target);
                auto const targetIt = labelMap.find(target);
                if (skipEdgeStore && targetIt != labelMap.end())
                    ignoredEdgeRoots.insert(targetIt->second);
                if (!skipEdgeStore && targetIt != labelMap.end())
                    succ[bb].push_back(targetIt->second);
                if (!isConditionalBranch(*term)) fallsThrough = false;
            }
            if (term != nullptr && isEndOfFunction(*term)) fallsThrough = false;
            if (fallsThrough && i + 1 < ordered.size()) {
                BasicBlock* next = ordered[i + 1];
                if (std::find(succ[bb].begin(), succ[bb].end(), next) == succ[bb].end()) {
                    succ[bb].push_back(next);
                }
            }
        }

        std::vector<char> excluded(ordered.size(), 0);
        markAlternateWaveLoopBlocks(ordered, excluded);

        // Each block is replayed from empty issue state. The other wave's copy is
        // skipped so its instructions are not added to the epilogue.
        std::unordered_map<BasicBlock*, uint32_t> blockCost;
        std::string primaryLoopLabels;
        std::vector<BasicBlock*> preBlocks;
        std::vector<BasicBlock*> postBlocks;
        for (size_t i = 0; i < ordered.size(); ++i) {
            BasicBlock& bb = *ordered[i];
            bool const inLoop = rangeBegin.has_value() && i >= *rangeBegin && i <= *rangeEnd;
            uint32_t blockCycles = 0;
            if (excluded[i] == 0 && passCtx.shouldProcessBasicBlock(bb)) {
                if (inLoop) {
                    if (!primaryLoopLabels.empty()) primaryLoopLabels += ',';
                    primaryLoopLabels += bb.getLabel();
                }
                calculateMathClocksInUnrolledLoop(
                    bb, passCtx, inLoop ? BlockCycleScope::LoopBody : BlockCycleScope::EntireBlock);
                blockCycles = totalCycles_;
            }
            blockCost[&bb] = blockCycles;
            if (excluded[i] != 0) continue;
            if (inLoop) {
                phaseCycles_.mainLoopCycles += blockCycles;
            } else if (rangeBegin.has_value() && i > *rangeEnd) {
                postBlocks.push_back(&bb);
            } else {
                preBlocks.push_back(&bb);
            }
        }
        phaseCycles_.preLoopCycles =
            longestPathCycles(preBlocks, succ, blockCost, ignoredEdgeRoots);
        phaseCycles_.postLoopCycles =
            longestPathCycles(postBlocks, succ, blockCost, ignoredEdgeRoots);
        PASS_DEBUG(std::cerr << "[EstimateAsmCyclesPass] primaryLoop=" << primaryLoopLabels
                             << "\n");
        totalCycles_ =
            phaseCycles_.preLoopCycles + phaseCycles_.mainLoopCycles + phaseCycles_.postLoopCycles;
        PASS_DEBUG(
            std::cerr << "[EstimateAsmCyclesPass] preLoopCycles=" << phaseCycles_.preLoopCycles
                      << " mainLoopCycles=" << phaseCycles_.mainLoopCycles << " postLoopCycles="
                      << phaseCycles_.postLoopCycles << " totalCycles=" << totalCycles_ << "\n");
        if (publishMetadata_) {
            func.setMetaData(kEstimateAsmTotalCyclesMetadataKey, totalCycles_);
            func.setMetaData(kEstimateAsmPreLoopCyclesMetadataKey, phaseCycles_.preLoopCycles);
            func.setMetaData(kEstimateAsmMainLoopCyclesMetadataKey, phaseCycles_.mainLoopCycles);
            func.setMetaData(kEstimateAsmPostLoopCyclesMetadataKey, phaseCycles_.postLoopCycles);
        }

        return PreservedAnalyses::all();
    }

    /// Get the total cycles calculated by this pass
    unsigned int getTotalCycles() const {
        return totalCycles_;
    }

    EstimateAsmCyclePhases getPhaseCycles() const {
        return phaseCycles_;
    }

    /// Per-instruction cumulative cycle positions recorded during the last run.
    const std::unordered_map<const StinkyInstruction*, uint32_t>& getPerInstructionCycles() const {
        return perInstCycles_;
    }

    /// When false, suppress the `<This is N-cycle>` comment annotations so the
    /// pass can be used purely as a cycle-position query by other passes.
    void setAnnotateComments(bool annotate) {
        annotateComments_ = annotate;
    }

    /// When false, suppress publishing the total-cycles function metadata
    /// (`EstimateAsmCyclesPass.totalCycles`). Read-only query callers set this
    /// alongside setAnnotateComments(false) so run() has no IR/function side
    /// effects — the AnalysisManager contract for a pure analysis.
    void setPublishMetadata(bool publish) {
        publishMetadata_ = publish;
    }

    /// When true, also model non-loop blocks that contain an embedded
    /// `label_LoopBeginL` (hierarchical / sub-loop IR). Used only by
    /// computeEstimatedCyclesPerInstruction(); the published pass total stays
    /// loop-block-only.
    void setPerInstructionCycleQuery(bool query) {
        perInstructionCycleQuery_ = query;
    }

   private:
    // (For future extension: If analysis needs to track cycles per basic block/label,
    // consider using a map like below)
    // std::unordered_map<std::string, unsigned int> labelCycles_;
    void appendComment(StinkyInstruction* inst, const std::string& suffix) {
        if (!inst || suffix.empty()) return;
        if (auto* c = inst->getModifier<CommentData>()) {
            if (!c->comment.empty()) c->comment += ' ';
            c->comment += suffix;
        } else {
            inst->addModifier<CommentData>(CommentData{suffix});
        }
    }

    /// Process a single basic block to estimate asm cycles. Issue state starts
    /// empty here and is discarded when the block ends.
    void calculateMathClocksInUnrolledLoop(BasicBlock& bb, PassContext& passCtx,
                                           BlockCycleScope scope) {
        totalCycles_ = 0;
        // Collect all instructions in this basic block
        std::vector<StinkyInstruction*> instructions;
        for (IRBase& irNode : bb) {
            if (irNode.getType() == IRBase::IRType::StinkyTofu) {
                instructions.push_back(cast<StinkyInstruction>(&irNode));
            }
        }

        if (instructions.empty()) return;

        CycleReplay replay;
        replay.queue = dag::chooseReadyQueue(passCtx);
        int const blockStart = replay.cycles;
        // Loop-body modelling runs in any basic block whose label contains label_LoopBeginL
        // (tensilelite names the block ^label_LoopBeginL, and the rocisa split adds suffixes
        // such as _CBWave0). computeEstimatedCyclesPerInstruction() also covers blocks that
        // embed that label.
        std::size_t startIdx = 0;
        if (scope == BlockCycleScope::Legacy &&
            bb.getLabel().find("label_LoopBeginL") == std::string::npos &&
            bb.getLabel() != "LoopBeginL") {
            startIdx = instructions.size();
            if (perInstructionCycleQuery_) {
                for (std::size_t i = 0; i < instructions.size(); ++i) {
                    const LabelData* labelData = instructions[i]->getModifier<LabelData>();
                    if (isLabel(*instructions[i]) && labelData != nullptr &&
                        labelData->label == "label_LoopBeginL") {
                        startIdx = i;
                        break;
                    }
                }
            }
        }
        // chooseReadyQueue is the only arch switch. A queue with no issue model
        // (anything other than CDNA5 today) leaves the total at 0.
        const bool useSchedulerTiming = replay.queue->modelsIssueTiming();
        const HWModel& hw = passCtx.getHWModel();
        const int numWaves = static_cast<int>(passCtx.getGemmTileConfig().NumWaves);
        for (std::size_t idx = startIdx; idx < instructions.size(); ++idx) {
            StinkyInstruction* inst = instructions[idx];
            if (useSchedulerTiming) {
                // A back-edge to this loop ends the unroll body. The branch itself
                // is not part of the total.
                if (scope != BlockCycleScope::EntireBlock && (isBranch(*inst) || isCall(*inst))) {
                    const LabelData* labelData = inst->getModifier<LabelData>();
                    if (labelData != nullptr && labelData->label.find("label_LoopBeginL") == 0) {
                        break;
                    }
                }
                // Co-issued s_set_vgpr_msb stays on the previous instruction's cycle.
                const bool coIssuedMsb = !isLabel(*inst) && isSetVgprMsb(*inst) &&
                                         replay.prevIssued != nullptr &&
                                         msbCoIssuesWith(*replay.prevIssued);
                if (coIssuedMsb) {
                    // Clock stays on the previous instruction.
                } else if (const std::optional<int> keep = dsWaitKeepCount(*inst)) {
                    // The stall is how long the waited-for LDS op still has left. The
                    // instruction still issues in one cycle when that op has already returned.
                    const int stall =
                        dsWaitStallCycles(hw, replay.inflightDs, *keep, replay.cycles, numWaves);
                    const int savedIssue = inst->issueCycles;
                    inst->issueCycles = std::max(stall, savedIssue);
                    replay.queue->replayIssue(*inst);
                    inst->issueCycles = savedIssue;
                    replay.cycles = replay.queue->elapsedCycles();
                    retireReturnedDsOps(hw, replay.inflightDs, replay.cycles, numWaves);
                } else {
                    const bool dsOp = isDSRead(*inst) || isDSWrite(*inst);
                    const int issueClock = replay.cycles;
                    DsLoadDrainEntry entry;
                    if (dsOp) {
                        const HwInstDesc* desc = inst->getHwInstDesc();
                        entry = makeDsLoadDrainEntry(hw, inst->latencyCycles,
                                                     desc != nullptr ? desc->dsThroughput : 0,
                                                     desc != nullptr ? desc->dsMaxDrain : 0);
                    }
                    replay.queue->replayIssue(*inst);
                    replay.cycles = replay.queue->elapsedCycles();
                    if (dsOp) replay.inflightDs.push_back(IssuedDsOp{issueClock, entry});
                }
            }
            if (!isLabel(*inst)) replay.prevIssued = inst;

            // Record the cumulative cycle position of every processed
            // instruction so other passes can query it (see
            // computeEstimatedCyclesPerInstruction).
            perInstCycles_[inst] = static_cast<uint32_t>(replay.cycles < 0 ? 0 : replay.cycles);

            // Update total cycles
            if (annotateComments_ && !isLabel(*inst)) {
                appendComment(inst, "<This is " + std::to_string(replay.cycles) + "-cycle>");
            }
        }
        int const blockCycles = replay.cycles - blockStart;
        totalCycles_ = static_cast<uint32_t>(blockCycles < 0 ? 0 : blockCycles);
        if (totalCycles_ > 0 || bb.getLabel().find("LoopBegin") != std::string::npos) {
            PASS_DEBUG(std::cerr << "[EstimateAsmCyclesPass] bb=" << bb.getLabel()
                                 << " cycles=" << totalCycles_ << " publish=" << publishMetadata_
                                 << " perInstQuery=" << perInstructionCycleQuery_ << "\n");
        }
    }

    bool annotateComments_ = true;
    bool perInstructionCycleQuery_ = false;
    bool publishMetadata_ = true;
    std::unordered_map<const StinkyInstruction*, uint32_t> perInstCycles_;

    unsigned int totalCycles_ = 0;
    EstimateAsmCyclePhases phaseCycles_{};
};

char EstimateAsmCyclesPassImpl::ID = 0;

}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createEstimateAsmCyclesPass() {
    return std::make_unique<EstimateAsmCyclesPassImpl>();
}

EstimateAsmCyclesAnalysis::Result EstimateAsmCyclesAnalysis::run(Function& func,
                                                                 AnalysisManager& AM) {
    (void)AM;
    PassContext passCtx;
    passCtx.setGemmTileConfig(func.getGemmTileConfig());
    passCtx.setBasicBlockFilter(BasicBlockFilterBuilder::all());
    return calculateEstimateAsmCycles(func, passCtx);
}

EstimateAsmCyclePhasesAnalysis::Result EstimateAsmCyclePhasesAnalysis::run(Function& func,
                                                                           AnalysisManager& AM) {
    (void)AM;
    PassContext passCtx;
    passCtx.setGemmTileConfig(func.getGemmTileConfig());
    passCtx.setBasicBlockFilter(BasicBlockFilterBuilder::all());
    return calculateEstimateAsmCyclePhases(func, passCtx);
}

EstimateAsmCyclesPerInstructionAnalysis::Result EstimateAsmCyclesPerInstructionAnalysis::run(
    Function& func, AnalysisManager& AM) {
    (void)AM;
    PassContext passCtx;
    passCtx.setGemmTileConfig(func.getGemmTileConfig());
    passCtx.setBasicBlockFilter(BasicBlockFilterBuilder::all());
    return computeEstimatedCyclesPerInstruction(func, passCtx);
}

unsigned int calculateEstimateAsmCycles(Function& func, PassContext& passCtx) {
    EstimateAsmCyclesPassImpl pass;
    pass.setAnnotateComments(false);
    pass.setPublishMetadata(false);
    AnalysisManager AM;
    (void)pass.run(func, passCtx, AM);
    return pass.getTotalCycles();
}

EstimateAsmCyclePhases calculateEstimateAsmCyclePhases(Function& func, PassContext& passCtx) {
    EstimateAsmCyclesPassImpl pass;
    pass.setAnnotateComments(false);
    pass.setPublishMetadata(false);
    AnalysisManager AM;
    (void)pass.run(func, passCtx, AM);
    return pass.getPhaseCycles();
}

std::unordered_map<const StinkyInstruction*, uint32_t> computeEstimatedCyclesPerInstruction(
    Function& func, PassContext& passCtx) {
    EstimateAsmCyclesPassImpl pass;
    pass.setAnnotateComments(false);
    pass.setPublishMetadata(false);
    pass.setPerInstructionCycleQuery(true);
    AnalysisManager AM;
    (void)pass.run(func, passCtx, AM);
    return pass.getPerInstructionCycles();
}
}  // namespace stinkytofu
