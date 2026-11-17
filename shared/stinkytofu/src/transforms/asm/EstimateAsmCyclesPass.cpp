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

#include <cassert>
#include <cstdint>
#include <deque>
#include <iostream>
#include <memory>
#include <optional>
#include <queue>
#include <span>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "dag/ReadyQueue.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/HWModel.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/support/Casting.hpp"

#define DEBUG_TYPE "EstimateAsmCyclesPass"

namespace {
using namespace stinkytofu;
constexpr const char* kEstimateAsmTotalCyclesMetadataKey = "EstimateAsmCyclesPass.totalCycles";

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
        // Reset total cycles before processing
        totalCycles_ = 0;

        // Process all basic blocks
        for (BasicBlock& bb : func) {
            // Skip filtered basic blocks
            if (!passCtx.shouldProcessBasicBlock(bb)) continue;

            // std::cout << "Processing basic block: " << bb.getLabel() << "\n";
            // Only process the loopWithPrefetch block

            // if(bb.getLabel() == "LocalReadDoA_I0" || bb.getLabel() == "LocalReadDoB_I0")
            // {
            //     calculateLocalReadBytes(bb, passCtx);

            // }
            // else if(bb.getLabel() == "LocalRead")
            // {
            //     analyzeBankConflicts(bb, passCtx);
            // }
            // else if(bb.getLabel() == "label_LoopBeginL")
            // {
            //     calculateMathClocksInUnrolledLoop(bb, passCtx);
            // }
            calculateMathClocksInUnrolledLoop(bb, passCtx);
        }
        PASS_DEBUG(std::cerr << "[EstimateAsmCyclesPass] totalCycles=" << totalCycles_ << "\n");
        if (publishMetadata_) func.setMetaData(kEstimateAsmTotalCyclesMetadataKey, totalCycles_);

        return PreservedAnalyses::all();
    }

    /// Get the total cycles calculated by this pass
    unsigned int getTotalCycles() const {
        return totalCycles_;
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
            if (!c->comment.empty()) c->comment += " ";
            c->comment += suffix;
        } else {
            inst->addModifier<CommentData>(CommentData{suffix});
        }
    }

    /// Process a single basic block to estimate asm cycles
    void calculateMathClocksInUnrolledLoop(BasicBlock& bb, PassContext& passCtx) {
        // Collect all instructions in this basic block
        std::vector<StinkyInstruction*> instructions;
        for (IRBase& irNode : bb) {
            if (irNode.getType() == IRBase::IRType::StinkyTofu) {
                instructions.push_back(cast<StinkyInstruction>(&irNode));
            }
        }

        if (instructions.empty()) return;

        int cycles = 0;
        uint32_t totalCycles = 0;
        // Loop-body modelling runs in any basic block whose label contains label_LoopBeginL
        // (tensilelite names the block ^label_LoopBeginL, and the rocisa split adds suffixes
        // such as _CBWave0). computeEstimatedCyclesPerInstruction() also covers blocks that
        // embed that label.
        std::size_t startIdx = 0;
        if (bb.getLabel().find("label_LoopBeginL") == std::string::npos &&
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
        std::unique_ptr<dag::ReadyQueue> issueQueue = dag::chooseReadyQueue(passCtx);
        const bool useSchedulerTiming = issueQueue->modelsIssueTiming();
        const HWModel& hw = passCtx.getHWModel();
        const int numWaves = static_cast<int>(passCtx.getGemmTileConfig().NumWaves);
        const StinkyInstruction* prevIssued = nullptr;
        std::deque<IssuedDsOp> inflightDs;
        for (std::size_t idx = startIdx; idx < instructions.size(); ++idx) {
            StinkyInstruction* inst = instructions[idx];
            if (useSchedulerTiming) {
                // A back-edge to this loop ends the unroll body. The branch itself
                // is not part of the total.
                if (isBranch(*inst) || isCall(*inst)) {
                    const LabelData* labelData = inst->getModifier<LabelData>();
                    if (labelData != nullptr && labelData->label.find("label_LoopBeginL") == 0) {
                        break;
                    }
                }
                // Co-issued s_set_vgpr_msb stays on the previous instruction's cycle.
                const bool coIssuedMsb = !isLabel(*inst) && isSetVgprMsb(*inst) &&
                                         prevIssued != nullptr && msbCoIssuesWith(*prevIssued);
                if (coIssuedMsb) {
                    // Clock stays on the previous instruction.
                } else if (const std::optional<int> keep = dsWaitKeepCount(*inst)) {
                    // The stall is how long the waited-for LDS op still has left. The
                    // instruction still issues in one cycle when that op has already returned.
                    const int stall = dsWaitStallCycles(hw, inflightDs, *keep, cycles, numWaves);
                    const int savedIssue = inst->issueCycles;
                    inst->issueCycles = std::max(stall, savedIssue);
                    issueQueue->replayIssue(*inst);
                    inst->issueCycles = savedIssue;
                    cycles = issueQueue->elapsedCycles();
                    retireReturnedDsOps(hw, inflightDs, cycles, numWaves);
                } else {
                    const bool dsOp = isDSRead(*inst) || isDSWrite(*inst);
                    const int issueClock = cycles;
                    DsLoadDrainEntry entry;
                    if (dsOp) {
                        const HwInstDesc* desc = inst->getHwInstDesc();
                        entry = makeDsLoadDrainEntry(hw, inst->latencyCycles,
                                                     desc != nullptr ? desc->dsThroughput : 0,
                                                     desc != nullptr ? desc->dsMaxDrain : 0);
                    }
                    issueQueue->replayIssue(*inst);
                    cycles = issueQueue->elapsedCycles();
                    if (dsOp) inflightDs.push_back(IssuedDsOp{issueClock, entry});
                }
            }
            if (!isLabel(*inst)) prevIssued = inst;

            // Record the cumulative cycle position of every processed
            // instruction so other passes can query it (see
            // computeEstimatedCyclesPerInstruction).
            perInstCycles_[inst] = static_cast<uint32_t>(cycles < 0 ? 0 : cycles);

            // Update total cycles
            if (annotateComments_ && !isLabel(*inst)) {
                // inst->dump(std::cout, false, "AsmCycles "+std::to_string(cycles - totalCycles));
                appendComment(inst, "<This is " + std::to_string(cycles) + "-cycle>");
            }
            // std::cout << "cycles: " << cycles - totalCycles << std::endl;
            totalCycles = cycles;
        }
        totalCycles_ = totalCycles;
        if (totalCycles > 0 || bb.getLabel().find("LoopBegin") != std::string::npos) {
            PASS_DEBUG(std::cerr << "[EstimateAsmCyclesPass] bb=" << bb.getLabel()
                                 << " cycles=" << totalCycles << " publish=" << publishMetadata_
                                 << " perInstQuery=" << perInstructionCycleQuery_ << "\n");
        }
    }

    bool annotateComments_ = true;
    bool perInstructionCycleQuery_ = false;
    bool publishMetadata_ = true;
    std::unordered_map<const StinkyInstruction*, uint32_t> perInstCycles_;

    unsigned int totalCycles_ = 0;
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
