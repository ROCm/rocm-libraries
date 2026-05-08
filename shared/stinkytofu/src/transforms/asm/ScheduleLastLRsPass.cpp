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
#include "stinkytofu/transforms/asm/ScheduleLastLRsPass.hpp"

#include <iostream>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

namespace {
using namespace stinkytofu;

static int getLRDistance(BasicBlock::iterator regStart, BasicBlock::iterator instsEnd,
                         BasicBlock::iterator instsBegin,
                         const std::vector<StinkyRegister>& lrDst) {
    int cycles = 0;
    for (BasicBlock::iterator it = regStart; it != instsEnd; ++it) {
        StinkyInstruction& inst = getStinkyInst(it);
        for (const StinkyRegister& reg : inst.getSrcRegs()) {
            // check overlap
            for (const StinkyRegister& dst : lrDst) {
                if (reg.isOverlap(dst)) {
                    return cycles;
                }
            }
        }
        if (isMFMA(inst) || isWMMA(inst)) {
            cycles += inst.latencyCycles;
        } else {
            cycles += inst.issueCycles;
        }
    }

    for (BasicBlock::iterator it = instsBegin; it != regStart; ++it) {
        StinkyInstruction& inst = getStinkyInst(it);
        for (const StinkyRegister& reg : inst.getSrcRegs()) {
            // check overlap
            for (const StinkyRegister& dst : lrDst) {
                if (reg.isOverlap(dst)) {
                    return cycles;
                }
            }
        }
        if (isMFMA(inst) || isWMMA(inst)) {
            cycles += inst.latencyCycles;
        } else {
            cycles += inst.issueCycles;
        }
    }
    return cycles;
}

/// True when both \p lhs and \p rhs are null, or when both are \c s_set_vgpr_msb
/// with identical source operands (same encoding / MSB set value).
static bool compareVgprMsb(StinkyInstruction* lhs, StinkyInstruction* rhs) {
    if (lhs == nullptr && rhs == nullptr) return true;
    if (lhs == nullptr || rhs == nullptr) return false;

    const auto& srcL = lhs->getSrcRegs();
    const auto& srcR = rhs->getSrcRegs();
    if (srcL.size() != srcR.size()) return false;
    for (size_t i = 0; i < srcL.size(); ++i) {
        if (srcL[i].getLiteralInt() != srcR[i].getLiteralInt()) return false;
    }
    return true;
}

// Schedule the Final PGR in the given BasicBlock.
// This will Move the PGR instructions to the suitable position to hide the latency.
//
// In the end, the instructions will be reordered in the block
// to reflect the scheduling order.
void scheduleFinalLocalReadWithLatency(BasicBlock& bb, PassContext& passCtx) {
    if (bb.empty()) return;

    std::vector<StinkyInstruction*> scheduled;
    scheduled.reserve(bb.size());

    BasicBlock::iterator beginIt = bb.begin();
    BasicBlock::iterator endIt = bb.end();

    BasicBlock::iterator regionStart = beginIt;
    BasicBlock::iterator regionEnd = endIt;

    // 1. Find the last barrier
    for (BasicBlock::iterator rit = beginIt; rit != endIt; ++rit) {
        StinkyInstruction& inst = getStinkyInst(rit);
        if (isBarrier(inst)) {
            // Start a new region after the side-effect instruction.
            regionStart = std::next(BasicBlock::iterator(rit.getNodePtr()));
        }
        const LabelData* labelData = inst.getModifier<LabelData>();
        if (labelData != nullptr) {
            const std::string& labelName = labelData->label;
            auto pos = labelName.find("label_LoopBeginL");
            if (pos != std::string::npos && pos == 0) {
                regionEnd = rit;
                break;
            }
        }
    }

    std::unordered_map<StinkyInstruction*, StinkyInstruction*> mfmaMapVgprMsb;
    std::unordered_map<StinkyInstruction*, StinkyInstruction*> lrMapVgprMsb;
    StinkyInstruction* currSetVgprMsb = nullptr;
    for (BasicBlock::iterator it = regionStart; it != regionEnd; ++it) {
        StinkyInstruction& inst = getStinkyInst(it);
        if (isBarrier(inst)) {
            break;
        }
        if (isDSRead(inst)) {
            lrMapVgprMsb[&inst] = currSetVgprMsb;
        }
        if (isMFMA(inst) || isWMMA(inst)) {
            mfmaMapVgprMsb[&inst] = currSetVgprMsb;
        }
        const std::string& opcode = inst.getHwInstDesc()->mnemonic;
        if (opcode.find("s_set_vgpr_msb") != std::string::npos) {
            currSetVgprMsb = &inst;
        }
    }

    // 2. Push the instructions before the barrier.
    for (BasicBlock::iterator it = beginIt; it != regionStart; ++it) {
        StinkyInstruction& inst = getStinkyInst(it);
        scheduled.push_back(&inst);
    }
    // 3. Count the number of MFMAs and LRs.
    // get the distance with cycles where a LR dst is used.
    std::queue<StinkyInstruction*> scheLR;

    auto scheduleRemainingLRs = [&]() {
        while (!scheLR.empty()) {
            // pop LR
            auto lr = scheLR.front();
            scheduled.push_back(lr);
            scheLR.pop();
        }
    };

    for (BasicBlock::iterator it = regionStart; it != regionEnd; ++it) {
        StinkyInstruction& inst = getStinkyInst(it);
        // inst.dump(std::cout);
        if (isDSRead(inst)) {
            auto dist = getLRDistance(it, endIt, beginIt, inst.getDestRegs());
            // std::cout<<", distance: "<<dist<<std::endl;
            if (dist < inst.latencyCycles) {
                // issue asap
                scheduled.push_back(&inst);
            } else {
                // issue later
                scheLR.push(&inst);
            }
        } else if (isWMMA(inst)) {
            scheduled.push_back(&inst);
            if (!scheLR.empty()) {
                // pop LR
                auto lr = scheLR.front();
                scheduled.push_back(lr);
                scheLR.pop();
            }
        } else {
            if (isBranch(inst)) {
                scheduleRemainingLRs();
            }
            scheduled.push_back(&inst);
        }
    }

    scheduleRemainingLRs();

    for (BasicBlock::iterator it = regionEnd; it != endIt; ++it) {
        StinkyInstruction& inst = getStinkyInst(it);
        scheduled.push_back(&inst);
    }

    assert(scheduled.size() == bb.size() &&
           "Scheduled instructions size must match original instructions size");

    // Now we have a scheduled list of instructions.
    // Reorder the block to reflect the scheduling (move each to end in order).
    for (StinkyInstruction* inst : scheduled) {
        bb.removeIR(inst);
        bb.appendIR(inst);
    }

    // Insert the s_set_vgpr_msb instructions if needed
    currSetVgprMsb = nullptr;
    for (BasicBlock::iterator it = regionStart; it != regionEnd; ++it) {
        StinkyInstruction& inst = getStinkyInst(it);
        if (isBarrier(inst)) {
            break;
        }
        if (isDSRead(inst)) {
            auto lr = lrMapVgprMsb[&inst];
            if (!compareVgprMsb(lr, currSetVgprMsb)) {
                StinkyInstruction* cloned = lr->clone();
                bb.insertIR(it, cloned);
                currSetVgprMsb = lr;
            }
        }
        if (isMFMA(inst) || isWMMA(inst)) {
            auto mfma = mfmaMapVgprMsb[&inst];
            if (!compareVgprMsb(mfma, currSetVgprMsb)) {
                StinkyInstruction* cloned = mfma->clone();
                bb.insertIR(it, cloned);
                currSetVgprMsb = mfma;
            }
        }
        const std::string& opcode = inst.getHwInstDesc()->mnemonic;
        if (opcode.find("s_set_vgpr_msb") != std::string::npos) {
            currSetVgprMsb = &inst;
        }
    }
}

class ScheduleLastLRsPass : public StinkyInstPass {
   public:
    static char ID;

    const char* getName() const override {
        return "ScheduleLastLRsPass";
    }

    PassID getPassID() const override {
        return &ScheduleLastLRsPass::ID;
    }

    void runOnBasicBlock(BasicBlock& bb, PassContext& passCtx) {
        scheduleFinalLocalReadWithLatency(bb, passCtx);
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        for (BasicBlock& bb : func) {
            if (passCtx.shouldProcessBasicBlock(bb)) runOnBasicBlock(bb, passCtx);
        }
        return preserveCFGAnalyses();
    }
};

char ScheduleLastLRsPass::ID = 0;
}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createScheduleLastLRsPass() {
    return std::make_unique<ScheduleLastLRsPass>();
}
}  // namespace stinkytofu
