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
#include "stinkytofu/transforms/asm/ScheduleFirstLRsPass.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <queue>
#include <unordered_map>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

namespace {
using namespace stinkytofu;
static int getUsedBeforeMFMAIndexOfLR(BasicBlock::iterator regStart, BasicBlock::iterator end,
                                      const std::vector<StinkyRegister>& lrDst) {
    int index = -1;
    int usedBeforeIndex = -1;
    for (BasicBlock::iterator it = regStart; it != end; ++it) {
        StinkyInstruction& inst = getStinkyInst(it);
        if (isMFMA(inst) || isWMMA(inst)) {
            index++;
        }
        for (const StinkyRegister& reg : inst.getSrcRegs()) {
            // check overlap
            for (const StinkyRegister& dst : lrDst) {
                if (reg.isOverlap(dst)) {
                    usedBeforeIndex = index;
                }
            }
        }
    }
    return usedBeforeIndex;
}

static int getMFMAIndexOfLR(BasicBlock::iterator regStart, BasicBlock::iterator end,
                            const std::vector<StinkyRegister>& lrDst) {
    int index = -1;
    for (BasicBlock::iterator it = regStart; it != end; ++it) {
        StinkyInstruction& inst = getStinkyInst(it);
        if (isMFMA(inst) || isWMMA(inst)) {
            index++;
        }
        for (const StinkyRegister& reg : inst.getSrcRegs()) {
            // check overlap
            for (const StinkyRegister& dst : lrDst) {
                if (reg.isOverlap(dst)) {
                    return index;
                }
            }
        }
    }
    return index;
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
        // std::cout<<"srcL["<<i<<"]: "<<srcL[i].getLiteralInt()<<", srcR["<<i<<"]:
        // "<<srcR[i].getLiteralInt()<<std::endl;
        if (srcL[i].getLiteralInt() != srcR[i].getLiteralInt()) return false;
    }
    return true;
}

/// If the next queued LR may be scheduled before MFMA index \p mfmaIndex, pop it into
/// \p scheduled. Returns \c false when the caller should break out of the per-MFMA LR loop.
static inline bool tryPopLrForMfmaSchedule(
    std::queue<StinkyInstruction*>& scheLR,
    std::unordered_map<StinkyInstruction*, int>& usedBeforeIndexMap, int mfmaIndex,
    std::vector<StinkyInstruction*>& scheduled) {
    StinkyInstruction* lr = scheLR.front();
    const int usedBeforeIndex = usedBeforeIndexMap[lr];
    if (usedBeforeIndex != -1 && usedBeforeIndex >= mfmaIndex) return false;
    // lr->dump(std::cout);
    scheduled.push_back(lr);
    scheLR.pop();
    return true;
}

// Schedule the First Group of PGRs in the given BasicBlock.
// This will Move the PGR instructions to the suitable position to hide the latency.
//
// In the end, the instructions will be reordered in the block
// to reflect the scheduling order.
void scheduleFirstLocalReadWithLatency(BasicBlock& bb, PassContext& passCtx) {
    if (bb.empty()) return;

    std::vector<StinkyInstruction*> scheduled;
    scheduled.reserve(bb.size());

    BasicBlock::iterator beginIt = bb.begin();
    BasicBlock::iterator endIt = bb.end();

    BasicBlock::iterator regionLoopBeginL = beginIt;

    // 1.0 Find the beginning
    BasicBlock::reverse_iterator revStartIt = bb.rbegin(), revEndIt = bb.rend();
    bool isLoopBeginL = false;
    for (auto rit = revStartIt; rit != revEndIt; ++rit) {
        StinkyInstruction& inst = getStinkyInst(rit);
        if (isLabel(inst) && inst.getModifier<LabelData>()->label == "label_LoopBeginL") {
            regionLoopBeginL = std::next(BasicBlock::iterator(rit.getNodePtr()));
            isLoopBeginL = true;
            break;
        }
    }
    if (!isLoopBeginL) {
        for (BasicBlock::iterator it = beginIt; it != endIt; ++it) {
            StinkyInstruction& inst = getStinkyInst(it);
            scheduled.push_back(&inst);
        }
        return;
    }

    // 1. Find the first barrier
    // Count the number of MFMAs and LRs.
    auto numMFMA = 0;
    auto numLR = 0;
    std::queue<StinkyInstruction*> scheLR;
    std::vector<int> scheLRIndices;
    BasicBlock::iterator regionFirstBarrier = endIt;
    uint32_t mfmaLatency = 0, mfmaIssueCycles = 0;
    std::unordered_map<StinkyInstruction*, int> usedBeforeIndexMap;
    StinkyInstruction* currSetVgprMsb = nullptr;
    for (BasicBlock::iterator it = regionLoopBeginL; it != endIt; ++it) {
        StinkyInstruction& inst = getStinkyInst(it);
        if (isBarrier(inst)) {
            // Start a new region after the side-effect instruction.
            regionFirstBarrier = it;
            break;
        }
        if (isDSRead(inst)) {
            numLR++;
            int index = numMFMA + getMFMAIndexOfLR(it, endIt, inst.getDestRegs());
            int usedBeforeIndex = getUsedBeforeMFMAIndexOfLR(beginIt, it, inst.getDestRegs());
            usedBeforeIndexMap[&inst] = usedBeforeIndex;
            scheLRIndices.push_back(index);
            scheLR.push(&inst);
        }
        if (isMFMA(inst) || isWMMA(inst)) {
            numMFMA++;
            mfmaLatency = inst.latencyCycles;
            mfmaIssueCycles = inst.issueCycles;
        }
    }
    uint32_t numLRPerMFMAAtMost = mfmaLatency - mfmaIssueCycles;
    uint32_t numLRPerMFMAAtLeast = mfmaLatency / 2;
    if (numMFMA > 0) {
        numLRPerMFMAAtLeast = (numLR + numMFMA - 1) / numMFMA;
    }
    // Each LR i must appear before MFMA index scheLRIndices[i]. With numLRPerMFMA LRs
    // issued before MFMA_0 and after each MFMA, LR i sits before MFMA floor(i/N); need
    // floor(i/N) <= S => N >= ceil((i+1)/(S+1)). Take max with the even-spread bound.
    uint32_t numLRPerMFMAFromDeps = 1;
    for (size_t i = 0; i < scheLRIndices.size(); ++i) {
        const int s = scheLRIndices[i];
        if (s >= 0) {
            const uint64_t denom = static_cast<uint64_t>(s) + 1u;
            const uint64_t numer = static_cast<uint64_t>(i) + 1u;
            const uint32_t need = static_cast<uint32_t>((numer + denom - 1u) / denom);
            // std::cout<<"s: "<<s<<", denom: "<<denom<<", numer: "<<numer<<", need:
            // "<<need<<std::endl;
            numLRPerMFMAFromDeps = std::max(numLRPerMFMAFromDeps, need);
        }
    }
    uint32_t numLRPerMFMA = std::max(numLRPerMFMAAtLeast, numLRPerMFMAFromDeps);
    // std::cout<<"numLR: "<<numLR<<", numMFMA: "<<numMFMA<<", numLRPerMFMAAtMost:
    // "<<numLRPerMFMAAtMost<<", numLRPerMFMAAtLeast: "<<numLRPerMFMAAtLeast<<",
    // numLRPerMFMAFromDeps: "<<numLRPerMFMAFromDeps<<", numLRPerMFMA: "<<numLRPerMFMA<<std::endl;

    // 2. Push the instructions before the barrier.
    int currentIndex = 0;
    int mfmaIndex = 0;
    int currentMFMAFreeSpaces = 0;
    for (BasicBlock::iterator it = beginIt; it != regionFirstBarrier; ++it) {
        StinkyInstruction& inst = getStinkyInst(it);
        if (isMFMA(inst) || isWMMA(inst)) {
            for (; currentMFMAFreeSpaces > 0; currentMFMAFreeSpaces--) {
                if (!tryPopLrForMfmaSchedule(scheLR, usedBeforeIndexMap, mfmaIndex, scheduled))
                    break;
            }
            mfmaIndex++;
            scheduled.push_back(&inst);
            currentMFMAFreeSpaces = numLRPerMFMAAtMost;
            // inst.dump(std::cout);
            for (auto i = 0; i < numLRPerMFMA; i++) {
                if (scheLR.empty()) continue;
                if (!tryPopLrForMfmaSchedule(scheLR, usedBeforeIndexMap, mfmaIndex, scheduled))
                    break;
                currentMFMAFreeSpaces--;
            }
            currentIndex++;
        } else if (!isDSRead(inst)) {
            scheduled.push_back(&inst);
            currentMFMAFreeSpaces--;
        }
    }
    while (!scheLR.empty()) {
        // pop LR
        // std::cout<<"pop LR number: "<<scheLR.size()<<std::endl;
        auto lr = scheLR.front();
        scheduled.push_back(lr);
        scheLR.pop();
    }

    for (BasicBlock::iterator it = regionFirstBarrier; it != bb.end(); ++it) {
        StinkyInstruction& inst = getStinkyInst(it);
        scheduled.push_back(&inst);
    }

    assert(scheduled.size() == bb.size() && "Must be the same size as the original block");

    // Reorder bb to match `scheduled`: for each entry, splice that node to the tail.
    // Duplicates are intentional (same StinkyInstruction* appears multiple times); each
    // pass removes the single list node and re-appends it, so |scheduled| may exceed bb.size().
    for (StinkyInstruction* inst : scheduled) {
        bb.removeIR(inst);
        bb.appendIR(inst);
    }
}

class ScheduleFirstLRsPass : public StinkyInstPass {
   public:
    static char ID;

    const char* getName() const override {
        return "ScheduleFirstLRsPass";
    }

    PassID getPassID() const override {
        return &ScheduleFirstLRsPass::ID;
    }

    void runOnBasicBlock(BasicBlock& bb, PassContext& passCtx) {
        scheduleFirstLocalReadWithLatency(bb, passCtx);
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        for (BasicBlock& bb : func) {
            if (passCtx.shouldProcessBasicBlock(bb)) runOnBasicBlock(bb, passCtx);
        }
        return preserveCFGAnalyses();
    }
};

char ScheduleFirstLRsPass::ID = 0;
}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createScheduleFirstLRsPass() {
    return std::make_unique<ScheduleFirstLRsPass>();
}
}  // namespace stinkytofu
