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
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

#include <memory>
#include <queue>

namespace
{
    using namespace stinkytofu;

    static int getLRDistance(BasicBlock::iterator                regStart,
                             BasicBlock::iterator                instsEnd,
                             BasicBlock::iterator                instsBegin,
                             const std::vector<StinkyRegister>&   lrDst)
    {
        int cycles = 0;
        for(BasicBlock::iterator it = regStart; it != instsEnd; ++it)
        {
            StinkyInstruction& inst = getStinkyInst(it);
            for(const StinkyRegister& reg : inst.getSrcRegs())
            {
                // check overlap
                for(const StinkyRegister& dst : lrDst)
                {
                    if(reg.isOverlap(dst))
                    {
                        return cycles;
                    }
                }
            }
            if(isMFMA(inst) || isWMMA(inst))
            {
                cycles += inst.latencyCycles;
            }
            else
            {
                cycles += inst.issueCycles;
            }
        }

        for(BasicBlock::iterator it = instsBegin; it != regStart; ++it)
        {
            StinkyInstruction& inst = getStinkyInst(it);
            for(const StinkyRegister& reg : inst.getSrcRegs())
            {
                // check overlap
                for(const StinkyRegister& dst : lrDst)
                {
                    if(reg.isOverlap(dst))
                    {
                        return cycles;
                    }
                }
            }
            if(isMFMA(inst) || isWMMA(inst))
            {
                cycles += inst.latencyCycles;
            }
            else
            {
                cycles += inst.issueCycles;
            }
        }
        return cycles;
    }
    // Schedule the Final PGR in the given BasicBlock.
    // This will Move the PGR instructions to the suitable position to hide the latency.
    //
    // In the end, the instructions will be reordered in the block
    // to reflect the scheduling order.
    void scheduleFinalLocalReadWithLatency(BasicBlock& bb, PassContext& passCtx)
    {
        if(bb.empty())
            return;

        std::vector<StinkyInstruction*> scheduled;
        scheduled.reserve(bb.size());

        BasicBlock::iterator beginIt = bb.begin();
        BasicBlock::iterator endIt   = bb.end();

        BasicBlock::iterator regionStart = beginIt;

        // 1. Find the last barrier
        BasicBlock::reverse_iterator revStartIt = bb.rbegin(), revEndIt = bb.rend();

        for(auto rit = revStartIt; rit != revEndIt; ++rit)
        {
            StinkyInstruction& inst = getStinkyInst(rit);
            if(isBarrier(inst))
            {
                // Start a new region after the side-effect instruction.
                regionStart = std::next(BasicBlock::iterator(rit.getNodePtr()));
                break;
            }
        }
        // 2. Push the instructions before the barrier.
        for(BasicBlock::iterator it = beginIt; it != regionStart; ++it)
        {
            StinkyInstruction& inst = getStinkyInst(it);
            scheduled.push_back(&inst);
        }
        // 3. Count the number of MFMAs and LRs.
        // get the distance with cycles where a LR dst is used.
        auto                           numMFMA = 0;
        auto                           numLR   = 0;
        std::queue<StinkyInstruction*> scheLR;

        auto scheduleRemainingLRs = [&]() {
            while(!scheLR.empty())
            {
                // pop LR
                auto lr = scheLR.front();
                scheduled.push_back(lr);
                scheLR.pop();
            }
        };

        for(BasicBlock::iterator it = regionStart; it != endIt; ++it)
        {
            StinkyInstruction& inst = getStinkyInst(it);
            if(isDSRead(inst))
            {
                numLR++;
                auto dist = getLRDistance(it, endIt, beginIt, inst.getDestRegs());
                if(dist < inst.latencyCycles)
                {
                    // issue asap
                    scheduled.push_back(&inst);
                }
                else
                {
                    // issue later
                    scheLR.push(&inst);
                }
            }
            else if(isMFMA(inst) || isWMMA(inst))
            {
                numMFMA++;
                scheduled.push_back(&inst);
                if(!scheLR.empty())
                {
                    // pop LR
                    auto lr = scheLR.front();
                    scheduled.push_back(lr);
                    scheLR.pop();
                }
            }
            else
            {
                if(isBranch(inst))
                {
                    scheduleRemainingLRs();
                }
                scheduled.push_back(&inst);
            }
        }

        scheduleRemainingLRs();

        if(endIt != bb.end())
        {
            for(BasicBlock::iterator it = endIt; it != bb.end(); ++it)
            {
                StinkyInstruction& inst = getStinkyInst(it);
                scheduled.push_back(&inst);
            }
        }

        assert(scheduled.size() == bb.size()
               && "Scheduled instructions size must match original instructions size");

        // Now we have a scheduled list of instructions.
        // Reorder the block to reflect the scheduling (move each to end in order).
        for(StinkyInstruction* inst : scheduled)
        {
            bb.removeIR(inst);
            bb.appendIR(inst);
        }
    }

    class ScheduleLastLRsPass : public StinkyInstPass
    {
    public:
        static char ID;

        const char* getName() const override
        {
            return "ScheduleLastLRsPass";
        }

        PassID getPassID() const override
        {
            return &ScheduleLastLRsPass::ID;
        }

        void runOnBasicBlock(BasicBlock& bb, PassContext& passCtx)
        {
            scheduleFinalLocalReadWithLatency(bb, passCtx);
        }

        void run(Function& func, PassContext& passCtx) override
        {
            for(BasicBlock& bb : func)
            {
                if(passCtx.shouldProcessBasicBlock(bb))
                    runOnBasicBlock(bb, passCtx);
            }
        }
    };

    char ScheduleLastLRsPass::ID = 0;
}

namespace stinkytofu
{
    std::unique_ptr<Pass> createScheduleLastLRsPass()
    {
        return std::make_unique<ScheduleLastLRsPass>();
    }
}
