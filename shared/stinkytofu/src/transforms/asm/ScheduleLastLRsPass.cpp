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

    static int getLRDistance(IRList&                            insts,
                             IRList::iterator                   regStart,
                             IRList::iterator                   instsBegin,
                             IRList::iterator                   instsEnd,
                             const std::vector<StinkyRegister>& lrDst)
    {
        int cycles = 0;
        for(IRList::iterator it = regStart; it != instsEnd; ++it)
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

        for(IRList::iterator it = instsBegin; it != regStart; ++it)
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
    // Schedule the Final PGR in the given IRList.
    // This will Move the PGR instructions to the suitable position to hide the latency.
    //
    // In the end, the instructions will be reordered in the IRList
    // to reflect the scheduling order.
    void scheduleFinalLocalReadWithLatency(IRList& insts, PassContext& passCtx)
    {
        if(insts.empty())
            return;

        std::vector<StinkyInstruction*> scheduled;
        scheduled.reserve(insts.size());

        IntrusiveListIterator<IRBase> beginIt = insts.begin();
        IntrusiveListIterator<IRBase> endIt   = insts.end();

        IRList::iterator regionStart = beginIt;

        // 1. Find the last barrier
        IRList::reverse_iterator revStartIt = insts.rbegin(), revEndIt = insts.rend();

        for(auto rit = revStartIt; rit != revEndIt; ++rit)
        {
            StinkyInstruction& inst = getStinkyInst(rit);
            if(isBarrier(inst))
            {
                // Start a new region after the side-effect instruction.
                regionStart = std::next(IRList::iterator(rit.getNodePtr()));
                break;
            }
        }
        // 2. Push the instructions before the barrier.
        for(IRList::iterator it = beginIt; it != regionStart; ++it)
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

        for(IRList::iterator it = regionStart; it != endIt; ++it)
        {
            StinkyInstruction& inst = getStinkyInst(it);
            if(isDSRead(inst))
            {
                numLR++;
                auto dist = getLRDistance(insts, it, beginIt, endIt, inst.getDestRegs());
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

        if(endIt != insts.end())
        {
            for(IRList::iterator it = endIt; it != insts.end(); ++it)
            {
                StinkyInstruction& inst = getStinkyInst(it);
                scheduled.push_back(&inst);
            }
        }

        assert(scheduled.size() == insts.size()
               && "Scheduled instructions size must match original instructions size");

        // Now we have a scheduled list of instructions.
        // Modify the original insts list to reflect the scheduling.
        for(StinkyInstruction* inst : scheduled)
        {
            insts.moveBefore(IRList::iterator(inst), insts.end());
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
            IRList& irlist = bb.getIR();
            scheduleFinalLocalReadWithLatency(irlist, passCtx);
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
