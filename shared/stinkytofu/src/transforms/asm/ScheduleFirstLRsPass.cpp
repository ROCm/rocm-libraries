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
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

#include <memory>
#include <queue>

namespace
{
    using namespace stinkytofu;

    static int getLRDistance(BasicBlock::iterator regStart,
                            BasicBlock::iterator end,
                            std::vector<StinkyRegister>& lrDst)
    {
        int cycles = 0;
        for(BasicBlock::iterator it = regStart; it != end; ++it)
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
    // Schedule the First Group of PGRs in the given BasicBlock.
    // This will Move the PGR instructions to the suitable position to hide the latency.
    //
    // In the end, the instructions will be reordered in the block
    // to reflect the scheduling order.
    void scheduleFirstLocalReadWithLatency(BasicBlock& bb, PassContext& passCtx)
    {
        if(bb.empty())
            return;

        std::vector<StinkyInstruction*> scheduled;
        scheduled.reserve(bb.size());

        BasicBlock::iterator beginIt = bb.begin();
        BasicBlock::iterator endIt   = bb.end();

        BasicBlock::iterator regionEnd = beginIt;

        // 1. Find the first barrier
        // Count the number of MFMAs and LRs.
        auto                           numMFMA = 0;
        auto                           numLR   = 0;
        std::queue<StinkyInstruction*> scheLR;
        for(BasicBlock::iterator it = beginIt; it != endIt; ++it)
        {
            StinkyInstruction& inst = getStinkyInst(it);
            if(isBarrier(inst))
            {
                // Start a new region after the side-effect instruction.
                regionEnd = it;
                break;
            }
            if(isDSRead(inst))
            {
                numLR++;
                scheLR.push(&inst);
            }
            if(isMFMA(inst) || isWMMA(inst))
            {
                numMFMA++;
            }
        }
        uint32_t numLRPerMFMA = numLR;
        if(numMFMA > 0)
        {
            numLRPerMFMA = (numLR + numMFMA - 1) / numMFMA;
        }

        // 2. Push the instructions before the barrier.
        bool canIssueLR = false;
        for(auto i = 0; i < numLRPerMFMA; i++)
        {
            // issue the first numLRPerMFMA LRs
            if(!scheLR.empty())
            {
                // pop LR
                auto lr = scheLR.front();
                scheduled.push_back(lr);
                scheLR.pop();
            }
        }
        for(BasicBlock::iterator it = beginIt; it != regionEnd; ++it)
        {
            StinkyInstruction& inst = getStinkyInst(it);
            if(isDSRead(inst))
            {
                //skip
            }
            if(isMFMA(inst) || isWMMA(inst))
            {
                scheduled.push_back(&inst);
                for(auto i = 0; i < numLRPerMFMA; i++)
                {
                    // issue the first numLRPerMFMA LRs
                    if(!scheLR.empty())
                    {
                        // pop LR
                        auto lr = scheLR.front();
                        scheduled.push_back(lr);
                        scheLR.pop();
                    }
                }
            }
            else
            {
                scheduled.push_back(&inst);
            }
        }
        while(!scheLR.empty())
        {
            // pop LR
            auto lr = scheLR.front();
            scheduled.push_back(lr);
            scheLR.pop();
        }

        for(BasicBlock::iterator it = regionEnd; it != bb.end(); ++it)
        {
            StinkyInstruction& inst = getStinkyInst(it);
            scheduled.push_back(&inst);
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

    class ScheduleFirstLRsPass : public StinkyInstPass
    {
    public:
        static char ID;

        const char* getName() const override
        {
            return "ScheduleFirstLRsPass";
        }

        PassID getPassID() const override
        {
            return &ScheduleFirstLRsPass::ID;
        }

        void runOnBasicBlock(BasicBlock& bb, PassContext& passCtx)
        {
            scheduleFirstLocalReadWithLatency(bb, passCtx);
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

    char ScheduleFirstLRsPass::ID = 0;
}

namespace stinkytofu
{
    std::unique_ptr<Pass> createScheduleFirstLRsPass()
    {
        return std::make_unique<ScheduleFirstLRsPass>();
    }
}
