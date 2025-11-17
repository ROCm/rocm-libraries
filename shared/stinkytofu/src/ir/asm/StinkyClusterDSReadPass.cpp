/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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

#include <memory>

#include "ir/asm/StinkyAsmIR.hpp"

namespace
{
    using namespace stinkytofu;

    // This pass will group ds_read as close as possible
    // and put them right after the barrier or fence.
    class StinkyClusterDSReadPass : public StinkyInstPass
    {
    public:
        static char ID;

        const char* getName() const override
        {
            return "StinkyClusterDSReadPass";
        }

        PassID getPassID() const override
        {
            return &StinkyClusterDSReadPass::ID;
        }

        void run(IRList& insts, PassContext& passCtx) override
        {
            GfxArchID arch = getGfxArchID(passCtx.getKernelInfo().arch[0],
                                          passCtx.getKernelInfo().arch[1],
                                          passCtx.getKernelInfo().arch[2]);

            StinkyInstIRBuilder& irBuilder
                = passCtx.getOrCreateIRBuilder<StinkyInstIRBuilder>(insts, arch);

            IntrusiveListIterator<IRBase> begin = insts.begin();
            IntrusiveListIterator<IRBase> end   = insts.end();
            if(passCtx.getProperties().containsLoop)
            {
                begin = passCtx.getProperties().loopBegin;
                end   = passCtx.getProperties().loopEnd;
            }
            // Add a dummy barrier at the beginning to handle ds_read at the start.
            StinkyInstruction* dummyBarrier
                = irBuilder.createStinkyInstBefore(begin, getMCIDByUOp(GFX::s_barrier, arch));

            // The barrier/ fence comes first, then the ds_read.
            // We memorize the position of the last barrier/ fence.
            IRList::iterator                lastBarrierIt = end;
            IRList::iterator                currBarrierIt = begin;
            std::vector<StinkyInstruction*> dsReadGroup;
            // Plan moves first to avoid modifying the list while iterating.
            struct MoveBatch
            {
                StinkyInstruction*              barrier;
                std::vector<StinkyInstruction*> reads;
            };
            std::vector<MoveBatch> moveBatches;

            for(auto it = begin; it != end; ++it)
            {
                StinkyInstruction& inst = getStinkyInst(it);
                if(isBarrier(inst) || isBranch(inst))
                {
                    lastBarrierIt = currBarrierIt;
                    if(currBarrierIt != end && !dsReadGroup.empty())
                    {
                        MoveBatch batch;
                        batch.barrier = &getStinkyInst(currBarrierIt);
                        batch.reads   = dsReadGroup;
                        moveBatches.push_back(std::move(batch));
                        dsReadGroup.clear();
                    }
                    currBarrierIt = it;
                }
                else if(isDSRead(inst))
                {
                    dsReadGroup.push_back(&inst);
                }
            }

            // Record remaining ds_read group to move after the last seen barrier/fence.
            if(currBarrierIt != lastBarrierIt && !dsReadGroup.empty())
            {
                MoveBatch batch;
                batch.barrier = &getStinkyInst(currBarrierIt);
                batch.reads   = dsReadGroup;
                moveBatches.push_back(std::move(batch));
                dsReadGroup.clear();
            }

            // Apply planned moves after iteration to avoid iterator invalidation.
            auto findIteratorByPtr = [&](StinkyInstruction* ptr) {
                for(auto it = insts.begin(); it != insts.end(); ++it)
                {
                    if(&(*it) == ptr)
                        return it;
                }
                return insts.end();
            };

            for(auto& batch : moveBatches)
            {
                auto barrierIt2 = findIteratorByPtr(batch.barrier);
                if(barrierIt2 == insts.end())
                    continue;

                auto insertPos = barrierIt2;
                for(auto dsPtr : batch.reads)
                {
                    auto dsIt = findIteratorByPtr(dsPtr);
                    insts.moveAfter(dsIt, insertPos);
                    insertPos = std::next(insertPos);
                }
            }

            // Remove the dummy barrier.
            irBuilder.erase(dummyBarrier);
        }
    };

    char StinkyClusterDSReadPass::ID = 0;
}

namespace stinkytofu
{
    std::unique_ptr<Pass> createStinkyClusterDSReadPass()
    {
        return std::make_unique<StinkyClusterDSReadPass>();
    }
}
