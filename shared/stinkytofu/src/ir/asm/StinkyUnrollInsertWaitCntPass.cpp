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

#include "ir/asm/StinkyAsmIR.hpp"

#define MAX_WAITCNT 255

namespace
{
    using namespace stinkytofu;

    class StinkyUnrollInsertWaitCntPass : public StinkyInstPass
    {
    public:
        static char ID;

        const char* getName() const override
        {
            return "StinkyUnrollInsertWaitCntPass";
        }

        PassID getPassID() const override
        {
            return &StinkyUnrollInsertWaitCntPass::ID;
        }

        void run(IRList& insts, PassContext& passCtx) override
        {
            if(insts.empty())
                return;

            IRList::iterator it = insts.begin();

            GfxArchID arch = getGfxArchID(passCtx.getKernelInfo().arch[0],
                                          passCtx.getKernelInfo().arch[1],
                                          passCtx.getKernelInfo().arch[2]);

            StinkyInstIRBuilder& irBuilder
                = passCtx.getOrCreateIRBuilder<StinkyInstIRBuilder>(insts, arch);

#ifndef NDEBUG
            // There should be no existing s_waitcnt in the instruction list.
            // while(it != insts.end())
            // {
            //     StinkyInstruction& inst = getStinkyInst(it);
            //     assert(isWaitCnt(inst) == false
            //            && "Internal Error: All s_waitcnt should have been removed in conversion to "
            //               "StinkyInstructions.");
            //     ++it;
            // }
#endif
            // Add a s_waitcnt local wait before each s_barrier
            it = insts.begin();
            while(it != insts.end())
            {
                StinkyInstruction& inst = getStinkyInst(it);
                if(isBarrier(inst))
                {
                    bool foundTensorLoad = false;
                    bool foundDSStore    = false;
                    if(it != insts.begin())
                    {
                        IRList::iterator prevIt = it;
                        do
                        {
                            --prevIt;
                            StinkyInstruction& prevInst = getStinkyInst(prevIt);
                            if(isTensorLoad(prevInst))
                            {
                                foundTensorLoad = true;
                            }
                            if(isDSWrite(prevInst))
                            {
                                foundDSStore = true;
                            }
                            // Stop scanning if another barrier is encountered.
                            if(isBarrier(prevInst))
                            {
                                break;
                            }
                        } while(prevIt != insts.begin());
                    }
                    if(foundTensorLoad)
                    {
                        StinkyInstruction* waitInst = irBuilder.createStinkyInstBefore(
                            it, getMCIDByUOp(GFX::s_wait_tensorcnt, arch));
                        SWaitTensorCntData waitTensorCnt(0);
                        waitInst->addModifier<SWaitTensorCntData>(waitTensorCnt);
                    }

                    int dscnt = foundDSStore ? 0 : -1;
                    // Insert a waitcnt before the barrier.
                    StinkyInstruction* barrier
                        = irBuilder.createStinkyInstBefore(it, getMCIDByUOp(GFX::s_waitcnt, arch));
                    SWaitCntData waitCnt(-1, -1, 0, dscnt, -1);
                    barrier->addModifier<SWaitCntData>(waitCnt);
                }
                ++it;
            }

            // When a global read/ ds read is found, find the next instruction
            // that uses the same register and add a waitcnt before it.
            it = insts.begin();
            while(it != insts.end())
            {
                StinkyInstruction& inst = getStinkyInst(it);

                if(isGlobalMemLoad(inst) || isDSRead(inst))
                {
                    // Find the next instruction that uses the same register.
                    for(const StinkyRegister& reg : inst.destRegs)
                    {
                        int globalReadCount  = isGlobalMemLoad(inst) ? 0 : -1;
                        int globalWriteCount = -1;
                        int dsReadCount      = isDSRead(inst) ? 0 : -1;
                        int dsWriteCount     = -1;

                        IRList::iterator nextIt = it;
                        ++nextIt;
                        if(passCtx.getProperties().containsLoop
                           && nextIt == passCtx.getProperties().loopEnd)
                        {
                            nextIt = passCtx.getProperties().loopBegin;
                        }
                        while(nextIt != it && nextIt != insts.end())
                        {
                            bool               found    = false;
                            StinkyInstruction& nextInst = getStinkyInst(nextIt);

                            if(isGlobalMemLoad(nextInst) || isGlobalMemStore(nextInst))
                            {
                                if(isGlobalMemLoad(nextInst) && isGlobalMemLoad(inst))
                                {
                                    ++globalReadCount;
                                }
                                else if(isGlobalMemStore(nextInst) && isGlobalMemLoad(inst))
                                {
                                    ++globalWriteCount;
                                }
                            }
                            else if(isDSRead(nextInst) || isDSWrite(nextInst))
                            {
                                if(isDSRead(nextInst) && isDSRead(inst))
                                {
                                    ++dsReadCount;
                                }
                                else if(isDSWrite(nextInst) && isDSRead(inst))
                                {
                                    ++dsWriteCount;
                                }
                            }
                            else if(isWaitCnt(nextInst))
                            {
                                const SWaitCntData* waitInst = nextInst.getModifier<SWaitCntData>();
                                assert(waitInst != nullptr);
                                // Update the counts based on the existing waitcnt.
                                auto updateCount = [](int waitValue, int& target) {
                                    int opCount = waitValue == -1 ? MAX_WAITCNT : waitValue;
                                    if(opCount == 0)
                                    {
                                        target = -1;
                                    }
                                    else
                                    {
                                        target = std::min(opCount, target);
                                    }
                                };
                                updateCount(waitInst->vlcnt, globalReadCount);
                                updateCount(waitInst->vscnt, globalWriteCount);
                                updateCount(waitInst->dlcnt, dsReadCount);
                                updateCount(waitInst->dscnt, dsWriteCount);

                                if(isGlobalMemLoad(inst) && globalReadCount < 0)
                                {
                                    // No need to insert waitcnt, as the existing one covers it.
                                    break;
                                }
                                else if(isDSRead(inst) && dsReadCount < 0 && dsWriteCount < 0)
                                {
                                    // No need to insert waitcnt, as the existing one covers it.
                                    break;
                                }
                            }

                            for(const StinkyRegister& nextReg : nextInst.srcRegs)
                            {
                                if(reg.isOverlap(nextReg))
                                {
                                    found = true;
                                    // Found the next instruction that uses this register.
                                    // Insert a waitcnt before this instruction.
                                    // Check if the previous instruction is already a waitcnt.
                                    // If it's already a waitcnt, we can merge it with the current one.
                                    // Otherwise, we need to insert a new waitcnt.
                                    if(nextIt != insts.begin())
                                    {
                                        IRList::iterator prevIt = nextIt;
                                        --prevIt;
                                        StinkyInstruction& prevInst = getStinkyInst(prevIt);
                                        if(isWaitCnt(prevInst))
                                        {
                                            SWaitCntData* waitCnt
                                                = prevInst.getModifier<SWaitCntData>();
                                            assert(waitCnt != nullptr);

                                            // make sure the counts do not exceed int8_t range
                                            assert(globalReadCount <= 127);
                                            assert(globalWriteCount <= 127);
                                            assert(dsReadCount <= 127);
                                            assert(dsWriteCount <= 127);
                                            assert(globalReadCount >= -128);
                                            assert(globalWriteCount >= -128);
                                            assert(dsReadCount >= -128);
                                            assert(dsWriteCount >= -128);

                                            // Merge with the previous waitcnt.
                                            waitCnt->vlcnt
                                                = std::min(waitCnt->vlcnt, (int8_t)globalReadCount);
                                            waitCnt->vscnt = std::min(waitCnt->vscnt,
                                                                      (int8_t)globalWriteCount);
                                            waitCnt->dlcnt
                                                = std::min(waitCnt->dlcnt, (int8_t)dsReadCount);
                                            waitCnt->dscnt
                                                = std::min(waitCnt->dscnt, (int8_t)dsWriteCount);
                                            break;
                                        }
                                    }

                                    // Insert a new waitcnt before the next instruction.
                                    SWaitCntData       waitCnt((int8_t)globalReadCount,
                                                         (int8_t)globalWriteCount,
                                                         (int8_t)(dsReadCount),
                                                         (int8_t)(dsWriteCount),
                                                         -1);
                                    StinkyInstruction* waitInst = irBuilder.createStinkyInstBefore(
                                        nextIt, getMCIDByUOp(GFX::s_waitcnt, arch));
                                    waitInst->addModifier<SWaitCntData>(waitCnt);
                                    break;
                                }
                            }
                            if(found)
                                break;
                            // Move to the next instruction.
                            ++nextIt;
                            if(passCtx.getProperties().containsLoop
                               && nextIt == passCtx.getProperties().loopEnd)
                            {
                                // Reached the end of the instruction list.
                                nextIt = passCtx.getProperties().loopBegin;
                            }
                        }
                    }
                }
                // Move to the next instruction.
                ++it;
            }
        }
    };

    char StinkyUnrollInsertWaitCntPass::ID = 0;
}

namespace stinkytofu
{
    std::unique_ptr<Pass> createStinkyUnrollInsertWaitCntPass()
    {
        return std::make_unique<StinkyUnrollInsertWaitCntPass>();
    }
} // namespace stinkytofu
