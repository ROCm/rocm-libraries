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
#include "stinkytofu/transforms/asm/StinkyBuildImplicitDependencyPass.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include <iostream>

namespace
{
    using namespace stinkytofu;

    void setPseudoRegistersInBlock(BasicBlock& bb, PassContext& passCtx)
    {
        if(!passCtx.getPassFeatureConfig().barrierConfig.unrollMovableBarrier)
            return;

        for(auto it = bb.begin(); it != bb.end(); ++it)
        {
            auto* stinkyInst = dyn_cast<StinkyInstruction>(it.getNodePtr());
            if(!stinkyInst || !isBarrier(*stinkyInst))
                continue;
            stinkyInst->addDestReg(StinkyRegister::getBarrierRegister());
            StinkyInstruction* prev = dyn_cast<StinkyInstruction>(stinkyInst->getPrev());
            while(prev != nullptr)
            {
                if(isMUBUFLoad(*prev))
                {
                    const MUBUFModifiers* mubuf = prev->getModifier<MUBUFModifiers>();
                    if(mubuf && mubuf->glc)
                    {
                        prev->addDestReg(StinkyRegister::getMUBUFLoadRegister());
                        stinkyInst->addSrcReg(StinkyRegister::getMUBUFLoadRegister());
                        prev->addSrcReg(stinkyInst->getDestReg(0));
                    }
                }
                else if(isTensorLoad(*prev))
                {
                    prev->addDestReg(StinkyRegister::getTensorLoadRegister());
                    stinkyInst->addSrcReg(prev->getDestReg(0));
                    prev->addSrcReg(stinkyInst->getDestReg(0));
                }
                else if(isDSRead(*prev))
                {
                    prev->addDestReg(StinkyRegister::getDSReadRegister());
                    stinkyInst->addSrcReg(StinkyRegister::getDSReadRegister());
                    prev->addSrcReg(stinkyInst->getDestReg(0));
                }
                else if(isDSWrite(*prev))
                {
                    prev->addDestReg(StinkyRegister::getDSWriteRegister());
                    stinkyInst->addSrcReg(prev->getDestReg(0));
                    prev->addSrcReg(stinkyInst->getDestReg(0));
                }
                else if(isBarrier(*prev))
                {
                    stinkyInst->addSrcReg(prev->getDestReg(0));
                    break;
                }
                IRBase* prevPrev = prev->getPrev();
                if(prevPrev == nullptr)
                    break;
                prev = dyn_cast<StinkyInstruction>(prevPrev);
            }
        }
    }

    class StinkyBuildImplicitDependencyPass : public StinkyInstPass
    {
    public:
        static char ID;

        const char* getName() const override
        {
            return "StinkyBuildImplicitDependencyPass";
        }

        PassID getPassID() const override
        {
            return &StinkyBuildImplicitDependencyPass::ID;
        }

        void run(Function& func, PassContext& passCtx) override
        {
            for(BasicBlock& bb : func)
            {
                if(passCtx.shouldProcessBasicBlock(bb))
                    setPseudoRegistersInBlock(bb, passCtx);
            }
        }
    };

    char StinkyBuildImplicitDependencyPass::ID = 0;
} // namespace

namespace stinkytofu
{
    std::unique_ptr<Pass> createStinkyBuildImplicitDependencyPass()
    {
        return std::make_unique<StinkyBuildImplicitDependencyPass>();
    }
}
