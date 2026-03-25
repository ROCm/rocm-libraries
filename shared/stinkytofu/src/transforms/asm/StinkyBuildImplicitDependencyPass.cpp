/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the Software), to deal
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
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include <iostream>
#include <vector>

#define DEBUG_TYPE "StinkyBuildImplicitDependencyPass"

namespace
{
    using namespace stinkytofu;

    // A linkable op between two barriers is wired twice; keep src/dest operand lists from
    // appending the same pseudo register again (list hygiene, not SSA "defines").
    static const StinkyRegister& uniquePseudoDest(StinkyInstruction& inst,
                                                  RegType                 kind,
                                                  const StinkyRegister&   proto)
    {
        for(const StinkyRegister& d : inst.getDestRegs())
            if(d.reg.type == kind)
                return d;
        inst.addDestReg(proto);
        return inst.getDestReg(inst.getNumDestRegs() - 1);
    }

    static void uniqueSrc(StinkyInstruction& inst, const StinkyRegister& r)
    {
        for(const StinkyRegister& s : inst.getSrcRegs())
            if(s == r)
                return;
        inst.addSrcReg(r);
    }

    static void linkNeighborToBarrier(StinkyInstruction& neighbor, StinkyInstruction& barrier)
    {
        if(isMUBUFLoad(neighbor))
        {
            const MUBUFModifiers* mubuf = neighbor.getModifier<MUBUFModifiers>();
            if(!mubuf || !mubuf->glc)
                return;
            uniquePseudoDest(neighbor, RegType::MUBUF_LOAD, StinkyRegister::getMUBUFLoadRegister());
            barrier.addSrcReg(StinkyRegister::getMUBUFLoadRegister());
            uniqueSrc(neighbor, barrier.getDestReg(0));
            PASS_DEBUG(std::cerr << "[BuildImplicitDep]   link GLC MUBUF load <-> barrier\n");
            return;
        }
        if(isTensorLoad(neighbor))
        {
            const StinkyRegister& p = uniquePseudoDest(
                neighbor, RegType::TENSOR_LOAD, StinkyRegister::getTensorLoadRegister());
            barrier.addSrcReg(p);
            uniqueSrc(neighbor, barrier.getDestReg(0));
            PASS_DEBUG(std::cerr << "[BuildImplicitDep]   link tensor_load_to_lds <-> barrier "
                                    "(tensor pseudo + barrier reg)\n");
            return;
        }
        if(isDSRead(neighbor))
        {
            uniquePseudoDest(neighbor, RegType::DS_READ, StinkyRegister::getDSReadRegister());
            barrier.addSrcReg(StinkyRegister::getDSReadRegister());
            uniqueSrc(neighbor, barrier.getDestReg(0));
            PASS_DEBUG(std::cerr << "[BuildImplicitDep]   link ds_read <-> barrier\n");
            return;
        }
        if(isDSWrite(neighbor))
        {
            const StinkyRegister& p = uniquePseudoDest(
                neighbor, RegType::DS_WRITE, StinkyRegister::getDSWriteRegister());
            barrier.addSrcReg(p);
            uniqueSrc(neighbor, barrier.getDestReg(0));
            PASS_DEBUG(std::cerr << "[BuildImplicitDep]   link ds_write <-> barrier\n");
        }
    }

    static bool isLinkableImplicitNeighbor(const StinkyInstruction& inst)
    {
        if(isTensorLoad(inst) || isDSRead(inst) || isDSWrite(inst))
            return true;
        if(isMUBUFLoad(inst))
        {
            const MUBUFModifiers* mubuf = inst.getModifier<MUBUFModifiers>();
            return mubuf && mubuf->glc;
        }
        return false;
    }

    void setPseudoRegistersInBlock(BasicBlock& bb, PassContext& passCtx)
    {
        if(!passCtx.getPassFeatureConfig().barrierConfig.unrollMovableBarrier)
        {
            PASS_DEBUG(std::cerr << "[BuildImplicitDep] skip BB label=\"" << bb.getLabel()
                                 << "\" (unrollMovableBarrier=false)\n");
            return;
        }

        StinkyInstruction*              lastBarrier = nullptr;
        std::vector<StinkyInstruction*> pending;

        for(auto it = bb.begin(); it != bb.end(); ++it)
        {
            auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
            if(!inst)
                continue;

            if(isBarrier(*inst))
            {
                inst->addDestReg(StinkyRegister::getBarrierRegister());
                PASS_DEBUG(std::cerr << "[BuildImplicitDep] movable barrier BB label=\"" << bb.getLabel()
                                     << "\" barrier hasBarrierDest=true\n");
                if(lastBarrier != nullptr)
                {
                    inst->addSrcReg(lastBarrier->getDestReg(0));
                    PASS_DEBUG(std::cerr << "[BuildImplicitDep]   stop at prev barrier\n");
                }
                for(StinkyInstruction* op : pending)
                    linkNeighborToBarrier(*op, *inst);
                pending.clear();
                lastBarrier = inst;
            }
            else if(isLinkableImplicitNeighbor(*inst))
            {
                pending.push_back(inst);
                if(lastBarrier != nullptr)
                    linkNeighborToBarrier(*inst, *lastBarrier);
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
