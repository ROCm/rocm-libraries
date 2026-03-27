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
#include <unordered_map>
#include <vector>

#define DEBUG_TYPE "StinkyBuildImplicitDependencyPass"

namespace
{
    using namespace stinkytofu;

    // A linkable op between two barriers is wired twice; keep src/dest operand lists from
    // appending the same pseudo register again (list hygiene, not SSA "defines").
    // For PSEUDO registers, also match by index since PSEUDO[0] != PSEUDO[1].
    static const StinkyRegister& uniquePseudoDest(StinkyInstruction& inst,
                                                  RegType                 kind,
                                                  const StinkyRegister&   proto)
    {
        for(const StinkyRegister& d : inst.getDestRegs())
        {
            if(d.reg.type == kind)
            {
                if(kind == RegType::PSEUDO)
                {
                    if(d.reg.idx == proto.reg.idx)
                        return d;
                }
                else
                    return d;
            }
        }
        inst.addDestReg(proto);
        return inst.getDestReg(inst.getNumDestRegs() - 1);
    }

    static void uniqueSrc(StinkyInstruction& inst, const StinkyRegister& r)
    {
        for(const StinkyRegister& s : inst.getSrcRegs())
            if(s.reg.type == r.reg.type && s.reg.idx == r.reg.idx)
                return;
        inst.addSrcReg(r);
    }

    static void linkViaMemToken(StinkyInstruction& neighbor, StinkyInstruction& barrier,
                                bool beforeBarrier)
    {
        const MemTokenData* neighborMt = neighbor.getModifier<MemTokenData>();
        const MemTokenData* barrierMt  = barrier.getModifier<MemTokenData>();
        if(!neighborMt || !barrierMt)
            return;

        for(int tokenId : neighborMt->tokens)
        {
            // Only connect when the same token ID exists in both
            if(std::find(barrierMt->tokens.begin(), barrierMt->tokens.end(), tokenId)
               == barrierMt->tokens.end())
                continue;

            StinkyRegister pseudo(RegType::PSEUDO, tokenId, 1);
            if(beforeBarrier)
            {
                uniquePseudoDest(neighbor, RegType::PSEUDO, pseudo);
                uniqueSrc(barrier, pseudo);
            }
            else
            {
                uniquePseudoDest(barrier, RegType::PSEUDO, pseudo);
                uniqueSrc(neighbor, pseudo);
            }
            PASS_DEBUG(std::cerr << "[BuildImplicitDep]   link via mem_token " << tokenId
                                 << " <-> barrier" << (beforeBarrier ? " (before)" : " (after)")
                                 << "\n");
        }
    }

    static void linkNeighborToBarrier(StinkyInstruction& neighbor, StinkyInstruction& barrier,
                                      bool beforeBarrier)
    {
        if(neighbor.getModifier<MemTokenData>())
        {
            linkViaMemToken(neighbor, barrier, beforeBarrier);
            return;
        }

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
        if(inst.getModifier<MemTokenData>())
            return true;
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

        std::unordered_map<int, StinkyInstruction*> lastBarrierByToken;
        // Track unpaired s_barrier_signal instructions awaiting their s_barrier_wait.
        std::unordered_map<int, StinkyInstruction*> pendingSignalByToken;
        std::vector<StinkyInstruction*> pending;

        // Helper lambda: process a barrier instruction that acts as the entry point
        // (drains pending ops) and optionally the exit point (updates lastBarrierByToken).
        // For s_barrier (gfx9): single inst is both entry and exit.
        // For split barriers (gfx1250): signal is entry, wait is exit.
        auto processBarrierEntry = [&](StinkyInstruction* inst, const MemTokenData* mt) {
            for(int tokenId : mt->tokens)
                uniquePseudoDest(
                    *inst, RegType::PSEUDO, StinkyRegister(RegType::PSEUDO, tokenId, 1));

            auto newEnd = std::remove_if(
                pending.begin(), pending.end(), [&](StinkyInstruction* op) {
                    const MemTokenData* opMt = op->getModifier<MemTokenData>();
                    assert(opMt
                           && "mixed mem-token and non-mem-token ops in same BB");
                    for(int t : opMt->tokens)
                    {
                        if(std::find(mt->tokens.begin(), mt->tokens.end(), t)
                           != mt->tokens.end())
                        {
                            linkNeighborToBarrier(*op, *inst, true);
                            return true;
                        }
                    }
                    return false;
                });
            pending.erase(newEnd, pending.end());
        };

        auto processBarrierExit = [&](StinkyInstruction* inst, const MemTokenData* mt) {
            for(int tokenId : mt->tokens)
                uniquePseudoDest(
                    *inst, RegType::PSEUDO, StinkyRegister(RegType::PSEUDO, tokenId, 1));
            for(int t : mt->tokens)
                lastBarrierByToken[t] = inst;
        };

        for(auto it = bb.begin(); it != bb.end(); ++it)
        {
            auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
            if(!inst)
                continue;

            if(isBarrier(*inst))
            {
                const MemTokenData* mt = inst->getModifier<MemTokenData>();
                if(!mt)
                {
                    PASS_DEBUG(std::cerr << "[BuildImplicitDep] non-mem-token barrier BB label=\""
                                         << bb.getLabel() << "\" — skipping (side-effect)\n");
                    assert(pending.empty()
                           && "mixed mem-token and non-mem-token ops in same BB");
                    continue;
                }

                // Split barriers with a non-(-1) operand are workgroup-sync mode:
                // only one wave signals, pattern is "if wave==0: signal; all wg wait;"
                // which splits signal/wait across different BBs. Skip these — we'll
                // never find a matching pair in the same BB.
                if((isBarrierSignal(*inst) || isBarrierWait(*inst))
                   && !isSplitBarrierAllWave(*inst))
                {
                    PASS_DEBUG(
                        std::cerr
                        << "[BuildImplicitDep] non-all-wave split barrier BB label=\""
                        << bb.getLabel()
                        << "\" — skipping (workgroup-sync mode, signal/wait in different BBs)\n");
                    continue;
                }

                if(isBarrierSignal(*inst))
                {
                    // s_barrier_signal: entry point only — drain pending ops, but don't
                    // update lastBarrierByToken (the paired s_barrier_wait will do that).
                    PASS_DEBUG(std::cerr << "[BuildImplicitDep] barrier_signal BB label=\""
                                         << bb.getLabel() << "\" tokens=[";
                               for(int t : mt->tokens) std::cerr << t << " ";
                               std::cerr << "]\n");
                    processBarrierEntry(inst, mt);
                    for(int t : mt->tokens)
                        pendingSignalByToken[t] = inst;
                }
                else if(isBarrierWait(*inst))
                {
                    // s_barrier_wait: exit point — link from the paired signal (if any)
                    // and become the lastBarrier for subsequent ops.
                    PASS_DEBUG(std::cerr << "[BuildImplicitDep] barrier_wait BB label=\""
                                         << bb.getLabel() << "\" tokens=[";
                               for(int t : mt->tokens) std::cerr << t << " ";
                               std::cerr << "]\n");
                    // Connect signal → wait via pseudo registers
                    for(int t : mt->tokens)
                    {
                        auto sit = pendingSignalByToken.find(t);
                        if(sit != pendingSignalByToken.end())
                        {
                            StinkyRegister pseudo(RegType::PSEUDO, t, 1);
                            uniquePseudoDest(*sit->second, RegType::PSEUDO, pseudo);
                            uniqueSrc(*inst, pseudo);
                            pendingSignalByToken.erase(sit);
                        }
                    }
                    processBarrierExit(inst, mt);
                }
                else
                {
                    // s_barrier (gfx9): single barrier is both entry and exit.
                    PASS_DEBUG(std::cerr << "[BuildImplicitDep] movable barrier BB label=\""
                                         << bb.getLabel() << "\" tokens=[";
                               for(int t : mt->tokens) std::cerr << t << " ";
                               std::cerr << "]\n");
                    processBarrierEntry(inst, mt);
                    processBarrierExit(inst, mt);
                }
            }
            else if(isLinkableImplicitNeighbor(*inst))
            {
                const MemTokenData* opMt = inst->getModifier<MemTokenData>();
                if(!opMt)
                    continue;
                pending.push_back(inst);
                for(int t : opMt->tokens)
                {
                    auto bit = lastBarrierByToken.find(t);
                    if(bit != lastBarrierByToken.end())
                        linkNeighborToBarrier(*inst, *bit->second, false);
                }
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
