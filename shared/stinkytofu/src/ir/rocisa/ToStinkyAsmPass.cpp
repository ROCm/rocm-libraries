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
#include <set>
#include <typeindex>

#include "code.hpp"
#include "instruction/branch.hpp"
#include "instruction/cmp.hpp"
#include "instruction/common.hpp"
#include "instruction/cvt.hpp"
#include "instruction/mem.hpp"
#include "instruction/mfma.hpp"
#include "ir/asm/StinkyAsmIR.hpp"
#include "ir/rocisa/AllHwMappings.hpp"
#include "ir/rocisa/ToStinkyAsm.hpp"

namespace stinkytofu
{
    static void getFlatItemsInDepthFirst(const rocisa::Module&       module,
                                         std::vector<rocisa::Item*>& flatItems)
    {
        for(const std::shared_ptr<rocisa::Item>& itemShared : module.itemList)
        {
            rocisa::Item* item = itemShared.get();
            if(rocisa::Module* subMod = dynamic_cast<rocisa::Module*>(item))
            {
                getFlatItemsInDepthFirst(*subMod, flatItems);
            }
            else
            {
                flatItems.push_back(item);
            }
        }
    }

    void RocisaDFSFlatItems::run(IRList& irlist, PassContext& passCtx)
    {
        getFlatItemsInDepthFirst(module, flatItems);
    }

    std::unique_ptr<AnalysisPass> createRocisaDFSFlatItemsPass(rocisa::Module& module)
    {
        return std::make_unique<RocisaDFSFlatItems>(module);
    }

    Pass::ID RocisaDFSFlatItems::ID = &RocisaDFSFlatItems::ID;

    std::unique_ptr<AnalysisPass> createRocisaStinkyMappingPass()
    {
        return std::make_unique<RocisaStinkyMapping>();
    }

    Pass::ID RocisaStinkyMapping::ID = &RocisaStinkyMapping::ID;
}

namespace
{
    using namespace rocisa;
    using namespace stinkytofu;

    StinkyRegister getStinkyRegister(const Container* container)
    {
        if(const RegisterContainer* regCont = dynamic_cast<const RegisterContainer*>(container))
        {
            return StinkyRegister{regCont->regType, regCont->regIdx, regCont->regNum};
        }
        return StinkyRegister{};
    }

    StinkyRegister getStinkyRegister(const InstructionInput& input)
    {
        if(auto pptr = std::get_if<std::shared_ptr<Container>>(&input))
        {
            if(auto regContainer = std::dynamic_pointer_cast<RegisterContainer>(*pptr))
            {
                return StinkyRegister{
                    regContainer->regType, regContainer->regIdx, regContainer->regNum};
            }
        }
        else if(const int* literalInt = std::get_if<int>(&input))
        {
            return StinkyRegister(*literalInt);
        }
        else if(const double* literalDouble = std::get_if<double>(&input))
        {
            return StinkyRegister(*literalDouble);
        }
        else if(const std::string* literalString = std::get_if<std::string>(&input))
        {
            return StinkyRegister(*literalString);
        }

        // TODO: Should we allow unhandled cases?
        return StinkyRegister{};
    }

    bool doesReadSCC(const Instruction* inst)
    {
        // See ISA: 5.3. Scalar Condition Code (SCC)
        //
        // TODO: Handle all instructions that read SCC.
        if(dynamic_cast<const SCSelectB32*>(inst))
        {
            return true;
        }
        return false;
    }

    bool doesWriteSCC(const Instruction* inst)
    {
        // See ISA: 5.3. Scalar Condition Code (SCC)
        //
        // TODO: Handle all instructions that write to SCC.
        if(dynamic_cast<const SCmpEQI32*>(inst) || dynamic_cast<const SCmpEQU32*>(inst)
           || dynamic_cast<const SSubU32*>(inst) || dynamic_cast<const SAddU32*>(inst)
           || dynamic_cast<const SAddCU32*>(inst) || dynamic_cast<const SSubBU32*>(inst))
        {
            return true;
        }
        return false;
    }

    bool isWaitCntInstruction(const rocisa::Instruction& inst)
    {
        return inst.instType == rocisa::InstType::INST_SWAIT;
    }

    // Convert the module's flat items into StinkyInstructions.
    // This will populate the `insts` list with StinkyInstructions.
    // Each StinkyInstruction will have its source and destination registers populated.
    // It will also handle SCC registers if the instruction reads or writes them.
    class RocisaToStinkyAsmPass : public StinkyInstPass
    {
    public:
        static char ID;
        const char* getName() const override
        {
            return "RocisaToStinkyAsmPass";
        }

        PassID getPassID() const override
        {
            return &RocisaToStinkyAsmPass::ID;
        }

        RocisaToStinkyAsmPass(bool doesIgnoreWaitCnt)
            : StinkyInstPass()
            , ignoreWaitCnt(doesIgnoreWaitCnt)
        {
        }

        bool ignoreWaitCnt = false;

        void run(IRList& insts, PassContext& passCtx) override
        {
            RocisaDFSFlatItems& flatItems
                = passCtx.getAnalysisManager().getResult<RocisaDFSFlatItems>(insts, passCtx);
            RocisaStinkyMapping& mapping
                = passCtx.getAnalysisManager().getResult<RocisaStinkyMapping>(insts, passCtx);

            GfxArchID arch = getGfxArchID(passCtx.getKernelInfo().arch[0],
                                          passCtx.getKernelInfo().arch[1],
                                          passCtx.getKernelInfo().arch[2]);

            StinkyInstIRBuilder& irBuilder
                = passCtx.getOrCreateIRBuilder<StinkyInstIRBuilder>(insts, arch);

            assert(insts.empty() && "Instruction list must be empty before populating");

            StinkyInstruction* lastBarrierInst = nullptr;
            for(Item* item : flatItems.getFlatItems())
            {
                if(dynamic_cast<Label*>(item))
                {
                    rocisa::Label* rocLabel = dynamic_cast<Label*>(item);
                    auto label = irBuilder.createStinkyLabel(insts.end(), rocLabel->getLabelName());

                    mapping.addMapping(label, rocLabel);
                    continue;
                }

                Instruction* inst = dynamic_cast<Instruction*>(item);
                if((inst == nullptr) || (ignoreWaitCnt && isWaitCntInstruction(*inst)))
                {
                    continue;
                }

                StinkyInstruction* stinkyInst = nullptr;
                std::type_index    rocisaTy   = std::type_index(typeid(*inst));
                const HwInstDesc*  hwInstDesc = getRocisaToMCID(rocisaTy, arch);

                if(hwInstDesc != nullptr)
                {
                    stinkyInst = irBuilder.createStinkyInstBefore(insts.end(), hwInstDesc);
                }
                else
                {
                    ConvertRocisaToHwInstFunc convFn = getConvertRocisaToHwInstFunc(rocisaTy, arch);
                    assert(convFn != nullptr && "TODO: unhandled rocisa type");
                    std::vector<StinkyInstruction*> stinkyInsts = convFn(*inst, irBuilder, insts);

                    assert(stinkyInsts.size() == 1 && "TODO: handle multiple stinky instructions.");
                    stinkyInst = stinkyInsts[0];
                }

                mapping.addMapping(stinkyInst, inst);

                // TODO: Use unordered_set when the StinkyRegister is no longer using std::string.
                std::set<StinkyRegister> uniqueSrcRegs, uniqueDstRegs;

                for(const InstructionInput& dst : inst->getDstParams())
                {
                    StinkyRegister reg = getStinkyRegister(dst);
                    if(uniqueDstRegs.find(reg) == uniqueDstRegs.end())
                    {
                        uniqueDstRegs.insert(reg);
                        stinkyInst->destRegs.push_back(reg);
                    }
                }

                for(const InstructionInput& src : inst->getSrcParams())
                {
                    StinkyRegister reg = getStinkyRegister(src);
                    if(reg.isValid() && uniqueSrcRegs.find(reg) == uniqueSrcRegs.end())
                    {
                        uniqueSrcRegs.insert(reg);
                        stinkyInst->srcRegs.push_back(reg);
                    }
                }

                if(doesReadSCC(inst))
                {
                    stinkyInst->srcRegs.push_back(StinkyRegister::getSCCRegister());
                }

                if(doesWriteSCC(inst))
                {
                    stinkyInst->destRegs.push_back(StinkyRegister::getSCCRegister());
                }

                if(isBranch(*stinkyInst))
                {
                    stinkyInst->addModifier<LabelData>(
                        LabelData{Modifier::Type::LABEL_NAME,
                                  dynamic_cast<BranchInstruction*>(inst)->labelName});
                }

                // should read from passCtx->getKernelInfo() to see if it's gemm loop
                // It's gemm specialized barrier handling
                if(passCtx.getOptInfo().unrollGemmMovableBarrier)
                {
                    if(dynamic_cast<const SBarrier*>(inst))
                    {
                        lastBarrierInst         = stinkyInst;
                        StinkyInstruction* prev = cast<StinkyInstruction>(stinkyInst->getPrev());
                        stinkyInst->destRegs.push_back(StinkyRegister::getBarrierRegister());
                        while(prev != nullptr)
                        {
                            if(isMUBUFLoad(*prev))
                            {
                                const stinkytofu::MUBUFModifiers* mubuf
                                    = prev->getModifier<stinkytofu::MUBUFModifiers>();
                                if(mubuf && mubuf->glc)
                                {
                                    stinkyInst->srcRegs.push_back(prev->destRegs[0]);
                                    prev->srcRegs.push_back(stinkyInst->destRegs[0]);
                                }
                            }
                            else if(isTensorLoad(*prev))
                            {
                                prev->destRegs.push_back(StinkyRegister::getTensorLoadRegister());
                                stinkyInst->srcRegs.push_back(prev->destRegs[0]);
                                prev->srcRegs.push_back(stinkyInst->destRegs[0]);
                            }
                            else if(isDSRead(*prev))
                            {
                                stinkyInst->srcRegs.push_back(prev->destRegs[0]);
                                prev->srcRegs.push_back(stinkyInst->destRegs[0]);
                            }
                            else if(isDSWrite(*prev))
                            {
                                prev->destRegs.push_back(StinkyRegister::getDSWriteRegister());
                                stinkyInst->srcRegs.push_back(prev->destRegs[0]);
                                prev->srcRegs.push_back(stinkyInst->destRegs[0]);
                            }
                            else if(isBarrier(*prev))
                            {
                                stinkyInst->srcRegs.push_back(prev->destRegs[0]);
                                break;
                            }

                            if(prev->getPrev() == nullptr)
                            {
                                prev = nullptr;
                                break;
                            }
                            prev = cast<StinkyInstruction>(prev->getPrev());
                        }
                    }
                }
            }

            if(lastBarrierInst == nullptr || lastBarrierInst->getNext() == nullptr)
            {
                return;
            }

            IRBase* nextIR = lastBarrierInst->getNext();
            while(nextIR != nullptr)
            {
                StinkyInstruction* next = cast<StinkyInstruction>(nextIR);
                if(isBarrier(*next))
                {
                    break;
                }
                else if(isMUBUFLoad(*next))
                {
                    const stinkytofu::MUBUFModifiers* mubuf
                        = next->getModifier<stinkytofu::MUBUFModifiers>();
                    if(mubuf && mubuf->glc)
                    {
                        next->srcRegs.push_back(StinkyRegister::getBarrierRegister());
                    }
                }
                else if(isDSRead(*next) || isDSWrite(*next))
                {
                    next->srcRegs.push_back(StinkyRegister::getBarrierRegister());
                }
                nextIR = next->getNext();
            }

            // Check if the input IRList contains a loop.
            std::vector<IntrusiveListIterator<IRBase>> loopLabels;
            for(auto it = insts.begin(); it != insts.end(); ++it)
            {
                StinkyInstruction* inst = cast<StinkyInstruction>(it.getNodePtr());
                if(inst->getUnifiedOpcode() == GFX::LABEL)
                {
                    loopLabels.push_back(it);
                }
                else if(isBranch(*inst))
                {
                    auto branchTarget = inst->getModifier<LabelData>();
                    if(branchTarget)
                    {
                        for(auto targetLabel : loopLabels)
                        {
                            StinkyInstruction* label
                                = cast<StinkyInstruction>(targetLabel.getNodePtr());
                            if(branchTarget->label == label->getModifier<LabelData>()->label)
                            {
                                passCtx.setLoopProperties(true, targetLabel, it);
                                break;
                            }
                        }
                    }
                }
            }
        }
    };

    char RocisaToStinkyAsmPass::ID = 0;
}

namespace stinkytofu
{
    std::unique_ptr<Pass> createRocisaToStinkyAsmPass(bool doesIgnoreWaitCnt)
    {
        return std::make_unique<RocisaToStinkyAsmPass>(doesIgnoreWaitCnt);
    }
}
