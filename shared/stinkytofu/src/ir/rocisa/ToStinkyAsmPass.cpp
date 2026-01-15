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
#include "isa/ArchHelper.hpp"

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

    void RocisaDFSFlatItems::run(Function& func, PassContext& passCtx)
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

    // Helper functions to convert rocisa modifiers to stinkytofu modifiers
    stinkytofu::DSModifiers convertDSModifiers(const rocisa::DSModifiers& rocMod)
    {
        return stinkytofu::DSModifiers(
            rocMod.na, rocMod.offset, rocMod.offset0, rocMod.offset1, rocMod.gds);
    }

    stinkytofu::FLATModifiers convertFLATModifiers(const rocisa::FLATModifiers& rocMod)
    {
        return stinkytofu::FLATModifiers(
            rocMod.offset12, rocMod.glc, rocMod.slc, rocMod.lds, rocMod.isStore);
    }

    stinkytofu::MUBUFModifiers convertMUBUFModifiers(const rocisa::MUBUFModifiers& rocMod)
    {
        return stinkytofu::MUBUFModifiers(rocMod.offen,
                                          rocMod.offset12,
                                          rocMod.glc,
                                          rocMod.slc,
                                          rocMod.nt,
                                          rocMod.lds,
                                          rocMod.isStore);
    }

    stinkytofu::SMEMModifiers convertSMEMModifiers(const rocisa::SMEMModifiers& rocMod)
    {
        return stinkytofu::SMEMModifiers(rocMod.glc, rocMod.nv, rocMod.offset);
    }

    StinkyRegister getStinkyRegister(const Container* container)
    {
        if(const RegisterContainer* regCont = dynamic_cast<const RegisterContainer*>(container))
        {
            // Convert string regType to RegType enum
            RegType        regType = stringToRegType(regCont->regType);
            StinkyRegister reg{regType, regCont->regIdx, regCont->regNum};

            // Capture symbolic register name if available
            // In rocisa, the symbolic name includes the type prefix and all offsets
            // (e.g., "vgprLocalWriteAddrA+0" or "vgprValuA_X0_I0+4")
            if(regCont->regName.has_value())
            {
                // regName->toString() includes the base name and all offsets
                std::string fullName = regCont->regType + "gpr" + regCont->regName->toString();
                reg.setSymbolicName(fullName);
            }

            return reg;
        }
        return StinkyRegister{};
    }

    StinkyRegister getStinkyRegister(const InstructionInput& input)
    {
        if(auto pptr = std::get_if<std::shared_ptr<Container>>(&input))
        {
            if(auto regContainer = std::dynamic_pointer_cast<RegisterContainer>(*pptr))
            {
                // Convert string regType to RegType enum
                RegType        regType = stringToRegType(regContainer->regType);
                StinkyRegister reg{regType, regContainer->regIdx, regContainer->regNum};

                // Capture symbolic register name if available
                // In rocisa, the symbolic name includes the type prefix and all offsets
                // (e.g., "vgprLocalWriteAddrA+0" or "vgprValuA_X0_I0+4")
                if(regContainer->regName.has_value())
                {
                    // regName->toString() includes the base name and all offsets
                    std::string fullName
                        = regContainer->regType + "gpr" + regContainer->regName->toString();
                    reg.setSymbolicName(fullName);
                }

                return reg;
            }
            else if(auto hwregContainer = std::dynamic_pointer_cast<HWRegContainer>(*pptr))
            {
                // Handle hardware register containers like hwreg(26,4,1)
                // These should be emitted as literal strings in the assembly
                return StinkyRegister(hwregContainer->toString());
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
            // Try to convert numeric strings to integers
            if(!literalString->empty())
            {
                bool   isNumeric = true;
                size_t start     = 0;

                // Check for optional leading minus sign
                if((*literalString)[0] == '-')
                {
                    start = 1;
                    if(literalString->length() == 1)
                    {
                        isNumeric = false;
                    }
                }

                // Check if all remaining characters are digits
                for(size_t i = start; i < literalString->length() && isNumeric; ++i)
                {
                    if(!std::isdigit(static_cast<unsigned char>((*literalString)[i])))
                    {
                        isNumeric = false;
                    }
                }

                if(isNumeric)
                {
                    int value = std::atoi(literalString->c_str());
                    return StinkyRegister(value);
                }
            }
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

        void run(Function& func, PassContext& passCtx) override
        {
            RocisaDFSFlatItems& flatItems
                = passCtx.getAnalysisManager().getResult<RocisaDFSFlatItems>(func, passCtx);
            RocisaStinkyMapping& mapping
                = passCtx.getAnalysisManager().getResult<RocisaStinkyMapping>(func, passCtx);

            GfxArchID arch = getGfxArchID(passCtx.getGemmTileConfig().arch[0],
                                          passCtx.getGemmTileConfig().arch[1],
                                          passCtx.getGemmTileConfig().arch[2]);

            // Create a single BasicBlock to hold all IR (common case)
            BasicBlock* bb = func.createBasicBlock("entry");
            func.setEntryBlock(bb);
            IRList& insts = bb->getIR();

            auto irBuilder = passCtx.getIRBuilder<StinkyInstIRBuilder>(insts, arch);

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
                        stinkyInst->addDestReg(reg);
                    }
                }

                for(const InstructionInput& src : inst->getSrcParams())
                {
                    StinkyRegister reg = getStinkyRegister(src);
                    if(reg.isValid() && uniqueSrcRegs.find(reg) == uniqueSrcRegs.end())
                    {
                        uniqueSrcRegs.insert(reg);
                        stinkyInst->addSrcReg(reg);
                    }
                }

                if(doesReadSCC(inst))
                {
                    stinkyInst->addSrcReg(StinkyRegister::getSCCRegister());
                }

                if(doesWriteSCC(inst))
                {
                    stinkyInst->addDestReg(StinkyRegister::getSCCRegister());
                }

                // Copy modifiers from rocisa instruction to StinkyInstruction
                // DS (Local Memory) modifiers
                if(auto dsLoad = dynamic_cast<const DSLoadInstruction*>(inst))
                {
                    if(dsLoad->ds.has_value())
                    {
                        stinkyInst->addModifier<stinkytofu::DSModifiers>(
                            convertDSModifiers(dsLoad->ds.value()));
                    }
                }
                else if(auto dsStore = dynamic_cast<const DSStoreInstruction*>(inst))
                {
                    if(dsStore->ds.has_value())
                    {
                        stinkyInst->addModifier<stinkytofu::DSModifiers>(
                            convertDSModifiers(dsStore->ds.value()));
                    }
                }
                // FLAT (Flat Memory) modifiers
                else if(auto flatRead = dynamic_cast<const FLATReadInstruction*>(inst))
                {
                    if(flatRead->flat.has_value())
                    {
                        stinkyInst->addModifier<stinkytofu::FLATModifiers>(
                            convertFLATModifiers(flatRead->flat.value()));
                    }
                }
                else if(auto flatStore = dynamic_cast<const FLATStoreInstruction*>(inst))
                {
                    if(flatStore->flat.has_value())
                    {
                        stinkyInst->addModifier<stinkytofu::FLATModifiers>(
                            convertFLATModifiers(flatStore->flat.value()));
                    }
                }
                // MUBUF (Buffer Memory) modifiers
                else if(auto mubufRead = dynamic_cast<const MUBUFReadInstruction*>(inst))
                {
                    if(mubufRead->mubuf.has_value())
                    {
                        stinkyInst->addModifier<stinkytofu::MUBUFModifiers>(
                            convertMUBUFModifiers(mubufRead->mubuf.value()));
                    }
                }
                else if(auto mubufStore = dynamic_cast<const MUBUFStoreInstruction*>(inst))
                {
                    if(mubufStore->mubuf.has_value())
                    {
                        stinkyInst->addModifier<stinkytofu::MUBUFModifiers>(
                            convertMUBUFModifiers(mubufStore->mubuf.value()));
                    }
                }
                // SMEM (Scalar Memory) modifiers
                else if(auto smemLoad = dynamic_cast<const SMemLoadInstruction*>(inst))
                {
                    if(smemLoad->smem.has_value())
                    {
                        stinkyInst->addModifier<stinkytofu::SMEMModifiers>(
                            convertSMEMModifiers(smemLoad->smem.value()));
                    }
                }
                else if(auto smemStore = dynamic_cast<const SMemStoreInstruction*>(inst))
                {
                    if(smemStore->smem.has_value())
                    {
                        stinkyInst->addModifier<stinkytofu::SMEMModifiers>(
                            convertSMEMModifiers(smemStore->smem.value()));
                    }
                }
                else if(auto smemAtomic = dynamic_cast<const SMemAtomicDecInstruction*>(inst))
                {
                    if(smemAtomic->smem.has_value())
                    {
                        stinkyInst->addModifier<stinkytofu::SMEMModifiers>(
                            convertSMEMModifiers(smemAtomic->smem.value()));
                    }
                }

                // Copy comment from rocisa instruction to StinkyInstruction
                if(!inst->comment.empty())
                {
                    stinkyInst->addModifier<CommentData>(CommentData{inst->comment});
                }

                if(isBranch(*stinkyInst))
                {
                    stinkyInst->addModifier<LabelData>(
                        LabelData{Modifier::Type::LABEL_NAME,
                                  dynamic_cast<BranchInstruction*>(inst)->labelName});
                }

                // should read from passCtx->getGemmTileConfig() to see if it's gemm loop
                // It's gemm specialized barrier handling
                if(passCtx.getPassFeatureConfig().barrierConfig.unrollMovableBarrier)
                {
                    if(dynamic_cast<const SBarrier*>(inst))
                    {
                        lastBarrierInst         = stinkyInst;
                        StinkyInstruction* prev = cast<StinkyInstruction>(stinkyInst->getPrev());
                        stinkyInst->addDestReg(StinkyRegister::getBarrierRegister());
                        while(prev != nullptr)
                        {
                            if(isMUBUFLoad(*prev))
                            {
                                const stinkytofu::MUBUFModifiers* mubuf
                                    = prev->getModifier<stinkytofu::MUBUFModifiers>();
                                if(mubuf && mubuf->glc)
                                {
                                    stinkyInst->addSrcReg(prev->getDestRegs()[0]);
                                    prev->addSrcReg(stinkyInst->getDestRegs()[0]);
                                }
                            }
                            else if(isTensorLoad(*prev))
                            {
                                prev->addDestReg(StinkyRegister::getTensorLoadRegister());
                                stinkyInst->addSrcReg(prev->getDestRegs()[0]);
                                prev->addSrcReg(stinkyInst->getDestRegs()[0]);
                            }
                            else if(isDSRead(*prev))
                            {
                                stinkyInst->addSrcReg(prev->getDestRegs()[0]);
                                prev->addSrcReg(stinkyInst->getDestRegs()[0]);
                            }
                            else if(isDSWrite(*prev))
                            {
                                prev->addDestReg(StinkyRegister::getDSWriteRegister());
                                stinkyInst->addSrcReg(prev->getDestRegs()[0]);
                                prev->addSrcReg(stinkyInst->getDestRegs()[0]);
                            }
                            else if(isBarrier(*prev))
                            {
                                stinkyInst->addSrcReg(prev->getDestRegs()[0]);
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
                        next->addSrcReg(StinkyRegister::getBarrierRegister());
                    }
                }
                else if(isDSRead(*next) || isDSWrite(*next))
                {
                    next->addSrcReg(StinkyRegister::getBarrierRegister());
                }
                nextIR = next->getNext();
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
