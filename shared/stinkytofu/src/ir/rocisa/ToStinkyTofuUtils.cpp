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
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include "code.hpp"
#include "container.hpp"
#include "instruction/branch.hpp"
#include "instruction/cmp.hpp"
#include "instruction/common.hpp"
#include "instruction/mem.hpp"
#include "ir/asm/StinkyAsmDirectives.hpp"
#include "ir/asm/StinkyAsmIR.hpp"
#include "ir/asm/StinkyAsmModule.hpp"
#include "ir/asm/StinkySignature.hpp"
#include "ir/rocisa/AllHwMappings.hpp"
#include "isa/ArchHelper.hpp"
#include "stinkytofu.hpp"

namespace nb = nanobind;

namespace stinkytofu
{
    // Forward declarations for use in anonymous namespace
    StinkyRegister                 toStinkyRegister(const rocisa::Container* container);
    StinkyRegister                 toStinkyRegister(const InstructionInput& input);
    std::shared_ptr<SignatureBase> toStinkySignature(const rocisa::SignatureBase& rocisaSig,
                                                     const std::array<int, 3>&    isaVersion,
                                                     int wavefrontSize = 64);
    std::shared_ptr<KernelBody>    toStinkyKernelBody(const rocisa::KernelBody& rocisaKernel,
                                                      const std::array<int, 3>& isaVersion);
} // namespace stinkytofu

namespace
{
    using namespace rocisa;
    using namespace stinkytofu;

    /**
     * @brief Check if an instruction reads the SCC register
     */
    bool doesReadSCC(const Instruction* inst)
    {
        if(dynamic_cast<const SCSelectB32*>(inst))
        {
            return true;
        }
        return false;
    }

    /**
     * @brief Check if an instruction writes the SCC register
     */
    bool doesWriteSCC(const Instruction* inst)
    {
        if(dynamic_cast<const SCmpEQI32*>(inst) || dynamic_cast<const SCmpEQU32*>(inst)
           || dynamic_cast<const SSubU32*>(inst) || dynamic_cast<const SAddU32*>(inst)
           || dynamic_cast<const SAddCU32*>(inst) || dynamic_cast<const SSubBU32*>(inst))
        {
            return true;
        }
        return false;
    }

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

    bool getVgprMsbSetValue(const rocisa::Instruction& inst, int& setVal)
    {
        int  msbSrc[3] = {0, 0, 0};
        int  msbDst    = 0;
        bool hasVgpr   = false;

        const auto srcs = inst.getSrcParams();
        for(size_t i = 0; i < srcs.size() && i < 3; ++i)
        {
            if(auto pptr = std::get_if<std::shared_ptr<rocisa::Container>>(&srcs[i]))
            {
                if(*pptr == nullptr)
                    continue;
                auto gpr = dynamic_cast<rocisa::RegisterContainer*>(pptr->get());
                if(gpr && gpr->regType == "v")
                {
                    msbSrc[i] = gpr->msb;
                    hasVgpr   = true;
                }
            }
        }

        const auto dsts = inst.getDstParams();
        for(const auto& dst : dsts)
        {
            if(auto pptr = std::get_if<std::shared_ptr<rocisa::Container>>(&dst))
            {
                if(*pptr == nullptr)
                    continue;
                auto gpr = dynamic_cast<rocisa::RegisterContainer*>(pptr->get());
                if(gpr && gpr->regType == "v")
                {
                    msbDst  = gpr->msb;
                    hasVgpr = true;
                    break;
                }
            }
        }

        setVal = msbSrc[0] + (msbSrc[1] << 2) + (msbSrc[2] << 4) + (msbDst << 6);
        return hasVgpr;
    }

    /**
     * @brief Handle VGPR MSB setting if needed
     * @return The new MSB value that was set, or -1 if no change
     */
    int handleVgprMsbSetting(const rocisa::Instruction& inst,
                             int                        currentVgprMsb,
                             StinkyInstIRBuilder&       irBuilder,
                             IRList&                    insts,
                             GfxArchID                  archId)
    {
        int setVal = 0;
        if(getVgprMsbSetValue(inst, setVal) && setVal != currentVgprMsb)
        {
            const HwInstDesc* desc = getMCIDByUOp(GFX::s_set_vgpr_msb, archId);
            assert(desc != nullptr && "s_set_vgpr_msb is not supported on this architecture");
            StinkyInstruction* msbInst = irBuilder.createStinkyInstBefore(insts.end(), desc);
            msbInst->addSrcReg(StinkyRegister(setVal));
            return setVal;
        }
        return currentVgprMsb;
    }

    /**
     * @brief Post-process IRList to expand VNop instructions into multiple individual v_nop instructions
     *
     * In rocisa, VNop(count=N) represents N separate v_nop instructions.
     * This function expands a single VNop instruction with count > 1 into multiple individual v_nop instructions.
     */
    void expandNopInstructions(IRList& insts, StinkyInstIRBuilder& irBuilder, GfxArchID archId)
    {
        // Expand VNop instructions in place with a single pass
        for(auto it = insts.begin(); it != insts.end(); ++it)
        {
            StinkyInstruction* inst = dyn_cast<StinkyInstruction>(&*it);
            if(!inst)
                continue;

            // Check for v_nop using unified opcode
            if(inst->getUnifiedOpcode() != GFX::v_nop)
                continue;

            const auto& srcRegs = inst->getSrcRegs();
            if(srcRegs.empty() || srcRegs[0].dataType != StinkyRegister::Type::LiteralInt)
                continue;

            int count = srcRegs[0].literalInt;
            if(count <= 1)
                continue;

            // Found a v_nop with count > 1, expand it in place
            const HwInstDesc* desc = inst->getHwInstDesc();

            // Get comment if present
            const CommentData* comment    = inst->getModifier<CommentData>();
            std::string        commentStr = comment ? comment->comment : "";

            // Create 'count' separate nop instructions before the original
            for(int i = 0; i < count; ++i)
            {
                StinkyInstruction* newInst = irBuilder.createStinkyInstBefore(it, desc);

                // v_nop has no operands - just add comment
                if(!commentStr.empty())
                {
                    newInst->addModifier<CommentData>(CommentData{commentStr});
                }
            }

            --it; // Move back so iterator points to the last expanded nop

            // Remove the original nop instruction and adjust iterator
            irBuilder.erase(inst);
        }
    }

    /**
     * @brief Post-process IRList to lower VCmpX instructions on architectures that need it
     *
     * On gfx10+ (RDNA) architectures, v_cmpx instructions don't directly write to SGPR.
     * We need to convert: v_cmpx_* exec_lo, src0, src1
     * Into: v_cmp_* vcc_lo, src0, src1
     *       s_mov_b32 exec_lo, vcc_lo
     */
    void lowerVCmpXInstructions(IRList& insts, StinkyInstIRBuilder& irBuilder, GfxArchID archId)
    {
        // Check if this architecture needs VCmpX lowering
        const std::map<std::string, int> asmCaps = rocisa::rocIsa::getInstance().getAsmCaps();
        auto                             it      = asmCaps.find("CMPXWritesSGPR");
        bool                             cmpxWritesSGPR = (it != asmCaps.end() && it->second);

        if(cmpxWritesSGPR)
        {
            // No lowering needed on this architecture
            return;
        }

        // Get wavefront size to determine if we use vcc/exec or vcc_lo/exec_lo
        int wavefrontSize = rocisa::rocIsa::getInstance().getKernel().wavefront;

        // Iterate through instructions and lower v_cmpx to v_cmp + s_mov
        for(auto it = insts.begin(); it != insts.end(); ++it)
        {
            StinkyInstruction* inst = dyn_cast<StinkyInstruction>(&*it);
            if(!inst)
                continue;

            const HwInstDesc* desc = inst->getHwInstDesc();
            assert(desc != nullptr
                   && "Instruction descriptor is not supported on this architecture");
            assert(desc->mnemonic != nullptr && "Missing mnemonic in instruction descriptor");

            std::string mnemonic(desc->mnemonic);
            size_t      pos = mnemonic.find("_cmpx_");
            if(pos == std::string::npos)
                continue;

            // This is a v_cmpx instruction that needs lowering
            // Check if destination is EXEC
            const auto& destRegs = inst->getDestRegs();
            if(destRegs.empty())
                continue;

            const StinkyRegister& destReg = destRegs[0];
            bool                  isExecDest
                = (destReg.reg.type == RegType::EXEC || destReg.reg.type == RegType::EXEC_LO);

            if(!isExecDest)
                continue;

            // Replace v_cmpx with v_cmp
            mnemonic.replace(pos, 6, "_cmp_");

            uint16_t          cmpOpcode = getMnemonicToIsaOpcode(mnemonic, archId);
            const HwInstDesc* cmpDesc   = getMCIDByIsaOp(cmpOpcode, archId);

            assert(cmpDesc != nullptr && "v_cmp_* is not supported on this architecture");

            // Create new v_cmp instruction
            StinkyInstruction* cmpInst = irBuilder.createStinkyInstBefore(it, cmpDesc);

            // Replace EXEC destination with VCC
            StinkyRegister vccReg;
            if(wavefrontSize == 32)
            {
                vccReg = StinkyRegister(RegType::VCC_LO, 0, 1);
            }
            else
            {
                vccReg = StinkyRegister(RegType::VCC, 0, 1);
            }
            cmpInst->addDestReg(vccReg);

            // Copy source registers
            for(const auto& srcReg : inst->getSrcRegs())
            {
                cmpInst->addSrcReg(srcReg);
            }

            // Copy modifiers (comment, etc.)
            if(const CommentData* comment = inst->getModifier<CommentData>())
            {
                cmpInst->addModifier<CommentData>(*comment);
            }

            // Create s_mov exec, vcc instruction
            const HwInstDesc* movDesc;
            StinkyRegister    execReg;
            if(wavefrontSize == 32)
            {
                movDesc = getMCIDByUOp(GFX::s_mov_b32, archId);
                execReg = StinkyRegister(RegType::EXEC_LO, 0, 1);
            }
            else
            {
                movDesc = getMCIDByUOp(GFX::s_mov_b64, archId);
                execReg = StinkyRegister(RegType::EXEC, 0, 1);
            }

            assert(movDesc != nullptr
                   && "s_mov_b32 or s_mov_b64 is not supported on this architecture");

            StinkyInstruction* movInst = irBuilder.createStinkyInstBefore(it, movDesc);
            movInst->addDestReg(execReg);
            movInst->addSrcReg(vccReg);

            // Adjust iterator to the new cmp instruction.
            ++it;

            // Remove the original v_cmpx instruction using builder
            irBuilder.erase(inst);
        }
    }

    /**
     * @brief Create StinkyInstruction from rocisa instruction
     */
    StinkyInstruction* createStinkyInstructionFromRocisa(rocisa::Instruction& inst,
                                                         std::type_index      rocisaTy,
                                                         StinkyInstIRBuilder& irBuilder,
                                                         IRList&              insts,
                                                         GfxArchID            archId)
    {
        const HwInstDesc*  hwInstDesc = getRocisaToMCID(rocisaTy, archId);
        StinkyInstruction* stinkyInst = nullptr;

        if(hwInstDesc != nullptr)
        {
            // Direct mapping exists - create instruction directly
            stinkyInst = irBuilder.createStinkyInstBefore(insts.end(), hwInstDesc);
        }
        else
        {
            // Need conversion function
            ConvertRocisaToHwInstFunc convFn = getConvertRocisaToHwInstFunc(rocisaTy, archId);
            if(convFn == nullptr)
            {
                return nullptr;
            }

            // Call conversion function
            std::vector<StinkyInstruction*> stinkyInsts = convFn(inst, irBuilder, insts);

            if(stinkyInsts.empty())
            {
                return nullptr;
            }

            // For now, only handle single instruction conversions
            stinkyInst = stinkyInsts[0];
        }

        return stinkyInst;
    }

    /**
     * @brief Add source and destination registers to StinkyInstruction
     */
    void addRegistersToInstruction(StinkyInstruction*                stinkyInst,
                                   const rocisa::Instruction*        inst,
                                   const std::map<std::string, int>& asmCaps,
                                   GfxArchID                         archId)
    {
        std::set<StinkyRegister> uniqueSrcRegs, uniqueDstRegs;

        // Add destination registers
        for(const InstructionInput& dst : inst->getDstParams())
        {
            StinkyRegister reg = stinkytofu::toStinkyRegister(dst);

            // Only add valid registers
            if(reg.isValid() && uniqueDstRegs.find(reg) == uniqueDstRegs.end())
            {
                uniqueDstRegs.insert(reg);
                stinkyInst->addDestReg(reg);
            }
        }

        // Add source registers
        std::vector<InstructionInput> srcParams = inst->getSrcParams();

        // Adjust source parameters for VLShiftLeftAddU32 instructions
        if(typeid(*inst) == typeid(rocisa::VLShiftLeftAddU32)
           || typeid(*inst) == typeid(rocisa::_VLShiftLeftAddU32))
        {
            auto it         = asmCaps.find("HasAddLshl");
            bool hasAddLshl = (it != asmCaps.end() && it->second);
            if(hasAddLshl)
            {
                // rocisa CompositeInstruction order is src0, src1, shift;
                // HasAddLshl expects src0, shift, src1
                std::swap(srcParams[1], srcParams[2]);
            }
        }

        for(const InstructionInput& src : srcParams)
        {
            StinkyRegister reg = stinkytofu::toStinkyRegister(src);
            if(reg.isValid() && uniqueSrcRegs.find(reg) == uniqueSrcRegs.end())
            {
                uniqueSrcRegs.insert(reg);
                stinkyInst->addSrcReg(reg);
            }
        }

        // Add SCC register if needed
        if(doesReadSCC(inst))
        {
            stinkyInst->addSrcReg(StinkyRegister::getSCCRegister());
        }

        if(doesWriteSCC(inst))
        {
            stinkyInst->addDestReg(StinkyRegister::getSCCRegister());
        }
    }

    /**
     * @brief Add modifiers to StinkyInstruction (DS, FLAT, MUBUF, SMEM, WaitCnt)
     */
    void addModifiersToInstruction(StinkyInstruction* stinkyInst, const rocisa::Instruction* inst)
    {
        // Macro to reduce boilerplate for memory modifier handling
#define TRY_ADD_MOD(RocisaInstType, modField, StinkyModType, converter)                 \
    if(auto typed = dynamic_cast<const RocisaInstType*>(inst))                          \
    {                                                                                   \
        if(typed->modField.has_value())                                                 \
        {                                                                               \
            stinkyInst->addModifier<StinkyModType>(converter(typed->modField.value())); \
        }                                                                               \
    }                                                                                   \
    else

        // Chain all memory instruction types
        TRY_ADD_MOD(DSLoadInstruction, ds, stinkytofu::DSModifiers, convertDSModifiers)
        TRY_ADD_MOD(DSStoreInstruction, ds, stinkytofu::DSModifiers, convertDSModifiers)
        TRY_ADD_MOD(FLATReadInstruction, flat, stinkytofu::FLATModifiers, convertFLATModifiers)
        TRY_ADD_MOD(FLATStoreInstruction, flat, stinkytofu::FLATModifiers, convertFLATModifiers)
        TRY_ADD_MOD(MUBUFReadInstruction, mubuf, stinkytofu::MUBUFModifiers, convertMUBUFModifiers)
        TRY_ADD_MOD(MUBUFStoreInstruction, mubuf, stinkytofu::MUBUFModifiers, convertMUBUFModifiers)
        TRY_ADD_MOD(SMemLoadInstruction, smem, stinkytofu::SMEMModifiers, convertSMEMModifiers)
        TRY_ADD_MOD(SMemStoreInstruction, smem, stinkytofu::SMEMModifiers, convertSMEMModifiers)
        TRY_ADD_MOD(SMemAtomicDecInstruction, smem, stinkytofu::SMEMModifiers, convertSMEMModifiers)
        {
        }
        // WaitCnt (special case - different structure, no optional field)
        if(auto waitCntInst = dynamic_cast<const rocisa::SWaitCnt*>(inst))
        {
            SWaitCntData waitCntData(
                waitCntInst->vlcnt, waitCntInst->vscnt, -1, waitCntInst->dscnt, waitCntInst->kmcnt);
            stinkyInst->addModifier<SWaitCntData>(waitCntData);
        }

#undef TRY_ADD_MOD

        // Always add comment if present
        if(!inst->comment.empty())
        {
            stinkyInst->addModifier<CommentData>(CommentData{inst->comment});
        }
    }

} // anonymous namespace

namespace stinkytofu
{
    /**
     * @brief Convert a rocisa::Container to StinkyRegister
     *
     * This function takes a rocisa::Container pointer and converts it to a
     * StinkyRegister. It handles RegisterContainer types by extracting the
     * register type, index, and number of registers.
     *
     * @param container Pointer to rocisa::Container to convert
     * @return StinkyRegister representing the container, or invalid register if conversion fails
     */
    StinkyRegister toStinkyRegister(const rocisa::Container* container)
    {
        if(const rocisa::RegisterContainer* regCont
           = dynamic_cast<const rocisa::RegisterContainer*>(container))
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
                std::string fullName = regCont->getCompleteRegNameWithType();
                reg.setSymbolicName(fullName);
            }

            return reg;
        }
        if(const rocisa::VCC* vccCont = dynamic_cast<const rocisa::VCC*>(container))
        {
            RegType regType = stringToRegType(vccCont->toString());
            return StinkyRegister(regType, 0, 1);
        }
        if(const rocisa::EXEC* execCont = dynamic_cast<const rocisa::EXEC*>(container))
        {
            RegType regType = stringToRegType(execCont->toString());
            return StinkyRegister(regType, 0, 1);
        }
        if(const rocisa::HWRegContainer* hwregContainer
           = dynamic_cast<const rocisa::HWRegContainer*>(container))
        {
            // Handle hardware register containers like hwreg(26,4,1)
            // These should be emitted as literal strings in the assembly
            return StinkyRegister(hwregContainer->toString());
        }
        return StinkyRegister{};
    }

    /**
     * @brief Convert a rocisa::InstructionInput to StinkyRegister
     *
     * This overload handles InstructionInput variants which can contain:
     * - shared_ptr<Container> (converted via RegisterContainer)
     * - int literals
     * - double literals
     * - string literals
     *
     * @param input The InstructionInput variant to convert
     * @return StinkyRegister representing the input value
     */
    StinkyRegister toStinkyRegister(const InstructionInput& input)
    {
        if(auto pptr = std::get_if<std::shared_ptr<rocisa::Container>>(&input))
        {
            return toStinkyRegister(pptr->get());
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
                size_t start = 0;

                // Check for optional leading minus sign
                if((*literalString)[0] == '-')
                {
                    start = 1;
                    if(literalString->length() == 1)
                    {
                        return StinkyRegister(*literalString);
                    }
                }

                // Check if all remaining characters are digits
                for(size_t i = start; i < literalString->length(); ++i)
                    if(!std::isdigit(static_cast<unsigned char>((*literalString)[i])))
                        return StinkyRegister(*literalString);

                int value = std::atoi(literalString->c_str());
                return StinkyRegister(value);
            }
            return StinkyRegister(*literalString);
        }

        return StinkyRegister{};
    }

    std::shared_ptr<StinkyAsmModule> toStinkyTofuModule(const rocisa::Module& module,
                                                        std::array<int, 3>    arch,
                                                        const std::string&    moduleName)
    {
        // Get GfxArchID from architecture array
        GfxArchID archId = getGfxArchID(arch[0], arch[1], arch[2]);

        StinkyAsmModule stinkyAsmModule(moduleName, arch);
        IRList&         insts = stinkyAsmModule.getIRList();

        // Create IRBuilder for lower-level instruction creation
        StinkyInstIRBuilder irBuilder(insts, archId);

        // Process each item
        const std::map<std::string, int> asmCaps = rocisa::rocIsa::getInstance().getAsmCaps();

        // '-1' is safe because vgprMsb is a 8 bit value.
        int currentVgprMsb = -1;

        for(auto itemShared : module.flatitems())
        {
            rocisa::Item* item = itemShared.get();

            // Skip text blocks
            if(rocisa::TextBlock* textBlock = dynamic_cast<rocisa::TextBlock*>(item))
            {
                continue;
            }

            // Handle labels
            if(rocisa::Label* rocLabel = dynamic_cast<rocisa::Label*>(item))
            {
                StinkyInstruction* labelInst
                    = irBuilder.createStinkyLabel(insts.end(), rocLabel->getLabelName());

                // Add comment if present
                if(!rocLabel->comment.empty())
                {
                    labelInst->addModifier<CommentData>(CommentData{rocLabel->comment});
                }

                currentVgprMsb = -1;
                continue;
            }

            // Handle ValueSet directives
            if(rocisa::ValueSet* valueSet = dynamic_cast<rocisa::ValueSet*>(item))
            {
                AsmDirective* directive = new AsmDirective();
                directive->kind         = AsmDirectiveKind::SET;
                directive->name         = ".set";
                directive->symbol       = valueSet->name;

                // get the last value after the last comma
                size_t pos = valueSet->toString().rfind(',');
                if(pos != std::string::npos)
                {
                    directive->value = valueSet->toString().substr(pos + 1);
                    directive->value.erase(0, directive->value.find_first_not_of(" \t\n\r"));
                    directive->value.erase(directive->value.find_last_not_of(" \t\n\r") + 1);
                }

                insts.insert(insts.end(), directive);
                continue;
            }

            // Handle macro directives
            if(rocisa::Macro* macro = dynamic_cast<rocisa::Macro*>(item))
            {
                AsmDirective* directive = new AsmDirective();
                directive->kind         = AsmDirectiveKind::MACRO;
                directive->name         = ".macro";
                directive->symbol       = macro->name;
                directive->value        = macro->toString();
                insts.insert(insts.end(), directive);
                continue;
            }

            // Handle instructions
            rocisa::Instruction* inst = dynamic_cast<rocisa::Instruction*>(item);
            if(inst == nullptr)
            {
                // TODO: Remove this once we have a better way to handle non-instruction items
                std::cout << "Skipping non-instruction item: " << item->toString() << std::endl;
                continue;
            }
            assert(dynamic_cast<rocisa::SSetVgprMsb*>(inst) == nullptr
                   && "SSetVgprMsb should not be created directly in TensileLite");

            // Handle VGPR MSB setting if needed
            currentVgprMsb = handleVgprMsbSetting(*inst, currentVgprMsb, irBuilder, insts, archId);

            // Create StinkyInstruction from rocisa instruction
            std::type_index    rocisaTy = std::type_index(typeid(*inst));
            StinkyInstruction* stinkyInst
                = createStinkyInstructionFromRocisa(*inst, rocisaTy, irBuilder, insts, archId);

            if(stinkyInst == nullptr)
            {
                continue;
            }

            // Add registers (sources and destinations) to the instruction
            addRegistersToInstruction(stinkyInst, inst, asmCaps, archId);

            // Add modifiers (DS, FLAT, MUBUF, SMEM, WaitCnt, comments)
            addModifiersToInstruction(stinkyInst, inst);

            // Handle branch instructions
            if(rocisa::BranchInstruction* branchInst
               = dynamic_cast<rocisa::BranchInstruction*>(inst))
            {
                stinkyInst->addModifier<LabelData>(
                    LabelData{Modifier::Type::LABEL_NAME, branchInst->labelName});
            }
        }

        // Post-process: Expand VNop instructions with count > 1 into multiple v_nop instructions
        expandNopInstructions(insts, irBuilder, archId);

        // Post-process: Lower VCmpX instructions on architectures that need it
        // This converts v_cmpx instructions into v_cmp + s_mov exec
        lowerVCmpXInstructions(insts, irBuilder, archId);

        // Iterate through the instructions in the IRList and add them to the module
        for(IRBase& ir : insts)
        {
            stinkyAsmModule.add(static_cast<IRBase*>(&ir));
        }

        return std::make_shared<StinkyAsmModule>(std::move(stinkyAsmModule));
    }

    /**
     * @brief Convert rocisa::SignatureValueKind to stinkytofu::SignatureValueKind
     */
    SignatureValueKind convertSignatureValueKind(rocisa::SignatureValueKind kind)
    {
        switch(kind)
        {
        case rocisa::SignatureValueKind::SIG_GLOBALBUFFER:
            return SignatureValueKind::SIG_GLOBALBUFFER;
        case rocisa::SignatureValueKind::SIG_VALUE:
            return SignatureValueKind::SIG_VALUE;
        default:
            return SignatureValueKind::SIG_VALUE;
        }
    }

    /**
     * @brief Convert rocisa::SignatureBase to stinkytofu::SignatureBase
     */
    std::shared_ptr<SignatureBase> toStinkySignature(const rocisa::SignatureBase& rocisaSig,
                                                     const std::array<int, 3>&    isaVersion,
                                                     int                          wavefrontSize)
    {
        // Extract kernel descriptor info
        const auto& kd = rocisaSig.kernelDescriptor;

        // Extract code metadata info
        const auto& cm = rocisaSig.codeMeta;

        // Use wavefrontSize passed from Python (kernel["WavefrontSize"])

        // Create stinkytofu signature
        auto stinkySig = std::make_shared<SignatureBase>(rocisaSig.name,
                                                         isaVersion,
                                                         cm.kernArgsVersion,
                                                         cm.codeObjectVersion,
                                                         kd.groupSegSize,
                                                         kd.sgprWorkGroup,
                                                         kd.vgprWorkItem,
                                                         cm.flatWgSize,
                                                         wavefrontSize,
                                                         kd.totalVgprs,
                                                         kd.totalAgprs,
                                                         kd.totalSgprs,
                                                         kd.enablePreloadKernArgs);

        // Convert arguments
        for(const auto& arg : cm.argList)
        {
            stinkySig->addArg(arg.name,
                              convertSignatureValueKind(arg.valueKind),
                              arg.valueType,
                              arg.addrSpaceQual);
        }

        // Note: Descriptions (descriptionTopic, descriptionList) are private in rocisa::SignatureBase
        // and cannot be directly accessed. They would need to be made public or accessed via getters
        // if description conversion is needed.

        return stinkySig;
    }

    /**
     * @brief Convert rocisa::KernelBody to stinkytofu::KernelBody
     */
    std::shared_ptr<KernelBody> toStinkyKernelBody(const rocisa::KernelBody& rocisaKernel,
                                                   const std::array<int, 3>& isaVersion)
    {
        auto stinkyKernel = std::make_shared<KernelBody>(rocisaKernel.name);

        // Convert signature if present
        if(rocisaKernel.signature)
        {
            auto stinkySig = toStinkySignature(*rocisaKernel.signature, isaVersion);
            stinkyKernel->addSignature(stinkySig);
        }

        // Convert body if present
        if(rocisaKernel.body)
        {
            auto stinkyModule
                = toStinkyTofuModule(*rocisaKernel.body, isaVersion, rocisaKernel.name);
            stinkyKernel->addBody(stinkyModule);
        }

        // Set GPR counts
        stinkyKernel->setGprs(
            rocisaKernel.totalVgprs, rocisaKernel.totalAgprs, rocisaKernel.totalSgprs);

        return stinkyKernel;
    }

} // namespace stinkytofu

/**
 * @brief Initialize StinkyTofu Python bindings
 *
 * This function binds the rocisa to StinkyTofu utilities to Python, allowing
 * Python code to convert rocisa to StinkyTofu IR.
 *
 * @param m The nanobind module to add bindings to
 */
void init_stinkytofu(nb::module_ m)
{
    // Bind toStinkyRegister for Container pointer
    m.def("toStinkyRegister",
          nb::overload_cast<const rocisa::Container*>(&stinkytofu::toStinkyRegister),
          nb::arg("container"),
          "Convert a rocisa Container to a StinkyRegister");

    // Bind toStinkyRegister for InstructionInput
    m.def("toStinkyRegister",
          nb::overload_cast<const InstructionInput&>(&stinkytofu::toStinkyRegister),
          nb::arg("input"),
          "Convert a rocisa InstructionInput to a StinkyRegister");

    // Wrapper class to add signature support to StinkyAsmModule
    class StinkyAsmModuleWithSignature
    {
    private:
        std::shared_ptr<StinkyAsmModule>           module_;
        std::shared_ptr<stinkytofu::SignatureBase> signature_;

    public:
        StinkyAsmModuleWithSignature(std::shared_ptr<StinkyAsmModule>           module,
                                     std::shared_ptr<stinkytofu::SignatureBase> signature)
            : module_(module)
            , signature_(signature)
        {
        }

        // Forward all StinkyAsmModule methods
        void runOptimizationPipeline()
        {
            module_->runOptimizationPipeline();
        }

        size_t size() const
        {
            return module_->size();
        }

        std::string getName() const
        {
            return module_->getName();
        }

        // Override emitAssembly to include signature
        std::string emitAssembly() const
        {
            std::string result;
            if(signature_)
            {
                result = signature_->toString();
            }
            result += module_->emitAssembly();
            return result;
        }

        // Provide access to underlying module if needed
        std::shared_ptr<StinkyAsmModule> getModule() const
        {
            return module_;
        }
    };

    // Bind the wrapper class
    nb::class_<StinkyAsmModuleWithSignature>(m, "StinkyAsmModule")
        .def("runOptimizationPipeline", &StinkyAsmModuleWithSignature::runOptimizationPipeline)
        .def("emitAssembly", &StinkyAsmModuleWithSignature::emitAssembly)
        .def("size", &StinkyAsmModuleWithSignature::size)
        .def("getName", &StinkyAsmModuleWithSignature::getName)
        .def("getModule", &StinkyAsmModuleWithSignature::getModule);

    // Bind toStinkyTofuModule with signature support
    m.def(
        "toStinkyTofuModule",
        [](const rocisa::Module&        module,
           nb::object                   arch_obj,
           const std::string&           moduleName,
           const rocisa::SignatureBase& signature,
           int                          wavefrontSize) {
            // Convert Python sequence (tuple or list) to std::array
            if(!nb::isinstance<nb::sequence>(arch_obj))
            {
                throw std::invalid_argument(
                    "arch must be a tuple or list of 3 integers [major, minor, stepping]");
            }

            auto arch_seq = nb::cast<nb::sequence>(arch_obj);
            if(nb::len(arch_seq) != 3)
            {
                throw std::invalid_argument(
                    "arch must have exactly 3 elements [major, minor, stepping]");
            }

            std::array<int, 3> archArray = {
                nb::cast<int>(arch_seq[0]), nb::cast<int>(arch_seq[1]), nb::cast<int>(arch_seq[2])};

            // Convert module to StinkyAsmModule
            auto stinkyModule = stinkytofu::toStinkyTofuModule(module, archArray, moduleName);

            // Convert signature to StinkyTofu format, using the wavefrontSize passed from Python
            auto stinkySig = stinkytofu::toStinkySignature(signature, archArray, wavefrontSize);

            // Create and return wrapper with both
            return std::make_shared<StinkyAsmModuleWithSignature>(stinkyModule, stinkySig);
        },
        nb::arg("module"),
        nb::arg("arch"),
        nb::arg("moduleName") = "",
        nb::arg("signature"),
        nb::arg("wavefrontSize") = 64,
        "Convert a rocisa.Module to a StinkyTofu StinkyAsmModule with signature support. "
        "The returned object's emitAssembly() will include both signature and instructions.");
}
