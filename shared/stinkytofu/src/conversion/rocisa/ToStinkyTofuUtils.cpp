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
#include "instruction/cvt.hpp"
#include "instruction/mem.hpp"
#include "instruction/mfma.hpp"
#include "stinkytofu/pipeline/BackendRegistry.hpp"
#include "stinkytofu/transforms/asm/LegalizationUtils.hpp"
#include "stinkytofu/ir/asm/StinkyAsmDirectives.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/core/StinkyAsmModule.hpp"
#include "stinkytofu/ir/asm/StinkySignature.hpp"
#include "AllHwMappings.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/core/stinkytofu.hpp"

#include "stinkytofu/serialization/asm/StinkyAsmEmitter.hpp"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <iostream>
#include <optional>
#include <string_view>
#include <vector>

namespace nb = nanobind;

namespace
{
    using namespace rocisa;
    using namespace stinkytofu;

    /**
     * @brief Get the instruction group indices from the rocisa item
     * @param item The rocisa item
     * @param rocisaModuleSet The set of rocisa modules
     * @param stinkyAsmModule The StinkyAsmModule to get the instruction group indices
     * @return The instruction group indices
     */
    std::vector<int> getInstructionGroupIndices(const rocisa::Item&              item,
                                                std::set<const rocisa::Module*>& rocisaModuleSet,
                                                StinkyAsmModule&                 stinkyAsmModule)
    {
        std::vector<int>    instructionGroups;
        const rocisa::Item* parentItem = item.parent;
        while(const auto* parentModule = dynamic_cast<const rocisa::Module*>(parentItem))
        {
            std::string moduleName = parentModule->name;
            if(rocisaModuleSet.find(parentModule) == rocisaModuleSet.end())
            {
                rocisaModuleSet.insert(parentModule);
                int duplicateCount = 0;
                while(stinkyAsmModule.hasGroup(moduleName))
                {
                    moduleName = parentModule->name + "_" + std::to_string(++duplicateCount);
                }
                stinkyAsmModule.addGroup(moduleName);
            }
            instructionGroups.push_back(stinkyAsmModule.getGroupIndex(moduleName));
            parentItem = parentModule->parent;
        }
        return instructionGroups;
    }
    StinkyRegister toStinkyRegister(const Container* container, bool hasVgprMsb);
    StinkyRegister toStinkyRegister(const InstructionInput& input, bool hasVgprMsb);

    /// Check if an instruction reads the SCC register
    bool doesReadSCC(const Instruction* inst)
    {
        if(dynamic_cast<const SCSelectB32*>(inst))
        {
            return true;
        }
        return false;
    }

    /// Check if an instruction writes the SCC register
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

    std::string itemToString(const rocisa::Item* item)
    {
        return item->toString();
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

    stinkytofu::SDelayAluData convertSDelayAluData(const rocisa::SDelayAlu* delayAluInst)
    {
        // Convert DelayALUType to SDelayAluData::InstType
        auto convertType = [](rocisa::DelayALUType type) -> SDelayAluData::InstType {
            switch(type)
            {
            case rocisa::DelayALUType::VALU:
                return SDelayAluData::InstType::VALU;
            case rocisa::DelayALUType::SALU:
                return SDelayAluData::InstType::SALU;
            case rocisa::DelayALUType::TRANS:
                return SDelayAluData::InstType::TRANS;
            default:
                return SDelayAluData::InstType::NO_DEP;
            }
        };

        auto params = delayAluInst->getParams();

        assert(params.size() >= 2 && "s_delay_alu should have at least 2 parameters");

        int instid0TypeInt = std::get<int>(params[0]);
        int instid0Cnt     = std::get<int>(params[1]);

        SDelayAluData::InstType instid0Type
            = convertType(static_cast<rocisa::DelayALUType>(instid0TypeInt));

        if(!delayAluInst->hasInstID1())
        {
            // Single dependency: {instid0type, instid0cnt}
            assert(params.size() == 2 && "s_delay_alu single dependency should have 2 parameters");
            return SDelayAluData(instid0Type, static_cast<int8_t>(instid0Cnt));
        }

        // Dual dependency: {instid0type, instid0cnt, instskipCnt, instid1type, instid1cnt}
        assert(params.size() == 5 && "s_delay_alu has 5 parameters");

        int instskipCnt    = std::get<int>(params[2]);
        int instid1TypeInt = std::get<int>(params[3]);
        int instid1Cnt     = std::get<int>(params[4]);

        SDelayAluData::InstType instid1Type
            = convertType(static_cast<rocisa::DelayALUType>(instid1TypeInt));

        return SDelayAluData(instid0Type,
                             static_cast<int8_t>(instid0Cnt),
                             static_cast<int8_t>(instskipCnt),
                             instid1Type,
                             static_cast<int8_t>(instid1Cnt));
    }

    stinkytofu::VOP3PModifiers convertVOP3PModifiers(const rocisa::VOP3PModifiers& rocMod)
    {
        return stinkytofu::VOP3PModifiers(rocMod.op_sel, rocMod.op_sel_hi, rocMod.byte_sel);
    }

    Legalized legalizeInstruction(StinkyInstruction*                inst,
                                  rocisa::Instruction*              rocisaInst,
                                  StinkyInstIRBuilder&              irBuilder,
                                  GfxArchID                         archId,
                                  const std::map<std::string, int>& asmCaps,
                                  bool                              hasVgprMsb)
    {
        if(isBranch(*inst))
        {
            // Handle branch instructions
            rocisa::BranchInstruction* branchInst
                = dynamic_cast<rocisa::BranchInstruction*>(rocisaInst);
            assert(branchInst != nullptr && "This should be a rocisa Branch.");
            inst->addModifier<LabelData>(
                LabelData{Modifier::Type::LABEL_NAME, branchInst->labelName});
            return {nullptr, nullptr};
        }

        if(inst->is(InstFlag::IF_VCmpX))
        {
            return legalizeVCmpX(inst, irBuilder, archId, asmCaps);
        }

        switch(inst->getUnifiedOpcode())
        {
        case GFX::v_nop:
            return legalizeVNop(inst, irBuilder, archId);

        case GFX::ds_load_b192:
            return legalizeDSLoadB192(inst, irBuilder, archId, hasVgprMsb);

        case GFX::ds_store_b192:
            return legalizeDSStoreB192(inst, irBuilder, archId, hasVgprMsb);

        case GFX::s_waitcnt:
            return legalizeWaitCnt(inst, irBuilder, archId);

        case GFX::s_barrier:
            return legalizeBarrier(inst, irBuilder, archId);

        default:
            break;
        }
        return {nullptr, nullptr};
    }

    /// Create StinkyInstruction from rocisa instruction
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

    /// Add source and destination registers to StinkyInstruction
    void addRegistersToInstruction(StinkyInstruction*                stinkyInst,
                                   const rocisa::Instruction*        inst,
                                   const std::map<std::string, int>& asmCaps,
                                   bool                              hasVgprMsb,
                                   GfxArchID                         archId)
    {
        // Skip adding registers for SDelayAlu - it uses SDelayAluData modifier instead
        if(stinkyInst->getUnifiedOpcode() == GFX::s_delay_alu)
        {
            return;
        }

        // Add destination registers
        for(const InstructionInput& dst : inst->getDstParams())
        {
            StinkyRegister reg = toStinkyRegister(dst, hasVgprMsb);
            if(reg.isValid())
            {
                stinkyInst->addDestReg(reg);
            }
        }

        // Add source registers
        std::vector<InstructionInput> srcParams = inst->getSrcParams();

        // Adjust source parameters for VLShiftLeftAddU32 CompositeInstruction
        // VLShiftLeftAddU32 stores parameters as: {src0, src1, shift}
        // _VLShiftLeftAddU32 stores parameters as: {src0, shift, src1}
        // Assembly format is: v_lshl_add_u32 dst, src0, shift, src1
        if(typeid(*inst) == typeid(rocisa::VLShiftLeftAddU32))
        {
            auto it         = asmCaps.find("HasAddLshl");
            bool hasAddLshl = (it != asmCaps.end() && it->second);
            if(hasAddLshl)
            {
                // VLShiftLeftAddU32 order is src0, src1, shift
                // Need to swap to get: src0, shift, src1
                std::swap(srcParams[1], srcParams[2]);
            }
        }

        for(size_t i = 0; i < srcParams.size(); ++i)
        {
            StinkyRegister reg = toStinkyRegister(srcParams[i], hasVgprMsb);
            if(reg.isValid())
            {
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

    /// Helper to extract neg_lo/neg_hi modifiers from instruction string
    ///
    /// Searches for patterns like:
    ///   neg_lo:[x,x] or neg_lo:[x,x,x]
    ///   neg_hi:[x,x] or neg_hi:[x,x,x]
    /// where x is a digit
    ///
    /// \param instString The instruction string to search
    /// \return Tuple of (negStr, has_neg_lo, has_neg_hi)
    std::tuple<std::string, bool, bool> extractNegModifiers(const std::string& instString)
    {
        std::string negStr   = "";
        bool        hasNegLo = false;
        bool        hasNegHi = false;

        // Helper to extract a neg modifier pattern
        auto extractPattern = [&](const std::string& pattern, bool& hasPattern) {
            size_t pos = instString.find(pattern);
            if(pos != std::string::npos)
            {
                size_t endPos = instString.find(']', pos);
                if(endPos != std::string::npos)
                {
                    if(!negStr.empty())
                        negStr += " ";
                    negStr += instString.substr(pos, endPos - pos + 1);
                    hasPattern = true;
                }
            }
        };

        extractPattern("neg_lo:", hasNegLo);
        extractPattern("neg_hi:", hasNegHi);

        return std::make_tuple(negStr, hasNegLo, hasNegHi);
    }

    /// Helper to extract matrix/scale format substring from instruction string.
    /// Matches: matrix_a_fmt:xxx matrix_b_fmt:yyy [matrix_a_scale_fmt:z] [matrix_b_scale_fmt:w]
    ///      or: matrix_a_scale_fmt:z [matrix_b_scale_fmt:w]
    static std::string extractMatrixFormatStr(std::string_view instString)
    {
        auto extractUntilSpace = [&](size_t pos) -> std::string_view {
            if(pos >= instString.size())
                return {};
            size_t end = instString.find(' ', pos);
            if(end == std::string_view::npos)
                end = instString.size();
            return instString.substr(pos, end - pos);
        };
        auto append = [](std::string& out, std::string_view tok) {
            if(tok.empty())
                return;
            if(!out.empty())
                out += ' ';
            out.append(tok);
        };

        std::string result;
        size_t      aFmt = instString.find("matrix_a_fmt:");
        size_t      bFmt = instString.find("matrix_b_fmt:");
        if(aFmt != std::string_view::npos && bFmt != std::string_view::npos && bFmt > aFmt)
        {
            append(result, extractUntilSpace(aFmt));
            append(result, extractUntilSpace(bFmt));
            size_t aScale = instString.find("matrix_a_scale_fmt:", bFmt);
            size_t bScale = instString.find("matrix_b_scale_fmt:", bFmt);
            if(aScale != std::string_view::npos)
                append(result, extractUntilSpace(aScale));
            if(bScale != std::string_view::npos)
                append(result, extractUntilSpace(bScale));
        }
        else
        {
            size_t aScale = instString.find("matrix_a_scale_fmt:");
            if(aScale != std::string_view::npos)
            {
                append(result, extractUntilSpace(aScale));
                size_t bScale = instString.find("matrix_b_scale_fmt:", aScale);
                if(bScale != std::string_view::npos)
                    append(result, extractUntilSpace(bScale));
            }
        }
        return result;
    }

    /// Helper to handle MXMFMA instruction modifiers
    void handleMXMFMAModifiers(StinkyInstruction*               stinkyInst,
                               const rocisa::MXMFMAInstruction* mxmfmaInst,
                               const std::string&               instString)
    {
        std::string inputPermuteStr = extractMatrixFormatStr(instString);

        // MXMFMA does not support neg_lo/neg_hi modifiers

        // Create and add MFMA modifiers with MXMFMA-specific fields
        MFMAModifiers mfmaModifiers(inputPermuteStr,
                                    "" /* negStr */,
                                    mxmfmaInst->reuseA,
                                    mxmfmaInst->reuseB,
                                    static_cast<int>(mxmfmaInst->instType),
                                    static_cast<int>(mxmfmaInst->mxScaleAType),
                                    static_cast<int>(mxmfmaInst->mxScaleBType),
                                    false /* hasNegLo */,
                                    false /* hasNegHi */);
        stinkyInst->addModifier<MFMAModifiers>(mfmaModifiers);
    }

    /// Helper to handle MFMA instruction modifiers
    void handleMFMAModifiers(StinkyInstruction*             stinkyInst,
                             const rocisa::MFMAInstruction* mfmaInst,
                             const std::string&             instString)
    {
        // Extract inputPermute string patterns like "matrix_a_fmt:xxxxx matrix_b_fmt:yyyyy"
        std::string inputPermuteStr;
        size_t      aFmt = instString.find("matrix_a_fmt:");
        size_t      bFmt = instString.find("matrix_b_fmt:");
        if(aFmt != std::string_view::npos && bFmt != std::string_view::npos && bFmt > aFmt)
        {
            size_t end = instString.find(' ', bFmt + 13);
            if(end == std::string_view::npos)
                end = instString.size();
            inputPermuteStr.assign(instString, aFmt, end - aFmt);
        }

        // Extract neg_lo/neg_hi modifiers
        auto [negStr, hasNegLo, hasNegHi] = extractNegModifiers(instString);

        // Only set reuseA and reuseB if the instruction type is not f8f6f4
        bool reuseA = mfmaInst->typeConvert(mfmaInst->instType) != "f8f6f4" && mfmaInst->reuseA;
        bool reuseB = mfmaInst->typeConvert(mfmaInst->instType) != "f8f6f4" && mfmaInst->reuseB;
        MFMAModifiers mfmaModifiers(inputPermuteStr, negStr, reuseA, reuseB, hasNegLo, hasNegHi);
        stinkyInst->addModifier<MFMAModifiers>(mfmaModifiers);
    }

    /// Helper to handle SMFMA instruction modifiers
    void handleSMFMAModifiers(StinkyInstruction*              stinkyInst,
                              const rocisa::SMFMAInstruction* smfmaInst,
                              const std::string&             instString)
    {
        // Extract neg_lo/neg_hi modifiers
        auto [negStr, hasNegLo, hasNegHi] = extractNegModifiers(instString);

        MFMAModifiers mfmaModifiers("" /* inputPermuteStr */,
                                    negStr,
                                    false /* reuseA */,
                                    false /* reuseB */,
                                    hasNegLo,
                                    hasNegHi);
        stinkyInst->addModifier<MFMAModifiers>(mfmaModifiers);
    }

    /// Helper to handle SWaitCnt instruction modifiers
    void handleSWaitCntModifiers(StinkyInstruction* stinkyInst, const rocisa::SWaitCnt* waitCntInst)
    {
        SWaitCntData waitCntData(
            waitCntInst->vlcnt, waitCntInst->vscnt, -1, waitCntInst->dscnt, waitCntInst->kmcnt);
        stinkyInst->addModifier<SWaitCntData>(waitCntData);
    }

    /// Helper to handle _SWaitDscnt instruction modifiers
    void handleSWaitDscntModifiers(StinkyInstruction*                stinkyInst,
                                   const rocisa::_SWaitDscnt*        waitCntInst,
                                   const std::map<std::string, int>& asmCaps)
    {
        auto it       = asmCaps.find("MaxDscnt");
        int  maxDscnt = it != asmCaps.end() ? it->second : waitCntInst->getDscnt();
        int  dscnt    = std::min(waitCntInst->getDscnt(), maxDscnt);
        if(auto sWaitCntData = stinkyInst->getModifier<SWaitCntData>())
        {
            sWaitCntData->dscnt = dscnt;
        }
        else
        {
            SWaitCntData waitCntData;
            waitCntData.dscnt = dscnt;
            stinkyInst->addModifier<SWaitCntData>(waitCntData);
        }
    }

    /// Helper to handle _SWaitLoadcnt instruction modifiers
    void handleSWaitLoadcntModifiers(StinkyInstruction*                stinkyInst,
                                     const rocisa::_SWaitLoadcnt*      waitLoadcntInst,
                                     const std::map<std::string, int>& asmCaps)
    {
        auto it         = asmCaps.find("MaxLoadcnt");
        int  maxLoadcnt = it != asmCaps.end() ? it->second : waitLoadcntInst->getLoadcnt();
        int  loadcnt    = std::min(waitLoadcntInst->getLoadcnt(), maxLoadcnt);
        if(auto sWaitCntData = stinkyInst->getModifier<SWaitCntData>())
        {
            sWaitCntData->vlcnt = loadcnt;
        }
        else
        {
            SWaitCntData waitCntData;
            waitCntData.vlcnt = loadcnt;
            stinkyInst->addModifier<SWaitCntData>(waitCntData);
        }
    }

    /// Helper to handle VCvt instruction True16 modifiers
    void handleVCvtTrue16Modifiers(StinkyInstruction*             stinkyInst,
                                   const rocisa::VCvtInstruction* vcvtInst)
    {
        if(vcvtInst->true16.empty())
        {
            return;
        }

        // Convert rocisa::True16Modifiers to stinkytofu True16Modifiers
        // rocisa uses indices: DST=0, DST1=1, SRC0=2, SRC1=3, ...
        stinkytofu::HighBitSel              dst0 = stinkytofu::HighBitSel::NONE;
        stinkytofu::HighBitSel              dst1 = stinkytofu::HighBitSel::NONE;
        std::vector<stinkytofu::HighBitSel> srcs;

        for(size_t i = 0; i < vcvtInst->true16.size(); ++i)
        {
            stinkytofu::HighBitSel highBit = static_cast<stinkytofu::HighBitSel>(
                static_cast<int>(vcvtInst->true16[i].high_bit));

            if(i == 0) // DST
            {
                dst0 = highBit;
            }
            else if(i == 1) // DST1
            {
                dst1 = highBit;
            }
            else // SRC0, SRC1, ...
            {
                srcs.push_back(highBit);
            }
        }

        // Assert that source count is within the 2-bit encoding limit (max 6 sources)
        assert(srcs.size() <= 6
               && "True16Modifiers: source count must be <= 6 for uint16_t 2-bit encoding");

        stinkyInst->addModifier<stinkytofu::True16Modifiers>(
            stinkytofu::True16Modifiers(dst0, dst1, srcs));
    }

    /// Add modifiers to StinkyInstruction (DS, FLAT, MUBUF, SMEM, WaitCnt, DelayAlu)
    void addModifiersToInstruction(StinkyInstruction*                stinkyInst,
                                   const rocisa::Instruction*        inst,
                                   const std::map<std::string, int>& asmCaps)
    {
#define TRY_ADD_MOD(RocisaInstType, modField, StinkyModType, converter)                 \
    if(auto typed = dynamic_cast<const RocisaInstType*>(inst))                          \
    {                                                                                   \
        if(typed->modField.has_value())                                                 \
        {                                                                               \
            stinkyInst->addModifier<StinkyModType>(converter(typed->modField.value())); \
        }                                                                               \
    }

#define HANDLE_INST_TYPE(RocisaInstType, handlerCall)              \
    if(auto typedInst = dynamic_cast<const RocisaInstType*>(inst)) \
    {                                                              \
        handlerCall;                                               \
    }

        // clang-format off
        // Chain all memory instruction types (mutually exclusive)
        TRY_ADD_MOD(DSLoadInstruction, ds, stinkytofu::DSModifiers, convertDSModifiers)
        else TRY_ADD_MOD(DSStoreInstruction, ds, stinkytofu::DSModifiers, convertDSModifiers)
        else TRY_ADD_MOD(FLATReadInstruction, flat, stinkytofu::FLATModifiers,convertFLATModifiers)
        else TRY_ADD_MOD(FLATStoreInstruction, flat, stinkytofu::FLATModifiers, convertFLATModifiers)
        else TRY_ADD_MOD(MUBUFReadInstruction, mubuf, stinkytofu::MUBUFModifiers, convertMUBUFModifiers)
        else TRY_ADD_MOD(MUBUFStoreInstruction, mubuf, stinkytofu::MUBUFModifiers, convertMUBUFModifiers)
        else TRY_ADD_MOD(SMemLoadInstruction, smem, stinkytofu::SMEMModifiers, convertSMEMModifiers)
        else TRY_ADD_MOD(SMemStoreInstruction, smem, stinkytofu::SMEMModifiers, convertSMEMModifiers)
        else TRY_ADD_MOD(SMemAtomicDecInstruction, smem, stinkytofu::SMEMModifiers, convertSMEMModifiers)
        else
        {
            // No memory modifier matched
            TRY_ADD_MOD(CommonInstruction, vop3, stinkytofu::VOP3PModifiers, convertVOP3PModifiers)

            // VOP/SOP instructions - these can overlap with CommonInstruction base class
            HANDLE_INST_TYPE(rocisa::MXMFMAInstruction, handleMXMFMAModifiers(stinkyInst, typedInst, itemToString(inst)))
            else HANDLE_INST_TYPE(rocisa::MFMAInstruction, handleMFMAModifiers(stinkyInst, typedInst, itemToString(inst)))
            else HANDLE_INST_TYPE(rocisa::SMFMAInstruction, handleSMFMAModifiers(stinkyInst, typedInst, itemToString(inst)))
            else HANDLE_INST_TYPE(rocisa::VCvtInstruction, handleVCvtTrue16Modifiers(stinkyInst, typedInst))

            // Control/Synchronization instructions, separate from VOP/SOP
            else HANDLE_INST_TYPE(rocisa::SDelayAlu,
                                stinkyInst->addModifier<SDelayAluData>(convertSDelayAluData(typedInst)))
            else HANDLE_INST_TYPE(rocisa::SWaitCnt, handleSWaitCntModifiers(stinkyInst, typedInst))
            else HANDLE_INST_TYPE(rocisa::_SWaitDscnt, handleSWaitDscntModifiers(stinkyInst, typedInst, asmCaps))
            else HANDLE_INST_TYPE(rocisa::_SWaitLoadcnt, handleSWaitLoadcntModifiers(stinkyInst, typedInst, asmCaps))
        }
        // clang-format on

#undef TRY_ADD_MOD
#undef HANDLE_INST_TYPE

        // Always add comment if present
        if(!inst->comment.empty())
        {
            stinkyInst->addModifier<CommentData>(CommentData{inst->comment});
        }
    }

    /// Get MSB value from StinkyRegister if it's a VGPR. Returns -1 for non-VGPR.
    int getMsbFromStinkyVgpr(const StinkyRegister& reg)
    {
        if(reg.dataType != StinkyRegister::Type::Register || reg.reg.type != RegType::V)
            return -1;
        return static_cast<int>(reg.reg.idx) / 256;
    }

    int getMsbOffsetFromStinkyVgpr(const StinkyRegister& reg)
    {
        if(reg.dataType != StinkyRegister::Type::Register || reg.reg.type != RegType::V)
            return 0;
        return getMsbFromStinkyVgpr(reg) * (-256);
    }

    /// Map StinkyInstruction operands to VGPR_OFF MSB slots per encoding type.
    /// Fills msbSrc[3], msbDst and returns hasVgpr.
    void collectVgprMsbFromStinkyInst(const StinkyInstruction* inst,
                                      int                      msbSrc[3],
                                      int&                     msbDst,
                                      bool&                    hasVgpr)
    {
        const auto& srcRegs  = inst->getSrcRegs();
        const auto& destRegs = inst->getDestRegs();

        auto setMsbFromReg = [&](int& slot, const StinkyRegister& reg) {
            int msb = getMsbFromStinkyVgpr(reg);
            if(msb >= 0)
            {
                slot    = msb;
                hasVgpr = true;
            }
        };

        // VOPC: No DST, SRC0=srcRegs[0], SRC1=srcRegs[1]
        if(inst->is(InstFlag::IF_VCmp))
        {
            setMsbFromReg(msbSrc[0], srcRegs[0]);
            setMsbFromReg(msbSrc[1], srcRegs[1]);
        }
        // VBUFFER (MUBUF): immediate layout {dst, src2, src1, src0}
        // Load:  DST=VDATA, SRC0=VADDR (srcRegs[0]), SRC1=RSRC (srcRegs[1])
        // Store: SRC0=DATA (srcRegs[0]), SRC1=VADDR (srcRegs[1]), SRC2=RSRC (srcRegs[2])
        else if(isMUBUFLoad(*inst))
        {
            setMsbFromReg(msbDst, destRegs[0]);
            setMsbFromReg(msbSrc[0], srcRegs[0]);
        }
        else if(isMUBUFStore(*inst))
        {
            setMsbFromReg(msbDst, srcRegs[0]);
            setMsbFromReg(msbSrc[0], srcRegs[1]);
        }
        else if(isDSRead(*inst))
        {
            setMsbFromReg(msbDst, destRegs[0]);
            setMsbFromReg(msbSrc[0], srcRegs[0]);
        }
        else if(isDSWrite(*inst))
        {
            setMsbFromReg(msbSrc[1], srcRegs[1]);
            setMsbFromReg(msbSrc[0], srcRegs[0]);
        }
        // VIMAGE (Tensor DMA): DST=VDATA, SRC0=VADDR0, SRC1=VADDR1, SRC2=VADDR2
        else if(isTensorLoad(*inst))
        {
            if(!destRegs.empty())
                setMsbFromReg(msbDst, destRegs[0]);
            for(size_t i = 0; i < srcRegs.size() && i < 3; i++)
                setMsbFromReg(msbSrc[i], srcRegs[i]);
        }
        // VOP1, VOP2, VOP3, VOP3P, VOPD, VOPD3, VDS, VFLAT, VGLOBAL, VSCRATCH:
        // Standard mapping: DST=destRegs[0], SRC0=srcRegs[0], SRC1=srcRegs[1], SRC2=srcRegs[2]
        else
        {
            if(!destRegs.empty())
                setMsbFromReg(msbDst, destRegs[0]);
            for(size_t i = 0; i < srcRegs.size() && i < 3; i++)
                setMsbFromReg(msbSrc[i], srcRegs[i]);
        }
    }

    /// Compute required MSB setVal from StinkyInstruction's VGPR register usage.
    /// Uses same encoding as rocisa: setVal = msbSrc[0] + (msbSrc[1]<<2) + (msbSrc[2]<<4) + (msbDst<<6)
    /// \return (setVal, hasVgpr). setVal is -1 when hasVgpr is false.
    std::pair<int, bool> computeRequiredMsbFromStinkyInst(const StinkyInstruction* inst,
                                                          bool                     hasVgprMsb)
    {
        if(!hasVgprMsb)
            return {-1, false};

        // ignore any not vop instructions
        if(inst->is(InstFlag::IF_SALU) || inst->is(InstFlag::IF_SMemLoad)
           || inst->is(InstFlag::IF_SMemStore) || inst->is(InstFlag::IF_SMemAtomic)
           || inst->is(InstFlag::IF_Branch) || inst->is(InstFlag::IF_Barrier)
           || inst->is(InstFlag::IF_WaitCnt) || inst->is(InstFlag::IF_HasSideEffect))
        {
            return {-1, false};
        }

        int  msbSrc[3] = {0, 0, 0};
        int  msbDst    = 0;
        bool hasVgpr   = false;

        collectVgprMsbFromStinkyInst(inst, msbSrc, msbDst, hasVgpr);

        if(!hasVgpr)
            return {-1, false};

        int setVal = msbSrc[0] + (msbSrc[1] << 2) + (msbSrc[2] << 4) + (msbDst << 6);
        return {setVal, true};
    }

    /// Emit s_set_vgpr_msb instruction if required MSB differs from current.
    /// \param insertBefore If valid, insert before this position; otherwise insert at end.
    void emitVgprMsbIfNeeded(int                             requiredSetVal,
                             bool                            hasVgpr,
                             int&                            currentVgprMsb,
                             StinkyInstIRBuilder&            irBuilder,
                             IRList&                         insts,
                             GfxArchID                       archId,
                             std::optional<IRList::iterator> insertBefore = std::nullopt)
    {
        if(!hasVgpr || requiredSetVal == currentVgprMsb)
            return;

        const HwInstDesc* desc = getMCIDByUOp(GFX::s_set_vgpr_msb, archId);
        assert(desc != nullptr && "s_set_vgpr_msb is not supported on this architecture");
        IRList::iterator   pos     = insertBefore ? *insertBefore : insts.end();
        StinkyInstruction* msbInst = irBuilder.createStinkyInstBefore(pos, desc);
        msbInst->addSrcReg(StinkyRegister(requiredSetVal));
        currentVgprMsb = requiredSetVal;
    }

    /// Convert a rocisa::Container to StinkyRegister
    ///
    /// This function takes a rocisa::Container pointer and converts it to a
    /// StinkyRegister. It handles RegisterContainer types by extracting the
    /// register type, index, and number of registers.
    ///
    /// \param container Pointer to rocisa::Container to convert
    /// \param hasVgprMsb Whether VGPR MSB is supported (affects register offset for VGPRs > 255)
    /// \return StinkyRegister representing the container, or invalid register if conversion fails
    StinkyRegister toStinkyRegister(const rocisa::Container* container, bool hasVgprMsb)
    {
        if(const rocisa::RegisterContainer* regCont
           = dynamic_cast<const rocisa::RegisterContainer*>(container))
        {
            // Convert string regType to RegType enum
            RegType regType = stringToRegType(regCont->regType);

            int physicalIdx = regCont->regIdx;
            if(regCont->regName)
            {
                physicalIdx = regCont->regName->getTotalIdx();
            }

            StinkyRegister reg{regType,
                               static_cast<uint32_t>(physicalIdx),
                               static_cast<uint16_t>(regCont->regNum)};

            // TODO: This is a hack to set the offset of the register for use case such as msb, etc.
            // Set offset for VGPR MSB when supported (use case: vgpr > 255)
            if(hasVgprMsb)
                reg.reg.offset = static_cast<int16_t>(getMsbOffsetFromStinkyVgpr(reg));

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

    /// Convert a rocisa::InstructionInput to StinkyRegister
    ///
    /// This overload handles InstructionInput variants which can contain:
    /// - shared_ptr<Container> (converted via RegisterContainer)
    /// - int literals
    /// - double literals
    /// - string literals
    ///
    /// \param input The InstructionInput variant to convert
    /// \param hasVgprMsb Whether VGPR MSB is supported
    /// \return StinkyRegister representing the input value
    StinkyRegister toStinkyRegister(const InstructionInput& input, bool hasVgprMsb)
    {
        if(auto pptr = std::get_if<std::shared_ptr<rocisa::Container>>(&input))
        {
            return toStinkyRegister(pptr->get(), hasVgprMsb);
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

    /// Convert rocisa::SignatureValueKind to stinkytofu::SignatureValueKind
    stinkytofu::SignatureValueKind convertSignatureValueKind(rocisa::SignatureValueKind kind)
    {
        switch(kind)
        {
        case rocisa::SignatureValueKind::SIG_GLOBALBUFFER:
            return stinkytofu::SignatureValueKind::SIG_GLOBALBUFFER;
        case rocisa::SignatureValueKind::SIG_VALUE:
            return stinkytofu::SignatureValueKind::SIG_VALUE;
        default:
            return stinkytofu::SignatureValueKind::SIG_VALUE;
        }
    }

    /// Convert rocisa::SignatureBase to stinkytofu::SignatureBase
    std::shared_ptr<stinkytofu::SignatureBase>
        toStinkySignature(const rocisa::SignatureBase& rocisaSig,
                          const std::array<int, 3>&    isaVersion,
                          int                          wavefrontSize)
    {
        // Extract kernel descriptor info
        const auto& kd = rocisaSig.kernelDescriptor;

        // Extract code metadata info
        const auto& cm = rocisaSig.codeMeta;

        // Use wavefrontSize passed from Python (kernel["WavefrontSize"])

        // Create stinkytofu signature
        auto stinkySig = std::make_shared<stinkytofu::SignatureBase>(rocisaSig.name,
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

        // Note: Optimization config (ThreadTile, SubGroup, VectorWidth, etc.) is now passed
        // directly from Python via toStinkyTofuModule's parameters and set via setOptimizationConfig()

        return stinkySig;
    }

} // anonymous namespace

namespace stinkytofu
{
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
        std::map<std::string, int> asmCaps = rocisa::rocIsa::getInstance().getAsmCaps();
        bool hasVgprMsb = asmCaps.count("HasVgprMSB") && asmCaps.at("HasVgprMSB");

        std::set<const rocisa::Module*> rocisaModuleSet;
        rocisa::Item*                   parentItem = nullptr;
        std::vector<int>                instructionGroups;
        int                             currentVgprMsb = -1;

        for(auto itemShared : module.flatitems())
        {
            rocisa::Item* item       = itemShared.get();
            const auto    instsCount = insts.size();
            if(parentItem != item->parent)
            {
                instructionGroups
                    = getInstructionGroupIndices(*item, rocisaModuleSet, stinkyAsmModule);
                parentItem = item->parent;
            }

            // Handle text blocks
            if(rocisa::TextBlock* textBlock = dynamic_cast<rocisa::TextBlock*>(item))
            {
                AsmDirective* directive = new AsmDirective();
                directive->kind         = AsmDirectiveKind::TEXTBLOCK;
                directive->value        = textBlock->text;
                insts.insert(insts.end(), directive);
                stinkyAsmModule.setInstructionGroups(directive, instructionGroups);
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

                stinkyAsmModule.setInstructionGroups(labelInst, instructionGroups);
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
                std::string itemString  = itemToString(item);

                // get the last value after the last comma
                size_t pos = itemString.rfind(',');
                if(pos != std::string::npos)
                {
                    directive->value = itemString.substr(pos + 1);
                    directive->value.erase(0, directive->value.find_first_not_of(" \t\n\r"));
                    directive->value.erase(directive->value.find_last_not_of(" \t\n\r") + 1);
                }

                insts.insert(insts.end(), directive);
                stinkyAsmModule.setInstructionGroups(directive, instructionGroups);
                continue;
            }

            // Handle macro directives
            if(rocisa::Macro* macro = dynamic_cast<rocisa::Macro*>(item))
            {
                AsmDirective* directive = new AsmDirective();
                directive->kind         = AsmDirectiveKind::MACRO;
                directive->name         = ".macro";
                directive->symbol       = macro->name;
                directive->value        = itemToString(item);
                insts.insert(insts.end(), directive);
                stinkyAsmModule.setInstructionGroups(directive, instructionGroups);
                continue;
            }

            // Handle ValueIf directives
            if(rocisa::ValueIf* valueIf = dynamic_cast<rocisa::ValueIf*>(item))
            {
                AsmDirective* directive = new AsmDirective();
                directive->kind         = AsmDirectiveKind::IF;
                directive->name         = ".if";
                directive->symbol       = std::to_string(valueIf->value);
                directive->value        = itemToString(item);
                insts.insert(insts.end(), directive);
                stinkyAsmModule.setInstructionGroups(directive, instructionGroups);
                continue;
            }

            // Handle ValueEndif directives
            if(rocisa::ValueEndif* valueEndif = dynamic_cast<rocisa::ValueEndif*>(item))
            {
                AsmDirective* directive = new AsmDirective();
                directive->kind         = AsmDirectiveKind::ENDIF;
                directive->name         = ".endif";
                directive->comment      = valueEndif->comment;
                directive->value        = itemToString(item);
                insts.insert(insts.end(), directive);
                stinkyAsmModule.setInstructionGroups(directive, instructionGroups);
                continue;
            }

            // Handle instructions
            rocisa::Instruction* inst = dynamic_cast<rocisa::Instruction*>(item);
            if(inst == nullptr)
            {
                // TODO: Remove this once we have a better way to handle non-instruction items
                std::cout << "Skipping non-instruction item: " << itemToString(item) << std::endl;
                continue;
            }
            assert(dynamic_cast<rocisa::SSetVgprMsb*>(inst) == nullptr
                   && "SSetVgprMsb should not be created directly in TensileLite");

            // Create StinkyInstruction from rocisa instruction
            std::type_index    rocisaTy = std::type_index(typeid(*inst));
            StinkyInstruction* stinkyInst
                = createStinkyInstructionFromRocisa(*inst, rocisaTy, irBuilder, insts, archId);

            assert(stinkyInst != nullptr
                   && "Failed to create StinkyInstruction from rocisa instruction");

            // Add registers (sources and destinations) to the instruction
            addRegistersToInstruction(stinkyInst, inst, asmCaps, hasVgprMsb, archId);

            // Add modifiers (DS, FLAT, MUBUF, SMEM, WaitCnt, comments)
            addModifiersToInstruction(stinkyInst, inst, asmCaps);

            Legalized legalizedInsts
                = legalizeInstruction(stinkyInst, inst, irBuilder, archId, asmCaps, hasVgprMsb);

            auto processStinkyInst = [&](StinkyInstruction* inst) {
                auto [requiredMsb, hasVgpr] = computeRequiredMsbFromStinkyInst(inst, hasVgprMsb);
                emitVgprMsbIfNeeded(requiredMsb,
                                    hasVgpr,
                                    currentVgprMsb,
                                    irBuilder,
                                    insts,
                                    archId,
                                    IRList::iterator(inst));
                stinkyAsmModule.setInstructionGroups(inst, instructionGroups);
            };

            if(legalizedInsts.first != nullptr)
            {
                StinkyInstruction* currentStinkyInst = legalizedInsts.first;
                while(currentStinkyInst != legalizedInsts.last->getNext())
                {
                    processStinkyInst(currentStinkyInst);
                    currentStinkyInst
                        = static_cast<StinkyInstruction*>(currentStinkyInst->getNext());
                }
            }
            else
            {
                processStinkyInst(stinkyInst);
            }
        }
        return std::make_shared<StinkyAsmModule>(std::move(stinkyAsmModule));
    }

} // namespace stinkytofu

namespace
{
    /**
     * @brief Convert a Python sequence (tuple or list) to a std::array<int, 3>
     * @param arch_obj The Python sequence (tuple or list)
     * @return The std::array<int, 3> [major, minor, stepping]
     */
    std::array<int, 3> convertArch(nb::object arch_obj)
    {
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

        return {nb::cast<int>(arch_seq[0]), nb::cast<int>(arch_seq[1]), nb::cast<int>(arch_seq[2])};
    }

} // anonymous namespace

/// Initialize StinkyTofu Python bindings
///
/// This function binds the rocisa to StinkyTofu utilities to Python, allowing
/// Python code to convert rocisa to StinkyTofu IR.
///
/// \param m The nanobind module to add bindings to
void init_stinkytofu(nb::module_ m)
{
    // Bind isSupportedByStinkyTofu to check if the architecture is supported by StinkyTofu
    m.def(
        "isSupportedByStinkyTofu",
        [](nb::object arch_obj) {
            std::array<int, 3> archArray = convertArch(arch_obj);
            return BackendRegistry::hasPipelines(archArray);
        },
        nb::arg("arch"),
        "Check if the architecture is supported by StinkyTofu");
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
           int                          wavefrontSize,
           nb::object                   tt_obj,
           nb::object                   sg_obj,
           int                          vwA,
           int                          vwB,
           int                          glvwA,
           int                          glvwB,
           bool                         d2lA,
           bool                         d2lB,
           int                          useSgprForGRO) {
            // Convert architecture to std::array<int, 3> [major, minor, stepping]
            std::array<int, 3> archArray = convertArch(arch_obj);

            // Convert tt and sg sequences to arrays
            std::array<int, 2> tt = {0, 0};
            std::array<int, 2> sg = {0, 0};

            if(nb::isinstance<nb::sequence>(tt_obj))
            {
                auto tt_seq = nb::cast<nb::sequence>(tt_obj);
                assert(nb::len(tt_seq) == 2 && "ThreadTile must have exactly 2 elements");
                tt = {nb::cast<int>(tt_seq[0]), nb::cast<int>(tt_seq[1])};
            }

            if(nb::isinstance<nb::sequence>(sg_obj))
            {
                auto sg_seq = nb::cast<nb::sequence>(sg_obj);
                assert(nb::len(sg_seq) == 2 && "SubGroup must have exactly 2 elements");
                sg = {nb::cast<int>(sg_seq[0]), nb::cast<int>(sg_seq[1])};
            }

            // Convert module to StinkyAsmModule
            auto stinkyModule = stinkytofu::toStinkyTofuModule(module, archArray, moduleName);

            // Convert signature to StinkyTofu format, using the wavefrontSize passed from Python
            auto stinkySig = toStinkySignature(signature, archArray, wavefrontSize);

            // Set optimization config
            stinkySig->setOptimizationConfig(
                tt, sg, vwA, vwB, glvwA, glvwB, d2lA, d2lB, useSgprForGRO);

            // Create and return wrapper with both
            return std::make_shared<StinkyAsmModuleWithSignature>(stinkyModule, stinkySig);
        },
        nb::arg("module"),
        nb::arg("arch"),
        nb::arg("moduleName") = "",
        nb::arg("signature"),
        nb::arg("wavefrontSize") = 64,
        nb::arg("tt")            = nb::make_tuple(0, 0),
        nb::arg("sg")            = nb::make_tuple(0, 0),
        nb::arg("vwA")           = 0,
        nb::arg("vwB")           = 0,
        nb::arg("glvwA")         = 0,
        nb::arg("glvwB")         = 0,
        nb::arg("d2lA")          = false,
        nb::arg("d2lB")          = false,
        nb::arg("useSgprForGRO") = 0,
        "Convert a rocisa.Module to a StinkyTofu StinkyAsmModule with signature support. "
        "The returned object's emitAssembly() will include both signature and instructions.");
}
