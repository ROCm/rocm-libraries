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
#include "ir/asm/LegalizationUtils.hpp"
#include "ir/asm/StinkyAsmDirectives.hpp"
#include "ir/asm/StinkyAsmIR.hpp"
#include "ir/asm/StinkyAsmModule.hpp"
#include "ir/asm/StinkySignature.hpp"
#include "ir/rocisa/AllHwMappings.hpp"
#include "isa/ArchHelper.hpp"
#include "stinkytofu.hpp"

#include <cassert>
#include <ranges>
#include <regex>
#include <string_view>

namespace nb = nanobind;

namespace stinkytofu
{
    // Forward declarations for use in anonymous namespace
    StinkyRegister                 toStinkyRegister(const rocisa::Container* container);
    StinkyRegister                 toStinkyRegister(const InstructionInput& input);
    std::shared_ptr<SignatureBase> toStinkySignature(const rocisa::SignatureBase& rocisaSig,
                                                     const std::array<int, 3>&    isaVersion,
                                                     int wavefrontSize = 64);
} // namespace stinkytofu

namespace
{
    using namespace rocisa;
    using namespace stinkytofu;

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
                                  const std::map<std::string, int>& asmCaps)
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
            return legalizeDSLoadB192(inst, irBuilder, archId);

        case GFX::ds_store_b192:
            return legalizeDSStoreB192(inst, irBuilder, archId);

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
                                   GfxArchID                         archId)
    {
        // Skip adding registers for SDelayAlu - it uses SDelayAluData modifier instead
        if(dynamic_cast<const rocisa::SDelayAlu*>(inst))
        {
            return;
        }

        std::set<StinkyRegister> uniqueDstRegs;

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
            StinkyRegister reg = stinkytofu::toStinkyRegister(srcParams[i]);
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

    /// Helper to handle MXMFMA instruction modifiers
    void handleMXMFMAModifiers(StinkyInstruction*               stinkyInst,
                               const rocisa::MXMFMAInstruction* mxmfmaInst,
                               const std::string&               instString)
    {
        // Extract matrix format and scale format patterns from the instruction string
        // Follows the pattern: matrix_a_fmt:xxx matrix_b_fmt:yyy [matrix_a_scale_fmt:z] [matrix_b_scale_fmt:w]
        // or just scale formats: matrix_a_scale_fmt:z [matrix_b_scale_fmt:w]
        std::regex  inputPermuteRegex("matrix_a_fmt:[^ ]+ matrix_b_fmt:[^ ]+"
                                      "( matrix_a_scale_fmt:[^ ]+)?( matrix_b_scale_fmt:[^ ]+)?"
                                      "|matrix_a_scale_fmt:[^ ]+( matrix_b_scale_fmt:[^ ]+)?");
        std::smatch match;
        std::string inputPermuteStr = "";
        if(std::regex_search(instString, match, inputPermuteRegex))
        {
            inputPermuteStr = match[0].str();
        }

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
        // extract inputPermute string patterns like "matrix_a_fmt:xxxxx matrix_b_fmt:yyyyy"
        std::regex  inputPermuteRegex("matrix_a_fmt:([^ ]+) matrix_b_fmt:([^ ]+)");
        std::smatch match;
        std::string inputPermuteStr = "";
        if(std::regex_search(instString, match, inputPermuteRegex))
        {
            inputPermuteStr = match[0].str();
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
                              const std::string&              instString)
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
                                   const std::string&                instString,
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
            HANDLE_INST_TYPE(rocisa::MXMFMAInstruction, handleMXMFMAModifiers(stinkyInst, typedInst, instString))
            else HANDLE_INST_TYPE(rocisa::MFMAInstruction, handleMFMAModifiers(stinkyInst, typedInst, instString))
            else HANDLE_INST_TYPE(rocisa::SMFMAInstruction, handleSMFMAModifiers(stinkyInst, typedInst, instString))
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

    inline std::string_view trimString(std::string_view sv)
    {
        const auto begin = sv.find_first_not_of(" \t\f\v\r\n");
        if(begin == std::string_view::npos)
            return {};
        const auto end = sv.find_last_not_of(" \t\f\v\r\n");
        return sv.substr(begin, end - begin + 1);
    }

    inline std::vector<std::string> splitLinesToStrings(std::string_view text)
    {
        std::vector<std::string> out;

        // views::split('\n') splits by LF. We'll drop a trailing '\r' (CRLF) per line.
        for(auto&& rng : text | std::views::split('\n'))
        {
            // Materialize the subrange into a string_view without allocation
            std::string_view line(&*rng.begin(),
                                  static_cast<std::size_t>(std::ranges::distance(rng)));

            if(!line.empty() && line.back() == '\r')
                line.remove_suffix(1); // handle CRLF

            line = trimString(line);
            if(line.empty())
                continue;

            // Store as owning string
            out.emplace_back(line);
        }
        return out;
    }

    /// Create a StinkyRegister from a string
    ///
    /// \param regStr The string representation of the register
    /// \return The StinkyRegister
    StinkyRegister createStinkyRegisterFromStr(const std::string& regStr)
    {
        // TODO: handle more register types. Currently only support v[start:end] and v[N]
        std::regex  srcRegex1("v\\[([^:]+):([^:]+)\\]");
        std::regex  srcRegex2("v\\[([^:]+)\\]");
        std::smatch srcMatch;

        StinkyRegister vReg("v", -1, 1);
        std::string    startIdxStr = "";
        if(std::regex_search(regStr, srcMatch, srcRegex1))
        {
            startIdxStr  = srcMatch[1].str();
            vReg.reg.num = std::stoi(srcMatch[2].str().substr(startIdxStr.length())) + 1;
        }
        else if(std::regex_search(regStr, srcMatch, srcRegex2))
        {
            startIdxStr  = srcMatch[1].str();
            vReg.reg.num = 1;
        }

        // match minus number from startIdxStr
        size_t minusPos = startIdxStr.find('-');
        if(minusPos != std::string::npos)
        {
            vReg.reg.offset = static_cast<int16_t>(std::stoi(startIdxStr.substr(minusPos)));
            vReg.setSymbolicName(startIdxStr.substr(0, minusPos));
        }
        else
        {
            vReg.setSymbolicName(startIdxStr);
        }

        return vReg;
    }

    /// Try to capture s_set_vgpr_msb instruction from the instruction string
    ///
    /// \param instString The instruction string
    /// \param irBuilder The IR builder
    /// \param insts The instruction list
    /// \param archId The architecture ID
    ///
    /// \return True if the s_set_vgpr_msb instruction is captured, false otherwise
    bool tryCaptureVgprMsb(const std::string&   instString,
                           StinkyInstIRBuilder& irBuilder,
                           IRList&              insts,
                           GfxArchID            archId)
    {
        std::smatch match;
        if(std::regex_search(instString, match, std::regex("s_set_vgpr_msb ([0-9]+)")))
        {
            int msbValue = std::stoi(match[1].str());

            const HwInstDesc* desc = getMCIDByUOp(GFX::s_set_vgpr_msb, archId);
            assert(desc != nullptr && "s_set_vgpr_msb is not supported on this architecture");
            StinkyInstruction* msbInst = irBuilder.createStinkyInstBefore(insts.end(), desc);
            msbInst->addSrcReg(StinkyRegister(msbValue));

            return true;
        }

        return false;
    }

    /// Handle ds_load_b192 and ds_store_b192 instructions
    ///
    /// TODO: This function is a temporary solution, we should use legalizeDSLoadB192 and legalizeDSStoreB192 instead.
    ///
    /// \return True if the instruction is a ds_load_b192 or ds_store_b192 instruction and it is handled, false otherwise
    bool handleDSLDSTb192Instructions(rocisa::Instruction* inst,
                                      const std::string&   instString,
                                      StinkyInstIRBuilder& irBuilder,
                                      IRList&              insts,
                                      GfxArchID            archId)
    {
        // DSLoadB192 is composite of 2 instructions: ds_load_b128 and ds_load_b64
        if(dynamic_cast<DSLoadB192*>(inst) || dynamic_cast<DSStoreB192*>(inst))
        {
            auto lines = splitLinesToStrings(instString);
            if(lines.size() > 1)
            {
                for(const auto& line : lines)
                {
                    if(tryCaptureVgprMsb(line, irBuilder, insts, archId))
                    {
                        continue;
                    }

                    std::regex  mnemonicRegex("([a-zA-Z0-9_]+) ([^ ]+), ([^ ]+)");
                    std::smatch match;
                    if(std::regex_search(line, match, mnemonicRegex))
                    {
                        std::string mnemonic = match[1].str();
                        std::string dstStr   = match[2].str();
                        std::string srcStr   = match[3].str();

                        StinkyRegister dstReg = createStinkyRegisterFromStr(dstStr);
                        StinkyRegister srcReg = createStinkyRegisterFromStr(srcStr);

                        auto              opcode = getMnemonicToIsaOpcode(mnemonic, archId);
                        const HwInstDesc* desc   = getMCIDByIsaOp(opcode, archId);
                        assert(desc != nullptr && "is not supported on this architecture");
                        StinkyInstruction* newStinkyInst
                            = irBuilder.createStinkyInstBefore(insts.end(), desc);
                        newStinkyInst->addDestReg(dstReg);
                        newStinkyInst->addSrcReg(srcReg);

                        // match offset:xxx modifier string
                        std::string modifierStr;
                        std::smatch offsetMatch;
                        if(std::regex_search(line, offsetMatch, std::regex("offset:([0-9]+)")))
                        {
                            stinkytofu::DSModifiers dsModifiers;
                            dsModifiers.offset = std::stoi(offsetMatch[1].str());
                            newStinkyInst->addModifier<stinkytofu::DSModifiers>(dsModifiers);
                        }

                        // Always add comment if present
                        if(!inst->comment.empty())
                        {
                            newStinkyInst->addModifier<stinkytofu::CommentData>(
                                stinkytofu::CommentData{inst->comment});
                        }
                    }
                }
                return true;
            }
        }

        return false;
    }

} // anonymous namespace

namespace stinkytofu
{
    /// Convert a rocisa::Container to StinkyRegister
    ///
    /// This function takes a rocisa::Container pointer and converts it to a
    /// StinkyRegister. It handles RegisterContainer types by extracting the
    /// register type, index, and number of registers.
    ///
    /// \param container Pointer to rocisa::Container to convert
    /// \return StinkyRegister representing the container, or invalid register if conversion fails
    StinkyRegister toStinkyRegister(const rocisa::Container* container)
    {
        if(const rocisa::RegisterContainer* regCont
           = dynamic_cast<const rocisa::RegisterContainer*>(container))
        {
            // Convert string regType to RegType enum
            RegType        regType = stringToRegType(regCont->regType);
            StinkyRegister reg{regType,
                               static_cast<uint32_t>(regCont->regIdx),
                               static_cast<uint16_t>(regCont->regNum)};

            // TODO: This is a hack to set the offset of the register for use case such as msb, etc.
            reg.reg.offset = static_cast<int16_t>(-256 * regCont->msb);

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
    /// \return StinkyRegister representing the input value
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
            rocisa::Item*     item       = itemShared.get();
            const std::string itemString = item->toString();

            // Handle text blocks
            if(rocisa::TextBlock* textBlock = dynamic_cast<rocisa::TextBlock*>(item))
            {
                AsmDirective* directive = new AsmDirective();
                directive->kind         = AsmDirectiveKind::TEXTBLOCK;
                directive->value        = textBlock->text;
                insts.insert(insts.end(), directive);
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
                size_t pos = itemString.rfind(',');
                if(pos != std::string::npos)
                {
                    directive->value = itemString.substr(pos + 1);
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
                directive->value        = itemString;
                insts.insert(insts.end(), directive);
                continue;
            }

            // Handle ValueIf directives
            if(rocisa::ValueIf* valueIf = dynamic_cast<rocisa::ValueIf*>(item))
            {
                AsmDirective* directive = new AsmDirective();
                directive->kind         = AsmDirectiveKind::IF;
                directive->name         = ".if";
                directive->symbol       = std::to_string(valueIf->value);
                directive->value        = itemString;
                insts.insert(insts.end(), directive);
                continue;
            }

            // Handle ValueEndif directives
            if(rocisa::ValueEndif* valueEndif = dynamic_cast<rocisa::ValueEndif*>(item))
            {
                AsmDirective* directive = new AsmDirective();
                directive->kind         = AsmDirectiveKind::ENDIF;
                directive->name         = ".endif";
                directive->comment      = valueEndif->comment;
                directive->value        = itemString;
                insts.insert(insts.end(), directive);
                continue;
            }

            // Handle instructions
            rocisa::Instruction* inst = dynamic_cast<rocisa::Instruction*>(item);
            if(inst == nullptr)
            {
                // TODO: Remove this once we have a better way to handle non-instruction items
                std::cout << "Skipping non-instruction item: " << itemString << std::endl;
                continue;
            }
            assert(dynamic_cast<rocisa::SSetVgprMsb*>(inst) == nullptr
                   && "SSetVgprMsb should not be created directly in TensileLite");

            // Handle ds_load_b192 and ds_store_b192 instructions, also capture s_set_vgpr_msb
            // TODO: This is a temporary solution, we should use legalizeDSLoadB192 and legalizeDSStoreB192 instead.
            if(handleDSLDSTb192Instructions(inst, itemString, irBuilder, insts, archId))
            {
                continue;
            }

            // try to capture s_set_vgpr_msb instruction from the instruction string
            tryCaptureVgprMsb(itemString, irBuilder, insts, archId);

            // Create StinkyInstruction from rocisa instruction
            std::type_index    rocisaTy = std::type_index(typeid(*inst));
            StinkyInstruction* stinkyInst
                = createStinkyInstructionFromRocisa(*inst, rocisaTy, irBuilder, insts, archId);

            assert(stinkyInst != nullptr
                   && "Failed to create StinkyInstruction from rocisa instruction");

            // Add registers (sources and destinations) to the instruction
            addRegistersToInstruction(stinkyInst, inst, asmCaps, archId);

            // Add modifiers (DS, FLAT, MUBUF, SMEM, WaitCnt, comments)
            addModifiersToInstruction(stinkyInst, inst, itemString, asmCaps);

            // Legalize instructions
            auto [firstLegalizedInst, lastLegalizedInst]
                = legalizeInstruction(stinkyInst, inst, irBuilder, archId, asmCaps);

            // Don't use stinkyInst if it was replaced by legalization
            if(firstLegalizedInst != nullptr)
                stinkyInst = nullptr;
        }

        // Iterate through the instructions in the IRList and add them to the module
        for(IRBase& ir : insts)
        {
            stinkyAsmModule.add(static_cast<IRBase*>(&ir));
        }

        return std::make_shared<StinkyAsmModule>(std::move(stinkyAsmModule));
    }

    /// Convert rocisa::SignatureValueKind to stinkytofu::SignatureValueKind
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

    /// Convert rocisa::SignatureBase to stinkytofu::SignatureBase
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

        // Note: Optimization config (ThreadTile, SubGroup, VectorWidth, etc.) is now passed
        // directly from Python via toStinkyTofuModule's parameters and set via setOptimizationConfig()

        return stinkySig;
    }

} // namespace stinkytofu

/// Initialize StinkyTofu Python bindings
///
/// This function binds the rocisa to StinkyTofu utilities to Python, allowing
/// Python code to convert rocisa to StinkyTofu IR.
///
/// \param m The nanobind module to add bindings to
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
            auto stinkySig = stinkytofu::toStinkySignature(signature, archArray, wavefrontSize);

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
