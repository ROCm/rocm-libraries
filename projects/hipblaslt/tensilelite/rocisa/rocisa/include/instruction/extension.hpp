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
#pragma once
#include "base.hpp"
#include "code.hpp"
#include "container.hpp"
#include "instruction/branch.hpp"
#include "instruction/cmp.hpp"
#include "instruction/common.hpp"
#include "instruction/cvt.hpp"

namespace rocisa
{
    ////////////////////////////////////////
    // Branch
    ////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////
    // longBranch - 32 bit offset
    // s_branch class instructions take a label operand which is truncated to 16 bit
    // If the target label address offset is greater than 16 bits, then
    // we must use a longer 32 bit version.
    // Use when erroring out "invalid operand due to label > SIMM16"
    //////////////////////////////////////////////////////////////////////////////
    inline std::shared_ptr<Module> SLongBranch(const Label&        label,
                                               ContinuousRegister& tmpSgprRes,
                                               const std::string&  positiveLabelStr,
                                               const std::string&  comment = "")
    {
        std::string labelName = label.getLabelName();
        auto        module    = std::make_shared<Module>("SLongBranch " + labelName);
        if(!comment.empty())
        {
            module->addComment(comment);
        }

        if(tmpSgprRes.size < 3)
        {
            throw std::runtime_error("ContinuousRegister size must be at least 3.");
        }
        int   tmpSgpr = tmpSgprRes.idx;
        Label positiveLabel(positiveLabelStr, "");
        module->addT<SGetPCB64>(sgpr(tmpSgpr, 2), "addr of next instr");
        module->addT<SAddI32>(sgpr(tmpSgpr + 2), labelName, 4, "target branch offset");
        module->addT<SCmpGeI32>(sgpr(tmpSgpr + 2), 0, "check positive or negative");
        module->addT<SCBranchSCC1>(positiveLabel.getLabelName(), "jump when positive");

        // negative offset
        module->addT<SAbsI32>(sgpr(tmpSgpr + 2), sgpr(tmpSgpr + 2), "abs offset");
        module->addT<SSubU32>(
            sgpr(tmpSgpr), sgpr(tmpSgpr), sgpr(tmpSgpr + 2), "sub target branch offset");
        module->addT<SSubBU32>(sgpr(tmpSgpr + 1), sgpr(tmpSgpr + 1), 0, "sub high and carry");
        module->addT<SSetPCB64>(sgpr(tmpSgpr, 2), "branch to " + labelName);

        // positive offset
        module->addT<Label>(positiveLabel);
        module->addT<SAddU32>(
            sgpr(tmpSgpr), sgpr(tmpSgpr), sgpr(tmpSgpr + 2), "add target branch offset");
        module->addT<SAddCU32>(sgpr(tmpSgpr + 1), sgpr(tmpSgpr + 1), 0, "add high and carry");
        module->addT<SSetPCB64>(sgpr(tmpSgpr, 2), "branch to " + labelName);

        return module;
    }

    ////////////////////////////////////////////////////////////////////////////////
    // longBranchPositive - 32 bit offset (positive offset only)
    // s_branch class instructions take a label operand which is truncated to 16 bit
    // If the target label address offset is greater than 16 bits, then
    // we must use a longer 32 bit version.
    // Use when erroring out "invalid operand due to label > SIMM16"
    ////////////////////////////////////////////////////////////////////////////////
    inline std::shared_ptr<Module>
        SGetPositivePCOffset(int sgprIdx, const Label& label, ContinuousRegister& tmpSgprRes)
    {
        std::string labelName = label.getLabelName();
        auto        module    = std::make_shared<Module>("SGetPositivePCOffset " + labelName);
        if(tmpSgprRes.size < 1)
        {
            throw std::runtime_error("ContinuousRegister size must be at least 1.");
        }
        int tmpSgpr = tmpSgprRes.idx;
        module->addT<SGetPCB64>(sgpr(sgprIdx, 2), "addr of next instr");
        module->addT<SAddI32>(sgpr(tmpSgpr), labelName, 4, "target branch offset");

        // positive offset
        module->addT<SAddU32>(
            sgpr(sgprIdx), sgpr(sgprIdx), sgpr(tmpSgpr), "add target branch offset");
        module->addT<SAddCU32>(sgpr(sgprIdx + 1), sgpr(sgprIdx + 1), 0, "add high and carry");

        return module;
    }

    inline std::shared_ptr<Module> SLongBranchPositive(const Label&        label,
                                                       ContinuousRegister& tmpSgprRes,
                                                       const std::string&  comment = "")
    {
        std::string labelName = label.getLabelName();
        auto        module    = std::make_shared<Module>("SLongBranchPositive " + labelName);
        if(!comment.empty())
        {
            module->addComment(comment);
        }

        if(rocIsa::getInstance().getAsmCaps()["HasAdd_PC_i64"])
        {
            //what '.' does is to create a label right before that instruction
            //So s_add_pc_i64 (target_label - .) effectively becomes:
            //      .temp_label
            //      s_add_pc_i64 (target_label - temp_label)
            //the size of s_add_pc_i64 (target_label - temp_label) is 8 + 4 bytes, so you will need to do -12 as well

            //TODO: remove hardcore "-.-12" when compiler update for label value calculation
            module->addT<SAddPCI64_SIMM>(
                labelName + "-.-12",
                "Add PC to " + labelName
                    + ", the constant correction is dependent on the current assembler behavior.");
        }
        else
        {
            if(tmpSgprRes.size < 3)
            {
                throw std::runtime_error("ContinuousRegister size must be at least 3.");
            }
            int tmpSgprX2, tmpSgprX1;
            if(tmpSgprRes.idx % 2 == 0)
            {
                tmpSgprX2 = tmpSgprRes.idx;
                tmpSgprX1 = tmpSgprRes.idx + 2;
            }
            else
            {
                tmpSgprX2 = tmpSgprRes.idx + 1;
                tmpSgprX1 = tmpSgprRes.idx;
            }
            auto cr = ContinuousRegister(tmpSgprX1, 1);
            module->add(SGetPositivePCOffset(tmpSgprX2, label, cr));
            module->addT<SSetPCB64>(sgpr(tmpSgprX2, 2), "branch to " + labelName);
        }

        return module;
    }

    inline std::shared_ptr<Module> SLongBranchNegative(const Label&        label,
                                                       ContinuousRegister& tmpSgprRes,
                                                       const std::string&  comment = "")
    {
        std::string labelName = label.getLabelName();
        auto        module    = std::make_shared<Module>("SLongBranchNegative " + labelName);
        if(!comment.empty())
        {
            module->addComment(comment);
        }

        if(tmpSgprRes.size < 3)
        {
            throw std::runtime_error("ContinuousRegister size must be at least 3.");
        }

        int tmpSgprX2, tmpSgprX1;
        if(tmpSgprRes.idx % 2 == 0)
        {
            tmpSgprX2 = tmpSgprRes.idx;
            tmpSgprX1 = tmpSgprRes.idx + 2;
        }
        else
        {
            tmpSgprX2 = tmpSgprRes.idx + 1;
            tmpSgprX1 = tmpSgprRes.idx;
        }

        module->addT<SGetPCB64>(sgpr(tmpSgprX2, 2), "addr of next instr");
        module->addT<SAddI32>(sgpr(tmpSgprX1), labelName, 4, "target branch offset");

        // negative offset
        module->addT<SAbsI32>(sgpr(tmpSgprX1), sgpr(tmpSgprX1), "abs offset");
        module->addT<SSubU32>(
            sgpr(tmpSgprX2), sgpr(tmpSgprX2), sgpr(tmpSgprX1), "sub target branch offset");
        module->addT<SSubBU32>(sgpr(tmpSgprX2 + 1), sgpr(tmpSgprX2 + 1), 0, "sub high and carry");
        module->addT<SSetPCB64>(sgpr(tmpSgprX2, 2), "branch to " + labelName);

        return module;
    }

    //////////////////////////////////////////////////////////////////////////////
    // longBranchScc0 - 32 bit offset
    // Conditional branch to label when SCC == 0
    // Use when erroring out "invalid operand due to label > SIMM16"
    //////////////////////////////////////////////////////////////////////////////
    inline std::shared_ptr<Module> SCLongBranchScc0(const Label&        label,
                                                    ContinuousRegister& tmpSgprRes,
                                                    const std::string&  noBranchLabelStr,
                                                    const std::string&  positiveLabelStr,
                                                    int                 posNeg  = 0,
                                                    const std::string&  comment = "")
    {
        auto  module = std::make_shared<Module>("SCLongBranchScc0 " + label.getLabelName());
        Label noBranchLabel(noBranchLabelStr, "");
        module->addT<SCBranchSCC1>(noBranchLabel.getLabelName(), "Only branch on scc0");
        if(posNeg > 0)
        {
            module->add(SLongBranchPositive(label, tmpSgprRes, comment));
        }
        else if(posNeg < 0)
        {
            module->add(SLongBranchNegative(label, tmpSgprRes, comment));
        }
        else
        {
            module->add(SLongBranch(label, tmpSgprRes, positiveLabelStr, comment));
        }
        module->addT<Label>(noBranchLabel);
        return module;
    }

    //////////////////////////////////////////////////////////////////////////////
    // longBranchScc1 - 32 bit offset
    // Conditional branch to label when SCC == 1
    // Use when erroring out "invalid operand due to label > SIMM16"
    //////////////////////////////////////////////////////////////////////////////
    inline std::shared_ptr<Module> SCLongBranchScc1(const Label&        label,
                                                    ContinuousRegister& tmpSgprRes,
                                                    const std::string&  noBranchLabelStr,
                                                    const std::string&  positiveLabelStr,
                                                    int                 posNeg  = 0,
                                                    const std::string&  comment = "")
    {
        auto  module = std::make_shared<Module>("SCLongBranchScc1 " + label.getLabelName());
        Label noBranchLabel(noBranchLabelStr, "");
        module->addT<SCBranchSCC0>(noBranchLabel.getLabelName(), "Only branch on scc1");
        if(posNeg > 0)
        {
            module->add(SLongBranchPositive(label, tmpSgprRes, comment));
        }
        else if(posNeg < 0)
        {
            module->add(SLongBranchNegative(label, tmpSgprRes, comment));
        }
        else
        {
            module->add(SLongBranch(label, tmpSgprRes, positiveLabelStr, comment));
        }
        module->addT<Label>(noBranchLabel);
        return module;
    }

    //////////////////////////////////////////////////////////////////////////////
    // longBranchVccnz - 32 bit offset
    // Conditional branch to label when VCC != 0
    // Use when erroring out "invalid operand due to label > SIMM16"
    // VCC != 0 executes long branch, VCC == 0 do SCBranchVCCZ to skip long branch.
    //////////////////////////////////////////////////////////////////////////////
    inline std::shared_ptr<Module> SCLongBranchVccnz(const Label&        label,
                                                     ContinuousRegister& tmpSgprRes,
                                                     const std::string&  noBranchLabelStr,
                                                     const std::string&  positiveLabelStr,
                                                     int                 posNeg  = 0,
                                                     const std::string&  comment = "")
    {
        auto  module = std::make_shared<Module>("SCLongBranchVccnz " + label.getLabelName());
        Label noBranchLabel(noBranchLabelStr, "");
        module->addT<SCBranchVCCZ>(noBranchLabel.getLabelName(), "Only branch on vccz");
        if(posNeg > 0)
        {
            module->add(SLongBranchPositive(label, tmpSgprRes, comment));
        }
        else if(posNeg < 0)
        {
            module->add(SLongBranchNegative(label, tmpSgprRes, comment));
        }
        else
        {
            module->add(SLongBranch(label, tmpSgprRes, positiveLabelStr, comment));
        }
        module->addT<Label>(noBranchLabel);
        return module;
    }

    // Perform 32-bit scalar mul and save 64-bit result in two SGPR
    // src0 and src1 are 32-bit ints in scalar sgpr or small int constants (<64?))
    // sign indicates if input and output data is signed
    // return returns in dst0:dest (lower 32-bit in dst0, high 64-bit in dst1))
    // Requires 2 tmp vgprs
    inline std::shared_ptr<Module> SMulInt64to32(const std::shared_ptr<RegisterContainer>& dst0,
                                                 const std::shared_ptr<RegisterContainer>& dst1,
                                                 const InstructionInput&                   src0,
                                                 const InstructionInput&                   src1,
                                                 const ContinuousRegister& tmpVgprRes,
                                                 bool                      hasSMulHi,
                                                 bool                      sign,
                                                 const std::string&        comment = "")
    {
        auto module = std::make_shared<Module>("SMulInt64to32");
        if(tmpVgprRes.size < 2)
        {
            throw std::runtime_error("ContinuousRegister size must be at least 2.");
        }
        if(auto src0data = std::get_if<std::shared_ptr<Container>>(&src0))
        {
            if(auto src0Reg = std::dynamic_pointer_cast<RegisterContainer>(*src0data))
            {
                if(dst1 == src0Reg)
                {
                    throw std::runtime_error("dst1 cannot be the same as src0.");
                }
            }
        }
        if(auto src1data = std::get_if<std::shared_ptr<Container>>(&src1))
        {
            if(auto src1Reg = std::dynamic_pointer_cast<RegisterContainer>(*src1data))
            {
                if(dst1 == src1Reg)
                {
                    throw std::runtime_error("dst1 cannot be the same as src1.");
                }
            }
        }
        // the else path below has less restrictions but prefer consistency
        if(hasSMulHi)
        {
            if(sign)
            {
                module->addT<SMulHII32>(dst1, src0, src1, comment);
            }
            else
            {
                module->addT<SMulHIU32>(dst1, src0, src1, comment);
            }
            module->addT<SMulI32>(dst0, src0, src1, comment);
        }
        else
        {
            auto swapSrc = [&src0, &src1]() {
                if(auto src1data = std::get_if<std::shared_ptr<Container>>(&src1))
                {
                    return std::make_pair(src0, src1);
                }
                return std::make_pair(src1, src0);
            };
            auto [swapSrc0, swapSrc1] = swapSrc();
            auto vgprTmp              = vgpr(tmpVgprRes.idx);
            auto vgprTmp1             = vgpr(tmpVgprRes.idx + 1);
            module->addT<VMovB32>(vgprTmp, swapSrc0, std::nullopt, comment);
            if(sign)
            {
                module->addT<VMulHII32>(vgprTmp1, vgprTmp, swapSrc1, comment);
            }
            else
            {
                module->addT<VMulHIU32>(vgprTmp1, vgprTmp, swapSrc1, comment);
            }
            module->addT<VReadfirstlaneB32>(dst1, vgprTmp1, comment);
            module->addT<VMulLOU32>(vgprTmp1, vgprTmp, swapSrc1, comment);
            module->addT<VReadfirstlaneB32>(dst0, vgprTmp1, comment);
        }
        return module;
    }

    ////////////////////////////////////////////////////////////////////////////////
    // True16 conversion helpers.
    // These query the NoSDWA arch cap at runtime and emit either:
    //   - true16 encoding (op_sel / byte_sel) on NoSDWA targets (gfx11+), or
    //   - SDWA encoding (src0_sel / dst_sel)  on legacy targets.
    // Callers only provide a semantic `sel` (HighBitSel::LOW or HighBitSel::HIGH).
    ////////////////////////////////////////////////////////////////////////////////
    /// Convert F16 → F32, selecting src half-word by `sel`.
    inline std::shared_ptr<Item>
        ECvtF16toF32(const std::shared_ptr<RegisterContainer>& dst,
                     const InstructionInput&                   src,
                     HighBitSel                                sel,
                     const std::string&                        comment = "")
    {
        rocIsa& instance = rocIsa::getInstance();
        if(instance.getArchCaps()["NoSDWA"])
        {
            return std::make_shared<VCvtF16toF32>(
                dst,
                src,
                std::nullopt,
                std::vector<int>{-1, -1, static_cast<int>(sel)},
                comment);
        }

        SDWAModifiers sdwa;
        sdwa.src0_sel = (sel == HighBitSel::HIGH) ? SelectBit::WORD_1
                                                  : SelectBit::WORD_0;
        return std::make_shared<VCvtF16toF32>(
            dst, src, sdwa, std::vector<int>{}, comment);
    }

    /**
     * Convert F32->F16 with optional half-word packing.
     *
     * @param sel  std::nullopt -> plain VCvtF32toF16, no half-word modifier.
     *             HighBitSel::LOW / HIGH -> pack result into the selected
     *             half-word via op_sel (NoSDWA / true16) or SDWA dst_sel
     *             (legacy).
     */
    inline std::shared_ptr<Item>
        ECvtF32toF16(const std::shared_ptr<RegisterContainer>& dst,
                     const InstructionInput&                   src,
                     std::optional<HighBitSel>                 sel     = std::nullopt,
                     const std::string&                        comment = "")
    {
        if(!sel.has_value())
        {
            return std::make_shared<VCvtF32toF16>(
                dst, src, std::nullopt, std::vector<int>{}, comment);
        }

        rocIsa& instance = rocIsa::getInstance();
        if(instance.getArchCaps()["NoSDWA"])
        {
            return std::make_shared<VCvtF32toF16>(
                dst,
                src,
                std::nullopt,
                std::vector<int>{static_cast<int>(*sel)},
                comment);
        }

        SDWAModifiers sdwa;
        sdwa.dst_sel = (*sel == HighBitSel::HIGH) ? SelectBit::WORD_1
                                                  : SelectBit::WORD_0;
        return std::make_shared<VCvtF32toF16>(
            dst, src, sdwa, std::vector<int>{}, comment);
    }

    /// Unpack packed-FP8 → 2×F32, selecting src half-word by `sel`.
    inline std::shared_ptr<Item>
        ECvtPkFP8toF32(const std::shared_ptr<RegisterContainer>& dst,
                       const InstructionInput&                   src,
                       HighBitSel                                sel,
                       const std::string&                        comment = "")
    {
        int     selInt   = static_cast<int>(sel);
        rocIsa& instance = rocIsa::getInstance();
        if(instance.getArchCaps()["NoSDWA"])
        {
            VOP3PModifiers vop3;
            vop3.op_sel.push_back(selInt);
            return std::make_shared<VCvtPkFP8toF32>(
                dst,
                src,
                std::nullopt,
                vop3,
                std::vector<int>{-1, -1, selInt},
                comment);
        }

        SDWAModifiers sdwa;
        sdwa.src0_sel = (sel == HighBitSel::HIGH) ? SelectBit::WORD_1
                                                  : SelectBit::WORD_0;
        return std::make_shared<VCvtPkFP8toF32>(
            dst, src, sdwa, std::nullopt, std::vector<int>{}, comment);
    }

    /// Unpack packed-BF8 → 2×F32, selecting src half-word by `sel`.
    inline std::shared_ptr<Item>
        ECvtPkBF8toF32(const std::shared_ptr<RegisterContainer>& dst,
                       const InstructionInput&                   src,
                       HighBitSel                                sel,
                       const std::string&                        comment = "")
    {
        int     selInt   = static_cast<int>(sel);
        rocIsa& instance = rocIsa::getInstance();
        if(instance.getArchCaps()["NoSDWA"])
        {
            VOP3PModifiers vop3;
            vop3.op_sel.push_back(selInt);
            return std::make_shared<VCvtPkBF8toF32>(
                dst,
                src,
                std::nullopt,
                vop3,
                std::vector<int>{-1, -1, selInt},
                comment);
        }

        SDWAModifiers sdwa;
        sdwa.src0_sel = (sel == HighBitSel::HIGH) ? SelectBit::WORD_1
                                                  : SelectBit::WORD_0;
        return std::make_shared<VCvtPkBF8toF32>(
            dst, src, sdwa, std::nullopt, std::vector<int>{}, comment);
    }

    inline std::shared_ptr<Item>
        VCvtBF16toFP32(const std::shared_ptr<RegisterContainer>&         dst,
                       const std::shared_ptr<RegisterContainer>&         src,
                       std::optional<std::shared_ptr<RegisterContainer>> vgprMask,
                       int                                               vi,
                       const std::string&                                comment = "")
    {
        rocIsa& instance = rocIsa::getInstance();
        if(!instance.getAsmCaps()["HasBF16CVT"])
        {
            if((vi % 2) == 1)
            {
                if(!vgprMask.has_value())
                    throw std::runtime_error("vgprMask is null");
                return std::make_shared<VAndB32>(
                    dst, src, *vgprMask, "cvt bf16 to fp32. " + comment);
            }
            return std::make_shared<VLShiftLeftB32>(
                dst, 16, src, "cvt bf16 to fp32. " + comment);
        }

        if(instance.getArchCaps()["NoSDWA"])
        {
            VOP3PModifiers vop3;
            vop3.op_sel.push_back(vi % 2);
            return std::make_shared<PVCvtBF16toFP32>(
                dst, src, std::nullopt, vop3, std::vector<int>{-1, -1, vi % 2}, "cvt bf16 to f32");
        }

        SDWAModifiers sdwa;
        sdwa.src0_sel = (vi % 2 == 1) ? SelectBit::WORD_1 : SelectBit::WORD_0;
        return std::make_shared<PVCvtBF16toFP32>(
            dst, src, sdwa, std::nullopt, std::vector<int>{}, "cvt bf16 to f32");
    }
} // namespace rocisa
