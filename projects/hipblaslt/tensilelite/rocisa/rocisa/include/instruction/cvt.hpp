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
#include "instruction.hpp"

namespace rocisa
{
    struct VCvtInstruction : public CommonInstruction
    {
        CvtType cvtType;

        VCvtInstruction(CvtType                                   cvtType,
                        const std::shared_ptr<RegisterContainer>& dst,
                        const std::vector<InstructionInput>&      srcs,
                        const std::optional<SDWAModifiers>&       sdwa    = std::nullopt,
                        const std::optional<VOP3PModifiers>&      vop3    = std::nullopt,
                        const std::string&                        comment = "")
            : CommonInstruction(InstType::INST_CVT, dst, srcs, std::nullopt, sdwa, vop3, comment)
            , cvtType(cvtType)
        {
        }

        VCvtInstruction(const VCvtInstruction& other)
            : CommonInstruction(other)
            , cvtType(other.cvtType)
        {
        }
    };

    struct VCvtF16toF32 : public VCvtInstruction
    {
        VCvtF16toF32(const std::shared_ptr<RegisterContainer>& dst,
                     const InstructionInput&                   src,
                     const std::optional<SDWAModifiers>&       sdwa    = std::nullopt,
                     const std::string&                        comment = "")
            : VCvtInstruction(CvtType::CVT_F16_to_F32, dst, {src}, sdwa, std::nullopt, comment)
        {
            setInst("v_cvt_f32_f16");
        }

        VCvtF16toF32(const VCvtF16toF32& other)
            : VCvtInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCvtF16toF32>(*this);
        }
    };

    struct VCvtF32toF16 : public VCvtInstruction
    {
        VCvtF32toF16(const std::shared_ptr<RegisterContainer>& dst,
                     const InstructionInput&                   src,
                     const std::optional<SDWAModifiers>&       sdwa    = std::nullopt,
                     const std::string&                        comment = "")
            : VCvtInstruction(CvtType::CVT_F32_to_F16, dst, {src}, sdwa, std::nullopt, comment)
        {
            setInst("v_cvt_f16_f32");
        }

        VCvtF32toF16(const VCvtF32toF16& other)
            : VCvtInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCvtF32toF16>(*this);
        }
    };

    struct VCvtF32toU32 : public VCvtInstruction
    {
        VCvtF32toU32(const std::shared_ptr<RegisterContainer>& dst,
                     const InstructionInput&                   src,
                     const std::optional<SDWAModifiers>&       sdwa    = std::nullopt,
                     const std::string&                        comment = "")
            : VCvtInstruction(CvtType::CVT_F32_to_U32, dst, {src}, sdwa, std::nullopt, comment)
        {
            setInst("v_cvt_u32_f32");
        }

        VCvtF32toU32(const VCvtF32toU32& other)
            : VCvtInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCvtF32toU32>(*this);
        }
    };

    struct VCvtU32toF32 : public VCvtInstruction
    {
        VCvtU32toF32(const std::shared_ptr<RegisterContainer>& dst,
                     const InstructionInput&                   src,
                     const std::optional<SDWAModifiers>&       sdwa    = std::nullopt,
                     const std::string&                        comment = "")
            : VCvtInstruction(CvtType::CVT_U32_to_F32, dst, {src}, sdwa, std::nullopt, comment)
        {
            setInst("v_cvt_f32_u32");
        }

        VCvtU32toF32(const VCvtU32toF32& other)
            : VCvtInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCvtU32toF32>(*this);
        }
    };

    struct VCvtF64toU32 : public VCvtInstruction
    {
        VCvtF64toU32(const std::shared_ptr<RegisterContainer>& dst,
                     const InstructionInput&                   src,
                     const std::optional<SDWAModifiers>&       sdwa    = std::nullopt,
                     const std::string&                        comment = "")
            : VCvtInstruction(CvtType::CVT_F64_to_U32, dst, {src}, sdwa, std::nullopt, comment)
        {
            setInst("v_cvt_u32_f64");
        }

        VCvtF64toU32(const VCvtF64toU32& other)
            : VCvtInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCvtF64toU32>(*this);
        }
    };

    struct VCvtU32toF64 : public VCvtInstruction
    {
        VCvtU32toF64(const std::shared_ptr<RegisterContainer>& dst,
                     const InstructionInput&                   src,
                     const std::optional<SDWAModifiers>&       sdwa    = std::nullopt,
                     const std::string&                        comment = "")
            : VCvtInstruction(CvtType::CVT_U32_to_F64, dst, {src}, sdwa, std::nullopt, comment)
        {
            setInst("v_cvt_f64_u32");
        }

        VCvtU32toF64(const VCvtU32toF64& other)
            : VCvtInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCvtU32toF64>(*this);
        }
    };

    struct VCvtI32toF32 : public VCvtInstruction
    {
        VCvtI32toF32(const std::shared_ptr<RegisterContainer>& dst,
                     const InstructionInput&                   src,
                     const std::optional<SDWAModifiers>&       sdwa    = std::nullopt,
                     const std::string&                        comment = "")
            : VCvtInstruction(CvtType::CVT_I32_to_F32, dst, {src}, sdwa, std::nullopt, comment)
        {
            setInst("v_cvt_f32_i32");
        }

        VCvtI32toF32(const VCvtI32toF32& other)
            : VCvtInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCvtI32toF32>(*this);
        }
    };

    struct VCvtF32toI32 : public VCvtInstruction
    {
        VCvtF32toI32(const std::shared_ptr<RegisterContainer>& dst,
                     const InstructionInput&                   src,
                     const std::optional<SDWAModifiers>&       sdwa    = std::nullopt,
                     const std::string&                        comment = "")
            : VCvtInstruction(CvtType::CVT_F32_to_I32, dst, {src}, sdwa, std::nullopt, comment)
        {
            setInst("v_cvt_i32_f32");
        }

        VCvtF32toI32(const VCvtF32toI32& other)
            : VCvtInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCvtF32toI32>(*this);
        }
    };

    struct VCvtFP8toF32 : public VCvtInstruction
    {
        VCvtFP8toF32(const std::shared_ptr<RegisterContainer>& dst,
                     const InstructionInput&                   src,
                     const std::optional<SDWAModifiers>&       sdwa    = std::nullopt,
                     const std::optional<VOP3PModifiers>&      vop3    = std::nullopt,
                     const std::string&                        comment = "")
            : VCvtInstruction(CvtType::CVT_FP8_to_F32, dst, {src}, sdwa, vop3, comment)
        {
            setInst("v_cvt_f32_fp8");
        }

        VCvtFP8toF32(const VCvtFP8toF32& other)
            : VCvtInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCvtFP8toF32>(*this);
        }
    };

    struct VCvtBF8toF32 : public VCvtInstruction
    {
        VCvtBF8toF32(const std::shared_ptr<RegisterContainer>& dst,
                     const InstructionInput&                   src,
                     const std::optional<SDWAModifiers>&       sdwa    = std::nullopt,
                     const std::optional<VOP3PModifiers>&      vop3    = std::nullopt,
                     const std::string&                        comment = "")
            : VCvtInstruction(CvtType::CVT_BF8_to_F32, dst, {src}, sdwa, vop3, comment)
        {
            setInst("v_cvt_f32_bf8");
        }

        VCvtBF8toF32(const VCvtBF8toF32& other)
            : VCvtInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCvtBF8toF32>(*this);
        }
    };

    struct VCvtPkFP8toF32 : public VCvtInstruction
    {
        VCvtPkFP8toF32(const std::shared_ptr<RegisterContainer>& dst,
                       const InstructionInput&                   src,
                       const std::optional<SDWAModifiers>&       sdwa    = std::nullopt,
                       const std::optional<VOP3PModifiers>&      vop3    = std::nullopt,
                       const std::string&                        comment = "")
            : VCvtInstruction(CvtType::CVT_PK_FP8_to_F32, dst, {src}, sdwa, vop3, comment)
        {
            setInst("v_cvt_pk_f32_fp8");
        }

        VCvtPkFP8toF32(const VCvtPkFP8toF32& other)
            : VCvtInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCvtPkFP8toF32>(*this);
        }
    };

    struct VCvtPkBF8toF32 : public VCvtInstruction
    {
        VCvtPkBF8toF32(const std::shared_ptr<RegisterContainer>& dst,
                       const InstructionInput&                   src,
                       const std::optional<SDWAModifiers>&       sdwa    = std::nullopt,
                       const std::optional<VOP3PModifiers>&      vop3    = std::nullopt,
                       const std::string&                        comment = "")
            : VCvtInstruction(CvtType::CVT_PK_BF8_to_F32, dst, {src}, sdwa, vop3, comment)
        {
            setInst("v_cvt_pk_f32_bf8");
        }

        VCvtPkBF8toF32(const VCvtPkBF8toF32& other)
            : VCvtInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCvtPkBF8toF32>(*this);
        }
    };

    struct VCvtPkF32toFP8 : public VCvtInstruction
    {
        VCvtPkF32toFP8(const std::shared_ptr<RegisterContainer>& dst,
                       const InstructionInput&                   src0,
                       const InstructionInput&                   src1,
                       const std::optional<VOP3PModifiers>&      vop3    = std::nullopt,
                       const std::string&                        comment = "")
            : VCvtInstruction(
                CvtType::CVT_PK_F32_to_FP8, dst, {src0, src1}, std::nullopt, vop3, comment)
        {
            setInst("v_cvt_pk_fp8_f32");
        }

        VCvtPkF32toFP8(const VCvtPkF32toFP8& other)
            : VCvtInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCvtPkF32toFP8>(*this);
        }
    };

    struct VCvtPkF32toBF8 : public VCvtInstruction
    {
        VCvtPkF32toBF8(const std::shared_ptr<RegisterContainer>& dst,
                       const InstructionInput&                   src0,
                       const InstructionInput&                   src1,
                       const std::optional<VOP3PModifiers>&      vop3    = std::nullopt,
                       const std::string&                        comment = "")
            : VCvtInstruction(
                CvtType::CVT_PK_F32_to_BF8, dst, {src0, src1}, std::nullopt, vop3, comment)
        {
            setInst("v_cvt_pk_bf8_f32");
        }

        VCvtPkF32toBF8(const VCvtPkF32toBF8& other)
            : VCvtInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCvtPkF32toBF8>(*this);
        }
    };

    struct VCvtSRF32toFP8 : public VCvtInstruction
    {
        VCvtSRF32toFP8(const std::shared_ptr<RegisterContainer>& dst,
                       const InstructionInput&                   src0,
                       const InstructionInput&                   src1,
                       const std::optional<VOP3PModifiers>&      vop3    = std::nullopt,
                       const std::string&                        comment = "")
            : VCvtInstruction(
                CvtType::CVT_SR_F32_to_FP8, dst, {src0, src1}, std::nullopt, vop3, comment)
        {
            setInst("v_cvt_sr_fp8_f32");
        }

        VCvtSRF32toFP8(const VCvtSRF32toFP8& other)
            : VCvtInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCvtSRF32toFP8>(*this);
        }
    };

    struct VCvtSRF32toBF8 : public VCvtInstruction
    {
        VCvtSRF32toBF8(const std::shared_ptr<RegisterContainer>& dst,
                       const InstructionInput&                   src0,
                       const InstructionInput&                   src1,
                       const std::optional<VOP3PModifiers>&      vop3    = std::nullopt,
                       const std::string&                        comment = "")
            : VCvtInstruction(
                CvtType::CVT_SR_F32_to_BF8, dst, {src0, src1}, std::nullopt, vop3, comment)
        {
            setInst("v_cvt_sr_bf8_f32");
        }

        VCvtSRF32toBF8(const VCvtSRF32toBF8& other)
            : VCvtInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCvtSRF32toBF8>(*this);
        }
    };

    struct VCvtScalePkFP8toF16 : public VCvtInstruction
    {
        VCvtScalePkFP8toF16(const std::shared_ptr<RegisterContainer>& dst,
                            const std::shared_ptr<Container>&         src,
                            const InstructionInput&                   scale,
                            std::optional<SDWAModifiers>              sdwa    = std::nullopt,
                            std::optional<VOP3PModifiers>             vop3    = std::nullopt,
                            const std::string&                        comment = "")
            : VCvtInstruction(
                CvtType::CVT_SCALEF32_PK_F16_FP8, dst, {src, scale}, sdwa, vop3, comment)
        {
            setInst("v_cvt_scalef32_pk_f16_fp8");
        }

        VCvtScalePkFP8toF16(const VCvtScalePkFP8toF16& other)
            : VCvtInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCvtScalePkFP8toF16>(*this);
        }
    };

    struct VCvtScalePkBF8toF16 : public VCvtInstruction
    {
        VCvtScalePkBF8toF16(const std::shared_ptr<RegisterContainer>& dst,
                            const std::shared_ptr<Container>&         src,
                            const InstructionInput&                   scale,
                            std::optional<SDWAModifiers>              sdwa    = std::nullopt,
                            std::optional<VOP3PModifiers>             vop3    = std::nullopt,
                            const std::string&                        comment = "")
            : VCvtInstruction(
                CvtType::CVT_SCALEF32_PK_F16_BF8, dst, {src, scale}, sdwa, vop3, comment)
        {
            setInst("v_cvt_scalef32_pk_f16_bf8");
        }

        VCvtScalePkBF8toF16(const VCvtScalePkBF8toF16& other)
            : VCvtInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCvtScalePkBF8toF16>(*this);
        }
    };

    struct VCvtScaleFP8toF16 : public VCvtInstruction
    {
        VCvtScaleFP8toF16(const std::shared_ptr<RegisterContainer>& dst,
                          const std::shared_ptr<Container>&         src,
                          const InstructionInput&                   scale,
                          std::optional<SDWAModifiers>              sdwa    = std::nullopt,
                          std::optional<VOP3PModifiers>             vop3    = std::nullopt,
                          const std::string&                        comment = "")
            : VCvtInstruction(CvtType::CVT_SCALEF32_F16_FP8, dst, {src, scale}, sdwa, vop3, comment)
        {
            setInst("v_cvt_scalef32_f16_fp8");
        }

        VCvtScaleFP8toF16(const VCvtScaleFP8toF16& other)
            : VCvtInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCvtScaleFP8toF16>(*this);
        }
    };

    struct VCvtScalePkF16toFP8 : public VCvtInstruction
    {
        VCvtScalePkF16toFP8(const std::shared_ptr<RegisterContainer>& dst,
                            const std::shared_ptr<Container>&         src,
                            const InstructionInput&                   scale,
                            std::optional<SDWAModifiers>              sdwa    = std::nullopt,
                            std::optional<VOP3PModifiers>             vop3    = std::nullopt,
                            const std::string&                        comment = "")
            : VCvtInstruction(
                CvtType::CVT_SCALEF32_PK_FP8_F16, dst, {src, scale}, sdwa, vop3, comment)
        {
            setInst("v_cvt_scalef32_pk_fp8_f16");
        }

        VCvtScalePkF16toFP8(const VCvtScalePkF16toFP8& other)
            : VCvtInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCvtScalePkF16toFP8>(*this);
        }
    };

    struct VCvtScalePkF16toBF8 : public VCvtInstruction
    {
        VCvtScalePkF16toBF8(const std::shared_ptr<RegisterContainer>& dst,
                            const std::shared_ptr<Container>&         src,
                            const InstructionInput&                   scale,
                            std::optional<SDWAModifiers>              sdwa    = std::nullopt,
                            std::optional<VOP3PModifiers>             vop3    = std::nullopt,
                            const std::string&                        comment = "")
            : VCvtInstruction(
                CvtType::CVT_SCALEF32_PK_BF8_F16, dst, {src, scale}, sdwa, vop3, comment)
        {
            setInst("v_cvt_scalef32_pk_bf8_f16");
        }

        VCvtScalePkF16toBF8(const VCvtScalePkF16toBF8& other)
            : VCvtInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCvtScalePkF16toBF8>(*this);
        }
    };

    struct VCvtScaleSRF16toFP8 : public VCvtInstruction
    {
        VCvtScaleSRF16toFP8(const std::shared_ptr<RegisterContainer>& dst,
                            const std::shared_ptr<Container>&         src,
                            const InstructionInput&                   scale,
                            std::optional<SDWAModifiers>              sdwa    = std::nullopt,
                            std::optional<VOP3PModifiers>             vop3    = std::nullopt,
                            const std::string&                        comment = "")
            : VCvtInstruction(
                CvtType::CVT_SCALEF32_SR_FP8_F16, dst, {src, scale}, sdwa, vop3, comment)
        {
            setInst("v_cvt_scalef32_sr_fp8_f16");
        }

        VCvtScaleSRF16toFP8(const VCvtScaleSRF16toFP8& other)
            : VCvtInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCvtScaleSRF16toFP8>(*this);
        }
    };

    struct VCvtScaleSRF16toBF8 : public VCvtInstruction
    {
        VCvtScaleSRF16toBF8(const std::shared_ptr<RegisterContainer>& dst,
                            const std::shared_ptr<Container>&         src,
                            const InstructionInput&                   scale,
                            std::optional<SDWAModifiers>              sdwa    = std::nullopt,
                            std::optional<VOP3PModifiers>             vop3    = std::nullopt,
                            const std::string&                        comment = "")
            : VCvtInstruction(
                CvtType::CVT_SCALEF32_SR_BF8_F16, dst, {src, scale}, sdwa, vop3, comment)
        {
            setInst("v_cvt_scalef32_sr_bf8_f16");
        }

        VCvtScaleSRF16toBF8(const VCvtScaleSRF16toBF8& other)
            : VCvtInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCvtScaleSRF16toBF8>(*this);
        }
    };

    struct PVCvtBF16toFP32 : public VCvtInstruction
    {
        PVCvtBF16toFP32(const std::shared_ptr<RegisterContainer>& dst,
                        const std::shared_ptr<Container>&         src,
                        std::optional<SDWAModifiers>              sdwa    = std::nullopt,
                        const std::string&                        comment = "")
            : VCvtInstruction(CvtType::CVT_BF16_to_F32, dst, {src}, sdwa, std::nullopt, comment)
        {
            setInst("v_cvt_f32_bf16");
        }

        PVCvtBF16toFP32(const PVCvtBF16toFP32& other)
            : VCvtInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<PVCvtBF16toFP32>(*this);
        }
    };

    struct VCvtPkF32toBF16 : public VCvtInstruction
    {
        VCvtPkF32toBF16(const std::shared_ptr<RegisterContainer>& dst,
                        const std::shared_ptr<Container>&         src0,
                        const std::shared_ptr<Container>&         src1,
                        std::optional<SDWAModifiers>              sdwa    = std::nullopt,
                        std::optional<VOP3PModifiers>             vop3    = std::nullopt,
                        const std::string&                        comment = "")
            : VCvtInstruction(CvtType::CVT_PK_F32_to_BF16, dst, {src0, src1}, sdwa, vop3, comment)
        {
            setInst("v_cvt_pk_bf16_f32");
        }

        VCvtPkF32toBF16(const VCvtPkF32toBF16& other)
            : VCvtInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCvtPkF32toBF16>(*this);
        }
    };
} // namespace rocisa
