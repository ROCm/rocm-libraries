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
#include "enum.hpp"
#include "instruction/instruction.hpp"
#include <algorithm>
#include <cctype>

namespace rocisa
{
    DataType instTypeToDataType(InstType instType);

    bool is8bitFloat(DataType value);

    template <bool isSparse>
    auto getMFMAIssueLatency(DataType dataType, int matrixInstM, int matrixInstB)
    {
        auto numBytes       = dataTypeToBytes(dataType);
        int  mi_divisor     = 2;
        int  miIssueLatency = 2;
        auto isaVersion     = rocIsa::getInstance().getKernel().isaVersion;
        if((isaVersion == std::array<int, 3>{9, 4, 0} || isaVersion == std::array<int, 3>{9, 4, 1}
            || isaVersion == std::array<int, 3>{9, 4, 2}
            || isaVersion == std::array<int, 3>{9, 5, 0})
           && matrixInstB == 1)
        {
            if(dataType == DataType::Half || dataType == DataType::BFloat16
               || dataType == DataType::Int8 || is8bitFloat(dataType))
            {
                mi_divisor     = 4;
                miIssueLatency = 1;
            }
        }

        // need some way to distinguish between sparse and non-sparse
        // for F32Xdl we can use InstType::XFloat32
        if(isSparse || dataType == DataType::XFloat32)
        {
            mi_divisor = 4;
        }

        // special checking : F8 MFMA takes 2x more cycles and computes 4xK in gfx950
        if(isaVersion == std::array<int, 3>{9, 5, 0} && is8bitFloat(dataType))
        {
            mi_divisor = 2;
        }
        return std::make_pair(matrixInstM / mi_divisor, miIssueLatency);
    }

    struct MFMAInstruction : public Instruction
    {
        InstType                           accType;
        std::vector<int>                   variant;
        bool                               mfma1k;
        std::shared_ptr<RegisterContainer> acc;
        std::shared_ptr<RegisterContainer> a;
        std::shared_ptr<RegisterContainer> b;
        std::shared_ptr<RegisterContainer> acc2;
        bool                               neg;

        MFMAInstruction(InstType                                  instType,
                        InstType                                  accType,
                        const std::vector<int>&                   variant,
                        bool                                      mfma1k,
                        const std::shared_ptr<RegisterContainer>& acc,
                        const std::shared_ptr<RegisterContainer>& a,
                        const std::shared_ptr<RegisterContainer>& b,
                        const std::shared_ptr<RegisterContainer>& acc2    = nullptr,
                        bool                                      neg     = false,
                        const std::string&                        comment = "")
            : Instruction(instType, comment)
            , accType(accType)
            , variant(variant)
            , mfma1k(mfma1k)
            , acc(acc)
            , a(a)
            , b(b)
            , acc2(acc2 ? acc2 : acc)
            , neg(neg)
        {
        }

        MFMAInstruction(const MFMAInstruction& other)
            : Instruction(other.instType, other.comment)
            , accType(other.accType)
            , variant(other.variant)
            , mfma1k(other.mfma1k)
            , acc(other.acc ? other.acc->clone2() : nullptr)
            , a(other.a ? other.a->clone2() : nullptr)
            , b(other.b ? other.b->clone2() : nullptr)
            , acc2(other.acc2 ? other.acc2->clone2() : nullptr)
            , neg(other.neg)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<MFMAInstruction>(*this);
        }

        std::string typeConvert(InstType iType) const
        {
            // gfx950 (MI350) uses v_mfma_scale_xxx_f8f6f4 for MX types
            size_t f8f6f4_k = 64;  // K threshold for f8f6f4 instruction on gfx950

            switch(iType)
            {
            case InstType::INST_F16:
                return "f16";
            case InstType::INST_F32:
                return "f32";
            case InstType::INST_F64:
                return "f64";
            case InstType::INST_BF16:
                return "bf16";
            case InstType::INST_I8:
                return "i8";
            case InstType::INST_U8:
                return "iu8";
            case InstType::INST_I32:
                return "i32";
            case InstType::INST_XF32:
                return "xf32";
            case InstType::INST_F8:
                // gfx950: K>=64 uses f8f6f4 format, K<64 (16,32) uses legacy format
                return (variant[2] >= f8f6f4_k) ? "f8f6f4" : "fp8_fp8";
            case InstType::INST_BF8:
                return (variant[2] >= f8f6f4_k) ? "f8f6f4" : "bf8_bf8";
            case InstType::INST_F8_BF8:
                return (variant[2] >= f8f6f4_k) ? "f8f6f4" : "fp8_bf8";
            case InstType::INST_BF8_F8:
                return (variant[2] >= f8f6f4_k) ? "f8f6f4" : "bf8_fp8";
            // gfx950 (MI350) MX F6/F4 types
            case InstType::INST_F6:
            case InstType::INST_BF6:
            case InstType::INST_F6_B6:
            case InstType::INST_B6_F6:
                return "f8f6f4";
            case InstType::INST_F4:
                return "f8f6f4";
            case InstType::INST_F8_F4:
            case InstType::INST_F4_F8:
            case InstType::INST_F6_F4:
            case InstType::INST_F4_F6:
            case InstType::INST_F8_F6:
            case InstType::INST_F6_F8:
            case InstType::INST_F8_B6:
            case InstType::INST_B6_F8:
            case InstType::INST_B8_F4:
            case InstType::INST_F4_B8:
            case InstType::INST_B6_F4:
            case InstType::INST_F4_B6:
            case InstType::INST_B8_F6:
            case InstType::INST_F6_B8:
            case InstType::INST_B8_B6:
            case InstType::INST_B6_B8:
                return "f8f6f4";
            default:
                throw std::runtime_error("Type not found");
            }
        }

        std::vector<InstructionInput> getParams() const override
        {
            std::string negStr
                = !neg ? "" : (getAsmCaps()["HasWMMA_V1"] ? " neg_lo:[1,1,1]" : " neg_lo:[1,1]");
            return {acc, a, b, acc2, negStr};
        }

        std::string preStr() const override
        {
            std::string variantStr = std::to_string(variant[0]) + "x" + std::to_string(variant[1])
                                     + "x" + std::to_string(variant[2]);
            if(getAsmCaps()["HasMFMA_explictB"] && !mfma1k)
            {
                std::string strB = variant[3] > 1 ? std::to_string(variant[3]) + "b_" : "";
                return "v_mfma_" + typeConvert(accType) + "_" + variantStr + "_" + strB
                       + typeConvert(instType);
            }
            else
            {
                bool        is_mfma         = getAsmCaps()["HasMFMA"];
                std::string instructionName = is_mfma ? "mfma" : "wmma";
                std::string instructionStep = is_mfma ? "" : "_";
                std::string mfma_1k         = mfma1k ? "_1k" : "";
                return "v_" + instructionName + "_" + typeConvert(accType) + "_" + variantStr
                       + instructionStep + typeConvert(instType) + mfma_1k;
            }
        }

        std::string getArgStr() const
        {
            std::string negStr
                = !neg ? "" : (getAsmCaps()["HasWMMA_V1"] ? " neg_lo:[1,1,1]" : " neg_lo:[1,1]");
            std::string inputPermuteStr = "";
            if(getAsmCaps()["HasMFMA_f8f6f4"])
            {
                // gfx950 cbsz/blgp values for v_mfma_f32_xxx_f8f6f4 instruction
                // cbsz (A matrix format): 0=FP8, 1=BF8, 2=FP6, 3=BF6, 4=FP4
                // blgp (B matrix format): 0=FP8, 1=BF8, 2=FP6, 3=BF6, 4=FP4
                switch(instType)
                {
                case InstType::INST_F8:
                    inputPermuteStr = variant[2] > 32 ? " cbsz:0 blgp:0" : "";
                    break;
                case InstType::INST_BF8:
                    inputPermuteStr = variant[2] > 32 ? " cbsz:1 blgp:1" : "";
                    break;
                case InstType::INST_F8_BF8:
                    inputPermuteStr = variant[2] > 32 ? " cbsz:0 blgp:1" : "";
                    break;
                case InstType::INST_BF8_F8:
                    inputPermuteStr = variant[2] > 32 ? " cbsz:1 blgp:0" : "";
                    break;
                // gfx950 (MI350) FP6/BF6 types
                case InstType::INST_F6:
                    inputPermuteStr = variant[2] > 32 ? " cbsz:2 blgp:2" : "";
                    break;
                case InstType::INST_BF6:
                    inputPermuteStr = variant[2] > 32 ? " cbsz:3 blgp:3" : "";
                    break;
                case InstType::INST_F6_B6:
                    inputPermuteStr = variant[2] > 32 ? " cbsz:2 blgp:3" : "";
                    break;
                case InstType::INST_B6_F6:
                    inputPermuteStr = variant[2] > 32 ? " cbsz:3 blgp:2" : "";
                    break;
                // gfx950 (MI350) FP4 type
                case InstType::INST_F4:
                    inputPermuteStr = variant[2] > 32 ? " cbsz:4 blgp:4" : "";
                    break;
                // gfx950 (MI350) Mixed FP8/FP6/FP4 types
                case InstType::INST_F8_F6:
                    inputPermuteStr = variant[2] > 32 ? " cbsz:0 blgp:2" : "";
                    break;
                case InstType::INST_F6_F8:
                    inputPermuteStr = variant[2] > 32 ? " cbsz:2 blgp:0" : "";
                    break;
                case InstType::INST_F8_B6:
                    inputPermuteStr = variant[2] > 32 ? " cbsz:0 blgp:3" : "";
                    break;
                case InstType::INST_B6_F8:
                    inputPermuteStr = variant[2] > 32 ? " cbsz:3 blgp:0" : "";
                    break;
                case InstType::INST_B8_F6:
                    inputPermuteStr = variant[2] > 32 ? " cbsz:1 blgp:2" : "";
                    break;
                case InstType::INST_F6_B8:
                    inputPermuteStr = variant[2] > 32 ? " cbsz:2 blgp:1" : "";
                    break;
                case InstType::INST_B8_B6:
                    inputPermuteStr = variant[2] > 32 ? " cbsz:1 blgp:3" : "";
                    break;
                case InstType::INST_B6_B8:
                    inputPermuteStr = variant[2] > 32 ? " cbsz:3 blgp:1" : "";
                    break;
                case InstType::INST_F8_F4:
                    inputPermuteStr = variant[2] > 32 ? " cbsz:0 blgp:4" : "";
                    break;
                case InstType::INST_F4_F8:
                    inputPermuteStr = variant[2] > 32 ? " cbsz:4 blgp:0" : "";
                    break;
                case InstType::INST_B8_F4:
                    inputPermuteStr = variant[2] > 32 ? " cbsz:1 blgp:4" : "";
                    break;
                case InstType::INST_F4_B8:
                    inputPermuteStr = variant[2] > 32 ? " cbsz:4 blgp:1" : "";
                    break;
                case InstType::INST_F6_F4:
                    inputPermuteStr = variant[2] > 32 ? " cbsz:2 blgp:4" : "";
                    break;
                case InstType::INST_F4_F6:
                    inputPermuteStr = variant[2] > 32 ? " cbsz:4 blgp:2" : "";
                    break;
                case InstType::INST_B6_F4:
                    inputPermuteStr = variant[2] > 32 ? " cbsz:3 blgp:4" : "";
                    break;
                case InstType::INST_F4_B6:
                    inputPermuteStr = variant[2] > 32 ? " cbsz:4 blgp:3" : "";
                    break;
                default:
                    break;
                }
            }
            return acc->toString() + ", " + a->toString() + ", " + b->toString() + ", "
                   + acc2->toString() + negStr + inputPermuteStr;
        }

        std::string toString() const override
        {
            auto        newInstStr = preStr();
            std::string kStr       = newInstStr + " " + getArgStr();
            return formatWithComment(kStr);
        }

        int getIssueLatency() const override
        {
            auto dataType = instTypeToDataType(instType);
            auto [issueLatency, miLatency]
                = getMFMAIssueLatency<false>(dataType, variant[0], variant[3]);
            return issueLatency;
        }
    };

    struct SMFMAInstruction : public Instruction
    {
        InstType                           accType;
        std::vector<int>                   variant;
        bool                               mfma1k;
        std::shared_ptr<RegisterContainer> acc;
        std::shared_ptr<RegisterContainer> a;
        std::shared_ptr<RegisterContainer> b;
        std::shared_ptr<RegisterContainer> metadata;

        SMFMAInstruction(InstType                                  instType,
                         InstType                                  accType,
                         const std::vector<int>&                   variant,
                         bool                                      mfma1k,
                         const std::shared_ptr<RegisterContainer>& acc,
                         const std::shared_ptr<RegisterContainer>& a,
                         const std::shared_ptr<RegisterContainer>& b,
                         const std::shared_ptr<RegisterContainer>& metadata,
                         const std::string&                        comment = "")
            : Instruction(instType, comment)
            , accType(accType)
            , variant(variant)
            , mfma1k(mfma1k)
            , acc(acc)
            , a(a)
            , b(b)
            , metadata(metadata)
        {
        }

        SMFMAInstruction(const SMFMAInstruction& other)
            : Instruction(other.instType, other.comment)
            , accType(other.accType)
            , variant(other.variant)
            , mfma1k(other.mfma1k)
            , acc(other.acc ? other.acc->clone2() : nullptr)
            , a(other.a ? other.a->clone2() : nullptr)
            , b(other.b ? other.b->clone2() : nullptr)
            , metadata(other.metadata ? other.metadata->clone2() : nullptr)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SMFMAInstruction>(*this);
        }

        std::string typeConvert(InstType iType) const
        {
            switch(iType)
            {
            case InstType::INST_F16:
                return "f16";
            case InstType::INST_F32:
                return "f32";
            case InstType::INST_BF16:
                return "bf16";
            case InstType::INST_I8:
                return "i8";
            case InstType::INST_I32:
                return "i32";
            case InstType::INST_F8:
                return "fp8_fp8";
            case InstType::INST_BF8:
                return "bf8_bf8";
            case InstType::INST_F8_BF8:
                return "fp8_bf8";
            case InstType::INST_BF8_F8:
                return "bf8_fp8";
            default:
                throw std::runtime_error("Type not found");
            }
        }

        std::vector<InstructionInput> getParams() const override
        {
            return {acc, a, b, metadata};
        }

        std::string preStr() const override
        {
            if(variant.size() == 4)
            {
                std::string variantStr = std::to_string(variant[0]) + "x"
                                         + std::to_string(variant[1]) + "x"
                                         + std::to_string(variant[2]);
                std::string strB = variant[3] > 1 ? std::to_string(variant[3]) + "ub_" : "";
                return "v_smfmac_" + typeConvert(accType) + "_" + variantStr + "_" + strB
                       + typeConvert(instType);
            }
            else
            {
                throw std::runtime_error("Currently only support smfma variant 4");
            }
        }

        std::string getArgStr() const
        {
            return acc->toString() + ", " + a->toString() + ", " + b->toString() + ", "
                   + metadata->toString();
        }

        std::string toString() const override
        {
            auto        newInstStr = preStr();
            std::string kStr       = newInstStr + " " + getArgStr();
            return formatWithComment(kStr);
        }

        int getIssueLatency() const override
        {
            auto dataType = instTypeToDataType(instType);
            auto [issueLatency, miLatency]
                = getMFMAIssueLatency<true>(dataType, variant[0], variant[3]);
            return issueLatency;
        }
    };

    // MX Scale MFMA Instruction for gfx950 (MI350)
    // Implements v_mfma_scale_f32_MxNxK_f8f6f4 instruction for Microscaling
    // Supported variants: 16x16x128, 32x32x64
    //
    // Key differences from MI450 (gfx1250):
    // - MI350 Scale Format: E8M0 only (MI450 has E8M0, E5M3, E4M3)
    // - MI350 MX Block Size: 32 only (MI450 uses 16) - one scale per 32 K elements
    // - MI350 uses cbsz/blgp modifiers (MI450 uses matrix_a_fmt/matrix_b_fmt)
    // - MI350 does NOT support matrix_a_reuse/matrix_b_reuse (MI450 does)
    //
    // Control Behavior:
    //   ABID[0] = 1'b1 : Must be set for V_MFMA_SCALE instructions (enables scale)
    //   ABID[0] = 1'b0 : Forces all scales to 1.0f (MFMA runs without scale source)
    //
    // Hardware calculation: d_exp = (a0_exp+b0_exp) + (a1_exp+b1_exp) + ... + c_exp + scale_a + scale_b
    //
    // Register requirements:
    //   V_MFMA_SCALE_F32_16x16x128_F8F6F4:
    //     A/B: F8=8 VGPRs, F6=6 VGPRs, F4=4 VGPRs
    //     Acc C/D (16x16 F32): 4 VGPRs
    //     scaleA, scaleB: 1 VGPR each
    //   V_MFMA_SCALE_F32_32x32x64_F8F6F4:
    //     A/B: F8=8 VGPRs, F6=6 VGPRs, F4=4 VGPRs
    //     Acc C/D (32x32 F32): 16 VGPRs
    //     scaleA, scaleB: 1 VGPR each
    //
    // Issue latency (cycles):
    //   V_MFMA_SCALE_F32_16x16x128_F8F6F4: F8=32 cycles, F6/F4=16 cycles
    //   V_MFMA_SCALE_F32_32x32x64_F8F6F4:  F8=64 cycles, F6/F4=32 cycles
    //   (If either A or B matrix is F8, use F8 latency)
    struct MXScaleMFMAInstruction : public Instruction
    {
        InstType                           accType;
        std::vector<int>                   variant;
        std::shared_ptr<RegisterContainer> acc;
        std::shared_ptr<RegisterContainer> a;
        std::shared_ptr<RegisterContainer> b;
        std::shared_ptr<RegisterContainer> acc2;
        std::shared_ptr<RegisterContainer> scaleA;  // MX scale for A (E8M0 format on MI350)
        std::shared_ptr<RegisterContainer> scaleB;  // MX scale for B (E8M0 format on MI350)
        InstType                           instTypeA;  // Data type for A matrix
        InstType                           instTypeB;  // Data type for B matrix
        bool                               enableScale;  // ABID[0]: true=enable scale, false=force 1.0f

        // Helper function to get cbsz/blgp value from InstType
        // gfx950 cbsz/blgp encoding: 0=FP8, 1=BF8, 2=FP6, 3=BF6, 4=FP4
        static int getDataFormatCode(InstType type)
        {
            switch(type)
            {
            case InstType::INST_F8:
                return 0;  // FP8
            case InstType::INST_BF8:
                return 1;  // BF8
            case InstType::INST_F6:
                return 2;  // FP6
            case InstType::INST_BF6:
                return 3;  // BF6
            case InstType::INST_F4:
                return 4;  // FP4
            default:
                return 0;  // Default to FP8
            }
        }

        MXScaleMFMAInstruction(InstType                                  instType,
                               InstType                                  accType,
                               const std::vector<int>&                   variant,
                               const std::shared_ptr<RegisterContainer>& acc,
                               const std::shared_ptr<RegisterContainer>& a,
                               const std::shared_ptr<RegisterContainer>& b,
                               const std::shared_ptr<RegisterContainer>& acc2,
                               const std::shared_ptr<RegisterContainer>& scaleA,
                               const std::shared_ptr<RegisterContainer>& scaleB,
                               InstType                                  instTypeA     = InstType::INST_F8,
                               InstType                                  instTypeB     = InstType::INST_F8,
                               bool                                      enableScale   = true,
                               const std::string&                        comment       = "")
            : Instruction(instType, comment)
            , accType(accType)
            , variant(variant)
            , acc(acc)
            , a(a)
            , b(b)
            , acc2(acc2 ? acc2 : acc)
            , scaleA(scaleA)
            , scaleB(scaleB)
            , instTypeA(instTypeA)
            , instTypeB(instTypeB)
            , enableScale(enableScale)
        {
        }

        MXScaleMFMAInstruction(const MXScaleMFMAInstruction& other)
            : Instruction(other.instType, other.comment)
            , accType(other.accType)
            , variant(other.variant)
            , acc(other.acc ? other.acc->clone2() : nullptr)
            , a(other.a ? other.a->clone2() : nullptr)
            , b(other.b ? other.b->clone2() : nullptr)
            , acc2(other.acc2 ? other.acc2->clone2() : nullptr)
            , scaleA(other.scaleA ? other.scaleA->clone2() : nullptr)
            , scaleB(other.scaleB ? other.scaleB->clone2() : nullptr)
            , instTypeA(other.instTypeA)
            , instTypeB(other.instTypeB)
            , enableScale(other.enableScale)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<MXScaleMFMAInstruction>(*this);
        }

        std::vector<InstructionInput> getParams() const override
        {
            if (enableScale) {
                return {acc, a, b, acc2, scaleA, scaleB};
            } else {
                return {acc, a, b, acc2};
            }
        }

        std::string preStr() const override
        {
            // gfx950 (MI350) supports both:
            // - v_mfma_scale_f32_MxNxK_f8f6f4 (with scale operands)
            // - v_mfma_f32_MxNxK_f8f6f4 (without scale operands, uses cbsz/blgp for format)
            std::string variantStr = std::to_string(variant[0]) + "x"
                                     + std::to_string(variant[1]) + "x"
                                     + std::to_string(variant[2]);
            if (enableScale) {
                return "v_mfma_scale_f32_" + variantStr + "_f8f6f4";
            } else {
                return "v_mfma_f32_" + variantStr + "_f8f6f4";
            }
        }

        // Helper function to get the number of VGPRs required for input matrix
        // based on data type and matrix dimensions
        // For gfx950 v_mfma_scale_f32_MxNxK_f8f6f4:
        // Each thread in wave64 processes: (M * K) / 64 elements for A, (K * N) / 64 for B
        // VGPR calculation for MX MFMA instructions on gfx950:
        //
        // CBSZ/BLGP encoding and VGPR per element:
        //   0 = FP8 (E4M3): 8 bits, 0.25 VGPR/element
        //   1 = BF8 (E5M2): 8 bits, 0.25 VGPR/element
        //   2 = FP6 (E2M3): 6 bits, 0.1875 VGPR/element
        //   3 = BF6 (E3M2): 6 bits, 0.1875 VGPR/element
        //   4 = FP4 (E2M1): 4 bits, 0.125 VGPR/element
        //
        // MIInputPerThread = 32 elements for both 16x16x128 and 32x32x64
        //
        // Total VGPRs = MIInputPerThread * VGPR_per_element:
        //   FP8/BF8: 32 * 0.25   = 8 VGPRs
        //   FP6/BF6: 32 * 0.1875 = 6 VGPRs
        //   FP4:     32 * 0.125  = 4 VGPRs
        //
        int getNumInputVgprs(InstType type) const
        {
            // MIInputPerThread = 32 elements for both instruction variants
            constexpr int miInputPerThread = 32;

            switch(type)
            {
            case InstType::INST_F8:
            case InstType::INST_BF8:
                // 32 * 0.25 = 8 VGPRs
                return miInputPerThread / 4;
            case InstType::INST_F6:
            case InstType::INST_BF6:
                // 32 * 0.1875 = 6 VGPRs (32 * 6 / 32)
                return (miInputPerThread * 6) / 32;
            case InstType::INST_F4:
                // 32 * 0.125 = 4 VGPRs
                return miInputPerThread / 8;
            default:
                return miInputPerThread / 4; // Default to F8
            }
        }

        // Helper to format register range with correct count
        // Input: original register like "v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7]"
        // Output: adjusted to use numVgprs, e.g., "v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+5]" for 6 VGPRs
        std::string formatRegWithCount(const std::shared_ptr<RegisterContainer>& reg, int numVgprs) const
        {
            std::string regStr = reg->toString();

            // Find the colon that separates start:end in range
            size_t colonPos = regStr.find(':');
            size_t bracketEnd = regStr.rfind(']');

            if (colonPos == std::string::npos || bracketEnd == std::string::npos)
            {
                // Not a range register, return as-is
                return regStr;
            }

            // Extract parts: "v[start" and "end]"
            std::string startPart = regStr.substr(0, colonPos);  // "v[vgprValuA_X0_I0+0+0+0"
            std::string endPartWithBracket = regStr.substr(colonPos + 1);  // "vgprValuA_X0_I0+0+0+0+7]"

            // Find the last '+' followed by a number in the end part (this is the range offset)
            size_t lastPlus = endPartWithBracket.rfind('+');
            if (lastPlus == std::string::npos)
            {
                // No '+' in end part, might be simple like "v[0:7]"
                // Just replace the number before ']'
                size_t numStart = 0;
                size_t numEnd = endPartWithBracket.find(']');
                if (numEnd != std::string::npos)
                {
                    return startPart + ":" + std::to_string(numVgprs - 1) + "]";
                }
                return regStr;
            }

            // Check if what follows '+' is a number
            std::string afterPlus = endPartWithBracket.substr(lastPlus + 1);
            size_t bracketInEnd = afterPlus.find(']');
            if (bracketInEnd != std::string::npos)
            {
                std::string numStr = afterPlus.substr(0, bracketInEnd);
                // Verify it's a number
                bool isNumber = !numStr.empty() && std::all_of(numStr.begin(), numStr.end(), ::isdigit);
                if (isNumber)
                {
                    // Replace the old range value with new one
                    std::string basePart = endPartWithBracket.substr(0, lastPlus + 1);  // "vgprValuA_X0_I0+0+0+0+"
                    return startPart + ":" + basePart + std::to_string(numVgprs - 1) + "]";
                }
            }

            // Fallback: return original
            return regStr;
        }

        std::string getArgStr() const
        {
            // gfx950 (MI350) instruction formats:
            //
            // With scale (v_mfma_scale_f32_MxNxK_f8f6f4):
            //   acc, a, b, acc2, scaleA, scaleB cbsz:X blgp:Y
            //
            // Without scale (v_mfma_f32_MxNxK_f8f6f4):
            //   acc, a, b, acc2 cbsz:X blgp:Y
            //
            // cbsz: A matrix format (0=FP8, 1=BF8, 2=FP6, 3=BF6, 4=FP4)
            // blgp: B matrix format (0=FP8, 1=BF8, 2=FP6, 3=BF6, 4=FP4)
            //
            // Input register requirements (same for both 16x16x128 and 32x32x64):
            //   F8/BF8: 8 VGPRs
            //   F6/BF6: 6 VGPRs
            //   F4:     4 VGPRs
            int cbsz = getDataFormatCode(instTypeA);
            int blgp = getDataFormatCode(instTypeB);

            // Get correct VGPR counts for A and B matrices
            int numVgprsA = getNumInputVgprs(instTypeA);
            int numVgprsB = getNumInputVgprs(instTypeB);

            // Format input registers with correct counts
            std::string aStr = formatRegWithCount(a, numVgprsA);
            std::string bStr = formatRegWithCount(b, numVgprsB);

            std::string result;
            if (enableScale) {
                // With scale: acc, a, b, acc2, scaleA, scaleB
                result = acc->toString() + ", " + aStr + ", " + bStr
                         + ", " + acc2->toString() + ", " + scaleA->toString() + ", "
                         + scaleB->toString();
            } else {
                // Without scale: acc, a, b, acc2
                result = acc->toString() + ", " + aStr + ", " + bStr
                         + ", " + acc2->toString();
            }
            result += " cbsz:" + std::to_string(cbsz);
            result += " blgp:" + std::to_string(blgp);
            return result;
        }

        std::string toString() const override
        {
            auto        newInstStr = preStr();
            std::string kStr       = newInstStr + " " + getArgStr();
            return formatWithComment(kStr);
        }

        int getIssueLatency() const override
        {
            // MI350 issue latency for V_MFMA_SCALE_F32_xxx_F8F6F4:
            //   16x16x128: F8=32 cycles, F6/F4=16 cycles
            //   32x32x64:  F8=64 cycles, F6/F4=32 cycles
            // If either A or B matrix is F8/BF8, use F8 latency
            bool isF8A = (instTypeA == InstType::INST_F8 || instTypeA == InstType::INST_BF8);
            bool isF8B = (instTypeB == InstType::INST_F8 || instTypeB == InstType::INST_BF8);
            bool hasF8 = isF8A || isF8B;

            if (variant[0] == 16) {
                // 16x16x128 variant
                return hasF8 ? 32 : 16;
            } else {
                // 32x32x64 variant
                return hasF8 ? 64 : 32;
            }
        }
    };
} // namespace rocisa
