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

        if(isaVersion == std::array<int, 3>{9, 5, 0} && numBytes == 2)
        {
            mi_divisor     = 2;
            miIssueLatency = 2;
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
                return variant[2] > 32 ? "f8f6f4" : "fp8_fp8";
            case InstType::INST_BF8:
                return variant[2] > 32 ? "f8f6f4" : "bf8_bf8";
            case InstType::INST_F8_BF8:
                return variant[2] > 32 ? "f8f6f4" : "fp8_bf8";
            case InstType::INST_BF8_F8:
                return variant[2] > 32 ? "f8f6f4" : "bf8_fp8";
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
} // namespace rocisa
