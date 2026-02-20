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

#include <optional>

namespace rocisa
{
    struct VCmpInstruction : public CommonInstruction
    {
        VCmpInstruction(const InstType                    instType,
                        const std::shared_ptr<Container>& dst,
                        const InstructionInput&           src0,
                        const InstructionInput&           src1,
                        std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                        const std::string                 comment = "")
            : CommonInstruction(
                instType, dst, {src0, src1}, std::nullopt, sdwa, std::nullopt, comment)
        {
        }

        VCmpInstruction(const VCmpInstruction& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const
        {
            throw std::runtime_error("You should override clone (VCmp) function in derived class");
            return std::make_shared<Item>("");
        }
    };

    struct VCmpXInstruction : public CommonInstruction
    {
        VCmpXInstruction(InstType                          instType,
                         const std::shared_ptr<Container>& dst,
                         const InstructionInput&           src0,
                         const InstructionInput&           src1,
                         std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                         std::string                       comment = "")
            : CommonInstruction(
                instType, dst, {src0, src1}, std::nullopt, sdwa, std::nullopt, comment)
        {
        }

        VCmpXInstruction(const VCmpXInstruction& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const
        {
            throw std::runtime_error("You should override clone (VCmpX) function in derived class");
            return std::make_shared<Item>("");
        }

        std::string getArgStr(const std::string& dstStr = "") const
        {
            std::string kStr;
            if(!dstStr.empty())
            {
                kStr += dstStr;
            }
            else if(dst)
            {
                kStr += dst->toString();
            }
            if(dst1)
            {
                if(!kStr.empty())
                {
                    kStr += ", ";
                }
                kStr += dst1->toString();
            }
            if(!srcs.empty())
            {
                if(!kStr.empty())
                {
                    kStr += ", ";
                }
                kStr += InstructionInputToString(srcs[0]);
            }
            for(size_t i = 1; i < srcs.size(); ++i)
            {
                kStr += ", " + InstructionInputToString(srcs[i]);
            }
            return kStr;
        }

        std::string toString() const
        {
            auto        newInstStr = preStr();
            std::string kStr;
            if(getArchCaps()["CMPXWritesSGPR"])
            {
                kStr = newInstStr + " " + getArgStr();
                if(sdwa)
                {
                    kStr += sdwa->toString();
                }
                if(vop3)
                {
                    kStr += vop3->toString();
                }
                kStr = formatWithComment(kStr);
            }
            else
            {
                kStr = newInstStr;
                try
                {
                    kStr = kStr.replace(kStr.find("_cmpx_"), 6, "_cmp_");
                }
                catch(const std::out_of_range& e)
                {
                    std::cerr << "Out of range error: " << e.what() << " " << kStr << std::endl;
                    throw std::runtime_error("Out of range error");
                }
                std::string dstStr;
                if(dynamic_cast<EXEC*>(dst.get()))
                {
                    VCC vcc;
                    dstStr = vcc.toString();
                }
                else
                {
                    dstStr = dst->toString();
                }
                kStr += " " + getArgStr(dstStr);
                if(sdwa)
                {
                    kStr += sdwa->toString();
                }
                kStr = formatWithComment(kStr);
                std::string kStr2;
                if(kernel().wavefront == 64)
                {
                    kStr2 = "s_mov_b64 exec " + dstStr;
                }
                else
                {
                    kStr2 = "s_mov_b32 exec_lo " + dstStr;
                }
                kStr2 = formatWithComment(kStr2);
                kStr += kStr2;
            }
            return kStr;
        }
    };

    struct SCmpEQI32 : public CommonInstruction
    {
        SCmpEQI32(const InstructionInput& src0,
                  const InstructionInput& src1,
                  const std::string&      comment = "")
            : CommonInstruction(InstType::INST_I32,
                                nullptr,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_cmp_eq_i32");
        }

        SCmpEQI32(const SCmpEQI32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const
        {
            return std::make_shared<SCmpEQI32>(*this);
        }
    };

    struct SCmpEQU32 : public CommonInstruction
    {
        SCmpEQU32(const InstructionInput& src0,
                  const InstructionInput& src1,
                  const std::string&      comment = "")
            : CommonInstruction(InstType::INST_U32,
                                nullptr,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_cmp_eq_u32");
        }

        SCmpEQU32(const SCmpEQU32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const
        {
            return std::make_shared<SCmpEQU32>(*this);
        }
    };

    struct SCmpEQU64 : public CommonInstruction
    {
        SCmpEQU64(const InstructionInput& src0,
                  const InstructionInput& src1,
                  const std::string&      comment = "")
            : CommonInstruction(InstType::INST_U64,
                                nullptr,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_cmp_eq_u64");
        }

        SCmpEQU64(const SCmpEQU64& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const
        {
            return std::make_shared<SCmpEQU64>(*this);
        }
    };

    struct SCmpGeI32 : public CommonInstruction
    {
        SCmpGeI32(const InstructionInput& src0,
                  const InstructionInput& src1,
                  const std::string&      comment = "")
            : CommonInstruction(InstType::INST_I32,
                                nullptr,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_cmp_ge_i32");
        }

        SCmpGeI32(const SCmpGeI32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const
        {
            return std::make_shared<SCmpGeI32>(*this);
        }
    };

    struct SCmpGeU32 : public CommonInstruction
    {
        SCmpGeU32(const InstructionInput& src0,
                  const InstructionInput& src1,
                  const std::string&      comment = "")
            : CommonInstruction(InstType::INST_U32,
                                nullptr,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_cmp_ge_u32");
        }

        SCmpGeU32(const SCmpGeU32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const
        {
            return std::make_shared<SCmpGeU32>(*this);
        }
    };

    struct SCmpGtI32 : public CommonInstruction
    {
        SCmpGtI32(const InstructionInput& src0,
                  const InstructionInput& src1,
                  const std::string&      comment = "")
            : CommonInstruction(InstType::INST_I32,
                                nullptr,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_cmp_gt_i32");
        }

        SCmpGtI32(const SCmpGtI32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const
        {
            return std::make_shared<SCmpGtI32>(*this);
        }
    };

    struct SCmpGtU32 : public CommonInstruction
    {
        SCmpGtU32(const InstructionInput& src0,
                  const InstructionInput& src1,
                  const std::string&      comment = "")
            : CommonInstruction(InstType::INST_U32,
                                nullptr,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_cmp_gt_u32");
        }

        SCmpGtU32(const SCmpGtU32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const
        {
            return std::make_shared<SCmpGtU32>(*this);
        }
    };

    struct SCmpLeI32 : public CommonInstruction
    {
        SCmpLeI32(const InstructionInput& src0,
                  const InstructionInput& src1,
                  const std::string&      comment = "")
            : CommonInstruction(InstType::INST_I32,
                                nullptr,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_cmp_le_i32");
        }

        SCmpLeI32(const SCmpLeI32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const
        {
            return std::make_shared<SCmpLeI32>(*this);
        }
    };

    struct SCmpLeU32 : public CommonInstruction
    {
        SCmpLeU32(const InstructionInput& src0,
                  const InstructionInput& src1,
                  const std::string&      comment = "")
            : CommonInstruction(InstType::INST_U32,
                                nullptr,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_cmp_le_u32");
        }

        SCmpLeU32(const SCmpLeU32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const
        {
            return std::make_shared<SCmpLeU32>(*this);
        }
    };

    struct SCmpLgU32 : public CommonInstruction
    {
        SCmpLgU32(const InstructionInput& src0,
                  const InstructionInput& src1,
                  const std::string&      comment = "")
            : CommonInstruction(InstType::INST_U32,
                                nullptr,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_cmp_lg_u32");
        }

        SCmpLgU32(const SCmpLgU32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const
        {
            return std::make_shared<SCmpLgU32>(*this);
        }
    };

    struct SCmpLgI32 : public CommonInstruction
    {
        SCmpLgI32(const InstructionInput& src0,
                  const InstructionInput& src1,
                  const std::string&      comment = "")
            : CommonInstruction(InstType::INST_I32,
                                nullptr,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_cmp_lg_i32");
        }

        SCmpLgI32(const SCmpLgI32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const
        {
            return std::make_shared<SCmpLgI32>(*this);
        }
    };

    struct SCmpLgU64 : public CommonInstruction
    {
        SCmpLgU64(const InstructionInput& src0,
                  const InstructionInput& src1,
                  const std::string&      comment = "")
            : CommonInstruction(InstType::INST_U64,
                                nullptr,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_cmp_lg_u64");
        }

        SCmpLgU64(const SCmpLgU64& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const
        {
            return std::make_shared<SCmpLgU64>(*this);
        }
    };

    struct SCmpLtI32 : public CommonInstruction
    {
        SCmpLtI32(const InstructionInput& src0,
                  const InstructionInput& src1,
                  const std::string&      comment = "")
            : CommonInstruction(InstType::INST_I32,
                                nullptr,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_cmp_lt_i32");
        }

        SCmpLtI32(const SCmpLtI32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const
        {
            return std::make_shared<SCmpLtI32>(*this);
        }
    };

    struct SCmpLtU32 : public CommonInstruction
    {
        SCmpLtU32(const InstructionInput& src0,
                  const InstructionInput& src1,
                  const std::string&      comment = "")
            : CommonInstruction(InstType::INST_U32,
                                nullptr,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_cmp_lt_u32");
        }

        SCmpLtU32(const SCmpLtU32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const
        {
            return std::make_shared<SCmpLtU32>(*this);
        }
    };

    struct SBitcmp1B32 : public CommonInstruction
    {
        SBitcmp1B32(const InstructionInput& src0,
                    const InstructionInput& src1,
                    const std::string&      comment = "")
            : CommonInstruction(InstType::INST_B32,
                                nullptr,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_bitcmp1_b32");
        }

        SBitcmp1B32(const SBitcmp1B32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const
        {
            return std::make_shared<SBitcmp1B32>(*this);
        }
    };

    struct SCmpKEQU32 : public CommonInstruction
    {
        SCmpKEQU32(const std::shared_ptr<Container>& src,
                   const int                         simm16,
                   const std::string&                comment = "")
            : CommonInstruction(InstType::INST_U32,
                                nullptr,
                                {src, simm16},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_cmpk_eq_u32");
        }

        SCmpKEQU32(const SCmpKEQU32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const
        {
            return std::make_shared<SCmpKEQU32>(*this);
        }
    };

    struct SCmpKGeU32 : public CommonInstruction
    {
        SCmpKGeU32(const std::shared_ptr<Container>& src,
                   const int                         simm16,
                   const std::string&                comment = "")
            : CommonInstruction(InstType::INST_U32,
                                nullptr,
                                {src, simm16},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_cmpk_ge_u32");
        }

        SCmpKGeU32(const SCmpKGeU32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const
        {
            return std::make_shared<SCmpKGeU32>(*this);
        }
    };

    struct SCmpKGtU32 : public CommonInstruction
    {
        SCmpKGtU32(const std::shared_ptr<Container>& src,
                   const int                         simm16,
                   const std::string&                comment = "")
            : CommonInstruction(InstType::INST_U32,
                                nullptr,
                                {src, simm16},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_cmpk_gt_u32");
        }

        SCmpKGtU32(const SCmpKGtU32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const
        {
            return std::make_shared<SCmpKGtU32>(*this);
        }
    };

    struct SCmpKLGU32 : public CommonInstruction
    {
        SCmpKLGU32(const std::shared_ptr<Container>& src,
                   const int                         simm16,
                   const std::string&                comment = "")
            : CommonInstruction(InstType::INST_U32,
                                nullptr,
                                {src, simm16},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_cmpk_lg_u32");
        }

        SCmpKLGU32(const SCmpKLGU32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const
        {
            return std::make_shared<SCmpKLGU32>(*this);
        }
    };

    struct VCmpEQF32 : public VCmpInstruction
    {
        VCmpEQF32(const std::shared_ptr<Container>& dst,
                  const InstructionInput&           src0,
                  const InstructionInput&           src1,
                  std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                  const std::string&                comment = "")
            : VCmpInstruction(InstType::INST_F32, dst, src0, src1, sdwa, comment)
        {
            setInst("v_cmp_eq_f32");
        }

        VCmpEQF32(const VCmpEQF32& other)
            : VCmpInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCmpEQF32>(*this);
        }
    };

    struct VCmpEQF64 : public VCmpInstruction
    {
        VCmpEQF64(const std::shared_ptr<Container>& dst,
                  const InstructionInput&           src0,
                  const InstructionInput&           src1,
                  std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                  const std::string&                comment = "")
            : VCmpInstruction(InstType::INST_F64, dst, src0, src1, sdwa, comment)
        {
            setInst("v_cmp_eq_f64");
        }

        VCmpEQF64(const VCmpEQF64& other)
            : VCmpInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCmpEQF64>(*this);
        }
    };

    struct VCmpEQU32 : public VCmpInstruction
    {
        VCmpEQU32(const std::shared_ptr<Container>& dst,
                  const InstructionInput&           src0,
                  const InstructionInput&           src1,
                  std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                  const std::string&                comment = "")
            : VCmpInstruction(InstType::INST_U32, dst, src0, src1, sdwa, comment)
        {
            setInst("v_cmp_eq_u32");
        }

        VCmpEQU32(const VCmpEQU32& other)
            : VCmpInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCmpEQU32>(*this);
        }
    };

    struct VCmpEQI32 : public VCmpInstruction
    {
        VCmpEQI32(const std::shared_ptr<Container>& dst,
                  const InstructionInput&           src0,
                  const InstructionInput&           src1,
                  std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                  const std::string&                comment = "")
            : VCmpInstruction(InstType::INST_I32, dst, src0, src1, sdwa, comment)
        {
            setInst("v_cmp_eq_i32");
        }

        VCmpEQI32(const VCmpEQI32& other)
            : VCmpInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCmpEQI32>(*this);
        }
    };

    struct VCmpGEF16 : public VCmpInstruction
    {
        VCmpGEF16(const std::shared_ptr<Container>& dst,
                  const InstructionInput&           src0,
                  const InstructionInput&           src1,
                  std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                  const std::string&                comment = "")
            : VCmpInstruction(InstType::INST_F16, dst, src0, src1, sdwa, comment)
        {
            setInst("v_cmp_ge_f16");
        }

        VCmpGEF16(const VCmpGEF16& other)
            : VCmpInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCmpGEF16>(*this);
        }
    };

    struct VCmpGTF16 : public VCmpInstruction
    {
        VCmpGTF16(const std::shared_ptr<Container>& dst,
                  const InstructionInput&           src0,
                  const InstructionInput&           src1,
                  std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                  const std::string&                comment = "")
            : VCmpInstruction(InstType::INST_F16, dst, src0, src1, sdwa, comment)
        {
            setInst("v_cmp_gt_f16");
        }

        VCmpGTF16(const VCmpGTF16& other)
            : VCmpInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCmpGTF16>(*this);
        }
    };

    struct VCmpGEF32 : public VCmpInstruction
    {
        VCmpGEF32(const std::shared_ptr<Container>& dst,
                  const InstructionInput&           src0,
                  const InstructionInput&           src1,
                  std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                  const std::string&                comment = "")
            : VCmpInstruction(InstType::INST_F32, dst, src0, src1, sdwa, comment)
        {
            setInst("v_cmp_ge_f32");
        }

        VCmpGEF32(const VCmpGEF32& other)
            : VCmpInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCmpGEF32>(*this);
        }
    };

    struct VCmpGTF32 : public VCmpInstruction
    {
        VCmpGTF32(const std::shared_ptr<Container>& dst,
                  const InstructionInput&           src0,
                  const InstructionInput&           src1,
                  std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                  const std::string&                comment = "")
            : VCmpInstruction(InstType::INST_F32, dst, src0, src1, sdwa, comment)
        {
            setInst("v_cmp_gt_f32");
        }

        VCmpGTF32(const VCmpGTF32& other)
            : VCmpInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCmpGTF32>(*this);
        }
    };

    struct VCmpGEF64 : public VCmpInstruction
    {
        VCmpGEF64(const std::shared_ptr<Container>& dst,
                  const InstructionInput&           src0,
                  const InstructionInput&           src1,
                  std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                  const std::string&                comment = "")
            : VCmpInstruction(InstType::INST_F64, dst, src0, src1, sdwa, comment)
        {
            setInst("v_cmp_ge_f64");
        }

        VCmpGEF64(const VCmpGEF64& other)
            : VCmpInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCmpGEF64>(*this);
        }
    };

    struct VCmpGTF64 : public VCmpInstruction
    {
        VCmpGTF64(const std::shared_ptr<Container>& dst,
                  const InstructionInput&           src0,
                  const InstructionInput&           src1,
                  std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                  const std::string&                comment = "")
            : VCmpInstruction(InstType::INST_F64, dst, src0, src1, sdwa, comment)
        {
            setInst("v_cmp_gt_f64");
        }

        VCmpGTF64(const VCmpGTF64& other)
            : VCmpInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCmpGTF64>(*this);
        }
    };

    struct VCmpGEI32 : public VCmpInstruction
    {
        VCmpGEI32(const std::shared_ptr<Container>& dst,
                  const InstructionInput&           src0,
                  const InstructionInput&           src1,
                  std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                  const std::string&                comment = "")
            : VCmpInstruction(InstType::INST_I32, dst, src0, src1, sdwa, comment)
        {
            setInst("v_cmp_ge_i32");
        }

        VCmpGEI32(const VCmpGEI32& other)
            : VCmpInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCmpGEI32>(*this);
        }
    };

    struct VCmpGTI32 : public VCmpInstruction
    {
        VCmpGTI32(const std::shared_ptr<Container>& dst,
                  const InstructionInput&           src0,
                  const InstructionInput&           src1,
                  std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                  const std::string&                comment = "")
            : VCmpInstruction(InstType::INST_I32, dst, src0, src1, sdwa, comment)
        {
            setInst("v_cmp_gt_i32");
        }

        VCmpGTI32(const VCmpGTI32& other)
            : VCmpInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCmpGTI32>(*this);
        }
    };

    struct VCmpGEU32 : public VCmpInstruction
    {
        VCmpGEU32(const std::shared_ptr<Container>& dst,
                  const InstructionInput&           src0,
                  const InstructionInput&           src1,
                  std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                  const std::string&                comment = "")
            : VCmpInstruction(InstType::INST_U32, dst, src0, src1, sdwa, comment)
        {
            setInst("v_cmp_ge_u32");
        }

        VCmpGEU32(const VCmpGEU32& other)
            : VCmpInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCmpGEU32>(*this);
        }
    };

    struct VCmpGtU32 : public VCmpInstruction
    {
        VCmpGtU32(const std::shared_ptr<Container>& dst,
                  const InstructionInput&           src0,
                  const InstructionInput&           src1,
                  std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                  const std::string&                comment = "")
            : VCmpInstruction(InstType::INST_U32, dst, src0, src1, sdwa, comment)
        {
            setInst("v_cmp_gt_u32");
        }

        VCmpGtU32(const VCmpGtU32& other)
            : VCmpInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCmpGtU32>(*this);
        }
    };

    struct VCmpLeU32 : public VCmpInstruction
    {
        VCmpLeU32(const std::shared_ptr<Container>& dst,
                  const InstructionInput&           src0,
                  const InstructionInput&           src1,
                  std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                  const std::string&                comment = "")
            : VCmpInstruction(InstType::INST_U32, dst, src0, src1, sdwa, comment)
        {
            setInst("v_cmp_le_u32");
        }

        VCmpLeU32(const VCmpLeU32& other)
            : VCmpInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCmpLeU32>(*this);
        }
    };

    struct VCmpLeI32 : public VCmpInstruction
    {
        VCmpLeI32(const std::shared_ptr<Container>& dst,
                  const InstructionInput&           src0,
                  const InstructionInput&           src1,
                  std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                  const std::string&                comment = "")
            : VCmpInstruction(InstType::INST_I32, dst, src0, src1, sdwa, comment)
        {
            setInst("v_cmp_le_i32");
        }

        VCmpLeI32(const VCmpLeI32& other)
            : VCmpInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCmpLeI32>(*this);
        }
    };

    struct VCmpLtI32 : public VCmpInstruction
    {
        VCmpLtI32(const std::shared_ptr<Container>& dst,
                  const InstructionInput&           src0,
                  const InstructionInput&           src1,
                  std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                  const std::string&                comment = "")
            : VCmpInstruction(InstType::INST_I32, dst, src0, src1, sdwa, comment)
        {
            setInst("v_cmp_lt_i32");
        }

        VCmpLtI32(const VCmpLtI32& other)
            : VCmpInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCmpLtI32>(*this);
        }
    };

    struct VCmpLtU32 : public VCmpInstruction
    {
        VCmpLtU32(const std::shared_ptr<Container>& dst,
                  const InstructionInput&           src0,
                  const InstructionInput&           src1,
                  std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                  const std::string&                comment = "")
            : VCmpInstruction(InstType::INST_U32, dst, src0, src1, sdwa, comment)
        {
            setInst("v_cmp_lt_u32");
        }

        VCmpLtU32(const VCmpLtU32& other)
            : VCmpInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCmpLtU32>(*this);
        }
    };

    struct VCmpUF32 : public VCmpInstruction
    {
        VCmpUF32(const std::shared_ptr<Container>& dst,
                 const InstructionInput&           src0,
                 const InstructionInput&           src1,
                 std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                 const std::string&                comment = "")
            : VCmpInstruction(InstType::INST_F32, dst, src0, src1, sdwa, comment)
        {
            setInst("v_cmp_u_f32");
        }

        VCmpUF32(const VCmpUF32& other)
            : VCmpInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCmpUF32>(*this);
        }
    };

    struct VCmpNeI32 : public VCmpInstruction
    {
        VCmpNeI32(const std::shared_ptr<Container>& dst,
                  const InstructionInput&           src0,
                  const InstructionInput&           src1,
                  std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                  const std::string&                comment = "")
            : VCmpInstruction(InstType::INST_I32, dst, src0, src1, sdwa, comment)
        {
            setInst("v_cmp_ne_i32");
        }

        VCmpNeI32(const VCmpNeI32& other)
            : VCmpInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCmpNeI32>(*this);
        }
    };

    struct VCmpNeU32 : public VCmpInstruction
    {
        VCmpNeU32(const std::shared_ptr<Container>& dst,
                  const InstructionInput&           src0,
                  const InstructionInput&           src1,
                  std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                  const std::string&                comment = "")
            : VCmpInstruction(InstType::INST_U32, dst, src0, src1, sdwa, comment)
        {
            setInst("v_cmp_ne_u32");
        }

        VCmpNeU32(const VCmpNeU32& other)
            : VCmpInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCmpNeU32>(*this);
        }
    };

    struct VCmpNeU64 : public VCmpInstruction
    {
        VCmpNeU64(const std::shared_ptr<Container>& dst,
                  const InstructionInput&           src0,
                  const InstructionInput&           src1,
                  std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                  const std::string&                comment = "")
            : VCmpInstruction(InstType::INST_U64, dst, src0, src1, sdwa, comment)
        {
            setInst("v_cmp_ne_u64");
        }

        VCmpNeU64(const VCmpNeU64& other)
            : VCmpInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCmpNeU64>(*this);
        }
    };

    struct VCmpClassF32 : public VCmpInstruction
    {
        VCmpClassF32(const std::shared_ptr<Container>& dst,
                     const InstructionInput&           src0,
                     const InstructionInput&           src1,
                     std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                     const std::string&                comment = "")
            : VCmpInstruction(InstType::INST_F32, dst, src0, src1, sdwa, comment)
        {
            setInst("v_cmp_class_f32");
        }

        VCmpClassF32(const VCmpClassF32& other)
            : VCmpInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCmpClassF32>(*this);
        }
    };

    struct VCmpXClassF32 : public VCmpXInstruction
    {
        VCmpXClassF32(const std::shared_ptr<Container>& dst,
                      const InstructionInput&           src0,
                      const InstructionInput&           src1,
                      std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                      const std::string&                comment = "")
            : VCmpXInstruction(InstType::INST_F32, dst, src0, src1, sdwa, comment)
        {
            setInst("v_cmpx_class_f32");
        }

        VCmpXClassF32(const VCmpXClassF32& other)
            : VCmpXInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCmpXClassF32>(*this);
        }
    };

    struct VCmpXEqU32 : public VCmpXInstruction
    {
        VCmpXEqU32(const std::shared_ptr<Container>& dst,
                   const InstructionInput&           src0,
                   const InstructionInput&           src1,
                   std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                   const std::string&                comment = "")
            : VCmpXInstruction(InstType::INST_U32, dst, src0, src1, sdwa, comment)
        {
            setInst("v_cmpx_eq_u32");
        }

        VCmpXEqU32(const VCmpXEqU32& other)
            : VCmpXInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCmpXEqU32>(*this);
        }
    };

    struct VCmpXGeU32 : public VCmpXInstruction
    {
        VCmpXGeU32(const std::shared_ptr<Container>& dst,
                   const InstructionInput&           src0,
                   const InstructionInput&           src1,
                   std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                   const std::string&                comment = "")
            : VCmpXInstruction(InstType::INST_U32, dst, src0, src1, sdwa, comment)
        {
            setInst("v_cmpx_ge_u32");
        }

        VCmpXGeU32(const VCmpXGeU32& other)
            : VCmpXInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCmpXGeU32>(*this);
        }
    };

    struct VCmpXGtU32 : public VCmpXInstruction
    {
        VCmpXGtU32(const std::shared_ptr<Container>& dst,
                   const InstructionInput&           src0,
                   const InstructionInput&           src1,
                   std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                   const std::string&                comment = "")
            : VCmpXInstruction(InstType::INST_U32, dst, src0, src1, sdwa, comment)
        {
            setInst("v_cmpx_gt_u32");
        }

        VCmpXGtU32(const VCmpXGtU32& other)
            : VCmpXInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCmpXGtU32>(*this);
        }
    };

    struct VCmpXLeU32 : public VCmpXInstruction
    {
        VCmpXLeU32(const std::shared_ptr<Container>& dst,
                   const InstructionInput&           src0,
                   const InstructionInput&           src1,
                   std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                   const std::string&                comment = "")
            : VCmpXInstruction(InstType::INST_U32, dst, src0, src1, sdwa, comment)
        {
            setInst("v_cmpx_le_u32");
        }

        VCmpXLeU32(const VCmpXLeU32& other)
            : VCmpXInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCmpXLeU32>(*this);
        }
    };

    struct VCmpXLeI32 : public VCmpXInstruction
    {
        VCmpXLeI32(const std::shared_ptr<Container>& dst,
                   const InstructionInput&           src0,
                   const InstructionInput&           src1,
                   std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                   const std::string&                comment = "")
            : VCmpXInstruction(InstType::INST_I32, dst, src0, src1, sdwa, comment)
        {
            setInst("v_cmpx_le_i32");
        }

        VCmpXLeI32(const VCmpXLeI32& other)
            : VCmpXInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCmpXLeI32>(*this);
        }
    };

    struct VCmpXLtF32 : public VCmpXInstruction
    {
        VCmpXLtF32(const std::shared_ptr<Container>& dst,
                   const InstructionInput&           src0,
                   const InstructionInput&           src1,
                   std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                   const std::string&                comment = "")
            : VCmpXInstruction(InstType::INST_F32, dst, src0, src1, sdwa, comment)
        {
            setInst("v_cmpx_lt_f32");
        }

        VCmpXLtF32(const VCmpXLtF32& other)
            : VCmpXInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCmpXLtF32>(*this);
        }
    };

    struct VCmpXLtI32 : public VCmpXInstruction
    {
        VCmpXLtI32(const std::shared_ptr<Container>& dst,
                   const InstructionInput&           src0,
                   const InstructionInput&           src1,
                   std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                   const std::string&                comment = "")
            : VCmpXInstruction(InstType::INST_I32, dst, src0, src1, sdwa, comment)
        {
            setInst("v_cmpx_lt_i32");
        }

        VCmpXLtI32(const VCmpXLtI32& other)
            : VCmpXInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCmpXLtI32>(*this);
        }
    };

    struct VCmpXLtU32 : public VCmpXInstruction
    {
        VCmpXLtU32(const std::shared_ptr<Container>& dst,
                   const InstructionInput&           src0,
                   const InstructionInput&           src1,
                   std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                   const std::string&                comment = "")
            : VCmpXInstruction(InstType::INST_U32, dst, src0, src1, sdwa, comment)
        {
            setInst("v_cmpx_lt_u32");
        }

        VCmpXLtU32(const VCmpXLtU32& other)
            : VCmpXInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCmpXLtU32>(*this);
        }
    };

    struct VCmpXLtU64 : public VCmpXInstruction
    {
        VCmpXLtU64(const std::shared_ptr<Container>& dst,
                   const InstructionInput&           src0,
                   const InstructionInput&           src1,
                   std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                   const std::string&                comment = "")
            : VCmpXInstruction(InstType::INST_U64, dst, src0, src1, sdwa, comment)
        {
            setInst("v_cmpx_lt_u64");
        }

        VCmpXLtU64(const VCmpXLtU64& other)
            : VCmpXInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCmpXLtU64>(*this);
        }
    };

    struct VCmpXNeU16 : public VCmpXInstruction
    {
        VCmpXNeU16(const std::shared_ptr<Container>& dst,
                   const InstructionInput&           src0,
                   const InstructionInput&           src1,
                   std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                   const std::string&                comment = "")
            : VCmpXInstruction(InstType::INST_U16, dst, src0, src1, sdwa, comment)
        {
            setInst("v_cmpx_ne_u16");
        }

        VCmpXNeU16(const VCmpXNeU16& other)
            : VCmpXInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCmpXNeU16>(*this);
        }
    };

    struct VCmpXNeU32 : public VCmpXInstruction
    {
        VCmpXNeU32(const std::shared_ptr<Container>& dst,
                   const InstructionInput&           src0,
                   const InstructionInput&           src1,
                   std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                   const std::string&                comment = "")
            : VCmpXInstruction(InstType::INST_U32, dst, src0, src1, sdwa, comment)
        {
            setInst("v_cmpx_ne_u32");
        }

        VCmpXNeU32(const VCmpXNeU32& other)
            : VCmpXInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCmpXNeU32>(*this);
        }
    };

} // namespace rocisa
