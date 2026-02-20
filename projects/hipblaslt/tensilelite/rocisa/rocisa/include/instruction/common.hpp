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
#include "instruction.hpp"

#include <string>
#include <vector>

namespace rocisa
{
    struct SAbsI32 : public CommonInstruction
    {
        SAbsI32(const std::shared_ptr<RegisterContainer>& dst,
                const std::shared_ptr<RegisterContainer>& src,
                const std::string&                        comment = "")
            : CommonInstruction(
                InstType::INST_I32, dst, {src}, std::nullopt, std::nullopt, std::nullopt, comment)
        {
            setInst("s_abs_i32");
        }

        SAbsI32(const SAbsI32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SAbsI32>(*this);
        }
    };

    struct SMaxI32 : public CommonInstruction
    {
        SMaxI32(const std::shared_ptr<RegisterContainer>& dst,
                const InstructionInput&                   src0,
                const InstructionInput&                   src1,
                const std::string&                        comment = "")
            : CommonInstruction(InstType::INST_I32,
                                dst,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_max_i32");
        }

        SMaxI32(const SMaxI32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SMaxI32>(*this);
        }
    };

    struct SMaxU32 : public CommonInstruction
    {
        SMaxU32(const std::shared_ptr<RegisterContainer>& dst,
                const InstructionInput&                   src0,
                const InstructionInput&                   src1,
                const std::string&                        comment = "")
            : CommonInstruction(InstType::INST_I32,
                                dst,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_max_u32");
        }

        SMaxU32(const SMaxU32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SMaxU32>(*this);
        }
    };

    struct SMinI32 : public CommonInstruction
    {
        SMinI32(const std::shared_ptr<RegisterContainer>& dst,
                const InstructionInput&                   src0,
                const InstructionInput&                   src1,
                const std::string&                        comment = "")
            : CommonInstruction(InstType::INST_I32,
                                dst,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_min_i32");
        }

        SMinI32(const SMinI32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SMinI32>(*this);
        }
    };

    struct SMinU32 : public CommonInstruction
    {
        SMinU32(const std::shared_ptr<RegisterContainer>& dst,
                const InstructionInput&                   src0,
                const InstructionInput&                   src1,
                const std::string&                        comment = "")
            : CommonInstruction(InstType::INST_I32,
                                dst,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_min_u32");
        }

        SMinU32(const SMinU32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SMinU32>(*this);
        }
    };

    struct SAddI32 : public CommonInstruction
    {
        SAddI32(const std::shared_ptr<RegisterContainer>& dst,
                const InstructionInput&                   src0,
                const InstructionInput&                   src1,
                const std::string&                        comment = "")
            : CommonInstruction(InstType::INST_I32,
                                dst,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_add_i32");
        }

        SAddI32(const SAddI32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SAddI32>(*this);
        }
    };

    struct SAddU32 : public CommonInstruction
    {
        SAddU32(const std::shared_ptr<RegisterContainer>& dst,
                const InstructionInput&                   src0,
                const InstructionInput&                   src1,
                const std::string&                        comment = "")
            : CommonInstruction(InstType::INST_U32,
                                dst,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_add_u32");
        }

        SAddU32(const SAddU32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SAddU32>(*this);
        }
    };

    struct SAddCU32 : public CommonInstruction
    {
        SAddCU32(const std::shared_ptr<RegisterContainer>& dst,
                 const InstructionInput&                   src0,
                 const InstructionInput&                   src1,
                 const std::string&                        comment = "")
            : CommonInstruction(InstType::INST_U32,
                                dst,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_addc_u32");
        }

        SAddCU32(const SAddCU32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SAddCU32>(*this);
        }
    };

    struct SMulI32 : public CommonInstruction
    {
        SMulI32(const std::shared_ptr<RegisterContainer>& dst,
                const InstructionInput&                   src0,
                const InstructionInput&                   src1,
                const std::string&                        comment = "")
            : CommonInstruction(InstType::INST_I32,
                                dst,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_mul_i32");
        }

        SMulI32(const SMulI32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SMulI32>(*this);
        }
    };

    struct SMulHII32 : public CommonInstruction
    {
        SMulHII32(const std::shared_ptr<RegisterContainer>& dst,
                  const InstructionInput&                   src0,
                  const InstructionInput&                   src1,
                  const std::string&                        comment = "")
            : CommonInstruction(InstType::INST_HI_I32,
                                dst,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_mul_hi_i32");
        }

        SMulHII32(const SMulHII32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SMulHII32>(*this);
        }
    };

    struct SMulHIU32 : public CommonInstruction
    {
        SMulHIU32(const std::shared_ptr<RegisterContainer>& dst,
                  const InstructionInput&                   src0,
                  const InstructionInput&                   src1,
                  const std::string&                        comment = "")
            : CommonInstruction(InstType::INST_HI_U32,
                                dst,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_mul_hi_u32");
        }

        SMulHIU32(const SMulHIU32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SMulHIU32>(*this);
        }
    };

    struct SMulLOU32 : public CommonInstruction
    {
        SMulLOU32(const std::shared_ptr<RegisterContainer>& dst,
                  const InstructionInput&                   src0,
                  const InstructionInput&                   src1,
                  const std::string&                        comment = "")
            : CommonInstruction(InstType::INST_HI_U32,
                                dst,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_mul_lo_u32");
        }

        SMulLOU32(const SMulLOU32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SMulLOU32>(*this);
        }
    };

    struct SSubI32 : public CommonInstruction
    {
        SSubI32(const std::shared_ptr<RegisterContainer>& dst,
                const InstructionInput&                   src0,
                const InstructionInput&                   src1,
                const std::string&                        comment = "")
            : CommonInstruction(InstType::INST_I32,
                                dst,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_sub_i32");
        }

        SSubI32(const SSubI32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SSubI32>(*this);
        }
    };

    struct SSubU32 : public CommonInstruction
    {
        SSubU32(const std::shared_ptr<RegisterContainer>& dst,
                const InstructionInput&                   src0,
                const InstructionInput&                   src1,
                const std::string&                        comment = "")
            : CommonInstruction(InstType::INST_U32,
                                dst,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_sub_u32");
        }

        SSubU32(const SSubU32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SSubU32>(*this);
        }
    };

    struct SSubBU32 : public CommonInstruction
    {
        SSubBU32(const std::shared_ptr<RegisterContainer>& dst,
                 const InstructionInput&                   src0,
                 const InstructionInput&                   src1,
                 const std::string&                        comment = "")
            : CommonInstruction(InstType::INST_U32,
                                dst,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_subb_u32");
        }

        SSubBU32(const SSubBU32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SSubBU32>(*this);
        }
    };

    struct SCSelectB32 : public CommonInstruction
    {
        SCSelectB32(const std::shared_ptr<Container>& dst,
                    const InstructionInput&           src0,
                    const InstructionInput&           src1,
                    const std::string&                comment = "")
            : CommonInstruction(InstType::INST_B32,
                                dst,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_cselect_b32");
        }

        SCSelectB32(const SCSelectB32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SCSelectB32>(*this);
        }
    };

    struct SCSelectB64 : public CommonInstruction
    {
        SCSelectB64(const std::shared_ptr<Container>& dst,
                    const InstructionInput&           src0,
                    const InstructionInput&           src1,
                    const std::string&                comment = "")
            : CommonInstruction(InstType::INST_B64,
                                dst,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_cselect_b64");
        }

        SCSelectB64(const SCSelectB64& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SCSelectB64>(*this);
        }
    };

    struct SAndB32 : public CommonInstruction
    {
        SAndB32(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src0,
                const InstructionInput&           src1,
                const std::string&                comment = "")
            : CommonInstruction(InstType::INST_B32,
                                dst,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_and_b32");
        }

        SAndB32(const SAndB32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SAndB32>(*this);
        }
    };

    struct SAndB64 : public CommonInstruction
    {
        SAndB64(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src0,
                const InstructionInput&           src1,
                const std::string&                comment = "")
            : CommonInstruction(InstType::INST_B64,
                                dst,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_and_b64");
        }

        SAndB64(const SAndB64& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SAndB64>(*this);
        }
    };

    struct SAndN2B32 : public CommonInstruction
    {
        SAndN2B32(const std::shared_ptr<Container>& dst,
                  const InstructionInput&           src0,
                  const InstructionInput&           src1,
                  const std::string&                comment = "")
            : CommonInstruction(InstType::INST_B32,
                                dst,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_andn2_b32");
        }

        SAndN2B32(const SAndN2B32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SAndN2B32>(*this);
        }
    };

    struct SOrB32 : public CommonInstruction
    {
        SOrB32(const std::shared_ptr<Container>& dst,
               const InstructionInput&           src0,
               const InstructionInput&           src1,
               const std::string&                comment = "")
            : CommonInstruction(InstType::INST_B32,
                                dst,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_or_b32");
        }

        SOrB32(const SOrB32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SOrB32>(*this);
        }
    };

    struct SXorB32 : public CommonInstruction
    {
        SXorB32(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src0,
                const InstructionInput&           src1,
                const std::string&                comment = "")
            : CommonInstruction(InstType::INST_B32,
                                dst,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_xor_b32");
        }

        SXorB32(const SXorB32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SXorB32>(*this);
        }
    };

    struct SOrB64 : public CommonInstruction
    {
        SOrB64(const std::shared_ptr<Container>& dst,
               const InstructionInput&           src0,
               const InstructionInput&           src1,
               const std::string&                comment = "")
            : CommonInstruction(InstType::INST_B64,
                                dst,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_or_b64");
        }

        SOrB64(const SOrB64& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SOrB64>(*this);
        }
    };

    struct SGetPCB64 : public CommonInstruction
    {
        SGetPCB64(const std::shared_ptr<Container>& dst, const std::string& comment = "")
            : CommonInstruction(
                InstType::INST_B64, dst, {}, std::nullopt, std::nullopt, std::nullopt, comment)
        {
            setInst("s_getpc_b64");
        }

        SGetPCB64(const SGetPCB64& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SGetPCB64>(*this);
        }
    };

    struct SLShiftLeftB32 : public CommonInstruction
    {
        SLShiftLeftB32(const std::shared_ptr<Container>& dst,
                       const InstructionInput&           shiftHex,
                       const InstructionInput&           src,
                       const std::string&                comment = "")
            : CommonInstruction(InstType::INST_B32,
                                dst,
                                {src, shiftHex},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_lshl_b32");
        }

        SLShiftLeftB32(const SLShiftLeftB32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SLShiftLeftB32>(*this);
        }
    };

    struct SLShiftRightB32 : public CommonInstruction
    {
        SLShiftRightB32(const std::shared_ptr<Container>& dst,
                        const InstructionInput&           shiftHex,
                        const InstructionInput&           src,
                        const std::string&                comment = "")
            : CommonInstruction(InstType::INST_B32,
                                dst,
                                {src, shiftHex},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_lshr_b32");
        }

        SLShiftRightB32(const SLShiftRightB32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SLShiftRightB32>(*this);
        }
    };

    struct SLShiftLeftB64 : public CommonInstruction
    {
        SLShiftLeftB64(const std::shared_ptr<Container>& dst,
                       const InstructionInput&           shiftHex,
                       const InstructionInput&           src,
                       const std::string&                comment = "")
            : CommonInstruction(InstType::INST_B64,
                                dst,
                                {src, shiftHex},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_lshl_b64");
        }

        SLShiftLeftB64(const SLShiftLeftB64& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SLShiftLeftB64>(*this);
        }
    };

    struct SLShiftRightB64 : public CommonInstruction
    {
        SLShiftRightB64(const std::shared_ptr<Container>& dst,
                        const InstructionInput&           shiftHex,
                        const InstructionInput&           src,
                        const std::string&                comment = "")
            : CommonInstruction(InstType::INST_B64,
                                dst,
                                {src, shiftHex},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_lshr_b64");
        }

        SLShiftRightB64(const SLShiftRightB64& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SLShiftRightB64>(*this);
        }
    };

    struct SAShiftRightI32 : public CommonInstruction
    {
        SAShiftRightI32(const std::shared_ptr<Container>& dst,
                        const InstructionInput&           shiftHex,
                        const InstructionInput&           src,
                        const std::string&                comment = "")
            : CommonInstruction(InstType::INST_I32,
                                dst,
                                {src, shiftHex},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_ashr_i32");
        }

        SAShiftRightI32(const SAShiftRightI32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SAShiftRightI32>(*this);
        }
    };

    struct SLShiftLeft1AddU32 : public CommonInstruction
    {
        SLShiftLeft1AddU32(const std::shared_ptr<Container>& dst,
                           const InstructionInput&           src0,
                           const InstructionInput&           src1,
                           const std::string&                comment = "")
            : CommonInstruction(InstType::INST_B32,
                                dst,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_lshl1_add_u32");
        }

        SLShiftLeft1AddU32(const SLShiftLeft1AddU32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SLShiftLeft1AddU32>(*this);
        }
    };

    struct SLShiftLeft2AddU32 : public CommonInstruction
    {
        SLShiftLeft2AddU32(const std::shared_ptr<Container>& dst,
                           const InstructionInput&           src0,
                           const InstructionInput&           src1,
                           const std::string&                comment = "")
            : CommonInstruction(InstType::INST_B32,
                                dst,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_lshl2_add_u32");
        }

        SLShiftLeft2AddU32(const SLShiftLeft2AddU32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SLShiftLeft2AddU32>(*this);
        }
    };

    struct SLShiftLeft3AddU32 : public CommonInstruction
    {
        SLShiftLeft3AddU32(const std::shared_ptr<Container>& dst,
                           const InstructionInput&           src0,
                           const InstructionInput&           src1,
                           const std::string&                comment = "")
            : CommonInstruction(InstType::INST_B32,
                                dst,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_lshl3_add_u32");
        }

        SLShiftLeft3AddU32(const SLShiftLeft3AddU32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SLShiftLeft3AddU32>(*this);
        }
    };

    struct SLShiftLeft4AddU32 : public CommonInstruction
    {
        SLShiftLeft4AddU32(const std::shared_ptr<Container>& dst,
                           const InstructionInput&           src0,
                           const InstructionInput&           src1,
                           const std::string&                comment = "")
            : CommonInstruction(InstType::INST_B32,
                                dst,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_lshl4_add_u32");
        }

        SLShiftLeft4AddU32(const SLShiftLeft4AddU32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SLShiftLeft4AddU32>(*this);
        }
    };

    struct SSetMask : public CommonInstruction
    {
        SSetMask(const std::shared_ptr<Container>& dst,
                 const InstructionInput&           src,
                 const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_B64, dst, {src}, std::nullopt, std::nullopt, std::nullopt, comment)
        {
            if(kernel().wavefront == 32)
            {
                instType = InstType::INST_B32;
                setInst("s_mov_b32");
            }
            else
            {
                setInst("s_mov_b64");
            }
        }

        SSetMask(const SSetMask& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SSetMask>(*this);
        }
    };

    struct SMovB32 : public CommonInstruction
    {
        SMovB32(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src,
                const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_B32, dst, {src}, std::nullopt, std::nullopt, std::nullopt, comment)
        {
            setInst("s_mov_b32");
        }

        SMovB32(const SMovB32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SMovB32>(*this);
        }
    };

    struct SMovB64 : public CommonInstruction
    {
        SMovB64(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src,
                const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_B64, dst, {src}, std::nullopt, std::nullopt, std::nullopt, comment)
        {
            setInst("s_mov_b64");
        }

        SMovB64(const SMovB64& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SMovB64>(*this);
        }
    };

    struct SCMovB32 : public CommonInstruction
    {
        SCMovB32(const std::shared_ptr<Container>& dst,
                 const InstructionInput&           src,
                 const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_B32, dst, {src}, std::nullopt, std::nullopt, std::nullopt, comment)
        {
            setInst("s_cmov_b32");
        }

        SCMovB32(const SCMovB32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SCMovB32>(*this);
        }
    };

    struct SCMovB64 : public CommonInstruction
    {
        SCMovB64(const std::shared_ptr<Container>& dst,
                 const InstructionInput&           src,
                 const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_B64, dst, {src}, std::nullopt, std::nullopt, std::nullopt, comment)
        {
            setInst("s_cmov_b64");
        }

        SCMovB64(const SCMovB64& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SCMovB64>(*this);
        }
    };

    struct SFf1B32 : public CommonInstruction
    {
        SFf1B32(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src,
                const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_B32, dst, {src}, std::nullopt, std::nullopt, std::nullopt, comment)
        {
            setInst("s_ff1_i32_b32");
        }

        SFf1B32(const SFf1B32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SFf1B32>(*this);
        }
    };

    struct SBfmB32 : public CommonInstruction
    {
        SBfmB32(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src0,
                const InstructionInput&           src1,
                const std::string&                comment = "")
            : CommonInstruction(InstType::INST_B32,
                                dst,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_bfm_b32");
        }

        SBfmB32(const SBfmB32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SBfmB32>(*this);
        }
    };

    struct SFlbitI32B32 : public CommonInstruction
    {
        SFlbitI32B32(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src,
                const std::string&                comment = "")
            : CommonInstruction(InstType::INST_B32,
                                dst,
                                {src},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("s_flbit_i32_b32");
        }

        SFlbitI32B32(const SFlbitI32B32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SFlbitI32B32>(*this);
        }
    };

    struct SMovkI32 : public CommonInstruction
    {
        SMovkI32(const std::shared_ptr<Container>& dst,
                 const InstructionInput&           src,
                 const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_I32, dst, {src}, std::nullopt, std::nullopt, std::nullopt, comment)
        {
            setInst("s_movk_i32");
        }

        SMovkI32(const SMovkI32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SMovkI32>(*this);
        }
    };

    struct SSExtI16toI32 : public CommonInstruction
    {
        SSExtI16toI32(const std::shared_ptr<Container>& dst,
                      const InstructionInput&           src,
                      const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_I32, dst, {src}, std::nullopt, std::nullopt, std::nullopt, comment)
        {
            setInst("s_sext_i32_i16");
        }

        SSExtI16toI32(const SSExtI16toI32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SSExtI16toI32>(*this);
        }
    };

    struct SAndSaveExecB32 : public CommonInstruction
    {
        SAndSaveExecB32(const std::shared_ptr<Container>& dst,
                        const InstructionInput&           src,
                        const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_B32, dst, {src}, std::nullopt, std::nullopt, std::nullopt, comment)
        {
            setInst("s_and_saveexec_b32");
        }

        SAndSaveExecB32(const SAndSaveExecB32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SAndSaveExecB32>(*this);
        }
    };

    struct SAndSaveExecB64 : public CommonInstruction
    {
        SAndSaveExecB64(const std::shared_ptr<Container>& dst,
                        const InstructionInput&           src,
                        const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_B64, dst, {src}, std::nullopt, std::nullopt, std::nullopt, comment)
        {
            setInst("s_and_saveexec_b64");
        }

        SAndSaveExecB64(const SAndSaveExecB64& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SAndSaveExecB64>(*this);
        }
    };

    struct SOrSaveExecB32 : public CommonInstruction
    {
        SOrSaveExecB32(const std::shared_ptr<Container>& dst,
                       const InstructionInput&           src,
                       const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_B32, dst, {src}, std::nullopt, std::nullopt, std::nullopt, comment)
        {
            setInst("s_or_saveexec_b32");
        }

        SOrSaveExecB32(const SOrSaveExecB32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SOrSaveExecB32>(*this);
        }
    };

    struct SOrSaveExecB64 : public CommonInstruction
    {
        SOrSaveExecB64(const std::shared_ptr<Container>& dst,
                       const InstructionInput&           src,
                       const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_B64, dst, {src}, std::nullopt, std::nullopt, std::nullopt, comment)
        {
            setInst("s_or_saveexec_b64");
        }

        SOrSaveExecB64(const SOrSaveExecB64& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SOrSaveExecB64>(*this);
        }
    };

    struct SSetPrior : public Instruction
    {
        SSetPrior(int prior, const std::string& comment = "")
            : Instruction(InstType::INST_NOTYPE, comment)
            , prior(prior)
        {
            setInst("s_setprio");
        }

        SSetPrior(const SSetPrior& other)
            : Instruction(other)
            , prior(other.prior)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SSetPrior>(*this);
        }

        std::vector<InstructionInput> getParams() const override
        {
            return {prior};
        }

        std::string toString() const override
        {
            return formatWithComment(instStr + " " + std::to_string(prior));
        }

    private:
        int prior;
    };

    struct SBarrier : public Instruction
    {
        SBarrier(const std::string& comment = "")
            : Instruction(InstType::INST_NOTYPE, comment)
        {
            if(getAsmCaps()["HasNewBarrier"])
            {
                setInst("s_barrier_signal -1 \ns_barrier_wait -1");
            }
            else
            {
                setInst("s_barrier");
            }
        }

        SBarrier(const SBarrier& other)
            : Instruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SBarrier>(*this);
        }

        std::vector<InstructionInput> getParams() const override
        {
            return {};
        }

        std::string toString() const override
        {
            return formatWithComment(instStr);
        }
    };

    struct SDcacheWb : public Instruction
    {
        SDcacheWb(const std::string& comment = "")
            : Instruction(InstType::INST_NOTYPE, comment)
        {
            setInst("s_dcache_wb");
        }

        SDcacheWb(const SDcacheWb& other)
            : Instruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SDcacheWb>(*this);
        }

        std::vector<InstructionInput> getParams() const override
        {
            return {};
        }

        std::string toString() const override
        {
            return formatWithComment(instStr);
        }
    };

    struct SNop : public Instruction
    {
        SNop(int waitState, const std::string& comment = "")
            : Instruction(InstType::INST_NOTYPE, comment)
            , waitState(waitState)
        {
            setInst("s_nop");
        }

        SNop(const SNop& other)
            : Instruction(other)
            , waitState(other.waitState)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SNop>(*this);
        }

        std::vector<InstructionInput> getParams() const override
        {
            return {waitState};
        }

        std::string toString() const override
        {
            return formatWithComment(instStr + " " + std::to_string(waitState));
        }

    private:
        int waitState;
    };

    struct SEndpgm : public Instruction
    {
        SEndpgm(const std::string& comment = "")
            : Instruction(InstType::INST_NOTYPE, comment)
        {
            setInst("s_endpgm");
        }

        SEndpgm(const SEndpgm& other)
            : Instruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SEndpgm>(*this);
        }

        std::vector<InstructionInput> getParams() const override
        {
            return {};
        }

        std::string toString() const override
        {
            return formatWithComment(instStr);
        }
    };

    struct SSleep : public Instruction
    {
        SSleep(const int simm16, const std::string& comment = "")
            : Instruction(InstType::INST_NOTYPE, comment)
            , simm16(simm16)
        {
            setInst("s_sleep");
        }

        SSleep(const SSleep& other)
            : Instruction(other)
            , simm16(other.simm16)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SSleep>(*this);
        }

        std::vector<InstructionInput> getParams() const override
        {
            return {simm16};
        }

        std::string toString() const override
        {
            return formatWithComment(instStr + " " + std::to_string(simm16));
        }

    private:
        int simm16;
    };

    struct SGetRegB32 : public CommonInstruction
    {
        SGetRegB32(const std::shared_ptr<Container>& dst,
                   const InstructionInput&           src,
                   const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_B32, dst, {src}, std::nullopt, std::nullopt, std::nullopt, comment)
        {
            setInst("s_getreg_b32");
        }

        SGetRegB32(const SGetRegB32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SGetRegB32>(*this);
        }
    };

    struct SSetRegB32 : public CommonInstruction
    {
        SSetRegB32(const std::shared_ptr<Container>& dst,
                   const InstructionInput&           src,
                   const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_B32, dst, {src}, std::nullopt, std::nullopt, std::nullopt, comment)
        {
            setInst("s_setreg_b32");
        }

        SSetRegB32(const SSetRegB32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SSetRegB32>(*this);
        }
    };

    struct SSetRegIMM32B32 : public CommonInstruction
    {
        SSetRegIMM32B32(const std::shared_ptr<Container>& dst,
                        const InstructionInput&           src,
                        const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_B32, dst, {src}, std::nullopt, std::nullopt, std::nullopt, comment)
        {
            setInst("s_setreg_IMM32_b32");
        }

        SSetRegIMM32B32(const SSetRegIMM32B32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SSetRegIMM32B32>(*this);
        }
    };

    struct _SWaitCnt : public Instruction
    {
        _SWaitCnt(int lgkmcnt = -1, int vmcnt = -1, const std::string& comment = "")
            : Instruction(InstType::INST_NOTYPE, comment)
            , lgkmcnt(lgkmcnt)
            , vmcnt(vmcnt)
        {
        }

        _SWaitCnt(const _SWaitCnt& other)
            : Instruction(other)
            , lgkmcnt(other.lgkmcnt)
            , vmcnt(other.vmcnt)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<_SWaitCnt>(*this);
        }

        std::vector<InstructionInput> getParams() const override
        {
            return {lgkmcnt, vmcnt};
        }

        std::string toString() const override
        {
            std::string waitStr;
            if(lgkmcnt == 0 && vmcnt == 0)
            {
                waitStr = "0";
            }
            else
            {
                if(lgkmcnt != -1)
                {
                    int maxLgkmcnt = getAsmCaps()["MaxLgkmcnt"];
                    waitStr = "lgkmcnt(" + std::to_string(std::min(lgkmcnt, maxLgkmcnt)) + ")";
                }
                if(vmcnt != -1)
                {
                    int maxVmcnt = getAsmCaps()["MaxVmcnt"];
                    waitStr += (waitStr != "" ? ", " : "");
                    waitStr += "vmcnt(" + std::to_string(std::min(vmcnt, maxVmcnt)) + ")";
                }
            }
            return formatWithComment("s_waitcnt " + waitStr);
        }

    private:
        int lgkmcnt;
        int vmcnt;
    };

    struct _SWaitCntVscnt : public Instruction
    {
        _SWaitCntVscnt(int vscnt = -1, const std::string& comment = "")
            : Instruction(InstType::INST_NOTYPE, comment)
            , vscnt(vscnt)
        {
        }

        _SWaitCntVscnt(const _SWaitCntVscnt& other)
            : Instruction(other)
            , vscnt(other.vscnt)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<_SWaitCntVscnt>(*this);
        }

        std::vector<InstructionInput> getParams() const override
        {
            return {vscnt};
        }

        std::string toString() const override
        {
            int maxVscnt = getAsmCaps()["MaxVscnt"];
            return formatWithComment("s_waitcnt_vscnt null " + std::to_string(std::min(vscnt, maxVscnt)));
        }

    private:
        int vscnt;
    };

    struct _SWaitStorecnt : public Instruction
    {
        _SWaitStorecnt(int storecnt = -1, const std::string& comment = "")
            : Instruction(InstType::INST_NOTYPE, comment)
            , storecnt(storecnt)
        {
        }

        _SWaitStorecnt(const _SWaitStorecnt& other)
            : Instruction(other)
            , storecnt(other.storecnt)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<_SWaitStorecnt>(*this);
        }

        std::vector<InstructionInput> getParams() const override
        {
            return {storecnt};
        }

        std::string toString() const override
        {
            int maxStorecnt = getAsmCaps()["MaxStorecnt"];
            return formatWithComment("s_wait_storecnt " + std::to_string(std::min(storecnt, maxStorecnt)));
        }

    private:
        int storecnt;
    };

    struct _SWaitLoadcnt : public Instruction
    {
        _SWaitLoadcnt(int loadcnt = -1, const std::string& comment = "")
            : Instruction(InstType::INST_NOTYPE, comment)
            , loadcnt(loadcnt)
        {
        }

        _SWaitLoadcnt(const _SWaitLoadcnt& other)
            : Instruction(other)
            , loadcnt(other.loadcnt)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<_SWaitLoadcnt>(*this);
        }

        std::vector<InstructionInput> getParams() const override
        {
            return {loadcnt};
        }

        std::string toString() const override
        {
            int maxLoadcnt = getAsmCaps()["MaxLoadcnt"];
            return formatWithComment("s_wait_loadcnt " + std::to_string(std::min(loadcnt, maxLoadcnt)));
        }

    private:
        int loadcnt;
    };

    struct _SWaitKMcnt : public Instruction
    {
        _SWaitKMcnt(int kmcnt = -1, const std::string& comment = "")
            : Instruction(InstType::INST_NOTYPE, comment)
            , kmcnt(kmcnt)
        {
        }

        _SWaitKMcnt(const _SWaitKMcnt& other)
            : Instruction(other)
            , kmcnt(other.kmcnt)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<_SWaitKMcnt>(*this);
        }

        std::vector<InstructionInput> getParams() const override
        {
            return {kmcnt};
        }

        std::string toString() const override
        {
            int maxKmcnt = getAsmCaps()["MaxKmcnt"];
            return formatWithComment("s_wait_kmcnt " + std::to_string(std::min(kmcnt, maxKmcnt)));
        }

    private:
        int kmcnt;
    };

    struct _SWaitDscnt : public Instruction
    {
        _SWaitDscnt(int dscnt = -1, const std::string& comment = "")
            : Instruction(InstType::INST_NOTYPE, comment)
            , dscnt(dscnt)
        {
        }

        _SWaitDscnt(const _SWaitDscnt& other)
            : Instruction(other)
            , dscnt(other.dscnt)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<_SWaitDscnt>(*this);
        }

        std::vector<InstructionInput> getParams() const override
        {
            return {dscnt};
        }

        std::string toString() const override
        {
            int maxDscnt = getAsmCaps()["MaxDscnt"];
            return formatWithComment("s_wait_dscnt " + std::to_string(std::min(dscnt, maxDscnt)));
        }

    private:
        int dscnt;
    };

    struct SWaitCnt : public CompositeInstruction
    {
        /*
        vlcnt: Number of VMEM load (and atomic with return value) instructions issued but not yet completed.
        vscnt: Number of VMEM store (and atomic w/o return value) instructions issued but not yet completed.
        dscnt: Number of LDS instructions issued but not yet completed.
        kmcnt: Number of constant-fetch (scalar memory read), and message instructions issued but not yet completed.

        In some ISA, VMEM load/store share the same counter(vmcnt). LDS, scalar memory read and message share
        the same counter(lgkmcnt). These counters are combined from the 4 counters above as:
            vmcnt   = vlcnt + vscnt
            lgkmcnt = dscnt + kmcnt

        Example: 4 VMEM instructions vl_0, vl_1, vs_0, vl_2 are issued and we want to wait for vl_1 to complete.

            If the target ISA has separate counters for load and store, use SWaitCnt(vlcnt=1),
            which means vl_2 is not completed yet.

            If the target ISA has a single counter for load and store, use SWaitCnt(vlcnt=1, vscnt=1),
            which means vs_0, vl_2 are not completed yet.
        */
        int vlcnt;
        int vscnt;
        int dscnt;
        int kmcnt;

        SWaitCnt(int                vlcnt   = -1,
                 int                vscnt   = -1,
                 int                dscnt   = -1,
                 int                kmcnt   = -1,
                 const std::string& comment = "",
                 bool               waitAll = false)
            : CompositeInstruction(InstType::INST_NOTYPE, nullptr, {}, comment)
            , vlcnt(vlcnt)
            , vscnt(vscnt)
            , dscnt(dscnt)
            , kmcnt(kmcnt)
            , waitAll(waitAll)
        {
        }

        SWaitCnt(const SWaitCnt& other)
            : CompositeInstruction(other)
            , vlcnt(other.vlcnt)
            , vscnt(other.vscnt)
            , dscnt(other.dscnt)
            , kmcnt(other.kmcnt)
            , waitAll(other.waitAll)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SWaitCnt>(*this);
        }

        std::vector<InstructionInput> getParams() const override
        {
            return {};
        }

        std::vector<std::shared_ptr<Instruction>> setupInstructions() const override
        {
            int         vlcnt   = this->vlcnt;
            int         vscnt   = this->vscnt;
            int         dscnt   = this->dscnt;
            int         kmcnt   = this->kmcnt;
            std::string comment = this->comment;

            // Currently these two capabilities should be both enabled or disabled together
            assert(getAsmCaps()["SeparateVMcnt"] == getAsmCaps()["SeparateLGKMcnt"]);

            if(waitAll)
            {
                vlcnt   = 0;
                vscnt   = 0;
                dscnt   = 0;
                kmcnt   = 0;
                comment = "(Wait all)";
            }

            std::vector<std::shared_ptr<Instruction>> instructions;

            if(getAsmCaps()["SeparateVscnt"])
            {
                int lgkmcnt = (dscnt != -1 || kmcnt != -1)? (dscnt != -1 ? dscnt : 0) + (kmcnt != -1 ? kmcnt : 0) : -1;
                int vmcnt   = vlcnt; // With SeparateVscnt, vmcnt only counts load instructions
                if(vlcnt != -1 || lgkmcnt != -1)
                {
                    instructions.push_back(std::make_shared<_SWaitCnt>(lgkmcnt, vmcnt, comment));
                }
                if(vscnt != -1)
                {
                    instructions.push_back(std::make_shared<_SWaitCntVscnt>(vscnt, comment));
                }
            }
            else if(getAsmCaps()["SeparateVMcnt"] && getAsmCaps()["SeparateLGKMcnt"])
            {
                if(dscnt != -1)
                {
                    instructions.push_back(std::make_shared<_SWaitDscnt>(dscnt, comment));
                }
                if(kmcnt != -1)
                {
                    instructions.push_back(std::make_shared<_SWaitKMcnt>(kmcnt, comment));
                }
                if(vlcnt != -1)
                {
                    instructions.push_back(std::make_shared<_SWaitLoadcnt>(vlcnt, comment));
                }
                if(vscnt != -1)
                {
                    instructions.push_back(std::make_shared<_SWaitStorecnt>(vscnt, comment));
                }
            }
            else
            {
                int lgkmcnt = (dscnt != -1 || kmcnt != -1)? (dscnt != -1 ? dscnt : 0) + (kmcnt != -1 ? kmcnt : 0) : -1;
                int vmcnt   = (vscnt != -1 || vlcnt != -1)? (vscnt != -1 ? vscnt : 0) + (vlcnt != -1 ? vlcnt : 0) : -1;
                if(vmcnt != -1 || lgkmcnt != -1)
                {
                    instructions.push_back(std::make_shared<_SWaitCnt>(lgkmcnt, vmcnt, comment));
                }
            }
            return std::move(instructions);
        }

    private:
        bool waitAll;
    };

    /*
        GFX12:
        +-----------------------+-----------------+---------+-----+-----------+-----------------+--------+---------+
        | 15  | 14  | 13  | 12  | 11  | 10  |  9  |    8    |  7  |  6  |  5  |  4  |  3  |  2  |    1   |    0    |
        |-----------------------+-----------------+---------+-----+-----------+-----------------+--------+---------|
        |        va_vdst        |     va_sdst     | va_ssrc | hc  |   rsvd    |     vm_vsrc     | va_vcc | sa_sdst |
        +-----------------------+-----------------+---------+-----+-----------+-----------------+--------+---------+
    */
    struct SWaitAlu : public Instruction
    {
        SWaitAlu(int                va_vdst  = -1,
                 int                va_sdst  = -1,
                 int                va_ssrc  = -1,
                 int                hold_cnt = -1,
                 int                vm_vsrc  = -1,
                 int                va_vcc   = -1,
                 int                sa_sdst  = -1,
                 const std::string& comment  = "")
            : Instruction(InstType::INST_NOTYPE, comment)
            , va_vdst(va_vdst)
            , va_sdst(va_sdst)
            , va_ssrc(va_ssrc)
            , hold_cnt(hold_cnt)
            , vm_vsrc(vm_vsrc)
            , va_vcc(va_vcc)
            , sa_sdst(sa_sdst)
        {
            if(kernel().isaVersion[0] < 12)
            {
                setInst("s_waitcnt_depctr");
            }
            else
            {
                setInst("s_wait_alu");
            }
        }

        SWaitAlu(const SWaitAlu& other)
            : SWaitAlu(other.va_vdst,
                       other.va_sdst,
                       other.va_ssrc,
                       other.hold_cnt,
                       other.vm_vsrc,
                       other.va_vcc,
                       other.sa_sdst,
                       other.comment)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SWaitAlu>(*this);
        }

        std::vector<InstructionInput> getParams() const override
        {
            return {va_vdst, va_sdst, va_ssrc, hold_cnt, vm_vsrc, va_vcc, sa_sdst};
        }

        std::string toString() const override
        {
            if(!getArchCaps()["HasSchedMode"])
                return "";

            std::string result;
            if(va_vdst != -1)
                result += " depctr_va_vdst(" + std::to_string(va_vdst) + ")";
            if(va_sdst != -1)
                result += " depctr_va_sdst(" + std::to_string(va_sdst) + ")";
            if(va_ssrc != -1)
                result += " depctr_va_ssrc(" + std::to_string(va_ssrc) + ")";
            if(hold_cnt != -1)
                result += " depctr_hold_cnt(" + std::to_string(hold_cnt) + ")";
            if(vm_vsrc != -1)
                result += " depctr_vm_vsrc(" + std::to_string(vm_vsrc) + ")";
            if(va_vcc != -1)
                result += " depctr_va_vcc(" + std::to_string(va_vcc) + ")";
            if(sa_sdst != -1)
                result += " depctr_sa_sdst(" + std::to_string(sa_sdst) + ")";

            if(result.empty())
                return "";

            return formatWithComment(instStr + result);
        }

    private:
        int va_vdst;
        int va_sdst;
        int va_ssrc;
        int hold_cnt;
        int vm_vsrc;
        int va_vcc;
        int sa_sdst;
    };

    struct SDelayAlu : public Instruction
    {
        SDelayAlu(const rocisa::DelayALUType          instid0type,
                  const int                           instid0cnt,
                  std::optional<int>                  instskipCnt = std::nullopt,
                  std::optional<rocisa::DelayALUType> instid1type = std::nullopt,
                  std::optional<int>                  instid1cnt  = std::nullopt,
                  const std::string&                  comment     = "")
            : Instruction(InstType::INST_NOTYPE, comment)
            , instid0type(instid0type)
            , instid0cnt(instid0cnt)
            , instskipCnt(instskipCnt)
            , instid1type(instid1type)
            , instid1cnt(instid1cnt)
        {
            setInst("s_delay_alu");
        }

        SDelayAlu(const SDelayAlu& other)
            : SDelayAlu(other.instid0type,
                        other.instid0cnt,
                        other.instskipCnt,
                        other.instid1type,
                        other.instid1cnt,
                        other.comment)
        {
        }

        bool hasInstID1() const
        {
            return this->instskipCnt != std::nullopt || this->instid1type != std::nullopt
                   || this->instid1cnt != std::nullopt;
        }

        bool setInstID1(const int&          instskipCnt,
                        const DelayALUType& instid1type,
                        const int&          instid1cnt)
        {
            if(hasInstID1())
            {
                return false;
            }

            this->instskipCnt = instskipCnt;
            this->instid1type = instid1type;
            this->instid1cnt  = instid1cnt;
            return true;
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SDelayAlu>(*this);
        }

        std::vector<InstructionInput> getParams() const override
        {
            if(hasInstID1())
            {
                return {static_cast<int>(instid0type),
                        instid0cnt,
                        instskipCnt.value_or(-1),
                        static_cast<int>(instid1type.value_or(DelayALUType::OTHER)),
                        instid1cnt.value_or(-1)};
            }

            return {static_cast<int>(instid0type), instid0cnt};
        }

        std::string toString() const override
        {
            if(!getAsmCaps()["s_delay_alu"])
                return "";

            std::string result;
            result += " instid0(" + ::rocisa::toString(instid0type, instid0cnt) + ")";
            if(!hasInstID1())
            {
                return formatWithComment(instStr + result);
            }

            result += " | instskip("
                      + ::rocisa::toString(static_cast<DelayALUSkip>(instskipCnt.value())) + ")";
            result += " | instid1(" + ::rocisa::toString(instid1type.value(), instid1cnt.value())
                      + ")";

            return formatWithComment(instStr + result);
        }

    private:
        DelayALUType                instid0type;
        int                         instid0cnt;
        std::optional<int>          instskipCnt;
        std::optional<DelayALUType> instid1type;
        std::optional<int>          instid1cnt;
    };

    struct VAddF16 : public CommonInstruction
    {
        VAddF16(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src0,
                const InstructionInput&           src1,
                std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_F16, dst, {src0, src1}, std::nullopt, sdwa, std::nullopt, comment)
        {
            setInst("v_add_f16");
        }

        VAddF16(const VAddF16& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VAddF16>(*this);
        }
    };

    struct VAddF32 : public CommonInstruction
    {
        VAddF32(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src0,
                const InstructionInput&           src1,
                std::optional<DPPModifiers>       dpp     = std::nullopt,
                std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_F32, dst, {src0, src1}, dpp, sdwa, std::nullopt, comment)
        {
            setInst("v_add_f32");
        }

        VAddF32(const std::shared_ptr<Container>&    dst,
                const std::vector<InstructionInput>& srcs,
                std::optional<DPPModifiers>          dpp     = std::nullopt,
                std::optional<SDWAModifiers>         sdwa    = std::nullopt,
                const std::string&                   comment = "")
            : CommonInstruction(InstType::INST_F32, dst, srcs, dpp, sdwa, std::nullopt, comment)
        {
            setInst("v_add_f32");
        }

        VAddF32(const VAddF32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VAddF32>(*this);
        }
    };

    struct VAddF64 : public CommonInstruction
    {
        VAddF64(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src0,
                const InstructionInput&           src1,
                std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_F64, dst, {src0, src1}, std::nullopt, sdwa, std::nullopt, comment)
        {
            setInst("v_add_f64");
        }

        VAddF64(const VAddF64& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VAddF64>(*this);
        }
    };

    struct VAddI32 : public CommonInstruction
    {
        VAddI32(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src0,
                const InstructionInput&           src1,
                std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_I32, dst, {src0, src1}, std::nullopt, sdwa, std::nullopt, comment)
        {
            if(getAsmBugs()["ExplicitNC"])
            {
                setInst("v_add_nc_i32");
            }
            else if(getAsmBugs()["ExplicitCO"])
            {
                setInst("v_add_i32");
            }
            else
            {
                setInst("v_add_i32");
            }
        }

        VAddI32(const VAddI32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VAddI32>(*this);
        }
    };

    struct VAddU32 : public CommonInstruction
    {
        VAddU32(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src0,
                const InstructionInput&           src1,
                const std::string&                comment = "")
            : CommonInstruction(InstType::INST_U32,
                                dst,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            if(getAsmBugs()["ExplicitNC"])
            {
                setInst("v_add_nc_u32");
                dst1 = nullptr;
            }
            else if(getAsmBugs()["ExplicitCO"])
            {
                setInst("v_add_u32");
                dst1 = nullptr;
            }
            else
            {
                setInst("v_add_u32");
                dst1 = std::make_shared<VCC>();
            }
        }

        VAddU32(const std::shared_ptr<Container>&    dst,
                const std::vector<InstructionInput>& srcs,
                const std::string&                   comment = "")
            : CommonInstruction(
                InstType::INST_U32, dst, srcs, std::nullopt, std::nullopt, std::nullopt, comment)
        {
            if(getAsmBugs()["ExplicitNC"])
            {
                setInst("v_add_nc_u32");
                dst1 = nullptr;
            }
            else if(getAsmBugs()["ExplicitCO"])
            {
                setInst("v_add_u32");
                dst1 = nullptr;
            }
            else
            {
                setInst("v_add_u32");
                dst1 = std::make_shared<VCC>();
            }
        }

        VAddU32(const VAddU32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VAddU32>(*this);
        }
    };

    struct VAddCOU32 : public CommonInstruction
    {
        VAddCOU32(const std::shared_ptr<Container>& dst,
                  const std::shared_ptr<Container>& dst1,
                  const InstructionInput&           src0,
                  const InstructionInput&           src1,
                  const std::string&                comment = "")
            : CommonInstruction(InstType::INST_U32,
                                dst,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            this->dst1 = dst1;
            if(getAsmBugs()["ExplicitCO"])
            {
                setInst("v_add_co_u32");
            }
            else
            {
                setInst("v_add_u32");
            }
        }

        VAddCOU32(const std::shared_ptr<Container>&    dst,
                  const std::shared_ptr<Container>&    dst1,
                  const std::vector<InstructionInput>& srcs,
                  const std::string&                   comment = "")
            : CommonInstruction(
                InstType::INST_U32, dst, srcs, std::nullopt, std::nullopt, std::nullopt, comment)
        {
            this->dst1 = dst1;
            if(getAsmBugs()["ExplicitCO"])
            {
                setInst("v_add_co_u32");
            }
            else
            {
                setInst("v_add_u32");
            }
        }

        VAddCOU32(const VAddCOU32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VAddCOU32>(*this);
        }
    };

    struct VAddCCOU32 : public CommonInstruction
    {
        VAddCCOU32(const std::shared_ptr<Container>& dst,
                   const std::shared_ptr<Container>& dst1,
                   const InstructionInput&           src0,
                   const InstructionInput&           src1,
                   const InstructionInput&           src2,
                   const std::string&                comment = "")
            : CommonInstruction(InstType::INST_U32,
                                dst,
                                {src0, src1, src2},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            this->dst1 = dst1;
            if(getAsmBugs()["ExplicitNC"])
            {
                setInst("v_add_co_ci_u32");
            }
            else if(getAsmBugs()["ExplicitCO"])
            {
                setInst("v_addc_co_u32");
            }
            else
            {
                setInst("v_addc_u32");
            }
        }

        VAddCCOU32(const std::shared_ptr<Container>&    dst,
                   const std::shared_ptr<Container>&    dst1,
                   const std::vector<InstructionInput>& srcs,
                   const std::string&                   comment = "")
            : CommonInstruction(
                InstType::INST_U32, dst, srcs, std::nullopt, std::nullopt, std::nullopt, comment)
        {
            this->dst1 = dst1;
            if(getAsmBugs()["ExplicitNC"])
            {
                setInst("v_add_co_ci_u32");
            }
            else if(getAsmBugs()["ExplicitCO"])
            {
                setInst("v_addc_co_u32");
            }
            else
            {
                setInst("v_addc_u32");
            }
        }

        VAddCCOU32(const VAddCCOU32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VAddCCOU32>(*this);
        }
    };

    struct VAddPKF16 : public CommonInstruction
    {
        VAddPKF16(const std::shared_ptr<Container>& dst,
                  const InstructionInput&           src0,
                  const InstructionInput&           src1,
                  std::optional<VOP3PModifiers>     vop3    = std::nullopt,
                  const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_F16, dst, {src0, src1}, std::nullopt, std::nullopt, vop3, comment)
        {
            setInst("v_pk_add_f16");
        }

        VAddPKF16(const VAddPKF16& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VAddPKF16>(*this);
        }
    };

    struct _VAddPKF32 : public CommonInstruction
    {
        _VAddPKF32(const std::shared_ptr<Container>& dst,
                   const InstructionInput&           src0,
                   const InstructionInput&           src1,
                   std::optional<VOP3PModifiers>     vop3    = std::nullopt,
                   const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_F32, dst, {src0, src1}, std::nullopt, std::nullopt, vop3, comment)
        {
            setInst("v_pk_add_f32");
        }

        _VAddPKF32(const std::shared_ptr<Container>&    dst,
                   const std::vector<InstructionInput>& srcs,
                   std::optional<VOP3PModifiers>        vop3    = std::nullopt,
                   const std::string&                   comment = "")
            : CommonInstruction(
                InstType::INST_F32, dst, srcs, std::nullopt, std::nullopt, vop3, comment)
        {
            setInst("v_pk_add_f32");
        }

        _VAddPKF32(const _VAddPKF32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<_VAddPKF32>(*this);
        }
    };

    struct VAddPKF32 : public CompositeInstruction
    {
        VAddPKF32(const std::shared_ptr<Container>& dst,
                  const InstructionInput&           src0,
                  const InstructionInput&           src1,
                  const std::string&                comment = "")
            : CompositeInstruction(InstType::INST_F32, dst, {src0, src1}, comment)
        {
            setInst("v_pk_add_f32");
        }

        std::vector<std::shared_ptr<Instruction>> setupInstructions() const override
        {
            std::vector<std::shared_ptr<Instruction>> instructions;
            if(getAsmCaps()["v_pk_add_f32"])
            {
                instructions = {std::make_shared<_VAddPKF32>(dst, srcs, std::nullopt, comment)};
            }
            else
            {
                auto [dst1, dst2]
                    = std::dynamic_pointer_cast<RegisterContainer>(dst)->splitRegContainer();
                std::vector<InstructionInput> srcs1;
                std::vector<InstructionInput> srcs2;
                splitSrcs(srcs, srcs1, srcs2);
                instructions
                    = {std::make_shared<VAddF32>(dst1, srcs1, std::nullopt, std::nullopt, comment),
                       std::make_shared<VAddF32>(dst2, srcs2, std::nullopt, std::nullopt, comment)};
            }
            return std::move(instructions);
        }

        VAddPKF32(const VAddPKF32& other)
            : CompositeInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VAddPKF32>(*this);
        }
    };

    struct VAdd3U32 : public CommonInstruction
    {
        VAdd3U32(const std::shared_ptr<Container>& dst,
                 const InstructionInput&           src0,
                 const InstructionInput&           src1,
                 const InstructionInput&           src2,
                 std::optional<VOP3PModifiers>     vop3    = std::nullopt,
                 const std::string&                comment = "")
            : CommonInstruction(InstType::INST_U32,
                                dst,
                                {src0, src1, src2},
                                std::nullopt,
                                std::nullopt,
                                vop3,
                                comment)
        {
            setInst("v_add3_u32");
        }

        VAdd3U32(const VAdd3U32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VAdd3U32>(*this);
        }
    };

    struct VMulF16 : public CommonInstruction
    {
        VMulF16(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src0,
                const InstructionInput&           src1,
                std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_F16, dst, {src0, src1}, std::nullopt, sdwa, std::nullopt, comment)
        {
            setInst("v_mul_f16");
        }

        VMulF16(const VMulF16& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VMulF16>(*this);
        }
    };

    struct VMulF32 : public CommonInstruction
    {
        VMulF32(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src0,
                const InstructionInput&           src1,
                std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_F32, dst, {src0, src1}, std::nullopt, sdwa, std::nullopt, comment)
        {
            setInst("v_mul_f32");
        }

        VMulF32(const std::shared_ptr<Container>&    dst,
                const std::vector<InstructionInput>& srcs,
                std::optional<SDWAModifiers>         sdwa    = std::nullopt,
                const std::string&                   comment = "")
            : CommonInstruction(
                InstType::INST_F32, dst, srcs, std::nullopt, sdwa, std::nullopt, comment)
        {
            setInst("v_mul_f32");
        }

        VMulF32(const VMulF32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VMulF32>(*this);
        }
    };

    struct VMulF64 : public CommonInstruction
    {
        VMulF64(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src0,
                const InstructionInput&           src1,
                std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_F64, dst, {src0, src1}, std::nullopt, sdwa, std::nullopt, comment)
        {
            setInst("v_mul_f64");
        }

        VMulF64(const VMulF64& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VMulF64>(*this);
        }
    };

    struct VMulPKF16 : public CommonInstruction
    {
        VMulPKF16(const std::shared_ptr<Container>& dst,
                  const InstructionInput&           src0,
                  const InstructionInput&           src1,
                  std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                  std::optional<VOP3PModifiers>     vop3    = std::nullopt,
                  const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_F16, dst, {src0, src1}, std::nullopt, sdwa, vop3, comment)
        {
            setInst("v_pk_mul_f16");
        }

        VMulPKF16(const VMulPKF16& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VMulPKF16>(*this);
        }
    };

    struct VMulPKF32S : public CommonInstruction
    {
        VMulPKF32S(const std::shared_ptr<Container>& dst,
                   const InstructionInput&           src0,
                   const InstructionInput&           src1,
                   std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                   std::optional<VOP3PModifiers>     vop3    = std::nullopt,
                   const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_F32, dst, {src0, src1}, std::nullopt, sdwa, vop3, comment)
        {
            setInst("v_pk_mul_f32");
        }

        VMulPKF32S(const VMulPKF32S& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VMulPKF32S>(*this);
        }
    };

    struct _VMulPKF32 : public CommonInstruction
    {
        _VMulPKF32(const std::shared_ptr<Container>& dst,
                   const InstructionInput&           src0,
                   const InstructionInput&           src1,
                   std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                   std::optional<VOP3PModifiers>     vop3    = std::nullopt,
                   const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_F32, dst, {src0, src1}, std::nullopt, sdwa, vop3, comment)
        {
            setInst("v_pk_mul_f32");
        }

        _VMulPKF32(const std::shared_ptr<Container>&    dst,
                   const std::vector<InstructionInput>& srcs,
                   std::optional<SDWAModifiers>         sdwa    = std::nullopt,
                   std::optional<VOP3PModifiers>        vop3    = std::nullopt,
                   const std::string&                   comment = "")
            : CommonInstruction(InstType::INST_F32, dst, srcs, std::nullopt, sdwa, vop3, comment)
        {
            setInst("v_pk_mul_f32");
        }

        _VMulPKF32(const _VMulPKF32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<_VMulPKF32>(*this);
        }
    };

    struct VMulPKF32 : public CompositeInstruction
    {
        VMulPKF32(const std::shared_ptr<Container>& dst,
                  const InstructionInput&           src0,
                  const InstructionInput&           src1,
                  std::optional<VOP3PModifiers>     vop3    = std::nullopt,
                  const std::string&                comment = "")
            : CompositeInstruction(InstType::INST_F32, dst, {src0, src1}, comment)
            , vop3(vop3)
        {
            setInst("v_pk_mul_f32");
        }

        std::vector<std::shared_ptr<Instruction>> setupInstructions() const override
        {
            std::vector<std::shared_ptr<Instruction>> instructions;
            if(getAsmCaps()["v_pk_mul_f32"])
            {
                instructions
                    = {std::make_shared<_VMulPKF32>(dst, srcs, std::nullopt, vop3, comment)};
            }
            else
            {
                auto [dst1, dst2]
                    = std::dynamic_pointer_cast<RegisterContainer>(dst)->splitRegContainer();
                std::vector<InstructionInput> srcs1;
                std::vector<InstructionInput> srcs2;
                splitSrcs(srcs, srcs1, srcs2);
                if(!vop3)
                {
                    instructions = {std::make_shared<VMulF32>(dst1, srcs1, std::nullopt, comment),
                                    std::make_shared<VMulF32>(dst2, srcs2, std::nullopt, comment)};
                }
                else
                {
                    if(vop3->op_sel.size() > 0)
                    {
                        assert(vop3->op_sel.size() == 3);
                    }
                    if(vop3->op_sel_hi.size() > 0)
                    {
                        assert(vop3->op_sel_hi.size() == 3);
                    }
                    if(!vop3->byte_sel.empty())
                    {
                        throw std::runtime_error("Byte sel not supported");
                    }
                    auto lowDst = !vop3->op_sel.empty() && vop3->op_sel[2] == 1 ? dst2 : dst1;
                    auto lowSrc1
                        = !vop3->op_sel.empty() && vop3->op_sel[0] == 1 ? srcs2[0] : srcs1[0];
                    auto lowSrc2
                        = !vop3->op_sel.empty() && vop3->op_sel[1] == 1 ? srcs2[1] : srcs1[1];
                    auto highDst
                        = !vop3->op_sel_hi.empty() && vop3->op_sel_hi[2] == 0 ? dst1 : dst2;
                    auto highSrc1
                        = !vop3->op_sel_hi.empty() && vop3->op_sel_hi[0] == 0 ? srcs1[0] : srcs2[0];
                    auto highSrc2
                        = !vop3->op_sel_hi.empty() && vop3->op_sel_hi[1] == 0 ? srcs1[1] : srcs2[1];
                    std::vector<InstructionInput> lowSrcs  = {lowSrc1, lowSrc2};
                    std::vector<InstructionInput> highSrcs = {highSrc1, highSrc2};
                    instructions
                        = {std::make_shared<VMulF32>(lowDst, lowSrcs, std::nullopt, comment),
                           std::make_shared<VMulF32>(highDst, highSrcs, std::nullopt, comment)};
                }
            }
            return std::move(instructions);
        }

        VMulPKF32(const VMulPKF32& other)
            : CompositeInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VMulPKF32>(*this);
        }

    private:
        std::optional<VOP3PModifiers> vop3;
    };

    struct VMulLOU32 : public CommonInstruction
    {
        VMulLOU32(const std::shared_ptr<Container>& dst,
                  const InstructionInput&           src0,
                  const InstructionInput&           src1,
                  const std::string&                comment = "")
            : CommonInstruction(InstType::INST_LO_U32,
                                dst,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("v_mul_lo_u32");
        }

        VMulLOU32(const VMulLOU32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VMulLOU32>(*this);
        }
    };

    struct VMulHII32 : public CommonInstruction
    {
        VMulHII32(const std::shared_ptr<Container>& dst,
                  const InstructionInput&           src0,
                  const InstructionInput&           src1,
                  const std::string&                comment = "")
            : CommonInstruction(InstType::INST_HI_I32,
                                dst,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("v_mul_hi_i32");
        }

        VMulHII32(const VMulHII32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VMulHII32>(*this);
        }
    };

    struct VMulHIU32 : public CommonInstruction
    {
        VMulHIU32(const std::shared_ptr<Container>& dst,
                  const InstructionInput&           src0,
                  const InstructionInput&           src1,
                  const std::string&                comment = "")
            : CommonInstruction(InstType::INST_HI_U32,
                                dst,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("v_mul_hi_u32");
        }

        VMulHIU32(const VMulHIU32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VMulHIU32>(*this);
        }
    };

    struct VMulI32I24 : public CommonInstruction
    {
        VMulI32I24(const std::shared_ptr<Container>& dst,
                   const InstructionInput&           src0,
                   const InstructionInput&           src1,
                   const std::string&                comment = "")
            : CommonInstruction(InstType::INST_I32,
                                dst,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("v_mul_i32_i24");
        }

        VMulI32I24(const VMulI32I24& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VMulI32I24>(*this);
        }
    };

    struct VMulU32U24 : public CommonInstruction
    {
        VMulU32U24(const std::shared_ptr<Container>& dst,
                   const InstructionInput&           src0,
                   const InstructionInput&           src1,
                   const std::string&                comment = "")
            : CommonInstruction(InstType::INST_U32,
                                dst,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("v_mul_u32_u24");
        }

        VMulU32U24(const VMulU32U24& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VMulU32U24>(*this);
        }
    };

    struct VSubF32 : public CommonInstruction
    {
        VSubF32(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src0,
                const InstructionInput&           src1,
                std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_F32, dst, {src0, src1}, std::nullopt, sdwa, std::nullopt, comment)
        {
            setInst("v_sub_f32");
        }

        VSubF32(const VSubF32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VSubF32>(*this);
        }
    };

    struct VSubI32 : public CommonInstruction
    {
        VSubI32(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src0,
                const InstructionInput&           src1,
                const std::string&                comment = "")
            : CommonInstruction(InstType::INST_I32,
                                dst,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            if(getAsmBugs()["ExplicitNC"])
            {
                setInst("v_sub_nc_i32");
            }
            else if(getAsmBugs()["ExplicitCO"])
            {
                setInst("v_sub_i32");
            }
            else
            {
                setInst("v_sub_i32");
            }
        }

        VSubI32(const VSubI32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VSubI32>(*this);
        }
    };

    struct VSubU32 : public CommonInstruction
    {
        VSubU32(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src0,
                const InstructionInput&           src1,
                const std::string&                comment = "")
            : CommonInstruction(InstType::INST_U32,
                                dst,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            if(getAsmBugs()["ExplicitNC"])
            {
                setInst("v_sub_nc_u32");
            }
            else if(getAsmBugs()["ExplicitCO"])
            {
                setInst("v_sub_u32");
            }
            else
            {
                setInst("v_sub_u32");
            }
        }

        VSubU32(const VSubU32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VSubU32>(*this);
        }
    };

    struct VSubCoU32 : public CommonInstruction
    {
        VSubCoU32(const std::shared_ptr<Container>& dst,
                  const std::shared_ptr<Container>& dst1,
                  const InstructionInput&           src0,
                  const InstructionInput&           src1,
                  const std::string&                comment = "")
            : CommonInstruction(InstType::INST_U32,
                                dst,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            this->dst1 = dst1;
            if(getAsmBugs()["ExplicitCO"])
            {
                setInst("v_sub_co_u32");
            }
            else
            {
                setInst("v_sub_u32");
            }
        }

        VSubCoU32(const VSubCoU32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VSubCoU32>(*this);
        }
    };

    struct VMacF32 : public CommonInstruction
    {
        VMacF32(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src0,
                const InstructionInput&           src1,
                std::optional<VOP3PModifiers>     vop3    = std::nullopt,
                const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_F32, dst, {src0, src1}, std::nullopt, std::nullopt, vop3, comment)
            , addDstToSrc(false)
        {
            if(getAsmCaps()["v_fmac_f32"])
            {
                setInst("v_fmac_f32");
            }
            else if(getAsmCaps()["v_fma_f32"])
            {
                addDstToSrc = true;
                setInst("v_fmac_f32");
            }
            else if(getAsmCaps()["v_mac_f32"])
            {
                setInst("v_mac_f32");
            }
            else
            {
                throw std::runtime_error("FMA and MAC instructions are not supported.");
            }
        }

        std::string getArgStr() const override
        {
            std::string kStr = CommonInstruction::getArgStr();
            if(addDstToSrc)
            {
                kStr += ", " + dst->toString();
            }
            return kStr;
        }

        VMacF32(const VMacF32& other)
            : CommonInstruction(other)
            , addDstToSrc(other.addDstToSrc)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VMacF32>(*this);
        }

    private:
        bool addDstToSrc;
    };

    struct VDot2CF32F16 : public CommonInstruction
    {
        VDot2CF32F16(const std::shared_ptr<Container>& dst,
                     const InstructionInput&           src0,
                     const InstructionInput&           src1,
                     std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                     const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_F32, dst, {src0, src1}, std::nullopt, sdwa, std::nullopt, comment)
        {
            if(kernel().isaVersion[0] >= 11)
            {
                setInst("v_dot2acc_f32_f16");
            }
            else
            {
                setInst("v_dot2c_f32_f16");
            }
        }

        VDot2CF32F16(const VDot2CF32F16& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VDot2CF32F16>(*this);
        }
    };

    struct VDot2F32F16 : public CommonInstruction
    {
        VDot2F32F16(const std::shared_ptr<Container>& dst,
                    const InstructionInput&           src0,
                    const InstructionInput&           src1,
                    const InstructionInput&           src2,
                    std::optional<VOP3PModifiers>     vop3    = std::nullopt,
                    const std::string&                comment = "")
            : CommonInstruction(InstType::INST_F32,
                                dst,
                                {src0, src1, src2},
                                std::nullopt,
                                std::nullopt,
                                vop3,
                                comment)
        {
            setInst("v_dot2_f32_f16");
        }

        VDot2F32F16(const VDot2F32F16& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VDot2F32F16>(*this);
        }
    };

    struct VDot2F32BF16 : public CommonInstruction
    {
        VDot2F32BF16(const std::shared_ptr<Container>& dst,
                     const InstructionInput&           src0,
                     const InstructionInput&           src1,
                     const InstructionInput&           src2,
                     std::optional<VOP3PModifiers>     vop3    = std::nullopt,
                     const std::string&                comment = "")
            : CommonInstruction(InstType::INST_F32,
                                dst,
                                {src0, src1, src2},
                                std::nullopt,
                                std::nullopt,
                                vop3,
                                comment)
        {
            setInst("v_dot2_f32_bf16");
        }

        VDot2F32BF16(const VDot2F32BF16& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VDot2F32BF16>(*this);
        }
    };

    struct VDot2CF32BF16 : public CommonInstruction
    {
        VDot2CF32BF16(const std::shared_ptr<Container>& dst,
                      const InstructionInput&           src0,
                      const InstructionInput&           src1,
                      std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                      const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_F32, dst, {src0, src1}, std::nullopt, sdwa, std::nullopt, comment)
        {
            if(kernel().isaVersion[0] >= 11)
            {
                setInst("v_dot2acc_f32_bf16");
            }
            else
            {
                setInst("v_dot2c_f32_bf16");
            }
        }

        VDot2CF32BF16(const VDot2CF32BF16& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VDot2CF32BF16>(*this);
        }
    };

    struct VFmaF16 : public CommonInstruction
    {
        VFmaF16(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src0,
                const InstructionInput&           src1,
                const InstructionInput&           src2,
                std::optional<VOP3PModifiers>     vop3    = std::nullopt,
                const std::string&                comment = "")
            : CommonInstruction(InstType::INST_F16,
                                dst,
                                {src0, src1, src2},
                                std::nullopt,
                                std::nullopt,
                                vop3,
                                comment)
        {
            setInst("v_fma_f16");
        }

        VFmaF16(const VFmaF16& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VFmaF16>(*this);
        }
    };

    struct VFmaF32 : public CommonInstruction
    {
        VFmaF32(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src0,
                const InstructionInput&           src1,
                const InstructionInput&           src2,
                std::optional<VOP3PModifiers>     vop3    = std::nullopt,
                const std::string&                comment = "")
            : CommonInstruction(InstType::INST_F32,
                                dst,
                                {src0, src1, src2},
                                std::nullopt,
                                std::nullopt,
                                vop3,
                                comment)
        {
            setInst("v_fma_f32");
        }

        VFmaF32(const VFmaF32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VFmaF32>(*this);
        }
    };

    struct VFmaF64 : public CommonInstruction
    {
        VFmaF64(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src0,
                const InstructionInput&           src1,
                const InstructionInput&           src2,
                std::optional<VOP3PModifiers>     vop3    = std::nullopt,
                const std::string&                comment = "")
            : CommonInstruction(InstType::INST_F64,
                                dst,
                                {src0, src1, src2},
                                std::nullopt,
                                std::nullopt,
                                vop3,
                                comment)
        {
            setInst("v_fma_f64");
        }

        VFmaF64(const VFmaF64& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VFmaF64>(*this);
        }
    };

    struct VFmaPKF16 : public CommonInstruction
    {
        VFmaPKF16(const std::shared_ptr<Container>& dst,
                  const InstructionInput&           src0,
                  const InstructionInput&           src1,
                  const InstructionInput&           src2,
                  std::optional<VOP3PModifiers>     vop3    = std::nullopt,
                  const std::string&                comment = "")
            : CommonInstruction(InstType::INST_F16,
                                dst,
                                {src0, src1, src2},
                                std::nullopt,
                                std::nullopt,
                                vop3,
                                comment)
        {
            setInst("v_pk_fma_f16");
        }

        VFmaPKF16(const VFmaPKF16& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VFmaPKF16>(*this);
        }
    };

    struct VFmaMixF32 : public CommonInstruction
    {
        VFmaMixF32(const std::shared_ptr<Container>& dst,
                   const InstructionInput&           src0,
                   const InstructionInput&           src1,
                   const InstructionInput&           src2,
                   std::optional<VOP3PModifiers>     vop3    = std::nullopt,
                   const std::string&                comment = "")
            : CommonInstruction(InstType::INST_F32,
                                dst,
                                {src0, src1, src2},
                                std::nullopt,
                                std::nullopt,
                                vop3,
                                comment)
        {
            setInst("v_fma_mix_f32");
        }

        VFmaMixF32(const VFmaMixF32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VFmaMixF32>(*this);
        }
    };

    struct VMadI32I24 : public CommonInstruction
    {
        VMadI32I24(const std::shared_ptr<Container>& dst,
                   const InstructionInput&           src0,
                   const InstructionInput&           src1,
                   const InstructionInput&           src2,
                   std::optional<VOP3PModifiers>     vop3    = std::nullopt,
                   const std::string&                comment = "")
            : CommonInstruction(InstType::INST_I32,
                                dst,
                                {src0, src1, src2},
                                std::nullopt,
                                std::nullopt,
                                vop3,
                                comment)
        {
            setInst("v_mad_i32_i24");
        }

        VMadI32I24(const VMadI32I24& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VMadI32I24>(*this);
        }
    };

    struct VMadU32U24 : public CommonInstruction
    {
        VMadU32U24(const std::shared_ptr<Container>& dst,
                   const InstructionInput&           src0,
                   const InstructionInput&           src1,
                   const InstructionInput&           src2,
                   std::optional<VOP3PModifiers>     vop3    = std::nullopt,
                   const std::string&                comment = "")
            : CommonInstruction(InstType::INST_U32,
                                dst,
                                {src0, src1, src2},
                                std::nullopt,
                                std::nullopt,
                                vop3,
                                comment)
        {
            setInst("v_mad_u32_u24");
        }

        VMadU32U24(const VMadU32U24& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VMadU32U24>(*this);
        }
    };

    struct VMadMixF32 : public CommonInstruction
    {
        VMadMixF32(const std::shared_ptr<Container>& dst,
                   const InstructionInput&           src0,
                   const InstructionInput&           src1,
                   const InstructionInput&           src2,
                   std::optional<VOP3PModifiers>     vop3    = std::nullopt,
                   const std::string&                comment = "")
            : CommonInstruction(InstType::INST_F32,
                                dst,
                                {src0, src1, src2},
                                std::nullopt,
                                std::nullopt,
                                vop3,
                                comment)
        {
            setInst("v_mad_mix_f32");
        }

        VMadMixF32(const VMadMixF32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VMadMixF32>(*this);
        }
    };

    struct VExpF16 : public CommonInstruction
    {
        VExpF16(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src,
                std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_F16, dst, {src}, std::nullopt, sdwa, std::nullopt, comment)
        {
            setInst("v_exp_f16");
        }

        VExpF16(const VExpF16& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VExpF16>(*this);
        }
    };

    struct VExpF32 : public CommonInstruction
    {
        VExpF32(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src,
                const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_F32, dst, {src}, std::nullopt, std::nullopt, std::nullopt, comment)
        {
            setInst("v_exp_f32");
        }

        VExpF32(const VExpF32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VExpF32>(*this);
        }
    };

    struct VRcpF16 : public CommonInstruction
    {
        VRcpF16(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src,
                std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_F16, dst, {src}, std::nullopt, sdwa, std::nullopt, comment)
        {
            setInst("v_rcp_f16");
        }

        VRcpF16(const VRcpF16& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VRcpF16>(*this);
        }
    };

    struct VRcpF32 : public CommonInstruction
    {
        VRcpF32(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src,
                const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_F32, dst, {src}, std::nullopt, std::nullopt, std::nullopt, comment)
        {
            setInst("v_rcp_f32");
        }

        VRcpF32(const VRcpF32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VRcpF32>(*this);
        }
    };

    struct VRcpIFlagF32 : public CommonInstruction
    {
        VRcpIFlagF32(const std::shared_ptr<Container>& dst,
                     const InstructionInput&           src,
                     const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_F32, dst, {src}, std::nullopt, std::nullopt, std::nullopt, comment)
        {
            setInst("v_rcp_iflag_f32");
        }

        VRcpIFlagF32(const VRcpIFlagF32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VRcpIFlagF32>(*this);
        }
    };

    struct VRcpF64 : public CommonInstruction
    {
        VRcpF64(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src,
                const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_F64, dst, {src}, std::nullopt, std::nullopt, std::nullopt, comment)
        {
            setInst("v_rcp_f64");
        }

        VRcpF64(const VRcpF64& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VRcpF64>(*this);
        }
    };

    struct VRsqF16 : public CommonInstruction
    {
        VRsqF16(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src,
                std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_F16, dst, {src}, std::nullopt, sdwa, std::nullopt, comment)
        {
            setInst("v_rsq_f16");
        }

        VRsqF16(const VRsqF16& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VRsqF16>(*this);
        }
    };

    struct VRsqF32 : public CommonInstruction
    {
        VRsqF32(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src,
                const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_F32, dst, {src}, std::nullopt, std::nullopt, std::nullopt, comment)
        {
            setInst("v_rsq_f32");
        }

        VRsqF32(const VRsqF32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VRsqF32>(*this);
        }
    };

    struct VRsqIFlagF32 : public CommonInstruction
    {
        VRsqIFlagF32(const std::shared_ptr<Container>& dst,
                     const InstructionInput&           src,
                     const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_F32, dst, {src}, std::nullopt, std::nullopt, std::nullopt, comment)
        {
            setInst("v_rsq_iflag_f32");
        }

        VRsqIFlagF32(const VRsqIFlagF32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VRsqIFlagF32>(*this);
        }
    };

    struct VMaxF16 : public CommonInstruction
    {
        VMaxF16(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src0,
                const InstructionInput&           src1,
                std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_F16, dst, {src0, src1}, std::nullopt, sdwa, std::nullopt, comment)
        {
            setInst("v_max_f16");
        }

        VMaxF16(const VMaxF16& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VMaxF16>(*this);
        }
    };

    struct VMaxF32 : public CommonInstruction
    {
        VMaxF32(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src0,
                const InstructionInput&           src1,
                std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_F32, dst, {src0, src1}, std::nullopt, sdwa, std::nullopt, comment)
        {
            setInst("v_max_f32");
        }

        VMaxF32(const VMaxF32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VMaxF32>(*this);
        }
    };

    struct VMaxF64 : public CommonInstruction
    {
        VMaxF64(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src0,
                const InstructionInput&           src1,
                std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_F64, dst, {src0, src1}, std::nullopt, sdwa, std::nullopt, comment)
        {
            setInst("v_max_f64");
        }

        VMaxF64(const VMaxF64& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VMaxF64>(*this);
        }
    };

    struct VMaxI32 : public CommonInstruction
    {
        VMaxI32(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src0,
                const InstructionInput&           src1,
                std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_I32, dst, {src0, src1}, std::nullopt, sdwa, std::nullopt, comment)
        {
            setInst("v_max_i32");
        }

        VMaxI32(const VMaxI32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VMaxI32>(*this);
        }
    };

    struct VMaxPKF16 : public CommonInstruction
    {
        VMaxPKF16(const std::shared_ptr<Container>& dst,
                  const InstructionInput&           src0,
                  const InstructionInput&           src1,
                  std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                  std::optional<VOP3PModifiers>     vop3    = std::nullopt,
                  const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_F16, dst, {src0, src1}, std::nullopt, sdwa, vop3, comment)
        {
            setInst("v_pk_max_f16");
        }

        VMaxPKF16(const VMaxPKF16& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VMaxPKF16>(*this);
        }
    };

    struct VMed3I32 : public CommonInstruction
    {
        VMed3I32(const std::shared_ptr<Container>& dst,
                 const InstructionInput&           src0,
                 const InstructionInput&           src1,
                 const InstructionInput&           src2,
                 const std::string&                comment = "")
            : CommonInstruction(InstType::INST_I32,
                                dst,
                                {src0, src1, src2},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("v_med3_i32");
        }

        VMed3I32(const VMed3I32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VMed3I32>(*this);
        }
    };

    struct VMed3F32 : public CommonInstruction
    {
        VMed3F32(const std::shared_ptr<Container>& dst,
                 const InstructionInput&           src0,
                 const InstructionInput&           src1,
                 const InstructionInput&           src2,
                 const std::string&                comment = "")
            : CommonInstruction(InstType::INST_F32,
                                dst,
                                {src0, src1, src2},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("v_med3_f32");
        }

        VMed3F32(const VMed3F32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VMed3F32>(*this);
        }
    };

    struct VMinF16 : public CommonInstruction
    {
        VMinF16(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src0,
                const InstructionInput&           src1,
                std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_F16, dst, {src0, src1}, std::nullopt, sdwa, std::nullopt, comment)
        {
            setInst("v_min_f16");
        }

        VMinF16(const VMinF16& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VMinF16>(*this);
        }
    };

    struct VMinF32 : public CommonInstruction
    {
        VMinF32(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src0,
                const InstructionInput&           src1,
                std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_F32, dst, {src0, src1}, std::nullopt, sdwa, std::nullopt, comment)
        {
            setInst("v_min_f32");
        }

        VMinF32(const VMinF32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VMinF32>(*this);
        }
    };

    struct VMinF64 : public CommonInstruction
    {
        VMinF64(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src0,
                const InstructionInput&           src1,
                std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_F64, dst, {src0, src1}, std::nullopt, sdwa, std::nullopt, comment)
        {
            setInst("v_min_f64");
        }

        VMinF64(const VMinF64& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VMinF64>(*this);
        }
    };

    struct VMinI32 : public CommonInstruction
    {
        VMinI32(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src0,
                const InstructionInput&           src1,
                std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_I32, dst, {src0, src1}, std::nullopt, sdwa, std::nullopt, comment)
        {
            setInst("v_min_i32");
        }

        VMinI32(const VMinI32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VMinI32>(*this);
        }
    };

    struct VAndB32 : public CommonInstruction
    {
        VAndB32(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src0,
                const InstructionInput&           src1,
                const std::string&                comment = "")
            : CommonInstruction(InstType::INST_B32,
                                dst,
                                {src0, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("v_and_b32");
        }

        VAndB32(const VAndB32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VAndB32>(*this);
        }
    };

    struct VAndOrB32 : public CommonInstruction
    {
        VAndOrB32(const std::shared_ptr<Container>& dst,
                  const InstructionInput&           src0,
                  const InstructionInput&           src1,
                  const InstructionInput&           src2,
                  const std::string&                comment = "")
            : CommonInstruction(InstType::INST_B32,
                                dst,
                                {src0, src1, src2},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("v_and_or_b32");
        }

        VAndOrB32(const VAndOrB32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VAndOrB32>(*this);
        }
    };

    struct VNotB32 : public CommonInstruction
    {
        VNotB32(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src,
                const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_B32, dst, {src}, std::nullopt, std::nullopt, std::nullopt, comment)
        {
            setInst("v_not_b32");
        }

        VNotB32(const VNotB32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VNotB32>(*this);
        }
    };

    struct VOrB32 : public CommonInstruction
    {
        VOrB32(const std::shared_ptr<Container>& dst,
               const InstructionInput&           src0,
               const InstructionInput&           src1,
               std::optional<SDWAModifiers>      sdwa    = std::nullopt,
               const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_B32, dst, {src0, src1}, std::nullopt, sdwa, std::nullopt, comment)
        {
            setInst("v_or_b32");
        }

        VOrB32(const std::shared_ptr<Container>&    dst,
               const std::vector<InstructionInput>& srcs,
               std::optional<SDWAModifiers>         sdwa    = std::nullopt,
               const std::string&                   comment = "")
            : CommonInstruction(
                InstType::INST_B32, dst, srcs, std::nullopt, sdwa, std::nullopt, comment)
        {
            setInst("v_or_b32");
        }

        VOrB32(const VOrB32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VOrB32>(*this);
        }
    };

    struct VXorB32 : public CommonInstruction
    {
        VXorB32(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src0,
                const InstructionInput&           src1,
                std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_B32, dst, {src0, src1}, std::nullopt, sdwa, std::nullopt, comment)
        {
            setInst("v_xor_b32");
        }

        VXorB32(const VXorB32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VXorB32>(*this);
        }
    };

    struct VPrngB32 : public CommonInstruction
    {
        VPrngB32(const std::shared_ptr<Container>& dst,
                 const InstructionInput&           src,
                 const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_B32, dst, {src}, std::nullopt, std::nullopt, std::nullopt, comment)
        {
            setInst("v_prng_b32");
        }

        VPrngB32(const VPrngB32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VPrngB32>(*this);
        }
    };

    struct VCndMaskB32 : public CommonInstruction
    {
        VCndMaskB32(const std::shared_ptr<Container>& dst,
                    const InstructionInput&           src0,
                    const InstructionInput&           src1,
                    const std::shared_ptr<Container>& src2    = std::make_shared<VCC>(),
                    std::optional<SDWAModifiers>      sdwa    = std::nullopt,
                    const std::string&                comment = "")
            : CommonInstruction(InstType::INST_B32,
                                dst,
                                {src0, src1, src2},
                                std::nullopt,
                                sdwa,
                                std::nullopt,
                                comment)
        {
            setInst("v_cndmask_b32");
        }

        VCndMaskB32(const VCndMaskB32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VCndMaskB32>(*this);
        }
    };

    struct VLShiftLeftB16 : public CommonInstruction
    {
        VLShiftLeftB16(const std::shared_ptr<Container>& dst,
                       const InstructionInput&           shiftHex,
                       const InstructionInput&           src,
                       const std::string&                comment = "")
            : CommonInstruction(InstType::INST_B16,
                                dst,
                                {shiftHex, src},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("v_lshlrev_b16");
        }

        VLShiftLeftB16(const VLShiftLeftB16& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VLShiftLeftB16>(*this);
        }
    };

    struct VLShiftLeftB32 : public CommonInstruction
    {
        VLShiftLeftB32(const std::shared_ptr<Container>& dst,
                       const InstructionInput&           shiftHex,
                       const InstructionInput&           src,
                       const std::string&                comment = "")
            : CommonInstruction(InstType::INST_B32,
                                dst,
                                {shiftHex, src},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("v_lshlrev_b32");
        }

        VLShiftLeftB32(const std::shared_ptr<Container>&    dst,
                       const std::vector<InstructionInput>& srcs,
                       const std::string&                   comment = "")
            : CommonInstruction(
                InstType::INST_B32, dst, srcs, std::nullopt, std::nullopt, std::nullopt, comment)
        {
            setInst("v_lshlrev_b32");
        }

        VLShiftLeftB32(const VLShiftLeftB32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VLShiftLeftB32>(*this);
        }
    };

    struct VLShiftRightB32 : public CommonInstruction
    {
        VLShiftRightB32(const std::shared_ptr<Container>& dst,
                        const InstructionInput&           shiftHex,
                        const InstructionInput&           src,
                        const std::string&                comment = "")
            : CommonInstruction(InstType::INST_B32,
                                dst,
                                {shiftHex, src},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("v_lshrrev_b32");
        }

        VLShiftRightB32(const VLShiftRightB32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VLShiftRightB32>(*this);
        }
    };

    struct VLShiftLeftB64 : public CommonInstruction
    {
        VLShiftLeftB64(const std::shared_ptr<Container>& dst,
                       const InstructionInput&           shiftHex,
                       const InstructionInput&           src,
                       const std::string&                comment = "")
            : CommonInstruction(InstType::INST_B64,
                                dst,
                                {shiftHex, src},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("v_lshlrev_b64");
        }

        VLShiftLeftB64(const VLShiftLeftB64& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VLShiftLeftB64>(*this);
        }
    };

    struct VLShiftRightB64 : public CommonInstruction
    {
        VLShiftRightB64(const std::shared_ptr<Container>& dst,
                        const InstructionInput&           shiftHex,
                        const InstructionInput&           src,
                        const std::string&                comment = "")
            : CommonInstruction(InstType::INST_B64,
                                dst,
                                {shiftHex, src},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("v_lshrrev_b64");
        }

        VLShiftRightB64(const VLShiftRightB64& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VLShiftRightB64>(*this);
        }
    };

    struct _VLShiftLeftOrB32 : public CommonInstruction
    {
        _VLShiftLeftOrB32(const std::shared_ptr<Container>& dst,
                          const InstructionInput&           shiftHex,
                          const InstructionInput&           src0,
                          const InstructionInput&           src1,
                          const std::string&                comment)
            : CommonInstruction(InstType::INST_B32,
                                dst,
                                {src0, shiftHex, src1},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("v_lshl_or_b32");
        }

        _VLShiftLeftOrB32(const std::shared_ptr<Container>&    dst,
                          const std::vector<InstructionInput>& srcs,
                          const std::string&                   comment)
            : CommonInstruction(
                InstType::INST_B32, dst, srcs, std::nullopt, std::nullopt, std::nullopt, comment)
        {
            setInst("v_lshl_or_b32");
        }

        _VLShiftLeftOrB32(const _VLShiftLeftOrB32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<_VLShiftLeftOrB32>(*this);
        }
    };

    struct VAShiftRightI32 : public CommonInstruction
    {
        VAShiftRightI32(const std::shared_ptr<Container>& dst,
                        const InstructionInput&           shiftHex,
                        const InstructionInput&           src,
                        const std::string&                comment)
            : CommonInstruction(InstType::INST_I32,
                                dst,
                                {shiftHex, src},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("v_ashrrev_i32");
        }

        VAShiftRightI32(const VAShiftRightI32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VAShiftRightI32>(*this);
        }
    };

    struct VLShiftLeftOrB32 : public CompositeInstruction
    {
        VLShiftLeftOrB32(const std::shared_ptr<Container>& dst,
                         const InstructionInput&           shiftHex,
                         const InstructionInput&           src0,
                         const InstructionInput&           src1,
                         const std::string&                comment = "")
            : CompositeInstruction(InstType::INST_B32, dst, {shiftHex, src0, src1}, comment)
        {
        }

        std::vector<std::shared_ptr<Instruction>> setupInstructions() const override
        {
            std::vector<std::shared_ptr<Instruction>> instructions;
            if(getAsmCaps()["HasLshlOr"])
            {
                std::vector<InstructionInput> srcs1 = {srcs[1], srcs[0], srcs[2]};
                instructions = {std::make_shared<_VLShiftLeftOrB32>(dst, srcs1, comment)};
            }
            else
            {
                std::vector<InstructionInput> srcs1 = {srcs[0], srcs[1]};
                std::vector<InstructionInput> srcs2 = {dst, srcs[2]};
                instructions = {std::make_shared<VLShiftLeftB32>(dst, srcs1, comment),
                                std::make_shared<VOrB32>(dst, srcs2, std::nullopt, comment)};
            }
            return std::move(instructions);
        }

        VLShiftLeftOrB32(const VLShiftLeftOrB32& other)
            : CompositeInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VLShiftLeftOrB32>(*this);
        }
    };

    struct _VAddLShiftLeftU32 : public CommonInstruction
    {
        _VAddLShiftLeftU32(const std::shared_ptr<Container>& dst,
                           const InstructionInput&           shiftHex,
                           const InstructionInput&           src0,
                           const InstructionInput&           src1,
                           const std::string&                comment = "")
            : CommonInstruction(InstType::INST_U32,
                                dst,
                                {src0, src1, shiftHex},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("v_add_lshl_u32");
        }

        _VAddLShiftLeftU32(const std::shared_ptr<Container>&    dst,
                           const std::vector<InstructionInput>& srcs,
                           const std::string&                   comment = "")
            : CommonInstruction(
                InstType::INST_U32, dst, srcs, std::nullopt, std::nullopt, std::nullopt, comment)
        {
            setInst("v_add_lshl_u32");
        }

        _VAddLShiftLeftU32(const _VAddLShiftLeftU32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<_VAddLShiftLeftU32>(*this);
        }
    };

    struct VAddLShiftLeftU32 : public CompositeInstruction
    {
        VAddLShiftLeftU32(const std::shared_ptr<Container>& dst,
                          const InstructionInput&           shiftHex,
                          const InstructionInput&           src0,
                          const InstructionInput&           src1,
                          const std::string&                comment = "")
            : CompositeInstruction(InstType::INST_U32, dst, {src0, src1, shiftHex}, comment)
        {
            setInst("v_add_lshl_u32");
        }

        std::vector<std::shared_ptr<Instruction>> setupInstructions() const override
        {
            std::vector<std::shared_ptr<Instruction>> instructions;
            if(getAsmCaps()["HasAddLshl"])
            {
                instructions = {std::make_shared<_VAddLShiftLeftU32>(dst, srcs, comment)};
            }
            else
            {
                std::vector<InstructionInput> srcs1 = {srcs[0], srcs[1]};
                std::vector<InstructionInput> srcs2 = {srcs[2], dst};
                if(getAsmBugs()["ExplicitCO"])
                {
                    instructions = {
                        std::make_shared<VAddCCOU32>(dst, std::make_shared<VCC>(), srcs1, comment),
                        std::make_shared<VLShiftLeftB32>(dst, srcs2, comment)};
                }
                else
                {
                    instructions = {std::make_shared<VAddU32>(dst, srcs1, comment),
                                    std::make_shared<VLShiftLeftB32>(dst, srcs2, comment)};
                }
            }
            return std::move(instructions);
        }

        VAddLShiftLeftU32(const VAddLShiftLeftU32& other)
            : CompositeInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VAddLShiftLeftU32>(*this);
        }
    };

    struct _VLShiftLeftAddU32 : public CommonInstruction
    {
        _VLShiftLeftAddU32(const std::shared_ptr<Container>& dst,
                           const InstructionInput&           shiftHex,
                           const InstructionInput&           src0,
                           const InstructionInput&           src1,
                           std::optional<VOP3PModifiers>     vop3    = std::nullopt,
                           const std::string&                comment = "")
            : CommonInstruction(InstType::INST_U32,
                                dst,
                                {src0, shiftHex, src1},
                                std::nullopt,
                                std::nullopt,
                                vop3,
                                comment)
        {
            setInst("v_lshl_add_u32");
        }

        _VLShiftLeftAddU32(const std::shared_ptr<Container>&    dst,
                           const std::vector<InstructionInput>& srcs,
                           std::optional<VOP3PModifiers>        vop3    = std::nullopt,
                           const std::string&                   comment = "")
            : CommonInstruction(
                InstType::INST_U32, dst, srcs, std::nullopt, std::nullopt, vop3, comment)
        {
            setInst("v_lshl_add_u32");
        }

        _VLShiftLeftAddU32(const _VLShiftLeftAddU32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<_VLShiftLeftAddU32>(*this);
        }
    };

    struct VLShiftLeftAddU32 : public CompositeInstruction
    {
        VLShiftLeftAddU32(const std::shared_ptr<Container>& dst,
                          const InstructionInput&           shiftHex,
                          const InstructionInput&           src0,
                          const InstructionInput&           src1,
                          const std::string&                comment = "")
            : CompositeInstruction(InstType::INST_U32, dst, {src0, src1, shiftHex}, comment)
        {
        }

        std::vector<std::shared_ptr<Instruction>> setupInstructions() const override
        {
            std::vector<std::shared_ptr<Instruction>> instructions;
            if(getAsmCaps()["HasAddLshl"])
            {
                std::vector<InstructionInput> srcs1 = {srcs[0], srcs[2], srcs[1]};
                instructions
                    = {std::make_shared<_VLShiftLeftAddU32>(dst, srcs1, std::nullopt, comment)};
            }
            else
            {
                std::vector<InstructionInput> srcs1 = {srcs[0], srcs[1]};
                std::vector<InstructionInput> srcs2 = {srcs[2], dst};
                if(getAsmBugs()["ExplicitCO"])
                {
                    instructions
                        = {std::make_shared<VAddCOU32>(dst, std::make_shared<VCC>(), srcs1),
                           std::make_shared<VLShiftLeftB32>(dst, srcs2, comment)};
                }
                else
                {
                    instructions = {std::make_shared<VAddU32>(dst, srcs1),
                                    std::make_shared<VLShiftLeftB32>(dst, srcs2, comment)};
                }
            }
            return std::move(instructions);
        }

        VLShiftLeftAddU32(const VLShiftLeftAddU32& other)
            : CompositeInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VLShiftLeftAddU32>(*this);
        }
    };

    struct VMovB32 : public CommonInstruction
    {
        VMovB32(const std::shared_ptr<Container>&   dst,
                const InstructionInput&             src,
                const std::optional<SDWAModifiers>& sdwa    = std::nullopt,
                const std::string&                  comment = "")
            : CommonInstruction(
                InstType::INST_B32, dst, {src}, std::nullopt, sdwa, std::nullopt, comment)
        {
            setInst("v_mov_b32");
        }

        VMovB32(const VMovB32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VMovB32>(*this);
        }
    };

    struct _VMovB64 : public CommonInstruction
    {
        _VMovB64(const std::shared_ptr<Container>& dst,
                 const InstructionInput&           src,
                 const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_B64, dst, {src}, std::nullopt, std::nullopt, std::nullopt, comment)
        {
            setInst("v_mov_b64");
        }

        _VMovB64(const std::shared_ptr<Container>&    dst,
                 const std::vector<InstructionInput>& srcs,
                 const std::string&                   comment = "")
            : CommonInstruction(
                InstType::INST_B64, dst, srcs, std::nullopt, std::nullopt, std::nullopt, comment)
        {
            setInst("v_mov_b64");
        }

        _VMovB64(const _VMovB64& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<_VMovB64>(*this);
        }
    };

    struct VMovB64 : public CompositeInstruction
    {
        VMovB64(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src,
                const std::string&                comment = "")
            : CompositeInstruction(InstType::INST_B64, dst, {src}, comment)
        {
            setInst("v_mov_b64");
        }

        std::vector<std::shared_ptr<Instruction>> setupInstructions() const override
        {
            std::vector<std::shared_ptr<Instruction>> instructions;
            if(getAsmCaps()["v_mov_b64"])
            {
                instructions = {std::make_shared<_VMovB64>(dst, srcs, comment)};
            }
            else
            {
                auto [dst1, dst2]
                    = std::dynamic_pointer_cast<RegisterContainer>(dst)->splitRegContainer();
                std::visit(
                    [&instructions, &dst1, &dst2, this](auto&& arg) -> void {
                        using T = std::decay_t<decltype(arg)>;
                        if constexpr(std::is_same_v<T, std::shared_ptr<Container>>)
                        {
                            auto [src1, src2] = std::dynamic_pointer_cast<RegisterContainer>(arg)
                                                    ->splitRegContainer();
                            instructions
                                = {std::make_shared<VMovB32>(dst1, src1, std::nullopt, comment),
                                   std::make_shared<VMovB32>(dst2, src2, std::nullopt, comment)};
                        }
                    },
                    srcs[0]);
            }
            return std::move(instructions);
        }

        VMovB64(const VMovB64& other)
            : CompositeInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VMovB64>(*this);
        }
    };

    struct VSwapB32 : public CommonInstruction
    {
        VSwapB32(const std::shared_ptr<Container>&   dst,
                const InstructionInput&             src,
                const std::optional<SDWAModifiers>& sdwa    = std::nullopt,
                const std::string&                  comment = "")
            : CommonInstruction(
                InstType::INST_B32, dst, {src}, std::nullopt, sdwa, std::nullopt, comment)
        {
            setInst("v_swap_b32");
        }

        VSwapB32(const VSwapB32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VSwapB32>(*this);
        }
    };

    struct VBfeI32 : public CommonInstruction
    {
        VBfeI32(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src0,
                const InstructionInput&           src1,
                const InstructionInput&           src2,
                const std::string&                comment = "")
            : CommonInstruction(InstType::INST_I32,
                                dst,
                                {src0, src1, src2},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("v_bfe_i32");
        }

        VBfeI32(const VBfeI32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VBfeI32>(*this);
        }
    };

    struct VBfeU32 : public CommonInstruction
    {
        VBfeU32(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src0,
                const InstructionInput&           src1,
                const InstructionInput&           src2,
                const std::string&                comment = "")
            : CommonInstruction(InstType::INST_U32,
                                dst,
                                {src0, src1, src2},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("v_bfe_u32");
        }

        VBfeU32(const VBfeU32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VBfeU32>(*this);
        }
    };

    struct VBfiB32 : public CommonInstruction
    {
        VBfiB32(const std::shared_ptr<Container>& dst,
                const InstructionInput&           src0,
                const InstructionInput&           src1,
                const InstructionInput&           src2,
                const std::string&                comment = "")
            : CommonInstruction(InstType::INST_B32,
                                dst,
                                {src0, src1, src2},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("v_bfi_b32");
        }

        VBfiB32(const VBfiB32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VBfiB32>(*this);
        }
    };

    struct VPackF16toB32 : public CommonInstruction
    {
        VPackF16toB32(const std::shared_ptr<Container>& dst,
                      const InstructionInput&           src0,
                      const InstructionInput&           src1,
                      std::optional<VOP3PModifiers>     vop3    = std::nullopt,
                      const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_B32, dst, {src0, src1}, std::nullopt, std::nullopt, vop3, comment)
        {
            setInst("v_pack_b32_f16");
        }

        VPackF16toB32(const VPackF16toB32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VPackF16toB32>(*this);
        }
    };

    struct VAccvgprReadB32 : public CommonInstruction
    {
        VAccvgprReadB32(const std::shared_ptr<Container>& dst,
                        const InstructionInput&           src,
                        const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_B32, dst, {src}, std::nullopt, std::nullopt, std::nullopt, comment)
        {
            setInst("v_accvgpr_read_b32");
        }

        VAccvgprReadB32(const VAccvgprReadB32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VAccvgprReadB32>(*this);
        }
    };

    struct VAccvgprWrite : public CommonInstruction
    {
        VAccvgprWrite(const std::shared_ptr<Container>& dst,
                      const InstructionInput&           src,
                      const std::string&                comment = "")
            : CommonInstruction(InstType::INST_NOTYPE,
                                dst,
                                {src},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("v_accvgpr_write");
        }

        VAccvgprWrite(const VAccvgprWrite& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VAccvgprWrite>(*this);
        }
    };

    struct VAccvgprWriteB32 : public CommonInstruction
    {
        VAccvgprWriteB32(const std::shared_ptr<Container>& dst,
                         const InstructionInput&           src,
                         const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_B32, dst, {src}, std::nullopt, std::nullopt, std::nullopt, comment)
        {
            setInst("v_accvgpr_write_b32");
        }

        VAccvgprWriteB32(const VAccvgprWriteB32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VAccvgprWriteB32>(*this);
        }
    };

    struct VReadfirstlaneB32 : public CommonInstruction
    {
        VReadfirstlaneB32(const std::shared_ptr<Container>& dst,
                          const InstructionInput&           src,
                          const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_B32, dst, {src}, std::nullopt, std::nullopt, std::nullopt, comment)
        {
            setInst("v_readfirstlane_b32");
        }

        VReadfirstlaneB32(const VReadfirstlaneB32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VReadfirstlaneB32>(*this);
        }
    };

    struct VReadlaneB32 : public CommonInstruction
    {
        VReadlaneB32(const std::shared_ptr<Container>& dst,
                          const InstructionInput&           src0,
                          const InstructionInput&           src1,
                          const std::string&                comment = "")
            : CommonInstruction(
                                InstType::INST_B32, dst, {src0, src1}, std::nullopt, std::nullopt, std::nullopt, comment)
        {
            setInst("v_readlane_b32");
        }

        VReadlaneB32(const VReadlaneB32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VReadlaneB32>(*this);
        }
    };

    struct VWritelaneB32 : public CommonInstruction
    {
        VWritelaneB32(const std::shared_ptr<Container>& dst,
                          const InstructionInput&           src0,
                          const InstructionInput&           src1,
                          const std::string&                comment = "")
            : CommonInstruction(
                                InstType::INST_B32, dst, {src0, src1}, std::nullopt, std::nullopt, std::nullopt, comment)
        {
            setInst("v_writelane_b32");
        }

        VWritelaneB32(const VWritelaneB32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VWritelaneB32>(*this);
        }
    };

    struct VRndneF32 : public CommonInstruction
    {
        VRndneF32(const std::shared_ptr<Container>& dst,
                  const InstructionInput&           src,
                  const std::string&                comment = "")
            : CommonInstruction(
                InstType::INST_F32, dst, {src}, std::nullopt, std::nullopt, std::nullopt, comment)
        {
            setInst("v_rndne_f32");
        }

        VRndneF32(const VRndneF32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VRndneF32>(*this);
        }
    };

    struct VPermB32 : public CommonInstruction
    {
        VPermB32(const std::shared_ptr<Container>& dst,
                 const InstructionInput&           src0,
                 const InstructionInput&           src1,
                 const InstructionInput&           src2,
                 const std::string&                comment = "")
            : CommonInstruction(InstType::INST_B32,
                                dst,
                                {src0, src1, src2},
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                comment)
        {
            setInst("v_perm_b32");
        }

        VPermB32(const VPermB32& other)
            : CommonInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<VPermB32>(*this);
        }
    };
} // namespace rocisa
