/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <rocRoller/CodeGen/AddInstruction.hpp>
#include <rocRoller/CodeGen/Arithmetic/Divide.hpp>
#include <rocRoller/CodeGen/BranchGenerator.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/SubInstruction.hpp>
#include <rocRoller/InstructionValues/LabelAllocator.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    // Register supported components
    RegisterComponentTemplateSpec(DivideGenerator, Register::Type::Scalar, DataType::Int32);
    RegisterComponentTemplateSpec(DivideGenerator, Register::Type::Vector, DataType::Int32);
    RegisterComponentTemplateSpec(DivideGenerator, Register::Type::Scalar, DataType::Int64);
    RegisterComponentTemplateSpec(DivideGenerator, Register::Type::Vector, DataType::Int64);

    template <>
    std::shared_ptr<BinaryArithmeticGenerator<Expression::Divide>>
        GetGenerator<Expression::Divide>(Register::ValuePtr dst,
                                         Register::ValuePtr lhs,
                                         Register::ValuePtr rhs,
                                         Expression::Divide const&)
    {
        // Choose the proper generator, based on the context, register type
        // and datatype.
        return Component::Get<BinaryArithmeticGenerator<Expression::Divide>>(
            getContextFromValues(dst, lhs, rhs),
            promoteRegisterType(dst, lhs, rhs),
            promoteDataType(dst, lhs, rhs));
    }

    template <>
    Generator<Instruction> DivideGenerator<Register::Type::Scalar, DataType::Int32>::generate(
        Register::ValuePtr dest,
        Register::ValuePtr lhs,
        Register::ValuePtr rhs,
        Expression::Divide const&)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction::Lock(Scheduling::Dependency::VCC, "Start of Division"));

        // Allocate temporary registers
        auto s_1 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 1);

        auto s_2 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 1);

        auto s_3 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 1);

        auto s_4 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 1);

        auto s_5 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 1);

        auto v_1 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);

        auto v_2 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);

        auto v_3 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);

        co_yield_(Instruction("s_ashr_i32", {s_1}, {rhs, Register::Value::Literal(31)}, {}, ""));
        co_yield ScalarAddInt32(m_context, s_4, rhs, s_1);
        co_yield_(Instruction("s_xor_b32", {s_4}, {s_4, s_1}, {}, ""));
        co_yield_(Instruction("v_cvt_f32_u32", {v_1}, {s_4}, {}, ""));
        co_yield_(Instruction("s_ashr_i32", {s_2}, {lhs, Register::Value::Literal(31)}, {}, ""));
        co_yield ScalarAddInt32(m_context, s_5, lhs, s_2);
        co_yield_(Instruction("s_xor_b32", {s_3}, {s_2, s_1}, {}, ""));
        co_yield_(Instruction("v_rcp_iflag_f32", {v_1}, {v_1}, {}, ""));
        co_yield_(Instruction("s_xor_b32", {s_5}, {s_5, s_2}, {}, ""));
        co_yield ScalarSubInt32(m_context, s_2, Register::Value::Literal(0), s_4);
        co_yield_(
            Instruction("v_mul_f32", {v_1}, {Register::Value::Literal(0x4f7ffffe), v_1}, {}, ""));
        co_yield_(Instruction("v_cvt_u32_f32", {v_1}, {v_1}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_2}, {s_2, v_1}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_2}, {v_1, v_2}, {}, ""));
        co_yield VectorAddUInt32(m_context, v_1, v_1, v_2);
        co_yield_(Instruction("v_mul_hi_u32", {v_1}, {s_5, v_1}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_2}, {v_1, s_4}, {}, ""));
        co_yield VectorSubUInt32(m_context, v_2, s_5, v_2);
        co_yield VectorAddUInt32(m_context, v_3, Register::Value::Literal(1), v_1);
        co_yield_(Instruction("v_cmp_le_u32", {m_context->getVCC()}, {s_4, v_2}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32", {v_1}, {v_1, v_3, m_context->getVCC()}, {}, ""));
        co_yield VectorSubRevUInt32(m_context, v_3, s_4, v_2);
        co_yield_(Instruction("v_cndmask_b32", {v_2}, {v_2, v_3, m_context->getVCC()}, {}, ""));
        co_yield VectorAddUInt32(m_context, v_3, Register::Value::Literal(1), v_1);
        co_yield_(Instruction("v_cmp_le_u32", {m_context->getVCC()}, {s_4, v_2}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32", {v_1}, {v_1, v_3, m_context->getVCC()}, {}, ""));
        co_yield_(Instruction("v_xor_b32", {v_1}, {s_3, v_1}, {}, ""));
        co_yield VectorSubRevUInt32(m_context, v_1, s_3, v_1);
        co_yield_(Instruction(
            "v_readlane_b32", {dest}, {v_1, Register::Value::Literal(0)}, {}, "Move value"));

        co_yield_(Instruction::Unlock("End of Division"));
    }

    template <>
    Generator<Instruction> DivideGenerator<Register::Type::Vector, DataType::Int32>::generate(
        Register::ValuePtr dest,
        Register::ValuePtr lhs,
        Register::ValuePtr rhs,
        Expression::Divide const&)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction::Lock(Scheduling::Dependency::VCC, "Start of Division"));

        // Allocate temporary registers

        auto v_1 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);

        auto v_2 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);

        auto v_3 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);

        auto v_4 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);

        auto v_5 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);

        auto v_6 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);

        auto v_7 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);

        if(rhs->regType() == Register::Type::Literal)
        {
            co_yield moveToVGPR(rhs);
        }

        if(lhs->regType() == Register::Type::Literal)
        {
            co_yield moveToVGPR(lhs);
        }

        co_yield_(Instruction("v_ashrrev_i32", {v_1}, {Register::Value::Literal(31), rhs}, {}, ""));
        co_yield VectorAddUInt32(m_context, v_7, rhs, v_1);
        co_yield_(Instruction("v_xor_b32", {v_7}, {v_7, v_1}, {}, ""));
        co_yield_(Instruction("v_cvt_f32_u32", {v_2}, {v_7}, {}, ""));
        co_yield_(Instruction("v_rcp_iflag_f32", {v_3}, {v_2}, {}, ""));
        co_yield_(
            Instruction("v_mul_f32", {v_3}, {Register::Value::Literal(0x4f7ffffe), v_3}, {}, ""));
        co_yield_(Instruction("v_cvt_u32_f32", {v_3}, {v_3}, {}, ""));
        co_yield_(Instruction("v_ashrrev_i32", {v_4}, {Register::Value::Literal(31), lhs}, {}, ""));
        co_yield VectorAddUInt32(m_context, v_6, lhs, v_4);
        co_yield_(Instruction("v_xor_b32", {v_1}, {v_4, v_1}, {}, ""));
        co_yield_(Instruction("v_xor_b32", {v_6}, {v_6, v_4}, {}, ""));
        co_yield VectorSubUInt32(m_context, v_4, Register::Value::Literal(0), v_7);
        co_yield_(Instruction("v_mul_lo_u32", {v_4}, {v_4, v_3}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_4}, {v_3, v_4}, {}, ""));
        co_yield VectorAddUInt32(m_context, v_3, v_3, v_4);
        co_yield_(Instruction("v_mul_hi_u32", {v_3}, {v_6, v_3}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_4}, {v_3, v_7}, {}, ""));
        co_yield VectorSubUInt32(m_context, v_6, v_6, v_4);
        co_yield VectorAddUInt32(m_context, v_5, Register::Value::Literal(1), v_3);
        co_yield_(Instruction("v_cmp_ge_u32", {m_context->getVCC()}, {v_6, v_7}, {}, ""));
        co_yield VectorSubUInt32(m_context, v_4, v_6, v_7);
        co_yield_(Instruction("v_cndmask_b32", {v_3}, {v_3, v_5, m_context->getVCC()}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32", {v_6}, {v_6, v_4, m_context->getVCC()}, {}, ""));
        co_yield VectorAddUInt32(m_context, v_4, Register::Value::Literal(1), v_3);
        co_yield_(Instruction("v_cmp_ge_u32", {m_context->getVCC()}, {v_6, v_7}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32", {v_6}, {v_3, v_4, m_context->getVCC()}, {}, ""));
        co_yield_(Instruction("v_xor_b32", {v_6}, {v_6, v_1}, {}, ""));
        co_yield VectorSubUInt32(m_context, dest, v_6, v_1);

        co_yield_(Instruction::Unlock("End of Division"));
    }

    template <>
    Generator<Instruction> DivideGenerator<Register::Type::Scalar, DataType::Int64>::generate(
        Register::ValuePtr dest,
        Register::ValuePtr lhs,
        Register::ValuePtr rhs,
        Expression::Divide const&)
    {

        auto const& architecture  = m_context->targetArchitecture();
        auto const  wavefrontSize = architecture.GetCapability(GPUCapability::DefaultWavefrontSize);
        AssertFatal(wavefrontSize == 32 || wavefrontSize == 64,
                    std::format("DivideGenerator<{}, {}> only implemented for wave32 or wave64",
                                toString(Register::Type::Scalar),
                                toString(DataType::Int64)));

        auto VCC = m_context->getVCC();

        // Generated using the assembly_to_instructions.py script with the following HIP code:
        //  extern "C"
        //  __global__ void hello_world(long int * ptr, long int value1, long int value2)
        //  {
        //    *ptr = value1 / value2 ;
        //  }
        //
        // Generated code was modified to use the provided dest, lhs and rhs registers and
        // to save the result in the dest register instead of memory.
        co_yield Instruction::Lock(Scheduling::Dependency::SCC, "Start of Divide64(SCC)");
        co_yield(Instruction::Lock(Scheduling::Dependency::VCC, "Start of Divide64(VCC)"));
        co_yield describeOpArgs("dest", dest, "lhs", lhs, "rhs", rhs);
        Register::ValuePtr l0, l1, r0, r1;
        co_yield get2DwordsScalar(l0, l1, lhs);
        co_yield get2DwordsScalar(r0, r1, rhs);

        auto s_0
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Scalar,
                                                DataType::Int32,
                                                2,
                                                Register::AllocationOptions::FullyContiguous());

        auto s_0_flag = (wavefrontSize == 64) ? s_0 : s_0->subset({0});

        auto s_2 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 1);
        auto s_5
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Scalar,
                                                DataType::Int32,
                                                2,
                                                Register::AllocationOptions::FullyContiguous());
        auto s_6
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Scalar,
                                                DataType::Int32,
                                                2,
                                                Register::AllocationOptions::FullyContiguous());
        auto v_7 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_8 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_10 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_11 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_12 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_13 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_14 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_15 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_16 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_17 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_18 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_19 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_20 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto s_22
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Scalar,
                                                DataType::Int32,
                                                2,
                                                Register::AllocationOptions::FullyContiguous());
        auto s_23
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Scalar,
                                                DataType::Int32,
                                                2,
                                                Register::AllocationOptions::FullyContiguous());

        auto label_24 = m_context->labelAllocator()->label("BB0_2");
        auto label_25 = m_context->labelAllocator()->label("BB0_3");
        auto label_26 = m_context->labelAllocator()->label("BB0_4");

        co_yield_(Instruction("s_or_b64", {s_0}, {lhs, rhs}, {}, ""));
        co_yield m_context->copier()->copy(s_0->subset({0}), Register::Value::Literal(0), "");
        co_yield_(Instruction("s_cmp_lg_u64", {s_0}, {Register::Value::Literal(0)}, {}, ""));
        co_yield m_context->brancher()->branchIfZero(label_24, m_context->getSCC());
        co_yield_(Instruction(
            "s_ashr_i32", {s_6->subset({0})}, {r1, Register::Value::Literal(31)}, {}, ""));
        co_yield ScalarAddUInt32(m_context, s_0->subset({0}), r0, s_6->subset({0}));
        co_yield m_context->copier()->copy(s_6->subset({1}), s_6->subset({0}), "");
        co_yield ScalarAddUInt32CarryInOut(m_context, s_0->subset({1}), r1, s_6->subset({0}));
        co_yield_(Instruction("s_xor_b64", {s_5}, {s_0, s_6}, {}, ""));
        co_yield_(Instruction("v_cvt_f32_u32", {v_7}, {s_5->subset({0})}, {}, ""));
        co_yield_(Instruction("v_cvt_f32_u32", {v_8}, {s_5->subset({1})}, {}, ""));
        co_yield ScalarSubUInt32(m_context, s_2, Register::Value::Literal(0), s_5->subset({0}));
        co_yield ScalarSubUInt32CarryInOut(
            m_context, s_23->subset({0}), Register::Value::Literal(0), s_5->subset({1}));
        co_yield m_context->copier()->copy(v_10, Register::Value::Literal(0), "");

        if(architecture.HasCapability(GPUCapability::v_fmac_f32))
        {
            co_yield_(Instruction(
                "v_fmac_f32", {v_7}, {Register::Value::Literal(0x4f800000), v_8}, {}, ""));
        }
        else if(architecture.HasCapability(GPUCapability::v_mac_f32))
        {
            co_yield_(Instruction(
                "v_mac_f32", {v_7}, {Register::Value::Literal(0x4f800000), v_8}, {}, ""));
        }
        else
        {
            Throw<FatalError>("Can not generate Divide: Neither v_fmac_f32 or v_mac_f32 known");
        }
        co_yield_(Instruction("v_rcp_f32", {v_7}, {v_7}, {}, ""));
        co_yield m_context->copier()->copy(v_11, Register::Value::Literal(0), "");
        co_yield_(
            Instruction("v_mul_f32", {v_7}, {Register::Value::Literal(0x5f7ffffc), v_7}, {}, ""));
        co_yield_(
            Instruction("v_mul_f32", {v_8}, {Register::Value::Literal(0x2f800000), v_7}, {}, ""));
        co_yield_(Instruction("v_trunc_f32", {v_8}, {v_8}, {}, ""));
        if(architecture.HasCapability(GPUCapability::v_fmac_f32))
        {
            co_yield_(Instruction(
                "v_fmac_f32", {v_7}, {Register::Value::Literal(0xcf800000), v_8}, {}, ""));
        }
        else if(architecture.HasCapability(GPUCapability::v_mac_f32))
        {
            co_yield_(Instruction(
                "v_mac_f32", {v_7}, {Register::Value::Literal(0xcf800000), v_8}, {}, ""));
        }
        else
        {
            Throw<FatalError>("Can not generate Divide: Neither v_fmac_f32 or v_mac_f32 known");
        }
        co_yield_(Instruction("v_cvt_u32_f32", {v_8}, {v_8}, {}, ""));
        co_yield_(Instruction("v_cvt_u32_f32", {v_7}, {v_7}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_12}, {s_2, v_8}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_13}, {s_2, v_7}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_14}, {s_23->subset({0}), v_7}, {}, ""));
        co_yield VectorAddUInt32(m_context, v_12, v_13, v_12);
        co_yield VectorAddUInt32(m_context, v_12, v_12, v_14);
        co_yield_(Instruction("v_mul_lo_u32", {v_15}, {s_2, v_7}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_13}, {v_7, v_12}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_16}, {v_7, v_15}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_14}, {v_7, v_12}, {}, ""));
        co_yield VectorAddUInt32CarryOut(m_context, v_13, v_16, v_13);
        co_yield VectorAddUInt32CarryInOut(m_context, v_14, v_10, v_14);
        co_yield_(Instruction("v_mul_hi_u32", {v_17}, {v_8, v_15}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_15}, {v_8, v_15}, {}, ""));
        co_yield VectorAddUInt32CarryOut(m_context, v_13, v_13, v_15);
        co_yield_(Instruction("v_mul_hi_u32", {v_16}, {v_8, v_12}, {}, ""));
        co_yield VectorAddUInt32CarryInOut(m_context, v_14, v_14, v_17);
        co_yield VectorAddUInt32CarryInOut(m_context, v_13, v_16, v_11);
        co_yield_(Instruction("v_mul_lo_u32", {v_12}, {v_8, v_12}, {}, ""));
        co_yield VectorAddUInt32CarryOut(m_context, v_12, v_14, v_12);
        co_yield VectorAddUInt32CarryInOut(m_context, v_14, v_10, v_13);
        co_yield VectorAddUInt32CarryOut(m_context, v_7, s_0_flag, v_7, v_12);
        co_yield VectorAddUInt32CarryInOut(m_context, v_12, VCC, s_0_flag, v_8, v_14);
        co_yield_(Instruction("v_mul_lo_u32", {v_13}, {s_2, v_12}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_15}, {s_2, v_7}, {}, ""));
        co_yield VectorAddUInt32(m_context, v_13, v_15, v_13);
        co_yield_(Instruction("v_mul_lo_u32", {v_15}, {s_23->subset({0}), v_7}, {}, ""));
        co_yield VectorAddUInt32(m_context, v_13, v_13, v_15);
        co_yield_(Instruction("v_mul_lo_u32", {v_16}, {s_2, v_7}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_17}, {v_12, v_16}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_18}, {v_12, v_16}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_19}, {v_7, v_13}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_16}, {v_7, v_16}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_20}, {v_7, v_13}, {}, ""));
        co_yield VectorAddUInt32CarryOut(m_context, v_16, v_16, v_19);
        co_yield VectorAddUInt32CarryInOut(m_context, v_20, v_10, v_20);
        co_yield VectorAddUInt32CarryOut(m_context, v_16, v_16, v_18);
        co_yield_(Instruction("v_mul_hi_u32", {v_15}, {v_12, v_13}, {}, ""));
        co_yield VectorAddUInt32CarryInOut(m_context, v_16, v_20, v_17);
        co_yield VectorAddUInt32CarryInOut(m_context, v_15, v_15, v_11);
        co_yield_(Instruction("v_mul_lo_u32", {v_12}, {v_12, v_13}, {}, ""));
        co_yield VectorAddUInt32CarryOut(m_context, v_12, v_16, v_12);
        co_yield VectorAddUInt32CarryInOut(m_context, v_13, v_10, v_15);
        co_yield VectorAddUInt32(m_context, v_8, v_8, v_14);
        co_yield_(Instruction(
            "s_ashr_i32", {s_23->subset({0})}, {l1, Register::Value::Literal(31)}, {}, ""));
        co_yield VectorAddUInt32CarryInOut(m_context, v_8, VCC, s_0_flag, v_8, v_13);
        co_yield ScalarAddUInt32(m_context, s_0->subset({0}), l0, s_23->subset({0}));
        co_yield VectorAddUInt32CarryOut(m_context, v_7, v_7, v_12);
        co_yield m_context->copier()->copy(s_23->subset({1}), s_23->subset({0}), "");
        co_yield ScalarAddUInt32CarryInOut(m_context, s_0->subset({1}), l1, s_23->subset({0}));
        co_yield VectorAddUInt32CarryInOut(m_context, v_8, Register::Value::Literal(0), v_8);
        co_yield_(Instruction("s_xor_b64", {s_22}, {s_0, s_23}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_14}, {s_22->subset({0}), v_8}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_13}, {s_22->subset({0}), v_7}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_12}, {s_22->subset({0}), v_8}, {}, ""));
        co_yield VectorAddUInt32CarryOut(m_context, v_14, v_13, v_14);
        co_yield VectorAddUInt32CarryInOut(m_context, v_12, v_10, v_12);
        co_yield_(Instruction("v_mul_hi_u32", {v_15}, {s_22->subset({1}), v_7}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_7}, {s_22->subset({1}), v_7}, {}, ""));
        co_yield VectorAddUInt32CarryOut(m_context, v_7, v_14, v_7);
        co_yield_(Instruction("v_mul_hi_u32", {v_13}, {s_22->subset({1}), v_8}, {}, ""));
        co_yield VectorAddUInt32CarryInOut(m_context, v_7, v_12, v_15);
        co_yield VectorAddUInt32CarryInOut(m_context, v_12, v_13, v_11);
        co_yield_(Instruction("v_mul_lo_u32", {v_8}, {s_22->subset({1}), v_8}, {}, ""));
        co_yield VectorAddUInt32CarryOut(m_context, v_7, v_7, v_8);
        co_yield VectorAddUInt32CarryInOut(m_context, v_8, v_10, v_12);
        co_yield_(Instruction("v_mul_lo_u32", {v_12}, {s_5->subset({0}), v_8}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_14}, {s_5->subset({0}), v_7}, {}, ""));
        co_yield VectorAddUInt32(m_context, v_12, v_14, v_12);
        co_yield_(Instruction("v_mul_lo_u32", {v_14}, {s_5->subset({1}), v_7}, {}, ""));
        co_yield VectorAddUInt32(m_context, v_12, v_12, v_14);
        co_yield_(Instruction("v_mul_lo_u32", {v_13}, {s_5->subset({0}), v_7}, {}, ""));
        co_yield VectorSubUInt32(m_context, v_14, s_22->subset({1}), v_12);
        co_yield m_context->copier()->copy(v_15, s_5->subset({1}), "");
        co_yield VectorSubUInt32CarryOut(m_context, v_13, s_22->subset({0}), v_13);
        co_yield VectorSubUInt32CarryInOut(m_context, v_14, s_0_flag, VCC, v_14, v_15);
        co_yield VectorSubRevUInt32CarryOut(m_context, v_15, s_0_flag, s_5->subset({0}), v_13);
        co_yield VectorSubRevUInt32CarryInOut(
            m_context, v_14, s_0_flag, s_0_flag, Register::Value::Literal(0), v_14);
        co_yield_(Instruction("v_cmp_le_u32", {s_0_flag}, {s_5->subset({1}), v_14}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32",
                              {v_16},
                              {Register::Value::Literal(0), Register::Value::Literal(-1), s_0_flag},
                              {},
                              ""));
        co_yield_(Instruction("v_cmp_le_u32", {s_0_flag}, {s_5->subset({0}), v_15}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32",
                              {v_15},
                              {Register::Value::Literal(0), Register::Value::Literal(-1), s_0_flag},
                              {},
                              ""));
        co_yield_(Instruction("v_cmp_eq_u32", {s_0_flag}, {s_5->subset({1}), v_14}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32", {v_14}, {v_16, v_15, s_0_flag}, {}, ""));
        co_yield m_context->copier()->copy(v_16, s_22->subset({1}), "");
        co_yield VectorSubUInt32CarryInOut(m_context, v_12, v_16, v_12);
        co_yield_(
            Instruction("v_cmp_le_u32", {m_context->getVCC()}, {s_5->subset({1}), v_12}, {}, ""));
        co_yield_(
            Instruction("v_cmp_ne_u32", {s_0_flag}, {Register::Value::Literal(0), v_14}, {}, ""));
        co_yield_(Instruction(
            "v_cndmask_b32",
            {v_16},
            {Register::Value::Literal(0), Register::Value::Literal(-1), m_context->getVCC()},
            {},
            ""));
        co_yield_(
            Instruction("v_cmp_le_u32", {m_context->getVCC()}, {s_5->subset({0}), v_13}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32",
                              {v_14},
                              {Register::Value::Literal(1), Register::Value::Literal(2), s_0_flag},
                              {},
                              ""));
        co_yield_(Instruction(
            "v_cndmask_b32",
            {v_13},
            {Register::Value::Literal(0), Register::Value::Literal(-1), m_context->getVCC()},
            {},
            ""));
        co_yield_(
            Instruction("v_cmp_eq_u32", {m_context->getVCC()}, {s_5->subset({1}), v_12}, {}, ""));
        co_yield VectorAddUInt32CarryOut(m_context, v_14, s_0_flag, v_7, v_14);
        co_yield_(Instruction("v_cndmask_b32", {v_12}, {v_16, v_13, m_context->getVCC()}, {}, ""));
        co_yield VectorAddUInt32CarryInOut(
            m_context, v_15, s_0_flag, s_0_flag, Register::Value::Literal(0), v_8);
        co_yield_(Instruction(
            "v_cmp_ne_u32", {m_context->getVCC()}, {Register::Value::Literal(0), v_12}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32", {v_7}, {v_7, v_14, m_context->getVCC()}, {}, ""));
        co_yield_(Instruction("s_xor_b64", {s_0}, {s_23, s_6}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32", {v_8}, {v_8, v_15, m_context->getVCC()}, {}, ""));
        co_yield_(Instruction("v_xor_b32", {v_7}, {s_0->subset({0}), v_7}, {}, ""));
        co_yield_(Instruction("v_xor_b32", {v_8}, {s_0->subset({1}), v_8}, {}, ""));
        co_yield m_context->copier()->copy(v_12, s_0->subset({1}), "");
        co_yield VectorSubRevUInt32CarryOut(m_context, v_7, s_0->subset({0}), v_7);
        co_yield VectorSubUInt32CarryInOut(m_context, v_8, v_8, v_12);
        co_yield_(Instruction("s_cbranch_execz", {}, {label_25}, {}, ""));
        co_yield m_context->brancher()->branch(label_26);
        co_yield_(Instruction::Label(label_24));
        co_yield_(Instruction::Label(label_25));
        co_yield_(Instruction("v_cvt_f32_u32", {v_7}, {r0}, {}, ""));
        co_yield ScalarSubInt32(m_context, s_0->subset({0}), Register::Value::Literal(0), r0);
        co_yield_(Instruction("v_rcp_iflag_f32", {v_7}, {v_7}, {}, ""));
        co_yield_(
            Instruction("v_mul_f32", {v_7}, {Register::Value::Literal(0x4f7ffffe), v_7}, {}, ""));
        co_yield_(Instruction("v_cvt_u32_f32", {v_7}, {v_7}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_8}, {s_0->subset({0}), v_7}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_8}, {v_7, v_8}, {}, ""));
        co_yield VectorAddUInt32(m_context, v_7, v_7, v_8);
        co_yield m_context->copier()->copy(v_8, l0, "");
        co_yield_(Instruction("v_mul_hi_u32", {v_7}, {v_8, v_7}, {}, ""));
        co_yield m_context->copier()->copy(v_12, r0, "");
        co_yield_(Instruction("v_mul_lo_u32", {v_12}, {v_7, v_12}, {}, ""));
        co_yield VectorSubUInt32(m_context, v_12, l0, v_12);
        co_yield VectorAddUInt32(m_context, v_8, Register::Value::Literal(1), v_7);
        co_yield_(Instruction("v_cmp_le_u32", {m_context->getVCC()}, {r0, v_12}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32", {v_7}, {v_7, v_8, m_context->getVCC()}, {}, ""));
        co_yield VectorSubRevUInt32(m_context, v_8, r0, v_12);
        co_yield_(Instruction("v_cndmask_b32", {v_8}, {v_12, v_8, m_context->getVCC()}, {}, ""));
        co_yield VectorAddUInt32(m_context, v_14, Register::Value::Literal(1), v_7);
        co_yield_(Instruction("v_cmp_le_u32", {m_context->getVCC()}, {r0, v_8}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32", {v_7}, {v_7, v_14, m_context->getVCC()}, {}, ""));
        co_yield m_context->copier()->copy(v_8, Register::Value::Literal(0), "");
        co_yield_(Instruction::Label(label_26));
        co_yield_(Instruction("v_readlane_b32",
                              {dest->subset({0})},
                              {v_7, Register::Value::Literal(0)},
                              {},
                              "Move value"));
        co_yield_(Instruction("v_readlane_b32",
                              {dest->subset({1})},
                              {v_8, Register::Value::Literal(0)},
                              {},
                              "Move value"));
        co_yield(Instruction::Unlock("End of Divide64(VCC)"));
        co_yield(Instruction::Unlock("End of Divide64(SCC)"));
    }

    template <>
    Generator<Instruction> DivideGenerator<Register::Type::Vector, DataType::Int64>::generate(
        Register::ValuePtr dest,
        Register::ValuePtr lhs,
        Register::ValuePtr rhs,
        Expression::Divide const&)
    {
        auto const& architecture  = m_context->targetArchitecture();
        auto const  wavefrontSize = architecture.GetCapability(GPUCapability::DefaultWavefrontSize);
        AssertFatal(wavefrontSize == 32 || wavefrontSize == 64,
                    std::format("DivideGenerator<{}, {}> only implemented for wave32 or wave64",
                                toString(Register::Type::Vector),
                                toString(DataType::Int64)));

        auto VCC  = m_context->getVCC();
        auto EXEC = m_context->getExec();

        // Generated using the assembly_to_instructions.py script with the following HIP code:
        //  extern "C"
        //  __global__ void hello_world(long int * ptr, long int *value1, long int *value2)
        //  {
        //    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        //    ptr[idx] = value1[idx] / value2[idx];
        //  }
        //
        // Generated code was modified to use the provided dest, lhs and rhs registers and
        // to save the result in the dest register instead of memory.
        co_yield(Instruction::Lock(Scheduling::Dependency::SCC, "Start of Divide64(SCC)"));
        co_yield(Instruction::Lock(Scheduling::Dependency::VCC, "Start of Divide64(VCC)"));
        co_yield describeOpArgs("dest", dest, "lhs", lhs, "rhs", rhs);

        Register::ValuePtr l0, l1, r0, r1;
        co_yield get2DwordsVector(l0, l1, lhs);
        co_yield get2DwordsVector(r0, r1, rhs);

        auto v_2 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_3 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto s_5
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Scalar,
                                                DataType::Int32,
                                                (wavefrontSize == 64) ? 2 : 1,
                                                Register::AllocationOptions::FullyContiguous());
        auto s_6
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Scalar,
                                                DataType::Int32,
                                                (wavefrontSize == 64) ? 2 : 1,
                                                Register::AllocationOptions::FullyContiguous());
        auto v_7 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_8 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_9 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_10 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_11 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_12 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_13 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_14 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_15 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_16 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_17 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_18 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_19 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_20 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);

        auto label_21 = m_context->labelAllocator()->label("BB0_2");
        auto label_22 = m_context->labelAllocator()->label("BB0_4");

        co_yield m_context->copier()->copy(v_20, l0, "");
        co_yield m_context->copier()->copy(v_2, l1, "");
        co_yield m_context->copier()->copy(v_7, r0, "");
        co_yield m_context->copier()->copy(v_3, r1, "");

        co_yield_(Instruction("v_or_b32", {dest->subset({1})}, {v_2, v_3}, {}, ""));

        if(wavefrontSize == 64)
        {
            co_yield_(
                Instruction("v_cmp_ne_u64", {VCC}, {Register::Value::Literal(0), dest}, {}, ""));
            co_yield_(Instruction("s_and_saveexec_b64", {s_5}, {VCC}, {}, ""));
            co_yield_(Instruction("s_xor_b64", {s_6}, {EXEC, s_5}, {}, ""));
        }
        else
        {
            co_yield_(Instruction(
                "v_cmp_ne_u64_e32", {VCC}, {Register::Value::Literal(0), dest}, {}, ""));
            co_yield_(Instruction("s_and_saveexec_b32", {s_5}, {VCC}, {}, ""));
            co_yield_(Instruction("s_xor_b32", {s_6}, {EXEC, s_5}, {}, ""));
        }

        co_yield_(Instruction("s_cbranch_execz", {}, {label_21}, {}, ""));
        co_yield_(Instruction(
            "v_ashrrev_i32", {dest->subset({0})}, {Register::Value::Literal(31), v_3}, {}, ""));
        co_yield VectorAddUInt32CarryOut(m_context, v_7, v_7, dest->subset({0}));
        co_yield VectorAddUInt32CarryInOut(m_context, v_3, v_3, dest->subset({0}));
        co_yield_(Instruction("v_xor_b32", {v_3}, {v_3, dest->subset({0})}, {}, ""));
        co_yield_(Instruction("v_xor_b32", {v_7}, {v_7, dest->subset({0})}, {}, ""));
        co_yield_(Instruction("v_cvt_f32_u32", {dest->subset({1})}, {v_7}, {}, ""));
        co_yield_(Instruction("v_cvt_f32_u32", {v_8}, {v_3}, {}, ""));
        co_yield VectorSubUInt32CarryOut(m_context, v_9, Register::Value::Literal(0), v_7);
        co_yield VectorSubUInt32CarryInOut(m_context, v_10, Register::Value::Literal(0), v_3);
        if(architecture.HasCapability(GPUCapability::v_fmac_f32))
        {
            co_yield_(Instruction("v_fmac_f32",
                                  {dest->subset({1})},
                                  {Register::Value::Literal(0x4f800000), v_8},
                                  {},
                                  ""));
        }
        else if(architecture.HasCapability(GPUCapability::v_mac_f32))
        {
            co_yield_(Instruction("v_mac_f32",
                                  {dest->subset({1})},
                                  {Register::Value::Literal(0x4f800000), v_8},
                                  {},
                                  ""));
        }
        else
        {
            Throw<FatalError>("Can not generate Divide: Neither v_fmac_f32 or v_mac_f32 known");
        }

        co_yield_(Instruction("v_rcp_f32", {dest->subset({1})}, {dest->subset({1})}, {}, ""));
        co_yield m_context->copier()->copy(v_11, Register::Value::Literal(0), "");
        co_yield m_context->copier()->copy(v_12, Register::Value::Literal(0), "");
        co_yield_(Instruction("v_mul_f32",
                              {dest->subset({1})},
                              {Register::Value::Literal(0x5f7ffffc), dest->subset({1})},
                              {},
                              ""));
        co_yield_(Instruction(
            "v_mul_f32", {v_8}, {Register::Value::Literal(0x2f800000), dest->subset({1})}, {}, ""));
        co_yield_(Instruction("v_trunc_f32", {v_8}, {v_8}, {}, ""));
        if(architecture.HasCapability(GPUCapability::v_fmac_f32))
        {
            co_yield_(Instruction("v_fmac_f32",
                                  {dest->subset({1})},
                                  {Register::Value::Literal(0xcf800000), v_8},
                                  {},
                                  ""));
        }
        else if(architecture.HasCapability(GPUCapability::v_mac_f32))
        {
            co_yield_(Instruction("v_mac_f32",
                                  {dest->subset({1})},
                                  {Register::Value::Literal(0xcf800000), v_8},
                                  {},
                                  ""));
        }
        else
        {
            Throw<FatalError>("Can not generate Divide: Neither v_fmac_f32 or v_mac_f32 known");
        }
        co_yield_(Instruction("v_cvt_u32_f32", {v_8}, {v_8}, {}, ""));
        co_yield_(Instruction("v_cvt_u32_f32", {dest->subset({1})}, {dest->subset({1})}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_13}, {v_9, v_8}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_14}, {v_10, dest->subset({1})}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_15}, {v_9, dest->subset({1})}, {}, ""));
        co_yield VectorAdd3UInt32(m_context, v_14, v_15, v_13, v_14);
        co_yield_(Instruction("v_mul_lo_u32", {v_16}, {v_9, dest->subset({1})}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_13}, {dest->subset({1}), v_14}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_17}, {dest->subset({1}), v_16}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_15}, {dest->subset({1}), v_14}, {}, ""));
        co_yield VectorAddUInt32CarryOut(m_context, v_13, v_17, v_13);
        co_yield VectorAddUInt32CarryInOut(m_context, v_15, v_11, v_15);
        co_yield_(Instruction("v_mul_hi_u32", {v_18}, {v_8, v_16}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_16}, {v_8, v_16}, {}, ""));
        co_yield VectorAddUInt32CarryOut(m_context, v_13, v_13, v_16);
        co_yield_(Instruction("v_mul_hi_u32", {v_17}, {v_8, v_14}, {}, ""));
        co_yield VectorAddUInt32CarryInOut(m_context, v_15, v_15, v_18);
        co_yield VectorAddUInt32CarryInOut(m_context, v_13, v_17, v_12);
        co_yield_(Instruction("v_mul_lo_u32", {v_14}, {v_8, v_14}, {}, ""));
        co_yield VectorAddUInt32CarryOut(m_context, v_14, v_15, v_14);
        co_yield VectorAddUInt32CarryInOut(m_context, v_15, v_11, v_13);
        co_yield VectorAddUInt32CarryOut(
            m_context, dest->subset({1}), s_5, dest->subset({1}), v_14);
        co_yield VectorAddUInt32CarryInOut(m_context, v_14, VCC, s_5, v_8, v_15);
        co_yield_(Instruction("v_mul_lo_u32", {v_13}, {v_9, v_14}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_10}, {v_10, dest->subset({1})}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_16}, {v_9, dest->subset({1})}, {}, ""));
        co_yield VectorAdd3UInt32(m_context, v_10, v_16, v_13, v_10);
        co_yield_(Instruction("v_mul_lo_u32", {v_9}, {v_9, dest->subset({1})}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_16}, {v_14, v_9}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_17}, {v_14, v_9}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_19}, {dest->subset({1}), v_10}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_9}, {dest->subset({1}), v_9}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_18}, {dest->subset({1}), v_10}, {}, ""));
        co_yield VectorAddUInt32CarryOut(m_context, v_9, v_9, v_19);
        co_yield VectorAddUInt32CarryInOut(m_context, v_18, v_11, v_18);
        co_yield VectorAddUInt32CarryOut(m_context, v_9, v_9, v_17);
        co_yield_(Instruction("v_mul_hi_u32", {v_13}, {v_14, v_10}, {}, ""));
        co_yield VectorAddUInt32CarryInOut(m_context, v_9, v_18, v_16);
        co_yield VectorAddUInt32CarryInOut(m_context, v_13, v_13, v_12);
        co_yield_(Instruction("v_mul_lo_u32", {v_10}, {v_14, v_10}, {}, ""));
        co_yield VectorAddUInt32CarryOut(m_context, v_9, v_9, v_10);
        co_yield VectorAddUInt32CarryInOut(m_context, v_10, v_11, v_13);
        co_yield VectorAddUInt32(m_context, v_8, v_8, v_15);
        co_yield VectorAddUInt32CarryInOut(m_context, v_8, VCC, s_5, v_8, v_10);
        co_yield VectorAddUInt32CarryOut(m_context, dest->subset({1}), dest->subset({1}), v_9);
        co_yield VectorAddUInt32CarryInOut(m_context, v_8, Register::Value::Literal(0), v_8);
        co_yield_(Instruction("v_ashrrev_i32", {v_9}, {Register::Value::Literal(31), v_2}, {}, ""));
        co_yield VectorAddUInt32CarryOut(m_context, v_20, v_20, v_9);
        co_yield_(Instruction("v_xor_b32", {v_20}, {v_20, v_9}, {}, ""));
        co_yield VectorAddUInt32CarryInOut(m_context, v_2, v_2, v_9);
        co_yield_(Instruction("v_mul_lo_u32", {v_14}, {v_20, v_8}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_15}, {v_20, dest->subset({1})}, {}, ""));
        co_yield_(Instruction("v_xor_b32", {v_2}, {v_2, v_9}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_10}, {v_20, v_8}, {}, ""));
        co_yield VectorAddUInt32CarryOut(m_context, v_14, v_15, v_14);
        co_yield VectorAddUInt32CarryInOut(m_context, v_10, v_11, v_10);
        co_yield_(Instruction("v_mul_hi_u32", {v_13}, {v_2, dest->subset({1})}, {}, ""));
        co_yield_(
            Instruction("v_mul_lo_u32", {dest->subset({1})}, {v_2, dest->subset({1})}, {}, ""));
        co_yield VectorAddUInt32CarryOut(m_context, dest->subset({1}), v_14, dest->subset({1}));
        co_yield_(Instruction("v_mul_hi_u32", {v_15}, {v_2, v_8}, {}, ""));
        co_yield VectorAddUInt32CarryInOut(m_context, dest->subset({1}), v_10, v_13);
        co_yield VectorAddUInt32CarryInOut(m_context, v_10, v_15, v_12);
        co_yield_(Instruction("v_mul_lo_u32", {v_8}, {v_2, v_8}, {}, ""));
        co_yield VectorAddUInt32CarryOut(m_context, dest->subset({1}), dest->subset({1}), v_8);
        co_yield VectorAddUInt32CarryInOut(m_context, v_8, v_11, v_10);
        co_yield_(Instruction("v_mul_lo_u32", {v_10}, {v_3, dest->subset({1})}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_14}, {v_7, v_8}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_15}, {v_7, dest->subset({1})}, {}, ""));
        co_yield VectorAdd3UInt32(m_context, v_10, v_15, v_14, v_10);
        co_yield_(Instruction("v_mul_lo_u32", {v_15}, {v_7, dest->subset({1})}, {}, ""));
        co_yield VectorSubUInt32(m_context, v_14, v_2, v_10);
        co_yield VectorSubUInt32CarryOut(m_context, v_20, v_20, v_15);
        co_yield VectorSubUInt32CarryInOut(m_context, v_14, s_5, VCC, v_14, v_3);
        co_yield VectorSubUInt32CarryOut(m_context, v_15, s_5, v_20, v_7);
        co_yield VectorSubRevUInt32CarryInOut(
            m_context, v_14, s_5, s_5, Register::Value::Literal(0), v_14);
        co_yield_(Instruction("v_cmp_ge_u32", {s_5}, {v_14, v_3}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32",
                              {v_13},
                              {Register::Value::Literal(0), Register::Value::Literal(-1), s_5},
                              {},
                              ""));
        co_yield_(Instruction("v_cmp_ge_u32", {s_5}, {v_15, v_7}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32",
                              {v_15},
                              {Register::Value::Literal(0), Register::Value::Literal(-1), s_5},
                              {},
                              ""));
        co_yield_(Instruction("v_cmp_eq_u32", {s_5}, {v_14, v_3}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32", {v_14}, {v_13, v_15, s_5}, {}, ""));
        co_yield VectorAddUInt32CarryOut(
            m_context, v_15, s_5, Register::Value::Literal(2), dest->subset({1}));
        co_yield VectorSubUInt32CarryInOut(m_context, v_2, v_2, v_10);
        co_yield VectorAddUInt32CarryInOut(
            m_context, v_13, s_5, s_5, Register::Value::Literal(0), v_8);
        co_yield_(Instruction("v_cmp_ge_u32", {m_context->getVCC()}, {v_2, v_3}, {}, ""));
        co_yield VectorAddUInt32CarryOut(
            m_context, v_16, s_5, Register::Value::Literal(1), dest->subset({1}));
        co_yield_(Instruction(
            "v_cndmask_b32",
            {v_10},
            {Register::Value::Literal(0), Register::Value::Literal(-1), m_context->getVCC()},
            {},
            ""));
        co_yield_(Instruction("v_cmp_ge_u32", {m_context->getVCC()}, {v_20, v_7}, {}, ""));
        co_yield VectorAddUInt32CarryInOut(
            m_context, v_17, s_5, s_5, Register::Value::Literal(0), v_8);
        co_yield_(Instruction(
            "v_cndmask_b32",
            {v_20},
            {Register::Value::Literal(0), Register::Value::Literal(-1), m_context->getVCC()},
            {},
            ""));
        co_yield_(Instruction("v_cmp_eq_u32", {m_context->getVCC()}, {v_2, v_3}, {}, ""));
        co_yield_(Instruction("v_cmp_ne_u32", {s_5}, {Register::Value::Literal(0), v_14}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32", {v_20}, {v_10, v_20, m_context->getVCC()}, {}, ""));
        co_yield_(Instruction(
            "v_cmp_ne_u32", {m_context->getVCC()}, {Register::Value::Literal(0), v_20}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32", {v_2}, {v_16, v_15, s_5}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32", {v_14}, {v_17, v_13, s_5}, {}, ""));
        co_yield_(Instruction(
            "v_cndmask_b32", {v_2}, {dest->subset({1}), v_2, m_context->getVCC()}, {}, ""));
        co_yield_(Instruction("v_xor_b32", {v_7}, {v_9, dest->subset({0})}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32", {v_20}, {v_8, v_14, m_context->getVCC()}, {}, ""));
        co_yield_(Instruction("v_xor_b32", {v_2}, {v_2, v_7}, {}, ""));
        co_yield_(Instruction("v_xor_b32", {v_20}, {v_20, v_7}, {}, ""));
        co_yield VectorSubUInt32CarryOut(m_context, dest->subset({0}), v_2, v_7);
        co_yield VectorSubUInt32CarryInOut(m_context, dest->subset({1}), v_20, v_7);
        co_yield_(Instruction::Label(label_21));
        if(wavefrontSize == 64)
        {
            co_yield_(Instruction("s_or_saveexec_b64", {s_5}, {s_6}, {}, ""));
            co_yield_(Instruction("s_xor_b64", {EXEC}, {EXEC, s_5}, {}, ""));
        }
        else
        {
            co_yield_(Instruction("s_or_saveexec_b32", {s_5}, {s_6}, {}, ""));
            co_yield_(Instruction("s_xor_b32", {EXEC}, {EXEC, s_5}, {}, ""));
        }
        co_yield_(Instruction("s_cbranch_execz", {}, {label_22}, {}, ""));
        co_yield_(Instruction("v_cvt_f32_u32", {v_2}, {v_7}, {}, ""));
        co_yield VectorSubUInt32(m_context, v_3, Register::Value::Literal(0), v_7);
        co_yield m_context->copier()->copy(dest->subset({1}), Register::Value::Literal(0), "");
        co_yield_(Instruction("v_rcp_iflag_f32", {v_2}, {v_2}, {}, ""));
        co_yield_(
            Instruction("v_mul_f32", {v_2}, {Register::Value::Literal(0x4f7ffffe), v_2}, {}, ""));
        co_yield_(Instruction("v_cvt_u32_f32", {v_2}, {v_2}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_3}, {v_3, v_2}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_3}, {v_2, v_3}, {}, ""));
        co_yield VectorAddUInt32(m_context, v_2, v_2, v_3);
        co_yield_(Instruction("v_mul_hi_u32", {v_2}, {v_20, v_2}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_3}, {v_2, v_7}, {}, ""));
        co_yield VectorSubUInt32(m_context, v_20, v_20, v_3);
        co_yield VectorAddUInt32(m_context, dest->subset({0}), Register::Value::Literal(1), v_2);
        co_yield VectorSubUInt32(m_context, v_3, v_20, v_7);
        co_yield_(Instruction("v_cmp_ge_u32", {m_context->getVCC()}, {v_20, v_7}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32", {v_20}, {v_20, v_3, m_context->getVCC()}, {}, ""));
        co_yield_(Instruction(
            "v_cndmask_b32", {v_2}, {v_2, dest->subset({0}), m_context->getVCC()}, {}, ""));
        co_yield VectorAddUInt32(m_context, v_3, Register::Value::Literal(1), v_2);
        co_yield_(Instruction("v_cmp_ge_u32", {m_context->getVCC()}, {v_20, v_7}, {}, ""));
        co_yield_(Instruction(
            "v_cndmask_b32", {dest->subset({0})}, {v_2, v_3, m_context->getVCC()}, {}, ""));
        co_yield_(Instruction::Label(label_22));
        if(wavefrontSize == 64)
        {
            co_yield_(Instruction("s_or_b64", {EXEC}, {EXEC, s_5}, {}, ""));
        }
        else
        {
            co_yield_(Instruction("s_or_b32", {EXEC}, {EXEC, s_5}, {}, ""));
        }
        co_yield(Instruction::Unlock("End of Divide64(VCC)"));
        co_yield(Instruction::Unlock("End of Divide64(SCC)"));
    }
}
