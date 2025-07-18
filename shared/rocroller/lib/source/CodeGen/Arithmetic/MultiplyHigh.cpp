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

#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Arithmetic/MultiplyHigh.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    RegisterComponent(MultiplyHighGenerator);

    template <>
    std::shared_ptr<BinaryArithmeticGenerator<Expression::MultiplyHigh>>
        GetGenerator<Expression::MultiplyHigh>(Register::ValuePtr dst,
                                               Register::ValuePtr lhs,
                                               Register::ValuePtr rhs,
                                               Expression::MultiplyHigh const&)
    {
        return Component::Get<BinaryArithmeticGenerator<Expression::MultiplyHigh>>(
            getContextFromValues(dst, lhs, rhs), dst->regType(), dst->variableType().dataType);
    }

    Generator<Instruction> MultiplyHighGenerator::generate(Register::ValuePtr dest,
                                                           Register::ValuePtr lhs,
                                                           Register::ValuePtr rhs,
                                                           Expression::MultiplyHigh const&)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        auto const& dataTypeInfoLhs  = DataTypeInfo::Get(lhs->variableType());
        auto const& dataTypeInfoRhs  = DataTypeInfo::Get(rhs->variableType());
        auto const& dataTypeInfoDest = DataTypeInfo::Get(dest->variableType());

        if(dataTypeInfoLhs.elementBits == 32u && dataTypeInfoRhs.elementBits == 32u)
        {
            if(dest->regType() == Register::Type::Scalar)
            {
                if(dataTypeInfoDest.isSigned)
                {
                    co_yield_(Instruction("s_mul_hi_i32", {dest}, {lhs, rhs}, {}, ""));
                }
                else
                {
                    co_yield_(Instruction("s_mul_hi_u32", {dest}, {lhs, rhs}, {}, ""));
                }
            }
            else if(dest->regType() == Register::Type::Vector)
            {
                if(dataTypeInfoDest.isSigned)
                {
                    if(rhs->regType() == Register::Type::Literal
                       && !m_context->targetArchitecture().isSupportedConstantValue(rhs))
                        co_yield m_context->copier()->ensureType(rhs, rhs, Register::Type::Vector);
                    co_yield_(Instruction("v_mul_hi_i32", {dest}, {lhs, rhs}, {}, ""));
                }
                else
                {
                    co_yield_(Instruction("v_mul_hi_u32", {dest}, {lhs, rhs}, {}, ""));
                }
            }
        }
        else if(dataTypeInfoDest.elementBits == 64u)
        {
            std::string mulHiInst
                = dest->regType() == Register::Type::Vector ? "v_mul_hi_u32" : "s_mul_hi_u32";
            // libdivide algorithm:

            // uint32_t mask = 0xFFFFFFFF;
            // uint32_t x0 = (uint32_t)(x & mask);
            // uint32_t y0 = (uint32_t)(y & mask);
            // int32_t x1 = (int32_t)(x >> 32);
            // int32_t y1 = (int32_t)(y >> 32);
            // uint32_t x0y0_hi = libdivide_mullhi_u32(x0, y0);
            // int64_t t = x1 * (int64_t)y0 + x0y0_hi;
            // int64_t w1 = x0 * (int64_t)y1 + (t & mask);

            // return x1 * (int64_t)y1 + (t >> 32) + (w1 >> 32);

            Register::ValuePtr x0, x1, y0, y1;

            co_yield get2DwordsVector(x0, x1, lhs);
            co_yield get2DwordsVector(y0, y1, rhs);

            x0->setVariableType(DataType::UInt32);
            x1->setVariableType(DataType::Int32);
            y0->setVariableType(DataType::UInt32);
            y1->setVariableType(DataType::Int32);

            auto t
                = std::make_shared<Register::Value>(m_context, dest->regType(), DataType::Int64, 1);
            {
                auto x0y0Hi = std::make_shared<Register::Value>(
                    m_context, dest->regType(), DataType::Int64, 1);

                co_yield_(Instruction(
                    mulHiInst, {x0y0Hi->subset({0})}, {x0, y0}, {}, "mul_hi: x0*y0 high bits"));

                co_yield m_context->copier()->copy(x0y0Hi->subset({1}),
                                                   Register::Value::Literal(0));

                auto x1y0 = std::make_shared<Register::Value>(
                    m_context, dest->regType(), DataType::Int64, 1);
                co_yield generateOp<Expression::Multiply>(x1y0, x1, y0);

                co_yield generateOp<Expression::Add>(t, x1y0, x0y0Hi);
            }

            auto w1
                = std::make_shared<Register::Value>(m_context, dest->regType(), DataType::Int64, 1);
            {
                auto x0y1 = std::make_shared<Register::Value>(
                    m_context, dest->regType(), DataType::Int64, 1);
                co_yield generateOp<Expression::Multiply>(x0y1, x0, y1);

                co_yield generateOp<Expression::Add>(w1, x0y1, t->subset({0}));
            }

            co_yield generateOp<Expression::ArithmeticShiftR>(t, t, Register::Value::Literal(32));
            co_yield generateOp<Expression::ArithmeticShiftR>(w1, w1, Register::Value::Literal(32));

            auto x1y1
                = std::make_shared<Register::Value>(m_context, dest->regType(), DataType::Int64, 1);
            co_yield generateOp<Expression::Multiply>(x1y1, x1, y1);

            co_yield generateOp<Expression::Add>(dest, x1y1, t);
            co_yield generateOp<Expression::Add>(dest, dest, w1);
        }
        else
        {
            Throw<FatalError>("Unsupported register type for multiply high operation: ",
                              ShowValue(dest->regType()));
        }
    }
}
