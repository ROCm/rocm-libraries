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
#include <rocRoller/CodeGen/Arithmetic/Multiply.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    // Register supported components
    RegisterComponentTemplateSpec(MultiplyGenerator, Register::Type::Scalar, DataType::Int32);
    RegisterComponentTemplateSpec(MultiplyGenerator, Register::Type::Vector, DataType::Int32);
    RegisterComponentTemplateSpec(MultiplyGenerator, Register::Type::Scalar, DataType::Int64);
    RegisterComponentTemplateSpec(MultiplyGenerator, Register::Type::Vector, DataType::Int64);
    RegisterComponentTemplateSpec(MultiplyGenerator, Register::Type::Vector, DataType::Halfx2);
    RegisterComponentTemplateSpec(MultiplyGenerator, Register::Type::Vector, DataType::Float);
    RegisterComponentTemplateSpec(MultiplyGenerator, Register::Type::Vector, DataType::Double);

    template <>
    std::shared_ptr<BinaryArithmeticGenerator<Expression::Multiply>>
        GetGenerator<Expression::Multiply>(Register::ValuePtr dst,
                                           Register::ValuePtr lhs,
                                           Register::ValuePtr rhs,
                                           Expression::Multiply const&)
    {
        // Choose the proper generator, based on the context, register type
        // and datatype.
        return Component::Get<BinaryArithmeticGenerator<Expression::Multiply>>(
            getContextFromValues(dst, lhs, rhs),
            promoteRegisterType(dst, lhs, rhs),
            promoteDataType(dst, lhs, rhs));
    }

    template <>
    Generator<Instruction> MultiplyGenerator<Register::Type::Scalar, DataType::Int32>::generate(
        Register::ValuePtr dest,
        Register::ValuePtr lhs,
        Register::ValuePtr rhs,
        Expression::Multiply const&)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("s_mul_i32", {dest}, {lhs, rhs}, {}, ""));
    }

    template <>
    Generator<Instruction> MultiplyGenerator<Register::Type::Vector, DataType::Int32>::generate(
        Register::ValuePtr dest,
        Register::ValuePtr lhs,
        Register::ValuePtr rhs,
        Expression::Multiply const&)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        // TODO Is this necessary?
        if(rhs->regType() == Register::Type::Literal)
            co_yield m_context->copier()->ensureType(rhs, rhs, Register::Type::Vector);
        co_yield_(Instruction("v_mul_lo_u32", {dest}, {lhs, rhs}, {}, ""));
    }

    template <>
    Generator<Instruction> MultiplyGenerator<Register::Type::Scalar, DataType::Int64>::generate(
        Register::ValuePtr dest,
        Register::ValuePtr lhs,
        Register::ValuePtr rhs,
        Expression::Multiply const&)
    {
        co_yield describeOpArgs("dest", dest, "lhs", lhs, "rhs", rhs);

        Register::ValuePtr l0, l1, r0, r1;
        co_yield get2DwordsScalar(l0, l1, lhs);
        co_yield get2DwordsScalar(r0, r1, rhs);

        AssertFatal(l0);
        AssertFatal(l1);
        AssertFatal(r0);
        AssertFatal(r1);

        // See the comments in the Vector Int64 generator below for more info

        AssertFatal(dest->regType() == Register::Type::Scalar);
        AssertFatal(dest->registerCount() == 2);

        // cppcheck-suppress nullPointer
        bool needLL = !l0->isZeroLiteral() && !r0->isZeroLiteral();
        // cppcheck-suppress nullPointer
        bool needLH = !l0->isZeroLiteral() && !r1->isZeroLiteral();
        // cppcheck-suppress nullPointer
        bool needHL = !l1->isZeroLiteral() && !r0->isZeroLiteral();

        int numHighComponents = 0;

        std::array<Register::ValuePtr, 3> highComponents
            = {dest->subset({1}),
               Register::Value::Placeholder(m_context, Register::Type::Scalar, DataType::Int32, 1),
               Register::Value::Placeholder(m_context, Register::Type::Scalar, DataType::Int32, 1)};

        if(needLH)
        {
            co_yield_(Instruction("s_mul_i32",
                                  {highComponents[numHighComponents]},
                                  {l0, r1},
                                  {},
                                  "most significant: low of low * high"));
            numHighComponents++;
        }
        else
        {
            co_yield Instruction::Comment("low of low * high omitted due to zero input.");
        }

        if(needHL)
        {
            co_yield_(Instruction("s_mul_i32",
                                  {highComponents[numHighComponents]},
                                  {l1, r0},
                                  {},
                                  "most significant: low of high * low"));
            numHighComponents++;
        }
        else
        {
            co_yield Instruction::Comment("low of high * low omitted due to zero input.");
        }

        if(needLL)
        {
            co_yield_(
                Instruction("s_mul_i32", {dest->subset({0})}, {l0, r0}, {}, "least significant"));

            co_yield_(Instruction("s_mul_hi_u32",
                                  {highComponents[numHighComponents]},
                                  {l0, r0},
                                  {},
                                  "most significant: high of low * low"));
            numHighComponents++;
        }
        else
        {
            co_yield m_context->copier()->copy(dest->subset({0}),
                                               Register::Value::Literal(0),
                                               "low * low optimized due to zero input");
        }

        if(numHighComponents == 0)
        {
            co_yield m_context->copier()->copy(dest->subset({1}),
                                               Register::Value::Literal(0),
                                               "high * high optimized due to zero input");
        }
        else if(numHighComponents == 1)
        {
            // The first high multiply writes into the high bits of the output, nothing to do.
            co_yield Instruction::Comment(
                "Most significant: sum omitted due to only one valid input.");
        }
        else if(numHighComponents >= 2)
        {
            co_yield ScalarAddUInt32(m_context,
                                     dest->subset({1}),
                                     highComponents[0],
                                     highComponents[1],
                                     "most significant: sum");

            if(numHighComponents == 3)
            {
                co_yield ScalarAddUInt32(m_context,
                                         dest->subset({1}),
                                         dest->subset({1}),
                                         highComponents[2],
                                         "most significant: sum");
            }
        }
        else
        {
            Throw<FatalError>(
                concatenate("Shouldn't get here: numHighComponents = ", numHighComponents));
        }
    }

    template <>
    Generator<Instruction> MultiplyGenerator<Register::Type::Vector, DataType::Int64>::generate(
        Register::ValuePtr dest,
        Register::ValuePtr lhs,
        Register::ValuePtr rhs,
        Expression::Multiply const&)
    {
        co_yield describeOpArgs("dest", dest, "lhs", lhs, "rhs", rhs);

        Register::ValuePtr l0, l1, r0, r1;
        co_yield get2DwordsVector(l0, l1, lhs);
        co_yield get2DwordsVector(r0, r1, rhs);

        AssertFatal(l0);
        AssertFatal(l1);
        AssertFatal(r0);
        AssertFatal(r1);

        // The high bits of the output consist of the sum of:
        //   - high bits of low * low,
        //   - low bits of low * high,
        //   - low bits of high * low.
        //
        // If any of these are known to be zero, they can be omitted.
        // If they are all zero, we still need to write a zero into the output.
        // Whichever of these is generated first will write into the destination,
        // additional required instructions will require registers to be allocated.

        // cppcheck-suppress nullPointer
        bool needLL = !l0->isZeroLiteral() && !r0->isZeroLiteral();
        // cppcheck-suppress nullPointer
        bool needLH = !l0->isZeroLiteral() && !r1->isZeroLiteral();
        // cppcheck-suppress nullPointer
        bool needHL = !l1->isZeroLiteral() && !r0->isZeroLiteral();

        // Multiply instructions only accept constant values, not literals.

        if(l0->regType() == Register::Type::Literal
           && !m_context->targetArchitecture().isSupportedConstantValue(l0))
            co_yield moveToVGPR(l0);

        if(l1->regType() == Register::Type::Literal
           && !m_context->targetArchitecture().isSupportedConstantValue(l1))
            co_yield moveToVGPR(l1);

        if(r0->regType() == Register::Type::Literal
           && !m_context->targetArchitecture().isSupportedConstantValue(r0))
            co_yield moveToVGPR(r0);

        if(r1->regType() == Register::Type::Literal
           && !m_context->targetArchitecture().isSupportedConstantValue(r1))
            co_yield moveToVGPR(r1);

        // These instructions don't support scalar * scalar.
        if(l0->regType() == Register::Type::Scalar && r0->regType() == Register::Type::Scalar)
        {
            co_yield moveToVGPR(l0);
            if(l1->regType() == Register::Type::Scalar)
                co_yield moveToVGPR(l1);
        }

        AssertFatal(dest->regType() == Register::Type::Vector);
        AssertFatal(dest->registerCount() == 2);

        int numHighComponents = 0;

        std::array<Register::ValuePtr, 3> highComponents
            = {dest->subset({1}),
               Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Int32, 1),
               Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Int32, 1)};

        if(needLH)
        {
            co_yield_(Instruction("v_mul_lo_u32",
                                  {highComponents[numHighComponents]},
                                  {l0, r1},
                                  {},
                                  "most significant: low of low * high"));
            numHighComponents++;
        }
        else
        {
            co_yield Instruction::Comment("low of low * high omitted due to zero input.");
        }

        if(needHL)
        {
            co_yield_(Instruction("v_mul_lo_u32",
                                  {highComponents[numHighComponents]},
                                  {l1, r0},
                                  {},
                                  "most significant: low of high * low"));
            numHighComponents++;
        }
        else
        {
            co_yield Instruction::Comment("low of high * low omitted due to zero input.");
        }

        if(needLL)
        {
            co_yield_(Instruction("v_mul_hi_u32",
                                  {highComponents[numHighComponents]},
                                  {l0, r0},
                                  {},
                                  "most significant: high of low * low"));
            numHighComponents++;

            co_yield_(Instruction("v_mul_lo_u32",
                                  {dest->subset({0})},
                                  {l0, r0},
                                  {},
                                  "least significant: low of low * low"));
        }
        else
        {
            co_yield m_context->copier()->copy(dest->subset({0}),
                                               Register::Value::Literal(0),
                                               "low * low optimized due to zero input.");
        }

        if(numHighComponents == 0)
        {
            co_yield m_context->copier()->copy(dest->subset({1}),
                                               Register::Value::Literal(0),
                                               "high bits of output optimized due to zero input.");
        }
        else if(numHighComponents == 1)
        {
            // The first high multiply writes into the high bits of the output, nothing to do.
            co_yield Instruction::Comment(
                "Most significant: sum omitted due to only one valid input.");
        }
        else if(numHighComponents == 2)
        {
            co_yield VectorAddUInt32(m_context,
                                     dest->subset({1}),
                                     highComponents[0],
                                     highComponents[1],
                                     "most significant: sum");
        }
        else if(numHighComponents == 3)
        {
            co_yield VectorAdd3UInt32(m_context,
                                      dest->subset({1}),
                                      highComponents[0],
                                      highComponents[1],
                                      highComponents[2],
                                      "most significant: sum");
        }
        else
        {
            Throw<FatalError>(
                concatenate("Shouldn't get here: numHighComponents = ", numHighComponents));
        }
    }

    template <>
    Generator<Instruction> MultiplyGenerator<Register::Type::Vector, DataType::Halfx2>::generate(
        Register::ValuePtr dest,
        Register::ValuePtr lhs,
        Register::ValuePtr rhs,
        Expression::Multiply const&)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_pk_mul_f16", {dest}, {lhs, rhs}, {}, ""));
    }

    template <>
    Generator<Instruction> MultiplyGenerator<Register::Type::Vector, DataType::Float>::generate(
        Register::ValuePtr dest,
        Register::ValuePtr lhs,
        Register::ValuePtr rhs,
        Expression::Multiply const&)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_mul_f32", {dest}, {lhs, rhs}, {}, ""));
    }

    template <>
    Generator<Instruction> MultiplyGenerator<Register::Type::Vector, DataType::Double>::generate(
        Register::ValuePtr dest,
        Register::ValuePtr lhs,
        Register::ValuePtr rhs,
        Expression::Multiply const&)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_mul_f64", {dest}, {lhs, rhs}, {}, ""));
    }

}
