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
#include <rocRoller/CodeGen/Arithmetic/Utility.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>

namespace rocRoller
{
    // Move a value to a single VGPR register.
    Generator<Instruction> ArithmeticGenerator::moveToVGPR(Register::ValuePtr& val)
    {
        Register::ValuePtr tmp = val;

        val = Register::Value::Placeholder(
            m_context, Register::Type::Vector, tmp->variableType(), 1);

        co_yield m_context->copier()->copy(val, tmp, "");
    }

    Generator<Instruction> ArithmeticGenerator::signExtendDWord(Register::ValuePtr dst,
                                                                Register::ValuePtr src)
    {
        auto l31 = Register::Value::Literal(31);

        co_yield generateOp<Expression::ArithmeticShiftR>(dst, src, l31);
    }

    Generator<Instruction> ArithmeticGenerator::get2DwordsScalar(Register::ValuePtr& lsd,
                                                                 Register::ValuePtr& msd,
                                                                 Register::ValuePtr  input)
    {
        if(input->regType() == Register::Type::Literal)
        {
            Arithmetic::get2LiteralDwords(lsd, msd, input);
            co_return;
        }

        if(input->regType() == Register::Type::Scalar)
        {
            auto varType = input->variableType();

            if(varType == DataType::Int32)
            {
                lsd = input;

                msd = Register::Value::Placeholder(
                    m_context, Register::Type::Scalar, DataType::Int32, 1);

                co_yield signExtendDWord(msd, input);

                co_return;
            }

            if(varType == DataType::UInt32
               || (varType == DataType::Raw32 && input->valueCount() == 1))
            {
                lsd = input->subset({0});
                msd = Register::Value::Literal(0);
                co_return;
            }

            if(varType == DataType::Raw32 && input->valueCount() >= 2)
            {
                lsd = input->subset({0});
                msd = input->subset({1});
                co_return;
            }

            if(varType.pointerType == PointerType::PointerGlobal || varType == DataType::Int64
               || varType == DataType::UInt64)
            {
                lsd = input->subset({0});
                msd = input->subset({1});
                co_return;
            }
        }

        Throw<FatalError>(
            concatenate("get2DwordsScalar: Conversion not implemented for register type ",
                        input->regType(),
                        "/",
                        input->variableType()));
    }

    Generator<Instruction> ArithmeticGenerator::get2DwordsVector(Register::ValuePtr& lsd,
                                                                 Register::ValuePtr& msd,
                                                                 Register::ValuePtr  input)
    {
        if(input->regType() == Register::Type::Literal)
        {
            Arithmetic::get2LiteralDwords(lsd, msd, input);
            co_return;
        }

        if(input->regType() == Register::Type::Scalar)
        {
            co_yield get2DwordsScalar(lsd, msd, input);
            co_return;
        }

        if(input->regType() == Register::Type::Vector)
        {
            auto varType = input->variableType();

            if(varType == DataType::Int32)
            {
                lsd = input;

                msd = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Raw32, 1);

                co_yield signExtendDWord(msd, input);

                co_return;
            }

            if(varType == DataType::UInt32
               || (varType == DataType::Raw32 && input->valueCount() == 1))
            {
                lsd = input->subset({0});
                msd = Register::Value::Literal(0);
                co_return;
            }

            if(varType == DataType::Raw32 && input->valueCount() >= 2)
            {
                lsd = input->subset({0});
                msd = input->subset({1});
                co_return;
            }

            if(varType.pointerType == PointerType::PointerGlobal || varType == DataType::Int64
               || varType == DataType::UInt64)
            {
                lsd = input->subset({0});
                msd = input->subset({1});
                co_return;
            }
        }

        Throw<FatalError>(
            concatenate("get2DwordsVector: Conversion not implemented for register type ",
                        input->regType(),
                        "/",
                        input->variableType()));
    }

    Generator<Instruction> ArithmeticGenerator::describeOpArgs(std::string const& argName0,
                                                               Register::ValuePtr arg0,
                                                               std::string const& argName1,
                                                               Register::ValuePtr arg1,
                                                               std::string const& argName2,
                                                               Register::ValuePtr arg2)
    {
        auto        opDesc = name() + ": ";
        std::string indent(opDesc.size(), ' ');

        co_yield Instruction::Comment(
            concatenate(opDesc, argName0, " (", arg0->description(), ") = "));
        co_yield Instruction::Comment(
            concatenate(indent, argName1, " (", arg1->description(), ") "));

        if(arg2)
            co_yield Instruction::Comment(
                concatenate(indent, argName2, " (", arg2->description(), ")"));
    }

    Generator<Instruction>
        ArithmeticGenerator::scalarCompareThroughVALU(std::string const  instruction,
                                                      Register::ValuePtr dst,
                                                      Register::ValuePtr lhs,
                                                      Register::ValuePtr rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        Register::ValuePtr tmp;
        co_yield m_context->copier()->ensureType(
            tmp, rhs, {Register::Type::Vector, Register::Type::Literal});

        auto wfp = Register::Value::WavefrontPlaceholder(m_context);

        co_yield_(Instruction(instruction, {wfp}, {lhs, tmp}, {}, ""));

        auto reduce = m_context->kernel()->wavefront_size() == 64 ? "s_and_b64" : "s_and_b32";
        if(dst != nullptr && !dst->isSCC())
        {
            co_yield(Instruction::Lock(Scheduling::Dependency::SCC,
                                       "Start Compare writing to non-SCC dest"));
        }
        co_yield_(Instruction(reduce, {wfp}, {wfp, m_context->getExec()}, {}, ""));
        if(dst != nullptr && !dst->isSCC())
        {
            co_yield m_context->copier()->copy(dst, m_context->getSCC(), "");
            co_yield(Instruction::Unlock("End Compare writing to non-SCC dest"));
        }
    }

    // -----------------------------
    // Helper Functions

    DataType getArithDataType(Register::ValuePtr const reg)
    {
        AssertFatal(reg != nullptr, "Null argument");

        auto variableType = reg->variableType();

        if(variableType == DataType::Raw32 && reg->registerCount() == 2)
        {
            return DataType::UInt64;
        }

        return variableType.getArithmeticType();
    }

    DataType promoteDataType(Register::ValuePtr const dst,
                             Register::ValuePtr       lhs,
                             Register::ValuePtr       rhs)
    {
        AssertFatal(lhs != nullptr, "Null argument");
        AssertFatal(rhs != nullptr, "Null argument");

        auto lhsVarType = lhs->variableType() == DataType::Raw32 && lhs->registerCount() == 2
                              ? DataType::UInt64
                              : lhs->variableType();
        auto rhsVarType = rhs->variableType() == DataType::Raw32 && rhs->registerCount() == 2
                              ? DataType::UInt64
                              : rhs->variableType();
        auto varType    = VariableType::Promote(lhsVarType, rhsVarType);

        if(dst)
        {
            auto dstVarType = dst->variableType();
            if(varType != dstVarType && dstVarType.dataType != DataType::Raw32)
            {
                auto const& varTypeInfo    = DataTypeInfo::Get(varType);
                auto const& dstVarTypeInfo = DataTypeInfo::Get(dstVarType);

                AssertFatal(varTypeInfo.elementBits <= dstVarTypeInfo.elementBits
                                && varTypeInfo.isIntegral == dstVarTypeInfo.isIntegral,
                            ShowValue(varType),
                            ShowValue(dstVarType));

                varType = dstVarType;
            }
        }

        return varType.isPointer() ? DataType::UInt64 : varType.dataType;
    }

    Register::Type
        promoteRegisterType(Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs)
    {
        if(dst)
            return dst->regType();

        AssertFatal(lhs != nullptr, "Null argument");
        AssertFatal(rhs != nullptr, "Null argument");

        auto regType = Register::PromoteType(lhs->regType(), rhs->regType());

        return regType;
    }
}
