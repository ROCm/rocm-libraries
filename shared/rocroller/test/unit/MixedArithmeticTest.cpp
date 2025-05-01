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

#include <cmath>
#include <memory>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Utilities/Generator.hpp>
#include <rocRoller/Utilities/HipUtils.hpp>

#include "GPUContextFixture.hpp"
#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"
#include "Utilities.hpp"
#include <common/TestValues.hpp>
#include <rocRoller/DataTypes/DataTypes.hpp>

using namespace rocRoller;

namespace MixedArithmeticTest
{
    struct BinaryArgument
    {
        Register::Type                    lhsRegType;
        VariableType                      lhsVarType;
        std::vector<CommandArgumentValue> lhsValues;

        Register::Type                    rhsRegType;
        VariableType                      rhsVarType;
        std::vector<CommandArgumentValue> rhsValues;

        Register::Type resultRegType;
        VariableType   resultVarType;
    };

    std::ostream& operator<<(std::ostream& stream, BinaryArgument const& arg)
    {
        stream << "LHS: " << arg.lhsRegType << ", " << arg.lhsVarType << ", "
               << "RHS: " << arg.rhsRegType << ", " << arg.rhsVarType << ", "
               << "Result: " << arg.resultRegType << ", " << arg.resultVarType;
        return stream;
    }

    struct TernaryArgument
    {
        Register::Type                    arg1RegType;
        VariableType                      arg1VarType;
        std::vector<CommandArgumentValue> arg1Values;

        Register::Type                    arg2RegType;
        VariableType                      arg2VarType;
        std::vector<CommandArgumentValue> arg2Values;

        Register::Type                    arg3RegType;
        VariableType                      arg3VarType;
        std::vector<CommandArgumentValue> arg3Values;

        Register::Type resultRegType;
        VariableType   resultVarType;
    };

    std::ostream& operator<<(std::ostream& stream, TernaryArgument const& arg)
    {
        stream << "ARG1: " << arg.arg1RegType << ", " << arg.arg1VarType << ", "
               << "ARG2: " << arg.arg2RegType << ", " << arg.arg2VarType << ", "
               << "ARG3: " << arg.arg3RegType << ", " << arg.arg3VarType << ", "
               << "Result: " << arg.resultRegType << ", " << arg.resultVarType;
        return stream;
    }

    std::shared_ptr<uint8_t> createDeviceArray(std::vector<CommandArgumentValue> const& values)
    {
        KernelArguments args;

        for(size_t i = 0; i < values.size(); i++)
        {
            args.append(concatenate("a", i), values[i]);
        }

        auto rv = make_shared_device<uint8_t>(args.size());
        HIP_CHECK(hipMemcpy(rv.get(), args.data(), args.size(), hipMemcpyHostToDevice));

        return rv;
    }

    template <typename T>
    std::vector<CommandArgumentValue> getDeviceValues(std::shared_ptr<void> array, size_t count)
    {
        std::vector<T> hostArray(count);

        HIP_CHECK(
            hipMemcpy(hostArray.data(), array.get(), count * sizeof(T), hipMemcpyDeviceToHost));

        std::vector<CommandArgumentValue> rv;
        rv.reserve(count);

        for(size_t i = 0; i < count; i++)
        {
            rv.emplace_back(hostArray[i]);
        }

        return rv;
    }

    std::vector<CommandArgumentValue>
        getDeviceValues(std::shared_ptr<void> array, size_t count, VariableType type)
    {
        if(type == DataType::Int64)
        {
            return getDeviceValues<int64_t>(array, count);
        }
        else if(type == DataType::Int32)
        {
            return getDeviceValues<int>(array, count);
        }
        else if(type == DataType::Float)
        {
            return getDeviceValues<float>(array, count);
        }
        else if(type == DataType::Double)
        {
            return getDeviceValues<double>(array, count);
        }

        throw std::runtime_error(concatenate("Not implemented for ", type));
    }

    std::shared_ptr<void> allocateSingleDeviceValue(VariableType type)
    {
        if(type == DataType::Int64)
        {
            return make_shared_device<int64_t>(1);
        }

        throw std::runtime_error(concatenate("Not implemented for ", type));
    }

    template <CCommandArgumentValue T>
    CommandArgumentValue getResultValue(std::shared_ptr<void> d_val)
    {
        T result;
        HIP_CHECK(hipMemcpy(&result, d_val.get(), sizeof(T), hipMemcpyHostToDevice));
        return result;
    }

    CommandArgumentValue getResultValue(std::shared_ptr<void> d_val, VariableType type)
    {
        if(type == DataType::Int64)
        {
            return getResultValue<int64_t>(d_val);
        }

        throw std::runtime_error(concatenate("Not implemented for ", type));
    }

    void compareResultValues(CommandArgumentValue reference,
                             CommandArgumentValue result,
                             CommandArgumentValue lhs,
                             CommandArgumentValue rhs)
    {
        ASSERT_EQ(reference.index(), result.index());

        std::visit(
            [&](auto refVal) {
                using RefType  = std::decay_t<decltype(refVal)>;
                auto resultVal = std::get<RefType>(result);
                EXPECT_EQ(refVal, resultVal)
                    << ", LHS: " << toString(lhs) << ", RHS: " << toString(rhs);
            },
            reference);
    }

    void compareResultValues(CommandArgumentValue reference,
                             CommandArgumentValue result,
                             CommandArgumentValue arg1,
                             CommandArgumentValue arg2,
                             CommandArgumentValue arg3)
    {
        ASSERT_EQ(reference.index(), result.index());

        std::visit(
            [&](auto refVal) {
                using RefType  = std::decay_t<decltype(refVal)>;
                auto resultVal = std::get<RefType>(result);
                if constexpr(std::is_floating_point_v<RefType>)
                {
                    if(refVal != 0.0)
                    {
                        EXPECT_LT(fabs((resultVal - refVal) / refVal), 5.0e-6)
                            << ", ARG1: " << toString(arg1) << ", ARG2: " << toString(arg2)
                            << ", ARG3: " << toString(arg3);
                    }
                    else
                    {
                        EXPECT_EQ(resultVal, refVal)
                            << ", ARG1: " << toString(arg1) << ", ARG2: " << toString(arg2)
                            << ", ARG3: " << toString(arg3);
                    }
                }
                else
                {
                    EXPECT_EQ(resultVal, refVal)
                        << ", ARG1: " << toString(arg1) << ", ARG2: " << toString(arg2)
                        << ", ARG3: " << toString(arg3);
                }
            },
            reference);
    }

    template <CCommandArgumentValue NewType>
    CommandArgumentValue castTo(CommandArgumentValue val)
    {
        return std::visit(
            [](auto v) -> CommandArgumentValue {
                using CurType = std::decay_t<decltype(v)>;
                if constexpr(std::is_pointer_v<CurType>)
                {
                    Throw<FatalError>("Pointer value present in arithmetic test.");
                }
                else
                {
                    return static_cast<NewType>(v);
                }
            },
            val);
    }

    CommandArgumentValue castToResult(CommandArgumentValue val, VariableType type)
    {
        if(type == DataType::Int64)
        {
            return castTo<int64_t>(val);
        }
        else if(type == DataType::Int32)
        {
            return castTo<int>(val);
        }
        else if(type == DataType::Float)
        {
            return castTo<float>(val);
        }
        else if(type == DataType::Double)
        {
            return castTo<double>(val);
        }

        throw std::runtime_error(concatenate("Not implemented for ", type));
    }

    Generator<Instruction> getValueReg(ContextPtr&          context,
                                       Register::ValuePtr&  reg,
                                       Register::Type       regType,
                                       VariableType         varType,
                                       CommandArgumentValue value,
                                       Register::ValuePtr   inputPtr,
                                       int                  idx)
    {
        if(regType == Register::Type::Literal)
        {
            reg = Register::Value::Literal(value);
        }
        else
        {
            reg = Register::Value::Placeholder(context, Register::Type::Scalar, varType, 1);

            size_t offset = idx * varType.getElementSize();

            co_yield context->mem()->loadScalar(reg, inputPtr, offset, varType.getElementSize());

            if(regType == Register::Type::Vector)
            {
                auto tmp = reg;
                reg      = Register::Value::Placeholder(
                    context, Register::Type::Vector, reg->variableType(), 1);

                co_yield reg->allocate();
                co_yield context->copier()->copy(
                    reg, tmp, concatenate("Move ", tmp->name(), " to VGPR"));
            }
        }
    }

    Generator<Instruction> getArgReg(ContextPtr&         context,
                                     Register::ValuePtr& reg,
                                     std::string const&  name,
                                     Register::Type      type)
    {
        co_yield context->argLoader()->getValue(name, reg);

        if(type != Register::Type::Scalar)
        {
            Register::ValuePtr tmp = reg;
            reg                    = tmp->placeholder(type, {});

            co_yield reg->allocate();

            co_yield context->copier()->copy(reg, tmp, "");
        }
    }

    struct GPU_MixedBinaryArithmeticTest : public CurrentGPUContextFixture,
                                           public ::testing::WithParamInterface<BinaryArgument>
    {

        /**
         * Generate an expression to calculate the reference value.  Can return
         * nullptr for an invalid combination of inputs (e.g. dividing by zero)
         */

        template <Expression::CBinary Operation>
        void testBody()
        {
            auto const& param = GetParam();
            std::string paramStr;
            {
                std::ostringstream msg;
                msg << param;
                paramStr = msg.str();
            }
            auto command = std::make_shared<Command>();

            CommandArgumentPtr lhsArg, rhsArg, destArg;

            auto tagDest = command->allocateTag();
            destArg      = command->allocateArgument(param.resultVarType.getPointer(),
                                                tagDest,
                                                ArgumentType::Value,
                                                DataDirection::WriteOnly,
                                                "dest");

            rocRoller::Operations::OperationTag tagLhs, tagRhs;
            if(param.lhsRegType != Register::Type::Literal)
            {
                tagLhs = command->allocateTag();
                lhsArg = command->allocateArgument(param.lhsVarType.getPointer(),
                                                   tagLhs,
                                                   ArgumentType::Value,
                                                   DataDirection::ReadOnly,
                                                   "lhs");
            }

            if(param.rhsRegType != Register::Type::Literal)
            {
                tagRhs = command->allocateTag();
                rhsArg = command->allocateArgument(param.rhsVarType.getPointer(),
                                                   tagRhs,
                                                   ArgumentType::Value,
                                                   DataDirection::ReadOnly,
                                                   "rhs");
            }

            m_context->kernel()->addCommandArguments(command->getArguments());
            auto one  = std::make_shared<Expression::Expression>(1u);
            auto zero = std::make_shared<Expression::Expression>(0u);

            m_context->kernel()->setKernelDimensions(1);
            m_context->kernel()->setWorkgroupSize({1, 1, 1});
            m_context->kernel()->setWorkitemCount({one, one, one});

            m_context->schedule(m_context->kernel()->preamble());
            m_context->schedule(m_context->kernel()->prolog());

            auto kb = [&]() -> Generator<Instruction> {
                co_yield Instruction::Comment(paramStr);

                Register::ValuePtr lhsPtr, rhsPtr, resultPtr;

                if(lhsArg)
                {
                    co_yield getArgReg(m_context, lhsPtr, "lhs", Register::Type::Scalar);
                }
                if(rhsArg)
                {
                    co_yield getArgReg(m_context, rhsPtr, "rhs", Register::Type::Scalar);
                }
                co_yield getArgReg(m_context, resultPtr, "dest", Register::Type::Vector);

                Register::ValuePtr lhs, rhs;

                for(size_t lhsIdx = 0; lhsIdx < param.lhsValues.size(); lhsIdx++)
                {
                    co_yield Instruction::Comment(concatenate("Loading lhs value ",
                                                              lhsIdx,
                                                              " (",
                                                              toString(param.lhsValues[lhsIdx]),
                                                              ")"));
                    co_yield getValueReg(m_context,
                                         lhs,
                                         param.lhsRegType,
                                         param.lhsVarType,
                                         param.lhsValues[lhsIdx],
                                         lhsPtr,
                                         lhsIdx);

                    for(size_t rhsIdx = 0; rhsIdx < param.rhsValues.size(); rhsIdx++)
                    {
                        co_yield Instruction::Comment(concatenate("Loading rhs value ",
                                                                  rhsIdx,
                                                                  " (",
                                                                  toString(param.rhsValues[rhsIdx]),
                                                                  ")"));
                        co_yield getValueReg(m_context,
                                             rhs,
                                             param.rhsRegType,
                                             param.rhsVarType,
                                             param.rhsValues[rhsIdx],
                                             rhsPtr,
                                             rhsIdx);

                        auto result = Register::Value::Placeholder(
                            m_context, param.resultRegType, param.resultVarType, 1);

                        co_yield generateOp<Operation>(result, lhs, rhs);

                        auto v_result = result;

                        if(result->regType() != Register::Type::Vector)
                        {
                            v_result = Register::Value::Placeholder(
                                m_context, Register::Type::Vector, param.resultVarType, 1);
                            co_yield v_result->allocate();

                            co_yield m_context->copier()->copy(
                                v_result, result, "Move result to VGPR");
                        }

                        co_yield m_context->mem()->storeGlobal(
                            resultPtr, v_result, 0, param.resultVarType.getElementSize());

                        co_yield generateOp<Expression::Add>(
                            resultPtr,
                            Register::Value::Literal(param.resultVarType.getElementSize()),
                            resultPtr);
                    }
                }
            };

            m_context->schedule(kb());
            m_context->schedule(m_context->kernel()->postamble());
            m_context->schedule(m_context->kernel()->amdgpu_metadata());

            CommandKernel commandKernel;
            commandKernel.setContext(m_context);
            commandKernel.generateKernel();

            int numResultValues = param.lhsValues.size() * param.rhsValues.size();
            int numResultBytes  = numResultValues * param.resultVarType.getElementSize();

            auto                     d_result = make_shared_device<uint8_t>(numResultBytes);
            std::shared_ptr<uint8_t> d_lhs, d_rhs;

            CommandArguments commandArgs = command->createArguments();

            commandArgs.setArgument(tagDest, ArgumentType::Value, d_result.get());

            if(lhsArg)
            {
                d_lhs = createDeviceArray(param.lhsValues);
                commandArgs.setArgument(tagLhs, ArgumentType::Value, d_lhs.get());
            }

            if(rhsArg)
            {
                d_rhs = createDeviceArray(param.rhsValues);
                commandArgs.setArgument(tagRhs, ArgumentType::Value, d_rhs.get());
            }

            commandKernel.launchKernel(commandArgs.runtimeArguments());

            auto result = getDeviceValues(d_result, numResultValues, param.resultVarType);

            int idx = 0;
            for(auto lhsVal : param.lhsValues)
            {
                for(auto rhsVal : param.rhsValues)
                {
                    if constexpr(std::is_same_v<
                                     Operation,
                                     Expression::
                                         Divide> || std::is_same_v<Operation, Expression::Modulo>)
                    {
                        if(std::get<int64_t>(Expression::evaluate(rhsVal)) == 0)
                        {
                            idx++;
                            continue;
                        }
                    }

                    auto lhs_rt  = castToResult(lhsVal, param.resultVarType);
                    auto rhs_rt  = castToResult(rhsVal, param.resultVarType);
                    auto lhs_exp = std::make_shared<Expression::Expression>(lhs_rt);
                    auto rhs_exp = std::make_shared<Expression::Expression>(rhs_rt);

                    auto expr
                        = std::make_shared<Expression::Expression>(Operation{lhs_exp, rhs_exp});

                    if(expr)
                    {
                        auto reference = Expression::evaluate(expr);
                        compareResultValues(reference, result[idx], lhsVal, rhsVal);
                    }

                    idx++;
                }
            }
        }
    };

    struct GPU_MixedTernaryArithmeticTest : public CurrentGPUContextFixture,
                                            public ::testing::WithParamInterface<TernaryArgument>
    {

        /**
         * Generate an expression to calculate the reference value.  Can return
         * nullptr for an invalid combination of inputs (e.g. dividing by zero)
         */

        template <typename Operation>
        requires(
            Expression::CTernary<Operation> || Expression::CTernaryMixed<Operation>) void testBody()
        {
            auto const& param = GetParam();
            std::string paramStr;
            {
                std::ostringstream msg;
                msg << param;
                paramStr = msg.str();
            }
            auto command = std::make_shared<Command>();

            CommandArgumentPtr arg1Arg, arg2Arg, arg3Arg, destArg;

            auto tagDest = command->allocateTag();
            destArg      = command->allocateArgument(param.resultVarType.getPointer(),
                                                tagDest,
                                                ArgumentType::Value,
                                                DataDirection::WriteOnly,
                                                "dest");

            rocRoller::Operations::OperationTag tagArg1, tagArg2, tagArg3;

            if(param.arg1RegType != Register::Type::Literal)
            {
                tagArg1 = command->allocateTag();
                arg1Arg = command->allocateArgument(param.arg1VarType.getPointer(),
                                                    tagArg1,
                                                    ArgumentType::Value,
                                                    DataDirection::ReadOnly,
                                                    "arg1");
            }

            if(param.arg2RegType != Register::Type::Literal)
            {
                tagArg2 = command->allocateTag();
                arg2Arg = command->allocateArgument(param.arg2VarType.getPointer(),
                                                    tagArg2,
                                                    ArgumentType::Value,
                                                    DataDirection::ReadOnly,
                                                    "arg2");
            }

            if(param.arg3RegType != Register::Type::Literal)
            {
                tagArg3 = command->allocateTag();
                arg3Arg = command->allocateArgument(param.arg3VarType.getPointer(),
                                                    tagArg3,
                                                    ArgumentType::Value,
                                                    DataDirection::ReadOnly,
                                                    "arg3");
            }

            m_context->kernel()->addCommandArguments(command->getArguments());
            auto one  = std::make_shared<Expression::Expression>(1u);
            auto zero = std::make_shared<Expression::Expression>(0u);

            m_context->kernel()->setKernelDimensions(1);
            m_context->kernel()->setWorkgroupSize({1, 1, 1});
            m_context->kernel()->setWorkitemCount({one, one, one});

            m_context->schedule(m_context->kernel()->preamble());
            m_context->schedule(m_context->kernel()->prolog());

            auto kb = [&]() -> Generator<Instruction> {
                co_yield Instruction::Comment(paramStr);

                Register::ValuePtr arg1Ptr, arg2Ptr, arg3Ptr, resultPtr;

                if(arg1Arg)
                {
                    co_yield getArgReg(m_context, arg1Ptr, "arg1", Register::Type::Scalar);
                }
                if(arg2Arg)
                {
                    co_yield getArgReg(m_context, arg2Ptr, "arg2", Register::Type::Scalar);
                }
                if(arg3Arg)
                {
                    co_yield getArgReg(m_context, arg3Ptr, "arg3", Register::Type::Scalar);
                }
                co_yield getArgReg(m_context, resultPtr, "dest", Register::Type::Vector);

                Register::ValuePtr arg1, arg2, arg3;

                for(size_t arg1Idx = 0; arg1Idx < param.arg1Values.size(); arg1Idx++)
                {
                    co_yield Instruction::Comment(concatenate("Loading arg1 value ",
                                                              arg1Idx,
                                                              " (",
                                                              toString(param.arg1Values[arg1Idx]),
                                                              ")"));
                    co_yield getValueReg(m_context,
                                         arg1,
                                         param.arg1RegType,
                                         param.arg1VarType,
                                         param.arg1Values[arg1Idx],
                                         arg1Ptr,
                                         arg1Idx);

                    for(size_t arg2Idx = 0; arg2Idx < param.arg2Values.size(); arg2Idx++)
                    {
                        co_yield Instruction::Comment(
                            concatenate("Loading arg2 value ",
                                        arg2Idx,
                                        " (",
                                        toString(param.arg2Values[arg2Idx]),
                                        ")"));
                        co_yield getValueReg(m_context,
                                             arg2,
                                             param.arg2RegType,
                                             param.arg2VarType,
                                             param.arg2Values[arg2Idx],
                                             arg2Ptr,
                                             arg2Idx);

                        for(size_t arg3Idx = 0; arg3Idx < param.arg3Values.size(); arg3Idx++)
                        {
                            co_yield Instruction::Comment(
                                concatenate("Loading arg3 value ",
                                            arg3Idx,
                                            " (",
                                            toString(param.arg3Values[arg3Idx]),
                                            ")"));
                            co_yield getValueReg(m_context,
                                                 arg3,
                                                 param.arg3RegType,
                                                 param.arg3VarType,
                                                 param.arg3Values[arg3Idx],
                                                 arg3Ptr,
                                                 arg3Idx);

                            auto result = Register::Value::Placeholder(
                                m_context, param.resultRegType, param.resultVarType, 1);

                            co_yield generateOp<Operation>(result, arg1, arg2, arg3);

                            auto v_result = result;

                            if(result->regType() != Register::Type::Vector)
                            {
                                v_result = Register::Value::Placeholder(
                                    m_context, Register::Type::Vector, param.resultVarType, 1);
                                co_yield v_result->allocate();

                                co_yield m_context->copier()->copy(
                                    v_result, result, "Move result to VGPR");
                            }

                            co_yield m_context->mem()->storeGlobal(
                                resultPtr, v_result, 0, param.resultVarType.getElementSize());

                            co_yield generateOp<Expression::Add>(
                                resultPtr,
                                Register::Value::Literal(param.resultVarType.getElementSize()),
                                resultPtr);
                        }
                    }
                }
            };

            m_context->schedule(kb());
            m_context->schedule(m_context->kernel()->postamble());
            m_context->schedule(m_context->kernel()->amdgpu_metadata());

            CommandKernel commandKernel;
            commandKernel.setContext(m_context);
            commandKernel.generateKernel();

            int numResultValues
                = param.arg1Values.size() * param.arg2Values.size() * param.arg3Values.size();
            int numResultBytes = numResultValues * param.resultVarType.getElementSize();

            auto                     d_result = make_shared_device<uint8_t>(numResultBytes);
            std::shared_ptr<uint8_t> d_arg1, d_arg2, d_arg3;

            CommandArguments commandArgs = command->createArguments();

            commandArgs.setArgument(tagDest, ArgumentType::Value, d_result.get());

            if(arg1Arg)
            {
                d_arg1 = createDeviceArray(param.arg1Values);
                commandArgs.setArgument(tagArg1, ArgumentType::Value, d_arg1.get());
            }

            if(arg2Arg)
            {
                d_arg2 = createDeviceArray(param.arg2Values);
                commandArgs.setArgument(tagArg2, ArgumentType::Value, d_arg2.get());
            }

            if(arg3Arg)
            {
                d_arg3 = createDeviceArray(param.arg3Values);
                commandArgs.setArgument(tagArg3, ArgumentType::Value, d_arg3.get());
            }

            commandKernel.launchKernel(commandArgs.runtimeArguments());

            auto result = getDeviceValues(d_result, numResultValues, param.resultVarType);

            int idx = 0;
            for(auto arg1Val : param.arg1Values)
            {
                for(auto arg2Val : param.arg2Values)
                {
                    for(auto arg3Val : param.arg3Values)
                    {
                        auto arg1_rt  = castToResult(arg1Val, param.resultVarType);
                        auto arg2_rt  = castToResult(arg2Val, param.resultVarType);
                        auto arg3_rt  = castToResult(arg3Val, param.resultVarType);
                        auto arg1_exp = std::make_shared<Expression::Expression>(arg1_rt);
                        auto arg2_exp = std::make_shared<Expression::Expression>(arg2_rt);
                        auto arg3_exp = std::make_shared<Expression::Expression>(arg3_rt);

                        auto expr = std::make_shared<Expression::Expression>(
                            Operation{arg1_exp, arg2_exp, arg3_exp});

                        if(expr)
                        {
                            auto reference = Expression::evaluate(expr);
                            compareResultValues(reference, result[idx], arg1Val, arg2Val, arg3Val);
                        }

                        idx++;
                    }
                }
            }
        }
    };

    TEST_P(GPU_MixedBinaryArithmeticTest, Add)
    {

        testBody<Expression::Add>();
    }

    TEST_P(GPU_MixedBinaryArithmeticTest, Multiply)
    {
        testBody<Expression::Multiply>();
    }

    TEST_P(GPU_MixedBinaryArithmeticTest, Subtract)
    {
        auto const& param = GetParam();

        if(param.resultRegType == Register::Type::Vector
           && (param.lhsRegType != Register::Type::Vector
               || param.rhsRegType != Register::Type::Vector))
        {
            GTEST_SKIP() << "Subtract not yet supported for mixed register types.";
        }

        testBody<Expression::Subtract>();
    }

    TEST_P(GPU_MixedBinaryArithmeticTest, Divide)
    {
        auto const& param = GetParam();

        if(param.resultRegType == Register::Type::Vector
           && (param.lhsRegType != Register::Type::Vector
               || param.rhsRegType != Register::Type::Vector))
        {
            GTEST_SKIP() << "Divide not yet supported for mixed register types.";
        }

        if(param.lhsVarType != DataType::Int64 || param.rhsVarType != DataType::Int64)
        {
            GTEST_SKIP() << "64-bit divide not yet supported for 32-bit inputs.";
        }

        if(param.lhsRegType == Register::Type::Literal
           || param.rhsRegType == Register::Type::Literal)
        {
            GTEST_SKIP() << "64-bit divide not yet supported for literal inputs.";
        }

        testBody<Expression::Divide>();
    }

    TEST_P(GPU_MixedBinaryArithmeticTest, Modulus)
    {
        auto const& param = GetParam();

        if(param.resultRegType == Register::Type::Vector
           && (param.lhsRegType != Register::Type::Vector
               || param.rhsRegType != Register::Type::Vector))
        {
            GTEST_SKIP() << "Modulus not yet supported for mixed register types.";
        }

        if(param.lhsVarType != DataType::Int64 || param.rhsVarType != DataType::Int64)
        {
            GTEST_SKIP() << "64-bit modulus not yet supported for 32-bit inputs.";
        }

        if(param.lhsRegType == Register::Type::Literal
           || param.rhsRegType == Register::Type::Literal)
        {
            GTEST_SKIP() << "64-bit modulus not yet supported for literal inputs.";
        }

        testBody<Expression::Modulo>();
    }

    TEST_P(GPU_MixedTernaryArithmeticTest, MultiplyAdd)
    {
        testBody<Expression::MultiplyAdd>();
    }

    template <typename T>
    std::vector<CommandArgumentValue> makeVectorOfValues(std::vector<T> const& input)
    {
        std::vector<CommandArgumentValue> rv(input.size());

        for(size_t i = 0; i < input.size(); i++)
        {
            rv[i] = input[i];
        }

        return rv;
    }

    std::vector<CommandArgumentValue> inputs(VariableType vtype)
    {
        if(vtype == DataType::UInt32)
        {
            return makeVectorOfValues(TestValues::uint32Values);
        }
        else if(vtype == DataType::Int32)
        {
            return makeVectorOfValues(TestValues::int32Values);
        }
        else if(vtype == DataType::Int64)
        {
            return makeVectorOfValues(TestValues::int64Values);
        }
        else if(vtype == DataType::Float)
        {
            return makeVectorOfValues(TestValues::floatValues);
        }
        else if(vtype == DataType::Double)
        {
            return makeVectorOfValues(TestValues::doubleValues);
        }

        return {};
    }

    std::vector<BinaryArgument> binaryArguments()
    {
        std::vector<Register::Type> regTypes{
            Register::Type::Literal, Register::Type::Scalar, Register::Type::Vector};

        std::vector<VariableType> varTypes{DataType::UInt32, DataType::Int32, DataType::Int64};

        std::vector<Register::Type> resultRegTypes{Register::Type::Scalar, Register::Type::Vector};

        std::vector<BinaryArgument> rv;

        for(auto lhsRegType : regTypes)
        {
            for(auto lhsVarType : varTypes)
            {
                for(auto rhsRegType : regTypes)
                {
                    for(auto rhsVarType : varTypes)
                    {
                        for(auto resultRegType : resultRegTypes)

                        {
                            if(resultRegType == Register::Type::Scalar
                               && (lhsRegType == Register::Type::Vector
                                   || rhsRegType == Register::Type::Vector))
                            {
                                continue;
                            }

                            if(lhsRegType == Register::Type::Literal
                               && rhsRegType == Register::Type::Literal)
                            {
                                continue;
                            }

                            rv.push_back(BinaryArgument{lhsRegType,
                                                        lhsVarType,
                                                        inputs(lhsVarType),

                                                        rhsRegType,
                                                        rhsVarType,
                                                        inputs(rhsVarType),

                                                        resultRegType,
                                                        DataType::Int64});
                        }
                    }
                }
            }
        }

        return rv;
    }

    /**
     * Generate inputs for ternary tests.
     *
     * Currently targets MultiplyAdd.  The code-gen for MultiplyAdd
     * emits FMA instructions when appropriate, and decomposes to
     * Add(Multiply(.,.), .) otherwise. In theory this could handle
     * mixed types, like f32*f64 + f64.  However, the code-gen path
     * for f32 * f64 isn't happy.
     */
    std::vector<TernaryArgument> ternaryArguments()
    {
        std::vector<Register::Type> regTypes{Register::Type::Scalar, Register::Type::Vector};
        std::vector<VariableType>   varTypes{DataType::Int32, DataType::Float, DataType::Double};
        std::vector<Register::Type> resultRegTypes{Register::Type::Vector};

        std::vector<TernaryArgument> rv;

        for(auto arg1RegType : regTypes)
        {
            for(auto arg1VarType : varTypes)
            {
                for(auto arg2RegType : regTypes)
                {
                    for(auto arg2VarType : varTypes)
                    {
                        for(auto arg3RegType : regTypes)
                        {
                            for(auto arg3VarType : varTypes)
                            {
                                for(auto resultRegType : resultRegTypes)

                                {
                                    // no literals for now
                                    int literalCount
                                        = int(arg1RegType == Register::Type::Literal)
                                          + int(arg2RegType == Register::Type::Literal)
                                          + int(arg3RegType == Register::Type::Literal);
                                    if(literalCount > 0)
                                        continue;

                                    // no scalars for now
                                    int scalarCount = int(arg1RegType == Register::Type::Scalar)
                                                      + int(arg2RegType == Register::Type::Scalar)
                                                      + int(arg3RegType == Register::Type::Scalar);
                                    if(scalarCount > 0)
                                        continue;

                                    // uniform types for now
                                    auto vtype = arg1VarType;
                                    if(arg1VarType != vtype)
                                        continue;
                                    if(arg2VarType != vtype)
                                        continue;
                                    if(arg3VarType != vtype)
                                        continue;

                                    auto resultVarType = VariableType::Promote(
                                        arg1VarType,
                                        VariableType::Promote(arg2VarType, arg3VarType));

                                    rv.push_back(TernaryArgument{arg1RegType,
                                                                 arg1VarType,
                                                                 inputs(arg1VarType),

                                                                 arg2RegType,
                                                                 arg2VarType,
                                                                 inputs(arg2VarType),

                                                                 arg3RegType,
                                                                 arg3VarType,
                                                                 inputs(arg3VarType),

                                                                 resultRegType,
                                                                 resultVarType});
                                }
                            }
                        }
                    }
                }
            }
        }
        return rv;
    }

    INSTANTIATE_TEST_SUITE_P(GPU_MixedBinaryArithmeticTests,
                             GPU_MixedBinaryArithmeticTest,
                             ::testing::ValuesIn(binaryArguments()));

    INSTANTIATE_TEST_SUITE_P(GPU_MixedTernaryArithmeticTests,
                             GPU_MixedTernaryArithmeticTest,
                             ::testing::ValuesIn(ternaryArguments()));
}
