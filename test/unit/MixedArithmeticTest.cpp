
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/Arithmetic.hpp>
#include <rocRoller/CodeGen/Arithmetic/Double.hpp>
#include <rocRoller/CodeGen/Arithmetic/Float.hpp>
#include <rocRoller/CodeGen/Arithmetic/Int32.hpp>
#include <rocRoller/CodeGen/Arithmetic/Int64.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Utilities/Generator.hpp>
#include <rocRoller/Utilities/HipUtils.hpp>

#include "GPUContextFixture.hpp"
#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"
#include "TestValues.hpp"
#include "Utilities.hpp"

using namespace rocRoller;

namespace MixedArithmeticTest
{
    struct Argument
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

    std::ostream& operator<<(std::ostream& stream, Argument const& arg)
    {
        stream << "LHS: " << arg.lhsRegType << ", " << arg.lhsVarType << ", "
               << "RHS: " << arg.rhsRegType << ", " << arg.rhsVarType << ", "
               << "Result: " << arg.resultRegType << ", " << arg.resultVarType;
        return stream;
    }

    struct GPU_MixedArithmeticTest : public CurrentGPUContextFixture,
                                     public ::testing::WithParamInterface<Argument>
    {
        std::shared_ptr<void> createDeviceArray(std::vector<CommandArgumentValue> const& values)
        {
            KernelArguments args;

            for(int i = 0; i < values.size(); i++)
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

        Generator<Instruction> getValueReg(Register::ValuePtr&  reg,
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
                reg = Register::Value::Placeholder(m_context, Register::Type::Scalar, varType, 1);

                size_t offset = idx * varType.getElementSize();

                co_yield m_context->mem()->loadScalar(
                    reg, inputPtr, Register::Value::Literal(offset), varType.getElementSize());

                if(regType == Register::Type::Vector)
                {
                    auto tmp = reg;
                    reg      = Register::Value::Placeholder(
                        m_context, Register::Type::Vector, reg->variableType(), 1);

                    co_yield reg->allocate();
                    co_yield m_context->copier()->copy(
                        reg, tmp, concatenate("Move ", name, " to VGPR"));
                }
            }
        }

        template <CCommandArgumentValue NewType>
        CommandArgumentValue castTo(CommandArgumentValue val)
        {
            return std::visit(
                [](auto v) {
                    using CurType = std::decay_t<decltype(v)>;
                    if constexpr(std::is_pointer_v<CurType>)
                    {
                        return reinterpret_cast<NewType>(v);
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
            throw std::runtime_error(concatenate("Not implemented for ", type));
        }

        Generator<Instruction>
            getArgReg(Register::ValuePtr& reg, std::string const& name, Register::Type type)
        {
            co_yield m_context->argLoader()->getValue(name, reg);

            if(type != Register::Type::Scalar)
            {
                Register::ValuePtr tmp = reg;
                reg                    = tmp->placeholder(type);

                co_yield reg->allocate();

                co_yield m_context->copier()->copy(reg, tmp, "");
            }
        }

        /**
         * Generate the GPU arithmetic for this specific test case.  Generally
         * should just call the appropriate member of Arithmetic.
         */
        using GenerateArithmetic = std::function<Generator<Instruction>(
            ArithmeticPtr, Register::ValuePtr, Register::ValuePtr, Register::ValuePtr)>;

        /**
         * Generate an expression to calculate the reference value.  Can return
         * nullptr for an invalid combination of inputs (e.g. dividing by zero)
         */
        using CreateExpression = std::function<Expression::ExpressionPtr(
            Expression::ExpressionPtr, Expression::ExpressionPtr)>;

        void testBody(GenerateArithmetic const& genArithmetic,
                      CreateExpression const&   createExpression)
        {
            auto const& param = GetParam();
            std::string paramStr;
            {
                std::ostringstream msg;
                msg << param;
                paramStr = msg.str();
            }
            auto command = std::make_shared<Command>();

            std::shared_ptr<CommandArgument> lhsArg, rhsArg, destArg;

            destArg = command->allocateArgument(
                param.resultVarType.getPointer(), DataDirection::WriteOnly, "dest");

            if(param.lhsRegType != Register::Type::Literal)
            {
                lhsArg = command->allocateArgument(
                    param.lhsVarType.getPointer(), DataDirection::ReadOnly, "lhs");
            }

            if(param.rhsRegType != Register::Type::Literal)
            {
                rhsArg = command->allocateArgument(
                    param.rhsVarType.getPointer(), DataDirection::ReadOnly, "rhs");
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
                    co_yield getArgReg(lhsPtr, "lhs", Register::Type::Scalar);
                }
                if(rhsArg)
                {
                    co_yield getArgReg(rhsPtr, "rhs", Register::Type::Scalar);
                }
                co_yield getArgReg(resultPtr, "dest", Register::Type::Vector);

                auto arith = Component::Get<Arithmetic>(
                    m_context, param.resultRegType, param.resultVarType);

                VariableType int64(DataType::Int64);
                auto         arithVector64
                    = Component::Get<Arithmetic>(m_context, Register::Type::Vector, int64);

                Register::ValuePtr lhs, rhs;

                for(int lhsIdx = 0; lhsIdx < param.lhsValues.size(); lhsIdx++)
                {
                    co_yield Instruction::Comment(concatenate("Loading lhs value ",
                                                              lhsIdx,
                                                              " (",
                                                              toString(param.lhsValues[lhsIdx]),
                                                              ")"));
                    co_yield getValueReg(lhs,
                                         param.lhsRegType,
                                         param.lhsVarType,
                                         param.lhsValues[lhsIdx],
                                         lhsPtr,
                                         lhsIdx);

                    for(int rhsIdx = 0; rhsIdx < param.rhsValues.size(); rhsIdx++)
                    {
                        co_yield Instruction::Comment(concatenate("Loading rhs value ",
                                                                  rhsIdx,
                                                                  " (",
                                                                  toString(param.rhsValues[rhsIdx]),
                                                                  ")"));
                        co_yield getValueReg(rhs,
                                             param.rhsRegType,
                                             param.rhsVarType,
                                             param.rhsValues[rhsIdx],
                                             rhsPtr,
                                             rhsIdx);

                        auto result = Register::Value::Placeholder(
                            m_context, param.resultRegType, param.resultVarType, 1);

                        co_yield genArithmetic(arith, result, lhs, rhs);

                        auto v_result = result;

                        if(result->regType() != Register::Type::Vector)
                        {
                            v_result = Register::Value::Placeholder(
                                m_context, Register::Type::Vector, param.resultVarType, 1);
                            co_yield v_result->allocate();

                            co_yield m_context->copier()->copy(
                                v_result, result, "Move result to VGPR");
                        }

                        co_yield m_context->mem()->storeFlat(
                            resultPtr, v_result, "0", param.resultVarType.getElementSize());

                        co_yield arithVector64->add(
                            resultPtr,
                            Register::Value::Literal(param.resultVarType.getElementSize()),
                            resultPtr);
                    }
                }
            };

            m_context->schedule(kb());
            m_context->schedule(m_context->kernel()->postamble());
            m_context->schedule(m_context->kernel()->amdgpu_metadata());

            CommandKernel commandKernel(m_context);

            int numResultValues = param.lhsValues.size() * param.rhsValues.size();
            int numResultBytes  = numResultValues * param.resultVarType.getElementSize();

            auto                  d_result = make_shared_device<uint8_t>(numResultBytes);
            std::shared_ptr<void> d_lhs;
            std::shared_ptr<void> d_rhs;

            KernelArguments commandArgs;

            commandArgs.append("result", (void*)d_result.get());

            if(lhsArg)
            {
                d_lhs = createDeviceArray(param.lhsValues);
                commandArgs.append("lhs", d_lhs.get());
            }

            if(rhsArg)
            {
                d_rhs = createDeviceArray(param.rhsValues);
                commandArgs.append("rhs", d_rhs.get());
            }

            commandKernel.launchKernel(commandArgs.runtimeArguments());

            auto result = getDeviceValues(d_result, numResultValues, param.resultVarType);

            int idx = 0;
            for(auto lhsVal : param.lhsValues)
            {
                for(auto rhsVal : param.rhsValues)
                {
                    auto lhs_rt  = castToResult(lhsVal, param.resultVarType);
                    auto rhs_rt  = castToResult(rhsVal, param.resultVarType);
                    auto lhs_exp = std::make_shared<Expression::Expression>(lhs_rt);
                    auto rhs_exp = std::make_shared<Expression::Expression>(rhs_rt);

                    auto expr = createExpression(lhs_exp, rhs_exp);

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

    TEST_P(GPU_MixedArithmeticTest, Add)
    {
        GenerateArithmetic arith = [](ArithmeticPtr      arith,
                                      Register::ValuePtr dest,
                                      Register::ValuePtr lhs,
                                      Register::ValuePtr rhs) -> Generator<Instruction> {
            co_yield arith->add(dest, lhs, rhs);
        };

        CreateExpression expr = [](Expression::ExpressionPtr lhs, Expression::ExpressionPtr rhs) {
            return lhs + rhs;
        };

        testBody(arith, expr);
    }

    TEST_P(GPU_MixedArithmeticTest, Multiply)
    {
        GenerateArithmetic arith = [](ArithmeticPtr      arith,
                                      Register::ValuePtr dest,
                                      Register::ValuePtr lhs,
                                      Register::ValuePtr rhs) -> Generator<Instruction> {
            co_yield arith->mul(dest, lhs, rhs);
        };

        CreateExpression expr = [](Expression::ExpressionPtr lhs, Expression::ExpressionPtr rhs) {
            return lhs * rhs;
        };

        testBody(arith, expr);
    }

    TEST_P(GPU_MixedArithmeticTest, Subtract)
    {
        auto const& param = GetParam();

        if(param.resultRegType == Register::Type::Vector
           && (param.lhsRegType != Register::Type::Vector
               || param.rhsRegType != Register::Type::Vector))
        {
            GTEST_SKIP() << "Subtract not yet supported for mixed register types.";
        }

        GenerateArithmetic arith = [](ArithmeticPtr      arith,
                                      Register::ValuePtr dest,
                                      Register::ValuePtr lhs,
                                      Register::ValuePtr rhs) -> Generator<Instruction> {
            co_yield arith->sub(dest, lhs, rhs);
        };

        CreateExpression expr = [](Expression::ExpressionPtr lhs, Expression::ExpressionPtr rhs) {
            return lhs - rhs;
        };

        testBody(arith, expr);
    }

    TEST_P(GPU_MixedArithmeticTest, Divide)
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

        GenerateArithmetic arith = [](ArithmeticPtr      arith,
                                      Register::ValuePtr dest,
                                      Register::ValuePtr lhs,
                                      Register::ValuePtr rhs) -> Generator<Instruction> {
            co_yield arith->div(dest, lhs, rhs);
        };

        CreateExpression expr = [](Expression::ExpressionPtr lhs,
                                   Expression::ExpressionPtr rhs) -> Expression::ExpressionPtr {
            if(std::get<int64_t>(Expression::evaluate(rhs)) != 0)
                return lhs / rhs;

            return nullptr;
        };

        testBody(arith, expr);
    }

    TEST_P(GPU_MixedArithmeticTest, Modulus)
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

        GenerateArithmetic arith = [](ArithmeticPtr      arith,
                                      Register::ValuePtr dest,
                                      Register::ValuePtr lhs,
                                      Register::ValuePtr rhs) -> Generator<Instruction> {
            co_yield arith->mod(dest, lhs, rhs);
        };

        CreateExpression expr = [](Expression::ExpressionPtr lhs,
                                   Expression::ExpressionPtr rhs) -> Expression::ExpressionPtr {
            if(std::get<int64_t>(Expression::evaluate(rhs)) != 0)
            {
                return lhs % rhs;
            }

            return nullptr;
        };

        testBody(arith, expr);
    }

    template <typename T>
    std::vector<CommandArgumentValue> makeVectorOfValues(std::vector<T> const& input)
    {
        std::vector<CommandArgumentValue> rv(input.size());

        for(int i = 0; i < input.size(); i++)
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

        return {};
    }

    std::vector<Argument> int64Arguments()
    {
        std::vector<Register::Type> regTypes{
            Register::Type::Literal, Register::Type::Scalar, Register::Type::Vector};

        std::vector<VariableType> varTypes{DataType::UInt32, DataType::Int32, DataType::Int64};

        std::vector<Register::Type> resultRegTypes{Register::Type::Scalar, Register::Type::Vector};

        std::vector<Argument> rv;

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

                            rv.push_back(Argument{lhsRegType,
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

    INSTANTIATE_TEST_SUITE_P(GPU_MixedArithmeticTests,
                             GPU_MixedArithmeticTest,
                             ::testing::ValuesIn(int64Arguments()));
}
