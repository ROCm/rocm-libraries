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

#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/InstructionValues/Register.hpp>

#include <common/Utilities.hpp>

#include "CustomMatchers.hpp"
#include "TestContext.hpp"
#include "TestKernels.hpp"

#include <catch2/catch_test_case_info.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/interfaces/catch_interfaces_testcase.hpp>

using namespace rocRoller;

namespace ExpressionTest
{
    struct ConcatenateExpressionKernel : public AssemblyTestKernel
    {
        explicit ConcatenateExpressionKernel(ContextPtr                   context,
                                             std::vector<DataType>&       operandDataTypes,
                                             std::vector<Register::Type>& operandRegisterTypes,
                                             DataType                     resultDataType,
                                             PointerType                  resultPointerType,
                                             Register::Type               resultRegisterType,
                                             int                          operandNumber)

            : AssemblyTestKernel(context)
            , m_operandDataTypes(operandDataTypes)
            , m_operandRegisterTypes(operandRegisterTypes)
            , m_resultDataType(resultDataType)
            , m_resultPointerType(resultPointerType)
            , m_resultRegisterType(resultRegisterType)
            , m_operandNumber(operandNumber)
        {
            REQUIRE(operandDataTypes.size() == m_operandNumber);
            REQUIRE(operandRegisterTypes.size() == m_operandNumber);

            auto operandTotalRegisters = 0;
            for(int i = 0; i < m_operandNumber; ++i)
            {
                operandTotalRegisters += DataTypeInfo::Get(m_operandDataTypes[i]).registerCount;
            }
            REQUIRE(operandTotalRegisters
                    == DataTypeInfo::Get(VariableType{m_resultDataType, m_resultPointerType})
                           .registerCount);
        }

        Generator<Instruction> codegen()
        {
            Register::ValuePtr              s_result;
            std::vector<Register::ValuePtr> s_operands(m_operandNumber);

            co_yield m_context->argLoader()->getValue("result", s_result);
            for(int i = 0; i < m_operandNumber; ++i)
            {
                co_yield m_context->argLoader()->getValue("operand" + std::to_string(i),
                                                          s_operands[i]);
            }

            auto result = s_result->placeholder(Register::Type::Vector, {});
            std::vector<Register::ValuePtr> operands(m_operandNumber);
            for(int i = 0; i < m_operandNumber; ++i)
            {
                operands[i] = s_operands[i]->placeholder(m_operandRegisterTypes[i], {});
            }
            auto v = Register::Value::Placeholder(
                m_context, m_resultRegisterType, {m_resultDataType, m_resultPointerType}, 1);

            co_yield m_context->copier()->copy(result, s_result);
            for(int i = 0; i < m_operandNumber; ++i)
            {
                co_yield m_context->copier()->copy(operands[i], s_operands[i]);
            }

            std::vector<Expression::ExpressionPtr> operands_expr(m_operandNumber);
            for(int i = 0; i < m_operandNumber; ++i)
            {
                operands_expr[i] = operands[i]->expression();
            }

            co_yield Expression::generate(
                v,
                std::make_shared<Expression::Expression>(Expression::Concatenate{
                    {operands_expr}, VariableType{m_resultDataType, m_resultPointerType}}),
                m_context);

            auto vv = v->placeholder(Register::Type::Vector, {});
            co_yield m_context->copier()->copy(vv, v);
            co_yield m_context->mem()->storeGlobal(
                result,
                vv,
                0,
                DataTypeInfo::Get(VariableType{m_resultDataType, m_resultPointerType})
                    .elementBytes);
        }

        void generate() override
        {
            auto k = m_context->kernel();

            k->addArgument({.name          = "result",
                            .variableType  = {m_resultDataType, PointerType::PointerGlobal},
                            .dataDirection = DataDirection::WriteOnly});
            for(int i = 0; i < m_operandNumber; ++i)
            {
                k->addArgument({.name          = "operand" + std::to_string(i),
                                .variableType  = {m_operandDataTypes[i], PointerType::Value},
                                .dataDirection = DataDirection::ReadOnly});
            }

            m_context->schedule(k->preamble());
            m_context->schedule(k->prolog());
            m_context->schedule(codegen());
            m_context->schedule(k->postamble());
            m_context->schedule(k->amdgpu_metadata());
        }

    protected:
        std::vector<DataType>       m_operandDataTypes;
        std::vector<Register::Type> m_operandRegisterTypes;
        DataType                    m_resultDataType;
        PointerType                 m_resultPointerType;
        Register::Type              m_resultRegisterType;
        int                         m_operandNumber;
    };

    TEST_CASE("Run concatenate expression kernel with scalars", "[expression][gpu]")
    {
        auto context = TestContext::ForTestDevice();

        std::uint32_t a = 0xaaaaaaaaul;
        std::uint32_t b = 0xbbbbbbbbul;

        std::uint64_t expectedResult = 0xbbbbbbbbaaaaaaaaull;
        auto          result         = make_shared_device<uint64_t>();

        std::vector<DataType>       operandDataTypes{DataType::UInt32, DataType::UInt32};
        std::vector<Register::Type> operandRegisterTypes{Register::Type::Scalar,
                                                         Register::Type::Scalar};

        DataType       resultDataType    = DataType::UInt64;
        PointerType    resultPointerType = PointerType::Value;
        Register::Type resultRegType     = Register::Type::Scalar;

        ConcatenateExpressionKernel kernel(context.get(),
                                           operandDataTypes,
                                           operandRegisterTypes,
                                           resultDataType,
                                           resultPointerType,
                                           resultRegType,
                                           2);
        kernel({}, result.get(), a, b);

        REQUIRE_THAT(result, HasDeviceScalarEqualTo(expectedResult));
    }

    TEST_CASE("Run concatenate expression kernel with vectors", "[expression][gpu]")
    {
        auto context = TestContext::ForTestDevice();

        std::uint32_t a = 0xaaaaaaaaul;
        std::uint32_t b = 0xbbbbbbbbul;

        std::uint64_t expectedResult = 0xbbbbbbbbaaaaaaaaull;
        auto          result         = make_shared_device<uint64_t>();

        std::vector<DataType>       operandDataTypes{DataType::UInt32, DataType::UInt32};
        std::vector<Register::Type> operandRegisterTypes{Register::Type::Vector,
                                                         Register::Type::Vector};

        DataType       resultDataType    = DataType::UInt64;
        PointerType    resultPointerType = PointerType::Value;
        Register::Type resultRegType     = Register::Type::Vector;

        ConcatenateExpressionKernel kernel(context.get(),
                                           operandDataTypes,
                                           operandRegisterTypes,
                                           resultDataType,
                                           resultPointerType,
                                           resultRegType,
                                           2);
        kernel({}, result.get(), a, b);

        REQUIRE_THAT(result, HasDeviceScalarEqualTo(expectedResult));
    }

    TEST_CASE("Run concatenate expression kernel with mixed scalars and vectors",
              "[expression][gpu]")
    {
        auto context = TestContext::ForTestDevice();

        std::uint32_t a = 0xaaaaaaaaul;
        std::uint32_t b = 0xbbbbbbbbul;

        std::uint64_t expectedResult = 0xbbbbbbbbaaaaaaaaull;
        auto          result         = make_shared_device<uint64_t>();

        std::vector<DataType>       operandDataTypes{DataType::UInt32, DataType::UInt32};
        std::vector<Register::Type> operandRegisterTypes{Register::Type::Scalar,
                                                         Register::Type::Vector};

        DataType       resultDataType    = DataType::UInt64;
        PointerType    resultPointerType = PointerType::Value;
        Register::Type resultRegType     = Register::Type::Vector;

        ConcatenateExpressionKernel kernel(context.get(),
                                           operandDataTypes,
                                           operandRegisterTypes,
                                           resultDataType,
                                           resultPointerType,
                                           resultRegType,
                                           2);
        kernel({}, result.get(), a, b);

        REQUIRE_THAT(result, HasDeviceScalarEqualTo(expectedResult));
    }

    TEST_CASE("Run concatenate expression kernel with buffer output", "[expression][gpu]")
    {
        auto context = TestContext::ForTestDevice();

        std::uint32_t a = 0xaaaaaaaaul;
        std::uint32_t b = 0xbbbbbbbbul;
        std::uint32_t c = 0xccccccccul;
        std::uint32_t d = 0xddddddddul;

        std::vector<std::uint32_t> expectedResult
            = {0xaaaaaaaaul, 0xbbbbbbbbul, 0xccccccccul, 0xddddddddul};
        auto result = make_shared_device<uint32_t>(4);

        std::vector<DataType> operandDataTypes{
            DataType::UInt32, DataType::UInt32, DataType::UInt32, DataType::UInt32};
        std::vector<Register::Type> operandRegisterTypes{Register::Type::Scalar,
                                                         Register::Type::Scalar,
                                                         Register::Type::Scalar,
                                                         Register::Type::Scalar};

        DataType       resultDataType    = DataType::None;
        PointerType    resultPointerType = PointerType::Buffer;
        Register::Type resultRegType     = Register::Type::Scalar;

        ConcatenateExpressionKernel kernel(context.get(),
                                           operandDataTypes,
                                           operandRegisterTypes,
                                           resultDataType,
                                           resultPointerType,
                                           resultRegType,
                                           4);
        kernel({}, result.get(), a, b, c, d);

        REQUIRE_THAT(result, HasDeviceVectorEqualTo(expectedResult));
    }
}
