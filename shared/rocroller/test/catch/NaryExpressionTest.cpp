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
    template <Register::Type OperandARegisterType,
              Register::Type OperandBRegisterType,
              Register::Type ResultRegisterType,
              DataType       OperandDataType,
              DataType       ResultDataType>
    struct ConcatenateExpressionKernel : public AssemblyTestKernel
    {
        explicit ConcatenateExpressionKernel(ContextPtr context)
            : AssemblyTestKernel(context)
        {
            REQUIRE(DataTypeInfo::Get(OperandDataType).registerCount * 2
                    == DataTypeInfo::Get(ResultDataType).registerCount);
        }

        Generator<Instruction> codegen()
        {
            Register::ValuePtr s_result;
            Register::ValuePtr s_operandA;
            Register::ValuePtr s_operandB;

            co_yield m_context->argLoader()->getValue("result", s_result);
            co_yield m_context->argLoader()->getValue("operandA", s_operandA);
            co_yield m_context->argLoader()->getValue("operandB", s_operandB);

            auto result   = Register::Value::Placeholder(m_context,
                                                       rocRoller::Register::Type::Vector,
                                                       {ResultDataType, PointerType::PointerGlobal},
                                                       1);
            auto operandA = s_operandA->placeholder(OperandARegisterType, {});
            auto operandB = s_operandB->placeholder(OperandBRegisterType, {});

            co_yield result->allocate();
            co_yield operandA->allocate();
            co_yield operandB->allocate();
            co_yield m_context->copier()->copy(result, s_result);
            co_yield m_context->copier()->copy(operandA, s_operandA);
            co_yield m_context->copier()->copy(operandB, s_operandB);

            Register::ValuePtr v;
            co_yield Expression::generate(
                v,
                std::make_shared<Expression::Expression>(Expression::Concatenate{
                    {{operandA->expression(), operandB->expression()}}, ResultDataType}),
                m_context);

            REQUIRE(v->regType() == ResultRegisterType);

            auto vv = v->placeholder(Register::Type::Vector, {});
            co_yield m_context->copier()->copy(vv, v);
            co_yield m_context->mem()->storeGlobal(
                result, vv, 0, DataTypeInfo::Get(DataType::UInt64).elementBytes);
        }

        void generate() override
        {
            auto k = m_context->kernel();

            k->addArgument({.name          = "result",
                            .variableType  = {DataType::UInt64, PointerType::PointerGlobal},
                            .dataDirection = DataDirection::WriteOnly});
            k->addArgument({.name          = "operandA",
                            .variableType  = DataType::UInt32,
                            .dataDirection = DataDirection::ReadOnly});
            k->addArgument({.name          = "operandB",
                            .variableType  = DataType::UInt32,
                            .dataDirection = DataDirection::ReadOnly});

            m_context->schedule(k->preamble());
            m_context->schedule(k->prolog());
            m_context->schedule(codegen());
            m_context->schedule(k->postamble());
            m_context->schedule(k->amdgpu_metadata());
        }
    };

    TEST_CASE("Run concatenate expression kernel with scalars", "[expression][gpu]")
    {
        auto          context        = TestContext::ForTestDevice();
        std::uint32_t a              = 0xaaaaaaaaul;
        std::uint32_t b              = 0xbbbbbbbbul;
        std::uint64_t expectedResult = 0xbbbbbbbbaaaaaaaaull;
        auto          result         = make_shared_device<uint64_t>();
        ConcatenateExpressionKernel<rocRoller::Register::Type::Scalar,
                                    rocRoller::Register::Type::Scalar,
                                    rocRoller::Register::Type::Scalar,
                                    DataType::UInt32,
                                    DataType::UInt64>
            kernel(context.get());
        kernel({}, result.get(), a, b);
        REQUIRE_THAT(result, HasDeviceScalarEqualTo(expectedResult));
    }

    TEST_CASE("Run concatenate expression kernel with vectors", "[expression][gpu]")
    {
        auto          context        = TestContext::ForTestDevice();
        std::uint32_t a              = 0xaaaaaaaaul;
        std::uint32_t b              = 0xbbbbbbbbul;
        std::uint64_t expectedResult = 0xbbbbbbbbaaaaaaaaull;
        auto          result         = make_shared_device<uint64_t>();
        ConcatenateExpressionKernel<rocRoller::Register::Type::Vector,
                                    rocRoller::Register::Type::Vector,
                                    rocRoller::Register::Type::Vector,
                                    DataType::UInt32,
                                    DataType::UInt64>
            kernel(context.get());
        kernel({}, result.get(), a, b);
        REQUIRE_THAT(result, HasDeviceScalarEqualTo(expectedResult));
    }

    TEST_CASE("Run concatenate expression kernel with mixed scalars and vectors",
              "[expression][gpu]")
    {
        auto          context        = TestContext::ForTestDevice();
        std::uint32_t a              = 0xaaaaaaaaul;
        std::uint32_t b              = 0xbbbbbbbbul;
        std::uint64_t expectedResult = 0xbbbbbbbbaaaaaaaaull;
        auto          result         = make_shared_device<uint64_t>();
        ConcatenateExpressionKernel<rocRoller::Register::Type::Scalar,
                                    rocRoller::Register::Type::Vector,
                                    rocRoller::Register::Type::Vector,
                                    DataType::UInt32,
                                    DataType::UInt64>
            kernel(context.get());
        kernel({}, result.get(), a, b);
        REQUIRE_THAT(result, HasDeviceScalarEqualTo(expectedResult));
    }
}
