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

#include "CustomMatchers.hpp"
#include "CustomSections.hpp"
#include "TestContext.hpp"
#include "TestKernels.hpp"

#include <common/SourceMatcher.hpp>
#include <common/TestValues.hpp>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>

#include <catch2/catch_test_macros.hpp>

#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <catch2/catch_test_case_info.hpp>
#include <catch2/interfaces/catch_interfaces_testcase.hpp>

using namespace rocRoller;

namespace ExpressionTest
{
    struct ScalarExpressionKernel : public AssemblyTestKernel
    {
        ScalarExpressionKernel(rocRoller::ContextPtr context)
            : AssemblyTestKernel(context)
        {
        }

        void generate() override
        {
            auto k = m_context->kernel();

            k->addArgument({"result",
                            {DataType::Int32, PointerType::PointerGlobal},
                            DataDirection::WriteOnly});
            k->addArgument({"a", DataType::Int32});
            k->addArgument({"b", DataType::UInt32});

            m_context->schedule(k->preamble());
            m_context->schedule(k->prolog());

            auto kb = [&]() -> Generator<Instruction> {
                Register::ValuePtr s_result, s_a, s_b, s_c, temp;
                co_yield m_context->argLoader()->getValue("result", s_result);
                co_yield m_context->argLoader()->getValue("a", s_a);
                co_yield m_context->argLoader()->getValue("b", s_b);

                auto a = s_a->expression();
                auto b = s_b->expression();

                auto v_result = s_result->placeholder(Register::Type::Vector, {});

                REQUIRE(v_result != nullptr);

                co_yield m_context->copier()->copy(v_result, s_result, "Move pointer");

                auto expr1 = b > Expression::literal(0);
                co_yield Expression::generate(temp, expr1, m_context);

                auto expr2 = Expression::fuseTernary((a + (a < Expression::literal(5))) << b)
                             + temp->expression();
                co_yield Expression::generate(s_c, expr2, m_context);

                auto v_c = s_c->placeholder(Register::Type::Vector, {});
                co_yield m_context->copier()->copy(v_c, s_c, "Copy result");

                co_yield m_context->mem()->storeGlobal(v_result, v_c, 0, 4);
            };

            m_context->schedule(kb());
            m_context->schedule(k->postamble());
            m_context->schedule(k->amdgpu_metadata());
        }
    };

    TEST_CASE("Run scalar expression kernel", "[expression][scalar-arithmetic][gpu]")
    {
        auto context = TestContext::ForTestDevice();

        ScalarExpressionKernel kernel(context.get());

        auto d_result = make_shared_device<int>();

        for(int a = -10; a < 10; a++)
        {
            for(unsigned int b = 0; b < 5; b++)
            {
                CAPTURE(a, b);

                kernel({}, d_result.get(), a, b);

                auto expectedResult = ((a + (a < 5)) << b) + (b > 0);

                CHECK_THAT(d_result, HasDeviceScalarEqualTo(expectedResult));
            }
        }
    }

    TEST_CASE("Assemble scalar expression kernel", "[expression][scalar-arithmetic][codegen]")
    {
        SUPPORTED_ARCH_SECTION(arch)
        {
            auto context = TestContext::ForTarget(arch);

            ScalarExpressionKernel kernel(context.get());

            CHECK(kernel.getAssembledKernel().size() > 0);
        }
    }
}
