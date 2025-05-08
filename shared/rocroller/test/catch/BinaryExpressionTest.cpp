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

using namespace rocRoller;

namespace ExpressionTest
{
    struct BinaryExpressionKernel : public AssemblyTestKernel
    {
        using ExpressionFunc = std::function<Expression::ExpressionPtr(Expression::ExpressionPtr,
                                                                       Expression::ExpressionPtr)>;
        BinaryExpressionKernel(ContextPtr     context,
                               ExpressionFunc func,
                               DataType       resultType,
                               DataType       aType,
                               DataType       bType)
            : AssemblyTestKernel(context)
            , m_func(func)
            , m_resultType(resultType)
            , m_aType(aType)
            , m_bType(bType)

        {
            auto const& arch = m_context->targetArchitecture().target();
            if(!arch.isCDNAGPU())
                SKIP("Test not yet supported on " << arch);
        }

        void generate() override
        {
            auto k = m_context->kernel();

            k->addArgument(
                {"result", {m_resultType, PointerType::PointerGlobal}, DataDirection::WriteOnly});
            k->addArgument({"a", m_aType});
            k->addArgument({"b", m_bType});

            m_context->schedule(k->preamble());
            m_context->schedule(k->prolog());

            auto kb = [&]() -> Generator<Instruction> {
                Register::ValuePtr s_result, s_a, s_b;
                co_yield m_context->argLoader()->getValue("result", s_result);
                co_yield m_context->argLoader()->getValue("a", s_a);
                co_yield m_context->argLoader()->getValue("b", s_b);

                auto v_result
                    = Register::Value::Placeholder(m_context,
                                                   Register::Type::Vector,
                                                   {m_resultType, PointerType::PointerGlobal},
                                                   1);

                auto v_a = s_a->placeholder(Register::Type::Vector, {});
                auto v_b = s_b->placeholder(Register::Type::Vector, {});

                auto a    = v_a->expression();
                auto b    = v_b->expression();
                auto expr = m_func(a, b);

                co_yield v_a->allocate();
                co_yield v_b->allocate();
                co_yield v_result->allocate();

                co_yield m_context->copier()->copy(v_result, s_result, "Move pointer");

                co_yield m_context->copier()->copy(v_a, s_a, "Move pointer");
                co_yield m_context->copier()->copy(v_b, s_b, "Move pointer");

                Register::ValuePtr v_c;
                co_yield Expression::generate(v_c, expr, m_context);

                co_yield m_context->mem()->storeGlobal(
                    v_result, v_c, 0, DataTypeInfo::Get(m_resultType).elementBytes);
            };

            m_context->schedule(kb());
            m_context->schedule(k->postamble());
            m_context->schedule(k->amdgpu_metadata());
        }

    protected:
        ExpressionFunc m_func;
        DataType       m_resultType, m_aType, m_bType;
    };

    void potentiallyMutate(int32_t& x)
    {
        // Just force the compiler to consider that we might have changed the value of x.
    }

    TEST_CASE("Run binary expression kernel 1", "[expression][gpu]")
    {
        auto context = TestContext::ForTestDevice();

        auto expr
            = [](Expression::ExpressionPtr a, Expression::ExpressionPtr b) { return -b * (a + b); };

        BinaryExpressionKernel kernel(
            context.get(), expr, DataType::Double, DataType::Double, DataType::Double);

        auto d_result = make_shared_device<double>();

        for(auto a : TestValues::doubleValues)
        {
            for(auto b : TestValues::doubleValues)
            {
                double r = -b * (a + b);
                CAPTURE(a, b, r);

                kernel({}, d_result.get(), a, b);

                double result;

                CHECK_THAT(d_result, HasDeviceScalar(Catch::Matchers::WithinULP(r, 2)));
                CHECK_THAT(d_result, HasDeviceScalarEqualTo(r));
            }
        }
    }

    TEST_CASE("Assemble binary expression kernel 1", "[expression][codegen]")
    {
        SUPPORTED_ARCH_SECTION(arch)
        {
            auto context = TestContext::ForTarget(arch);

            auto expr = [](Expression::ExpressionPtr a, Expression::ExpressionPtr b) {
                return -b * (a + b);
            };

            BinaryExpressionKernel kernel(
                context.get(), expr, DataType::Double, DataType::Double, DataType::Double);

            CHECK(kernel.getAssembledKernel().size() > 0);
        }
    }

    TEST_CASE("Run binary expression kernel 2", "[expression][gpu]")
    {
        auto context = TestContext::ForTestDevice();

        auto expr = [](Expression::ExpressionPtr a, Expression::ExpressionPtr b) {
            return Expression::fuseTernary((a + b) << b);
        };

        BinaryExpressionKernel kernel(
            context.get(), expr, DataType::Int32, DataType::Int32, DataType::UInt32);

        auto d_result = make_shared_device<int>();

        for(auto a : TestValues::int32Values)
        {
            for(auto b : {5, 2, 3, 1, 12, 4})
            {
                int r = (a + b) << b;
                CAPTURE(a, b, r);

                kernel({}, d_result.get(), a, b);

                CHECK_THAT(d_result, HasDeviceScalarEqualTo(r));
            }
        }
    }

    TEST_CASE("Assemble binary expression kernel 2", "[expression][codegen]")
    {
        SUPPORTED_ARCH_SECTION(arch)
        {
            auto context = TestContext::ForTarget(arch);

            auto expr = [](Expression::ExpressionPtr a, Expression::ExpressionPtr b) {
                return Expression::fuseTernary((a + b) << b);
            };

            BinaryExpressionKernel kernel(
                context.get(), expr, DataType::Int32, DataType::Int32, DataType::UInt32);

            CHECK(kernel.getAssembledKernel().size() > 0);
            using namespace Catch::Matchers;
            CHECK_THAT(context.output(),
                       ContainsSubstring("v_lshlrev_b32", Catch::CaseSensitive::No));
        }
    }

}
