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
    enum class DestRegisterStatus
    {
        NullPointer = 0,
        Placeholder,
        Allocated,
        Count
    };

    std::ostream& operator<<(std::ostream& stream, DestRegisterStatus status)
    {
        switch(status)
        {
        case DestRegisterStatus::NullPointer:
            return stream << "NullPointer";
        case DestRegisterStatus::Placeholder:
            return stream << "Placeholder";
        case DestRegisterStatus::Allocated:
            return stream << "Allocated";
        case DestRegisterStatus::Count:
            return stream << "Count";
        }
        Throw<rocRoller::FatalError>("Bad value!");
    }

    struct TernaryExpressionKernel : public AssemblyTestKernel
    {
        using ExpressionFunc = std::function<Expression::ExpressionPtr(
            Expression::ExpressionPtr, Expression::ExpressionPtr, Expression::ExpressionPtr)>;
        TernaryExpressionKernel(ContextPtr         context,
                                ExpressionFunc     func,
                                DataType           resultType,
                                DataType           aType,
                                DataType           bType,
                                DataType           cType,
                                Register::Type     regType     = Register::Type::Vector,
                                DestRegisterStatus destRegMode = DestRegisterStatus::Placeholder)
            : AssemblyTestKernel(context)
            , m_func(func)
            , m_resultType(resultType)
            , m_aType(aType)
            , m_bType(bType)
            , m_cType(cType)
            , m_regType(regType)
            , m_destRegMode(destRegMode)

        {
        }

        void generate() override
        {
            auto k = m_context->kernel();

            k->addArgument(
                {"result", {m_resultType, PointerType::PointerGlobal}, DataDirection::WriteOnly});
            k->addArgument({"a", m_aType});
            k->addArgument({"b", m_bType});
            k->addArgument({"c", m_cType});

            m_context->schedule(k->preamble());
            m_context->schedule(k->prolog());

            auto kb = [&]() -> Generator<Instruction> {
                Register::ValuePtr s_result, s_a, s_b, s_c;
                co_yield m_context->argLoader()->getValue("result", s_result);
                co_yield m_context->argLoader()->getValue("a", s_a);
                co_yield m_context->argLoader()->getValue("b", s_b);
                co_yield m_context->argLoader()->getValue("c", s_c);

                auto v_result
                    = Register::Value::Placeholder(m_context,
                                                   Register::Type::Vector,
                                                   {m_resultType, PointerType::PointerGlobal},
                                                   1);

                auto v_a = s_a->placeholder(m_regType, {});
                auto v_b = s_b->placeholder(m_regType, {});
                auto v_c = s_c->placeholder(m_regType, {});

                auto a    = v_a->expression();
                auto b    = v_b->expression();
                auto c    = v_c->expression();
                auto expr = m_func(a, b, c);

                co_yield m_context->copier()->copy(v_result, s_result, "Move pointer");

                co_yield m_context->copier()->copy(v_a, s_a, "Move value");
                co_yield m_context->copier()->copy(v_b, s_b, "Move value");
                co_yield m_context->copier()->copy(v_c, s_c, "Move value");

                Register::ValuePtr resultValue, expressionDest;
                if(m_destRegMode == DestRegisterStatus::Placeholder)
                {
                    // resultValue and expressionDest are both pointing to the same object, which is an unallocated register.
                    expressionDest
                        = Register::Value::Placeholder(m_context, m_regType, m_resultType, 1);
                    resultValue = expressionDest;
                }
                else if(m_destRegMode == DestRegisterStatus::Allocated)
                {
                    // resultValue and expressionDest pointing to the different objects, which are allocated and aliased to the same register.
                    expressionDest
                        = Register::Value::Placeholder(m_context, m_regType, m_resultType, 1);
                    expressionDest->allocateNow();
                    resultValue = expressionDest->element({0});
                    REQUIRE(expressionDest != resultValue);
                }

                co_yield Expression::generate(expressionDest, expr, m_context);

                if(m_destRegMode == DestRegisterStatus::NullPointer)
                {
                    // both pointers were null, but expressionDest isn't any more.
                    REQUIRE(expressionDest != nullptr);
                    REQUIRE(resultValue == nullptr);
                    resultValue = expressionDest;
                }
                else if(m_destRegMode == DestRegisterStatus::Placeholder)
                {
                    // the pointers should still equal each other.
                    REQUIRE(expressionDest == resultValue);
                }
                else if(m_destRegMode == DestRegisterStatus::Allocated)
                {
                    // the pointers should still not equal each other.
                    REQUIRE(expressionDest != resultValue);
                }

                Register::ValuePtr v_resultValue;
                if(m_regType == Register::Type::Vector)
                    v_resultValue = resultValue;
                else
                {
                    v_resultValue = resultValue->placeholder(Register::Type::Vector, {});

                    co_yield Expression::generate(
                        v_resultValue, resultValue->expression(), m_context);
                }

                co_yield m_context->mem()->storeGlobal(
                    v_result, v_resultValue, 0, DataTypeInfo::Get(m_resultType).elementBytes);
            };

            m_context->schedule(kb());
            m_context->schedule(k->postamble());
            m_context->schedule(k->amdgpu_metadata());
        }

    protected:
        ExpressionFunc     m_func;
        DataType           m_resultType, m_aType, m_bType, m_cType;
        Register::Type     m_regType;
        DestRegisterStatus m_destRegMode;
    };

    TEST_CASE("Run ternary expression kernel 1", "[expression][ternary][fma][gpu]")
    {
        auto context = TestContext::ForTestDevice();

        auto expr = [](Expression::ExpressionPtr a,
                       Expression::ExpressionPtr b,
                       Expression::ExpressionPtr c) { //
            return a * b + c;
        };

        TernaryExpressionKernel kernel(context.get(),
                                       expr,
                                       DataType::Float,
                                       DataType::Float,
                                       DataType::Float,
                                       DataType::Float);

        auto d_result = make_shared_device<float>();

        for(float a : TestValues::floatValues)
        {
            for(float b : TestValues::floatValues)
            {
                for(float c : TestValues::floatValues)
                {
                    float r = std::fma(a, b, c);
                    CAPTURE(a, b, c, r);

                    kernel({}, d_result.get(), a, b, c);

                    REQUIRE_THAT(d_result, HasDeviceScalar(Catch::Matchers::WithinULP(r, 1)));
                }
            }
        }
    }

    TEST_CASE("Assemble ternary expression kernel 1", "[expression][codegen]")
    {
        SUPPORTED_ARCH_SECTION(arch)
        {
            auto destMode = GENERATE(DestRegisterStatus::NullPointer,
                                     DestRegisterStatus::Placeholder,
                                     DestRegisterStatus::Allocated);
            DYNAMIC_SECTION(destMode)
            {
                auto context = TestContext::ForTarget(arch);

                auto expr = [](Expression::ExpressionPtr a,
                               Expression::ExpressionPtr b,
                               Expression::ExpressionPtr c) { //
                    return a * b + c;
                };

                TernaryExpressionKernel kernel(context.get(),
                                               expr,
                                               DataType::Float,
                                               DataType::Float,
                                               DataType::Float,
                                               DataType::Float);

                CHECK(kernel.getAssembledKernel().size() > 0);
                using namespace Catch::Matchers;
                CHECK_THAT(context.output(), ContainsSubstring("v_fma_f32"));
            }
        }
    }

    TEST_CASE("Run ternary conversion expression kernel 1", "[expression][conversions][gpu]")
    {
        auto regType = GENERATE(Register::Type::Scalar, Register::Type::Vector);
        DYNAMIC_SECTION(regType)
        {
            auto destMode = GENERATE(DestRegisterStatus::NullPointer,
                                     DestRegisterStatus::Placeholder,
                                     DestRegisterStatus::Allocated);
            DYNAMIC_SECTION(destMode)
            {
                auto context = TestContext::ForTestDevice({}, regType, destMode);

                auto expr = [](Expression::ExpressionPtr a,
                               Expression::ExpressionPtr b,
                               Expression::ExpressionPtr c) {
                    namespace Ex = Expression;
                    return convert<DataType::UInt64>(convert<DataType::UInt64>(
                               ((b << Ex::literal(3)) * a) >> Ex::literal(4)))
                           + convert(DataType::Int64, c);
                };

                TernaryExpressionKernel kernel(context.get(),
                                               expr,
                                               DataType::UInt64,
                                               DataType::Int32,
                                               DataType::Int64,
                                               DataType::UInt64,
                                               regType,
                                               destMode);

                auto d_result = make_shared_device<uint64_t>();

                for(auto a : TestValues::int32Values)
                {
                    for(auto b : TestValues::int64Values)
                    {
                        for(auto c : TestValues::uint64Values)
                        {
                            auto r = static_cast<uint64_t>(((b << 3) * a) >> 4)
                                     + static_cast<int64_t>(c);
                            CAPTURE(a, b, c, r);

                            kernel({}, d_result.get(), a, b, c);

                            REQUIRE_THAT(d_result, HasDeviceScalarEqualTo(r));
                        }
                    }
                }
            }
        }
    }

    TEST_CASE("Assemble ternary conversion expression kernel 1",
              "[expression][conversions][codegen]")
    {
        SUPPORTED_ARCH_SECTION(arch)
        {
            auto destMode = GENERATE(DestRegisterStatus::NullPointer,
                                     DestRegisterStatus::Placeholder,
                                     DestRegisterStatus::Allocated);
            DYNAMIC_SECTION(destMode)
            {
                auto context = TestContext::ForTarget(arch);

                auto expr = [](Expression::ExpressionPtr a,
                               Expression::ExpressionPtr b,
                               Expression::ExpressionPtr c) {
                    namespace Ex = Expression;
                    return convert<DataType::UInt64>(((b << Ex::literal(3)) * a) >> Ex::literal(4))
                           + convert(DataType::Int64, c);
                };

                TernaryExpressionKernel kernel(context.get(),
                                               expr,
                                               DataType::UInt64,
                                               DataType::Int32,
                                               DataType::Int64,
                                               DataType::UInt64);

                CHECK(kernel.getAssembledKernel().size() > 0);
                using namespace Catch::Matchers;
                CHECK_THAT(context.output(), !ContainsSubstring("v_fma_f32"));
            }
        }
    }

    TEST_CASE("Run ternary conversion expression kernel 2", "[expression][conversions][gpu]")
    {
        auto context = TestContext::ForTestDevice();

        auto expr = [](Expression::ExpressionPtr a,
                       Expression::ExpressionPtr b,
                       Expression::ExpressionPtr c) {
            namespace Ex = Expression;
            return ((convert<DataType::UInt64>(a) * Ex::literal(23)) + b)
                   - convert(DataType::UInt64, c);
        };

        TernaryExpressionKernel kernel(context.get(),
                                       expr,
                                       DataType::UInt64,
                                       DataType::UInt32,
                                       DataType::UInt32,
                                       DataType::Int64);

        auto d_result = make_shared_device<uint64_t>();

        for(auto a : TestValues::uint32Values)
        {
            for(auto b : TestValues::uint32Values)
            {
                for(auto c : TestValues::int64Values)
                {
                    auto r = ((static_cast<uint64_t>(a) * 23) + b) - static_cast<uint64_t>(c);
                    CAPTURE(a, b, c, r);

                    kernel({}, d_result.get(), a, b, c);

                    REQUIRE_THAT(d_result, HasDeviceScalarEqualTo(r));
                }
            }
        }
    }

    TEST_CASE("Assemble ternary conversion expression kernel 2",
              "[expression][conversions][codegen]")
    {
        SUPPORTED_ARCH_SECTION(arch)
        {
            auto context = TestContext::ForTarget(arch);

            auto expr = [](Expression::ExpressionPtr a,
                           Expression::ExpressionPtr b,
                           Expression::ExpressionPtr c) {
                namespace Ex = Expression;
                return ((convert<DataType::UInt64>(a) * Ex::literal(23)) + b)
                       - convert(DataType::UInt64, c);
            };

            TernaryExpressionKernel kernel(context.get(),
                                           expr,
                                           DataType::UInt64,
                                           DataType::UInt32,
                                           DataType::UInt32,
                                           DataType::Int64);

            CHECK(kernel.getAssembledKernel().size() > 0);
            using namespace Catch::Matchers;
            CHECK_THAT(context.output(), !ContainsSubstring("v_fma_f32"));
        }
    }

    TEST_CASE("Run ternary conversion expression kernel 3", "[expression][conversions][gpu]")
    {
        auto context = TestContext::ForTestDevice();

        auto expr = [](Expression::ExpressionPtr a,
                       Expression::ExpressionPtr b,
                       Expression::ExpressionPtr c) {
            namespace Ex = Expression;
            return ((convert(DataType::Int32, b) + a) >> Ex::literal(4))
                   + convert(DataType::Int32, c);
        };

        TernaryExpressionKernel kernel(context.get(),
                                       expr,
                                       DataType::Int32,
                                       DataType::Int32,
                                       DataType::Int64,
                                       DataType::UInt32);

        auto d_result = make_shared_device<int32_t>();

        for(auto a : TestValues::int32Values)
        {
            for(auto b : TestValues::int64Values)
            {
                for(auto c : TestValues::uint32Values)
                {
                    auto r = ((static_cast<int32_t>(b) + a) >> 4) + static_cast<int32_t>(c);
                    CAPTURE(a, b, c, r);

                    kernel({}, d_result.get(), a, b, c);

                    REQUIRE_THAT(d_result, HasDeviceScalarEqualTo(r));
                }
            }
        }
    }

    TEST_CASE("Assemble ternary conversion expression kernel 3",
              "[expression][conversions][codegen]")
    {
        SUPPORTED_ARCH_SECTION(arch)
        {
            auto context = TestContext::ForTarget(arch);

            auto expr = [](Expression::ExpressionPtr a,
                           Expression::ExpressionPtr b,
                           Expression::ExpressionPtr c) {
                namespace Ex = Expression;
                return ((convert(DataType::Int32, b) + a) >> Ex::literal(4))
                       + convert(DataType::Int32, c);
            };

            TernaryExpressionKernel kernel(context.get(),
                                           expr,
                                           DataType::Int32,
                                           DataType::Int32,
                                           DataType::Int64,
                                           DataType::UInt32);

            CHECK(kernel.getAssembledKernel().size() > 0);
            using namespace Catch::Matchers;
            CHECK_THAT(context.output(), !ContainsSubstring("v_fma_f32"));
        }
    }

    TEST_CASE("Run ternary conversion expression kernel 4", "[expression][conversions][gpu]")
    {
        auto context = TestContext::ForTestDevice();

        auto expr = [](Expression::ExpressionPtr a,
                       Expression::ExpressionPtr b,
                       Expression::ExpressionPtr c) {
            namespace Ex = Expression;
            return logicalShiftR(convert(DataType::UInt32, b) + a, Ex::literal(4))
                   + convert(DataType::UInt32, c);
        };

        TernaryExpressionKernel kernel(context.get(),
                                       expr,
                                       DataType::UInt32,
                                       DataType::UInt32,
                                       DataType::UInt64,
                                       DataType::Int32);

        auto d_result = make_shared_device<uint32_t>();

        for(auto a : TestValues::uint32Values)
        {
            for(auto b : TestValues::uint64Values)
            {
                for(auto c : TestValues::int32Values)
                {
                    auto r = ((static_cast<uint32_t>(b) + a) >> 4) + static_cast<uint32_t>(c);
                    CAPTURE(a, b, c, r);

                    kernel({}, d_result.get(), a, b, c);

                    REQUIRE_THAT(d_result, HasDeviceScalarEqualTo(r));
                }
            }
        }
    }

    TEST_CASE("Assemble ternary conversion expression kernel 4",
              "[expression][conversions][codegen]")
    {
        SUPPORTED_ARCH_SECTION(arch)
        {
            auto context = TestContext::ForTarget(arch);

            auto expr = [](Expression::ExpressionPtr a,
                           Expression::ExpressionPtr b,
                           Expression::ExpressionPtr c) {
                namespace Ex = Expression;
                return logicalShiftR(convert(DataType::UInt32, b) + a, Ex::literal(4))
                       + convert(DataType::UInt32, c);
            };

            TernaryExpressionKernel kernel(context.get(),
                                           expr,
                                           DataType::UInt32,
                                           DataType::UInt32,
                                           DataType::UInt64,
                                           DataType::Int32);

            CHECK(kernel.getAssembledKernel().size() > 0);
            using namespace Catch::Matchers;
            CHECK_THAT(context.output(), !ContainsSubstring("v_fma_f32"));
        }
    }

    TEST_CASE("Run ternary conversion expression kernel 5", "[expression][conversions][gpu]")
    {
        auto regType = GENERATE(Register::Type::Scalar, Register::Type::Vector);
        DYNAMIC_SECTION(regType)
        {
            auto destMode = GENERATE(DestRegisterStatus::NullPointer,
                                     DestRegisterStatus::Placeholder,
                                     DestRegisterStatus::Allocated);
            DYNAMIC_SECTION(destMode)
            {
                auto context = TestContext::ForTestDevice({}, regType, destMode);

                auto expr = [](Expression::ExpressionPtr a,
                               Expression::ExpressionPtr b,
                               Expression::ExpressionPtr c) {
                    namespace Ex = Expression;
                    return convert<DataType::UInt64>(a);
                };

                TernaryExpressionKernel kernel(context.get(),
                                               expr,
                                               DataType::UInt64,
                                               DataType::UInt64,
                                               DataType::UInt64,
                                               DataType::UInt64,
                                               regType,
                                               destMode);

                auto d_result = make_shared_device<uint64_t>();

                for(auto a : TestValues::uint64Values)
                {
                    for(auto b : TestValues::uint64Values)
                    {
                        for(auto c : TestValues::uint64Values)
                        {
                            auto r = a;
                            CAPTURE(a, b, c, r);

                            kernel({}, d_result.get(), a, b, c);

                            REQUIRE_THAT(d_result, HasDeviceScalarEqualTo(r));
                        }
                    }
                }
            }
        }
    }

    TEST_CASE("Expression implicitly converts ShiftL 1", "[expression][conversions][gpu]")
    {
        auto context = TestContext::ForTestDevice();

        auto expr = [](Expression::ExpressionPtr a,
                       Expression::ExpressionPtr b,
                       Expression::ExpressionPtr c) {
            namespace Ex = Expression;
            return (a + b) << c;
        };

        TernaryExpressionKernel kernel(context.get(),
                                       expr,
                                       DataType::UInt64,
                                       DataType::UInt32,
                                       DataType::UInt32,
                                       DataType::Int32);

        auto d_result = make_shared_device<uint64_t>();

        for(auto a : TestValues::uint32Values)
        {
            for(auto b : TestValues::uint32Values)
            {
                for(auto c : TestValues::int32Values)
                {
                    uint64_t r = static_cast<uint64_t>(a + b) << c;
                    CAPTURE(a, b, c, r);

                    kernel({}, d_result.get(), a, b, c);

                    REQUIRE_THAT(d_result, HasDeviceScalarEqualTo(r));
                }
            }
        }
    }

    // This function must be defined in a different compilation unit or the compiler will
    // realize that it does nothing.
    void potentiallyMutate(int32_t& x);

    TEST_CASE("Expression implicitly converts ShiftL 2", "[expression][conversions][gpu]")
    {
        auto context = TestContext::ForTestDevice();

        auto expr = [](Expression::ExpressionPtr a,
                       Expression::ExpressionPtr b,
                       Expression::ExpressionPtr c) {
            namespace Ex = Expression;
            return (a + b) << c;
        };

        TernaryExpressionKernel kernel(context.get(),
                                       expr,
                                       DataType::Int64,
                                       DataType::Int32,
                                       DataType::Int32,
                                       DataType::Int32);

        auto d_result = make_shared_device<int64_t>();

        for(auto a : TestValues::int32Values)
        {
            for(auto b : TestValues::int32Values)
            {
                for(int32_t c : TestValues::int32Values)
                {
                    int32_t tmp = a + b;
                    // If this is not placed here, the compiler will directly calculate a + b
                    // as int64_t, and we will not have the same overflow behaviour that we
                    // have on the GPU.
                    potentiallyMutate(tmp);
                    int64_t r = static_cast<int64_t>(tmp) << c;
                    CAPTURE(a, b, c, r, tmp);

                    kernel({}, d_result.get(), a, b, c);

                    REQUIRE_THAT(d_result, HasDeviceScalarEqualTo(r));
                }
            }
        }
    }

    TEST_CASE("Expression implicitly converts ShiftR 1", "[expression][conversions][gpu]")
    {
        auto context = TestContext::ForTestDevice();

        auto expr = [](Expression::ExpressionPtr a,
                       Expression::ExpressionPtr b,
                       Expression::ExpressionPtr c) {
            namespace Ex = Expression;
            return Ex::logicalShiftR((a + b), c);
        };

        TernaryExpressionKernel kernel(context.get(),
                                       expr,
                                       DataType::UInt64,
                                       DataType::UInt32,
                                       DataType::UInt32,
                                       DataType::Int32);

        auto d_result = make_shared_device<uint64_t>();

        for(auto a : TestValues::uint32Values)
        {
            for(auto b : TestValues::uint32Values)
            {
                for(auto c : TestValues::int32Values)
                {
                    uint64_t r = static_cast<uint64_t>(a + b) >> c;
                    CAPTURE(a, b, c, r);

                    kernel({}, d_result.get(), a, b, c);

                    REQUIRE_THAT(d_result, HasDeviceScalarEqualTo(r));
                }
            }
        }
    }

    TEST_CASE("Run expression implicitly converts ShiftR 2", "[expression][conversions][gpu]")
    {
        auto context = TestContext::ForTestDevice();

        auto expr = [](Expression::ExpressionPtr a,
                       Expression::ExpressionPtr b,
                       Expression::ExpressionPtr c) {
            namespace Ex = Expression;
            return (a + b) >> c;
        };

        TernaryExpressionKernel kernel(context.get(),
                                       expr,
                                       DataType::UInt64,
                                       DataType::UInt32,
                                       DataType::UInt32,
                                       DataType::Int32);

        auto d_result = make_shared_device<uint64_t>();

        for(auto a : TestValues::uint32Values)
        {
            for(auto b : TestValues::uint32Values)
            {
                for(auto c : TestValues::int32Values)
                {
                    uint64_t r = static_cast<uint64_t>(a + b) >> c;
                    CAPTURE(a, b, c, r);

                    kernel({}, d_result.get(), a, b, c);

                    REQUIRE_THAT(d_result, HasDeviceScalarEqualTo(r));
                }
            }
        }
    }

    TEST_CASE("Run expression implicitly converts ShiftR 3", "[expression][conversions][gpu]")
    {
        auto context = TestContext::ForTestDevice();

        auto expr = [](Expression::ExpressionPtr a,
                       Expression::ExpressionPtr b,
                       Expression::ExpressionPtr c) {
            namespace Ex = Expression;
            return (a + b) >> c;
        };

        TernaryExpressionKernel kernel(context.get(),
                                       expr,
                                       DataType::Int64,
                                       DataType::Int32,
                                       DataType::Int32,
                                       DataType::Int32);

        auto d_result = make_shared_device<int64_t>();

        for(int32_t a : TestValues::int32Values)
        {
            for(int32_t b : TestValues::int32Values)
            {
                for(int32_t c : TestValues::int32Values)
                {
                    int32_t tmp1 = a + b;
                    potentiallyMutate(tmp1);
                    int64_t tmp2 = tmp1;
                    int64_t r    = tmp2 >> c;

                    // if(a == 1141374976 && b == 1141374976)
                    // {
                    //     // 1141374976 has a 1 in the second-from-MSB.
                    //     // 1141374976 + 1141374976 will overflow int32.
                    //     // In release mode if tmp1 is never used, the compiler will
                    //     // do 64-bit addition, which won't overflow and the result
                    //     // will be different from the GPU code which doesn't have
                    //     // this optimization.

                    //     // TODO: Find change compiler flags in this file to prevent
                    //     // this optimization?
                    //     CHECK(tmp1 < 0);
                    //     CHECK(tmp2 < 0);
                    //     CHECK(r < 0);
                    // }
                    // int64_t r = static_cast<int64_t>(a + b) >> c;
                    CAPTURE(a, b, c, r);

                    kernel({}, d_result.get(), a, b, c);

                    REQUIRE_THAT(d_result, HasDeviceScalarEqualTo(r));
                }
            }
        }
    }

    struct Less
    {
        static constexpr auto name = "<";

        static constexpr auto compare(auto a, auto b)
        {
            return a < b;
        }
    };

    struct LessEqual
    {
        static constexpr auto name = "<=";

        static constexpr auto compare(auto a, auto b)
        {
            return a <= b;
        }
    };

    struct Equal
    {
        static constexpr auto name = "==";

        static constexpr auto compare(auto a, auto b)
        {
            return a == b;
        }
    };

    struct GreaterEqual
    {
        static constexpr auto name = ">=";

        static constexpr auto compare(auto a, auto b)
        {
            return a >= b;
        }
    };

    struct Greater
    {
        static constexpr auto name = ">";

        static constexpr auto compare(auto a, auto b)
        {
            return a > b;
        }
    };

    TEST_CASE("Run expression Conditional", "[expression][conditional][gpu]")
    {
        auto regType = GENERATE(Register::Type::Scalar, Register::Type::Vector);
        DYNAMIC_SECTION(regType)
        {
            auto testFunc = [&](auto oper, auto dtype) {
                using Operator = std::decay_t<decltype(oper)>;
                using DType    = std::decay_t<decltype(dtype)>;
                auto dataType  = rocRoller::TypeInfo<DType>::Var.dataType;
                DYNAMIC_SECTION(Operator::name << ", " << dataType)
                {

                    auto context
                        = TestContext::ForTestDevice({}, regType, Operator::name, dataType);

                    auto expr = [](Expression::ExpressionPtr a,
                                   Expression::ExpressionPtr b,
                                   Expression::ExpressionPtr c) {
                        namespace Ex = Expression;
                        auto cond    = Operator::compare(a, b);
                        // auto d       = a > b;
                        return Ex::conditional(cond, b, c);
                    };

                    TernaryExpressionKernel kernel(
                        context.get(), expr, dataType, dataType, dataType, dataType, regType);

                    auto d_result = make_shared_device<DType>();

                    for(auto a : TestValues::ByType<DType>::values)
                    {
                        for(auto b : TestValues::ByType<DType>::values)
                        {
                            for(auto c : TestValues::ByType<DType>::values)
                            {
                                auto cond = Operator::compare(a, b);
                                auto r    = cond ? b : c;

                                CAPTURE(a, b, c, r);

                                kernel({}, d_result.get(), a, b, c);

                                REQUIRE_THAT(d_result, HasDeviceScalarEqualTo(r));
                            }
                        }
                    }
                }
            };

            testFunc(Less{}, uint32_t{});
            testFunc(LessEqual{}, uint32_t{});
            testFunc(Equal{}, uint32_t{});
            testFunc(GreaterEqual{}, uint32_t{});
            testFunc(Greater{}, uint32_t{});

            testFunc(Less{}, int32_t{});
            testFunc(LessEqual{}, int32_t{});
            testFunc(Equal{}, int32_t{});
            testFunc(GreaterEqual{}, int32_t{});
            testFunc(Greater{}, int32_t{});

            testFunc(Less{}, uint64_t{});
            testFunc(LessEqual{}, uint64_t{});
            testFunc(Equal{}, uint64_t{});
            testFunc(GreaterEqual{}, uint64_t{});
            testFunc(Greater{}, uint64_t{});

            testFunc(Less{}, int64_t{});
            testFunc(LessEqual{}, int64_t{});
            testFunc(Equal{}, int64_t{});
            testFunc(GreaterEqual{}, int64_t{});
            testFunc(Greater{}, int64_t{});

            if(regType == Register::Type::Vector)
            {
                testFunc(Less{}, float{});
                testFunc(LessEqual{}, float{});
                testFunc(Equal{}, float{});
                testFunc(GreaterEqual{}, float{});
                testFunc(Greater{}, float{});

                testFunc(Less{}, double{});
                testFunc(LessEqual{}, double{});
                testFunc(Equal{}, double{});
                testFunc(GreaterEqual{}, double{});
                testFunc(Greater{}, double{});
            }
        }
    }

}
