// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>

#include <rocRoller/KernelGraph/CoordinateGraph/Transformer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/RegisterTagManager.hpp>
#include <rocRoller/Operations/Command.hpp>

#include <catch2/catch_test_macros.hpp>

#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

using namespace rocRoller;

namespace ExpressionTest
{
    struct ConvertExpressionKernel : public AssemblyTestKernel
    {
        using ExpressionFunc = std::function<Expression::ExpressionPtr(
            Expression::ExpressionPtr, Expression::ExpressionPtr, Expression::ExpressionPtr)>;
        ConvertExpressionKernel(ContextPtr     context,
                                ExpressionFunc func,
                                DataType       resultType,
                                DataType       aType,
                                DataType       bType,
                                DataType       cType)
            : AssemblyTestKernel(context)
            , m_func(func)
            , m_resultType(resultType)
            , m_aType(aType)
            , m_bType(bType)
            , m_cType(cType)

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

                auto v_a = s_a->placeholder(Register::Type::Vector, {});
                auto v_b = s_b->placeholder(Register::Type::Vector, {});
                auto v_c = s_c->placeholder(Register::Type::Vector, {});

                auto a    = v_a->expression();
                auto b    = v_b->expression();
                auto c    = v_c->expression();
                auto expr = m_func(a, b, c);

                co_yield v_a->allocate();
                co_yield v_b->allocate();
                co_yield v_result->allocate();

                co_yield m_context->copier()->copy(v_result, s_result, "Move pointer");

                co_yield m_context->copier()->copy(v_a, s_a, "Move pointer");
                co_yield m_context->copier()->copy(v_b, s_b, "Move pointer");
                co_yield m_context->copier()->copy(v_c, s_c, "Move pointer");

                Register::ValuePtr v_d;
                co_yield Expression::generate(v_d, expr, m_context);

                co_yield m_context->mem()->storeGlobal(
                    v_result, v_d, 0, DataTypeInfo::Get(m_resultType).elementBytes);
            };

            m_context->schedule(kb());
            m_context->schedule(k->postamble());
            m_context->schedule(k->amdgpu_metadata());
        }

    protected:
        ExpressionFunc m_func;
        DataType       m_resultType, m_aType, m_bType, m_cType;
    };

    TEST_CASE("Convert propagation with Subtraction", "[gpu][convert-propagation]")
    {
        auto context = TestContext::ForTestDevice();

        auto expr
            = [](Expression::ExpressionPtr a,
                 Expression::ExpressionPtr b,
                 Expression::ExpressionPtr c) { return convert(DataType::Int32, (a - b) + c); };

        ConvertExpressionKernel kernel(context.get(),
                                       expr,
                                       DataType::Int32,
                                       DataType::Int64,
                                       DataType::Int64,
                                       DataType::Int64);

        int32_t c        = 0;
        auto    d_result = make_shared_device<int32_t>();

        for(auto const a : TestValues::int64Values)
        {
            for(auto const b : TestValues::int64Values)
            {
                kernel({}, d_result.get(), a, b, c);

                int32_t r = int32_t(int64_t(a - b) + c);
                CHECK_THAT(d_result, HasDeviceScalarEqualTo(r));
            }
        }

        auto const assembly = NormalizedSource(context.output());

        // Should use a 32-bit subtraction instruction and then 32-bit addition
        CHECK_THAT(assembly,
                   Catch::Matchers::ContainsSubstring("v_sub_i32")
                       or Catch::Matchers::ContainsSubstring("v_sub_nc_i32"));
        CHECK_THAT(assembly,
                   Catch::Matchers::ContainsSubstring("v_add_i32")
                       or Catch::Matchers::ContainsSubstring("v_add_nc_i32"));
    }

    TEST_CASE("Convert propagation with ArithmeticShiftR", "[gpu][convert-propagation]")
    {
        auto context = TestContext::ForTestDevice();

        auto expr
            = [](Expression::ExpressionPtr a,
                 Expression::ExpressionPtr b,
                 Expression::ExpressionPtr c) { return convert(DataType::Int32, (a >> b) + c); };

        ConvertExpressionKernel kernel(context.get(),
                                       expr,
                                       DataType::Int32,
                                       DataType::Int64,
                                       DataType::Int64,
                                       DataType::Int64);

        int32_t c        = 0;
        auto    d_result = make_shared_device<int32_t>();

        for(auto const a : TestValues::int64Values)
        {
            for(auto const b : TestValues::int64Values)
            {
                kernel({}, d_result.get(), a, b, c);

                int32_t r = int32_t(int64_t(a >> b) + c);
                CHECK_THAT(d_result, HasDeviceScalarEqualTo(r));
            }
        }

        auto const assembly = NormalizedSource(context.output());

        // Should use a 64-bit arithmetic shift instruction and then 32-bit addition
        CHECK_THAT(assembly, Catch::Matchers::ContainsSubstring("v_ashrrev_i64"));
        CHECK_THAT(assembly,
                   Catch::Matchers::ContainsSubstring("v_add_i32")
                       or Catch::Matchers::ContainsSubstring("v_add_nc_i32"));

        CHECK_THAT(assembly, not Catch::Matchers::ContainsSubstring("v_addc_co_u32"));
    }

    TEST_CASE("Convert propagation with LogicalShiftR", "[gpu][convert-propagation]")
    {
        auto context = TestContext::ForTestDevice();

        auto expr = [](Expression::ExpressionPtr a,
                       Expression::ExpressionPtr b,
                       Expression::ExpressionPtr c) {
            return convert(DataType::UInt32, logicalShiftR(a, b) + c);
        };

        ConvertExpressionKernel kernel(context.get(),
                                       expr,
                                       DataType::UInt32,
                                       DataType::UInt64,
                                       DataType::UInt64,
                                       DataType::UInt64);

        uint32_t c        = 0;
        auto     d_result = make_shared_device<uint32_t>();

        for(auto const a : TestValues::uint64Values)
        {
            for(auto const b : TestValues::uint64Values)
            {
                kernel({}, d_result.get(), a, b, c);

                uint32_t r = uint32_t(uint64_t(a >> b) + c);
                CHECK_THAT(d_result, HasDeviceScalarEqualTo(r));
            }
        }

        // Should use a 64-bit logical shift instruction and then 32-bit addition
        auto const assembly = NormalizedSource(context.output());
        CHECK_THAT(assembly, Catch::Matchers::ContainsSubstring("v_lshrrev_b64"));
        CHECK_THAT(assembly,
                   Catch::Matchers::ContainsSubstring("v_add_u32")
                       or Catch::Matchers::ContainsSubstring("v_add_nc_u32"));

        CHECK_THAT(assembly, not Catch::Matchers::ContainsSubstring("v_addc_co_u32"));
    }

    TEST_CASE("Convert propagation with Division/Modulo", "[gpu][convert-propagation]")
    {
        enum class TestOperation
        {
            Division,
            Modulo,
        };

        auto testOp = GENERATE(as<TestOperation>{}, TestOperation::Division, TestOperation::Modulo);

        SECTION("Test division and modulo")
        {

            auto context = TestContext::ForTestDevice({{.enableFullDivision = true}});

            auto const& arch = context->targetArchitecture().target();
            if(arch.isGFX12GPU())
                SKIP("Instruction not supported on gfx12");

            auto expr = [&](Expression::ExpressionPtr a,
                            Expression::ExpressionPtr b,
                            Expression::ExpressionPtr c) {
                return testOp == TestOperation::Division ? convert(DataType::Int32, (a / b) + c)
                                                         : convert(DataType::Int32, (a % b) + c);
            };

            ConvertExpressionKernel kernel(context.get(),
                                           expr,
                                           DataType::Int32,
                                           DataType::Int64,
                                           DataType::Int64,
                                           DataType::Int64);

            int64_t c        = 0;
            auto    d_result = make_shared_device<int32_t>();

            for(auto a : TestValues::int64Values)
            {
                for(auto b : TestValues::int64Values)
                {
                    if(b == 0) // cannot divide/modulo by 0
                        continue;

                    kernel({}, d_result.get(), a, b, c);

                    int32_t r = testOp == TestOperation::Division ? int32_t(int64_t(a / b) + c)
                                                                  : int32_t(int64_t(a % b) + c);
                    CHECK_THAT(d_result, HasDeviceScalarEqualTo(r));
                }
            }

            auto const assembly = NormalizedSource(context.output());

            // Should not do sign extension and should do 32-bit addition
            CHECK_THAT(assembly, Catch::Matchers::ContainsSubstring("v_ashrrev_i32"));
            CHECK_THAT(assembly, Catch::Matchers::ContainsSubstring("v_add_i32"));
        }
    }

    TEST_CASE("Convert propagation with enableDivideBy", "[gpu][convert-propagation]")
    {
        using namespace rocRoller::Expression;
        using namespace rocRoller::KernelGraph;
        using namespace rocRoller::KernelGraph::CoordinateGraph;

        enum class TestOperation
        {
            Division,
            Modulo,
        };

        auto testOp = GENERATE(as<TestOperation>{}, TestOperation::Division, TestOperation::Modulo);

        SECTION("Test division and modulo")
        {

            // This kernel does either:
            //   result = convert(Int32, A/B + C). A, B and C are 64-bit
            //   result = convert(Int32, A%B + C). A, B and C are 64-bit

            auto        context = TestContext::ForTestDevice({{.enableFullDivision = true}});
            auto const& arch    = context->targetArchitecture().target();
            if(arch.isGFX12GPU())
                SKIP("Instruction not supported on gfx12");

            auto typeResult = DataType::Int32;

            DataType typeA, typeB, typeC;
            typeA = typeB = typeC = DataType::Int64;

            auto command = std::make_shared<rocRoller::Command>();

            auto allocateTagAndArg
                = [&](DataType const d, PointerType const p, ArgumentType const a) {
                      auto tag = command->allocateTag();
                      auto arg = command->allocateArgument({d, p}, tag, a);
                      return std::make_pair(tag, arg);
                  };

            auto [tagResult, argResult]
                = allocateTagAndArg(typeResult, PointerType::PointerGlobal, ArgumentType::Value);
            auto& argResultRef = argResult;
            auto [tagA, argA]  = allocateTagAndArg(typeA, PointerType::Value, ArgumentType::Value);
            auto [tagB, argB]  = allocateTagAndArg(typeB, PointerType::Value, ArgumentType::Value);
            auto [tagC, argC]  = allocateTagAndArg(typeC, PointerType::Value, ArgumentType::Value);

            auto kernel = context->kernel();
            kernel->addArgument({argResult->name(),
                                 {typeResult, PointerType::PointerGlobal},
                                 DataDirection::WriteOnly,
                                 std::make_shared<rocRoller::Expression::Expression>(argResult)});
            auto exprA = kernel->addCommandArgument(argA);
            auto exprB = kernel->addCommandArgument(argB);
            auto exprC = kernel->addCommandArgument(argC);

            auto expr = testOp == TestOperation::Division
                            ? convert(DataType::Int32, exprA / exprB + exprC)
                            : convert(DataType::Int32, exprA % exprB + exprC);

            enableDivideBy(exprB, context.get());

            auto one  = std::make_shared<rocRoller::Expression::Expression>(1u);
            auto zero = std::make_shared<rocRoller::Expression::Expression>(0u);
            kernel->setWorkgroupSize({1, 1, 1});
            kernel->setWorkitemCount({one, one, one});
            kernel->setDynamicSharedMemBytes(zero);

            context->schedule(kernel->preamble());
            context->schedule(kernel->prolog());

            auto kb = [&]() -> Generator<Instruction> {
                Register::ValuePtr s_result;
                co_yield context->argLoader()->getValue(argResultRef->name(), s_result);

                auto v_result
                    = Register::Value::Placeholder(context.get(),
                                                   Register::Type::Vector,
                                                   {typeResult, PointerType::PointerGlobal},
                                                   1);

                co_yield v_result->allocate();

                co_yield context->copier()->copy(v_result, s_result, "Move result pointer");

                Register::ValuePtr s_tmp;
                // Generate the target expression
                co_yield rocRoller::Expression::generate(s_tmp, expr, context.get());

                auto v_tmp = Register::Value::Placeholder(
                    context.get(), Register::Type::Vector, typeResult, 1);
                co_yield context->copier()->copy(
                    v_tmp, s_tmp, "Move result to a temporary VGPR to store.");

                co_yield context->mem()->storeGlobal(
                    v_result, v_tmp, 0, DataTypeInfo::Get(typeResult).elementBytes);
            };

            context->schedule(kb());
            context->schedule(kernel->postamble());
            context->schedule(kernel->amdgpu_metadata());

            CommandKernel commandKernel;
            commandKernel.setContext(context.get());
            commandKernel.generateKernel();

            int64_t c        = 0;
            auto    d_result = make_shared_device<int32_t>();

            for(auto const a : TestValues::int64Values)
            {
                for(auto const b : TestValues::int64Values)
                {
                    if(b == 0)
                        continue;

                    int32_t r = testOp == TestOperation::Division ? int32_t(int64_t(a / b) + c)
                                                                  : int32_t(int64_t(a % b) + c);

                    auto commandArgs = command->createArguments();

                    commandArgs.setArgument(tagResult, ArgumentType::Value, d_result.get());
                    commandArgs.setArgument(tagA, ArgumentType::Value, a);
                    commandArgs.setArgument(tagB, ArgumentType::Value, b);
                    commandArgs.setArgument(tagC, ArgumentType::Value, c);

                    commandKernel.launchKernel(commandArgs.runtimeArguments());

                    CHECK_THAT(d_result, HasDeviceScalarEqualTo(r));
                }
            }
        }
    }
}
