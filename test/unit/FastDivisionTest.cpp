
#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/Operations/Command.hpp>

#include "GPUContextFixture.hpp"
#include "GenericContextFixture.hpp"
#include "TestValues.hpp"
#include "Utilities.hpp"

#include <limits>
#include <typeinfo>

#include <libdivide.h>

using namespace rocRoller;

namespace FastDivisionTest
{
    class FastDivisionTest : public GenericContextFixture
    {
    };

    TEST_F(FastDivisionTest, DivisionByConstantExpressions)
    {
        auto command = std::make_shared<Command>();

        auto a = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Int32, PointerType::Value}));

        auto expr      = a / Expression::literal(8u);
        auto expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        EXPECT_EQ(Expression::toString(expr_fast),
                  "ArithmeticShiftR(CommandArgument(user_Int32_Value_0), 3j)");

        expr      = a / Expression::literal(8);
        expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        EXPECT_EQ(Expression::toString(expr_fast),
                  "ArithmeticShiftR(Add(CommandArgument(user_Int32_Value_0), "
                  "LogicalShiftR(ArithmeticShiftR(CommandArgument(user_Int32_Value_0), 31j), "
                  "29j)), 3i)");

        expr      = a / Expression::literal(7u);
        expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        EXPECT_EQ(Expression::toString(expr_fast),
                  "LogicalShiftR(Add(LogicalShiftR(Subtract(CommandArgument(user_Int32_Value_0), "
                  "MultiplyHigh(CommandArgument(user_Int32_Value_0), 613566757j)), 1j), "
                  "MultiplyHigh(CommandArgument(user_Int32_Value_0), 613566757j)), 2j)");

        expr      = a / Expression::literal(1);
        expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        EXPECT_EQ(Expression::toString(expr_fast), "CommandArgument(user_Int32_Value_0)");

        expr      = a / Expression::literal(-5);
        expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        EXPECT_EQ(
            Expression::toString(expr_fast),
            "Add(ArithmeticShiftR(MultiplyHigh(CommandArgument(user_Int32_Value_0), -1717986919i), "
            "1j), LogicalShiftR(MultiplyHigh(CommandArgument(user_Int32_Value_0), -1717986919i), "
            "31i))");

        expr      = a / std::make_shared<Expression::Expression>(8u);
        expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        EXPECT_EQ(Expression::toString(expr_fast),
                  "ArithmeticShiftR(CommandArgument(user_Int32_Value_0), 3j)");

        expr      = a / std::make_shared<Expression::Expression>(128u);
        expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        EXPECT_EQ(Expression::toString(expr_fast),
                  "ArithmeticShiftR(CommandArgument(user_Int32_Value_0), 7j)");
    }

    TEST_F(FastDivisionTest, DivisionByArgumentExpressions)
    {
        auto command = std::make_shared<Command>();

        auto a = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Int32, PointerType::Value}));

        auto b_signed = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Int32, PointerType::Value}));

        auto b_unsigned = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::UInt32, PointerType::Value}));

        auto expr      = a / b_signed;
        auto expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        EXPECT_EQ(Expression::toString(expr_fast),
                  "Subtract(BitwiseXor(ArithmeticShiftR(Add(Add(MultiplyHigh(CommandArgument(user_"
                  "Int32_Value_0), magic_num_0), CommandArgument(user_Int32_Value_0)), "
                  "BitwiseAnd(ArithmeticShiftR(Add(MultiplyHigh(CommandArgument(user_Int32_Value_0)"
                  ", magic_num_0), CommandArgument(user_Int32_Value_0)), 31j), Add(ShiftL(1i, "
                  "magic_shifts_0), Subtract(LogicalShiftR(BitwiseOr(Negate(magic_num_0), "
                  "magic_num_0), 31j), 1i)))), magic_shifts_0), magic_sign_0), magic_sign_0)");

        expr      = a / b_unsigned;
        expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        EXPECT_EQ(
            Expression::toString(expr_fast),
            "Divide(CommandArgument(user_Int32_Value_0), CommandArgument(user_UInt32_Value_2))");
    }

    TEST_F(FastDivisionTest, ModuloByConstantExpressions)
    {
        auto command = std::make_shared<Command>();

        auto a = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Int32, PointerType::Value}));

        auto expr      = a % Expression::literal(8u);
        auto expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        EXPECT_EQ(Expression::toString(expr_fast),
                  "BitwiseAnd(CommandArgument(user_Int32_Value_0), 7j)");

        expr      = a % Expression::literal(8);
        expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        EXPECT_EQ(Expression::toString(expr_fast),
                  "Subtract(CommandArgument(user_Int32_Value_0), "
                  "BitwiseAnd(Add(CommandArgument(user_Int32_Value_0), "
                  "LogicalShiftR(ArithmeticShiftR(CommandArgument(user_Int32_Value_0), 31j), "
                  "29j)), -8i))");

        expr      = a % Expression::literal(7u);
        expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        EXPECT_EQ(Expression::toString(expr_fast),
                  "Subtract(CommandArgument(user_Int32_Value_0), "
                  "Multiply(LogicalShiftR(Add(LogicalShiftR(Subtract(CommandArgument(user_Int32_"
                  "Value_0), MultiplyHigh(CommandArgument(user_Int32_Value_0), 613566757j)), 1j), "
                  "MultiplyHigh(CommandArgument(user_Int32_Value_0), 613566757j)), 2j), 7j))");

        expr      = a % Expression::literal(1);
        expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        EXPECT_EQ(Expression::toString(expr_fast), "0i");

        expr      = a % Expression::literal(-5);
        expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        EXPECT_EQ(
            Expression::toString(expr_fast),
            "Subtract(CommandArgument(user_Int32_Value_0), "
            "Multiply(Add(ArithmeticShiftR(MultiplyHigh(CommandArgument(user_Int32_Value_0), "
            "-1717986919i), 1j), LogicalShiftR(MultiplyHigh(CommandArgument(user_Int32_Value_0), "
            "-1717986919i), 31i)), -5i))");

        expr      = a % std::make_shared<Expression::Expression>(8u);
        expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        EXPECT_EQ(Expression::toString(expr_fast),
                  "BitwiseAnd(CommandArgument(user_Int32_Value_0), 7j)");

        expr      = a % std::make_shared<Expression::Expression>(128u);
        expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        EXPECT_EQ(Expression::toString(expr_fast),
                  "BitwiseAnd(CommandArgument(user_Int32_Value_0), 127j)");
    }

    TEST_F(FastDivisionTest, ModuloByArgumentExpressions)
    {
        auto command = std::make_shared<Command>();

        auto a = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Int32, PointerType::Value}));

        auto b_signed = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Int32, PointerType::Value}));

        auto b_unsigned = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::UInt32, PointerType::Value}));

        auto expr      = a % b_signed;
        auto expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        EXPECT_EQ(
            Expression::toString(expr_fast),
            "Subtract(CommandArgument(user_Int32_Value_0), "
            "Multiply(Subtract(BitwiseXor(ArithmeticShiftR(Add(Add(MultiplyHigh(CommandArgument("
            "user_Int32_Value_0), magic_num_0), CommandArgument(user_Int32_Value_0)), "
            "BitwiseAnd(ArithmeticShiftR(Add(MultiplyHigh(CommandArgument(user_Int32_Value_0), "
            "magic_num_0), CommandArgument(user_Int32_Value_0)), 31j), Add(ShiftL(1i, "
            "magic_shifts_0), Subtract(LogicalShiftR(BitwiseOr(Negate(magic_num_0), magic_num_0), "
            "31j), 1i)))), magic_shifts_0), magic_sign_0), magic_sign_0), "
            "CommandArgument(user_Int32_Value_1)))");

        expr      = a % b_unsigned;
        expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        EXPECT_EQ(
            Expression::toString(expr_fast),
            "Modulo(CommandArgument(user_Int32_Value_0), CommandArgument(user_UInt32_Value_2))");
    }

    class FastDivisionTestCurrentGPU : public CurrentGPUContextFixture
    {
    public:
        template <typename R, typename A, typename B>
        void executeFastDivision(bool           isModulo,
                                 std::vector<A> numerators,
                                 std::vector<B> denominators)
        {
            auto command = std::make_shared<Command>();

            DataType dataTypeA;
            DataType dataTypeB;
            DataType dataTypeResult;

            if(typeid(A) == typeid(int32_t))
            {
                dataTypeA = DataType::Int32;
            }
            else if(typeid(A) == typeid(int64_t))
            {
                dataTypeA = DataType::Int64;
            }
            else if(typeid(A) == typeid(uint32_t))
            {
                dataTypeA = DataType::UInt32;
            }
            else
            {
                FAIL() << "Testing for unknown data type " << typeid(A).name();
            }

            if(typeid(B) == typeid(int32_t))
            {
                dataTypeB = DataType::Int32;
            }
            else if(typeid(B) == typeid(int64_t))
            {
                dataTypeB = DataType::Int64;
            }
            else if(typeid(B) == typeid(uint32_t))
            {
                dataTypeB = DataType::UInt32;
            }
            else
            {
                FAIL() << "Testing for unknown data type " << typeid(B).name();
            }

            if(typeid(R) == typeid(int32_t))
            {
                dataTypeResult = DataType::Int32;
            }
            else if(typeid(R) == typeid(int64_t))
            {
                dataTypeResult = DataType::Int64;
            }
            else if(typeid(R) == typeid(uint32_t))
            {
                dataTypeResult = DataType::UInt32;
            }
            else
            {
                FAIL() << "Testing for unknown data type " << typeid(B).name();
            }

            auto infoResult = DataTypeInfo::Get(dataTypeResult);

            auto result_arg
                = command->allocateArgument({dataTypeResult, PointerType::PointerGlobal});
            auto a_arg = command->allocateArgument({dataTypeA, PointerType::Value});
            auto b_arg = command->allocateArgument({dataTypeB, PointerType::Value});

            auto result_exp = std::make_shared<Expression::Expression>(result_arg);
            auto a_exp      = std::make_shared<Expression::Expression>(a_arg);
            auto b_exp      = std::make_shared<Expression::Expression>(b_arg);

            auto k = m_context->kernel();

            k->addArgument({result_arg->name(),
                            {dataTypeResult, PointerType::PointerGlobal},
                            DataDirection::WriteOnly,
                            result_exp});
            k->addArgument({a_arg->name(), dataTypeA, DataDirection::ReadOnly, a_exp});
            k->addArgument({b_arg->name(), dataTypeB, DataDirection::ReadOnly, b_exp});

            auto one  = std::make_shared<Expression::Expression>(1u);
            auto zero = std::make_shared<Expression::Expression>(0u);

            k->setWorkgroupSize({1, 1, 1});
            k->setWorkitemCount({one, one, one});
            k->setDynamicSharedMemBytes(zero);

            std::shared_ptr<Expression::Expression> expr;
            if(isModulo)
                expr = fastDivision(a_exp % b_exp, m_context);
            else
                expr = fastDivision(a_exp / b_exp, m_context);
            expr = KernelGraph::cleanArguments(expr, k);

            m_context->schedule(k->preamble());
            m_context->schedule(k->prolog());

            auto kb = [&]() -> Generator<Instruction> {
                Register::ValuePtr s_result, s_a, s_b;
                co_yield m_context->argLoader()->getValue(result_arg->name(), s_result);
                co_yield m_context->argLoader()->getValue(a_arg->name(), s_a);
                co_yield m_context->argLoader()->getValue(b_arg->name(), s_b);

                auto v_result
                    = Register::Value::Placeholder(m_context,
                                                   Register::Type::Vector,
                                                   {dataTypeResult, PointerType::PointerGlobal},
                                                   1);

                auto v_c = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, dataTypeResult, 1);

                co_yield v_result->allocate();

                co_yield m_context->copier()->copy(v_result, s_result, "Move pointer");

                Register::ValuePtr s_c;
                co_yield Expression::generate(s_c, expr, m_context);

                co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
                co_yield m_context->mem()->storeFlat(v_result, v_c, 0, infoResult.elementSize);
            };

            m_context->schedule(kb());
            m_context->schedule(k->postamble());
            m_context->schedule(k->amdgpu_metadata());

            std::shared_ptr<rocRoller::ExecutableKernel> executableKernel
                = m_context->instructions()->getExecutableKernel();

            auto d_result = make_shared_device<R>();

            CommandKernel commandKernel(m_context);

            for(A a : numerators)
            {
                if(dataTypeA == DataType::UInt32 && a > std::numeric_limits<int32_t>::max())
                {
                    // FIXME: Tests fail on large unsigned integers between max int32 and max uint32
                    continue;
                }

                for(B b : denominators)
                {
                    if(b == 0)
                        continue;

                    KernelArguments runtimeArgs;
                    runtimeArgs.append("result", d_result.get());
                    runtimeArgs.append("a", a);
                    runtimeArgs.append("b", b);

                    commandKernel.launchKernel(runtimeArgs.runtimeArguments());

                    R result;
                    ASSERT_THAT(
                        hipMemcpy(
                            &result, d_result.get(), infoResult.elementSize, hipMemcpyDefault),
                        HasHipSuccess(0));

                    if(isModulo)
                    {
                        EXPECT_EQ(result, a % b) << ShowValue(a) << ShowValue(dataTypeA)
                                                 << ShowValue(b) << ShowValue(dataTypeB);
                    }
                    else
                    {
                        auto bLibDivide = libdivide::libdivide_s64_branchfree_gen(b);
                        EXPECT_EQ(result, libdivide::libdivide_s64_branchfree_do(a, &bLibDivide))
                            << ShowValue(a) << ShowValue(dataTypeA) << ShowValue(b)
                            << ShowValue(dataTypeB);

                        // Sanity check
                        EXPECT_EQ(a / b, libdivide::libdivide_s64_branchfree_do(a, &bLibDivide));
                    }
                }
            }
        }
    };

    TEST_F(FastDivisionTestCurrentGPU, GPU_FastDivision)
    {
        executeFastDivision<int32_t>(false, TestValues::int32Values, TestValues::int32Values);
    }

    TEST_F(FastDivisionTestCurrentGPU, GPU_FastModulo)
    {
        executeFastDivision<int32_t>(true, TestValues::int32Values, TestValues::int32Values);
    }

    TEST_F(FastDivisionTestCurrentGPU, GPU_FastDivisionUnsigned)
    {
        executeFastDivision<uint32_t>(false, TestValues::uint32Values, TestValues::uint32Values);
    }

    TEST_F(FastDivisionTestCurrentGPU, GPU_FastModuloUnsigned)
    {
        executeFastDivision<uint32_t>(true, TestValues::uint32Values, TestValues::uint32Values);
    }

    TEST_F(FastDivisionTestCurrentGPU, GPU_FastDivisionMixed)
    {
        executeFastDivision<int64_t>(false, TestValues::int32Values, TestValues::int64Values);
    }

    TEST_F(FastDivisionTestCurrentGPU, GPU_FastModuloMixed)
    {
        executeFastDivision<int64_t>(true, TestValues::int32Values, TestValues::int64Values);
    }

    TEST_F(FastDivisionTestCurrentGPU, GPU_FastDivisionMixed2)
    {
        executeFastDivision<int64_t>(false, TestValues::int64Values, TestValues::int32Values);
    }

    TEST_F(FastDivisionTestCurrentGPU, GPU_FastModuloMixed2)
    {
        executeFastDivision<int64_t>(true, TestValues::int64Values, TestValues::int32Values);
    }

    TEST_F(FastDivisionTestCurrentGPU, GPU_FastDivision64)
    {
        executeFastDivision<int64_t>(true, TestValues::int64Values, TestValues::int64Values);
    }

    TEST_F(FastDivisionTestCurrentGPU, GPU_FastModulo64)
    {
        executeFastDivision<int64_t>(true, TestValues::int64Values, TestValues::int64Values);
    }

    class FastDivisionTestByConstantCurrentGPU : public CurrentGPUContextFixture
    {
    public:
        template <typename A, typename B>
        void executeFastDivisionByConstant(std::vector<A> numerators, B divisor, bool isModulo)
        {
            if(divisor == 0)
            {
                return;
            }

            DataType dataTypeA;

            if(typeid(A) == typeid(int32_t))
            {
                dataTypeA = DataType::Int32;
            }
            else if(typeid(A) == typeid(int64_t))
            {
                dataTypeA = DataType::Int64;
            }
            else if(typeid(A) == typeid(uint32_t))
            {
                dataTypeA = DataType::UInt32;
            }
            else
            {
                FAIL() << "Testing for unknown data type " << typeid(A).name();
            }

            auto command = std::make_shared<Command>();

            auto result_arg = command->allocateArgument({dataTypeA, PointerType::PointerGlobal});
            auto a_arg      = command->allocateArgument({dataTypeA, PointerType::Value});

            auto result_exp = std::make_shared<Expression::Expression>(result_arg);
            auto a_exp      = std::make_shared<Expression::Expression>(a_arg);

            auto k = m_context->kernel();

            k->addArgument({"result",
                            {dataTypeA, PointerType::PointerGlobal},
                            DataDirection::WriteOnly,
                            result_exp});
            k->addArgument({"a", dataTypeA, DataDirection::ReadOnly, a_exp});

            auto one  = std::make_shared<Expression::Expression>(1u);
            auto zero = std::make_shared<Expression::Expression>(0u);

            k->setWorkgroupSize({1, 1, 1});
            k->setWorkitemCount({one, one, one});
            k->setDynamicSharedMemBytes(zero);

            m_context->schedule(k->preamble());
            m_context->schedule(k->prolog());

            auto kb = [&]() -> Generator<Instruction> {
                Register::ValuePtr s_result, s_a;
                co_yield m_context->argLoader()->getValue("result", s_result);
                co_yield m_context->argLoader()->getValue("a", s_a);

                auto v_result
                    = Register::Value::Placeholder(m_context,
                                                   Register::Type::Vector,
                                                   DataType::Raw32,
                                                   2,
                                                   Register::AllocationOptions::FullyContiguous());

                auto v_c
                    = Register::Value::Placeholder(m_context, Register::Type::Vector, dataTypeA, 1);

                co_yield v_result->allocate();

                co_yield m_context->copier()->copy(v_result, s_result, "Move pointer");

                auto a = s_a->expression();

                std::shared_ptr<Expression::Expression> expr;
                if(isModulo)
                    expr = fastDivision(a % Expression::literal(divisor), m_context);
                else
                    expr = fastDivision(a / Expression::literal(divisor), m_context);

                Register::ValuePtr s_c;
                co_yield Expression::generate(s_c, expr, m_context);

                co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
                co_yield m_context->mem()->storeFlat(v_result, v_c, 0, 4);
            };

            m_context->schedule(kb());
            m_context->schedule(k->postamble());
            m_context->schedule(k->amdgpu_metadata());

            std::shared_ptr<rocRoller::ExecutableKernel> executableKernel
                = m_context->instructions()->getExecutableKernel();

            auto d_result = make_shared_device<A>();

            CommandKernel commandKernel(m_context);

            for(A a : numerators)
            {
                KernelArguments runtimeArgs;
                runtimeArgs.append("result", d_result.get());
                runtimeArgs.append("a", a);

                commandKernel.launchKernel(runtimeArgs.runtimeArguments());

                A result;
                ASSERT_THAT(hipMemcpy(&result, d_result.get(), sizeof(A), hipMemcpyDefault),
                            HasHipSuccess(0));

                if(isModulo)
                {
                    EXPECT_EQ(result, a % divisor) << ShowValue(a) << ShowValue(divisor);
                }
                else
                {
                    EXPECT_EQ(result, a / divisor) << ShowValue(a) << ShowValue(divisor);
                }
            }
        }
    };

    class FastDivisionTestByConstantSignedCurrentGPU : public FastDivisionTestByConstantCurrentGPU,
                                                       public ::testing::WithParamInterface<int>
    {
    };

    TEST_P(FastDivisionTestByConstantSignedCurrentGPU, GPU_FastDivisionByConstant32)
    {
        executeFastDivisionByConstant(TestValues::int32Values, GetParam(), false);
    }

    TEST_P(FastDivisionTestByConstantSignedCurrentGPU, GPU_FastModuloByConstant32)
    {
        executeFastDivisionByConstant(TestValues::int32Values, GetParam(), true);
    }

    // FIXME: Types not supported yet
    // TEST_P(FastDivisionTestByConstantSignedCurrentGPU, GPU_FastDivisionByConstantU32)
    // {
    //     executeFastDivisionByConstant(TestValues::uint32Values, GetParam(), false);
    // }

    // FIXME: Types not supported yet
    // TEST_P(FastDivisionTestByConstantSignedCurrentGPU, GPU_FastModuloByConstantU32)
    // {
    //     executeFastDivisionByConstant(TestValues::uint32Values, GetParam(), true);
    // }

    // FIXME: Types not supported yet
    // TEST_P(FastDivisionTestByConstantSignedCurrentGPU, GPU_FastDivisionByConstant64)
    // {
    //     executeFastDivisionByConstant(TestValues::int64Values, GetParam(), false);
    // }

    // FIXME: Types not supported yet
    // TEST_P(FastDivisionTestByConstantSignedCurrentGPU, GPU_FastModuloByConstant64)
    // {
    //     executeFastDivisionByConstant(TestValues::int64Values, GetParam(), true);
    // }

    INSTANTIATE_TEST_SUITE_P(FastDivisionTestByConstantSignedCurrentGPUs,
                             FastDivisionTestByConstantSignedCurrentGPU,
                             ::testing::ValuesIn(TestValues::int32Values));

    class FastDivisionTestByConstantUnsignedCurrentGPU
        : public FastDivisionTestByConstantCurrentGPU,
          public ::testing::WithParamInterface<unsigned int>
    {
    };

    // FIXME: Types not supported yet
    // TEST_P(FastDivisionTestByConstantUnsignedCurrentGPU, GPU_FastDivisionByConstant32)
    // {
    //     executeFastDivisionByConstant(TestValues::int32Values, GetParam(), false);
    // }

    // FIXME: Types not supported yet
    // TEST_P(FastDivisionTestByConstantUnsignedCurrentGPU, GPU_FastModuloByConstant32)
    // {
    //     executeFastDivisionByConstant(TestValues::int32Values, GetParam(), true);
    // }

    TEST_P(FastDivisionTestByConstantUnsignedCurrentGPU, GPU_FastDivisionByConstantU32)
    {
        executeFastDivisionByConstant(TestValues::uint32Values, GetParam(), false);
    }

    TEST_P(FastDivisionTestByConstantUnsignedCurrentGPU, GPU_FastModuloByConstantU32)
    {
        executeFastDivisionByConstant(TestValues::uint32Values, GetParam(), true);
    }

    // FIXME: Types not supported yet
    // TEST_P(FastDivisionTestByConstantUnsignedCurrentGPU, GPU_FastDivisionByConstant64)
    // {
    //     executeFastDivisionByConstant(TestValues::int64Values, GetParam(), false);
    // }

    // FIXME: Types not supported yet
    // TEST_P(FastDivisionTestByConstantUnsignedCurrentGPU, GPU_FastModuloByConstant64)
    // {
    //     executeFastDivisionByConstant(TestValues::int64Values, GetParam(), true);
    // }

    INSTANTIATE_TEST_SUITE_P(FastDivisionTestByConstantUnsignedCurrentGPUs,
                             FastDivisionTestByConstantUnsignedCurrentGPU,
                             ::testing::ValuesIn(TestValues::uint32Values));
}
