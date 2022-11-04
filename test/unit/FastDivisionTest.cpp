
#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/Operations/Command.hpp>

#include "GPUContextFixture.hpp"
#include "GenericContextFixture.hpp"

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
                  "ShiftR(CommandArgument(user_Int32_Value_0), 3j)");

        expr      = a / Expression::literal(8);
        expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        EXPECT_EQ(Expression::toString(expr_fast),
                  "SignedShiftR(Add(CommandArgument(user_Int32_Value_0), "
                  "ShiftR(SignedShiftR(CommandArgument(user_Int32_Value_0), 31i), "
                  "29i)), 3i)");

        expr      = a / Expression::literal(7u);
        expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        EXPECT_EQ(Expression::toString(expr_fast),
                  "ShiftR(Add(ShiftR(Subtract(CommandArgument(user_Int32_Value_0), "
                  "MultiplyHigh(CommandArgument(user_Int32_Value_0), "
                  "613566757m)), 1i), MultiplyHigh(CommandArgument(user_Int32_Value_0), "
                  "613566757m)), 2j)");

        expr      = a / Expression::literal(1);
        expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        EXPECT_EQ(Expression::toString(expr_fast), "CommandArgument(user_Int32_Value_0)");

        expr      = a / Expression::literal(-5);
        expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        EXPECT_EQ(
            Expression::toString(expr_fast),
            "Add(SignedShiftR(MultiplyHigh(CommandArgument(user_Int32_Value_0), 2576980377l), 1j), "
            "ShiftR(MultiplyHigh(CommandArgument(user_Int32_Value_0), 2576980377l), 31i))");

        expr      = a / std::make_shared<Expression::Expression>(8u);
        expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        EXPECT_EQ(Expression::toString(expr_fast),
                  "ShiftR(CommandArgument(user_Int32_Value_0), 3j)");

        expr      = a / std::make_shared<Expression::Expression>(128u);
        expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        EXPECT_EQ(Expression::toString(expr_fast),
                  "ShiftR(CommandArgument(user_Int32_Value_0), 7j)");
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
                  "Subtract(BitwiseXor(SignedShiftR(Add(Add(MultiplyHigh(CommandArgument(user_"
                  "Int32_Value_0), "
                  "magic_num_0), CommandArgument(user_Int32_Value_0)), "
                  "BitwiseAnd(SignedShiftR(Add(MultiplyHigh(CommandArgument(user_Int32_Value_0), "
                  "magic_num_0), "
                  "CommandArgument(user_Int32_Value_0)), 31i), Subtract(ShiftL(1i, "
                  "magic_shifts_0), Equal(magic_num_0, "
                  "0i)))), magic_shifts_0), magic_sign_0), magic_sign_0)");

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
                  "ShiftR(SignedShiftR(CommandArgument(user_Int32_Value_0), 31i), 29i)), -8i))");

        expr      = a % Expression::literal(7u);
        expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        EXPECT_EQ(Expression::toString(expr_fast),
                  "Subtract(CommandArgument(user_Int32_Value_0), "
                  "Multiply(ShiftR(Add(ShiftR(Subtract(CommandArgument(user_Int32_Value_0), "
                  "MultiplyHigh(CommandArgument(user_Int32_Value_0), 613566757m)), 1i), "
                  "MultiplyHigh(CommandArgument(user_Int32_Value_0), "
                  "613566757m)), 2j), 7j))");

        expr      = a % Expression::literal(1);
        expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        EXPECT_EQ(Expression::toString(expr_fast), "0i");

        expr      = a % Expression::literal(-5);
        expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        EXPECT_EQ(Expression::toString(expr_fast),
                  "Subtract(CommandArgument(user_Int32_Value_0), "
                  "Multiply(Add(SignedShiftR(MultiplyHigh(CommandArgument(user_Int32_Value_0), "
                  "2576980377l), 1j), ShiftR(MultiplyHigh(CommandArgument(user_Int32_Value_0), "
                  "2576980377l), 31i)), -5i))");

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
        EXPECT_EQ(Expression::toString(expr_fast),
                  "Subtract(CommandArgument(user_Int32_Value_0), "
                  "Multiply(Subtract(BitwiseXor(SignedShiftR(Add(Add(MultiplyHigh(CommandArgument("
                  "user_Int32_Value_0)"
                  ", magic_num_0), CommandArgument(user_Int32_Value_0)), "
                  "BitwiseAnd(SignedShiftR(Add(MultiplyHigh(CommandArgument(user_Int32_Value_0), "
                  "magic_num_0), "
                  "CommandArgument(user_Int32_Value_0)), 31i), Subtract(ShiftL(1i, "
                  "magic_shifts_0), Equal(magic_num_0, "
                  "0i)))), magic_shifts_0), magic_sign_0), magic_sign_0), "
                  "CommandArgument(user_Int32_Value_1)))");

        expr      = a % b_unsigned;
        expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        EXPECT_EQ(
            Expression::toString(expr_fast),
            "Modulo(CommandArgument(user_Int32_Value_0), CommandArgument(user_UInt32_Value_2))");
    }

    class FastDivisionTestCurrentGPU : public CurrentGPUContextFixture
    {
    };

    void executeFastDivision(bool isModulo, std::shared_ptr<Context> m_context)
    {
        auto command = std::make_shared<Command>();

        auto result_arg = command->allocateArgument({DataType::Int32, PointerType::PointerGlobal});
        auto a_arg      = command->allocateArgument({DataType::Int32, PointerType::Value});
        auto b_arg      = command->allocateArgument({DataType::Int32, PointerType::Value});

        auto result_exp = std::make_shared<Expression::Expression>(result_arg);
        auto a_exp      = std::make_shared<Expression::Expression>(a_arg);
        auto b_exp      = std::make_shared<Expression::Expression>(b_arg);

        auto k = m_context->kernel();

        k->addArgument({result_arg->name(),
                        {DataType::Int32, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        result_exp});
        k->addArgument({a_arg->name(), DataType::Int32, DataDirection::ReadOnly, a_exp});
        k->addArgument({b_arg->name(), DataType::Int32, DataDirection::ReadOnly, b_exp});

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

            auto v_result = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Raw32, 2);

            auto v_c = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int32, 1);

            co_yield v_result->allocate();

            co_yield m_context->copier()->copy(v_result, s_result, "Move pointer");

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

        auto d_result = make_shared_device<int>();

        CommandKernel commandKernel(m_context);

        std::vector<int> values
            = {-50002, -146, -1, 0, 1, 2, 4, 5, 7, 12, 19, 33, 63, 906, 3017, 8000, 12344, 40221};
        //= {5, 7, 12, -1};

        for(int a : values)
        {
            for(int b : values)
            {
                if(b == 0)
                    continue;

                KernelArguments runtimeArgs;
                runtimeArgs.append("result", d_result.get());
                runtimeArgs.append("a", a);
                runtimeArgs.append("b", b);

                commandKernel.launchKernel(runtimeArgs.runtimeArguments());

                int result;
                ASSERT_THAT(hipMemcpy(&result, d_result.get(), sizeof(int), hipMemcpyDefault),
                            HasHipSuccess(0));

                if(isModulo)
                {
                    EXPECT_EQ(result, a % b) << "A:" << a << " b: " << b;
                }
                else
                {
                    EXPECT_EQ(result, a / b) << "A:" << a << " b: " << b;
                }
            }
        }
    }

    TEST_F(FastDivisionTestCurrentGPU, GPU_FastDivision)
    {
        executeFastDivision(false, m_context);
    }

    TEST_F(FastDivisionTestCurrentGPU, GPU_FastModulo)
    {
        executeFastDivision(true, m_context);
    }

    class FastDivisionTestByConstantCurrentGPU : public CurrentGPUContextFixture,
                                                 public ::testing::WithParamInterface<int>
    {
    };

    void executeFastDivisionByConstant(int                      divisor,
                                       bool                     isModulo,
                                       std::shared_ptr<Context> m_context)
    {
        auto command = std::make_shared<Command>();

        auto result_arg = command->allocateArgument({DataType::Int32, PointerType::PointerGlobal});
        auto a_arg      = command->allocateArgument({DataType::Int32, PointerType::Value});

        auto result_exp = std::make_shared<Expression::Expression>(result_arg);
        auto a_exp      = std::make_shared<Expression::Expression>(a_arg);

        auto k = m_context->kernel();

        k->addArgument({"result",
                        {DataType::Int32, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        result_exp});
        k->addArgument({"a", DataType::Int32, DataDirection::ReadOnly, a_exp});

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

            auto v_result = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Raw32, 2);

            auto v_c = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int32, 1);

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

        auto d_result = make_shared_device<int>();

        CommandKernel commandKernel(m_context);

        std::vector<int> values
            = {-50002, -146, -1, 0, 1, 2, 4, 5, 7, 12, 19, 33, 63, 906, 3017, 8000, 12344, 40221};

        for(int a : values)
        {
            KernelArguments runtimeArgs;
            runtimeArgs.append("result", d_result.get());
            runtimeArgs.append("a", a);

            commandKernel.launchKernel(runtimeArgs.runtimeArguments());

            int result;
            ASSERT_THAT(hipMemcpy(&result, d_result.get(), sizeof(int), hipMemcpyDefault),
                        HasHipSuccess(0));

            if(isModulo)
            {
                EXPECT_EQ(result, a % divisor) << "A:" << a << " divisor: " << divisor;
            }
            else
            {
                EXPECT_EQ(result, a / divisor) << "A:" << a << " divisor: " << divisor;
            }
        }
    }

    TEST_P(FastDivisionTestByConstantCurrentGPU, GPU_FastDivisionByConstant)
    {
        executeFastDivisionByConstant(GetParam(), false, m_context);
    }

    TEST_P(FastDivisionTestByConstantCurrentGPU, GPU_FastModuloByConstant)
    {
        executeFastDivisionByConstant(GetParam(), true, m_context);
    }

    INSTANTIATE_TEST_SUITE_P(
        FastDivisionTestByConstantCurrentGPUs,
        FastDivisionTestByConstantCurrentGPU,
        ::testing::Values(
            -12346, -12345, -128, -7, -5, -3, -2, -1, 1, 2, 3, 5, 7, 128, 12345, 12346));
}
