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
#include "Utilities.hpp"
#include <common/TestValues.hpp>

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

        auto aTag = command->allocateTag();
        auto a    = std::make_shared<Expression::Expression>(command->allocateArgument(
            {DataType::Int32, PointerType::Value}, aTag, ArgumentType::Value));

        auto expr      = a / Expression::literal(8u);
        auto expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        EXPECT_EQ(Expression::toString(expr_fast),
                  "ArithmeticShiftR(CommandArgument(user_Int32_Value_0)I, 3:U32)I");

        expr      = a / Expression::literal(8);
        expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        EXPECT_EQ(Expression::toString(expr_fast),
                  "ArithmeticShiftR(Add(CommandArgument(user_Int32_Value_0)I, "
                  "LogicalShiftR(ArithmeticShiftR(CommandArgument(user_Int32_Value_0)I, 31:U32)I, "
                  "29:U32)I)I, 3:I)I");

        expr      = a / Expression::literal(7u);
        expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        EXPECT_EQ(
            Expression::toString(expr_fast),
            "LogicalShiftR(Add(LogicalShiftR(Subtract(CommandArgument(user_Int32_Value_0)I, "
            "MultiplyHigh(CommandArgument(user_Int32_Value_0)I, 613566757:U32)U32)U32, 1:U32)U32, "
            "MultiplyHigh(CommandArgument(user_Int32_Value_0)I, 613566757:U32)U32)U32, 2:U32)U32");

        expr      = a / Expression::literal(1);
        expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        EXPECT_EQ(Expression::toString(expr_fast), "CommandArgument(user_Int32_Value_0)I");

        expr      = a / Expression::literal(-5);
        expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        EXPECT_EQ(Expression::toString(expr_fast),
                  "Add(ArithmeticShiftR(MultiplyHigh(CommandArgument(user_Int32_Value_0)I, "
                  "-1717986919:I)I, 1:U32)I, "
                  "LogicalShiftR(ArithmeticShiftR(MultiplyHigh(CommandArgument(user_Int32_Value_0)"
                  "I, -1717986919:I)I, 1:U32)I, 31:U32)I)I");

        expr      = a / std::make_shared<Expression::Expression>(8u);
        expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        EXPECT_EQ(Expression::toString(expr_fast),
                  "ArithmeticShiftR(CommandArgument(user_Int32_Value_0)I, 3:U32)I");

        expr      = a / std::make_shared<Expression::Expression>(128u);
        expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        EXPECT_EQ(Expression::toString(expr_fast),
                  "ArithmeticShiftR(CommandArgument(user_Int32_Value_0)I, 7:U32)I");
    }

    TEST_F(FastDivisionTest, DivisionByArgumentExpressions)
    {
        auto command = std::make_shared<Command>();

        auto reg
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Int32, 1);
        auto a = reg->expression();

        auto reg2
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::UInt32, 1);
        auto a_unsigned = reg2->expression();

        auto bSignedTag = command->allocateTag();
        auto b_signed   = std::make_shared<Expression::Expression>(command->allocateArgument(
            {DataType::Int32, PointerType::Value}, bSignedTag, ArgumentType::Value));

        auto bUnsignedTag = command->allocateTag();
        auto b_unsigned   = std::make_shared<Expression::Expression>(command->allocateArgument(
            {DataType::UInt32, PointerType::Value}, bUnsignedTag, ArgumentType::Value));

        auto expr      = a / b_signed;
        auto expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        auto expected  = "Subtract(BitwiseXor(ArithmeticShiftR(Add(Add(MultiplyHigh(UNALLOCATED:I, "
                        "MagicMultiple_0:I)I, UNALLOCATED:I)I, "
                        "BitwiseAnd(ArithmeticShiftR(Add(MultiplyHigh(UNALLOCATED:I, "
                        "MagicMultiple_0:I)I, UNALLOCATED:I)I, 31:I)I, Add(ShiftL(1:I, "
                        "MagicShifts_1:I)I, Conditional(Equal(MagicMultiple_0:I, 0:I)BL, -1:I, "
                        "0:I)I)I)I)I, MagicShifts_1:I)I, MagicSign_2:I)I, MagicSign_2:I)I";
        EXPECT_EQ(Expression::toString(expr_fast), expected);

        expr      = a_unsigned / b_unsigned;
        expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        setComment(expr_fast, "");
        EXPECT_EQ(Expression::toString(expr_fast),
                  "ArithmeticShiftR(Add(ArithmeticShiftR(Subtract(UNALLOCATED:U32, "
                  "MultiplyHigh(UNALLOCATED:U32, MagicMultiple_3:U32)U32)U32, 1:U32)U32, "
                  "MultiplyHigh(UNALLOCATED:U32, MagicMultiple_3:U32)U32)U32, MagicShifts_4:I)U32");
    }

    TEST_F(FastDivisionTest, ModuloByConstantExpressions)
    {
        auto command = std::make_shared<Command>();

        auto regPtr
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::UInt32, 1);
        auto reg = regPtr->expression();

        auto aTag = command->allocateTag();
        auto a    = std::make_shared<Expression::Expression>(command->allocateArgument(
            {DataType::Int32, PointerType::Value}, aTag, ArgumentType::Value));

        auto expr      = a % Expression::literal(8u);
        auto expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        EXPECT_EQ(Expression::toString(expr_fast),
                  "BitwiseAnd(CommandArgument(user_Int32_Value_0)I, 7:U32)U32");

        expr      = a % Expression::literal(8);
        expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        EXPECT_EQ(Expression::toString(expr_fast),
                  "Subtract(CommandArgument(user_Int32_Value_0)I, "
                  "BitwiseAnd(Add(CommandArgument(user_Int32_Value_0)I, "
                  "LogicalShiftR(ArithmeticShiftR(CommandArgument(user_Int32_Value_0)I, 31:U32)I, "
                  "29:U32)I)I, -8:I)I)I");

        expr      = a % Expression::literal(7u);
        expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        EXPECT_EQ(Expression::toString(expr_fast),
                  "Subtract(CommandArgument(user_Int32_Value_0)I, "
                  "Multiply(LogicalShiftR(Add(LogicalShiftR(Subtract(CommandArgument(user_Int32_"
                  "Value_0)I, MultiplyHigh(CommandArgument(user_Int32_Value_0)I, "
                  "613566757:U32)U32)U32, 1:U32)U32, "
                  "MultiplyHigh(CommandArgument(user_Int32_Value_0)I, 613566757:U32)U32)U32, "
                  "2:U32)U32, 7:U32)U32)U32");

        expr      = a % Expression::literal(1);
        expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        EXPECT_EQ(Expression::toString(expr_fast), "0:I");

        expr      = reg % Expression::literal(1u);
        expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        EXPECT_EQ(Expression::toString(expr_fast), "0:U32");

        expr      = a % Expression::literal(-5);
        expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        EXPECT_EQ(
            Expression::toString(expr_fast),
            "Subtract(CommandArgument(user_Int32_Value_0)I, "
            "Multiply(Add(ArithmeticShiftR(MultiplyHigh(CommandArgument(user_Int32_Value_0)I, "
            "-1717986919:I)I, 1:U32)I, "
            "LogicalShiftR(ArithmeticShiftR(MultiplyHigh(CommandArgument(user_Int32_Value_0)I, "
            "-1717986919:I)I, 1:U32)I, 31:U32)I)I, -5:I)I)I");

        expr      = a % std::make_shared<Expression::Expression>(8u);
        expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        EXPECT_EQ(Expression::toString(expr_fast),
                  "BitwiseAnd(CommandArgument(user_Int32_Value_0)I, 7:U32)U32");

        expr      = a % std::make_shared<Expression::Expression>(128u);
        expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        EXPECT_EQ(Expression::toString(expr_fast),
                  "BitwiseAnd(CommandArgument(user_Int32_Value_0)I, 127:U32)U32");
    }

    TEST_F(FastDivisionTest, ModuloByArgumentExpressions)
    {
        auto command = std::make_shared<Command>();

        auto reg
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Int32, 1);
        auto a = reg->expression();

        auto reg2
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::UInt32, 1);
        auto a_unsigned = reg2->expression();

        auto bSignedTag = command->allocateTag();
        auto b_signed   = std::make_shared<Expression::Expression>(command->allocateArgument(
            {DataType::Int32, PointerType::Value}, bSignedTag, ArgumentType::Value));

        auto bUnsignedTag = command->allocateTag();
        auto b_unsigned   = std::make_shared<Expression::Expression>(command->allocateArgument(
            {DataType::UInt32, PointerType::Value}, bUnsignedTag, ArgumentType::Value));

        auto        expr      = a % b_signed;
        auto        expr_fast = rocRoller::Expression::fastDivision(expr, m_context);
        std::string expected
            = "Subtract(UNALLOCATED:I, "
              "Multiply(Subtract(BitwiseXor(ArithmeticShiftR(Add(Add(MultiplyHigh(UNALLOCATED:I, "
              "MagicMultiple_0:I)I, UNALLOCATED:I)I, "
              "BitwiseAnd(ArithmeticShiftR(Add(MultiplyHigh(UNALLOCATED:I, MagicMultiple_0:I)I, "
              "UNALLOCATED:I)I, 31:I)I, Add(ShiftL(1:I, MagicShifts_1:I)I, "
              "Conditional(Equal(MagicMultiple_0:I, 0:I)BL, -1:I, 0:I)I)I)I)I, MagicShifts_1:I)I, "
              "MagicSign_2:I)I, MagicSign_2:I)I, CommandArgument(user_Int32_Value_0)I)I)I";
        EXPECT_EQ(Expression::toString(expr_fast), expected);

        expr      = a_unsigned % b_unsigned;
        expr_fast = rocRoller::Expression::fastDivision(expr, m_context);

        EXPECT_THAT(
            Expression::toString(expr_fast),
            testing::HasSubstr(
                "Subtract(UNALLOCATED:U32, "
                "Multiply(ArithmeticShiftR(Add(ArithmeticShiftR(Subtract("
                "UNALLOCATED:U32, MultiplyHigh(UNALLOCATED:U32, MagicMultiple_3:U32)U32)U32, "
                "1:U32)U32, MultiplyHigh(UNALLOCATED:U32, MagicMultiple_3:U32)U32)U32, "
                "MagicShifts_4:I)U32, CommandArgument(user_UInt32_Value_1)U32)U32)U32"));
    }

    namespace GPUFastDivisionTest
    {

        template <bool Modulo,
                  bool Defer,
                  typename Result,
                  typename AType = Result,
                  typename BType = AType>
        struct Options
        {
            inline static const bool IsModulo        = Modulo;
            inline static const bool DeferExpression = Defer;
            using R                                  = Result;
            using A                                  = AType;
            using B                                  = BType;
        };

        template <typename Options>
        class FastDivisionTestCurrentGPU : public CurrentGPUContextFixture
        {
        };

#define BoolCombinations(...)                                              \
    Options<false, false, __VA_ARGS__>, Options<false, true, __VA_ARGS__>, \
        Options<true, false, __VA_ARGS__>, Options<true, true, __VA_ARGS__>

        using TestedOptions = ::testing::Types<BoolCombinations(int32_t),
                                               BoolCombinations(uint32_t),
                                               BoolCombinations(int64_t, int32_t, int64_t),
                                               BoolCombinations(int64_t, int64_t, int32_t),
                                               BoolCombinations(int64_t)>;

#undef BoolCombinations

        TYPED_TEST_SUITE(FastDivisionTestCurrentGPU, TestedOptions);

        TYPED_TEST(FastDivisionTestCurrentGPU, GPU_FastDivision)
        {

            bool IsModulo        = TypeParam::IsModulo;
            bool DeferExpression = TypeParam::DeferExpression;
            using R              = typename TypeParam::R;
            using A              = typename TypeParam::A;
            using B              = typename TypeParam::B;

            auto command = std::make_shared<Command>();

            auto dataTypeA      = TypeInfo<A>::Var.dataType;
            auto dataTypeB      = TypeInfo<B>::Var.dataType;
            auto dataTypeResult = TypeInfo<R>::Var.dataType;

            auto infoResult = DataTypeInfo::Get(dataTypeResult);

            auto resultTag  = command->allocateTag();
            auto result_arg = command->allocateArgument(
                {dataTypeResult, PointerType::PointerGlobal}, resultTag, ArgumentType::Value);
            auto aTag  = command->allocateTag();
            auto a_arg = command->allocateArgument(
                {dataTypeA, PointerType::Value}, aTag, ArgumentType::Value);
            auto bTag  = command->allocateTag();
            auto b_arg = command->allocateArgument(
                {dataTypeB, PointerType::Value}, bTag, ArgumentType::Value);

            auto result_exp = std::make_shared<Expression::Expression>(result_arg);

            auto k = this->m_context->kernel();

            k->addArgument({result_arg->name(),
                            {dataTypeResult, PointerType::PointerGlobal},
                            DataDirection::WriteOnly,
                            result_exp});
            auto a_exp = k->addCommandArgument(a_arg);
            auto b_exp = k->addCommandArgument(b_arg);

            auto one  = std::make_shared<Expression::Expression>(1u);
            auto zero = std::make_shared<Expression::Expression>(0u);

            auto a_reg = Register::Value::Placeholder(
                this->m_context, Register::Type::Scalar, dataTypeResult, 1);

            if(dataTypeA != dataTypeResult)
                a_exp = convert(dataTypeResult, a_exp);

            if(dataTypeB != dataTypeResult)
                b_exp = convert(dataTypeResult, b_exp);

            std::shared_ptr<Expression::Expression> expr;

            if(IsModulo)
                expr = a_reg->expression() % b_exp;
            else
                expr = a_reg->expression() / b_exp;

            if(DeferExpression)
            {
                enableDivideBy(b_exp, this->m_context);
            }
            else
            {
                expr = fastDivision(expr, this->m_context);
                AssertFatal(!std::holds_alternative<Expression::Divide>(*expr), toString(expr));
                AssertFatal(!std::holds_alternative<Expression::Modulo>(*expr), toString(expr));

                AssertFatal(!contains<Expression::Divide>(expr), toString(expr));
                AssertFatal(!contains<Expression::Modulo>(expr), toString(expr));
            }

            k->setWorkgroupSize({1, 1, 1});
            k->setWorkitemCount({one, one, one});
            k->setDynamicSharedMemBytes(zero);

            this->m_context->schedule(k->preamble());
            this->m_context->schedule(k->prolog());

            auto kb = [&]() -> Generator<Instruction> {
                Register::ValuePtr s_result, s_a, s_b;
                co_yield this->m_context->argLoader()->getValue(result_arg->name(), s_result);
                co_yield this->m_context->argLoader()->getValue(a_arg->name(), s_a);
                co_yield this->m_context->argLoader()->getValue(b_arg->name(), s_b);

                co_yield generate(a_reg, a_exp, this->m_context);

                auto v_result
                    = Register::Value::Placeholder(this->m_context,
                                                   Register::Type::Vector,
                                                   {dataTypeResult, PointerType::PointerGlobal},
                                                   1);

                auto v_c = Register::Value::Placeholder(
                    this->m_context, Register::Type::Vector, dataTypeResult, 1);

                co_yield v_result->allocate();

                co_yield this->m_context->copier()->copy(v_result, s_result, "Move pointer");

                if(DeferExpression)
                {
                    expr = fastDivision(expr, this->m_context);
                }

                AssertFatal(!std::holds_alternative<Expression::Divide>(*expr), toString(expr));
                AssertFatal(!std::holds_alternative<Expression::Modulo>(*expr), toString(expr));

                AssertFatal(!contains<Expression::Divide>(expr), toString(expr));
                AssertFatal(!contains<Expression::Modulo>(expr), toString(expr));

                Register::ValuePtr s_c;
                co_yield Expression::generate(s_c, expr, this->m_context);
                AssertFatal(s_c->variableType() == dataTypeResult,
                            ShowValue(s_c->variableType()),
                            ShowValue(dataTypeResult),
                            ShowValue(expr));

                co_yield this->m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
                co_yield this->m_context->mem()->storeGlobal(
                    v_result, v_c, 0, CeilDivide(infoResult.elementBits, 8u));
            };

            this->m_context->schedule(kb());
            this->m_context->schedule(k->postamble());
            this->m_context->schedule(k->amdgpu_metadata());

            std::shared_ptr<rocRoller::ExecutableKernel> executableKernel
                = this->m_context->instructions()->getExecutableKernel();

            auto d_result = make_shared_device<R>();

            CommandKernel commandKernel;
            commandKernel.setContext(this->m_context);
            commandKernel.generateKernel();

            auto numerators   = TestValues::ByType<A>::values;
            auto denominators = TestValues::ByType<B>::values;

            for(A a : numerators)
            {
                for(B b : denominators)
                {
                    if(b == 0)
                        continue;

                    CommandArguments commandArgs = command->createArguments();

                    commandArgs.setArgument(resultTag, ArgumentType::Value, d_result.get());
                    commandArgs.setArgument(aTag, ArgumentType::Value, a);
                    commandArgs.setArgument(bTag, ArgumentType::Value, b);

                    if(dataTypeB == DataType::UInt32 && b == 1)
                    {
                        EXPECT_THROW(commandKernel.launchKernel(commandArgs.runtimeArguments()),
                                     FatalError);
                    }
                    else
                    {
                        commandKernel.launchKernel(commandArgs.runtimeArguments());

                        R result;
                        ASSERT_THAT(hipMemcpy(&result,
                                              d_result.get(),
                                              CeilDivide(infoResult.elementBits, 8u),
                                              hipMemcpyDefault),
                                    HasHipSuccess(0));

                        if(IsModulo)
                        {
                            EXPECT_EQ(result, a % b) << ShowValue(a) << ShowValue(dataTypeA)
                                                     << ShowValue(b) << ShowValue(dataTypeB);
                        }
                        else
                        {
                            auto bLibDivide = libdivide::libdivide_s64_branchfree_gen(b);
                            EXPECT_EQ(result,
                                      libdivide::libdivide_s64_branchfree_do(a, &bLibDivide))
                                << ShowValue(a) << ShowValue(dataTypeA) << ShowValue(b)
                                << ShowValue(dataTypeB);

                            // Sanity check
                            EXPECT_EQ(a / b,
                                      libdivide::libdivide_s64_branchfree_do(a, &bLibDivide));
                        }
                    }
                }
            }
        }
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

            DataType dataTypeA = rocRoller::TypeInfo<A>::Var.dataType;

            auto command = std::make_shared<Command>();

            auto resultTag  = command->allocateTag();
            auto result_arg = command->allocateArgument(
                {dataTypeA, PointerType::PointerGlobal}, resultTag, ArgumentType::Value);
            auto aTag  = command->allocateTag();
            auto a_arg = command->allocateArgument(
                {dataTypeA, PointerType::Value}, aTag, ArgumentType::Value);

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

                auto denom = Expression::literal(divisor);

                std::shared_ptr<Expression::Expression> expr = isModulo ? (a % denom) : (a / denom);

                Register::ValuePtr s_c;
                co_yield Expression::generate(s_c, expr, m_context);

                co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
                co_yield m_context->mem()->storeGlobal(
                    v_result, v_c, 0, v_c->variableType().getElementSize());
            };

            m_context->schedule(kb());
            m_context->schedule(k->postamble());
            m_context->schedule(k->amdgpu_metadata());

            std::shared_ptr<rocRoller::ExecutableKernel> executableKernel
                = m_context->instructions()->getExecutableKernel();

            auto d_result = make_shared_device<A>();

            CommandKernel commandKernel;
            commandKernel.setContext(m_context);
            commandKernel.generateKernel();

            for(A a : numerators)
            {
                CommandArguments commandArgs = command->createArguments();

                commandArgs.setArgument(resultTag, ArgumentType::Value, d_result.get());
                commandArgs.setArgument(aTag, ArgumentType::Value, a);

                commandKernel.launchKernel(commandArgs.runtimeArguments());

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
