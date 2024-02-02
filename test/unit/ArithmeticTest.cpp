
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>

#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureLibrary.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Utilities/Generator.hpp>
#include <rocRoller/Utilities/Utils.hpp>

#include "GPUContextFixture.hpp"
#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"
#include "TestValues.hpp"
#include "Utilities.hpp"

namespace ArithmeticTest
{

    template <typename T>
    struct identity
    {
        using type = T;
    };

    template <typename T>
    using try_make_unsigned =
        typename std::conditional_t<std::is_integral_v<T>, std::make_unsigned<T>, identity<T>>;

    template <typename T>
    using try_make_signed =
        typename std::conditional_t<std::is_integral_v<T>, std::make_signed<T>, identity<T>>;

    using namespace rocRoller;

    const int LITERAL_TEST = 227;

    struct IntegralArithmeticTest : public GPUContextFixtureParam<Register::Type>
    {
        template <typename T>
        void testBody()
        {
            auto dataType = TypeInfo<T>::Var.dataType;
            auto regType  = std::get<1>(GetParam());

            auto k = m_context->kernel();

            k->setKernelName("IntegralArithmeticTest");
            k->setKernelDimensions(1);

            auto command = std::make_shared<Command>();

            auto result_exp = std::make_shared<Expression::Expression>(
                command->allocateArgument({dataType, PointerType::PointerGlobal}));
            auto a_exp = std::make_shared<Expression::Expression>(
                command->allocateArgument({dataType, PointerType::Value}));
            auto b_exp = std::make_shared<Expression::Expression>(
                command->allocateArgument({dataType, PointerType::Value}));
            auto sh_exp = std::make_shared<Expression::Expression>(
                command->allocateArgument({DataType::UInt32, PointerType::Value}));

            auto one  = std::make_shared<Expression::Expression>(1u);
            auto zero = std::make_shared<Expression::Expression>(0u);

            k->addArgument({"result",
                            {dataType, PointerType::PointerGlobal},
                            DataDirection::WriteOnly,
                            result_exp});
            k->addArgument({"a", dataType, DataDirection::ReadOnly, a_exp});
            k->addArgument({"b", dataType, DataDirection::ReadOnly, b_exp});
            k->addArgument({"shift", DataType::UInt32, DataDirection::ReadOnly, sh_exp});

            k->setWorkgroupSize({1, 1, 1});
            k->setWorkitemCount({one, one, one});
            k->setDynamicSharedMemBytes(zero);

            m_context->schedule(k->preamble());
            m_context->schedule(k->prolog());

            auto kb = [&]() -> Generator<Instruction> {
                Register::ValuePtr s_result, s_a, s_b, s_shift;
                co_yield m_context->argLoader()->getValue("result", s_result);
                co_yield m_context->argLoader()->getValue("a", s_a);
                co_yield m_context->argLoader()->getValue("b", s_b);
                co_yield m_context->argLoader()->getValue("shift", s_shift);

                auto v_result = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, {dataType, PointerType::PointerGlobal}, 1);

                auto v_a
                    = Register::Value::Placeholder(m_context, Register::Type::Vector, dataType, 1);

                auto v_b
                    = Register::Value::Placeholder(m_context, Register::Type::Vector, dataType, 1);

                auto v_c
                    = Register::Value::Placeholder(m_context, Register::Type::Vector, dataType, 1);

                auto v_shift = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::UInt32, 1);

                auto mask = Register::Value::Placeholder(
                    m_context, Register::Type::Scalar, {dataType, PointerType::PointerGlobal}, 1);

                auto s_c
                    = Register::Value::Placeholder(m_context, Register::Type::Scalar, dataType, 1);

                co_yield v_a->allocate();
                co_yield v_b->allocate();
                co_yield v_c->allocate();
                co_yield v_shift->allocate();
                co_yield mask->allocate();
                co_yield s_c->allocate();
                co_yield v_result->allocate();

                co_yield m_context->copier()->copy(v_result, s_result, "Move pointer");

                co_yield m_context->copier()->copy(v_a, s_a, "Move value");
                co_yield m_context->copier()->copy(v_b, s_b, "Move value");
                co_yield m_context->copier()->copy(v_shift, s_shift, "Move value");

                auto const a            = regType == Register::Type::Vector ? v_a : s_a;
                auto const b            = regType == Register::Type::Vector ? v_b : s_b;
                auto const c            = regType == Register::Type::Vector ? v_c : s_c;
                auto const mask_or_bool = regType == Register::Type::Vector ? mask : s_c;
                auto const shift        = regType == Register::Type::Vector ? v_shift : s_shift;

                enum LaneWise_t
                {
                    LaneWise,
                    NotLaneWise
                };

                auto store
                    = [&](size_t idx, LaneWise_t laneWise = NotLaneWise) -> Generator<Instruction> {
                    if(laneWise == LaneWise)
                    {
                        // E.g. vgpr comparsion requires copying result from vcc-sized sgpr to a T-sized vgpr
                        // E.g. sgpr comparsion requries copying result from bool sgpr to a T-sized vgpr
                        if constexpr(sizeof(T) == 4)
                        {
                            co_yield m_context->copier()->copy(
                                v_c, mask->subset({0}), "Move result to vgpr to store.");
                            co_yield m_context->mem()->store(
                                MemoryInstructions::Flat,
                                v_result,
                                v_c->subset({0}),
                                Register::Value::Literal(idx * sizeof(T)),
                                sizeof(T));
                        }
                        else
                        {
                            co_yield m_context->copier()->copy(
                                v_c, mask, "Move result to vgpr to store.");
                            co_yield m_context->mem()->store(
                                MemoryInstructions::Flat,
                                v_result,
                                v_c,
                                Register::Value::Literal(idx * sizeof(T)),
                                sizeof(T));
                        }
                    }
                    else
                    {
                        if(regType == Register::Type::Scalar)
                        {
                            co_yield m_context->copier()->copy(
                                v_c, s_c, "Move result to vgpr to store.");
                        }
                        co_yield m_context->mem()->store(MemoryInstructions::Flat,
                                                         v_result,
                                                         v_c,
                                                         Register::Value::Literal(idx * sizeof(T)),
                                                         sizeof(T));
                    }
                };

                co_yield generateOp<Expression::Add>(c, a, b);
                co_yield store(0);

                co_yield generateOp<Expression::Subtract>(c, a, b);
                co_yield store(1);

                co_yield generateOp<Expression::Multiply>(c, a, b);
                co_yield store(2);

                co_yield generateOp<Expression::Divide>(c, a, b);
                co_yield store(3);

                co_yield generateOp<Expression::Modulo>(c, a, b);
                co_yield store(4);

                co_yield generateOp<Expression::ShiftL>(c, a, b);
                co_yield store(5);

                co_yield generateOp<Expression::LogicalShiftR>(c, a, b);
                co_yield store(6);

                co_yield generateOp<Expression::ArithmeticShiftR>(c, a, b);
                co_yield store(7);

                if(!std::is_same_v<T, uint64_t>)
                {
                    co_yield generateOp<Expression::GreaterThan>(mask, a, b);
                    co_yield store(8, LaneWise);

                    co_yield generateOp<Expression::GreaterThanEqual>(mask, a, b);
                    co_yield store(9, LaneWise);

                    co_yield generateOp<Expression::LessThan>(mask, a, b);
                    co_yield store(10, LaneWise);

                    co_yield generateOp<Expression::LessThanEqual>(mask, a, b);
                    co_yield store(11, LaneWise);

                    co_yield generateOp<Expression::Equal>(mask, a, b);
                    co_yield store(12, LaneWise);
                }

                co_yield generateOp<Expression::BitwiseAnd>(c, a, b);
                co_yield store(13);

                co_yield generateOp<Expression::MultiplyHigh>(c, a, b);
                co_yield store(14);

                co_yield generateOp<Expression::Negate>(c, a);
                co_yield store(15);

                co_yield generateOp<Expression::BitwiseXor>(c, a, b);
                co_yield store(16);

                co_yield generateOp<Expression::AddShiftL>(c, a, b, shift);
                co_yield store(17);

                co_yield generateOp<Expression::ShiftLAdd>(c, a, shift, b);
                co_yield store(18);

                co_yield generateOp<Expression::BitwiseOr>(c, a, b);
                co_yield store(19);

                co_yield generateOp<Expression::Divide>(
                    c, a, Register::Value::Literal(LITERAL_TEST));
                co_yield store(20);

                co_yield generateOp<Expression::Divide>(
                    c, Register::Value::Literal(LITERAL_TEST), b);
                co_yield store(21);

                co_yield generateOp<Expression::Modulo>(
                    c, a, Register::Value::Literal(LITERAL_TEST));
                co_yield store(22);

                co_yield generateOp<Expression::Modulo>(
                    c, Register::Value::Literal(LITERAL_TEST), b);
                co_yield store(23);

                // Logical
                auto A = regType == Register::Type::Vector ? v_a->expression() : s_a->expression();
                auto B = regType == Register::Type::Vector ? v_b->expression() : s_b->expression();
                if constexpr(sizeof(T) == 4)
                {
                    co_yield generate(mask,
                                      (A < Expression::literal(0, dataType))
                                          && (B < Expression::literal(0, dataType)),
                                      m_context);
                    co_yield store(24, LaneWise);

                    co_yield generateOp<Expression::GreaterThanEqual>(mask_or_bool, a, b);
                    co_yield generateOp<Expression::Conditional>(c, mask_or_bool, a, b);
                    co_yield store(25);

                    co_yield generate(mask,
                                      (A < Expression::literal(0, dataType))
                                          || (B < Expression::literal(0, dataType)),
                                      m_context);
                    co_yield store(26, LaneWise);

                    co_yield generateOp<Expression::NotEqual>(mask, a, b);
                    co_yield store(27, LaneWise);

                    auto s_c_lower = mask->subset({0});
                    co_yield generate(s_c_lower, Expression::logicalNot(A < B), m_context);
                    co_yield store(28, LaneWise);

                    co_yield generate(s_c_lower,
                                      Expression::logicalNot(Expression::logicalNot(A > B)),
                                      m_context);
                    co_yield store(29, LaneWise);
                }

                co_yield generateOp<Expression::BitwiseNegate>(c, a);
                co_yield store(30);
            };

            m_context->schedule(kb());
            m_context->schedule(k->postamble());
            m_context->schedule(k->amdgpu_metadata());

            REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);

            if(isLocalDevice())
            {
                CommandKernel commandKernel(m_context);

                size_t const result_size = 31;
                auto         d_result    = make_shared_device<T>(result_size);

                for(T a : TestValues::ByType<T>::values)
                {
                    for(T b : TestValues::ByType<T>::values)
                    {
                        using T_unsigned = typename try_make_unsigned<T>::type;
                        using T_signed   = typename try_make_signed<T>::type;

                        // Signed division/modulo is used for unsigned
                        auto withinDivisionDomain = b != 0
                                                    && a <= std::numeric_limits<T_signed>::max()
                                                    && b <= std::numeric_limits<T_signed>::max();

                        for(uint32_t shift : TestValues::shiftValues)
                        {
                            KernelArguments runtimeArgs;
                            runtimeArgs.append("result", d_result.get());
                            runtimeArgs.append("a", a);
                            runtimeArgs.append("b", b);
                            runtimeArgs.append("shift", shift);

                            commandKernel.launchKernel(runtimeArgs.runtimeArguments());

                            std::vector<T> result(result_size);
                            ASSERT_THAT(hipMemcpy(result.data(),
                                                  d_result.get(),
                                                  result.size() * sizeof(T),
                                                  hipMemcpyDefault),
                                        HasHipSuccess(0));

                            EXPECT_EQ(result[0], a + b);
                            EXPECT_EQ(result[1], a - b);
                            EXPECT_EQ(result[2], a * b);
                            if(withinDivisionDomain)
                            {
                                EXPECT_EQ(result[3], a / b);
                                EXPECT_EQ(result[4], a % b);
                            }
                            if(b < 32 && b >= 0)
                            {
                                EXPECT_EQ(result[5], a << b) << a << " " << b;
                                EXPECT_EQ(result[6], static_cast<T_unsigned>(a) >> b);
                                EXPECT_EQ(result[7], a >> b);
                            }
                            if constexpr(!std::is_same_v<T, uint64_t>)
                            {
                                EXPECT_EQ(result[8], (a > b ? 1 : 0));
                                EXPECT_EQ(result[9], (a >= b ? 1 : 0));
                                EXPECT_EQ(result[10], (a < b ? 1 : 0));
                                EXPECT_EQ(result[11], (a <= b ? 1 : 0));
                                EXPECT_EQ(result[12], (a == b ? 1 : 0));
                            }
                            EXPECT_EQ(result[13], a & b);
                            if constexpr(std::is_same_v<T_signed, int32_t>)
                            {
                                EXPECT_EQ(result[14],
                                          (a * (int64_t)b)
                                              >> std::numeric_limits<T_unsigned>::digits);
                            }
                            else if constexpr(std::is_same_v<T, int64_t>)
                            {
                                EXPECT_EQ(result[14],
                                          (int64_t)(((__int128_t)a * (__int128_t)b)
                                                    >> std::numeric_limits<T_unsigned>::digits));
                            }
                            EXPECT_EQ(result[15], -a);
                            EXPECT_EQ(result[16], a ^ b);
                            EXPECT_EQ(result[17], (a + b) << shift);
                            EXPECT_EQ(result[18], (a << shift) + b);
                            EXPECT_EQ(result[19], a | b);
                            if(withinDivisionDomain)
                            {
                                EXPECT_EQ(result[20], a / LITERAL_TEST);
                                EXPECT_EQ(result[21], LITERAL_TEST / b);
                                EXPECT_EQ(result[22], a % LITERAL_TEST);
                                EXPECT_EQ(result[23], LITERAL_TEST % b);
                            }
                            if constexpr(sizeof(T) == 4)
                            {
                                EXPECT_EQ(result[24], ((a < 0) && (b < 0)) ? 1 : 0);
                                EXPECT_EQ(result[25], (a >= b ? a : b))
                                    << "a: " << a << ", b: " << b << ", shift: " << shift;
                                EXPECT_EQ(result[26], ((a < 0) || (b < 0)) ? 1 : 0);
                                EXPECT_EQ(result[27], (a != b ? 1 : 0))
                                    << "a: " << a << ", b: " << b << ", shift: " << shift;
                                EXPECT_EQ(result[28], !(a < b) ? 1 : 0);
                                EXPECT_EQ(result[29], !!(a > b) ? 1 : 0);
                            }
                            EXPECT_EQ(result[30], ~a);
                        }
                    }
                }
            }
            else
            {
                std::vector<char> assembledKernel = m_context->instructions()->assemble();
                EXPECT_GT(assembledKernel.size(), 0);
            }
        }
    };

    TEST_P(IntegralArithmeticTest, Int32)
    {
        testBody<int32_t>();
    }

    TEST_P(IntegralArithmeticTest, UInt32)
    {
        testBody<uint32_t>();
    }

    TEST_P(IntegralArithmeticTest, Int64)
    {
        testBody<int64_t>();
    }

    TEST_P(IntegralArithmeticTest, UInt64)
    {
        testBody<uint64_t>();
    }

    INSTANTIATE_TEST_SUITE_P(NewArithmeticTests,
                             IntegralArithmeticTest,
                             ::testing::Combine(supportedISAValues(),
                                                ::testing::Values(Register::Type::Vector,
                                                                  Register::Type::Scalar)));

    class FPArithmeticTest : public GPUContextFixture
    {
    public:
        FPArithmeticTest() {}
    };

    TEST_P(FPArithmeticTest, ArithFloat)
    {
        auto k = m_context->kernel();

        k->setKernelName("ArithFloat");
        k->setKernelDimensions(1);

        auto command = std::make_shared<Command>();

        auto result_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Float, PointerType::PointerGlobal}));
        auto cond_result_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Int32, PointerType::PointerGlobal}));
        auto a_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Float, PointerType::Value}));
        auto b_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Float, PointerType::Value}));
        auto c_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Float, PointerType::Value}));

        auto one  = std::make_shared<Expression::Expression>(1u);
        auto zero = std::make_shared<Expression::Expression>(0u);

        k->addArgument({"result",
                        {DataType::Float, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        result_exp});
        k->addArgument({"cond_result",
                        {DataType::Int32, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        cond_result_exp});
        k->addArgument({"a", DataType::Float, DataDirection::ReadOnly, a_exp});
        k->addArgument({"b", DataType::Float, DataDirection::ReadOnly, b_exp});
        k->addArgument({"c", DataType::Float, DataDirection::ReadOnly, c_exp});

        k->setWorkgroupSize({1, 1, 1});
        k->setWorkitemCount({one, one, one});
        k->setDynamicSharedMemBytes(zero);

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_result, s_cond_result, s_a, s_b, s_c;
            co_yield m_context->argLoader()->getValue("result", s_result);
            co_yield m_context->argLoader()->getValue("cond_result", s_cond_result);
            co_yield m_context->argLoader()->getValue("a", s_a);
            co_yield m_context->argLoader()->getValue("b", s_b);
            co_yield m_context->argLoader()->getValue("c", s_c);

            auto v_result
                = Register::Value::Placeholder(m_context,
                                               Register::Type::Vector,
                                               {DataType::Float, PointerType::PointerGlobal},
                                               1);

            auto v_cond_result
                = Register::Value::Placeholder(m_context,
                                               Register::Type::Vector,
                                               {DataType::Float, PointerType::PointerGlobal},
                                               1);

            auto v_a = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Float, 1);

            auto v_b = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Float, 1);

            auto v_c = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Float, 1);

            auto v_r = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Float, 1);

            auto s_r = Register::Value::Placeholder(m_context,
                                                    Register::Type::Scalar,
                                                    {DataType::Float, PointerType::PointerGlobal},
                                                    1);

            co_yield v_a->allocate();
            co_yield v_b->allocate();
            co_yield v_c->allocate();
            co_yield v_r->allocate();
            co_yield s_r->allocate();
            co_yield v_result->allocate();
            co_yield v_cond_result->allocate();

            co_yield m_context->copier()->copy(v_result, s_result, "Move pointer");

            co_yield m_context->copier()->copy(v_cond_result, s_cond_result, "Move pointer");

            co_yield m_context->copier()->copy(v_a, s_a, "Move value");
            co_yield m_context->copier()->copy(v_b, s_b, "Move value");
            co_yield m_context->copier()->copy(v_c, s_c, "Move value");

            co_yield generateOp<Expression::Add>(v_r, v_a, v_b);
            co_yield m_context->mem()->storeFlat(v_result, v_r, 0, 4);

            co_yield generateOp<Expression::Subtract>(v_r, v_a, v_b);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_r, Register::Value::Literal(4), 4);

            co_yield generateOp<Expression::Multiply>(v_r, v_a, v_b);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_r, Register::Value::Literal(8), 4);

            co_yield generateOp<Expression::Negate>(v_r, v_a);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_r, Register::Value::Literal(12), 4);

            co_yield generateOp<Expression::MultiplyAdd>(v_r, v_a, v_b, v_c);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_r, Register::Value::Literal(16), 4);

            auto vcc = m_context->getVCC();
            co_yield generate(vcc, v_c->expression() >= v_a->expression(), m_context);
            co_yield generateOp<Expression::Conditional>(v_r, vcc, v_a, v_b);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_r, Register::Value::Literal(20), 4);

            co_yield generateOp<Expression::GreaterThan>(s_r, v_a, v_b);
            co_yield m_context->copier()->copy(
                v_r, s_r->subset({0}), "Move result to vgpr to store.");
            co_yield m_context->mem()->storeFlat(v_cond_result, v_r->subset({0}), 0, 4);

            co_yield generateOp<Expression::GreaterThanEqual>(s_r, v_a, v_b);
            co_yield m_context->copier()->copy(
                v_r, s_r->subset({0}), "Move result to vgpr to store.");
            co_yield m_context->mem()->storeFlat(v_cond_result, v_r->subset({0}), 4, 4);

            co_yield generateOp<Expression::LessThan>(s_r, v_a, v_b);
            co_yield m_context->copier()->copy(
                v_r, s_r->subset({0}), "Move result to vgpr to store.");
            co_yield m_context->mem()->storeFlat(v_cond_result, v_r->subset({0}), 8, 4);

            co_yield generateOp<Expression::LessThanEqual>(s_r, v_a, v_b);
            co_yield m_context->copier()->copy(
                v_r, s_r->subset({0}), "Move result to vgpr to store.");
            co_yield m_context->mem()->storeFlat(v_cond_result, v_r->subset({0}), 12, 4);

            co_yield generateOp<Expression::Equal>(s_r, v_a, v_b);
            co_yield m_context->copier()->copy(
                v_r, s_r->subset({0}), "Move result to vgpr to store.");
            co_yield m_context->mem()->storeFlat(v_cond_result, v_r->subset({0}), 16, 4);

            co_yield generateOp<Expression::NotEqual>(s_r, v_a, v_b);
            co_yield m_context->copier()->copy(
                v_r, s_r->subset({0}), "Move result to vgpr to store.");
            co_yield m_context->mem()->storeFlat(v_cond_result, v_r->subset({0}), 20, 4);
        };

        m_context->schedule(kb());
        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        if(m_context->targetArchitecture().target().getMajorVersion() != 9)
        {
            GTEST_SKIP() << "Skipping GPU arithmetic tests for " << GetParam();
        }

        // Only execute the kernels if running on the architecture that the kernel was built for,        // otherwise just assemble the instructions.
        if(isLocalDevice())
        {
            CommandKernel commandKernel(m_context);

            auto d_result      = make_shared_device<float>(6);
            auto d_cond_result = make_shared_device<int>(6);

            for(float a : TestValues::floatValues)
            {
                for(float b : TestValues::floatValues)
                {
                    for(float c : TestValues::floatValues)
                    {

                        KernelArguments runtimeArgs;
                        runtimeArgs.append("result", d_result.get());
                        runtimeArgs.append("cond_result", d_cond_result.get());
                        runtimeArgs.append("a", a);
                        runtimeArgs.append("b", b);
                        runtimeArgs.append("c", c);

                        commandKernel.launchKernel(runtimeArgs.runtimeArguments());

                        std::vector<float> result(6);
                        ASSERT_THAT(hipMemcpy(result.data(),
                                              d_result.get(),
                                              result.size() * sizeof(float),
                                              hipMemcpyDefault),
                                    HasHipSuccess(0));

                        std::vector<int> cond_result(6);
                        ASSERT_THAT(hipMemcpy(cond_result.data(),
                                              d_cond_result.get(),
                                              cond_result.size() * sizeof(int),
                                              hipMemcpyDefault),
                                    HasHipSuccess(0));

                        EXPECT_EQ(result[0], a + b);
                        EXPECT_EQ(result[1], a - b);
                        EXPECT_EQ(result[2], a * b);
                        EXPECT_EQ(result[3], -a);
                        float fma = a * b + c;
                        if(fma != 0.0)
                        {
                            EXPECT_LT((result[4] - fma) / fma, 5.e-7);
                        }
                        EXPECT_EQ(result[5], c >= a ? a : b)
                            << "a: " << a << ", b: " << b << ", c: " << c;
                        ;
                        EXPECT_EQ(cond_result[0], (a > b ? 1 : 0));
                        EXPECT_EQ(cond_result[1], (a >= b ? 1 : 0));
                        EXPECT_EQ(cond_result[2], (a < b ? 1 : 0));
                        EXPECT_EQ(cond_result[3], (a <= b ? 1 : 0));
                        EXPECT_EQ(cond_result[4], (a == b ? 1 : 0));
                        EXPECT_EQ(cond_result[5], (a != b ? 1 : 0));
                    }
                }
            }
        }
        else
        {
            std::vector<char> assembledKernel = m_context->instructions()->assemble();
            EXPECT_GT(assembledKernel.size(), 0);
        }
    }

    TEST_P(FPArithmeticTest, ArithFMAMixed)
    {
        auto k = m_context->kernel();

        k->setKernelName("ArithFMAMixed");
        k->setKernelDimensions(1);

        auto command = std::make_shared<Command>();

        auto result_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Float, PointerType::PointerGlobal}));
        auto a_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Float, PointerType::Value}));
        auto b_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Half, PointerType::PointerGlobal}));
        auto c_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Float, PointerType::PointerGlobal}));

        auto one  = std::make_shared<Expression::Expression>(1u);
        auto zero = std::make_shared<Expression::Expression>(0u);

        k->addArgument({"result",
                        {DataType::Float, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        result_exp});
        k->addArgument({"a", DataType::Float, DataDirection::ReadOnly, a_exp});
        k->addArgument(
            {"b", {DataType::Half, PointerType::PointerGlobal}, DataDirection::ReadOnly, b_exp});
        k->addArgument(
            {"c", {DataType::Float, PointerType::PointerGlobal}, DataDirection::ReadOnly, c_exp});

        k->setWorkgroupSize({1, 1, 1});
        k->setWorkitemCount({one, one, one});
        k->setDynamicSharedMemBytes(zero);

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
                                               {DataType::Int64, PointerType::PointerGlobal},
                                               1);

            auto vbPtr = Register::Value::Placeholder(m_context,
                                                      Register::Type::Vector,
                                                      {DataType::Int64, PointerType::PointerGlobal},
                                                      1);

            auto vcPtr = Register::Value::Placeholder(m_context,
                                                      Register::Type::Vector,
                                                      {DataType::Int64, PointerType::PointerGlobal},
                                                      1);

            auto v_b = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Halfx2, 1);

            auto v_c = Register::Value::Placeholder(m_context,
                                                    Register::Type::Vector,
                                                    DataType::Float,
                                                    2,
                                                    Register::AllocationOptions::FullyContiguous());

            auto v_r = Register::Value::Placeholder(m_context,
                                                    Register::Type::Vector,
                                                    DataType::Float,
                                                    2,
                                                    Register::AllocationOptions::FullyContiguous());

            co_yield vbPtr->allocate();
            co_yield vcPtr->allocate();
            co_yield v_b->allocate();
            co_yield v_c->allocate();
            co_yield v_r->allocate();
            co_yield v_result->allocate();

            co_yield m_context->copier()->copy(v_result, s_result, "Move pointer");

            co_yield m_context->copier()->copy(vbPtr, s_b, "Move Pointer");
            co_yield m_context->copier()->copy(vcPtr, s_c, "Move Pointer");
            co_yield m_context->mem()->loadFlat(v_b, vbPtr, 0, 4);
            co_yield m_context->mem()->loadFlat(v_c, vcPtr, 0, 8);

            // fp32 = fp32 * fp16 + fp32
            co_yield generateOp<Expression::MultiplyAdd>(v_r, s_a, v_b, v_c);
            co_yield m_context->mem()->store(MemoryInstructions::Flat, v_result, v_r, 0, 8);

            // fp32 = fp16 * fp16 + fp32
            co_yield generateOp<Expression::MultiplyAdd>(v_r, v_b, v_b, v_c);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_r, Register::Value::Literal(8), 8);

            // fp32 = fp16 * fp32 + fp16
            co_yield generateOp<Expression::MultiplyAdd>(v_r, v_b, v_c, v_b);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_r, Register::Value::Literal(16), 8);

            // fp32 = fp32 * fp16 + fp16
            co_yield generateOp<Expression::MultiplyAdd>(v_r, s_a, v_b, v_b);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_r, Register::Value::Literal(24), 8);

            // fp32 = fp32 * fp32 + fp16
            co_yield generateOp<Expression::MultiplyAdd>(v_r, s_a, v_c, v_b);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_r, Register::Value::Literal(32), 8);

            // fp32 = fp16 * fp32 + fp32
            co_yield generateOp<Expression::MultiplyAdd>(v_r, v_b, v_c, v_c);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_r, Register::Value::Literal(40), 8);
        };

        m_context->schedule(kb());
        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        if(m_context->targetArchitecture().target().getMajorVersion() != 9
           || m_context->targetArchitecture().target().getVersionString() == "gfx900")
        {
            GTEST_SKIP() << "Skipping GPU arithmetic tests for " << GetParam();
        }

        // Only execute the kernels if running on the architecture that the kernel was built for,        // otherwise just assemble the instructions.
        if(isLocalDevice())
        {
            CommandKernel commandKernel(m_context);

            float               a       = 2.f;
            std::vector<__half> b       = {static_cast<__half>(1.), static_cast<__half>(2.)};
            std::vector<float>  c       = {1.f, 2.f};
            auto                bDevice = make_shared_device<__half>(b);
            auto                cDevice = make_shared_device<float>(c);
            auto                dResult = make_shared_device<float>(12);

            KernelArguments runtimeArgs;
            runtimeArgs.append("result", dResult.get());
            runtimeArgs.append("a", a);
            runtimeArgs.append("b", bDevice.get());
            runtimeArgs.append("c", cDevice.get());
            commandKernel.launchKernel(runtimeArgs.runtimeArguments());
            //6 different options
            std::vector<float> result(12);
            ASSERT_THAT(
                hipMemcpy(
                    result.data(), dResult.get(), result.size() * sizeof(float), hipMemcpyDefault),
                HasHipSuccess(0));

            // fp32 * fp16 + f32
            float fma1 = a * static_cast<float>(b[0]) + c[0];
            float fma2 = a * static_cast<float>(b[1]) + c[1];

            // fp16 * fp16 + fp32
            float fma3 = static_cast<float>(b[0]) * static_cast<float>(b[0]) + c[0];
            float fma4 = static_cast<float>(b[1]) * static_cast<float>(b[1]) + c[1];

            // fp16 * fp32 + fp16
            float fma5 = static_cast<float>(b[0]) * c[0] + static_cast<float>(b[0]);
            float fma6 = static_cast<float>(b[1]) * c[1] + static_cast<float>(b[1]);

            // fp32 * fp16 + fp16
            float fma7 = a * static_cast<float>(b[0]) + static_cast<float>(b[0]);
            float fma8 = a * static_cast<float>(b[1]) + static_cast<float>(b[1]);

            // fp32 * fp32 + fp16
            float fma9  = a * c[0] + static_cast<float>(b[0]);
            float fma10 = a * c[1] + static_cast<float>(b[1]);

            // fp16 * fp32 + fp32
            float fma11 = static_cast<float>(b[0]) * c[0] + c[0];
            float fma12 = static_cast<float>(b[1]) * c[1] + c[1];

            EXPECT_LT(std::abs(result[0] - fma1) / fma1, 5.e-7);
            EXPECT_LT(std::abs(result[1] - fma2) / fma2, 5.e-7);
            EXPECT_LT(std::abs(result[2] - fma3) / fma3, 5.e-7);
            EXPECT_LT(std::abs(result[3] - fma4) / fma4, 5.e-7);
            EXPECT_LT(std::abs(result[4] - fma5) / fma5, 5.e-7);
            EXPECT_LT(std::abs(result[5] - fma6) / fma6, 5.e-7);
            EXPECT_LT(std::abs(result[6] - fma7) / fma7, 5.e-7);
            EXPECT_LT(std::abs(result[7] - fma8) / fma8, 5.e-7);
            EXPECT_LT(std::abs(result[8] - fma9) / fma9, 5.e-7);
            EXPECT_LT(std::abs(result[9] - fma10) / fma10, 5.e-7);
            EXPECT_LT(std::abs(result[10] - fma11) / fma11, 5.e-7);
            EXPECT_LT(std::abs(result[11] - fma12) / fma12, 5.e-7);
        }
        else
        {
            std::vector<char> assembledKernel = m_context->instructions()->assemble();
            EXPECT_GT(assembledKernel.size(), 0);
        }
    }

    TEST_P(FPArithmeticTest, ArithDouble)
    {
        auto k = m_context->kernel();

        k->setKernelName("ArithDouble");
        k->setKernelDimensions(1);

        auto command = std::make_shared<Command>();

        auto result_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Double, PointerType::PointerGlobal}));
        auto cond_result_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Int32, PointerType::PointerGlobal}));
        auto a_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Double, PointerType::Value}));
        auto b_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Double, PointerType::Value}));

        auto one  = std::make_shared<Expression::Expression>(1u);
        auto zero = std::make_shared<Expression::Expression>(0u);

        k->addArgument({"result",
                        {DataType::Double, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        result_exp});
        k->addArgument({"cond_result",
                        {DataType::Int32, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        cond_result_exp});
        k->addArgument({"a", DataType::Double, DataDirection::ReadOnly, a_exp});
        k->addArgument({"b", DataType::Double, DataDirection::ReadOnly, b_exp});

        k->setWorkgroupSize({1, 1, 1});
        k->setWorkitemCount({one, one, one});
        k->setDynamicSharedMemBytes(zero);

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_result, s_cond_result, s_a, s_b;
            co_yield m_context->argLoader()->getValue("result", s_result);
            co_yield m_context->argLoader()->getValue("cond_result", s_cond_result);
            co_yield m_context->argLoader()->getValue("a", s_a);
            co_yield m_context->argLoader()->getValue("b", s_b);

            auto v_result
                = Register::Value::Placeholder(m_context,
                                               Register::Type::Vector,
                                               {DataType::Double, PointerType::PointerGlobal},
                                               1);

            auto v_cond_result
                = Register::Value::Placeholder(m_context,
                                               Register::Type::Vector,
                                               {DataType::Double, PointerType::PointerGlobal},
                                               1);

            auto v_a = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Double, 1);

            auto v_b = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Double, 1);

            auto v_c = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Double, 1);

            auto s_c = Register::Value::Placeholder(m_context,
                                                    Register::Type::Scalar,
                                                    {DataType::Double, PointerType::PointerGlobal},
                                                    1);

            co_yield v_a->allocate();
            co_yield v_b->allocate();
            co_yield v_c->allocate();
            co_yield s_c->allocate();
            co_yield v_result->allocate();
            co_yield v_cond_result->allocate();

            co_yield m_context->copier()->copy(v_result, s_result, "Move pointer");

            co_yield m_context->copier()->copy(v_cond_result, s_cond_result, "Move pointer");

            co_yield m_context->copier()->copy(v_a, s_a, "Move value");
            co_yield m_context->copier()->copy(v_b, s_b, "Move value");

            co_yield generateOp<Expression::Add>(v_c, v_a, v_b);
            co_yield m_context->mem()->storeFlat(v_result, v_c, 0, 8);

            co_yield generateOp<Expression::Subtract>(v_c, v_a, v_b);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(8), 8);

            co_yield generateOp<Expression::Multiply>(v_c, v_a, v_b);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(16), 8);

            co_yield generateOp<Expression::Negate>(v_c, v_a);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(24), 8);

            auto vcc = m_context->getVCC();
            co_yield generate(vcc, v_a->expression() >= v_b->expression(), m_context);
            co_yield generateOp<Expression::Conditional>(v_c, vcc, v_a, v_b);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(32), 8);

            co_yield generateOp<Expression::GreaterThan>(s_c, v_a, v_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");

            co_yield m_context->mem()->store(MemoryInstructions::Flat,
                                             v_cond_result,
                                             v_c->subset({0}),
                                             Register::Value::Literal(0),
                                             4);

            co_yield generateOp<Expression::GreaterThanEqual>(s_c, v_a, v_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");

            co_yield m_context->mem()->store(MemoryInstructions::Flat,
                                             v_cond_result,
                                             v_c->subset({0}),
                                             Register::Value::Literal(4),
                                             4);

            co_yield generateOp<Expression::LessThan>(s_c, v_a, v_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(MemoryInstructions::Flat,
                                             v_cond_result,
                                             v_c->subset({0}),
                                             Register::Value::Literal(8),
                                             4);

            co_yield generateOp<Expression::LessThanEqual>(s_c, v_a, v_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(MemoryInstructions::Flat,
                                             v_cond_result,
                                             v_c->subset({0}),
                                             Register::Value::Literal(12),
                                             4);

            co_yield generateOp<Expression::Equal>(s_c, v_a, v_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(MemoryInstructions::Flat,
                                             v_cond_result,
                                             v_c->subset({0}),
                                             Register::Value::Literal(16),
                                             4);

            co_yield generateOp<Expression::NotEqual>(s_c, v_a, v_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(MemoryInstructions::Flat,
                                             v_cond_result,
                                             v_c->subset({0}),
                                             Register::Value::Literal(20),
                                             4);
        };

        m_context->schedule(kb());
        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        if(m_context->targetArchitecture().target().getMajorVersion() != 9)
        {
            GTEST_SKIP() << "Skipping GPU arithmetic tests for " << GetParam();
        }

        // Only execute the kernels if running on the architecture that the kernel was built for,        // otherwise just assemble the instructions.
        if(isLocalDevice())
        {
            CommandKernel commandKernel(m_context);

            auto d_result      = make_shared_device<double>(5);
            auto d_cond_result = make_shared_device<int>(6);

            for(double a : TestValues::doubleValues)
            {
                for(double b : TestValues::doubleValues)
                {
                    KernelArguments runtimeArgs;
                    runtimeArgs.append("result", d_result.get());
                    runtimeArgs.append("cond_result", d_cond_result.get());
                    runtimeArgs.append("a", a);
                    runtimeArgs.append("b", b);

                    commandKernel.launchKernel(runtimeArgs.runtimeArguments());

                    std::vector<double> result(5);
                    ASSERT_THAT(hipMemcpy(result.data(),
                                          d_result.get(),
                                          result.size() * sizeof(double),
                                          hipMemcpyDefault),
                                HasHipSuccess(0));

                    std::vector<int> cond_result(6);
                    ASSERT_THAT(hipMemcpy(cond_result.data(),
                                          d_cond_result.get(),
                                          cond_result.size() * sizeof(int),
                                          hipMemcpyDefault),
                                HasHipSuccess(0));

                    EXPECT_EQ(result[0], a + b);
                    EXPECT_EQ(result[1], a - b);
                    EXPECT_EQ(result[2], a * b);
                    EXPECT_EQ(result[3], -a);
                    EXPECT_EQ(result[4], a >= b ? a : b);
                    EXPECT_EQ(cond_result[0], (a > b ? 1 : 0));
                    EXPECT_EQ(cond_result[1], (a >= b ? 1 : 0));
                    EXPECT_EQ(cond_result[2], (a < b ? 1 : 0));
                    EXPECT_EQ(cond_result[3], (a <= b ? 1 : 0));
                    EXPECT_EQ(cond_result[4], (a == b ? 1 : 0));
                    EXPECT_EQ(cond_result[5], (a != b ? 1 : 0));
                }
            }
        }
        else
        {
            std::vector<char> assembledKernel = m_context->instructions()->assemble();
            EXPECT_GT(assembledKernel.size(), 0);
        }
    }

    TEST_P(FPArithmeticTest, NullChecks)
    {
        (void)(::testing::GTEST_FLAG(death_test_style) = "threadsafe");

        auto kb_null = [&]() -> Generator<Instruction> {
            co_yield generateOp<Expression::Subtract>(nullptr, nullptr, nullptr);
        };
        ASSERT_THROW(m_context->schedule(kb_null()), FatalError);

        auto kb32 = [&]() -> Generator<Instruction> {
            auto v_c = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int32, 1);

            co_yield v_c->allocate();
            co_yield generateOp<Expression::Subtract>(v_c, nullptr, nullptr);
        };
        ASSERT_THROW(m_context->schedule(kb32()), FatalError);

        auto kb64 = [&]() -> Generator<Instruction> {
            auto v_c = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int64, 1);

            co_yield v_c->allocate();
            co_yield generateOp<Expression::Subtract>(v_c, nullptr, nullptr);
        };
        ASSERT_THROW(m_context->schedule(kb64()), FatalError);

        auto kb_dst = [&]() -> Generator<Instruction> {
            auto v_a = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int64, 1);
            auto v_b = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int64, 1);

            co_yield v_a->allocate();
            co_yield v_b->allocate();
            co_yield generateOp<Expression::Subtract>(nullptr, v_a, v_b);
        };
        ASSERT_THROW(m_context->schedule(kb_dst()), FatalError);
    }

    INSTANTIATE_TEST_SUITE_P(ArithmeticTests, FPArithmeticTest, supportedISATuples());
}
