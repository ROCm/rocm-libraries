
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/Operations/Command.hpp>

#include "GPUContextFixture.hpp"
#include "Utilities.hpp"

using namespace rocRoller;

namespace rocRollerTest
{
    class HalfPrecisionTest : public CurrentGPUContextFixture
    {
    };

    void genHalfPrecisionMultiplyAdd(std::shared_ptr<rocRoller::Context> m_context, int N)
    {
        auto k = m_context->kernel();

        k->setKernelDimensions(1);
        auto command = std::make_shared<Command>();

        auto result_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Half, PointerType::PointerGlobal}));
        auto a_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Half, PointerType::PointerGlobal}));
        auto b_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Half, PointerType::PointerGlobal}));

        auto one  = std::make_shared<Expression::Expression>(1u);
        auto zero = std::make_shared<Expression::Expression>(0u);

        k->addArgument({"result",
                        {DataType::Half, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        result_exp});
        k->addArgument(
            {"a", {DataType::Half, PointerType::PointerGlobal}, DataDirection::ReadOnly, a_exp});
        k->addArgument(
            {"b", {DataType::Half, PointerType::PointerGlobal}, DataDirection::ReadOnly, b_exp});

        k->setWorkgroupSize({1, 1, 1});
        k->setWorkitemCount({one, one, one});
        k->setDynamicSharedMemBytes(zero);

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_result, s_a, s_b;
            co_yield m_context->argLoader()->getValue("result", s_result);
            co_yield m_context->argLoader()->getValue("a", s_a);
            co_yield m_context->argLoader()->getValue("b", s_b);

            auto v_result = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Raw32, 2);

            auto a_ptr = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Raw32, 2);

            auto b_ptr = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Raw32, 2);

            auto v_a = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Halfx2, 1);

            auto v_b = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Halfx2, 1);

            co_yield v_a->allocate();
            co_yield v_b->allocate();
            co_yield a_ptr->allocate();
            co_yield v_result->allocate();

            co_yield m_context->copier()->copy(v_result, s_result, "Move pointer.");

            co_yield m_context->copier()->copy(a_ptr, s_a, "Move pointer.");
            co_yield m_context->copier()->copy(b_ptr, s_b, "Move pointer.");

            for(int i = 0; i < N / 2; i++)
            {
                co_yield m_context->mem()->loadFlat(v_a, a_ptr, i * 4, 4);
                co_yield m_context->mem()->loadFlat(v_b, b_ptr, i * 4, 4);

                co_yield generateOp<Expression::Multiply>(v_b, v_a, v_b);
                co_yield generateOp<Expression::Add>(v_a, v_a, v_b);

                co_yield m_context->mem()->storeFlat(v_result, v_a, i * 4, 4);
            }
        };

        m_context->schedule(kb());
        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());
    }

    void genHalfPrecisionMultiplyAddConvert(std::shared_ptr<rocRoller::Context> m_context, int N)
    {
        auto k = m_context->kernel();

        k->setKernelDimensions(1);
        auto command = std::make_shared<Command>();

        auto result_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Half, PointerType::PointerGlobal}));
        auto a_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Half, PointerType::PointerGlobal}));
        auto b_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Half, PointerType::PointerGlobal}));

        auto one  = std::make_shared<Expression::Expression>(1u);
        auto zero = std::make_shared<Expression::Expression>(0u);

        k->addArgument({"result",
                        {DataType::Half, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        result_exp});
        k->addArgument(
            {"a", {DataType::Half, PointerType::PointerGlobal}, DataDirection::ReadOnly, a_exp});
        k->addArgument(
            {"b", {DataType::Half, PointerType::PointerGlobal}, DataDirection::ReadOnly, b_exp});

        k->setWorkgroupSize({1, 1, 1});
        k->setWorkitemCount({one, one, one});
        k->setDynamicSharedMemBytes(zero);

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_result, s_a, s_b;
            co_yield m_context->argLoader()->getValue("result", s_result);
            co_yield m_context->argLoader()->getValue("a", s_a);
            co_yield m_context->argLoader()->getValue("b", s_b);

            auto v_result = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Raw32, 2);

            auto a_ptr = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Raw32, 2);

            auto b_ptr = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Raw32, 2);

            auto v_a = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Half, 1);

            auto v_b = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Half, 1);

            auto v_a_float = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Float, 1);

            auto v_b_float = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Float, 1);

            auto a    = v_a->expression();
            auto b    = v_b->expression();
            auto expr = Expression::convert<DataType::Half>(
                Expression::convert<DataType::Float>(a)
                + (Expression::convert<DataType::Float>(a)
                   * Expression::convert<DataType::Float>(b)));

            co_yield v_a->allocate();
            co_yield v_b->allocate();
            co_yield v_a_float->allocate();
            co_yield v_b_float->allocate();
            co_yield a_ptr->allocate();
            co_yield v_result->allocate();

            co_yield m_context->copier()->copy(v_result, s_result, "Move pointer.");

            co_yield m_context->copier()->copy(a_ptr, s_a, "Move pointer.");
            co_yield m_context->copier()->copy(b_ptr, s_b, "Move pointer.");

            for(int i = 0; i < N; i++)
            {
                co_yield m_context->mem()->loadFlat(v_a, a_ptr, i * 2, 2);
                co_yield m_context->mem()->loadFlat(v_b, b_ptr, i * 2, 2);

                co_yield Expression::generate(v_a, expr, m_context);

                co_yield m_context->mem()->storeFlat(v_result, v_a, i * 2, 2);
            }
        };

        m_context->schedule(kb());
        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());
    }

    void executeHalfPrecisionMultiplyAdd(std::shared_ptr<rocRoller::Context> m_context,
                                         int                                 N,
                                         bool                                convertToFloat)
    {
        if(convertToFloat)
            genHalfPrecisionMultiplyAddConvert(m_context, N);
        else
            genHalfPrecisionMultiplyAdd(m_context, N);

        CommandKernel     commandKernel(m_context);
        RandomGenerator   random(314273u);
        auto              a = random.vector<Half>(N, -1.0, 1.0);
        auto              b = random.vector<Half>(N, -1.0, 1.0);
        std::vector<Half> result(N);

        auto d_a      = make_shared_device(a);
        auto d_b      = make_shared_device(b);
        auto d_result = make_shared_device<Half>(N);

        KernelArguments runtimeArgs;
        runtimeArgs.append("result", d_result.get());
        runtimeArgs.append("a", d_a.get());
        runtimeArgs.append("b", d_b.get());

        commandKernel.launchKernel(runtimeArgs.runtimeArguments());

        ASSERT_THAT(hipMemcpy(result.data(), d_result.get(), sizeof(Half) * N, hipMemcpyDefault),
                    HasHipSuccess(0));

        for(int i = 0; i < N; i++)
            ASSERT_NEAR(result[i], a[i] + a[i] * b[i], 0.001);
    }

    TEST_F(HalfPrecisionTest, GPU_ExecuteHalfPrecisionMultiplyAdd)
    {
        executeHalfPrecisionMultiplyAdd(m_context, 8, false);
    }

    TEST_F(HalfPrecisionTest, GPU_ExecuteHalfPrecisionMultiplyAddConvert)
    {
        executeHalfPrecisionMultiplyAdd(m_context, 8, true);
    }

    void genHalfPrecisionPack(std::shared_ptr<rocRoller::Context> m_context, int N)
    {
        AssertFatal(N % 2 == 0, "HalfPrecisionPack tests should only operate on even sizes");

        auto k = m_context->kernel();

        k->setKernelDimensions(1);
        auto command = std::make_shared<Command>();

        auto result_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Half, PointerType::PointerGlobal}));
        auto a_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Half, PointerType::PointerGlobal}));
        auto b_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Half, PointerType::PointerGlobal}));

        auto one  = std::make_shared<Expression::Expression>(1u);
        auto zero = std::make_shared<Expression::Expression>(0u);

        k->addArgument({"result",
                        {DataType::Half, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        result_exp});
        k->addArgument(
            {"a", {DataType::Half, PointerType::PointerGlobal}, DataDirection::ReadOnly, a_exp});
        k->addArgument(
            {"b", {DataType::Half, PointerType::PointerGlobal}, DataDirection::ReadOnly, b_exp});

        k->setWorkgroupSize({1, 1, 1});
        k->setWorkitemCount({one, one, one});
        k->setDynamicSharedMemBytes(zero);

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_result, s_a, s_b;
            co_yield m_context->argLoader()->getValue("result", s_result);
            co_yield m_context->argLoader()->getValue("a", s_a);
            co_yield m_context->argLoader()->getValue("b", s_b);

            auto v_result = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Raw32, 2);

            auto a_ptr = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Raw32, 2);

            auto b_ptr = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Raw32, 2);

            auto v_a = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Halfx2, 1);

            auto v_hi = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Half, 1);

            auto v_lo = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Half, 1);

            auto lds_offset = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int32, 1);

            auto mask = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int32, 1);

            auto lds = Register::Value::AllocateLDS(m_context, DataType::Half, N * 2);

            co_yield v_a->allocate();
            co_yield a_ptr->allocate();
            co_yield b_ptr->allocate();
            co_yield v_result->allocate();
            co_yield v_hi->allocate();
            co_yield v_lo->allocate();
            co_yield lds_offset->allocate();
            co_yield mask->allocate();

            co_yield m_context->copier()->copy(v_result, s_result, "Move pointer.");

            co_yield m_context->copier()->copy(a_ptr, s_a, "Move pointer.");
            co_yield m_context->copier()->copy(b_ptr, s_b, "Move pointer.");

            co_yield m_context->copier()->copy(
                lds_offset, Register::Value::Literal(lds->getLDSAllocation()->offset()));
            co_yield m_context->copier()->copy(
                mask, Register::Value::Literal(0xFFFF), "Create register with mask for lower bits");

            for(int i = 0; i < N; i++)
            {
                // Load and pack from flat into registers
                co_yield m_context->mem()->loadAndPack(MemoryInstructions::Flat,
                                                       v_a,
                                                       a_ptr,
                                                       Register::Value::Literal(i * 2),
                                                       b_ptr,
                                                       Register::Value::Literal(i * 2));

                // Perform addition. This will be a+a and b+b
                co_yield generateOp<Expression::Add>(v_a, v_a, v_a);

                // Store and Pack into LDS
                co_yield generateOp<Expression::BitwiseAnd>(v_lo, v_a, mask);
                co_yield generateOp<Expression::ShiftR>(v_hi, v_a, Register::Value::Literal(16));
                co_yield m_context->mem()->packAndStore(MemoryInstructions::Local,
                                                        lds_offset,
                                                        v_lo,
                                                        v_hi,
                                                        Register::Value::Literal(i * 4));
            }

            co_yield m_context->mem()->barrier();

            // Load all values of a+a from LDS, then store in flat
            for(int i = 0; i < N / 2; i++)
            {
                // Load and pack from LDS
                co_yield m_context->mem()->loadAndPack(MemoryInstructions::Local,
                                                       v_a,
                                                       lds_offset,
                                                       Register::Value::Literal(i * 8),
                                                       lds_offset,
                                                       Register::Value::Literal(i * 8 + 4));

                // Store and pack into flat
                co_yield generateOp<Expression::BitwiseAnd>(v_lo, v_a, mask);
                co_yield generateOp<Expression::ShiftR>(v_hi, v_a, Register::Value::Literal(16));

                co_yield m_context->mem()->packAndStore(MemoryInstructions::Flat,
                                                        v_result,
                                                        v_lo,
                                                        v_hi,
                                                        Register::Value::Literal(i * 4));
            }

            // Load all values of b+b from LDS, then store in flat
            for(int i = 0; i < N / 2; i++)
            {
                // Load and pack from LDS
                co_yield m_context->mem()->loadAndPack(MemoryInstructions::Local,
                                                       v_a,
                                                       lds_offset,
                                                       Register::Value::Literal(i * 8 + 2),
                                                       lds_offset,
                                                       Register::Value::Literal(i * 8 + 6));

                // Store and pack into flat
                co_yield generateOp<Expression::BitwiseAnd>(v_lo, v_a, mask);
                co_yield generateOp<Expression::ShiftR>(v_hi, v_a, Register::Value::Literal(16));

                co_yield m_context->mem()->packAndStore(MemoryInstructions::Flat,
                                                        v_result,
                                                        v_lo,
                                                        v_hi,
                                                        Register::Value::Literal((N / 2 + i) * 4));
            }
        };

        m_context->schedule(kb());
        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());
    }

    void executeHalfPrecisionPack(std::shared_ptr<rocRoller::Context> m_context, int N)
    {
        genHalfPrecisionPack(m_context, N);

        CommandKernel     commandKernel(m_context);
        RandomGenerator   random(316473u);
        auto              a = random.vector<Half>(N, -1.0, 1.0);
        auto              b = random.vector<Half>(N, -1.0, 1.0);
        std::vector<Half> result(N * 2);

        auto d_a      = make_shared_device(a);
        auto d_b      = make_shared_device(b);
        auto d_result = make_shared_device<Half>(N * 2);

        KernelArguments runtimeArgs;
        runtimeArgs.append("result", d_result.get());
        runtimeArgs.append("a", d_a.get());
        runtimeArgs.append("b", d_b.get());

        commandKernel.launchKernel(runtimeArgs.runtimeArguments());

        ASSERT_THAT(
            hipMemcpy(result.data(), d_result.get(), sizeof(Half) * N * 2, hipMemcpyDefault),
            HasHipSuccess(0));

        for(int i = 0; i < N; i++)
            EXPECT_NEAR(result[i], a[i] + a[i], 0.001);

        for(int i = 0; i < N; i++)
            EXPECT_NEAR(result[i + N], b[i] + b[i], 0.001);
    }

    TEST_F(HalfPrecisionTest, GPU_ExecuteHalfPrecisionPack)
    {
        executeHalfPrecisionPack(m_context, 8);
    }

    TEST_F(HalfPrecisionTest, HalfPrecisionAsserts)
    {

        auto vf32
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Float, 1);

        auto vh16_1
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Half, 1);
        auto vh16_2
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Half, 1);
        auto vh16x2
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Halfx2, 1);

        auto addr
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Int64, 1);

        auto buff_desc = std::make_shared<BufferDescriptor>(m_context);
        auto buff_opts = BufferInstructionOptions();

        // copy
        EXPECT_THROW(m_context->schedule(m_context->copier()->pack(vf32, vh16_1, vh16_2)),
                     FatalError);

        EXPECT_THROW(m_context->schedule(m_context->copier()->pack(vh16x2, vf32, vf32)),
                     FatalError);

        // memory instructions
        EXPECT_THROW(m_context->schedule(m_context->mem()->loadFlat(vh16x2, addr, 0, 4, true)),
                     FatalError);

        EXPECT_THROW(m_context->schedule(m_context->mem()->loadLocal(vh16x2, addr, 0, 4, "", true)),
                     FatalError);

        EXPECT_THROW(m_context->schedule(m_context->mem()->loadAndPack(
                         MemoryInstructions::MemoryKind::Flat, vf32, addr, addr, addr, addr, "")),
                     FatalError);

        EXPECT_THROW(m_context->schedule(
                         m_context->mem()->loadAndPack(MemoryInstructions::MemoryKind::Buffer,
                                                       vf32,
                                                       addr,
                                                       addr,
                                                       addr,
                                                       addr,
                                                       "",
                                                       buff_desc)),
                     FatalError);
    }

}
