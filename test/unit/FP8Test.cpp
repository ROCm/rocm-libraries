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
    struct FP8Problem
    {
        std::shared_ptr<Command>            command;
        rocRoller::Operations::OperationTag resultTag, aTag;
    };

    class FP8Test : public GPUContextFixture
    {
    };

    const size_t numFP8PerElement = 4;

    /**
     * Loads FP8x4 to GPU, unpacks to individual FP8s, convert to float, store to CPU
    */
    void genFP8x4LoadToFloatStore(rocRoller::ContextPtr context, FP8Problem& prob, int N)
    {
        auto k = context->kernel();

        k->setKernelDimensions(1);
        prob.command = std::make_shared<Command>();

        prob.resultTag  = prob.command->allocateTag();
        auto result_exp = std::make_shared<Expression::Expression>(prob.command->allocateArgument(
            {DataType::Float, PointerType::PointerGlobal}, prob.resultTag, ArgumentType::Value));
        prob.aTag       = prob.command->allocateTag();
        auto a_exp      = std::make_shared<Expression::Expression>(prob.command->allocateArgument(
            {DataType::FP8_NANOO, PointerType::PointerGlobal}, prob.aTag, ArgumentType::Value));

        auto one  = std::make_shared<Expression::Expression>(1u);
        auto zero = std::make_shared<Expression::Expression>(0u);

        k->addArgument({"result",
                        {DataType::Float, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        result_exp});
        k->addArgument({"a",
                        {DataType::FP8_NANOO, PointerType::PointerGlobal},
                        DataDirection::ReadOnly,
                        a_exp});

        k->setWorkgroupSize({1, 1, 1});
        k->setWorkitemCount({one, one, one});
        k->setDynamicSharedMemBytes(zero);

        context->schedule(k->preamble());
        context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_result, s_a;
            co_yield context->argLoader()->getValue("result", s_result);
            co_yield context->argLoader()->getValue("a", s_a);

            auto result_ptr
                = Register::Value::Placeholder(context,
                                               Register::Type::Vector,
                                               {DataType::Float, PointerType::PointerGlobal},
                                               1,
                                               Register::AllocationOptions::FullyContiguous());

            auto a_ptr
                = Register::Value::Placeholder(context,
                                               Register::Type::Vector,
                                               {DataType::UInt32, PointerType::PointerGlobal},
                                               1,
                                               Register::AllocationOptions::FullyContiguous());
            auto v_a = Register::Value::Placeholder(
                context, Register::Type::Vector, DataType::UInt32, 1);

            auto v_temp = Register::Value::Placeholder(
                context, Register::Type::Vector, DataType::FP8_NANOO, 1);

            co_yield v_a->allocate();
            co_yield a_ptr->allocate();
            co_yield result_ptr->allocate();

            co_yield context->copier()->copy(result_ptr, s_result, "Move pointer.");
            co_yield context->copier()->copy(a_ptr, s_a, "Move pointer.");

            auto bpi = DataTypeInfo::Get(a_ptr->variableType().dataType).elementSize;
            auto bpo = DataTypeInfo::Get(result_ptr->variableType().dataType).elementSize;

            for(int i = 0; i < N; i++)
            {
                co_yield context->mem()->loadFlat(v_a, a_ptr, i * bpi, bpi);

                // Bitmask each FP8 of FP8x4, convert to float, then store
                for(int fp8_idx = 0; fp8_idx < numFP8PerElement; fp8_idx++)
                {
                    co_yield context->copier()->copy(v_temp, v_a, "Move to temp");

                    co_yield_(Instruction::Comment("Mask for lower 8 bits"));
                    co_yield generateOp<Expression::BitwiseAnd>(
                        v_temp, v_temp, Register::Value::Literal(0xFF));

                    co_yield_(Instruction::Comment("Convert to float"));
                    co_yield generateOp<Expression::Convert<DataType::Float>>(v_temp, v_temp);

                    co_yield context->mem()->storeFlat(result_ptr,
                                                       v_temp,
                                                       (i * numFP8PerElement + fp8_idx) * bpo,
                                                       bpo,
                                                       "Store to result");

                    co_yield_(Instruction::Comment(
                        "Shift right so lower 8 bits is the fp8 to manipulate"));
                    co_yield generateOp<Expression::LogicalShiftR>(
                        v_a, v_a, Register::Value::Literal(8));
                }
            }
        };

        context->schedule(kb());
        context->schedule(k->postamble());
        context->schedule(k->amdgpu_metadata());
    }

    /**
     * @param N number of FP8x4; so Nx4 float results
     */
    void executeFP8x4LoadToFloatStore(rocRoller::ContextPtr context)
    {
        int N = 256;

        auto rng = RandomGenerator(316473u);
        auto a   = rng.vector<uint>(
            N, std::numeric_limits<uint>::min(), std::numeric_limits<uint>::max());

        std::vector<float> result(a.size() * numFP8PerElement);

        FP8Problem prob;
        genFP8x4LoadToFloatStore(context, prob, a.size());
        CommandKernel commandKernel(context);

        auto d_a      = make_shared_device(a);
        auto d_result = make_shared_device<float>(result.size());

        CommandArguments commandArgs = prob.command->createArguments();

        commandArgs.setArgument(prob.resultTag, ArgumentType::Value, d_result.get());
        commandArgs.setArgument(prob.aTag, ArgumentType::Value, d_a.get());

        commandKernel.launchKernel(commandArgs.runtimeArguments());

        ASSERT_THAT(
            hipMemcpy(
                result.data(), d_result.get(), sizeof(float) * result.size(), hipMemcpyDefault),
            HasHipSuccess(0));

        for(int i = 0; i < a.size(); i++)
        {
            union
            {
                uint32_t word;
                uint8_t  bytes[4];
            } u;
            u.word = a[i];

            for(int fp8_idx = 0; fp8_idx < numFP8PerElement; fp8_idx++)
            {
                FP8_NANOO expected_fp8;
                expected_fp8.data = u.bytes[fp8_idx];

                float actual   = result.at(i * numFP8PerElement + fp8_idx);
                float expected = expected_fp8.operator float();

                if(std::isnan(expected))
                    EXPECT_TRUE(std::isnan(actual));
                else
                    EXPECT_EQ(actual, expected);
            }
        }
    }

    TEST_P(FP8Test, GPU_FP8x4LoadToFloatStore)
    {
        if(isLocalDevice())
        {
            executeFP8x4LoadToFloatStore(m_context);
        }
        else
        {
            FP8Problem prob;
            genFP8x4LoadToFloatStore(m_context, prob, 2);
            std::vector<char> assembledKernel = m_context->instructions()->assemble();
            EXPECT_GT(assembledKernel.size(), 0);
        }
    }

    /**
     * Loads sparse FP8s to GPU, packs into FP8x4s, stores to CPU
    */
    void genFP8LoadGather(rocRoller::ContextPtr m_context, int N)
    {
        auto k = m_context->kernel();

        k->setKernelDimensions(1);
        auto command = std::make_shared<Command>();

        auto resultTag  = command->allocateTag();
        auto result_exp = std::make_shared<Expression::Expression>(command->allocateArgument(
            {DataType::UInt32, PointerType::PointerGlobal}, resultTag, ArgumentType::Value));
        auto aTag       = command->allocateTag();
        auto a_exp      = std::make_shared<Expression::Expression>(command->allocateArgument(
            {DataType::FP8_NANOO, PointerType::PointerGlobal}, aTag, ArgumentType::Value));

        auto one  = std::make_shared<Expression::Expression>(1u);
        auto zero = std::make_shared<Expression::Expression>(0u);

        k->addArgument({"result",
                        {DataType::UInt32, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        result_exp});
        k->addArgument({"a",
                        {DataType::FP8_NANOO, PointerType::PointerGlobal},
                        DataDirection::ReadOnly,
                        a_exp});

        k->setWorkgroupSize({1, 1, 1});
        k->setWorkitemCount({one, one, one});
        k->setDynamicSharedMemBytes(zero);

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_result, s_a;
            co_yield m_context->argLoader()->getValue("result", s_result);
            co_yield m_context->argLoader()->getValue("a", s_a);

            auto result_ptr
                = Register::Value::Placeholder(m_context,
                                               Register::Type::Vector,
                                               {DataType::UInt32, PointerType::PointerGlobal},
                                               1,
                                               Register::AllocationOptions::FullyContiguous());

            auto a_ptr
                = Register::Value::Placeholder(m_context,
                                               Register::Type::Vector,
                                               {DataType::FP8_NANOO, PointerType::PointerGlobal},
                                               1,
                                               Register::AllocationOptions::FullyContiguous());
            auto v_a = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::FP8_NANOO, 1);

            auto v_temp
                = Register::Value::Placeholder(m_context,
                                               Register::Type::Vector,
                                               DataType::FP8_NANOO,
                                               4,
                                               Register::AllocationOptions::FullyContiguous());

            co_yield v_a->allocate();
            co_yield a_ptr->allocate();
            co_yield result_ptr->allocate();
            co_yield v_temp->allocate();

            co_yield m_context->copier()->copy(result_ptr, s_result, "Move pointer.");
            co_yield m_context->copier()->copy(a_ptr, s_a, "Move pointer.");

            auto bpi = DataTypeInfo::Get(a_ptr->variableType().dataType).elementSize;
            auto bpo = DataTypeInfo::Get(result_ptr->variableType().dataType).elementSize;

            auto bufDesc = std::make_shared<rocRoller::BufferDescriptor>(m_context);
            co_yield bufDesc->setup();
            co_yield bufDesc->setSize(Register::Value::Literal(N));
            co_yield bufDesc->setOptions(Register::Value::Literal(131072)); //0x00020000

            auto bufInstOpts = rocRoller::BufferInstructionOptions();

            auto vgprSerial = m_context->kernel()->workitemIndex()[0];

            co_yield bufDesc->setBasePointer(s_a);
            for(int i = 0; i < N; ++i)
            {
                co_yield m_context->mem()->loadBuffer(
                    v_temp->element({i}), vgprSerial, i, bufDesc, bufInstOpts, 1);
            }
            co_yield bufDesc->setBasePointer(s_result);
            co_yield m_context->mem()->storeBuffer(v_temp, vgprSerial, 0, bufDesc, bufInstOpts, N);
        };

        m_context->schedule(kb());
        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());
    }

    /**
     * @param N number of FP8x4; so Nx4 float results
     */
    void executeFP8LoadGather(rocRoller::ContextPtr context, int N)
    {
        genFP8LoadGather(context, N);
        std::shared_ptr<rocRoller::ExecutableKernel> executableKernel
            = context->instructions()->getExecutableKernel();

        std::vector<uint8_t> a(N);
        for(int i = 0; i < N; i++)
            a[i] = i + 10;

        auto d_a      = make_shared_device(a);
        auto d_result = make_shared_device<uint32_t>(N / 4);

        KernelArguments kargs;
        kargs.append<void*>("result", d_result.get());
        kargs.append<void*>("a", d_a.get());

        KernelInvocation invocation;

        executableKernel->executeKernel(kargs, invocation);

        std::vector<uint32_t> result(N / 4);
        ASSERT_THAT(
            hipMemcpy(result.data(), d_result.get(), sizeof(uint32_t) * N / 4, hipMemcpyDefault),
            HasHipSuccess(0));

        auto bpi = 1;
        auto bpo = 4;
        for(int i = 0; i < N / 4; i++)
        {
            uint32_t expected = a[i] | (a[i + 1] << 8) | (a[i + 2] << 16) | (a[i + 3] << 24);
            EXPECT_EQ(result[i], expected) << std::hex << result[i] << " " << expected;
        }
    }

    TEST_P(FP8Test, GPU_FP8LoadGather)
    {
        constexpr int N = 4;
        if(isLocalDevice())
        {
            executeFP8LoadGather(m_context, N);
        }
        else
        {
            genFP8LoadGather(m_context, N);
            std::vector<char> assembledKernel = m_context->instructions()->assemble();
            EXPECT_GT(assembledKernel.size(), 0);
        }
    }

    TEST(FP8Test, CPUConversions)
    {
        auto singleTest = [](auto fp64) {
            // FP8 to FP32
            rocRoller::FP8_NANOO fp8(fp64);
            float                fp32(fp8);
            EXPECT_FLOAT_EQ(fp32, fp64);

            // FP32 to FP8
            fp8 = rocRoller::FP8_NANOO(fp32);
            EXPECT_FLOAT_EQ((double)fp8, fp64);
        };

        constexpr auto cases = std::to_array<double>({-1.76e2, 1 / 1024, 240});

        for(auto const& c : cases)
        {
            singleTest(c);
        }
    }

    INSTANTIATE_TEST_SUITE_P(FP8Test,
                             FP8Test,
                             ::testing::Combine(::testing::Values("gfx942:sramecc+")));
}
