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
    class FP4MemoryInstructionTest : public GPUContextFixture
    {
    };

    const size_t numValuesPerByte = 2;
    const size_t numFP4PerElement = 8;

    // /**
    //  * buffer_load that into FP4x8 to GPU, buffer_store to CPU
    // */
    void genFP4x8BufferLoadAndStore(rocRoller::ContextPtr m_context, int num_fp4)
    {
        AssertFatal(num_fp4 % numFP4PerElement == 0,
                    "Number of FP4 values must be multiple times of 8");

        int N = num_fp4 / numValuesPerByte;

        auto k = m_context->kernel();
        k->setKernelName("BufferLoadAndStoreFP4x8");
        k->setKernelDimensions(1);

        k->addArgument(
            {"result", {DataType::Int32, PointerType::PointerGlobal}, DataDirection::WriteOnly});
        k->addArgument(
            {"a", {DataType::Int32, PointerType::PointerGlobal}, DataDirection::ReadOnly});

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_result, s_a;
            co_yield m_context->argLoader()->getValue("result", s_result);
            co_yield m_context->argLoader()->getValue("a", s_a);

            auto vgprSerial = m_context->kernel()->workitemIndex()[0];

            size_t size = N / 4;
            auto   v_a  = Register::Value::Placeholder(m_context,
                                                    Register::Type::Vector,
                                                    DataType::FP4x8,
                                                    size,
                                                    Register::AllocationOptions::FullyContiguous());

            co_yield v_a->allocate();

            auto bufDesc = std::make_shared<rocRoller::BufferDescriptor>(m_context);
            co_yield bufDesc->setup();
            co_yield bufDesc->setBasePointer(s_a);
            co_yield bufDesc->setSize(Register::Value::Literal(N));
            co_yield bufDesc->setOptions(Register::Value::Literal(131072)); //0x00020000

            auto bufInstOpts = rocRoller::BufferInstructionOptions();

            co_yield m_context->mem()->loadBuffer(v_a, vgprSerial, 0, bufDesc, bufInstOpts, N);
            co_yield bufDesc->setBasePointer(s_result);
            co_yield m_context->mem()->storeBuffer(v_a, vgprSerial, 0, bufDesc, bufInstOpts, N);
        };

        m_context->schedule(kb());
        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());
    }

    // /**
    //  * flat_load that into FP4x8 to GPU, flat_store to CPU
    // */
    void genFP4x8FlatLoadAndStore(rocRoller::ContextPtr m_context, int num_fp4)
    {
        AssertFatal(num_fp4 % numFP4PerElement == 0,
                    "Number of FP4 values must be multiple times of 8");
        int  N = num_fp4 / numValuesPerByte;
        auto k = m_context->kernel();

        k->setKernelName("FlatLoadAndStoreFP4x8");
        k->setKernelDimensions(1);

        k->addArgument(
            {"result", {DataType::UInt32, PointerType::PointerGlobal}, DataDirection::WriteOnly});
        k->addArgument(
            {"a", {DataType::UInt32, PointerType::PointerGlobal}, DataDirection::ReadOnly});

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_result, s_a;
            co_yield m_context->argLoader()->getValue("result", s_result);
            co_yield m_context->argLoader()->getValue("a", s_a);

            auto v_result
                = Register::Value::Placeholder(m_context,
                                               Register::Type::Vector,
                                               {DataType::UInt32, PointerType::PointerGlobal},
                                               1);

            auto v_ptr
                = Register::Value::Placeholder(m_context,
                                               Register::Type::Vector,
                                               {DataType::UInt32, PointerType::PointerGlobal},
                                               1);

            int  size = N / 4;
            auto v_a  = Register::Value::Placeholder(m_context,
                                                    Register::Type::Vector,
                                                    DataType::FP4x8,
                                                    size,
                                                    Register::AllocationOptions::FullyContiguous());

            co_yield v_a->allocate();

            co_yield v_ptr->allocate();
            co_yield v_result->allocate();

            co_yield m_context->copier()->copy(v_result, s_result, "Move pointer.");

            co_yield m_context->copier()->copy(v_ptr, s_a, "Move pointer.");

            co_yield m_context->mem()->loadFlat(v_a, v_ptr, 0, N);
            co_yield m_context->mem()->storeFlat(v_result, v_a, 0, N);
        };

        m_context->schedule(kb());
        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());
    }

    /**
     *
     */
    void executeFP4x8LoadAndStore(rocRoller::ContextPtr m_context, int num_fp4)
    {
        AssertFatal(num_fp4 % numFP4PerElement == 0,
                    "Number of FP4 values must be multiple times of 16");

        auto rng = RandomGenerator(316473u);
        auto a   = rng.vector<uint>(num_fp4 / numFP4PerElement,
                                  std::numeric_limits<uint>::min(),
                                  std::numeric_limits<uint>::max());

        std::vector<uint32_t> result(a.size());

        std::shared_ptr<rocRoller::ExecutableKernel> executableKernel
            = m_context->instructions()->getExecutableKernel();

        auto d_a      = make_shared_device(a);
        auto d_result = make_shared_device<uint32_t>(result.size());

        KernelArguments runtimeArgs;
        runtimeArgs.append("result", d_result.get());
        runtimeArgs.append("a", d_a.get());
        KernelInvocation invocation;
        executableKernel->executeKernel(runtimeArgs, invocation);

        ASSERT_THAT(
            hipMemcpy(
                result.data(), d_result.get(), sizeof(uint32_t) * result.size(), hipMemcpyDefault),
            HasHipSuccess(0));

        for(int i = 0; i < a.size(); i++)
        {
            EXPECT_EQ(result[i], a[i]);
        }
    }

    TEST_P(FP4MemoryInstructionTest, GPU_FP4x8BufferLoadAndStore)
    {
        int num_fp4 = 8;
        genFP4x8BufferLoadAndStore(m_context, num_fp4);
        std::vector<char> assembledKernel = m_context->instructions()->assemble();
        EXPECT_GT(assembledKernel.size(), 0);

        if(isLocalDevice())
        {
            executeFP4x8LoadAndStore(m_context, num_fp4);
        }
    }

    TEST_P(FP4MemoryInstructionTest, GPU_FP4x8FlatLoadAndStore)
    {

        int num_fp4 = 8;
        genFP4x8FlatLoadAndStore(m_context, num_fp4);
        std::vector<char> assembledKernel = m_context->instructions()->assemble();
        EXPECT_GT(assembledKernel.size(), 0);

        if(isLocalDevice())
        {
            executeFP4x8LoadAndStore(m_context, num_fp4);
        }
    }

    TEST(FP4ConversionTest, CPUConversions)
    {
        auto singleTest = [](auto fp64) {
            // FP4 to FP32
            rocRoller::FP4 fp4(fp64);
            float          fp32(fp4);
            EXPECT_FLOAT_EQ(fp32, fp64);

            // FP32 to FP4
            fp4 = rocRoller::FP4(fp32);
            EXPECT_FLOAT_EQ((double)fp4, fp64);
        };

        constexpr auto cases = std::to_array<double>(
            {0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6});

        for(auto const& c : cases)
        {
            singleTest(c);
        }
    }

    INSTANTIATE_TEST_SUITE_P(
        FP4MemoryInstructionTest,
        FP4MemoryInstructionTest,
        ::testing::Combine(::testing::Values("gfx90a:sramecc+, gfx942:sramecc+")));
}
