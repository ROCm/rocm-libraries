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

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#endif /* ROCROLLER_USE_HIP */

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/BranchGenerator.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/ExecutableKernel.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureLibrary.hpp>
#include <rocRoller/InstructionValues/LabelAllocator.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Utilities/Generator.hpp>

#include "GPUContextFixture.hpp"
#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"
#include "Utilities.hpp"

using namespace rocRoller;

namespace rocRollerTest
{
    class KernelTest : public GenericContextFixture
    {
        void SetUp() override
        {
            GenericContextFixture::SetUp();
            Settings::getInstance()->set(Settings::SerializeKernelGraph, true);
        }
    };

    class ARCH_KernelTest : public GPUContextFixture
    {
    };

    TEST_F(KernelTest, Preamble)
    {
        AssemblyKernel k(m_context, "hello_world");

        m_context->schedule(k.preamble());

        std::string expected = R"(
            .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
            .set .amdgcn.next_free_vgpr, 0
            .set .amdgcn.next_free_sgpr, 0
            .text
            .globl hello_world
            .p2align 8
            .type hello_world,@function
            hello_world:
        )";

        EXPECT_THAT(output(), MatchesSource(expected));
        EXPECT_EQ(NormalizedSource(expected), NormalizedSource(output()));
    }

    TEST_F(KernelTest, AddArgument)
    {
        auto kernel = m_context->kernel();

        kernel->addArgument({"foo", {DataType::Float}});
        EXPECT_EQ(0, kernel->arguments()[0].offset);
        EXPECT_EQ(4, kernel->arguments()[0].size);

        kernel->addArgument(
            {"bar", {DataType::Float, PointerType::PointerGlobal}, DataDirection::ReadOnly});
        EXPECT_EQ(8, kernel->arguments()[1].offset);
        EXPECT_EQ(8, kernel->arguments()[1].size);

        kernel->addArgument({"baz", {DataType::Double}});
        EXPECT_EQ(16, kernel->arguments()[2].offset);
        EXPECT_EQ(8, kernel->arguments()[2].size);
    }

    TEST_F(KernelTest, Postamble)
    {
        AssemblyKernel k(m_context, "hello_world");

        m_context->schedule(k.postamble());

        std::string expected = R"(
            s_endpgm
            .Lhello_world_end:
            .size hello_world, .Lhello_world_end-hello_world
            .rodata
            .p2align 6
            .amdhsa_kernel hello_world
            .amdhsa_next_free_vgpr 0
            .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
            .amdhsa_user_sgpr_kernarg_segment_ptr 1
            .amdhsa_system_sgpr_workgroup_id_x 1
            .amdhsa_system_sgpr_workgroup_id_y 1
            .amdhsa_system_sgpr_workgroup_id_z 1
            .amdhsa_system_sgpr_workgroup_info 0
            .amdhsa_system_vgpr_workitem_id 2
            .end_amdhsa_kernel
        )";

        EXPECT_EQ(NormalizedSource(output()), NormalizedSource(expected));
    }

    class MaxRegisterKernelTest : public KernelTest
    {
    protected:
        GPUArchitectureTarget targetArchitecture() override
        {
            return {GPUArchitectureGFX::GFX90A};
        }

        void SetUp() override
        {
            m_kernelOptions->maxVGPRs             = 234;
            m_kernelOptions->setNextFreeVGPRToMax = true;

            GenericContextFixture::SetUp();
        }
    };

    TEST_F(MaxRegisterKernelTest, MaxVGPRsInPostamble)
    {
        AssemblyKernel k(m_context, "hello_world");

        m_context->schedule(k.postamble());

        EXPECT_EQ(countSubstring(output(), ".amdhsa_next_free_vgpr 234"), 1);
    }

    TEST_F(KernelTest, Metadata)
    {
#ifdef ROCROLLER_TESTS_USE_YAML_CPP

        AssemblyKernel k(m_context, "hello_world");

        m_context->schedule(k.amdgpu_metadata());

        std::string expected = R"(
.amdgpu_metadata
---
amdhsa.version:
 - 1
 - 2
amdhsa.kernels:
  - .name: hello_world
    .symbol: hello_world.kd
    .kernarg_segment_size: 0
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 0
    .vgpr_count: 0
    .agpr_count: 0
    .max_flat_workgroup_size: 1
    .kernel_dimensions: 3
    .workgroup_size: [1, 1, 1]
    .wavefront_size: 64
    .workitem_count: [{is-null: true}, {is-null: true}, {is-null: true}]
    .dynamic_sharedmemory_bytes:
      is-null: true
    .args:
      []
    .kernel_graph:
      is-null: true
    .command:
      is-null: true
...
.end_amdgpu_metadata
        )";

        EXPECT_EQ(NormalizedSource(output()), NormalizedSource(expected));
#else
        GTEST_SKIP() << "Skipping NormalizedYAML test (Only implemented for yaml-cpp)";
#endif
    }

    TEST_F(KernelTest, ArgumentsAndRegisters)
    {
#ifdef ROCROLLER_TESTS_USE_YAML_CPP
        AssemblyKernel k(m_context, "hello_world");
        k.addArgument({"foo", {DataType::Float}});
        k.setWorkgroupSize({16, 8, 2});

        auto alloc0 = std::make_shared<Register::Allocation>(
            m_context, Register::Type::Scalar, DataType::Float);
        m_context->allocator(Register::Type::Scalar)->allocate(alloc0);

        auto alloc1 = std::make_shared<Register::Allocation>(
            m_context, Register::Type::Vector, DataType::Double);
        m_context->allocator(Register::Type::Vector)->allocate(alloc1);

        m_context->schedule(k.amdgpu_metadata());

        std::string expected = R"(
.amdgpu_metadata
---
amdhsa.version:
    - 1
    - 2
amdhsa.kernels:
  - .name: hello_world
    .symbol: hello_world.kd
    .kernarg_segment_size: 4
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 1
    .vgpr_count: 2
    .agpr_count: 0
    .max_flat_workgroup_size: 256
    .kernel_dimensions: 3
    .workgroup_size: [16, 8, 2]
    .wavefront_size: 64
    .workitem_count: [{is-null: true}, {is-null: true}, {is-null: true}]
    .dynamic_sharedmemory_bytes:
      is-null: true
    .args:
      - .name: foo
        .size: 4
        .offset: 0
        .expression:
          is-null: true
        .variableType: {dataType: Float, pointerType: Value}
        .value_kind: by_value
    .kernel_graph:
      is-null: true
    .command:
      is-null: true
...
.end_amdgpu_metadata)";

        EXPECT_EQ(NormalizedSource(output()), NormalizedSource(expected));
#else
        GTEST_SKIP() << "Skipping NormalizedYAML test (Only implemented for yaml-cpp)";
#endif
    }

    TEST_P(ARCH_KernelTest, GPU_WholeKernel)
    {
        auto k = m_context->kernel();

        k->setKernelName("lds_bank_conflict");
        k->setKernelDimensions(1);

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            const auto count = 64;

            auto dst = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int32, count);

            auto lds = Register::Value::AllocateLDS(m_context, DataType::Int32, count * 4 * 32);
            auto ldsOffset = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int32, 1);
            co_yield m_context->copier()->copy(
                ldsOffset, Register::Value::Literal(lds->getLDSAllocation()->offset()));
            auto workitemIndex = m_context->kernel()->workitemIndex()[0];
            co_yield Expression::generate(
                ldsOffset,
                ldsOffset->expression() + workitemIndex->expression() * Expression::literal(4 * 32),
                m_context);
            for(int i = 0; i < count; ++i)
            {
                co_yield m_context->mem()->loadLocal(dst->element({i}), ldsOffset, 0, 4);
            }
        };

        m_context->schedule(kb());

        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        // Only execute the kernels if running on the architecture that the kernel was built for,
        // otherwise just assemble the instructions.
        if(isLocalDevice())
        {
            std::shared_ptr<rocRoller::ExecutableKernel> executableKernel
                = m_context->instructions()->getExecutableKernel();

            KernelArguments  kargs;
            KernelInvocation invocation{{1, 1, 1}, {64, 1, 1}, 0};
            executableKernel->executeKernel(kargs, invocation);
        }
        else
        {
            AssertFatal(false);
            std::vector<char> assembledKernel = m_context->instructions()->assemble();
            EXPECT_GT(assembledKernel.size(), 0);
        }

        // kernel.s can be assembled with:
        // /opt/rocm/bin/amdclang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx90a:xnack+ build/kernel.s -o build/kernel.o
    }

    INSTANTIATE_TEST_SUITE_P(ARCH_KernelTests, ARCH_KernelTest, supportedISATuples());

    class GPU_KernelTest : public CurrentGPUContextFixture,
                           public ::testing::WithParamInterface<AssemblerType>
    {
    };

    TEST_P(GPU_KernelTest, GPU_WholeKernel)
    {
        //FIXME fix the arch check with InProcess assembler for gfx94X and gfx95X
        if(GetParam() == AssemblerType::InProcess
           && m_context->targetArchitecture().HasCapability(GPUCapability::HasMFMA_fp8))
        {
            GTEST_SKIP() << "Skipping InProcess Assembler on "
                         << m_context->targetArchitecture().target() << std::endl;
        }

        Settings::getInstance()->set(Settings::KernelAssembler, GetParam());

        ASSERT_EQ(true, isLocalDevice());

        auto command = std::make_shared<Command>();

        auto k = m_context->kernel();

        k->setKernelDimensions(1);

        const auto one  = std::make_shared<Expression::Expression>(1u);
        const auto zero = std::make_shared<Expression::Expression>(0u);

        const auto workgroupSize = 256u;
        auto       workitemCount
            = Expression::literal(1024 * workgroupSize); // Have high workgroup count for rocprofv3
        k->setWorkgroupSize({workgroupSize, 1, 1});
        k->setWorkitemCount({workitemCount, one, one});
        k->setDynamicSharedMemBytes(zero);

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto strideMultiplier = 1;
        if(const char* env_p = std::getenv("BYTE_STRIDE"))
            strideMultiplier = atoi(env_p); // adjust for bank conflict

        auto instrDwords = 1;
        if(const char* env_p = std::getenv("INSTR_WIDTH"))
            instrDwords = atoi(env_p); // e.g. 1 for ds_read_b32

        auto kb = [&]() -> Generator<Instruction> {
            const auto loops
                = 0; // Loop due to suspecting instruction cache causing latency, doesn't seem to change anything from testing
            const auto dstRegCount = 192;
            const auto count
                = dstRegCount / instrDwords; // number of instructions put one after another

            auto dst = Register::Value::Placeholder(
                m_context,
                Register::Type::Vector,
                DataType::Raw32,
                count * instrDwords,
                Register::AllocationOptions{
                    .contiguousChunkWidth = Register::FULLY_CONTIGUOUS,
                    .alignment            = instrDwords,
                });

            auto lds = Register::Value::AllocateLDS(
                m_context, DataType::Raw32, workgroupSize * strideMultiplier * instrDwords);
            auto ldsOffset = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int32, 1);
            auto workitemIndex = m_context->kernel()->workitemIndex()[0];
            co_yield Expression::generate(
                ldsOffset,
                Expression::literal(lds->getLDSAllocation()->offset())
                    + workitemIndex->expression()
                          * Expression::literal(4 * strideMultiplier * instrDwords),
                m_context);

            auto i = Register::Value::Placeholder(
                m_context, Register::Type::Scalar, DataType::UInt64, 1);
            co_yield Expression::generate(i, Expression::literal(loops + 1), m_context);

            auto label = m_context->labelAllocator()->label("label");
            co_yield Instruction::Label(label);

            for(int i = 0; i < count * instrDwords; i += instrDwords)
            {
                co_yield m_context->mem()->loadLocal(
                    dst->subset(Generated(iota(i, i + instrDwords))),
                    ldsOffset,
                    0,
                    4 * instrDwords);
            }

            co_yield Instruction::Wait(WaitCount::Zero(m_context->targetArchitecture()));
            co_yield Expression::generate(i, i->expression() - Expression::literal(1), m_context);
            co_yield m_context->brancher()->branchIfNonZero(label, i);
        };

        m_context->schedule(kb());

        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        CommandKernel commandKernel;
        commandKernel.setContext(m_context);
        commandKernel.generateKernel();

        CommandArguments commandArgs = command->createArguments();

        commandKernel.launchKernel(commandArgs.runtimeArguments());
    }

    INSTANTIATE_TEST_SUITE_P(GPU_KernelTests,
                             GPU_KernelTest,
                             ::testing::Values(AssemblerType::InProcess,
                                               AssemblerType::Subprocess));
}
