
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/BranchGenerator.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/ExecutableKernel.hpp>
#include <rocRoller/InstructionValues/LabelAllocator.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Scheduling/Scheduler.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Generator.hpp>

#include "GPUContextFixture.hpp"
#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"

using namespace rocRoller;

namespace rocRollerTest
{
    class BranchGeneratorTest : public GenericContextFixture
    {
    };

    class ARCH_BranchGeneratorTest : public GPUContextFixture
    {
    };

    /**
      * This test is making sure that the code produced by the branch generator compiles.
      **/
    TEST_F(BranchGeneratorTest, Basic)
    {
        auto brancher = BranchGenerator(m_context);

        auto l0 = Register::Value::Label("l0");
        auto s0
            = Register::Value::Placeholder(m_context, Register::Type::Scalar, DataType::UInt32, 1);
        s0->allocateNow();

        EXPECT_THROW(m_context->schedule(brancher.branch(s0)), FatalError);
        EXPECT_THROW(m_context->schedule(brancher.branchConditional(s0, s0, true)), FatalError);
    }

    TEST_P(ARCH_BranchGeneratorTest, Basic)
    {
        auto k = m_context->kernel();

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            auto l0 = m_context->labelAllocator()->label("l0");
            co_yield_(Instruction::Label(l0));

            auto wavefront_size = k->wavefront_size();

            auto scc = m_context->getSCC();
            auto vcc = m_context->getVCC();

            auto s0 = Register::Value::Placeholder(
                m_context, Register::Type::Scalar, DataType::UInt32, wavefront_size / 32);
            co_yield s0->allocate();

            co_yield m_context->brancher()->branch(l0);

            co_yield m_context->brancher()->branchConditional(l0, s0, true);
            co_yield m_context->brancher()->branchConditional(l0, s0, false);
            co_yield m_context->brancher()->branchIfZero(l0, s0);
            co_yield m_context->brancher()->branchIfNonZero(l0, s0);

            co_yield m_context->brancher()->branchConditional(l0, scc, true);
            co_yield m_context->brancher()->branchConditional(l0, scc, false);
            co_yield m_context->brancher()->branchIfZero(l0, scc);
            co_yield m_context->brancher()->branchIfNonZero(l0, scc);

            co_yield m_context->brancher()->branchConditional(l0, vcc, true);
            co_yield m_context->brancher()->branchConditional(l0, vcc, false);
            co_yield m_context->brancher()->branchIfZero(l0, vcc);
            co_yield m_context->brancher()->branchIfNonZero(l0, vcc);
        };

        m_context->schedule(kb());
        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        if(m_context->targetArchitecture().target().getMajorVersion() != 9)
            GTEST_SKIP() << "Skipping BranchGenerator tests for " << GetParam();

        std::vector<char> assembledKernel = m_context->instructions()->assemble();
        EXPECT_GT(assembledKernel.size(), 0);
    }

    TEST_P(ARCH_BranchGeneratorTest, Wait)
    {
        auto k = m_context->kernel();

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            auto l0 = m_context->labelAllocator()->label("l0");
            co_yield_(Instruction::Label(l0));
            co_yield m_context->brancher()->branch(l0);
        };

        std::string expected = std::string("s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)")
                               + std::string("\n") + std::string("s_branch");

        auto scheduler = Component::Get<Scheduling::Scheduler>(
            Scheduling::SchedulerProcedure::Sequential, m_context);
        std::vector<Generator<Instruction>> generators;
        generators.push_back(kb());
        m_context->schedule((*scheduler)(generators));
        if(m_context->targetArchitecture().target().getMajorVersion() != 9)
            GTEST_SKIP() << "Skipping BranchGeneratorWait tests for " << GetParam();
        auto found = NormalizedSource(output()).find(expected) != std::string::npos;
        EXPECT_EQ(found, true);
    }

    TEST_P(ARCH_BranchGeneratorTest, NoWait)
    {
        auto k = m_context->kernel();

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());
        m_context->kernelOptions().alwaysWaitBeforeBranch = false;
        auto kb                                           = [&]() -> Generator<Instruction> {
            auto l0 = m_context->labelAllocator()->label("l0");
            co_yield_(Instruction::Label(l0));
            co_yield m_context->brancher()->branch(l0);
        };

        std::string expected = std::string("s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)")
                               + std::string("\n") + std::string("s_branch");

        auto scheduler = Component::Get<Scheduling::Scheduler>(
            Scheduling::SchedulerProcedure::Sequential, m_context);
        std::vector<Generator<Instruction>> generators;
        generators.push_back(kb());
        m_context->schedule((*scheduler)(generators));
        if(m_context->targetArchitecture().target().getMajorVersion() != 9)
            GTEST_SKIP() << "Skipping BranchGeneratorWait tests for " << GetParam();
        auto found = NormalizedSource(output()).find(expected) != std::string::npos;
        EXPECT_EQ(found, false);
    }

    INSTANTIATE_TEST_SUITE_P(
        ARCH_BranchGeneratorTests,
        ARCH_BranchGeneratorTest,
        ::testing::ValuesIn(rocRoller::GPUArchitectureLibrary::getAllSupportedISAs()));

}
