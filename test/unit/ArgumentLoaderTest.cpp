
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdio>
#include <iterator>

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/Utilities/Generator.hpp>

#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"

using namespace rocRoller;

namespace rocRollerTest
{
    class ArgumentLoaderTest : public GenericContextFixture
    {
    };

    TEST_F(ArgumentLoaderTest, IsContiguousRange)
    {
        std::vector<int> numbers{0, 1, 2, 3, 5, 6, 7, 9};

        EXPECT_EQ(true, IsContiguousRange(numbers.begin(), numbers.begin() + 4));
        EXPECT_EQ(false, IsContiguousRange(numbers.begin(), numbers.begin() + 5));
        EXPECT_EQ(false, IsContiguousRange(numbers.begin() + 3, numbers.begin() + 5));
        EXPECT_EQ(true, IsContiguousRange(numbers.begin() + 4, numbers.begin() + 7));
        EXPECT_EQ(false, IsContiguousRange(numbers.begin() + 4, numbers.begin() + 8));
    }

    TEST_F(ArgumentLoaderTest, pickInstructionWidth)
    {
        auto indices = Generated(iota(0, 16));

        std::vector<int> gap{0, 1, 3, 4, 5, 6, 7, 8};

        EXPECT_EQ(4, rocRoller::PickInstructionWidthBytes(0, 4, indices.begin(), indices.end()));
        EXPECT_EQ(8, rocRoller::PickInstructionWidthBytes(0, 8, indices.begin(), indices.end()));
        EXPECT_EQ(16, rocRoller::PickInstructionWidthBytes(0, 16, indices.begin(), indices.end()));
        EXPECT_EQ(32, rocRoller::PickInstructionWidthBytes(0, 32, indices.begin(), indices.end()));
        EXPECT_EQ(64, rocRoller::PickInstructionWidthBytes(0, 64, indices.begin(), indices.end()));
        EXPECT_EQ(64, rocRoller::PickInstructionWidthBytes(0, 128, indices.begin(), indices.end()));

        EXPECT_EQ(4, rocRoller::PickInstructionWidthBytes(4, 16, indices.begin(), indices.end()));

        // only the first 2 registers in `gap` are contiguous.
        EXPECT_EQ(8, rocRoller::PickInstructionWidthBytes(0, 16, gap.begin(), gap.end()));

        // registers must be aligned
        EXPECT_EQ(4, rocRoller::PickInstructionWidthBytes(0, 16, gap.begin() + 1, gap.end()));

        EXPECT_EQ(16, rocRoller::PickInstructionWidthBytes(0, 16, gap.begin() + 3, gap.end()));
    }

    TEST_F(ArgumentLoaderTest, loadRangeUnaligned)
    {
        Register::ValuePtr r;
        auto               kernel = m_context->kernel();
        auto               loader = m_context->argLoader();

        // kernargs: 2, workgroup id: 1, next SGPR: 3
        kernel->setKernelDimensions(1);

        m_context->schedule(kernel->allocateInitialRegisters());
        m_context->schedule(loader->loadRange(0, 16, r));

        {
            std::string expected = R"(
                s_load_dword s3, s[0:1], 0
                s_load_dword s4, s[0:1], 4
                s_load_dword s5, s[0:1], 8
                s_load_dword s6, s[0:1], 12
            )";

            EXPECT_EQ(NormalizedSource(expected), NormalizedSource(output()));
        }

        clearOutput();

        m_context->schedule(loader->loadRange(4, 20, r));

        {
            std::string expected = R"(
                s_load_dword s3, s[0:1], 4        // Unaligned
                s_load_dwordx2 s[4:5], s[0:1], 8  // Aligned
                s_load_dword s6, s[0:1], 16       // Aligned but only 1 dword left to load.
            )";

            EXPECT_EQ(NormalizedSource(expected), NormalizedSource(output()));
        }
    }

    TEST_F(ArgumentLoaderTest, loadRangeAligned)
    {
        Register::ValuePtr r;
        auto               kernel = m_context->kernel();
        auto               loader = m_context->argLoader();

        // kernargs: 2, workgroup id: 2, next SGPR: 4
        kernel->setKernelDimensions(2);

        m_context->schedule(kernel->allocateInitialRegisters());
        m_context->schedule(loader->loadRange(0, 16, r));

        std::string expected = R"(
            s_load_dwordx4 s[4:7], s[0:1], 0
        )";

        EXPECT_EQ(NormalizedSource(expected), NormalizedSource(output()));
        EXPECT_EQ(DataType::Raw32, r->variableType());
    }

    TEST_F(ArgumentLoaderTest, loadArgExtra)
    {
        auto kernel = m_context->kernel();
        auto loader = m_context->argLoader();

        kernel->setKernelDimensions(1);

        AssemblyKernelArgument arg{
            "bar", {DataType::Float}, DataDirection::ReadOnly, nullptr, 4, 4};

        m_context->schedule(kernel->allocateInitialRegisters());

        m_context->schedule(loader->loadArgument(arg));

        std::string expected = R"(
            s_load_dword s3, s[0:1], 4
        )";

        EXPECT_EQ(NormalizedSource(expected), NormalizedSource(output()));

        clearOutput();

        Register::ValuePtr reg;
        m_context->schedule(loader->getValue("bar", reg));
        ASSERT_NE(nullptr, reg);
        EXPECT_EQ(DataType::Float, reg->variableType());
        EXPECT_EQ(std::vector<int>{3}, Generated(reg->registerIndices()));

        // Should not need to load a second time.
        EXPECT_EQ(NormalizedSource(""), NormalizedSource(output()));
    }

    TEST_F(ArgumentLoaderTest, loadAllArguments)
    {
        auto kernel = m_context->kernel();
        kernel->setKernelDimensions(2);

        VariableType floatPtr{DataType::Float, PointerType::PointerGlobal};

        kernel->addArgument({"a", floatPtr, DataDirection::WriteOnly});
        kernel->addArgument({"b", floatPtr, DataDirection::ReadOnly});
        kernel->addArgument({"c", {DataType::UInt64}});
        auto loader = m_context->argLoader();

        m_context->schedule(kernel->allocateInitialRegisters());
        m_context->schedule(loader->loadAllArguments());

        std::string expected = R"(
            s_load_dwordx4 s[4:7], s[0:1], 0
            s_load_dwordx2 s[8:9], s[0:1], 16
        )";

        EXPECT_EQ(NormalizedSource(expected), NormalizedSource(output()));

        clearOutput();

        Register::ValuePtr a, b, c;
        m_context->schedule(loader->getValue("a", a));
        m_context->schedule(loader->getValue("b", b));
        m_context->schedule(loader->getValue("c", c));

        ASSERT_NE(nullptr, a);
        EXPECT_EQ((std::vector<int>{4, 5}), Generated(a->registerIndices()));
        EXPECT_EQ(floatPtr, a->variableType());

        ASSERT_NE(nullptr, b);
        EXPECT_EQ((std::vector<int>{6, 7}), Generated(b->registerIndices()));
        EXPECT_EQ(floatPtr, b->variableType());

        ASSERT_NE(nullptr, c);
        EXPECT_EQ((std::vector<int>{8, 9}), Generated(c->registerIndices()));
        EXPECT_EQ(DataType::UInt64, c->variableType());

        EXPECT_EQ(NormalizedSource(""), NormalizedSource(output()));

        // force 'a' to be loaded again.
        loader->releaseArgument("a");
        EXPECT_EQ(1, a.use_count());
        a.reset();
        m_context->schedule(loader->getValue("a", a));

        // Since args were loaded using loadAllArguments(), they are allocated and freed as
        // a block using a single Register::Allocation object, and then partitioned into
        // multiple Register::Value objects.  We won't get registers s[4:5] back until all
        // the shared_ptrs referring to the arguments have been destroyed.
        // A future optimization may be to allow an Allocation to be freed piecemeal.
        EXPECT_EQ(NormalizedSource("s_load_dwordx2 s[10:11], s[0:1], 0"),
                  NormalizedSource(output()));
        EXPECT_EQ((std::vector<int>{10, 11}), Generated(a->registerIndices()));
        VariableType expectedType = {DataType::Float, PointerType::PointerGlobal};
        EXPECT_EQ(expectedType, a->variableType());

        // This should destroy all remaining pointers to the original Allocation.
        loader->releaseAllArguments();
        b.reset();
        c.reset();
        clearOutput();

        m_context->schedule(loader->getValue("c", c));

        EXPECT_EQ(NormalizedSource("s_load_dwordx2 s[4:5], s[0:1], 16"),
                  NormalizedSource(output()));
        EXPECT_EQ((std::vector<int>{4, 5}), Generated(c->registerIndices()));
        EXPECT_EQ(DataType::UInt64, c->variableType());
    }

}
