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

#include <cstdio>
#include <iterator>

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#endif /* ROCROLLER_USE_HIP */

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
        void SetUp() override
        {
            Settings::getInstance()->set(Settings::AllowUnknownInstructions, true);
            GenericContextFixture::SetUp();
        }
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

    TEST_F(ArgumentLoaderTest, loadArgExtra)
    {
        auto kernel = m_context->kernel();
        auto loader = m_context->argLoader();

        kernel->setKernelDimensions(1);

        AssemblyKernelArgument arg{
            "bar", {DataType::Float}, DataDirection::ReadOnly, nullptr, 4, 4};
        kernel->addArgument(arg);

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
            s_load_dwordx2 s[4:5], s[0:1], 0
            s_load_dwordx2 s[6:7], s[0:1], 8
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
    }

    TEST_F(ArgumentLoaderTest, releaseArguments)
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

        Register::ValuePtr a, b, c;
        m_context->schedule(loader->getValue("a", a));
        m_context->schedule(loader->getValue("b", b));
        m_context->schedule(loader->getValue("c", c));

        const auto prevUseCount = a.use_count();
        // Releasing the argument from the loader will reduce the ValuePtr use_count by 1.
        loader->releaseArgument("a");
        EXPECT_EQ(prevUseCount - 1, a.use_count());
        a.reset();

        // Clearing the KMQueue via a WaitCount::KMCnt ensures that reloading the Value uses
        // the same set of registers.
        auto inst = Instruction::Wait(WaitCount::KMCnt(m_context->targetArchitecture(), 0, ""));
        m_context->schedule(inst);
        clearOutput();
        m_context->schedule(loader->getValue("a", a));

        EXPECT_EQ(NormalizedSource("s_load_dwordx2 s[4:5], s[0:1], 0"), NormalizedSource(output()));
        EXPECT_EQ((std::vector<int>{4, 5}), Generated(a->registerIndices()));
        VariableType expectedType = {DataType::Float, PointerType::PointerGlobal};
        EXPECT_EQ(expectedType, a->variableType());

        // Reset all remaining pointers to the original Allocation.
        loader->releaseAllArguments();
        a.reset();
        b.reset();
        c.reset();
        clearOutput();

        m_context->schedule(inst);
        clearOutput();
        m_context->schedule(loader->getValue("c", c));

        EXPECT_EQ(NormalizedSource("s_load_dwordx2 s[4:5], s[0:1], 16"),
                  NormalizedSource(output()));
        EXPECT_EQ((std::vector<int>{4, 5}), Generated(c->registerIndices()));
        EXPECT_EQ(DataType::UInt64, c->variableType());
    }

}
