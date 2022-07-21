
#include <gtest/gtest.h>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/ExecutableKernel.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureLibrary.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/InstructionValues/RegisterAllocator.hpp>

#include "GPUContextFixture.hpp"
#include "GenericContextFixture.hpp"

using namespace rocRoller;

namespace RegisterAllocatorTest
{
    class RegisterAllocatorTest : public GenericContextFixture
    {
    };

    TEST_F(RegisterAllocatorTest, WeakPtrBehaviour)
    {
        std::weak_ptr<int> test;
        EXPECT_EQ(true, test.expired());

        auto sh = std::make_shared<int>(4);
        test    = sh;

        EXPECT_EQ(false, test.expired());

        sh.reset();

        EXPECT_EQ(true, test.expired());
    }

    TEST_F(RegisterAllocatorTest, Simple)
    {
        auto allocator = std::make_shared<Register::Allocator>(Register::Type::Scalar, 10);

        EXPECT_EQ(-1, allocator->maxUsed());
        EXPECT_EQ(0, allocator->useCount());
        EXPECT_EQ(allocator->regType(), Register::Type::Scalar);
        EXPECT_EQ(allocator->size(), 10);

        auto alloc0 = std::make_shared<Register::Allocation>(
            m_context, Register::Type::Scalar, DataType::Float);

        EXPECT_EQ(-1, allocator->maxUsed());
        EXPECT_EQ(0, allocator->useCount());

        EXPECT_EQ(0, allocator->findContiguousRange(0, 1, alloc0->options()));

        allocator->allocate(alloc0);

        EXPECT_EQ(std::vector{0}, alloc0->registerIndices());
        EXPECT_EQ(false, allocator->isFree(0));
        EXPECT_EQ(0, allocator->maxUsed());
        EXPECT_EQ(1, allocator->useCount());

        auto alloc1 = std::make_shared<Register::Allocation>(
            m_context, Register::Type::Scalar, DataType::Float, 3);

        allocator->allocate(alloc1);

        EXPECT_EQ((std::vector{1, 2, 3}), alloc1->registerIndices());
        EXPECT_EQ(false, allocator->isFree(1));
        EXPECT_EQ(false, allocator->isFree(2));
        EXPECT_EQ(false, allocator->isFree(3));
        EXPECT_EQ(true, allocator->isFree(4));
        EXPECT_EQ(3, allocator->maxUsed());
        EXPECT_EQ(4, allocator->useCount());

        alloc0.reset();
        EXPECT_EQ(true, allocator->isFree(0));
        EXPECT_EQ(3, allocator->maxUsed());
        EXPECT_EQ(4, allocator->useCount());

        Register::Allocation::Options options;
        options.contiguous = false;

        auto alloc2 = std::make_shared<Register::Allocation>(
            m_context, Register::Type::Scalar, DataType::Float, 3, options);

        allocator->allocate(alloc2);

        EXPECT_EQ((std::vector{0, 4, 5}), alloc2->registerIndices());
        EXPECT_EQ(false, allocator->isFree(0));
        EXPECT_EQ(false, allocator->isFree(4));
        EXPECT_EQ(false, allocator->isFree(5));
        EXPECT_EQ(true, allocator->isFree(6));

        auto alloc3 = std::make_shared<Register::Allocation>(
            m_context, Register::Type::Scalar, DataType::Float, 5, options);

        EXPECT_EQ(false, allocator->canAllocate(alloc3));

        alloc1->free();
        EXPECT_EQ(true, allocator->isFree(1));
        EXPECT_EQ(true, allocator->isFree(2));
        EXPECT_EQ(true, allocator->isFree(3));
        EXPECT_EQ(5, allocator->maxUsed());
        EXPECT_EQ(6, allocator->useCount());
    }

    class ARCH_RegisterAllocatorTest : public GPUContextFixture
    {
    };

    TEST_P(ARCH_RegisterAllocatorTest, AlignedSGPR)
    {
        auto k = m_context->kernel();

        k->setKernelName("AlignedSGPR");
        k->setKernelDimensions(1);

        k->addArgument(
            {"result", {DataType::Int32, PointerType::PointerGlobal}, DataDirection::WriteOnly});
        k->addArgument({"a", DataType::Int32});
        k->addArgument({"b", DataType::Int32});

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_result, s_a, s_b;
            co_yield m_context->argLoader()->getValue("result", s_result);
            co_yield m_context->argLoader()->getValue("a", s_a);
            co_yield m_context->argLoader()->getValue("b", s_b);

            auto s_c = Register::Value::Placeholder(
                m_context, Register::Type::Scalar, DataType::Raw32, 2);

            auto v_a = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int32, 1);

            auto v_b = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int32, 1);

            co_yield s_c->allocate();
            co_yield v_a->allocate();
            co_yield v_b->allocate();

            // this will trip "invalid register alignment" if s_c
            // isn't aligned properly on some archs
            co_yield_(Instruction("v_cmp_ge_i32", {s_c}, {v_a, v_b}, {}, ""));
        };

        m_context->schedule(kb());
        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        if(m_context->targetArchitecture().target().getMajorVersion() != 9)
            GTEST_SKIP() << "Skipping SGPR alignment tests for " << GetParam();

        std::vector<char> assembledKernel = m_context->instructions()->assemble();
        EXPECT_GT(assembledKernel.size(), 0);
    }

    INSTANTIATE_TEST_SUITE_P(
        ARCH_RegisterAllocatorTests,
        ARCH_RegisterAllocatorTest,
        ::testing::ValuesIn(rocRoller::GPUArchitectureLibrary::getAllSupportedISAs()));

}
