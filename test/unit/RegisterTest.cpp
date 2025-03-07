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
#include <rocRoller/Context.hpp>
#include <rocRoller/InstructionValues/LabelAllocator.hpp>
#include <rocRoller/InstructionValues/Register.hpp>

#include "GPUContextFixture.hpp"
#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"

using namespace rocRoller;

class RegisterTest : public GenericContextFixture
{
};

TEST_F(RegisterTest, Simple)
{
    auto r
        = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Float, 1);
    EXPECT_EQ(false, r->canUseAsOperand());
    EXPECT_ANY_THROW(r->assertCanUseAsOperand());
    r->allocateNow();
    EXPECT_EQ(true, r->canUseAsOperand());
    EXPECT_NO_THROW(r->assertCanUseAsOperand());
}

TEST_F(RegisterTest, Promote)
{
    auto ExpectEqual = [&](Register::Type result, Register::Type lhs, Register::Type rhs) {
        EXPECT_EQ(result, Register::PromoteType(lhs, rhs));
        EXPECT_EQ(result, Register::PromoteType(rhs, lhs));
    };

    ExpectEqual(Register::Type::Literal, Register::Type::Literal, Register::Type::Literal);
    ExpectEqual(Register::Type::Scalar, Register::Type::Scalar, Register::Type::Scalar);
    ExpectEqual(Register::Type::Vector, Register::Type::Vector, Register::Type::Vector);
    ExpectEqual(
        Register::Type::Accumulator, Register::Type::Accumulator, Register::Type::Accumulator);

    ExpectEqual(Register::Type::Scalar, Register::Type::Scalar, Register::Type::Literal);
    ExpectEqual(Register::Type::Vector, Register::Type::Vector, Register::Type::Literal);
    ExpectEqual(Register::Type::Accumulator, Register::Type::Accumulator, Register::Type::Literal);

    ExpectEqual(Register::Type::Vector, Register::Type::Vector, Register::Type::Scalar);
    ExpectEqual(Register::Type::Accumulator, Register::Type::Accumulator, Register::Type::Scalar);
    ExpectEqual(Register::Type::Accumulator, Register::Type::Accumulator, Register::Type::Vector);

    EXPECT_ANY_THROW(Register::PromoteType(Register::Type::Literal, Register::Type::Label));
}

TEST_F(RegisterTest, RegisterToString)
{
    auto allocOptions = Register::AllocationOptions::FullyContiguous();

    auto r
        = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Float, 1);
    r->allocateNow();
    EXPECT_EQ("v0", r->toString());

    r = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Double, 1);
    r->allocateNow();
    EXPECT_EQ("v[0:1]", r->toString());
    EXPECT_EQ("neg(v[0:1])", r->negate()->toString());
    EXPECT_EQ("v0", r->subset({0})->toString());
    EXPECT_EQ("v1", r->subset({1})->toString());

    r = std::make_shared<Register::Value>(
        m_context, Register::Type::Vector, DataType::Double, 4, allocOptions);
    r->allocateNow();
    EXPECT_EQ("v[0:7]", r->toString());
    EXPECT_EQ("v4", r->subset({4})->toString());
    EXPECT_EQ("v[1:4]", r->subset({1, 2, 3, 4})->toString());
    EXPECT_EQ("v2", r->subset({1, 2, 3, 4})->subset({1})->toString());
    EXPECT_EQ("v[2:3]", r->element({1})->toString());
    EXPECT_EQ("v[4:5]", r->element({1, 2})->element({1})->toString());
    EXPECT_ANY_THROW(r->subset({22}));
    EXPECT_ANY_THROW(r->element({22}));

    r = std::make_shared<Register::Value>(
        m_context, Register::Type::Scalar, DataType::Float, 2, allocOptions);
    r->allocateNow();

    EXPECT_EQ("s[0:1]", r->toString());
    r.reset();

    auto a = std::make_shared<Register::Allocation>(
        m_context, Register::Type::Accumulator, DataType::Float, 2, allocOptions);

    auto             allocator      = m_context->allocator(Register::Type::Vector);
    std::vector<int> whichRegisters = {0, 4};
    allocator->allocate(a, whichRegisters);

    r = **a;
    EXPECT_EQ("[a0, a4]", r->toString());
}

TEST_F(RegisterTest, SimpleTests)
{
    auto r
        = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Float, 1);
    EXPECT_EQ(r->canAllocateNow(), true);
    auto r_large = std::make_shared<Register::Value>(
        m_context, Register::Type::Vector, DataType::Float, 10000);
    EXPECT_EQ(r_large->canAllocateNow(), false);
    auto r_placeholder
        = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Int32, 1);
    auto r_literal = Register::Value::Literal(3.5);

    r->allocateNow();
    EXPECT_EQ(r->allocationState(), Register::AllocationState::Allocated);

    EXPECT_EQ(r->isPlaceholder(), false);
    EXPECT_EQ(r_placeholder->isPlaceholder(), true);
    EXPECT_EQ(r_literal->isPlaceholder(), false);
}

TEST_F(RegisterTest, Literal)
{
    auto r = Register::Value::Literal(5);
    EXPECT_EQ("5", r->toString());
    EXPECT_EQ(true, r->canUseAsOperand());

    r = Register::Value::Literal(100.0f);
    EXPECT_EQ("100.000", r->toString());
    EXPECT_EQ(true, r->canUseAsOperand());

    r = Register::Value::Literal(3.14159f);
    EXPECT_EQ("3.14159", r->toString());

    r = Register::Value::Literal(2.7182818f);
    EXPECT_EQ("2.71828", r->toString());
}

TEST_F(RegisterTest, LiteralFromCommandArgumentValue)
{
    CommandArgumentValue v1 = 5;

    auto r1 = Register::Value::Literal(v1);

    EXPECT_EQ("5", r1->toString());

    CommandArgumentValue v2 = 7.5;

    auto r2 = Register::Value::Literal(v2);

    EXPECT_EQ("7.50000", r2->toString());

    int intVal;

    CommandArgumentValue v3 = &intVal;

    auto r3 = Register::Value::Literal(v3);

    VariableType expected{DataType::Int32, PointerType::PointerGlobal};

    EXPECT_EQ(r3->variableType(), expected);
}

TEST_F(RegisterTest, Name)
{
    auto r
        = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Float, 1);
    r->setName("D0");
    r->allocateNow();

    EXPECT_EQ("D0", r->name());
    EXPECT_EQ("D0", r->allocation()->name());

    r->allocation()->setName("D2");
    EXPECT_EQ("D2", r->allocation()->name());
    EXPECT_EQ("D0", r->name());

    r->setName("D3");
    EXPECT_EQ("D2", r->allocation()->name());
}

/**
 * This tests that emitting a label instruction from a label register works.
 */
TEST_F(RegisterTest, LabelRegister)
{
    {
        auto label = m_context->labelAllocator()->label("main_loop");
        EXPECT_EQ(true, label->canUseAsOperand());
        auto inst = Instruction::Label(label);
        m_context->schedule(inst);
    }

    auto expected0 = testKernelName() + "_0_main_loop:\n";

    EXPECT_EQ(NormalizedSource(output()), NormalizedSource(expected0));

    {
        auto label = m_context->labelAllocator()->label("main_loop");
        auto inst  = Instruction::Label(label);
        m_context->schedule(inst);
    }

    auto expected1 = expected0 + '\n' + testKernelName() + "_1_main_loop:\n";

    EXPECT_EQ(NormalizedSource(output()), NormalizedSource(expected1));
}

#define CHECK_ALLOCATION_STATE(state) EXPECT_EQ(#state, toString(Register::AllocationState::state))

TEST_F(RegisterTest, PrintAllocationState)
{
    CHECK_ALLOCATION_STATE(Unallocated);
    CHECK_ALLOCATION_STATE(Allocated);
    CHECK_ALLOCATION_STATE(Freed);
    CHECK_ALLOCATION_STATE(NoAllocation);
    CHECK_ALLOCATION_STATE(Error);
    CHECK_ALLOCATION_STATE(Count);

    std::stringstream stream;
    stream << Register::AllocationState::Allocated;
    EXPECT_THAT("Allocated", stream.str());
}

TEST_F(RegisterTest, NoAllocationSubsetAndElement)
{
    auto r
        = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Int64, 4);

    auto r2 = r->subset({0});
    EXPECT_THROW(r2->getRegisterIds().to<std::vector>(), FatalError);

    auto r3 = r->element({1});
    EXPECT_THROW(r3->getRegisterIds().to<std::vector>(), FatalError);

    auto r4 = r->subset({2});

    EXPECT_TRUE(r2->sameAs(r->subset({0})));
    EXPECT_TRUE(r2->sameAs(r2));
    EXPECT_FALSE(r2->sameAs(r));
    EXPECT_FALSE(r2->sameAs(r3));
    EXPECT_FALSE(r2->sameAs(r4));

    r->allocateNow();

    auto rIDs = r->getRegisterIds().to<std::vector>();
    ASSERT_EQ(8, rIDs.size());

    EXPECT_EQ(std::vector{rIDs[0]}, r2->getRegisterIds().to<std::vector>());

    EXPECT_EQ((std::vector{rIDs[2], rIDs[3]}), r3->getRegisterIds().to<std::vector>());

    EXPECT_TRUE(r2->sameAs(r->subset({0})));
    EXPECT_TRUE(r2->sameAs(r2));
    EXPECT_FALSE(r2->sameAs(r));
    EXPECT_FALSE(r2->sameAs(r3));
    EXPECT_FALSE(r2->sameAs(r4));
}

TEST_F(RegisterTest, AllocateAsNeededMultiLevel)
{
    auto r
        = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::UInt64, 4);

    auto r2 = r->subset({0, 2});
    EXPECT_THROW(r2->getRegisterIds().to<std::vector>(), FatalError);

    auto r3 = r2->element({1});
    EXPECT_THROW(r3->getRegisterIds().to<std::vector>(), FatalError);

    EXPECT_FALSE(r->sameAs(r2));
    EXPECT_FALSE(r3->sameAs(r2));
    EXPECT_TRUE(r3->sameAs(r->subset({2})));
    EXPECT_FALSE(r3->sameAs(r->subset({0})));

    r3->allocateNow(); //Allocating r3 should allocate the underlying allocation, even though it's 2 levels down.

    auto rIDs = r->getRegisterIds().to<std::vector>();
    ASSERT_EQ(8, rIDs.size());

    auto r2IDs = r2->getRegisterIds().to<std::vector>();
    ASSERT_EQ(2, r2IDs.size());

    auto r3IDs = r3->getRegisterIds().to<std::vector>();
    ASSERT_EQ(1, r3IDs.size());

    EXPECT_EQ((std::vector{rIDs[0], rIDs[2]}), r2->getRegisterIds().to<std::vector>());

    EXPECT_EQ((std::vector{rIDs[2]}), r3->getRegisterIds().to<std::vector>());

    EXPECT_FALSE(r->sameAs(r2));
    EXPECT_FALSE(r3->sameAs(r2));
    EXPECT_TRUE(r3->sameAs(r->subset({2})));
}

TEST_F(RegisterTest, SubsetOutOfBounds)
{
    auto r
        = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Int32, 4);
    r->allocateNow();
    EXPECT_THROW(r->subset({5}), FatalError);
}

TEST_F(RegisterTest, RegisterReuse)
{
    auto r
        = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Float, 8);
    r->allocateNow();
    EXPECT_EQ(r->allocationState(), Register::AllocationState::Allocated);

    auto val1 = r->subset({3, 4});

    auto val2 = std::make_shared<Register::Value>(
        r->allocation(), Register::Type::Vector, DataType::Half, val1->allocationCoord());

    EXPECT_EQ(val1->toString(), val2->toString());
}

TEST_F(RegisterTest, Split)
{
    std::vector<Register::ValuePtr> individualValues;

    std::vector<int> blockIndices;

    {
        auto block = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Raw32, 8);
        block->allocateNow();
        EXPECT_EQ(block->allocationState(), Register::AllocationState::Allocated);
        blockIndices = block->registerIndices().to<std::vector>();
        ASSERT_EQ(8, blockIndices.size());

        std::vector<std::vector<int>> indices = {{0}, {2, 3}, {4, 5, 6, 7}};

        individualValues = block->split(indices);

        ASSERT_EQ(3, individualValues.size());
    }

    EXPECT_EQ(std::vector<int>{blockIndices[0]},
              individualValues[0]->registerIndices().to<std::vector>());

    EXPECT_EQ((std::vector<int>{blockIndices[2], blockIndices[3]}),
              individualValues[1]->registerIndices().to<std::vector>());
    EXPECT_EQ(
        (std::vector<int>{blockIndices[4], blockIndices[5], blockIndices[6], blockIndices[7]}),
        individualValues[2]->registerIndices().to<std::vector>());

    auto allocator = m_context->allocator(Register::Type::Vector);
    EXPECT_EQ(true, allocator->isFree(blockIndices[1]));
}

TEST_F(RegisterTest, RegisterIDBasic)
{
    auto rs
        = std::make_shared<Register::Value>(m_context, Register::Type::Scalar, DataType::Float, 8);
    rs->allocateNow();

    for(auto& id1 : rs->getRegisterIds())
    {
        for(auto& id2 : rs->getRegisterIds())
        {
            if(id1 != id2)
            {
                EXPECT_NE(Register::RegisterIdHash()(id1), Register::RegisterIdHash()(id2));
            }
            else
            {
                EXPECT_EQ(Register::RegisterIdHash()(id1), Register::RegisterIdHash()(id2));
            }
        }
    }

    auto rv
        = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Float, 8);
    rv->allocateNow();

    for(auto& id1 : rv->getRegisterIds())
    {
        for(auto& id2 : rv->getRegisterIds())
        {
            if(id1 != id2)
            {
                EXPECT_NE(Register::RegisterIdHash()(id1), Register::RegisterIdHash()(id2));
            }
            else
            {
                EXPECT_EQ(Register::RegisterIdHash()(id1), Register::RegisterIdHash()(id2));
            }
        }
    }

    for(auto& id1 : rv->getRegisterIds())
    {
        for(auto& id2 : rs->getRegisterIds())
        {
            EXPECT_NE(Register::RegisterIdHash()(id1), Register::RegisterIdHash()(id2));
        }
    }
}

class ARCH_RegisterTest : public GPUContextFixture
{
};

TEST_P(ARCH_RegisterTest, GPU_RegisterIDSpecial)
{
    std::vector<std::shared_ptr<rocRoller::Register::Value>> specialRegisters(
        {m_context->getVCC_LO(),
         m_context->getVCC_HI(),
         m_context->getSCC(),
         m_context->getExec()});
    std::vector<Register::RegisterId> specialRegisterIDs;
    for(auto& specialRegister : specialRegisters)
    {
        for(auto& id : specialRegister->getRegisterIds())
        {
            specialRegisterIDs.push_back(id);
        }
    }
    EXPECT_EQ(specialRegisterIDs.size(), specialRegisters.size());

    for(int i = 0; i < specialRegisters.size(); i++)
    {
        for(int j = 0; j < specialRegisters.size(); j++)
        {
            if(specialRegisters[i]->toString() != specialRegisters[j]->toString())
            {
                EXPECT_NE(specialRegisterIDs[i], specialRegisterIDs[j]);
                EXPECT_NE(Register::RegisterIdHash()(specialRegisterIDs[i]),
                          Register::RegisterIdHash()(specialRegisterIDs[j]));
            }
            else
            {
                EXPECT_EQ(specialRegisterIDs[i], specialRegisterIDs[j]);
                EXPECT_EQ(Register::RegisterIdHash()(specialRegisterIDs[i]),
                          Register::RegisterIdHash()(specialRegisterIDs[j]));
            }
        }
    }

    std::vector<Register::RegisterId> vccRegisterIDs;
    for(auto& id : m_context->getVCC()->getRegisterIds())
    {
        vccRegisterIDs.push_back(id);
    }
    if(m_context->kernel()->wavefront_size() == 64)
    {
        EXPECT_EQ(vccRegisterIDs.size(), 2);
        EXPECT_NE(vccRegisterIDs[0], vccRegisterIDs[1]);
        EXPECT_NE(Register::RegisterIdHash()(vccRegisterIDs[0]),
                  Register::RegisterIdHash()(vccRegisterIDs[1]));
        EXPECT_EQ(vccRegisterIDs[1], specialRegisterIDs[1]);
    }
    else
    {
        EXPECT_EQ(vccRegisterIDs.size(), 1);
    }
    EXPECT_EQ(vccRegisterIDs[0], specialRegisterIDs[0]);
}

INSTANTIATE_TEST_SUITE_P(ARCH_RegisterTest, ARCH_RegisterTest, supportedISATuples());
