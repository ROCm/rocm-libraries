
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rocRoller/Context.hpp>
#include <rocRoller/InstructionValues/LabelAllocator.hpp>
#include <rocRoller/InstructionValues/Register.hpp>

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
    r->allocateNow();
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

    r = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Double, 4);
    r->allocateNow();
    EXPECT_EQ("v[0:7]", r->toString());
    EXPECT_EQ("v4", r->subset({4})->toString());
    EXPECT_EQ("v[1:4]", r->subset({1, 2, 3, 4})->toString());
    EXPECT_EQ("v2", r->subset({1, 2, 3, 4})->subset({1})->toString());
    EXPECT_EQ("v[2:3]", r->element({1})->toString());
    EXPECT_EQ("v[4:5]", r->element({1, 2})->element({1})->toString());
    EXPECT_ANY_THROW(r->subset({22}));
    EXPECT_ANY_THROW(r->element({22}));

    r = std::make_shared<Register::Value>(m_context, Register::Type::Scalar, DataType::Float, 2);
    r->allocateNow();

    EXPECT_EQ("s[0:1]", r->toString());
    r.reset();

    auto a = std::make_shared<Register::Allocation>(
        m_context, Register::Type::Accumulator, DataType::Float, 2);

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

    r->freeNow();
    EXPECT_EQ(r->allocationState(), Register::AllocationState::Unallocated);
}

TEST_F(RegisterTest, Literal)
{
    auto r = Register::Value::Literal(5);
    EXPECT_EQ("5", r->toString());

    r = Register::Value::Literal(100.0f);
    EXPECT_EQ("100.000", r->toString());

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
}

/**
 * This tests that emitting a label instruction from a label register works.
 */
TEST_F(RegisterTest, LabelRegister)
{
    {
        auto label = m_context->labelAllocator()->label("main_loop");
        auto inst  = Instruction::Label(label);
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

#define CHECK_ALLOCATION_STATE(state) EXPECT_EQ(#state, ToString(Register::AllocationState::state))

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

TEST_F(RegisterTest, NoAllocationSubset)
{
    auto r
        = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Int32, 4);
    EXPECT_THROW(r->subset({0}), FatalError);
}

TEST_F(RegisterTest, SubsetOutOfBounds)
{
    auto r
        = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Int32, 4);
    r->allocateNow();
    EXPECT_THROW(r->subset({5}), FatalError);
}
