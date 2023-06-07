#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>

#include <rocRoller/Scheduling/Observers/FunctionalUnit/MFMAObserver.hpp>

#include "GPUContextFixture.hpp"

using namespace rocRoller;

class MFMAUnitObserverTest : public GPUContextFixture
{
public:
    MFMAUnitObserverTest() {}
};

TEST_P(MFMAUnitObserverTest, Direct)
{
    Scheduling::MFMAObserver observer(m_context);

    auto agpr
        = Register::Value::Placeholder(m_context, Register::Type::Accumulator, DataType::Float, 16);

    auto v0 = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Half, 4);
    auto v1 = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Half, 4);

    agpr->allocateNow();
    v0->allocateNow();
    v1->allocateNow();

    auto mfmaInst = Instruction("v_mfma_f32_32x32x8f16", {agpr}, {v0, v1, agpr}, {}, "");
    auto valuInst = Instruction("v_add_f32", {v0}, {v0, v1}, {}, "");

    EXPECT_EQ(0, observer.peek(mfmaInst).stallCycles);
    EXPECT_EQ(0, observer.peek(valuInst).stallCycles);

    observer.observe(mfmaInst);

    EXPECT_EQ(16, observer.peek(mfmaInst).stallCycles);
    EXPECT_EQ(0, observer.peek(valuInst).stallCycles);

    observer.observe(valuInst);

    EXPECT_EQ(15, observer.peek(mfmaInst).stallCycles);
    EXPECT_EQ(0, observer.peek(valuInst).stallCycles);

    observer.observe(mfmaInst);

    EXPECT_EQ(16, observer.peek(mfmaInst).stallCycles);
    EXPECT_EQ(0, observer.peek(valuInst).stallCycles);
}

TEST_P(MFMAUnitObserverTest, InContext)
{
    auto agpr
        = Register::Value::Placeholder(m_context, Register::Type::Accumulator, DataType::Float, 16);

    auto v0 = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Half, 4);
    auto v1 = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Half, 4);

    agpr->allocateNow();
    v0->allocateNow();
    v1->allocateNow();

    auto mfmaInst = Instruction("v_mfma_f32_32x32x8f16", {agpr}, {v0, v1, agpr}, {}, "");
    auto valuInst = Instruction("v_add_f32", {v0}, {v0, v1}, {}, "");

    EXPECT_EQ(0, m_context->peek(mfmaInst).stallCycles);
    EXPECT_EQ(0, m_context->peek(valuInst).stallCycles);

    m_context->schedule(mfmaInst);

    EXPECT_EQ(16, m_context->peek(mfmaInst).stallCycles);
    EXPECT_EQ(0, m_context->peek(valuInst).stallCycles);

    m_context->schedule(valuInst);

    EXPECT_EQ(15, m_context->peek(mfmaInst).stallCycles);
    EXPECT_EQ(0, m_context->peek(valuInst).stallCycles);

    m_context->schedule(mfmaInst);

    EXPECT_EQ(16, m_context->peek(mfmaInst).stallCycles);
    EXPECT_EQ(0, m_context->peek(valuInst).stallCycles);
}

INSTANTIATE_TEST_SUITE_P(MFMAUnitObserverTest, MFMAUnitObserverTest, mfmaSupportedISATuples());
