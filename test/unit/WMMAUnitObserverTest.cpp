#include <rocRoller/Scheduling/Observers/FunctionalUnit/WMMAObserver.hpp>

#include "GPUContextFixture.hpp"

using namespace rocRoller;

class WMMAUnitObserverTest : public GPUContextFixtureParam<std::string>
{
public:
    WMMAUnitObserverTest() {}

    std::string opCode()
    {
        return std::get<1>(GetParam());
    }
};

TEST_P(WMMAUnitObserverTest, GPU_Direct)
{
    Scheduling::WMMAObserver observer(m_context);

    const auto v0
        = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Float, 8);
    const auto v1
        = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Half, 4);
    const auto v2
        = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Half, 4);

    v0->allocateNow();
    v1->allocateNow();
    v2->allocateNow();

    const auto wmmaInst = Instruction(opCode(), {v0}, {v1, v2, v0}, {}, "");
    const auto valuInst = Instruction("v_add_f32", {v1}, {v1, v2}, {}, "");

    EXPECT_EQ(0, observer.peek(wmmaInst).stallCycles);
    EXPECT_EQ(0, observer.peek(valuInst).stallCycles);

    observer.observe(wmmaInst);

    const auto info    = m_context->targetArchitecture().GetInstructionInfo(wmmaInst.getOpCode());
    const auto latency = info.getLatency();

    EXPECT_EQ(latency, observer.peek(wmmaInst).stallCycles);
    EXPECT_EQ(0, observer.peek(valuInst).stallCycles);

    observer.observe(valuInst);

    EXPECT_EQ(latency - 1, observer.peek(wmmaInst).stallCycles);
    EXPECT_EQ(0, observer.peek(valuInst).stallCycles);

    observer.observe(wmmaInst);

    EXPECT_EQ(latency, observer.peek(wmmaInst).stallCycles);
    EXPECT_EQ(0, observer.peek(valuInst).stallCycles);
}

TEST_P(WMMAUnitObserverTest, GPU_InContext)
{
    const auto v0
        = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Float, 8);
    const auto v1
        = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Half, 4);
    const auto v2
        = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Half, 4);

    v0->allocateNow();
    v1->allocateNow();
    v2->allocateNow();

    auto wmmaInst = Instruction(opCode(), {v0}, {v1, v2, v0}, {}, "");
    auto valuInst = Instruction("v_add_f32", {v1}, {v1, v2}, {}, "");

    EXPECT_EQ(0, m_context->peek(wmmaInst).stallCycles);
    EXPECT_EQ(0, m_context->peek(valuInst).stallCycles);

    const auto info    = m_context->targetArchitecture().GetInstructionInfo(wmmaInst.getOpCode());
    const auto latency = info.getLatency();

    m_context->schedule(wmmaInst);

    EXPECT_EQ(latency, m_context->peek(wmmaInst).stallCycles);
    EXPECT_EQ(0, m_context->peek(valuInst).stallCycles);

    m_context->schedule(valuInst);

    EXPECT_EQ(latency - 1, m_context->peek(wmmaInst).stallCycles);
    EXPECT_EQ(0, m_context->peek(valuInst).stallCycles);

    m_context->schedule(wmmaInst);

    EXPECT_EQ(latency, m_context->peek(wmmaInst).stallCycles);
    EXPECT_EQ(0, m_context->peek(valuInst).stallCycles);
}

INSTANTIATE_TEST_SUITE_P(WMMAUnitObserverTests,
                         WMMAUnitObserverTest,
                         ::testing::Combine(wmmaSupportedISAValues(),
                                            ::testing::Values("v_wmma_f32_16x16x16_f16")));
