#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rocRoller/Context.hpp>
#include <rocRoller/Scheduling/MetaObserver.hpp>
#include <rocRoller/Scheduling/Observers/AllocatingObserver.hpp>
#include <rocRoller/Scheduling/Observers/WaitcntObserver.hpp>

using namespace rocRoller;

class MetaObserverTest : public ::testing::Test
{
};

class TestFalseObserver
{
public:
    TestFalseObserver() {}

    TestFalseObserver(ContextPtr context)
        : m_context(context){

        };

    Scheduling::InstructionStatus peek(Instruction const& inst) const
    {
        return Scheduling::InstructionStatus();
    };

    void modify(Instruction& inst) const {}

    void observe(Instruction const& inst) {}

    static bool required(ContextPtr)
    {
        return false;
    }

private:
    std::weak_ptr<Context> m_context;
};

class TestTrueObserver
{
public:
    TestTrueObserver() {}

    TestTrueObserver(ContextPtr context)
        : m_context(context){

        };

    Scheduling::InstructionStatus peek(Instruction const& inst) const
    {
        return Scheduling::InstructionStatus();
    };

    void modify(Instruction& inst) const {}

    void observe(Instruction const& inst) {}

    static bool required(ContextPtr)
    {
        return true;
    }

private:
    std::weak_ptr<Context> m_context;
};

static_assert(Scheduling::CObserver<TestTrueObserver>);
static_assert(Scheduling::CObserver<TestFalseObserver>);

TEST_F(MetaObserverTest, MultipleObserverTest)
{
    rocRoller::ContextPtr m_context = std::make_shared<Context>();

    std::tuple<Scheduling::AllocatingObserver,
               Scheduling::WaitcntObserver,
               Scheduling::AllocatingObserver,
               Scheduling::WaitcntObserver>
        constructedObservers = {Scheduling::AllocatingObserver(m_context),
                                Scheduling::WaitcntObserver(m_context),
                                Scheduling::AllocatingObserver(m_context),
                                Scheduling::WaitcntObserver(m_context)};

    using MyObserver      = Scheduling::MetaObserver<Scheduling::AllocatingObserver,
                                                Scheduling::WaitcntObserver,
                                                Scheduling::AllocatingObserver,
                                                Scheduling::WaitcntObserver>;
    m_context->observer() = std::make_shared<MyObserver>(constructedObservers);
}

TEST_F(MetaObserverTest, Required)
{
    using FalseFalseObserver = Scheduling::MetaObserver<TestFalseObserver, TestFalseObserver>;
    using TrueTrueObserver   = Scheduling::MetaObserver<TestTrueObserver, TestTrueObserver>;
    using FalseTrueObserver  = Scheduling::MetaObserver<TestFalseObserver, TestTrueObserver>;
    using TrueFalseObserver  = Scheduling::MetaObserver<TestTrueObserver, TestFalseObserver>;

    rocRoller::ContextPtr m_context = std::make_shared<Context>();

    EXPECT_TRUE(TrueTrueObserver::required(m_context));
    EXPECT_FALSE(FalseFalseObserver::required(m_context));
    EXPECT_FALSE(FalseTrueObserver::required(m_context));
    EXPECT_FALSE(TrueFalseObserver::required(m_context));
}
