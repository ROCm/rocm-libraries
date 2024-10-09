#include <rocRoller/Context.hpp>
#include <rocRoller/Scheduling/MetaObserver.hpp>
#include <rocRoller/Scheduling/Observers/AllocatingObserver.hpp>
#include <rocRoller/Scheduling/Observers/WaitcntObserver.hpp>

#include "SimpleFixture.hpp"

using namespace rocRoller;

class MetaObserverTest : public SimpleFixture
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

    constexpr static bool required(GPUArchitectureTarget const& target)
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

    constexpr static bool required(GPUArchitectureTarget const& target)
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
