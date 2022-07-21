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

TEST_F(MetaObserverTest, MultipleObserverTest)
{
    std::shared_ptr<rocRoller::Context> m_context = std::make_shared<Context>();

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
