
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "GenericContextFixture.hpp"

#include <rocRoller/Utilities/Timer.hpp>

#include <chrono>
#include <thread>

template <typename T>
T abs(T x)
{
    return x > 0 ? x : -x;
}

using namespace rocRoller;

class TimerTest : public GenericContextFixture
{
};

void timed_sleep()
{
    auto timer = Timer("rocRoller::timed_sleep");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

void untimed_sleep()
{
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

TEST_F(TimerTest, Base01)
{
    auto t = Timer("rocRoller");

    t.tic();
    timed_sleep();
    untimed_sleep();
    t.toc();

    auto t1 = TimerPool::milliseconds("rocRoller::timed_sleep");
    auto t2 = TimerPool::milliseconds("rocRoller");

    EXPECT_TRUE(abs(t.milliseconds() - 20) < 2);
    EXPECT_TRUE(abs(t1 - 10) < 2);
    EXPECT_TRUE(abs(t2 - 20) < 2);
}
