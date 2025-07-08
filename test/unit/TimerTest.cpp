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

#include "GPUContextFixture.hpp"
#include "GenericContextFixture.hpp"

#include <rocRoller/Utilities/HIPTimer.hpp>
#include <rocRoller/Utilities/Timer.hpp>

#include <chrono>
#include <thread>

using namespace rocRoller;

class TimerTest : public GenericContextFixture
{
};

class TimerTestGPU : public CurrentGPUContextFixture
{
    void SetUp() override
    {
        CurrentGPUContextFixture::SetUp();
    }
};

void timed_sleep()
{
    auto timer = Timer("rocRoller::TimerTest::timed_sleep");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

void untimed_sleep()
{
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

TEST_F(TimerTest, Base01)
{
    auto t = Timer("rocRoller::TimerTest.Base01");

    t.tic();
    timed_sleep();
    untimed_sleep();
    t.toc();

    auto t1 = TimerPool::milliseconds("rocRoller::TimerTest::timed_sleep");
    auto t2 = TimerPool::milliseconds("rocRoller::TimerTest.Base01");

    EXPECT_EQ(t.milliseconds(), t2);
    EXPECT_GT(t2, t1);
    EXPECT_GE(t1, 10);
    EXPECT_GE(t2, 20);
}

TEST_F(TimerTest, Destructor)
{
    TimerPool::clear();

    size_t t1, t2;
    {
        auto t = Timer("rocRoller::TimerTest.Destructor");

        t.tic();
        untimed_sleep();
        t.toc();

        t1 = t.milliseconds();
        t2 = TimerPool::milliseconds("rocRoller::TimerTest.Destructor");
        EXPECT_EQ(t1, t2);
    }
    std::cout << t1;
    EXPECT_EQ(t1, TimerPool::milliseconds("rocRoller::TimerTest.Destructor"));
}

TEST_F(TimerTestGPU, GPU_Destructor)
{
    TimerPool::clear();

    size_t t1, t2;
    {
        auto t = HIPTimer("rocRoller::TimerTestGPU.Destructor");

        t.tic();
        untimed_sleep();
        t.toc();
        t.sync();

        t1 = t.milliseconds();
        t2 = TimerPool::milliseconds("rocRoller::TimerTestGPU.Destructor");
        EXPECT_EQ(t1, t2);
    }
    std::cout << t1;
    EXPECT_EQ(t1, TimerPool::milliseconds("rocRoller::TimerTestGPU.Destructor"));
}
