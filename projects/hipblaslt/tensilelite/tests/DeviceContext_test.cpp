// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "DeviceContext.hpp"
#include "TimingEvents.hpp"

#include <hip/hip_runtime.h>

#include <memory>
#include <vector>

using TensileLite::Client::DeviceContext;
using TensileLite::Client::ScopedDevice;
using TensileLite::Client::TimingEvents;

namespace
{
    int deviceCount()
    {
        int n = 0;
        if(hipGetDeviceCount(&n) != hipSuccess)
            return 0;
        return n;
    }

    int currentDevice()
    {
        int d = -1;
        EXPECT_EQ(hipGetDevice(&d), hipSuccess);
        return d;
    }
}

TEST(ScopedDevice, SwitchesAndRestores)
{
    if(deviceCount() < 2)
        GTEST_SKIP() << "needs two devices";
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    {
        ScopedDevice guard(1);
        EXPECT_EQ(currentDevice(), 1);
    }
    EXPECT_EQ(currentDevice(), 0);
}

TEST(DeviceContext, StreamLivesOnItsDevice)
{
    if(deviceCount() < 2)
        GTEST_SKIP() << "needs two devices";
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    DeviceContext ctx(1, false);
    EXPECT_EQ(currentDevice(), 0);
    ASSERT_NE(ctx.stream, nullptr);
    hipDevice_t dev = -1;
    ASSERT_EQ(hipStreamGetDevice(ctx.stream, &dev), hipSuccess);
    EXPECT_EQ(dev, 1);
    EXPECT_NE(ctx.adapter.get(), nullptr);
}

TEST(DeviceContext, DefaultStreamIsNull)
{
    if(deviceCount() < 1)
        GTEST_SKIP() << "needs a device";
    DeviceContext ctx(0, true);
    EXPECT_EQ(ctx.stream, nullptr);
}

TEST(TimingEvents, CountsInvocationsWithoutCards)
{
    TimingEvents events({}, 4, 1);
    EXPECT_EQ(events.numInvocations(), 4u);
    EXPECT_EQ(events.numCards(), 0u);
}

TEST(TimingEvents, CreatesEachCardsEventsOnItsDevice)
{
    if(deviceCount() < 2)
        GTEST_SKIP() << "needs two devices";
    std::vector<std::shared_ptr<DeviceContext>> devices{std::make_shared<DeviceContext>(0, false),
                                                        std::make_shared<DeviceContext>(1, false)};
    TimingEvents                                events(devices, 3, 2);
    ASSERT_EQ(events.numCards(), 2u);
    for(size_t d = 0; d < devices.size(); d++)
    {
        ASSERT_EQ(events.card(d).size(), 3u);
        ASSERT_EQ(events.card(d)[2].size(), 2u);
        ScopedDevice device(devices[d]->deviceId);
        EXPECT_EQ(hipEventRecord(events.card(d)[2][1], devices[d]->stream), hipSuccess);
        EXPECT_EQ(hipEventSynchronize(events.card(d)[2][1]), hipSuccess);
    }
}
