// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "DeviceContext.hpp"
#include "TimingEvents.hpp"

#include <hip/hip_runtime.h>

#include <memory>
#include <stdexcept>
#include <string>
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

TEST(DeviceContext, SdmaRotateWalksTheEvenXgmiEngines)
{
    using TensileLite::Client::sdmaRotateEngine;
    EXPECT_EQ(sdmaRotateEngine(4, 0, 14), 4u);
    EXPECT_EQ(sdmaRotateEngine(2, -1, 14), 14u);
    EXPECT_EQ(sdmaRotateEngine(14, 1, 14), 2u);
    EXPECT_THROW(sdmaRotateEngine(3, -1, 14), std::runtime_error);
}

TEST(DeviceContext, SdmaRotateRejectsOddAndPcieEnginesAtAnyRotate)
{
    using TensileLite::Client::sdmaRotateEngine;
    EXPECT_THROW(sdmaRotateEngine(3, 0, 14), std::runtime_error);
    EXPECT_THROW(sdmaRotateEngine(7, 0, 6), std::runtime_error);
    EXPECT_THROW(sdmaRotateEngine(0, 0, 14), std::runtime_error);
    EXPECT_EQ(sdmaRotateEngine(6, 0, 6), 6u);
}

TEST(DeviceContext, SdmaRotateReportsAnUnusableEngineFirst)
{
    using TensileLite::Client::sdmaRotateEngine;
    try
    {
        sdmaRotateEngine(3, -1, 6);
        FAIL();
    }
    catch(const std::runtime_error& e)
    {
        EXPECT_EQ(std::string(e.what()).find("--fused-a2a-sdma-rotate=0"), std::string::npos);
    }
}

TEST(DeviceContext, SdmaRotateRequiresFourteenXgmiEngines)
{
    using TensileLite::Client::sdmaRotateEngine;
    EXPECT_THROW(sdmaRotateEngine(2, -1, 6), std::runtime_error);
    EXPECT_THROW(sdmaRotateEngine(2, 1, 0), std::runtime_error);
    EXPECT_EQ(sdmaRotateEngine(2, 0, 6), 2u);
}
