// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>

#include <utility>

#include "HipStreamGuard.hpp"

namespace
{
::testing::AssertionResult hasHipDevice()
{
    int        deviceCount = 0;
    hipError_t err         = hipGetDeviceCount(&deviceCount);
    if(err != hipSuccess)
    {
        return ::testing::AssertionFailure()
               << "hipGetDeviceCount failed: " << hipGetErrorString(err);
    }
    if(deviceCount <= 0)
    {
        return ::testing::AssertionFailure() << "No HIP devices available";
    }
    return ::testing::AssertionSuccess();
}
} // namespace

TEST(HipStreamGuard, CreatesGetsAndSynchronizes)
{
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
    {
        GTEST_SKIP() << hipDevice.message();
    }

    TensileLite::testing::HipStreamGuard stream;
    ASSERT_NE(stream.get(), nullptr);
    stream.synchronize();
}

TEST(HipStreamGuard, SupportsMoveConstruction)
{
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
    {
        GTEST_SKIP() << hipDevice.message();
    }

    TensileLite::testing::HipStreamGuard a;
    auto                                 raw = a.get();

    TensileLite::testing::HipStreamGuard b(std::move(a));

    EXPECT_EQ(a.get(), nullptr);
    EXPECT_EQ(b.get(), raw);
    b.synchronize();
}

TEST(HipStreamGuard, SupportsMoveAssignment)
{
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
    {
        GTEST_SKIP() << hipDevice.message();
    }

    TensileLite::testing::HipStreamGuard a;
    TensileLite::testing::HipStreamGuard b;
    auto                                 rawA = a.get();
    auto                                 rawB = b.get();

    b = std::move(a);

    EXPECT_EQ(a.get(), nullptr);
    EXPECT_EQ(b.get(), rawA);
    EXPECT_NE(rawB, nullptr);
    b.synchronize();
}

TEST(HipStreamGuard, MovedFromSynchronizeIsNoop)
{
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
    {
        GTEST_SKIP() << hipDevice.message();
    }

    TensileLite::testing::HipStreamGuard a;
    TensileLite::testing::HipStreamGuard b(std::move(a));

    EXPECT_EQ(a.get(), nullptr);
    EXPECT_NE(b.get(), nullptr);
    EXPECT_NO_THROW(a.synchronize());
}
