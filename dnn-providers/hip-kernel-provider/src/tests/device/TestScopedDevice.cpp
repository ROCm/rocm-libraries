// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

#include "device/ScopedDevice.hpp"

#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

namespace hip_kernel_provider::device
{
namespace
{

/// Compared across a scope rather than against a literal, so the suite does not assume which
/// device it was started on.
int currentDevice()
{
    int ordinal = -1;
    if(hipGetDevice(&ordinal) != hipSuccess)
    {
        return -1;
    }
    return ordinal;
}

int deviceCount()
{
    int count = 0;
    if(hipGetDeviceCount(&count) != hipSuccess)
    {
        return 0;
    }
    return count;
}

TEST(TestGpuScopedDevice, BindsTheDeviceThatIsAlreadyCurrent)
{
    SKIP_IF_NO_DEVICES();

    const int before = currentDevice();
    ASSERT_GE(before, 0);

    {
        const ScopedDevice binding(before);
        EXPECT_TRUE(binding.bound());
        EXPECT_EQ(currentDevice(), before);
    }

    EXPECT_EQ(currentDevice(), before);
}

TEST(TestGpuScopedDevice, PutsBackThePreviousDeviceAfterASwitch)
{
    if(deviceCount() < 2)
    {
        GTEST_SKIP() << "needs two devices: with one, ScopedDevice never switches";
    }

    const int before = currentDevice();
    ASSERT_GE(before, 0);
    const int other = (before == 0) ? 1 : 0;

    {
        const ScopedDevice binding(other);
        EXPECT_TRUE(binding.bound());
        EXPECT_EQ(currentDevice(), other);
    }

    EXPECT_EQ(currentDevice(), before);
}

TEST(TestGpuScopedDevice, ReportsARefusedBindAndLeavesTheCurrentDeviceAlone)
{
    SKIP_IF_NO_DEVICES();

    const int before = currentDevice();
    ASSERT_GE(before, 0);

    {
        // One past the last device: the cheapest refusal there is.
        const ScopedDevice binding(deviceCount());
        EXPECT_FALSE(binding.bound());
        EXPECT_EQ(currentDevice(), before);
    }

    EXPECT_EQ(currentDevice(), before);

    // hipSetDevice left an error behind on purpose; clear it so the HipErrorHandler
    // listener does not attribute it to the next test.
    static_cast<void>(hipGetLastError());
    static_cast<void>(hipExtGetLastError());
}

} // namespace
} // namespace hip_kernel_provider::device
