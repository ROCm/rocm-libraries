// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipBuffer.hpp"

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <memory>

TEST(DeviceVectorGuard, smoke_DetectsZeroStoreInWideGuard)
{
    struct RestoreGuardPad
    {
        std::size_t previous = g_DVEC_PAD;
        ~RestoreGuardPad()
        {
            g_DVEC_PAD = previous;
        }
    } restore;
    g_DVEC_PAD = 2048;

    auto device = std::make_unique<HipDeviceBuffer>(HIP_R_64F, 1);
    ASSERT_EQ(device->memcheck(), hipSuccess);

    const double input = 17.0;
    double       observed{};
    ASSERT_EQ(hipMemcpy(device->buf(), &input, sizeof(input), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(&observed, device->buf(), sizeof(observed), hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_EQ(observed, input);

    // This byte is inside the 16384-byte trailing guard, beyond its first 8192 bytes.
    auto* guardByte = static_cast<char*>(device->buf()) + device->getNumBytes() + 8192;
    ASSERT_EQ(hipMemset(guardByte, 0, 1), hipSuccess);
    EXPECT_NONFATAL_FAILURE(device.reset(), "memcmp");
}
