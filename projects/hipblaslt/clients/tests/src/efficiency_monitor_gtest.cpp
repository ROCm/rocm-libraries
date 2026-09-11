// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only smoke test: AMD-SMI status classification for optional telemetry
// (ROCM-30983). Pure logic, no HIP/AMDSMI device calls.

#include "efficiency_monitor.hpp"

#include <gtest/gtest.h>

// isAmdsmiTelemetryUnavailable() and the AMD-SMI status enum it classifies are
// only declared under !_WIN32 (efficiency_monitor.hpp mirrors the platform
// split already in efficiency_monitor.cpp; AMD-SMI is not used on Windows).
#ifndef _WIN32
namespace
{
    TEST(EfficiencyMonitorSmoke, NotSupportedIsTolerated)
    {
        EXPECT_TRUE(isAmdsmiTelemetryUnavailable(AMDSMI_STATUS_NOT_SUPPORTED));
    }

    TEST(EfficiencyMonitorSmoke, SuccessIsNotTelemetryUnavailable)
    {
        EXPECT_FALSE(isAmdsmiTelemetryUnavailable(AMDSMI_STATUS_SUCCESS));
    }

    TEST(EfficiencyMonitorSmoke, OtherFailuresStayFatal)
    {
        EXPECT_FALSE(isAmdsmiTelemetryUnavailable(AMDSMI_STATUS_UNKNOWN_ERROR));
    }
} // namespace
#endif // !_WIN32
