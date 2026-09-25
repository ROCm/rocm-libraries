// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only regression test for ROCM-30983 (WSL2/DXG hipblaslt-bench crash).
//
// GetAMDSMIIndex() itself needs a real AMD-SMI/HIP session, which this
// client test binary can't fake. So the NOT_SUPPORTED-tolerance and
// fallback-index policy that fixes the crash is factored into
// decideBdfMatch() / selectFallbackAmdsmiIndex() (efficiency_monitor.hpp) -
// pure functions with no HIP/AMD-SMI device calls - and exercised directly
// here. A regression that deletes the isAmdsmiTelemetryUnavailable() branch
// from decideBdfMatch() and restores an unconditional throw would turn
// smoke_NotSupportedFallsBackInsteadOfThrowing's ReturnIndex into a Throw and
// fail the test.
//
// Smoke tier: PR CI runs hipBLASLt with TEST_TYPE=quick, which selects
// --gtest_filter=*smoke* (test/therock/test_hipblaslt.py). The test names
// below carry the `smoke` token so this fast, host-only guard runs on the PR
// gate, not only in the full/nightly lane.

#include "efficiency_monitor.hpp"

#include <gtest/gtest.h>

// isAmdsmiTelemetryUnavailable()/decideBdfMatch() and the AMD-SMI status enum
// they classify are only declared under !_WIN32 (efficiency_monitor.hpp
// mirrors the platform split already in efficiency_monitor.cpp; AMD-SMI is
// not used on Windows).
#ifndef _WIN32
namespace
{
    constexpr uint64_t kHipPciId   = 0x1234;
    constexpr uint64_t kOtherPciId = 0x5678;

    TEST(EfficiencyMonitorSmoke, smoke_NotSupportedIsTolerated)
    {
        EXPECT_TRUE(isAmdsmiTelemetryUnavailable(AMDSMI_STATUS_NOT_SUPPORTED));
    }

    TEST(EfficiencyMonitorSmoke, smoke_SuccessIsNotTelemetryUnavailable)
    {
        EXPECT_FALSE(isAmdsmiTelemetryUnavailable(AMDSMI_STATUS_SUCCESS));
    }

    TEST(EfficiencyMonitorSmoke, smoke_OtherFailuresStayFatal)
    {
        EXPECT_FALSE(isAmdsmiTelemetryUnavailable(AMDSMI_STATUS_UNKNOWN_ERROR));
    }

    // The actual ROCM-30983 regression check: NOT_SUPPORTED must fall back,
    // not throw.
    TEST(EfficiencyMonitorSmoke, smoke_NotSupportedFallsBackInsteadOfThrowing)
    {
        BdfMatchDecision decision = decideBdfMatch(AMDSMI_STATUS_NOT_SUPPORTED,
                                                    /*smiIndex=*/0,
                                                    /*amdSmiPciId=*/0,
                                                    kHipPciId,
                                                    /*hipDeviceIndex=*/0,
                                                    /*amdsmiDeviceCount=*/1);
        EXPECT_EQ(decision.action, BdfMatchAction::ReturnIndex);
        EXPECT_EQ(decision.index, 0u);
    }

    TEST(EfficiencyMonitorSmoke, smoke_OtherFailuresThrow)
    {
        BdfMatchDecision decision = decideBdfMatch(AMDSMI_STATUS_UNKNOWN_ERROR,
                                                    /*smiIndex=*/0,
                                                    /*amdSmiPciId=*/0,
                                                    kHipPciId,
                                                    /*hipDeviceIndex=*/0,
                                                    /*amdsmiDeviceCount=*/1);
        EXPECT_EQ(decision.action, BdfMatchAction::Throw);
    }

    TEST(EfficiencyMonitorSmoke, smoke_SuccessfulMatchReturnsMatchingIndex)
    {
        BdfMatchDecision decision = decideBdfMatch(AMDSMI_STATUS_SUCCESS,
                                                    /*smiIndex=*/2,
                                                    /*amdSmiPciId=*/kHipPciId,
                                                    kHipPciId,
                                                    /*hipDeviceIndex=*/0,
                                                    /*amdsmiDeviceCount=*/4);
        EXPECT_EQ(decision.action, BdfMatchAction::ReturnIndex);
        EXPECT_EQ(decision.index, 2u);
    }

    TEST(EfficiencyMonitorSmoke, smoke_SuccessfulNonMatchContinuesSearching)
    {
        BdfMatchDecision decision = decideBdfMatch(AMDSMI_STATUS_SUCCESS,
                                                    /*smiIndex=*/0,
                                                    /*amdSmiPciId=*/kOtherPciId,
                                                    kHipPciId,
                                                    /*hipDeviceIndex=*/0,
                                                    /*amdsmiDeviceCount=*/1);
        EXPECT_EQ(decision.action, BdfMatchAction::ContinueSearch);
    }

    // Reviewer concern: a fixed "processor 0" fallback silently attributes
    // telemetry to the wrong GPU whenever the user targets a device other
    // than 0 on a multi-GPU WSL host. The fallback must track
    // hipDeviceIndex instead of ignoring it.
    TEST(EfficiencyMonitorSmoke, smoke_FallbackUsesHipDeviceIndexNotAlwaysZero)
    {
        EXPECT_EQ(selectFallbackAmdsmiIndex(/*hipDeviceIndex=*/2, /*amdsmiDeviceCount=*/4), 2u);
    }

    TEST(EfficiencyMonitorSmoke, smoke_FallbackClampsOutOfRangeDeviceIndex)
    {
        EXPECT_EQ(selectFallbackAmdsmiIndex(/*hipDeviceIndex=*/9, /*amdsmiDeviceCount=*/4), 3u);
    }

    TEST(EfficiencyMonitorSmoke, smoke_FallbackHandlesNegativeDeviceIndex)
    {
        EXPECT_EQ(selectFallbackAmdsmiIndex(/*hipDeviceIndex=*/-1, /*amdsmiDeviceCount=*/4), 0u);
    }
} // namespace
#endif // !_WIN32
