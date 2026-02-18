// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

namespace hipdnn_test_sdk::utilities
{

/// @brief Test event listener that ensures HIP errors are cleaned up after every test
///
/// This listener is registered with Google Test and automatically checks for HIP errors
/// after each test completes. If a test generates HIP errors but doesn't clean them up,
/// those errors will be flagged as test failures, preventing them from affecting subsequent
/// tests. This eliminates flaky test behavior caused by error propagation between tests.
class HipErrorHandler : public testing::EmptyTestEventListener
{
public:
    /// @brief Called after each test completes
    /// @param testInfo Information about the test that just completed
    void OnTestEnd(const testing::TestInfo& testInfo) override
    {
        // Check and clear standard HIP error state
        auto hipError = hipGetLastError();

        // Special case: hipErrorNoDevice cannot be cleared and persists even after
        // hipGetLastError() is called. In non-GPU environments, we should not fail
        // the test for this specific error.
        if(hipError == hipErrorNoDevice)
        {
            return;
        }

        // Check and clear extended HIP error state
        auto hipExtError = hipExtGetLastError();

        // If there were any errors, fail the test that generated them
        EXPECT_EQ(hipError, hipSuccess)
            << " hipGetLastError returned error code " << hipError << " after test "
            << testInfo.test_suite_name() << "." << testInfo.name()
            << ". Error string: " << hipGetErrorString(hipError);

        EXPECT_EQ(hipExtError, hipSuccess)
            << " hipExtGetLastError returned error code " << hipExtError << " after test "
            << testInfo.test_suite_name() << "." << testInfo.name()
            << ". Error string: " << hipGetErrorString(hipExtError);
    }
};

} // namespace hipdnn_data_sdk::test_utilities
