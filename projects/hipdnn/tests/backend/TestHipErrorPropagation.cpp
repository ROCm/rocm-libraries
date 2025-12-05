// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

// This test suite demonstrates the HIP error propagation issue.
// Without proper error handling, errors from one test can affect subsequent tests.

TEST(HipErrorPropagation, GenerateHipError)
{
    // Intentionally generate a HIP error by trying to set an invalid device
    // This simulates a test that generates an error but doesn't clean it up

    // Use -1 as device ID, which is guaranteed to be invalid since device IDs
    // must be non-negative. This will generate hipErrorInvalidDevice.
    auto error = hipSetDevice(-1);

    // We know this will fail, but we're not clearing the error
    // In a real scenario, this could be an accidental error in the test
    EXPECT_NE(error, hipSuccess);

    // Note: We are NOT calling hipGetLastError() to clear the error
    // This is intentional to demonstrate the problem
}

TEST(HipErrorPropagation, ExpectErrorPropagation)
{
    // WITHOUT HipErrorHandler: This test PASSES (finds expected error from previous test)
    // WITH HipErrorHandler: This test FAILS (handler clears errors, so no error found)

    // Check if there are any HIP errors present
    auto hipError = hipGetLastError();

    // We EXPECT to find the error from the previous test (without HipErrorHandler)
    EXPECT_NE(hipError, hipSuccess)
        << "Should have found uncaught HIP error from previous test (error propagation)";

    // Verify it's the expected error
    if(hipError != hipSuccess)
    {
        EXPECT_STREQ(hipGetErrorString(hipError), "invalid device ordinal");
    }
}

TEST(HipErrorPropagation, ExpectExtErrorPropagation)
{
    // WITHOUT HipErrorHandler: This test PASSES (finds expected ext error from first test)
    // WITH HipErrorHandler: This test FAILS (handler clears errors, so no error found)

    auto hipExtError = hipExtGetLastError();

    // We EXPECT to find the ext error from the first test (without HipErrorHandler)
    EXPECT_NE(hipExtError, hipSuccess)
        << "Should have found uncaught HIP ext error from first test (error propagation)";

    // Verify it's the expected error
    if(hipExtError != hipSuccess)
    {
        EXPECT_STREQ(hipGetErrorString(hipExtError), "invalid device ordinal");
    }
}
