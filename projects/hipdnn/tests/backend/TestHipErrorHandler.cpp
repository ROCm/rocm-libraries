// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

// This test suite demonstrates HipErrorHandler's role in preventing error propagation.
// HipErrorHandler is registered as a test event listener that runs AFTER each test.
// Its job is to verify tests properly clean up their HIP errors.

TEST(TestGpuHipErrorHandler, GenerateHipError)
{
    // Generate a HIP error by trying to set an invalid device
    // This simulates real test scenarios where HIP API calls may fail
    auto error = hipSetDevice(-1); // -1 is guaranteed invalid
    EXPECT_NE(error, hipSuccess);

    // BEST PRACTICE: Tests should clean up errors they generate
    // This prevents errors from affecting subsequent tests
    // We must clear BOTH error states since HIP maintains two separate error queues
    std::ignore = hipGetLastError(); // Clear regular error state
    std::ignore = hipExtGetLastError(); // Clear extended error state

    // AFTER this test completes, HipErrorHandler::OnTestEnd() will:
    // 1. Call hipGetLastError() - will return hipSuccess (we cleaned up)
    // 2. Call hipExtGetLastError() - will return hipSuccess (we cleaned up)
    // 3. EXPECT both to be hipSuccess - test PASSES
    //
    // The handler acts as a safety net: if we forgot to clean up, it would
    // FAIL this test, preventing error propagation to subsequent tests.
}

TEST(TestGpuHipErrorHandler, NoErrorPropagation)
{
    // Verify that errors from the previous test do NOT propagate here
    // WITH HipErrorHandler: This test PASSES (no propagation)
    // WITHOUT HipErrorHandler: Would fail if previous test didn't clean up

    auto hipError = hipGetLastError();
    EXPECT_EQ(hipError, hipSuccess) << "No errors should propagate from previous test";
}

TEST(TestGpuHipErrorHandler, NoExtErrorPropagation)
{
    // Verify that extended errors from the first test do NOT propagate here
    auto hipExtError = hipExtGetLastError();
    EXPECT_EQ(hipExtError, hipSuccess) << "No extended errors should propagate from previous tests";
}
