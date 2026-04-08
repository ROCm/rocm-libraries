// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hipdnn_gpu_ref/detail/GpuRefHipError.hpp>

#include <stdexcept>

using namespace hipdnn_gpu_ref;

// ============================================================================
// TestGpuRefHipError — throwOnHipError coverage
// ============================================================================

TEST(TestGpuRefHipError, ThrowsOnError)
{
    EXPECT_THROW(detail::throwOnHipError(hipErrorMemoryAllocation, "test"), std::runtime_error);
}

TEST(TestGpuRefHipError, NoThrowOnSuccess)
{
    EXPECT_NO_THROW(detail::throwOnHipError(hipSuccess, "test"));
}
