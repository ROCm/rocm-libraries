// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// CONTRACT TESTS for experimental feature DEMO.
//
// These tests define the promised behavior of the feature. Adding a test is free. Changing
// or deleting one is a breaking change: tools/scripts/check_experimental_api.py fails the PR
// unless it also bumps HIPBLASLT_EXPERIMENTAL_API_VERSION.

#define HIPBLASLT_ENABLE_EXPERIMENTAL_API HIPBLASLT_EXPERIMENTAL_API_VERSION
#include <hipblaslt/hipblaslt-experimental.h>

#include <gtest/gtest.h>

#include <limits>

namespace
{
    TEST(ExperimentalContractDemo, RoundsDownToMultiple)
    {
        size_t r = 0;
        EXPECT_EQ(hipblasLtExperimentalDemoAlignSize(100, 64, HIPBLASLT_EXPERIMENTAL_DEMO_ROUND_DOWN, &r),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(r, 64u);
    }

    TEST(ExperimentalContractDemo, RoundsUpToMultiple)
    {
        size_t r = 0;
        EXPECT_EQ(hipblasLtExperimentalDemoAlignSize(100, 64, HIPBLASLT_EXPERIMENTAL_DEMO_ROUND_UP, &r),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(r, 128u);
    }

    TEST(ExperimentalContractDemo, AlignedValueIsUnchanged)
    {
        for(auto rounding : {HIPBLASLT_EXPERIMENTAL_DEMO_ROUND_DOWN, HIPBLASLT_EXPERIMENTAL_DEMO_ROUND_UP})
        {
            size_t r = 0;
            EXPECT_EQ(hipblasLtExperimentalDemoAlignSize(128, 64, rounding, &r), HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(r, 128u);
        }
    }

    TEST(ExperimentalContractDemo, RejectsNonPowerOfTwoAlignment)
    {
        size_t r = 7;
        for(size_t alignment : {size_t(0), size_t(3), size_t(48)})
        {
            EXPECT_EQ(hipblasLtExperimentalDemoAlignSize(100, alignment, HIPBLASLT_EXPERIMENTAL_DEMO_ROUND_UP, &r),
                      HIPBLAS_STATUS_INVALID_VALUE);
            EXPECT_EQ(r, 7u) << "result must not be written on error";
        }
    }

    TEST(ExperimentalContractDemo, RejectsNullResult)
    {
        EXPECT_EQ(hipblasLtExperimentalDemoAlignSize(100, 64, HIPBLASLT_EXPERIMENTAL_DEMO_ROUND_UP, nullptr),
                  HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST(ExperimentalContractDemo, RejectsOverflowOnRoundUp)
    {
        size_t r = 7;
        EXPECT_EQ(hipblasLtExperimentalDemoAlignSize(
                      std::numeric_limits<size_t>::max(), 64, HIPBLASLT_EXPERIMENTAL_DEMO_ROUND_UP, &r),
                  HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_EQ(r, 7u);
    }
}
