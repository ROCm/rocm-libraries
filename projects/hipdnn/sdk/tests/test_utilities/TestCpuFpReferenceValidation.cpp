// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "Helpers.hpp"

#include <gtest/gtest.h>

#include <hipdnn_sdk/test_utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>
#include <hipdnn_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_sdk/utilities/UtilsFp16.hpp>

using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::utilities;
using namespace helpers;

TEST(TestCpuFpReferenceValidationBfp16, BasicUsage)
{
    CpuFpReferenceValidation<hip_bfloat16> refValidation;

    auto buffer1 = createBuffer<hip_bfloat16>(100, 1.0_bf);
    auto buffer2 = createBuffer<hip_bfloat16>(100, 1.0_bf);

    EXPECT_TRUE(refValidation.allClose(buffer1, buffer2));
}

TEST(TestCpuFpReferenceValidationFp16, BasicUsage)
{
    CpuFpReferenceValidation<half> refValidation;

    auto buffer1 = createBuffer<half>(100, 1.0_h);
    auto buffer2 = createBuffer<half>(100, 1.0_h);

    EXPECT_TRUE(refValidation.allClose(buffer1, buffer2));
}

TEST(TestCpuFpReferenceValidationFp32, BasicUsage)
{
    CpuFpReferenceValidation<float> refValidation;

    auto buffer1 = createBuffer<float>(100, 1.0f);
    auto buffer2 = createBuffer<float>(100, 1.0f);

    EXPECT_TRUE(refValidation.allClose(buffer1, buffer2));
}

TEST(TestCpuFpReferenceValidationFp64, BasicUsage)
{
    CpuFpReferenceValidation<double> refValidation;

    auto buffer1 = createBuffer<double>(100, 1.0);
    auto buffer2 = createBuffer<double>(100, 1.0);

    EXPECT_TRUE(refValidation.allClose(buffer1, buffer2));
}

TEST(TestCpuFpReferenceValidationBfp16, NotComparable)
{
    CpuFpReferenceValidation<hip_bfloat16> refValidation;

    auto buffer1 = createBuffer<hip_bfloat16>(100, 1.0_bf);
    auto buffer2 = createBuffer<hip_bfloat16>(100, 2.0_bf);

    EXPECT_FALSE(refValidation.allClose(buffer1, buffer2));
}

TEST(TestCpuFpReferenceValidationFp16, NotComparable)
{
    CpuFpReferenceValidation<half> refValidation;

    auto buffer1 = createBuffer<half>(100, 1.0_h);
    auto buffer2 = createBuffer<half>(100, 2.0_h);

    EXPECT_FALSE(refValidation.allClose(buffer1, buffer2));
}

TEST(TestCpuFpReferenceValidationFp32, NotComparable)
{
    CpuFpReferenceValidation<float> refValidation;

    auto buffer1 = createBuffer<float>(100, 1.0f);
    auto buffer2 = createBuffer<float>(100, 2.0f);

    EXPECT_FALSE(refValidation.allClose(buffer1, buffer2));
}

TEST(TestCpuFpReferenceValidationFp64, NotComparable)
{
    CpuFpReferenceValidation<double> refValidation;

    auto buffer1 = createBuffer<double>(100, 1.0);
    auto buffer2 = createBuffer<double>(100, 2.0);

    EXPECT_FALSE(refValidation.allClose(buffer1, buffer2));
}

TEST(TestCpuFpReferenceValidation, ToleranceComparison)
{
    CpuFpReferenceValidation<double> refValidationLowTolerance(1e-7, 1e-7);
    CpuFpReferenceValidation<double> refValidationHighTolerance(1e-5, 1e-5);

    auto buffer1 = createBuffer<double>(100, 1.0);
    auto buffer2 = createBuffer<double>(100, 1.000001);

    // Set a very small tolerance
    EXPECT_TRUE(refValidationHighTolerance.allClose(buffer1, buffer2));

    // Change the tolerance to a larger value
    EXPECT_FALSE(refValidationLowTolerance.allClose(buffer1, buffer2));
}

TEST(TestCpuFpReferenceValidation, DefaultTolerance)
{
    CpuFpReferenceValidation<float> refValidation;

    MigratableMemory<float> buffer1(1);
    MigratableMemory<float> buffer2(1);

    buffer1.hostData()[0] = 1.0f;
    buffer2.hostData()[0] = 1.0f + std::numeric_limits<float>::epsilon();

    EXPECT_TRUE(refValidation.allClose(buffer1, buffer2));
}

TEST(TestCpuFpReferenceValidation, NegativeToleranceThrows)
{
    EXPECT_THROW(CpuFpReferenceValidation<float> refValidation(-1e-5f), std::invalid_argument);
}
