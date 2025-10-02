// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "Helpers.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceMiopenRmsValidation.hpp>
#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>
#include <hipdnn_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_sdk/utilities/UtilsFp16.hpp>
#include <limits>

using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::utilities;
using namespace helpers;

TEST(TestCpuFpReferenceMiopenRmsValidationBfp16, BasicUsage)
{
    CpuFpReferenceMiopenRmsValidation<hip_bfloat16> refValidation;

    auto buffer1 = createBuffer<hip_bfloat16>(100, 1.0_bf);
    auto buffer2 = createBuffer<hip_bfloat16>(100, 1.0_bf);

    EXPECT_TRUE(refValidation.allClose(buffer1, buffer2));
}

TEST(TestCpuFpReferenceMiopenRmsValidationFp16, BasicUsage)
{
    CpuFpReferenceMiopenRmsValidation<half> refValidation;

    auto buffer1 = createBuffer<half>(100, 1.0_h);
    auto buffer2 = createBuffer<half>(100, 1.0_h);

    EXPECT_TRUE(refValidation.allClose(buffer1, buffer2));
}

TEST(TestCpuFpReferenceMiopenRmsValidationFp32, BasicUsage)
{
    CpuFpReferenceMiopenRmsValidation<float> refValidation;

    auto buffer1 = createBuffer<float>(100, 1.0f);
    auto buffer2 = createBuffer<float>(100, 1.0f);

    EXPECT_TRUE(refValidation.allClose(buffer1, buffer2));
}

TEST(TestCpuFpReferenceMiopenRmsValidationFp64, BasicUsage)
{
    CpuFpReferenceMiopenRmsValidation<double> refValidation;

    auto buffer1 = createBuffer<double>(100, 1.0);
    auto buffer2 = createBuffer<double>(100, 1.0);

    EXPECT_TRUE(refValidation.allClose(buffer1, buffer2));
}

TEST(TestCpuFpReferenceMiopenRmsValidationBfp16, NotComparable)
{
    CpuFpReferenceMiopenRmsValidation<hip_bfloat16> refValidation;

    auto buffer1 = createBuffer<hip_bfloat16>(100, 1.0_bf);
    auto buffer2 = createBuffer<hip_bfloat16>(100, 2.0_bf);

    EXPECT_FALSE(refValidation.allClose(buffer1, buffer2));
}

TEST(TestCpuFpReferenceMiopenRmsValidationFp16, NotComparable)
{
    CpuFpReferenceMiopenRmsValidation<half> refValidation;

    auto buffer1 = createBuffer<half>(100, 1.0_h);
    auto buffer2 = createBuffer<half>(100, 2.0_h);

    EXPECT_FALSE(refValidation.allClose(buffer1, buffer2));
}

TEST(TestCpuFpReferenceMiopenRmsValidationFp32, NotComparable)
{
    CpuFpReferenceMiopenRmsValidation<float> refValidation;

    auto buffer1 = createBuffer<float>(100, 1.0f);
    auto buffer2 = createBuffer<float>(100, 2.0f);

    EXPECT_FALSE(refValidation.allClose(buffer1, buffer2));
}

TEST(TestCpuFpReferenceMiopenRmsValidationFp64, NotComparable)
{
    CpuFpReferenceMiopenRmsValidation<double> refValidation;

    auto buffer1 = createBuffer<double>(100, 1.0);
    auto buffer2 = createBuffer<double>(100, 2.0);

    EXPECT_FALSE(refValidation.allClose(buffer1, buffer2));
}

TEST(TestCpuFpReferenceMiopenRmsValidation, ToleranceComparison)
{
    CpuFpReferenceMiopenRmsValidation<double> refValidationLowTolerance(1e-7);
    CpuFpReferenceMiopenRmsValidation<double> refValidationHighTolerance(1e-3);

    // Create buffers with small differences
    auto buffer1 = createBuffer<double>(100, 1.0);
    auto buffer2 = createBuffer<double>(100, 1.0001);

    // High tolerance should pass
    EXPECT_TRUE(refValidationHighTolerance.allClose(buffer1, buffer2));

    // Low tolerance should fail
    EXPECT_FALSE(refValidationLowTolerance.allClose(buffer1, buffer2));
}

TEST(TestCpuFpReferenceMiopenRmsValidation, EmptyBuffers)
{
    CpuFpReferenceMiopenRmsValidation<float> refValidation;

    MigratableMemory<float> buffer1(0);
    MigratableMemory<float> buffer2(0);

    EXPECT_TRUE(refValidation.allClose(buffer1, buffer2));
}

TEST(TestCpuFpReferenceMiopenRmsValidation, DifferentSizeBuffers)
{
    CpuFpReferenceMiopenRmsValidation<float> refValidation;

    auto buffer1 = createBuffer<float>(100, 1.0f);
    auto buffer2 = createBuffer<float>(50, 1.0f);

    EXPECT_FALSE(refValidation.allClose(buffer1, buffer2));
}

// Test MIOpen-specific RMS calculation behavior
TEST(TestCpuFpReferenceMiopenRmsValidation, MiopenRmsCalculation)
{
    // Test that RMS error is calculated correctly
    CpuFpReferenceMiopenRmsValidation<double> refValidation(0.1);

    MigratableMemory<double> buffer1(4);
    MigratableMemory<double> buffer2(4);

    double* data1 = buffer1.hostData();
    double* data2 = buffer2.hostData();

    // Set up test data
    data1[0] = 1.0;
    data1[1] = 2.0;
    data1[2] = 3.0;
    data1[3] = 4.0;

    data2[0] = 1.1;
    data2[1] = 2.1;
    data2[2] = 3.1;
    data2[3] = 4.1;

    // Expected RMS calculation:
    // Square differences: 0.01, 0.01, 0.01, 0.01
    // Sum of square differences: 0.04
    // sqrt(0.04) = 0.2
    // Max magnitude in either buffer: 4.1
    // Element count: 4
    // Relative RMS error = 0.2 / (sqrt(4) * 4.1) = 0.2 / (2 * 4.1) = 0.0244

    EXPECT_TRUE(refValidation.allClose(buffer1, buffer2)); // 0.0244 < 0.1

    // Now with tighter tolerance it should fail
    CpuFpReferenceMiopenRmsValidation<double> refValidationTight(0.02);
    EXPECT_FALSE(refValidationTight.allClose(buffer1, buffer2)); // 0.0244 > 0.02
}

TEST(TestCpuFpReferenceMiopenRmsValidation, DefaultTolerance)
{
    CpuFpReferenceMiopenRmsValidation<float> refValidation;

    MigratableMemory<float> buffer1(1);
    MigratableMemory<float> buffer2(1);

    buffer1.hostData()[0] = 1.0f;
    buffer2.hostData()[0] = 1.0f + std::numeric_limits<float>::epsilon();

    EXPECT_TRUE(refValidation.allClose(buffer1, buffer2));
}

TEST(TestCpuFpReferenceMiopenRmsValidation, NegativeToleranceThrows)
{
    EXPECT_THROW(CpuFpReferenceMiopenRmsValidation<float> refValidation(-1e-5f),
                 std::invalid_argument);
}
