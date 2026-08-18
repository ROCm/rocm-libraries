/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *
 * ************************************************************************ */

#include "clientcommon.hpp"

using ::testing::Matcher;
using ::testing::MatchesRegex;
using ::testing::UnitTest;

// The floating-point emulation configuration API forwards to cuSOLVER on the
// NVIDIA backend and returns HIPSOLVER_STATUS_NOT_SUPPORTED on rocSOLVER. Where
// a getter can round-trip, we only assert the value if the backend supports it.

class checkin_misc_EMULATION : public ::testing::Test
{
protected:
    checkin_misc_EMULATION() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_F(checkin_misc_EMULATION, math_mode)
{
    hipsolver_local_handle handle;
    hipsolverMathMode_t    mode;

    hipsolverStatus_t stat = hipsolverSetMathMode(handle, HIPSOLVER_FP32_EMULATED_BF16X9_MATH);
    if(stat != HIPSOLVER_STATUS_NOT_SUPPORTED)
    {
        EXPECT_ROCBLAS_STATUS(stat, HIPSOLVER_STATUS_SUCCESS);

        stat = hipsolverGetMathMode(handle, &mode);
        EXPECT_ROCBLAS_STATUS(stat, HIPSOLVER_STATUS_SUCCESS);
        EXPECT_EQ(mode, HIPSOLVER_FP32_EMULATED_BF16X9_MATH);
    }
}

TEST_F(checkin_misc_EMULATION, emulation_strategy)
{
    hipsolver_local_handle       handle;
    hipsolverEmulationStrategy_t strategy;

    hipsolverStatus_t stat
        = hipsolverSetEmulationStrategy(handle, HIPSOLVER_EMULATION_STRATEGY_PERFORMANT);
    if(stat != HIPSOLVER_STATUS_NOT_SUPPORTED)
    {
        EXPECT_ROCBLAS_STATUS(stat, HIPSOLVER_STATUS_SUCCESS);

        stat = hipsolverGetEmulationStrategy(handle, &strategy);
        EXPECT_ROCBLAS_STATUS(stat, HIPSOLVER_STATUS_SUCCESS);
        EXPECT_EQ(strategy, HIPSOLVER_EMULATION_STRATEGY_PERFORMANT);
    }
}

TEST_F(checkin_misc_EMULATION, mantissa_control)
{
    hipsolver_local_handle              handle;
    hipsolverEmulationMantissaControl_t control;

    hipsolverStatus_t stat = hipsolverSetFixedPointEmulationMantissaControl(
        handle, HIPSOLVER_EMULATION_MANTISSA_CONTROL_FIXED);
    if(stat != HIPSOLVER_STATUS_NOT_SUPPORTED)
    {
        EXPECT_ROCBLAS_STATUS(stat, HIPSOLVER_STATUS_SUCCESS);

        stat = hipsolverGetFixedPointEmulationMantissaControl(handle, &control);
        EXPECT_ROCBLAS_STATUS(stat, HIPSOLVER_STATUS_SUCCESS);
        EXPECT_EQ(control, HIPSOLVER_EMULATION_MANTISSA_CONTROL_FIXED);
    }
}

TEST_F(checkin_misc_EMULATION, mantissa_bit_count)
{
    hipsolver_local_handle handle;
    int                    bits = 0;

    hipsolverStatus_t stat = hipsolverSetFixedPointEmulationMaxMantissaBitCount(handle, 24);
    if(stat != HIPSOLVER_STATUS_NOT_SUPPORTED)
    {
        EXPECT_ROCBLAS_STATUS(stat, HIPSOLVER_STATUS_SUCCESS);

        stat = hipsolverGetFixedPointEmulationMaxMantissaBitCount(handle, &bits);
        EXPECT_ROCBLAS_STATUS(stat, HIPSOLVER_STATUS_SUCCESS);
    }
}

TEST_F(checkin_misc_EMULATION, mantissa_bit_offset)
{
    hipsolver_local_handle handle;
    int                    offset = 0;

    hipsolverStatus_t stat = hipsolverSetFixedPointEmulationMantissaBitOffset(handle, 0);
    if(stat != HIPSOLVER_STATUS_NOT_SUPPORTED)
    {
        EXPECT_ROCBLAS_STATUS(stat, HIPSOLVER_STATUS_SUCCESS);

        stat = hipsolverGetFixedPointEmulationMantissaBitOffset(handle, &offset);
        EXPECT_ROCBLAS_STATUS(stat, HIPSOLVER_STATUS_SUCCESS);
    }
}

TEST_F(checkin_misc_EMULATION, special_values_support)
{
    hipsolver_local_handle                   handle;
    hipsolverEmulationSpecialValuesSupport_t mask;

    hipsolverStatus_t stat = hipsolverSetEmulationSpecialValuesSupport(
        handle, HIPSOLVER_EMULATION_SPECIAL_VALUES_SUPPORT_NONE);
    if(stat != HIPSOLVER_STATUS_NOT_SUPPORTED)
    {
        EXPECT_ROCBLAS_STATUS(stat, HIPSOLVER_STATUS_SUCCESS);

        stat = hipsolverGetEmulationSpecialValuesSupport(handle, &mask);
        EXPECT_ROCBLAS_STATUS(stat, HIPSOLVER_STATUS_SUCCESS);
        EXPECT_EQ(mask, HIPSOLVER_EMULATION_SPECIAL_VALUES_SUPPORT_NONE);
    }
}

TEST_F(checkin_misc_EMULATION, set_null_handle)
{
    EXPECT_ROCBLAS_STATUS(hipsolverSetMathMode(nullptr, HIPSOLVER_DEFAULT_MATH),
                          HIPSOLVER_STATUS_NOT_INITIALIZED);
    EXPECT_ROCBLAS_STATUS(
        hipsolverSetEmulationStrategy(nullptr, HIPSOLVER_EMULATION_STRATEGY_DEFAULT),
        HIPSOLVER_STATUS_NOT_INITIALIZED);
    EXPECT_ROCBLAS_STATUS(hipsolverSetFixedPointEmulationMaxMantissaBitCount(nullptr, 24),
                          HIPSOLVER_STATUS_NOT_INITIALIZED);
}

TEST_F(checkin_misc_EMULATION, get_null_handle)
{
    hipsolverMathMode_t mode;

    EXPECT_ROCBLAS_STATUS(hipsolverGetMathMode(nullptr, &mode), HIPSOLVER_STATUS_NOT_INITIALIZED);
}

TEST_F(checkin_misc_EMULATION, get_null_out)
{
    hipsolver_local_handle handle;

    EXPECT_ROCBLAS_STATUS(hipsolverGetMathMode(handle, nullptr), HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolverGetEmulationStrategy(handle, nullptr),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolverGetFixedPointEmulationMaxMantissaBitCount(handle, nullptr),
                          HIPSOLVER_STATUS_INVALID_VALUE);
}
