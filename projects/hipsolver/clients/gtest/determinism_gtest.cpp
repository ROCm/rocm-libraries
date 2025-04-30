/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

class checkin_misc_DETERMINISM : public ::testing::Test
{
protected:
    checkin_misc_DETERMINISM() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_F(checkin_misc_DETERMINISM, normal_execution)
{
    hipsolver_local_handle       handle;
    hipsolverDeterministicMode_t mode;

    EXPECT_ROCBLAS_STATUS(
        hipsolverSetDeterministicMode(handle, HIPSOLVER_ALLOW_NON_DETERMINISTIC_RESULTS),
        HIPSOLVER_STATUS_SUCCESS);

    hipsolverStatus_t stat = hipsolverGetDeterministicMode(handle, &mode);
    EXPECT_ROCBLAS_STATUS(stat, HIPSOLVER_STATUS_SUCCESS);
    if(stat != HIPSOLVER_STATUS_NOT_SUPPORTED)
        EXPECT_EQ(mode, HIPSOLVER_ALLOW_NON_DETERMINISTIC_RESULTS);

    EXPECT_ROCBLAS_STATUS(hipsolverSetDeterministicMode(handle, HIPSOLVER_DETERMINISTIC_RESULTS),
                          HIPSOLVER_STATUS_SUCCESS);

    stat = hipsolverGetDeterministicMode(handle, &mode);
    EXPECT_ROCBLAS_STATUS(stat, HIPSOLVER_STATUS_SUCCESS);
    if(stat != HIPSOLVER_STATUS_NOT_SUPPORTED)
        EXPECT_EQ(mode, HIPSOLVER_DETERMINISTIC_RESULTS);
}

TEST_F(checkin_misc_DETERMINISM, get_null_handle)
{
    hipsolverDeterministicMode_t mode;

    EXPECT_ROCBLAS_STATUS(hipsolverGetDeterministicMode(nullptr, &mode),
                          HIPSOLVER_STATUS_NOT_INITIALIZED);
}

TEST_F(checkin_misc_DETERMINISM, get_null_mode)
{
    hipsolver_local_handle handle;

    EXPECT_ROCBLAS_STATUS(hipsolverGetDeterministicMode(handle, nullptr),
                          HIPSOLVER_STATUS_INVALID_VALUE);
}

TEST_F(checkin_misc_DETERMINISM, set_null_handle)
{
    EXPECT_ROCBLAS_STATUS(hipsolverSetDeterministicMode(nullptr, HIPSOLVER_DETERMINISTIC_RESULTS),
                          HIPSOLVER_STATUS_NOT_INITIALIZED);
}

TEST_F(checkin_misc_DETERMINISM, set_bad_mode)
{
    hipsolver_local_handle handle;

    EXPECT_ROCBLAS_STATUS(hipsolverSetDeterministicMode(handle, hipsolverDeterministicMode_t(-1)),
                          HIPSOLVER_STATUS_INVALID_ENUM);
}
