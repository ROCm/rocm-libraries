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

class checkin_misc_PARAMS : public ::testing::Test
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }
};

TEST_F(checkin_misc_PARAMS, normal_execution)
{
    hipsolverDnParams_t params = hipsolverDnParams_t();

    EXPECT_ROCBLAS_STATUS(hipsolverDnCreateParams(&params), HIPSOLVER_STATUS_SUCCESS);

    EXPECT_ROCBLAS_STATUS(hipsolverDnSetAdvOptions(params, HIPSOLVERDN_GETRF, HIPSOLVER_ALG_0),
                          HIPSOLVER_STATUS_SUCCESS);

    EXPECT_ROCBLAS_STATUS(hipsolverDnDestroyParams(params), HIPSOLVER_STATUS_SUCCESS);
}

TEST_F(checkin_misc_PARAMS, create_nullptr)
{
    EXPECT_ROCBLAS_STATUS(hipsolverDnCreateParams(nullptr), HIPSOLVER_STATUS_INVALID_VALUE);
}

TEST_F(checkin_misc_PARAMS, setoptions_nullptr)
{
    hipsolverDnParams_t params = hipsolverDnParams_t();

    EXPECT_ROCBLAS_STATUS(hipsolverDnSetAdvOptions(params, HIPSOLVERDN_GETRF, HIPSOLVER_ALG_0),
                          HIPSOLVER_STATUS_INVALID_VALUE);
}

TEST_F(checkin_misc_PARAMS, setoptions_bad_function)
{
    hipsolverDnParams_t params = hipsolverDnParams_t();

    EXPECT_ROCBLAS_STATUS(hipsolverDnCreateParams(&params), HIPSOLVER_STATUS_SUCCESS);

    EXPECT_ROCBLAS_STATUS(
        hipsolverDnSetAdvOptions(params, hipsolverDnFunction_t(-1), HIPSOLVER_ALG_0),
        HIPSOLVER_STATUS_INVALID_ENUM);

    EXPECT_ROCBLAS_STATUS(hipsolverDnDestroyParams(params), HIPSOLVER_STATUS_SUCCESS);
}

TEST_F(checkin_misc_PARAMS, setoptions_bad_algmode)
{
    hipsolverDnParams_t params = hipsolverDnParams_t();

    EXPECT_ROCBLAS_STATUS(hipsolverDnCreateParams(&params), HIPSOLVER_STATUS_SUCCESS);

    EXPECT_ROCBLAS_STATUS(
        hipsolverDnSetAdvOptions(params, HIPSOLVERDN_GETRF, hipsolverAlgMode_t(-1)),
        HIPSOLVER_STATUS_INVALID_ENUM);

    EXPECT_ROCBLAS_STATUS(hipsolverDnDestroyParams(params), HIPSOLVER_STATUS_SUCCESS);
}

TEST_F(checkin_misc_PARAMS, destroy_nullptr)
{
    hipsolverDnParams_t params = hipsolverDnParams_t();

    EXPECT_ROCBLAS_STATUS(hipsolverDnDestroyParams(params), HIPSOLVER_STATUS_INVALID_VALUE);
}
