/*******************************************************************************
 *
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 *******************************************************************************/

// StreamK=5 API-level tests.
//
// Tests that exercise the HIPBLASLT_MATMUL_DESC_STREAMK_TILE_SCHEDULING_EXT
// attribute at the API layer. Device-library-dependent numerics tests will
// be added in a follow-up PR that integrates SK5 kernels into the library.

#include <gtest/gtest.h>
#include <hipblaslt/hipblaslt.h>

namespace
{
    inline bool gpuAvailable()
    {
        int deviceCount = 0;
        return hipGetDeviceCount(&deviceCount) == hipSuccess && deviceCount > 0;
    }

    TEST(StreamK5HybridApi, AttributeRoundTrip)
    {
        if(!gpuAvailable())
            GTEST_SKIP() << "No GPU available for hipblasLt handle";

        hipblasLtHandle_t handle = nullptr;
        ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

        hipblasLtMatmulDesc_t desc = nullptr;
        ASSERT_EQ(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                  HIPBLAS_STATUS_SUCCESS);

        const int32_t want = 1;
        ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                      desc, HIPBLASLT_MATMUL_DESC_STREAMK_TILE_SCHEDULING_EXT, &want, sizeof(want)),
                  HIPBLAS_STATUS_SUCCESS);

        int32_t got      = -1;
        size_t  writtenN = 0;
        ASSERT_EQ(hipblasLtMatmulDescGetAttribute(desc,
                                                  HIPBLASLT_MATMUL_DESC_STREAMK_TILE_SCHEDULING_EXT,
                                                  &got,
                                                  sizeof(got),
                                                  &writtenN),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(got, want);
        EXPECT_EQ(writtenN, sizeof(int32_t));

        const int32_t want0 = 0;
        ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                      desc, HIPBLASLT_MATMUL_DESC_STREAMK_TILE_SCHEDULING_EXT, &want0, sizeof(want0)),
                  HIPBLAS_STATUS_SUCCESS);
        got = -1;
        ASSERT_EQ(hipblasLtMatmulDescGetAttribute(desc,
                                                  HIPBLASLT_MATMUL_DESC_STREAMK_TILE_SCHEDULING_EXT,
                                                  &got,
                                                  sizeof(got),
                                                  &writtenN),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(got, want0);

        const int32_t want2 = 2;
        ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                      desc, HIPBLASLT_MATMUL_DESC_STREAMK_TILE_SCHEDULING_EXT, &want2, sizeof(want2)),
                  HIPBLAS_STATUS_SUCCESS);
        got = -1;
        ASSERT_EQ(hipblasLtMatmulDescGetAttribute(desc,
                                                  HIPBLASLT_MATMUL_DESC_STREAMK_TILE_SCHEDULING_EXT,
                                                  &got,
                                                  sizeof(got),
                                                  &writtenN),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(got, want2);

        const int32_t bad = 3;
        EXPECT_EQ(hipblasLtMatmulDescSetAttribute(
                      desc, HIPBLASLT_MATMUL_DESC_STREAMK_TILE_SCHEDULING_EXT, &bad, sizeof(bad)),
                  HIPBLAS_STATUS_INVALID_VALUE);
        const int32_t neg = -1;
        EXPECT_EQ(hipblasLtMatmulDescSetAttribute(
                      desc, HIPBLASLT_MATMUL_DESC_STREAMK_TILE_SCHEDULING_EXT, &neg, sizeof(neg)),
                  HIPBLAS_STATUS_INVALID_VALUE);

        (void)hipblasLtMatmulDescDestroy(desc);
        (void)hipblasLtDestroy(handle);
    }

} // namespace
