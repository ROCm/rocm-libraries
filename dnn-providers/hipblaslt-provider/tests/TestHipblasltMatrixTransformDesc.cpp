// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HipblasltMatrixTransformDesc.hpp"
#include <gtest/gtest.h>
#include <hipblaslt/hipblaslt.h>

using namespace hipblaslt_plugin;

TEST(TestHipblasltMatrixTransformDesc, CanCreateAndDestroy)
{
    EXPECT_NO_THROW({
        HipblasltMatrixTransformDesc const desc(HIP_R_32F, HIPBLAS_OP_T);
        EXPECT_NE(desc.transformDesc(), nullptr);
    });
}

TEST(TestHipblasltMatrixTransformDesc, SetsTransA)
{
    HipblasltMatrixTransformDesc const desc(HIP_R_32F, HIPBLAS_OP_T);

    hipblasOperation_t transA{};
    size_t sizeWritten = 0;
    EXPECT_EQ(hipblasLtMatrixTransformDescGetAttribute(desc.transformDesc(),
                                                       HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSA,
                                                       static_cast<void*>(&transA),
                                                       sizeof(transA),
                                                       &sizeWritten),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(sizeWritten, sizeof(transA));
    EXPECT_EQ(transA, HIPBLAS_OP_T);
}

TEST(TestHipblasltMatrixTransformDesc, DefaultIsNull)
{
    EXPECT_NO_THROW({
        HipblasltMatrixTransformDesc const desc;
        EXPECT_EQ(desc.transformDesc(), nullptr);
    });
}
