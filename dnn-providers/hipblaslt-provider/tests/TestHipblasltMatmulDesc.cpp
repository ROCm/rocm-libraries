// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HipblasltMatmulDesc.hpp"
#include <gtest/gtest.h>
#include <hipblaslt/hipblaslt.h>

using namespace hipblaslt_plugin;

TEST(TestHipblasltMatmulDesc, CanCreateAndDestroy)
{
    EXPECT_NO_THROW({
        HipblasltMatmulDesc desc(HIPBLAS_OP_N, HIPBLAS_OP_N, HIPBLAS_COMPUTE_32F, HIP_R_32F);
        EXPECT_NE(desc.matmulDesc(), nullptr);
    });
}
