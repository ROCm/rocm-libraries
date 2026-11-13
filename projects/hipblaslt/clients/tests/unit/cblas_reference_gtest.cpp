// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "cblas_interface.hpp"
#include "hipblaslt_arguments.hpp"

#include <gtest/gtest.h>

#include <vector>

// The FP64 CPU reference must compute in double. The small-size path used to accumulate in float,
// which is ~1e-8 relative error with non-integer data and alpha. Allow only double rounding
// differences (operation order), K = 1, beta = 0: D = alpha * a * b.
TEST(HostUnitCblasReference, smoke_Fp64GemmKeepsDoublePrecision)
{
    constexpr int64_t   m = 4, n = 3, k = 1;
    std::vector<double> A(m * k), B(k * n), D(m * n, 0.0);
    for(int64_t i = 0; i < m; i++)
        A[i] = 0.1 + 1e-3 * double(i);
    for(int64_t j = 0; j < n; j++)
        B[j] = 0.3 - 1e-3 * double(j);
    const double alphaValue = 0.1;

    computeTypeInterface alpha{}, beta{}, scale{};
    set_computeInterface(alpha, alphaValue, HIP_R_64F, HIP_R_64F);
    set_computeInterface(beta, 0.0, HIP_R_64F, HIP_R_64F);
    set_computeInterface(scale, 1.0, HIP_R_64F, HIP_R_64F);

    cblas_gemm(HIPBLAS_OP_N,
               HIPBLAS_OP_N,
               m,
               n,
               k,
               alpha,
               A.data(),
               m,
               B.data(),
               k,
               beta,
               D.data(),
               m,
               nullptr,
               &scale,
               &scale,
               &scale,
               false,
               false,
               HIP_R_64F,
               HIP_R_64F,
               HIP_R_64F,
               HIP_R_64F,
               HIP_R_64F,
               HIP_R_64F);

    for(int64_t j = 0; j < n; j++)
        for(int64_t i = 0; i < m; i++)
            EXPECT_DOUBLE_EQ(D[i + j * m], alphaValue * A[i] * B[j]) << "i=" << i << " j=" << j;
}
