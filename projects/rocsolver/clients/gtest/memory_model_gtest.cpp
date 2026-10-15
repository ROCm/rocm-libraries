/* **************************************************************************
 * Copyright (C) 2020-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#include <algorithm>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

#include "common/misc/client_environment_helpers.hpp"

/*************************************/
/***** Workspace Helper Implicit Tests *****/
/*************************************/

// Test fixture for workspace management tests
class checkin_misc_memory_model : public ::testing::Test
{
protected:
    rocblas_handle handle;

    void SetUp() override
    {
        ASSERT_EQ(rocblas_create_handle(&handle), rocblas_status_success);
    }

    void TearDown() override
    {
        ASSERT_EQ(rocblas_destroy_handle(handle), rocblas_status_success);
    }

    // Helper function to query workspace size
    template <typename Func, typename... Args>
    size_t query_workspace_size(Func func, Args... args)
    {
        size_t size;
        rocblas_start_device_memory_size_query(handle);
        func(handle, args...);
        rocblas_stop_device_memory_size_query(handle, &size);
        return size;
    }
};

/*************************************/
/***** 1. Device Memory Size Query Tests *****/
/*************************************/

TEST_F(checkin_misc_memory_model, MemorySizeQuery_GETRF_Deterministic)
{
    const rocblas_int n = 100;
    const rocblas_int lda = n;
    const rocblas_int batch_count = 10;
    const rocblas_stride stA = lda * n;
    const rocblas_stride stP = n;

    double* dA;
    rocblas_int *dP, *dinfo;
    ASSERT_EQ(hipMalloc(&dA, sizeof(double) * stA * batch_count), hipSuccess);
    ASSERT_EQ(hipMalloc(&dP, sizeof(rocblas_int) * stP * batch_count), hipSuccess);
    ASSERT_EQ(hipMalloc(&dinfo, sizeof(rocblas_int) * batch_count), hipSuccess);

    // Query size twice - should be identical (deterministic)
    size_t size1 = query_workspace_size(rocsolver_dgetrf_strided_batched, n, n, dA, lda, stA, dP,
                                        stP, dinfo, batch_count);

    size_t size2 = query_workspace_size(rocsolver_dgetrf_strided_batched, n, n, dA, lda, stA, dP,
                                        stP, dinfo, batch_count);

    EXPECT_EQ(size1, size2);
    EXPECT_GT(size1, 0);

    ASSERT_EQ(hipFree(dA), hipSuccess);
    ASSERT_EQ(hipFree(dP), hipSuccess);
    ASSERT_EQ(hipFree(dinfo), hipSuccess);
}

TEST_F(checkin_misc_memory_model, MemorySizeQuery_GETRF_SizeScaling)
{
    const rocblas_int n = 100;
    const rocblas_int lda = n;
    const rocblas_stride stA = lda * n;
    const rocblas_stride stP = n;

    double* dA;
    rocblas_int *dP, *dinfo;
    ASSERT_EQ(hipMalloc(&dA, sizeof(double) * stA * 100), hipSuccess);
    ASSERT_EQ(hipMalloc(&dP, sizeof(rocblas_int) * stP * 100), hipSuccess);
    ASSERT_EQ(hipMalloc(&dinfo, sizeof(rocblas_int) * 100), hipSuccess);

    // Query with increasing batch counts
    size_t size_bc1 = query_workspace_size(rocsolver_dgetrf_strided_batched, n, n, dA, lda, stA, dP,
                                           stP, dinfo, 1);

    size_t size_bc10 = query_workspace_size(rocsolver_dgetrf_strided_batched, n, n, dA, lda, stA,
                                            dP, stP, dinfo, 10);

    size_t size_bc100 = query_workspace_size(rocsolver_dgetrf_strided_batched, n, n, dA, lda, stA,
                                             dP, stP, dinfo, 100);

    // Size should increase or stay the same with batch count (workspace may be shared)
    EXPECT_LE(size_bc1 * 9, size_bc10);
    EXPECT_GE(size_bc1 * 11, size_bc10);
    EXPECT_LE(size_bc10 * 9, size_bc100);
    EXPECT_GE(size_bc10 * 11, size_bc100);

    ASSERT_EQ(hipFree(dA), hipSuccess);
    ASSERT_EQ(hipFree(dP), hipSuccess);
    ASSERT_EQ(hipFree(dinfo), hipSuccess);
}

TEST_F(checkin_misc_memory_model, MemorySizeQuery_ComplexVsReal)
{
    const rocblas_int n = 100;
    const rocblas_int lda = n;
    const rocblas_int batch_count = 10;
    const rocblas_stride stA = lda * n;
    const rocblas_stride stP = n;

    // Real version
    double* dA_real;
    rocblas_int *dP_real, *dinfo_real;
    ASSERT_EQ(hipMalloc(&dA_real, sizeof(double) * stA * batch_count), hipSuccess);
    ASSERT_EQ(hipMalloc(&dP_real, sizeof(rocblas_int) * stP * batch_count), hipSuccess);
    ASSERT_EQ(hipMalloc(&dinfo_real, sizeof(rocblas_int) * batch_count), hipSuccess);

    size_t size_real = query_workspace_size(rocsolver_dgetrf_strided_batched, n, n, dA_real, lda,
                                            stA, dP_real, stP, dinfo_real, batch_count);

    // Complex version
    rocblas_double_complex* dA_complex;
    rocblas_int *dP_complex, *dinfo_complex;
    ASSERT_EQ(hipMalloc(&dA_complex, sizeof(rocblas_double_complex) * stA * batch_count), hipSuccess);
    ASSERT_EQ(hipMalloc(&dP_complex, sizeof(rocblas_int) * stP * batch_count), hipSuccess);
    ASSERT_EQ(hipMalloc(&dinfo_complex, sizeof(rocblas_int) * batch_count), hipSuccess);

    size_t size_complex = query_workspace_size(rocsolver_zgetrf_strided_batched, n, n, dA_complex,
                                               lda, stA, dP_complex, stP, dinfo_complex, batch_count);

    // Complex should require more memory (different scalar arrays)
    // At minimum, sizes should be positive
    EXPECT_GT(size_real, 0);
    EXPECT_LE(size_real * 1.9, size_complex);
    EXPECT_GE(size_real * 2.1, size_complex);

    ASSERT_EQ(hipFree(dA_real), hipSuccess);
    ASSERT_EQ(hipFree(dP_real), hipSuccess);
    ASSERT_EQ(hipFree(dinfo_real), hipSuccess);
    ASSERT_EQ(hipFree(dA_complex), hipSuccess);
    ASSERT_EQ(hipFree(dP_complex), hipSuccess);
    ASSERT_EQ(hipFree(dinfo_complex), hipSuccess);
}

TEST_F(checkin_misc_memory_model, MemorySizeQuery_EdgeCase_ZeroSize)
{
    const rocblas_int n = 0;
    const rocblas_int lda = 1;
    const rocblas_int batch_count = 0;
    const rocblas_stride stA = 1;
    const rocblas_stride stP = 1;

    double* dA = nullptr;
    rocblas_int *dP = nullptr, *dinfo = nullptr;

    size_t size = query_workspace_size(rocsolver_dgetrf_strided_batched, n, n, dA, lda, stA, dP,
                                       stP, dinfo, batch_count);

    // Zero-sized problem should require minimal or no workspace
    // The exact behavior depends on implementation
    EXPECT_EQ(size, 0);
}

TEST_F(checkin_misc_memory_model, MemorySizeQuery_GETRF_SmallVsLarge)
{
    const rocblas_int n_large = 100;
    const rocblas_int n_small = 10;
    const rocblas_int lda_large = n_large;
    const rocblas_int lda_small = n_small;
    const rocblas_stride stA_large = lda_large * n_large;
    const rocblas_stride stP_large = n_large;
    const rocblas_stride stA_small = n_small * n_small;
    const rocblas_stride stP_small = n_small;

    // Large problem
    double* dA_large;
    rocblas_int *dP_large, *dinfo_large;
    ASSERT_EQ(hipMalloc(&dA_large, sizeof(double) * stA_large * 10), hipSuccess);
    ASSERT_EQ(hipMalloc(&dP_large, sizeof(rocblas_int) * stP_large * 10), hipSuccess);
    ASSERT_EQ(hipMalloc(&dinfo_large, sizeof(rocblas_int) * 10), hipSuccess);

    size_t size_large
        = query_workspace_size(rocsolver_dgetrf_strided_batched, n_large, n_large, dA_large,
                               lda_large, stA_large, dP_large, stP_large, dinfo_large, 10);

    // Small problem
    double* dA_small;
    rocblas_int *dP_small, *dinfo_small;
    ASSERT_EQ(hipMalloc(&dA_small, sizeof(double) * stA_small * 10), hipSuccess);
    ASSERT_EQ(hipMalloc(&dP_small, sizeof(rocblas_int) * stP_small * 10), hipSuccess);
    ASSERT_EQ(hipMalloc(&dinfo_small, sizeof(rocblas_int) * 10), hipSuccess);

    size_t size_small
        = query_workspace_size(rocsolver_dgetrf_strided_batched, n_small, n_small, dA_small,
                               lda_small, stA_small, dP_small, stP_small, dinfo_small, 10);

    // Large problem should require more workspace
    EXPECT_GT(size_large, size_small);

    ASSERT_EQ(hipFree(dA_large), hipSuccess);
    ASSERT_EQ(hipFree(dP_large), hipSuccess);
    ASSERT_EQ(hipFree(dinfo_large), hipSuccess);
    ASSERT_EQ(hipFree(dA_small), hipSuccess);
    ASSERT_EQ(hipFree(dP_small), hipSuccess);
    ASSERT_EQ(hipFree(dinfo_small), hipSuccess);
}

/*************************************/
/***** 2. Numerical Correctness Tests *****/
/*************************************/

TEST_F(checkin_misc_memory_model, NumericalCorrectness_MultipleInvocations_GETRF)
{
    const rocblas_int n = 50;
    const rocblas_int lda = n;
    const rocblas_stride stA = lda * n;
    const rocblas_stride stP = n;

    // Allocate host and device memory
    std::vector<double> hA(stA);
    std::vector<double> hA_results[10];
    std::vector<rocblas_int> hP(stP);
    rocblas_int hinfo;

    // Initialize a simple test matrix (identity + small perturbation)
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            hA[i + j * lda] = (i == j) ? 1.0 : 0.01;
        }
    }

    double* dA;
    rocblas_int *dP, *dinfo;
    ASSERT_EQ(hipMalloc(&dA, sizeof(double) * stA), hipSuccess);
    ASSERT_EQ(hipMalloc(&dP, sizeof(rocblas_int) * stP), hipSuccess);
    ASSERT_EQ(hipMalloc(&dinfo, sizeof(rocblas_int)), hipSuccess);

    // Execute GETRF 10 times with same input
    for(int iter = 0; iter < 10; iter++)
    {
        // Reset matrix
        ASSERT_EQ(hipMemcpy(dA, hA.data(), sizeof(double) * stA, hipMemcpyHostToDevice), hipSuccess);

        // Execute GETRF
        rocblas_status status = rocsolver_dgetrf(handle, n, n, dA, lda, dP, dinfo);
        EXPECT_EQ(status, rocblas_status_success);

        // Copy results back
        hA_results[iter].resize(stA);
        ASSERT_EQ(hipMemcpy(hA_results[iter].data(), dA, sizeof(double) * stA, hipMemcpyDeviceToHost),
                  hipSuccess);
        ASSERT_EQ(hipMemcpy(&hinfo, dinfo, sizeof(rocblas_int), hipMemcpyDeviceToHost), hipSuccess);
        EXPECT_EQ(hinfo, 0);
    }

    // Verify all results are identical
    for(int iter = 1; iter < 10; iter++)
    {
        for(size_t i = 0; i < stA; i++)
        {
            EXPECT_NEAR(hA_results[iter][i], hA_results[0][i], 1e-10)
                << "Mismatch at iteration " << iter << " index " << i;
        }
    }

    ASSERT_EQ(hipFree(dA), hipSuccess);
    ASSERT_EQ(hipFree(dP), hipSuccess);
    ASSERT_EQ(hipFree(dinfo), hipSuccess);
}

TEST_F(checkin_misc_memory_model, NumericalCorrectness_AlternatingSizes)
{
    const rocblas_int n_large = 100;
    const rocblas_int n_small = 20;
    const rocblas_int lda_large = n_large;
    const rocblas_int lda_small = n_small;
    const rocblas_stride stA_large = lda_large * n_large;
    const rocblas_stride stA_small = lda_small * n_small;
    const rocblas_stride stP_large = n_large;
    const rocblas_stride stP_small = n_small;

    // Allocate for large size
    double* dA;
    rocblas_int *dP, *dinfo;
    ASSERT_EQ(hipMalloc(&dA, sizeof(double) * stA_large), hipSuccess);
    ASSERT_EQ(hipMalloc(&dP, sizeof(rocblas_int) * stP_large), hipSuccess);
    ASSERT_EQ(hipMalloc(&dinfo, sizeof(rocblas_int)), hipSuccess);

    std::vector<double> hA_large(stA_large);
    std::vector<double> hA_small(stA_small);

    // Initialize matrices
    for(int i = 0; i < n_large; i++)
        for(int j = 0; j < n_large; j++)
            hA_large[i + j * lda_large] = (i == j) ? 2.0 : 0.01;

    for(int i = 0; i < n_small; i++)
        for(int j = 0; j < n_small; j++)
            hA_small[i + j * lda_small] = (i == j) ? 3.0 : 0.02;

    rocblas_int hinfo;

    // Alternate between large and small problems
    for(int iter = 0; iter < 5; iter++)
    {
        // Large problem
        ASSERT_EQ(hipMemcpy(dA, hA_large.data(), sizeof(double) * stA_large, hipMemcpyHostToDevice),
                  hipSuccess);
        rocblas_status status = rocsolver_dgetrf(handle, n_large, n_large, dA, lda_large, dP, dinfo);
        EXPECT_EQ(status, rocblas_status_success);
        ASSERT_EQ(hipMemcpy(&hinfo, dinfo, sizeof(rocblas_int), hipMemcpyDeviceToHost), hipSuccess);
        EXPECT_EQ(hinfo, 0);

        // Small problem
        ASSERT_EQ(hipMemcpy(dA, hA_small.data(), sizeof(double) * stA_small, hipMemcpyHostToDevice),
                  hipSuccess);
        status = rocsolver_dgetrf(handle, n_small, n_small, dA, lda_small, dP, dinfo);
        EXPECT_EQ(status, rocblas_status_success);
        ASSERT_EQ(hipMemcpy(&hinfo, dinfo, sizeof(rocblas_int), hipMemcpyDeviceToHost), hipSuccess);
        EXPECT_EQ(hinfo, 0);
    }

    ASSERT_EQ(hipFree(dA), hipSuccess);
    ASSERT_EQ(hipFree(dP), hipSuccess);
    ASSERT_EQ(hipFree(dinfo), hipSuccess);
}

/*************************************/
/***** 3. Nested Workspace Tests *****/
/*************************************/

TEST_F(checkin_misc_memory_model, NestedWorkspace_GEBLTTRS_reuses_GETRS)
{
    // GEBLTTRS solves a block-tridiagonal system by looping GETRS (plus GEMM,
    // which needs no extra workspace) over the nblocks diagonal blocks. Its
    // workspace is a pure pass-through to a single nb-by-nrhs GETRS with no
    // additional buffer, so the two queries must match exactly and GEBLTTRS
    // must not grow with nblocks. (Contrast GESV, which adds an n*nrhs buffer
    // to copy B and therefore cannot satisfy a clean reuse invariant.)
    const rocblas_int nb = 100;
    const rocblas_int nrhs = 10;
    const rocblas_int lda = nb;
    const rocblas_int ldb = nb;
    const rocblas_int ldc = nb;
    const rocblas_int ldx = nb;

    // void lambda (writes size through an out-param) so the ASSERT_EQ macros,
    // which expand to `return;` on failure, remain valid here.
    auto query_geblttrs = [&](rocblas_int nblocks, size_t& size) {
        double *dA, *dB, *dC, *dX;
        ASSERT_EQ(hipMalloc(&dA, sizeof(double) * lda * nb * nblocks), hipSuccess);
        ASSERT_EQ(hipMalloc(&dB, sizeof(double) * ldb * nb * nblocks), hipSuccess);
        ASSERT_EQ(hipMalloc(&dC, sizeof(double) * ldc * nb * nblocks), hipSuccess);
        ASSERT_EQ(hipMalloc(&dX, sizeof(double) * ldx * nrhs * nblocks), hipSuccess);

        size = query_workspace_size(rocsolver_dgeblttrs_npvt, nb, nblocks, nrhs, dA, lda, dB, ldb,
                                    dC, ldc, dX, ldx);

        ASSERT_EQ(hipFree(dA), hipSuccess);
        ASSERT_EQ(hipFree(dB), hipSuccess);
        ASSERT_EQ(hipFree(dC), hipSuccess);
        ASSERT_EQ(hipFree(dX), hipSuccess);
    };

    // Standalone GETRS over a single nb-by-nrhs block, matching the inner solve.
    double *dGA, *dGB;
    rocblas_int* dGP;
    ASSERT_EQ(hipMalloc(&dGA, sizeof(double) * lda * nb), hipSuccess);
    ASSERT_EQ(hipMalloc(&dGB, sizeof(double) * ldb * nrhs), hipSuccess);
    ASSERT_EQ(hipMalloc(&dGP, sizeof(rocblas_int) * nb), hipSuccess);

    size_t getrs_size = query_workspace_size(rocsolver_dgetrs, rocblas_operation_none, nb, nrhs,
                                             dGA, lda, dGP, dGB, ldb);

    ASSERT_EQ(hipFree(dGA), hipSuccess);
    ASSERT_EQ(hipFree(dGB), hipSuccess);
    ASSERT_EQ(hipFree(dGP), hipSuccess);

    size_t geblttrs_size_1 = 0, geblttrs_size_8 = 0;
    query_geblttrs(1, geblttrs_size_1);
    query_geblttrs(8, geblttrs_size_8);

    // GEBLTTRS reuses the GETRS workspace verbatim, with no extra buffer.
    EXPECT_EQ(geblttrs_size_1, getrs_size);

    // Workspace is shared across the per-block solves, so it must not grow with nblocks.
    EXPECT_EQ(geblttrs_size_8, geblttrs_size_1);
}

TEST_F(checkin_misc_memory_model, NestedWorkspace_GESV_NumericalCorrectness)
{
    const rocblas_int n = 50;
    const rocblas_int nrhs = 5;
    const rocblas_int lda = n;
    const rocblas_int ldb = n;

    // Create a simple linear system Ax = b
    std::vector<double> hA(lda * n);
    std::vector<double> hB(ldb * nrhs);
    std::vector<double> hB_original(ldb * nrhs);
    std::vector<double> hX(ldb * nrhs);
    std::vector<rocblas_int> hP(n);
    rocblas_int hinfo;

    // Initialize A as diagonally dominant
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            if(i == j)
                hA[i + j * lda] = 10.0;
            else
                hA[i + j * lda] = 0.1;
        }
    }

    // Initialize B with known values
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < nrhs; j++)
        {
            hB[i + j * ldb] = 1.0 + i * 0.1 + j * 0.01;
            hB_original[i + j * ldb] = hB[i + j * ldb];
        }
    }

    double *dA, *dB;
    rocblas_int *dP, *dinfo;
    ASSERT_EQ(hipMalloc(&dA, sizeof(double) * lda * n), hipSuccess);
    ASSERT_EQ(hipMalloc(&dB, sizeof(double) * ldb * nrhs), hipSuccess);
    ASSERT_EQ(hipMalloc(&dP, sizeof(rocblas_int) * n), hipSuccess);
    ASSERT_EQ(hipMalloc(&dinfo, sizeof(rocblas_int)), hipSuccess);

    ASSERT_EQ(hipMemcpy(dA, hA.data(), sizeof(double) * lda * n, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(dB, hB.data(), sizeof(double) * ldb * nrhs, hipMemcpyHostToDevice),
              hipSuccess);

    // Execute GESV
    rocblas_status status = rocsolver_dgesv(handle, n, nrhs, dA, lda, dP, dB, ldb, dinfo);
    EXPECT_EQ(status, rocblas_status_success);

    // Copy results back
    ASSERT_EQ(hipMemcpy(hX.data(), dB, sizeof(double) * ldb * nrhs, hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(&hinfo, dinfo, sizeof(rocblas_int), hipMemcpyDeviceToHost), hipSuccess);
    EXPECT_EQ(hinfo, 0);

    // Verify solution: compute residual ||Ax - b||
    // Reload original A
    ASSERT_EQ(hipMemcpy(dA, hA.data(), sizeof(double) * lda * n, hipMemcpyHostToDevice), hipSuccess);

    std::vector<double> hAx(ldb * nrhs, 0.0);
    for(int j = 0; j < nrhs; j++)
    {
        for(int i = 0; i < n; i++)
        {
            for(int k = 0; k < n; k++)
            {
                hAx[i + j * ldb] += hA[i + k * lda] * hX[k + j * ldb];
            }
        }
    }

    // Check residual
    double max_residual = 0.0;
    for(int j = 0; j < nrhs; j++)
    {
        for(int i = 0; i < n; i++)
        {
            double residual = std::abs(hAx[i + j * ldb] - hB_original[i + j * ldb]);
            max_residual = std::max(max_residual, residual);
        }
    }

    EXPECT_LT(max_residual, 1e-6) << "Solution residual too large";

    ASSERT_EQ(hipFree(dA), hipSuccess);
    ASSERT_EQ(hipFree(dB), hipSuccess);
    ASSERT_EQ(hipFree(dP), hipSuccess);
    ASSERT_EQ(hipFree(dinfo), hipSuccess);
}

/*************************************/
/***** 4. User-Managed Memory Tests *****/
/*************************************/

TEST_F(checkin_misc_memory_model, UserManagedMemory_ExactAllocation)
{
    const rocblas_int n = 100;
    const rocblas_int lda = n;
    const rocblas_stride stA = lda * n;
    const rocblas_stride stP = n;

    double* dA;
    rocblas_int *dP, *dinfo;
    ASSERT_EQ(hipMalloc(&dA, sizeof(double) * stA), hipSuccess);
    ASSERT_EQ(hipMalloc(&dP, sizeof(rocblas_int) * stP), hipSuccess);
    ASSERT_EQ(hipMalloc(&dinfo, sizeof(rocblas_int)), hipSuccess);

    // Query required workspace size
    size_t required_size = query_workspace_size(rocsolver_dgetrf, n, n, dA, lda, dP, dinfo);

    EXPECT_GT(required_size, 0);

    // Allocate exact amount and set as user workspace
    void* workspace;
    ASSERT_EQ(hipMalloc(&workspace, required_size), hipSuccess);
    ASSERT_EQ(rocblas_set_workspace(handle, workspace, required_size), rocblas_status_success);

    // Verify memory is now user-managed
    EXPECT_FALSE(rocblas_is_managing_device_memory(handle));

    // Initialize matrix
    std::vector<double> hA(stA);
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            hA[i + j * lda] = (i == j) ? 2.0 : 0.01;

    ASSERT_EQ(hipMemcpy(dA, hA.data(), sizeof(double) * stA, hipMemcpyHostToDevice), hipSuccess);

    // Execute should succeed with exact allocation
    rocblas_status status = rocsolver_dgetrf(handle, n, n, dA, lda, dP, dinfo);
    EXPECT_EQ(status, rocblas_status_success);

    rocblas_int hinfo;
    ASSERT_EQ(hipMemcpy(&hinfo, dinfo, sizeof(rocblas_int), hipMemcpyDeviceToHost), hipSuccess);
    EXPECT_EQ(hinfo, 0);

    ASSERT_EQ(hipFree(workspace), hipSuccess);
    ASSERT_EQ(hipFree(dA), hipSuccess);
    ASSERT_EQ(hipFree(dP), hipSuccess);
    ASSERT_EQ(hipFree(dinfo), hipSuccess);
}

TEST_F(checkin_misc_memory_model, UserManagedMemory_InsufficientAllocation)
{
    const rocblas_int n = 100;
    const rocblas_int lda = n;
    const rocblas_stride stA = lda * n;
    const rocblas_stride stP = n;

    double* dA;
    rocblas_int *dP, *dinfo;
    ASSERT_EQ(hipMalloc(&dA, sizeof(double) * stA), hipSuccess);
    ASSERT_EQ(hipMalloc(&dP, sizeof(rocblas_int) * stP), hipSuccess);
    ASSERT_EQ(hipMalloc(&dinfo, sizeof(rocblas_int)), hipSuccess);

    // Query required workspace size
    size_t required_size = query_workspace_size(rocsolver_dgetrf, n, n, dA, lda, dP, dinfo);

    // Allocate less than required
    size_t insufficient_size = required_size / 2;
    void* workspace;
    ASSERT_EQ(hipMalloc(&workspace, insufficient_size), hipSuccess);
    ASSERT_EQ(rocblas_set_workspace(handle, workspace, insufficient_size), rocblas_status_success);

    // Initialize matrix
    std::vector<double> hA(stA);
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            hA[i + j * lda] = (i == j) ? 2.0 : 0.01;

    ASSERT_EQ(hipMemcpy(dA, hA.data(), sizeof(double) * stA, hipMemcpyHostToDevice), hipSuccess);

    // Execute should fail with insufficient memory
    rocblas_status status = rocsolver_dgetrf(handle, n, n, dA, lda, dP, dinfo);
    EXPECT_EQ(status, rocblas_status_memory_error);

    ASSERT_EQ(hipFree(workspace), hipSuccess);
    ASSERT_EQ(hipFree(dA), hipSuccess);
    ASSERT_EQ(hipFree(dP), hipSuccess);
    ASSERT_EQ(hipFree(dinfo), hipSuccess);
}

/*************************************/
/***** 5. Batched Functions Tests *****/
/*************************************/

TEST_F(checkin_misc_memory_model, BatchedFunction_GETRF_Correctness)
{
    const rocblas_int n = 30;
    const rocblas_int lda = n;
    const rocblas_stride stA = lda * n;
    const rocblas_int batch_count = 5;

    // Allocate batched arrays (pointers array)
    std::vector<double*> hA_array(batch_count);
    std::vector<double> hA_data(stA);

    for(int b = 0; b < batch_count; b++)
    {
        ASSERT_EQ(hipMalloc(&hA_array[b], sizeof(double) * stA), hipSuccess);

        // Initialize each batch with different matrix
        for(int i = 0; i < n; i++)
            for(int j = 0; j < n; j++)
                hA_data[i + j * lda] = (i == j) ? (2.0 + b * 0.1) : 0.01;

        ASSERT_EQ(hipMemcpy(hA_array[b], hA_data.data(), sizeof(double) * stA, hipMemcpyHostToDevice),
                  hipSuccess);
    }

    // Copy pointer array to device
    double** dA_array;
    ASSERT_EQ(hipMalloc(&dA_array, sizeof(double*) * batch_count), hipSuccess);
    ASSERT_EQ(
        hipMemcpy(dA_array, hA_array.data(), sizeof(double*) * batch_count, hipMemcpyHostToDevice),
        hipSuccess);

    rocblas_int *dP, *dinfo;
    ASSERT_EQ(hipMalloc(&dP, sizeof(rocblas_int) * n * batch_count), hipSuccess);
    ASSERT_EQ(hipMalloc(&dinfo, sizeof(rocblas_int) * batch_count), hipSuccess);

    // Execute batched GETRF
    rocblas_status status
        = rocsolver_dgetrf_batched(handle, n, n, dA_array, lda, dP, n, dinfo, batch_count);
    EXPECT_EQ(status, rocblas_status_success);

    // Verify all batches succeeded
    std::vector<rocblas_int> hinfo(batch_count);
    ASSERT_EQ(hipMemcpy(hinfo.data(), dinfo, sizeof(rocblas_int) * batch_count, hipMemcpyDeviceToHost),
              hipSuccess);

    for(int b = 0; b < batch_count; b++)
    {
        EXPECT_EQ(hinfo[b], 0) << "Batch " << b << " failed";
    }

    // Cleanup
    for(int b = 0; b < batch_count; b++)
    {
        ASSERT_EQ(hipFree(hA_array[b]), hipSuccess);
    }
    ASSERT_EQ(hipFree(dA_array), hipSuccess);
    ASSERT_EQ(hipFree(dP), hipSuccess);
    ASSERT_EQ(hipFree(dinfo), hipSuccess);
}

/*************************************/
/***** 6. Stress Tests *****/
/*************************************/

TEST_F(checkin_misc_memory_model, StressTest_RapidAllocationDeallocation)
{
    const rocblas_int n = 50;
    const rocblas_int lda = n;
    const rocblas_stride stA = lda * n;

    double* dA;
    rocblas_int *dP, *dinfo;
    ASSERT_EQ(hipMalloc(&dA, sizeof(double) * lda * n), hipSuccess);
    ASSERT_EQ(hipMalloc(&dP, sizeof(rocblas_int) * n), hipSuccess);
    ASSERT_EQ(hipMalloc(&dinfo, sizeof(rocblas_int)), hipSuccess);

    std::vector<double> hA(stA);
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            hA[i + j * lda] = (i == j) ? 2.0 : 0.01;

    // Execute 100 times rapidly
    for(int iter = 0; iter < 100; iter++)
    {
        ASSERT_EQ(hipMemcpy(dA, hA.data(), sizeof(double) * stA, hipMemcpyHostToDevice), hipSuccess);

        rocblas_status status = rocsolver_dgetrf(handle, n, n, dA, lda, dP, dinfo);
        EXPECT_EQ(status, rocblas_status_success) << "Failed at iteration " << iter;

        rocblas_int hinfo;
        ASSERT_EQ(hipMemcpy(&hinfo, dinfo, sizeof(rocblas_int), hipMemcpyDeviceToHost), hipSuccess);
        EXPECT_EQ(hinfo, 0) << "Non-zero info at iteration " << iter;
    }

    ASSERT_EQ(hipFree(dA), hipSuccess);
    ASSERT_EQ(hipFree(dP), hipSuccess);
    ASSERT_EQ(hipFree(dinfo), hipSuccess);
}

TEST_F(checkin_misc_memory_model, StressTest_RandomSizes)
{
    const int num_iterations = 50;
    std::vector<rocblas_int> sizes = {10, 20, 30, 50, 70, 100, 150, 200};

    double* dA;
    rocblas_int *dP, *dinfo;

    // Allocate for maximum size
    rocblas_int max_size = 200;
    ASSERT_EQ(hipMalloc(&dA, sizeof(double) * max_size * max_size), hipSuccess);
    ASSERT_EQ(hipMalloc(&dP, sizeof(rocblas_int) * max_size), hipSuccess);
    ASSERT_EQ(hipMalloc(&dinfo, sizeof(rocblas_int)), hipSuccess);

    // Execute with varying sizes
    for(int iter = 0; iter < num_iterations; iter++)
    {
        rocblas_int n = sizes[iter % sizes.size()];
        rocblas_int lda = n;
        rocblas_stride stA = lda * n;

        std::vector<double> hA(stA);
        for(int i = 0; i < n; i++)
            for(int j = 0; j < n; j++)
                hA[i + j * lda] = (i == j) ? 2.0 : 0.01;

        ASSERT_EQ(hipMemcpy(dA, hA.data(), sizeof(double) * stA, hipMemcpyHostToDevice), hipSuccess);

        rocblas_status status = rocsolver_dgetrf(handle, n, n, dA, lda, dP, dinfo);
        EXPECT_EQ(status, rocblas_status_success)
            << "Failed at iteration " << iter << " with size " << n;

        rocblas_int hinfo;
        ASSERT_EQ(hipMemcpy(&hinfo, dinfo, sizeof(rocblas_int), hipMemcpyDeviceToHost), hipSuccess);
        EXPECT_EQ(hinfo, 0) << "Non-zero info at iteration " << iter << " with size " << n;
    }

    ASSERT_EQ(hipFree(dA), hipSuccess);
    ASSERT_EQ(hipFree(dP), hipSuccess);
    ASSERT_EQ(hipFree(dinfo), hipSuccess);
}
