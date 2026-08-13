// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstring>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <iostream>
#include <string>

#include "helper.h"

// A shape for which the default heuristic pick splits the K reduction differently
// for different output rows, so the first run below reports non-uniform rows.
constexpr int64_t kM = 6144;
constexpr int64_t kN = 5120;
constexpr int64_t kK = 8192;

constexpr int64_t kMaxWorkspaceSize = 128 * 1024 * 1024;

using GemmRunner = Runner<float, float, float, float, float>;

// a is column-major with lda = m, so replicating a row across m means every
// element of a given column of the buffer holds the same value.
static void fillInputs(GemmRunner& runner)
{
    float* a = static_cast<float*>(runner.a);
    for(int64_t j = 0; j < runner.k; ++j)
    {
        const float v = static_cast<float>((rand() % 2001) - 1000) / 997.0f;
        for(int64_t i = 0; i < runner.m; ++i)
            a[i + j * runner.m] = v;
    }

    float* b = static_cast<float*>(runner.b);
    for(int64_t i = 0; i < runner.k * runner.n; ++i)
        b[i] = static_cast<float>((rand() % 2001) - 1000) / 991.0f;
}

// d is column-major with ldd = m. Every row of d holds the same value exactly
// when every column of the buffer is bitwise constant.
static int64_t countNonUniformColumns(const float* d, int64_t m, int64_t n)
{
    int64_t differing = 0;
    for(int64_t j = 0; j < n; ++j)
    {
        const float* col = d + j * m;
        for(int64_t i = 1; i < m; ++i)
        {
            if(std::memcmp(&col[i], &col[0], sizeof(float)) != 0)
            {
                ++differing;
                break;
            }
        }
    }
    return differing;
}

static void runOnce(GemmRunner& runner, bool uniformSummationOrder, const char* label)
{
    std::cout << label << std::endl;

    // Poison d so that a buffer left over from the previous run cannot be
    // mistaken for a uniform result.
    CHECK_HIP_ERROR(
        hipMemsetAsync(runner.d_d, 0xAB, runner.m * runner.n * sizeof(float), runner.stream));

    hipblasLtMatrixLayout_t matA, matB, matC, matD;
    CHECK_HIPBLASLT_ERROR(
        hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, runner.m, runner.k, runner.m));
    CHECK_HIPBLASLT_ERROR(
        hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, runner.k, runner.n, runner.k));
    CHECK_HIPBLASLT_ERROR(
        hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, runner.m, runner.n, runner.m));
    CHECK_HIPBLASLT_ERROR(
        hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, runner.m, runner.n, runner.m));

    hipblasLtMatmulDesc_t matmul;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F));

    const hipblasOperation_t trans = HIPBLAS_OP_N;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(int32_t)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &trans, sizeof(int32_t)));

    if(uniformSummationOrder)
    {
        // Only 0 (off, the default) and 1 (on) are accepted.
        const int32_t uniform = 1;
        CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_UNIFORM_SUMMATION_ORDER_EXT, &uniform, sizeof(uniform)));
    }

    hipblasLtMatmulPreference_t pref;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceCreate(&pref));
    CHECK_HIPBLASLT_ERROR(
        hipblasLtMatmulPreferenceSetAttribute(pref,
                                              HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &runner.max_workspace_size,
                                              sizeof(runner.max_workspace_size)));

    hipblasLtMatmulHeuristicResult_t heuristicResult[1];
    int                              returnedAlgoCount = 0;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulAlgoGetHeuristic(runner.handle,
                                                          matmul,
                                                          matA,
                                                          matB,
                                                          matC,
                                                          matD,
                                                          pref,
                                                          1,
                                                          heuristicResult,
                                                          &returnedAlgoCount));

    if(returnedAlgoCount == 0)
    {
        std::cout << "  no solution returned by the heuristic" << std::endl;
    }
    else
    {
        std::cout << "  solution: "
                  << hipblaslt_ext::getSolutionNameFromAlgo(runner.handle,
                                                            heuristicResult[0].algo)
                  << std::endl;

        hipblasStatus_t status = hipblasLtMatmul(runner.handle,
                                                 matmul,
                                                 &runner.alpha,
                                                 runner.d_a,
                                                 matA,
                                                 runner.d_b,
                                                 matB,
                                                 &runner.beta,
                                                 runner.d_c,
                                                 matC,
                                                 runner.d_d,
                                                 matD,
                                                 &heuristicResult[0].algo,
                                                 runner.d_workspace,
                                                 heuristicResult[0].workspaceSize,
                                                 runner.stream);

        if(status == HIPBLAS_STATUS_INVALID_VALUE)
        {
            // With the mode on, this means no configuration honoring the uniformity
            // guarantee exists for this problem on this device. The library refuses
            // the call instead of returning rows that are not bitwise identical, so
            // it is an outcome to handle rather than a fatal error.
            std::cout << "  HIPBLAS_STATUS_INVALID_VALUE: no configuration honoring the "
                         "guarantee exists here, the matmul did not run"
                      << std::endl;
        }
        else
        {
            CHECK_HIPBLASLT_ERROR(status);
            runner.deviceToHost();
            CHECK_HIP_ERROR(hipStreamSynchronize(runner.stream));

            const int64_t differing
                = countNonUniformColumns(static_cast<const float*>(runner.d), runner.m, runner.n);
            if(differing == 0)
                std::cout << "  rows of d bitwise identical: yes" << std::endl;
            else
                std::cout << "  rows of d bitwise identical: no (" << differing << " of "
                          << runner.n << " columns of d are not constant down m)" << std::endl;
        }
    }

    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matD));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(pref));
}

int main()
{
    /** This is an NN example with
     *  a = (m, k). lda = m, every row of a holds the same k-vector
     *  b = (k, n). ldb = k
     *  c = d = (m, n). ldc = ldd = m, beta = 0
     *
     *  With the mode on, every row of d must be bitwise identical within one run.
     *  That is not run-to-run determinism; see
     *  docs/how-to/how-to-use-uniform-summation-order.rst for the full semantics.
     */
    GemmRunner runner(kM, kN, kK, 1, 1.f, 0.f, kMaxWorkspaceSize);

    fillInputs(runner);
    runner.hostToDevice();

    std::cout << "fp32 NN GEMM, m=" << runner.m << " n=" << runner.n << " k=" << runner.k
              << ", beta=0, every row of a is the same k-vector" << std::endl;

    runOnce(runner, false, "Run 1: uniform summation order off (library default)");
    runOnce(runner,
            true,
            "Run 2: uniform summation order on "
            "(HIPBLASLT_MATMUL_DESC_UNIFORM_SUMMATION_ORDER_EXT)");

    return 0;
}
