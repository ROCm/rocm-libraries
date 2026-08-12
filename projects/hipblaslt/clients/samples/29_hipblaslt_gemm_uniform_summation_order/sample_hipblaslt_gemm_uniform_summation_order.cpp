// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstring>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <iostream>
#include <string>
#include <vector>

#include "helper.h"

// Measured on gfx950 / ROCm 7.1.1: for this shape the default heuristic pick
// splits the K reduction differently for different output rows, so rows of D
// that should be identical are not. M=3072 N=3072 K=12288 behaves the same way.
constexpr int64_t kM = 6144;
constexpr int64_t kN = 5120;
constexpr int64_t kK = 8192;

constexpr int64_t kMaxWorkspaceSize = 128 * 1024 * 1024;

using GemmRunner = Runner<float, float, float, float, float>;

static const char* statusName(hipblasStatus_t status)
{
    switch(status)
    {
    case HIPBLAS_STATUS_SUCCESS:
        return "HIPBLAS_STATUS_SUCCESS";
    case HIPBLAS_STATUS_INVALID_VALUE:
        return "HIPBLAS_STATUS_INVALID_VALUE";
    case HIPBLAS_STATUS_NOT_SUPPORTED:
        return "HIPBLAS_STATUS_NOT_SUPPORTED";
    default:
        return "other hipblasStatus_t";
    }
}

// A is column-major with lda = m, so replicating a row across M means every
// element of a given column of the buffer holds the same value.
static void fillRowReplicatedA(float* a, int64_t m, int64_t k)
{
    for(int64_t j = 0; j < k; ++j)
    {
        const float v = static_cast<float>((rand() % 2001) - 1000) / 997.0f;
        for(int64_t i = 0; i < m; ++i)
            a[i + j * m] = v;
    }
}

static void fillB(float* b, int64_t k, int64_t n)
{
    for(int64_t i = 0; i < k * n; ++i)
        b[i] = static_cast<float>((rand() % 2001) - 1000) / 991.0f;
}

// D is column-major with ldd = m. Every row of D holds the same value exactly
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

static void poisonD(GemmRunner& runner)
{
    CHECK_HIP_ERROR(
        hipMemsetAsync(runner.d_d, 0xAB, runner.m * runner.n * sizeof(float), runner.stream));
    CHECK_HIP_ERROR(hipStreamSynchronize(runner.stream));
}

static void reportRun(const char*        label,
                      GemmRunner&        runner,
                      hipblasStatus_t    status,
                      const std::string& solutionName)
{
    std::cout << label << std::endl;
    std::cout << "  solution: " << (solutionName.empty() ? "<none selected>" : solutionName)
              << std::endl;
    std::cout << "  status  : " << statusName(status) << std::endl;

    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        // With the mode on, HIPBLAS_STATUS_INVALID_VALUE means no configuration
        // that can honor the uniformity guarantee exists for this problem on this
        // device. The library refuses the call instead of silently returning rows
        // that are not bitwise identical.
        std::cout << "  rows of D bitwise identical: not evaluated, the matmul did not run"
                  << std::endl;
        return;
    }

    runner.deviceToHost();
    CHECK_HIP_ERROR(hipStreamSynchronize(runner.stream));

    const int64_t differing
        = countNonUniformColumns(static_cast<const float*>(runner.d), runner.m, runner.n);
    if(differing == 0)
        std::cout << "  rows of D bitwise identical: yes" << std::endl;
    else
        std::cout << "  rows of D bitwise identical: no (" << differing << " of " << runner.n
                  << " columns of D are not constant down M)" << std::endl;
}

static hipblasStatus_t
    runWithCApi(GemmRunner& runner, bool uniformSummationOrder, std::string& solutionName)
{
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

    const hipblasOperation_t trans_a = HIPBLAS_OP_N;
    const hipblasOperation_t trans_b = HIPBLAS_OP_N;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(int32_t)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(int32_t)));

    // Only 0 (off, the default) and 1 (on) are accepted; any other value makes
    // the setter return HIPBLAS_STATUS_INVALID_VALUE.
    if(uniformSummationOrder)
    {
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

    hipblasStatus_t status = HIPBLAS_STATUS_NOT_SUPPORTED;
    if(returnedAlgoCount > 0)
    {
        solutionName
            = hipblaslt_ext::getSolutionNameFromAlgo(runner.handle, heuristicResult[0].algo);
        status = hipblasLtMatmul(runner.handle,
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
        if(status == HIPBLAS_STATUS_SUCCESS)
            CHECK_HIP_ERROR(hipStreamSynchronize(runner.stream));
    }

    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matD));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(pref));
    return status;
}

static hipblasStatus_t
    runWithExtApi(GemmRunner& runner, bool uniformSummationOrder, std::string& solutionName)
{
    hipblaslt_ext::GemmPreference gemmPref;
    gemmPref.setMaxWorkspaceBytes(runner.max_workspace_size);
    gemmPref.setUniformSummationOrder(uniformSummationOrder);

    hipblaslt_ext::Gemm gemm(runner.handle,
                             HIPBLAS_OP_N,
                             HIPBLAS_OP_N,
                             HIP_R_32F,
                             HIP_R_32F,
                             HIP_R_32F,
                             HIP_R_32F,
                             HIPBLAS_COMPUTE_32F);

    hipblaslt_ext::GemmEpilogue epilogue;
    hipblaslt_ext::GemmInputs   inputs;
    inputs.setA(runner.d_a);
    inputs.setB(runner.d_b);
    inputs.setC(runner.d_c);
    inputs.setD(runner.d_d);
    inputs.setAlpha(&runner.alpha);
    inputs.setBeta(&runner.beta);
    CHECK_HIPBLASLT_ERROR(
        gemm.setProblem(runner.m, runner.n, runner.k, runner.batch_count, epilogue, inputs));

    std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;
    hipblasStatus_t status = gemm.algoGetHeuristic(1, gemmPref, heuristicResult);
    if(status != HIPBLAS_STATUS_SUCCESS)
        return status;
    if(heuristicResult.empty())
        return HIPBLAS_STATUS_NOT_SUPPORTED;

    solutionName = hipblaslt_ext::getSolutionNameFromAlgo(runner.handle, heuristicResult[0].algo);

    gemm.setMaxWorkspaceBytes(runner.max_workspace_size);
    status = gemm.initialize(heuristicResult[0].algo, runner.d_workspace);
    if(status != HIPBLAS_STATUS_SUCCESS)
        return status;

    status = gemm.run(runner.stream);
    if(status == HIPBLAS_STATUS_SUCCESS)
        CHECK_HIP_ERROR(hipStreamSynchronize(runner.stream));
    return status;
}

int main()
{
    /** This is an NN example with
     *  a = (m, k). lda = m, every row of a holds the same k-vector
     *  b = (k, n). ldb = k
     *  c = d = (m, n). ldc = ldd = m
     *  beta = 0, so d depends on a and b only.
     *
     *  With uniform summation order on, every row of d must be bitwise identical.
     *  That is uniformity across m within one run; it says nothing about whether
     *  two separate runs produce the same bits.
     */
    GemmRunner runner(kM, kN, kK, 1, 1.f, 0.f, kMaxWorkspaceSize);

    fillRowReplicatedA(static_cast<float*>(runner.a), runner.m, runner.k);
    fillB(static_cast<float*>(runner.b), runner.k, runner.n);
    runner.hostToDevice();
    CHECK_HIP_ERROR(hipStreamSynchronize(runner.stream));

    std::cout << "fp32 NN GEMM, m=" << runner.m << " n=" << runner.n << " k=" << runner.k
              << ", beta=0, every row of A is the same k-vector" << std::endl;

    std::string     solutionName;
    hipblasStatus_t status;

    poisonD(runner);
    status = runWithCApi(runner, false, solutionName);
    reportRun("Run 1: uniform summation order off (library default)", runner, status, solutionName);

    solutionName.clear();
    poisonD(runner);
    status = runWithCApi(runner, true, solutionName);
    reportRun(
        "Run 2: uniform summation order on (HIPBLASLT_MATMUL_DESC_UNIFORM_SUMMATION_ORDER_EXT)",
        runner,
        status,
        solutionName);

    solutionName.clear();
    poisonD(runner);
    status = runWithExtApi(runner, true, solutionName);
    reportRun("Run 3: uniform summation order on (GemmPreference::setUniformSummationOrder)",
              runner,
              status,
              solutionName);

    return 0;
}
