/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

/*
 * Sample exercising HIPBLASLT_CHECK_NUMERICS on hipblasLtMatmul.
 *
 * Runs two 4x4 FP32 NN matmuls with alpha=1, beta=0:
 *   1) "dirty" -- A = ones with A[0] overwritten to NaN; B = ones.
 *      The first row of D is therefore NaN for every column.
 *   2) "clean" -- A = ones, B = ones. D is the all-fours matrix.
 *
 * Run with the env var to exercise each mode:
 *   HIPBLASLT_CHECK_NUMERICS=1  -> info: one log line per matmul.
 *   HIPBLASLT_CHECK_NUMERICS=2  -> warn: one log line on dirty only.
 *   HIPBLASLT_CHECK_NUMERICS=4  -> fail: dirty returns INVALID_VALUE.
 *   HIPBLASLT_CHECK_NUMERICS=6  -> warn + fail.
 *
 * Without the env var set the program just runs both matmuls and prints D.
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>
#include <iostream>
#include <vector>

#include "helper.h"

namespace
{
constexpr int64_t M = 4;
constexpr int64_t N = 4;
constexpr int64_t K = 4;

void print_d(const char* tag, const std::vector<float>& D)
{
    std::printf("D (%s):\n", tag);
    for(int64_t i = 0; i < M; ++i)
    {
        for(int64_t j = 0; j < N; ++j)
        {
            // Column-major layout matches the matmul call below: D[i + ldd*j].
            const float v = D[i + M * j];
            std::printf("%10.3f ", v);
        }
        std::printf("\n");
    }
}

// Returns the hipBLASLt status of the matmul itself so the caller can confirm
// the fail bit converts NaN-in-D into HIPBLAS_STATUS_INVALID_VALUE.
hipblasStatus_t run_one(hipblasLtHandle_t handle,
                        hipStream_t       stream,
                        const float*      h_A,
                        const float*      h_B,
                        const char*       tag)
{
    // Device buffers. Column-major, ld = M = K = N = 4.
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    float* d_D = nullptr;
    CHECK_HIP_ERROR(hipMalloc(&d_A, sizeof(float) * M * K));
    CHECK_HIP_ERROR(hipMalloc(&d_B, sizeof(float) * K * N));
    CHECK_HIP_ERROR(hipMalloc(&d_C, sizeof(float) * M * N));
    CHECK_HIP_ERROR(hipMalloc(&d_D, sizeof(float) * M * N));

    CHECK_HIP_ERROR(hipMemcpy(d_A, h_A, sizeof(float) * M * K, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_B, h_B, sizeof(float) * K * N, hipMemcpyHostToDevice));
    // beta = 0 so C contents are irrelevant -- zero it for hygiene only.
    CHECK_HIP_ERROR(hipMemset(d_C, 0, sizeof(float) * M * N));
    CHECK_HIP_ERROR(hipMemset(d_D, 0, sizeof(float) * M * N));

    hipblasLtMatrixLayout_t matA{}, matB{}, matC{}, matD{};
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, M));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M));

    hipblasLtMatmulDesc_t matmul{};
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F));

    hipblasOperation_t trans_a = HIPBLAS_OP_N;
    hipblasOperation_t trans_b = HIPBLAS_OP_N;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(int32_t)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(int32_t)));

    hipblasLtEpilogue_t epi = HIPBLASLT_EPILOGUE_DEFAULT;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));

    hipblasLtMatmulPreference_t pref{};
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceCreate(&pref));
    uint64_t max_workspace = 32 * 1024 * 1024;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceSetAttribute(
        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace, sizeof(max_workspace)));

    hipblasLtMatmulHeuristicResult_t heur[1]{};
    int                              ret_count = 0;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulAlgoGetHeuristic(
        handle, matmul, matA, matB, matC, matD, pref, 1, heur, &ret_count));
    if(ret_count == 0)
    {
        std::cerr << "[" << tag << "] no heuristic algorithm returned\n";
        // Tear down everything allocated above before bailing so the sample is
        // a clean reference even on the unhappy path.
        CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(pref));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matD));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
        CHECK_HIP_ERROR(hipFree(d_D));
        CHECK_HIP_ERROR(hipFree(d_C));
        CHECK_HIP_ERROR(hipFree(d_B));
        CHECK_HIP_ERROR(hipFree(d_A));
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    void* d_workspace = nullptr;
    if(heur[0].workspaceSize)
        CHECK_HIP_ERROR(hipMalloc(&d_workspace, heur[0].workspaceSize));

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // Capture status WITHOUT CHECK_HIPBLASLT_ERROR -- under fail mode (4) we
    // *expect* the dirty matmul to return non-success and want to report it,
    // not exit.
    const hipblasStatus_t st = hipblasLtMatmul(handle,
                                               matmul,
                                               &alpha,
                                               d_A,
                                               matA,
                                               d_B,
                                               matB,
                                               &beta,
                                               d_C,
                                               matC,
                                               d_D,
                                               matD,
                                               &heur[0].algo,
                                               d_workspace,
                                               heur[0].workspaceSize,
                                               stream);

    // hipblasLtMatmul is async w.r.t. the host. Sync before reading D back.
    CHECK_HIP_ERROR(hipStreamSynchronize(stream));

    std::vector<float> h_D(M * N, 0.f);
    CHECK_HIP_ERROR(hipMemcpy(h_D.data(), d_D, sizeof(float) * M * N, hipMemcpyDeviceToHost));

    std::printf("[%s] hipblasLtMatmul returned %d", tag, static_cast<int>(st));
    if(st == HIPBLAS_STATUS_SUCCESS)
        std::printf(" (success)\n");
    else if(st == HIPBLAS_STATUS_INVALID_VALUE)
        std::printf(" (INVALID_VALUE -- expected when CHECK_NUMERICS fail bit is set)\n");
    else
        std::printf(" (unexpected)\n");

    int nan_count = 0;
    for(float v : h_D)
        if(std::isnan(v))
            ++nan_count;
    std::printf("[%s] NaNs in returned D: %d / %d\n", tag, nan_count, (int)(M * N));
    print_d(tag, h_D);

    if(d_workspace)
        CHECK_HIP_ERROR(hipFree(d_workspace));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(pref));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matD));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
    CHECK_HIP_ERROR(hipFree(d_D));
    CHECK_HIP_ERROR(hipFree(d_C));
    CHECK_HIP_ERROR(hipFree(d_B));
    CHECK_HIP_ERROR(hipFree(d_A));
    return st;
}

// C++ extension API probe: dispatches through hipblaslt_ext::Gemm::run() ->
// rocblaslt_run_cpp -> runKernelFromInvocation, which is the path the new
// tensile_host.cpp wiring covers. The dirty call should produce a log line
// tagged "hipblasLtMatmul (ext)" under HIPBLASLT_CHECK_NUMERICS=2|6, and
// return INVALID_VALUE under =4|6.
hipblasStatus_t run_one_ext(hipblasLtHandle_t handle,
                            hipStream_t       stream,
                            const float*      h_A,
                            const float*      h_B,
                            const char*       tag)
{
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    float* d_D = nullptr;
    CHECK_HIP_ERROR(hipMalloc(&d_A, sizeof(float) * M * K));
    CHECK_HIP_ERROR(hipMalloc(&d_B, sizeof(float) * K * N));
    CHECK_HIP_ERROR(hipMalloc(&d_C, sizeof(float) * M * N));
    CHECK_HIP_ERROR(hipMalloc(&d_D, sizeof(float) * M * N));

    CHECK_HIP_ERROR(hipMemcpy(d_A, h_A, sizeof(float) * M * K, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_B, h_B, sizeof(float) * K * N, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemset(d_C, 0, sizeof(float) * M * N));
    CHECK_HIP_ERROR(hipMemset(d_D, 0, sizeof(float) * M * N));

    const uint64_t max_workspace = 32 * 1024 * 1024;
    void*          d_workspace   = nullptr;
    CHECK_HIP_ERROR(hipMalloc(&d_workspace, max_workspace));

    hipblaslt_ext::GemmPreference gemmPref;
    gemmPref.setMaxWorkspaceBytes(max_workspace);

    hipblaslt_ext::Gemm gemm(handle,
                             HIPBLAS_OP_N,
                             HIPBLAS_OP_N,
                             HIP_R_32F,
                             HIP_R_32F,
                             HIP_R_32F,
                             HIP_R_32F,
                             HIPBLAS_COMPUTE_32F);

    hipblaslt_ext::GemmEpilogue epilogue; // default
    hipblaslt_ext::GemmInputs   inputs;
    inputs.setA(d_A);
    inputs.setB(d_B);
    inputs.setC(d_C);
    inputs.setD(d_D);
    float alpha = 1.0f;
    float beta  = 0.0f;
    inputs.setAlpha(&alpha);
    inputs.setBeta(&beta);
    CHECK_HIPBLASLT_ERROR(gemm.setProblem(M, N, K, /*batch_count=*/1, epilogue, inputs));

    std::vector<hipblasLtMatmulHeuristicResult_t> heur;
    CHECK_HIPBLASLT_ERROR(gemm.algoGetHeuristic(/*request_solutions=*/1, gemmPref, heur));
    if(heur.empty())
    {
        std::cerr << "[" << tag << "] (ext) no heuristic algorithm returned\n";
        CHECK_HIP_ERROR(hipFree(d_workspace));
        CHECK_HIP_ERROR(hipFree(d_D));
        CHECK_HIP_ERROR(hipFree(d_C));
        CHECK_HIP_ERROR(hipFree(d_B));
        CHECK_HIP_ERROR(hipFree(d_A));
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    gemm.setMaxWorkspaceBytes(max_workspace);
    CHECK_HIPBLASLT_ERROR(gemm.initialize(heur[0].algo, d_workspace));

    // Capture status WITHOUT CHECK_HIPBLASLT_ERROR so fail-bit non-success is
    // observed, not fatal.
    const hipblasStatus_t st = gemm.run(stream);

    CHECK_HIP_ERROR(hipStreamSynchronize(stream));

    std::vector<float> h_D(M * N, 0.f);
    CHECK_HIP_ERROR(hipMemcpy(h_D.data(), d_D, sizeof(float) * M * N, hipMemcpyDeviceToHost));

    std::printf("[%s] (ext) Gemm::run returned %d", tag, static_cast<int>(st));
    if(st == HIPBLAS_STATUS_SUCCESS)
        std::printf(" (success)\n");
    else if(st == HIPBLAS_STATUS_INVALID_VALUE)
        std::printf(" (INVALID_VALUE -- expected when CHECK_NUMERICS fail bit is set)\n");
    else
        std::printf(" (unexpected)\n");

    int nan_count = 0;
    for(float v : h_D)
        if(std::isnan(v))
            ++nan_count;
    std::printf("[%s] (ext) NaNs in returned D: %d / %d\n", tag, nan_count, (int)(M * N));

    CHECK_HIP_ERROR(hipFree(d_workspace));
    CHECK_HIP_ERROR(hipFree(d_D));
    CHECK_HIP_ERROR(hipFree(d_C));
    CHECK_HIP_ERROR(hipFree(d_B));
    CHECK_HIP_ERROR(hipFree(d_A));
    return st;
}
} // namespace

int main()
{
    if(const char* cn = std::getenv("HIPBLASLT_CHECK_NUMERICS"))
        std::printf("HIPBLASLT_CHECK_NUMERICS=\"%s\"\n", cn);
    else
        std::printf("HIPBLASLT_CHECK_NUMERICS not set -- scanner is disabled.\n");

    hipblasLtHandle_t handle = nullptr;
    hipStream_t       stream = nullptr;
    CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));
    CHECK_HIP_ERROR(hipStreamCreate(&stream));

    // ---- dirty ----
    // A is column-major MxK = 4x4. A[0,0] is at host index 0; setting it to
    // qNaN guarantees row 0 of A is NaN (well, A[0,0] is). With B = ones,
    // every dot product reading A[0,*] sums in NaN, so the entire first row
    // of D is NaN.
    std::vector<float> A_dirty(M * K, 1.0f);
    A_dirty[0] = std::nanf(""); // qNaN
    std::vector<float> B_ones(K * N, 1.0f);
    const hipblasStatus_t st_dirty = run_one(handle, stream, A_dirty.data(), B_ones.data(), "dirty");

    // ---- clean control ----
    std::vector<float>    A_ones(M * K, 1.0f);
    const hipblasStatus_t st_clean = run_one(handle, stream, A_ones.data(), B_ones.data(), "clean");

    // ---- C++ extension API probe ----
    // Same dirty + clean pair, but dispatched through hipblaslt_ext::Gemm::run().
    // This exercises the new tensile_host.cpp wiring; expect log lines tagged
    // "hipblasLtMatmul (ext)" under HIPBLASLT_CHECK_NUMERICS=2|6.
    std::printf("\n--- C++ extension API (Gemm::run) probe ---\n");
    const hipblasStatus_t st_dirty_ext
        = run_one_ext(handle, stream, A_dirty.data(), B_ones.data(), "dirty");
    const hipblasStatus_t st_clean_ext
        = run_one_ext(handle, stream, A_ones.data(), B_ones.data(), "clean");

    CHECK_HIP_ERROR(hipStreamDestroy(stream));
    CHECK_HIPBLASLT_ERROR(hipblasLtDestroy(handle));

    // Exit codes:
    //   0 -- ran end to end. (The dirty matmul returning INVALID_VALUE under
    //        CHECK_NUMERICS=4|6 is the *expected* path, not an error.)
    //   1 -- a *clean* call came back non-success; that means the scanner is
    //        producing false positives, which is a real bug to investigate.
    if(st_clean != HIPBLAS_STATUS_SUCCESS)
    {
        std::fprintf(stderr,
                     "FAIL: clean matmul returned %d -- false positive in the scanner?\n",
                     static_cast<int>(st_clean));
        return 1;
    }
    if(st_clean_ext != HIPBLAS_STATUS_SUCCESS)
    {
        std::fprintf(stderr,
                     "FAIL: clean Gemm::run returned %d -- false positive in the scanner?\n",
                     static_cast<int>(st_clean_ext));
        return 1;
    }
    (void)st_dirty;
    (void)st_dirty_ext;
    return 0;
}
