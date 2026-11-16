// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// [sphinx-start]
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK_HIP(expr)                                                   \
    do {                                                                  \
        hipError_t status_ = (expr);                                      \
        if(status_ != hipSuccess) {                                       \
            std::fprintf(stderr, "HIP error %s at %s:%d\n",               \
                         hipGetErrorString(status_), __FILE__, __LINE__); \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while(0)

#define CHECK_HIPBLASLT(expr)                                             \
    do {                                                                  \
        hipblasStatus_t status_ = (expr);                                 \
        if(status_ != HIPBLAS_STATUS_SUCCESS) {                           \
            std::fprintf(stderr, "hipBLASLt error %d at %s:%d\n",         \
                         static_cast<int>(status_), __FILE__, __LINE__);  \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while(0)

int main()
{
    const int64_t m = 8192, n = 8192, k = 8192;
    const double  alpha = 1.0, beta = 0.0;

    // Column-major A (m x k) and B (k x n), with values in [-0.5, 0.5].
    std::vector<double> hA(m * k), hB(k * n);
    for(auto& x : hA) x = std::rand() / static_cast<double>(RAND_MAX) - 0.5;
    for(auto& x : hB) x = std::rand() / static_cast<double>(RAND_MAX) - 0.5;

    double *dA, *dB, *dD;
    CHECK_HIP(hipMalloc(&dA, hA.size() * sizeof(double)));
    CHECK_HIP(hipMalloc(&dB, hB.size() * sizeof(double)));
    CHECK_HIP(hipMalloc(&dD, m * n * sizeof(double)));
    CHECK_HIP(hipMemcpy(dA, hA.data(), hA.size() * sizeof(double), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dB, hB.data(), hB.size() * sizeof(double), hipMemcpyHostToDevice));

    hipblasLtHandle_t handle;
    hipStream_t       stream;
    CHECK_HIPBLASLT(hipblasLtCreate(&handle));
    CHECK_HIP(hipStreamCreate(&stream));

    hipblasLtMatrixLayout_t layoutA, layoutB, layoutD;
    CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(&layoutA, HIP_R_64F, m, k, m));
    CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(&layoutB, HIP_R_64F, k, n, k));
    CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(&layoutD, HIP_R_64F, m, n, m));

    // Request FP64 emulation through the compute type.
    hipblasLtMatmulDesc_t matmul;
    CHECK_HIPBLASLT(hipblasLtMatmulDescCreate(
        &matmul, HIPBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, HIP_R_64F));

    // Lower the precision target to about 1e-8 relative accuracy.
    hipblasLtEmulationDesc_t emulation;
    CHECK_HIPBLASLT(hipblasLtEmulationDescCreate(&emulation));
    int32_t mantissa_bits = 26;
    CHECK_HIPBLASLT(hipblasLtEmulationDescSetAttribute(
        emulation, HIPBLASLT_EMULATION_DESC_FIXEDPOINT_MAX_MANTISSA_BIT_COUNT,
        &mantissa_bits, sizeof(mantissa_bits)));
    CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_EMULATION_DESCRIPTOR,
        &emulation, sizeof(emulation)));

    // Allocate the workspace. A size of 0 means emulation can't apply,
    // and the GEMM runs natively.
    size_t ws_bytes  = hipblasLtEmulationWorkspaceSize(handle, matmul, HIP_R_64F, m, n, k, 1);
    void*  workspace = nullptr;
    if(ws_bytes > 0)
        CHECK_HIP(hipMalloc(&workspace, ws_bytes));

    // Select an algorithm. The native GEMM uses it if emulation falls back.
    hipblasLtMatmulPreference_t pref;
    CHECK_HIPBLASLT(hipblasLtMatmulPreferenceCreate(&pref));
    uint64_t max_ws = ws_bytes;
    CHECK_HIPBLASLT(hipblasLtMatmulPreferenceSetAttribute(
        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_ws, sizeof(max_ws)));

    hipblasLtMatmulHeuristicResult_t heuristic;
    int                              algo_count = 0;
    CHECK_HIPBLASLT(hipblasLtMatmulAlgoGetHeuristic(handle, matmul,
                                                    layoutA, layoutB, layoutD, layoutD,
                                                    pref, 1, &heuristic, &algo_count));
    if(algo_count == 0)
    {
        std::fprintf(stderr, "No algorithm found\n");
        return EXIT_FAILURE;
    }

    // D = alpha * A * B + beta * C. With beta = 0, C isn't read, so D is passed as C.
    CHECK_HIPBLASLT(hipblasLtMatmul(handle, matmul,
                                    &alpha, dA, layoutA, dB, layoutB,
                                    &beta, dD, layoutD, dD, layoutD,
                                    &heuristic.algo, workspace, ws_bytes, stream));
    CHECK_HIP(hipStreamSynchronize(stream));

    // Destroy the emulation descriptor only after the last matmul that uses it.
    CHECK_HIPBLASLT(hipblasLtMatmulPreferenceDestroy(pref));
    CHECK_HIPBLASLT(hipblasLtEmulationDescDestroy(emulation));
    CHECK_HIPBLASLT(hipblasLtMatmulDescDestroy(matmul));
    CHECK_HIPBLASLT(hipblasLtMatrixLayoutDestroy(layoutA));
    CHECK_HIPBLASLT(hipblasLtMatrixLayoutDestroy(layoutB));
    CHECK_HIPBLASLT(hipblasLtMatrixLayoutDestroy(layoutD));
    CHECK_HIPBLASLT(hipblasLtDestroy(handle));
    CHECK_HIP(hipStreamDestroy(stream));
    CHECK_HIP(hipFree(workspace));
    CHECK_HIP(hipFree(dA));
    CHECK_HIP(hipFree(dB));
    CHECK_HIP(hipFree(dD));
    return EXIT_SUCCESS;
}
// [sphinx-end]
