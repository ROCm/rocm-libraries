// SPDX-License-Identifier: MIT
//
// 20-GEMM driver: runs N matmuls in one handle, injecting NaN at one
// configurable call_id. Lets us precisely verify what the scanner reports
// under SCAN_EVERY / SCAN_FROM / SCAN_UNTIL.
//
//   ./sample_check_numerics_multi [N=20] [inject_at=7]
//
// inject_at=0 means "no injection" (clean-run sanity test).
//
// Use cases:
//   inject_at=7, SCAN_EVERY=5 -> calls 5,10,15,20 sampled; #7 is NOT in
//                                that set, so no NaN reported (expected
//                                miss for sampling).
//   inject_at=10, SCAN_EVERY=5 -> #10 IS sampled, expect
//                                  "first NaN observed at sampled matmul call #10".
//   inject_at=12, SCAN_FROM=10 SCAN_UNTIL=15 -> in window, expect #12.
//   inject_at=2, SCAN_FROM=10 SCAN_UNTIL=15 -> outside window, no NaN.

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK_HIP(x)                                                                               \
    do                                                                                             \
    {                                                                                              \
        hipError_t e = (x);                                                                        \
        if(e != hipSuccess)                                                                        \
        {                                                                                          \
            std::fprintf(stderr, "HIP fail %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e)); \
            std::exit(2);                                                                          \
        }                                                                                          \
    } while(0)

#define CHECK_HBL(x)                                                                            \
    do                                                                                          \
    {                                                                                           \
        hipblasStatus_t s = (x);                                                                \
        if(s != HIPBLAS_STATUS_SUCCESS)                                                         \
        {                                                                                       \
            std::fprintf(stderr, "hipBLASLt fail %s:%d (%d)\n", __FILE__, __LINE__, (int)s);    \
            std::exit(2);                                                                       \
        }                                                                                       \
    } while(0)

constexpr int M = 8, N = 8, K = 8;

static hipblasStatus_t one_matmul(hipblasLtHandle_t handle,
                                  hipStream_t       stream,
                                  const float*      hA,
                                  const float*      hB,
                                  int               call_idx_for_log,
                                  bool              expect_nan)
{
    float *dA = nullptr, *dB = nullptr, *dC = nullptr, *dD = nullptr;
    CHECK_HIP(hipMalloc(&dA, sizeof(float) * M * K));
    CHECK_HIP(hipMalloc(&dB, sizeof(float) * K * N));
    CHECK_HIP(hipMalloc(&dC, sizeof(float) * M * N));
    CHECK_HIP(hipMalloc(&dD, sizeof(float) * M * N));
    CHECK_HIP(hipMemcpy(dA, hA, sizeof(float) * M * K, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dB, hB, sizeof(float) * K * N, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemset(dC, 0, sizeof(float) * M * N));

    hipblasLtMatrixLayout_t lA{}, lB{}, lC{}, lD{};
    CHECK_HBL(hipblasLtMatrixLayoutCreate(&lA, HIP_R_32F, M, K, M));
    CHECK_HBL(hipblasLtMatrixLayoutCreate(&lB, HIP_R_32F, K, N, K));
    CHECK_HBL(hipblasLtMatrixLayoutCreate(&lC, HIP_R_32F, M, N, M));
    CHECK_HBL(hipblasLtMatrixLayoutCreate(&lD, HIP_R_32F, M, N, M));

    hipblasLtMatmulDesc_t md{};
    CHECK_HBL(hipblasLtMatmulDescCreate(&md, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    int32_t no_op = HIPBLAS_OP_N;
    CHECK_HBL(hipblasLtMatmulDescSetAttribute(md, HIPBLASLT_MATMUL_DESC_TRANSA, &no_op, sizeof(no_op)));
    CHECK_HBL(hipblasLtMatmulDescSetAttribute(md, HIPBLASLT_MATMUL_DESC_TRANSB, &no_op, sizeof(no_op)));

    hipblasLtMatmulPreference_t pref{};
    CHECK_HBL(hipblasLtMatmulPreferenceCreate(&pref));
    uint64_t ws = 32 * 1024 * 1024;
    CHECK_HBL(hipblasLtMatmulPreferenceSetAttribute(
        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));

    hipblasLtMatmulHeuristicResult_t heur[1]{};
    int                              ret = 0;
    CHECK_HBL(hipblasLtMatmulAlgoGetHeuristic(handle, md, lA, lB, lC, lD, pref, 1, heur, &ret));
    if(ret == 0)
    {
        std::fprintf(stderr, "no algo for call #%d\n", call_idx_for_log);
        std::exit(2);
    }

    void* dW = nullptr;
    if(heur[0].workspaceSize > 0)
        CHECK_HIP(hipMalloc(&dW, heur[0].workspaceSize));

    float alpha = 1.0f, beta = 0.0f;
    hipblasStatus_t st = hipblasLtMatmul(
        handle, md, &alpha, dA, lA, dB, lB, &beta, dC, lC, dD, lD, &heur[0].algo,
        dW, heur[0].workspaceSize, stream);
    CHECK_HIP(hipStreamSynchronize(stream));

    std::vector<float> hD(M * N);
    CHECK_HIP(hipMemcpy(hD.data(), dD, sizeof(float) * M * N, hipMemcpyDeviceToHost));
    int nans = 0;
    for(float v : hD)
        if(std::isnan(v))
            ++nans;
    std::printf("[#%2d] status=%d nans=%d/%d %s%s\n",
                call_idx_for_log, (int)st, nans, M * N,
                expect_nan ? "(expect NaN)" : "(expect clean)",
                ((expect_nan && nans > 0) || (!expect_nan && nans == 0)) ? " OK" : " MISMATCH");

    if(dW)
        CHECK_HIP(hipFree(dW));
    CHECK_HIP(hipFree(dD));
    CHECK_HIP(hipFree(dC));
    CHECK_HIP(hipFree(dB));
    CHECK_HIP(hipFree(dA));
    hipblasLtMatmulPreferenceDestroy(pref);
    hipblasLtMatmulDescDestroy(md);
    hipblasLtMatrixLayoutDestroy(lD);
    hipblasLtMatrixLayoutDestroy(lC);
    hipblasLtMatrixLayoutDestroy(lB);
    hipblasLtMatrixLayoutDestroy(lA);
    return st;
}

int main(int argc, char** argv)
{
    int total     = (argc >= 2) ? std::atoi(argv[1]) : 20;
    int inject_at = (argc >= 3) ? std::atoi(argv[2]) : 7;
    if(total < 1)
        total = 20;
    // inject_at == 0 is an explicit opt-out: no NaN is ever injected
    // (used for clean-run sanity tests). Other out-of-range values are
    // clamped so accidental misconfiguration still exercises the scanner.
    if(inject_at != 0 && (inject_at < 1 || inject_at > total))
        inject_at = total / 2;

    if(const char* cn = std::getenv("HIPBLASLT_CHECK_NUMERICS"))
        std::printf("HIPBLASLT_CHECK_NUMERICS=\"%s\"  total=%d  inject_at=%d\n", cn, total, inject_at);
    else
        std::printf("HIPBLASLT_CHECK_NUMERICS not set. total=%d  inject_at=%d\n", total, inject_at);

    hipblasLtHandle_t handle = nullptr;
    hipStream_t       stream = nullptr;
    CHECK_HBL(hipblasLtCreate(&handle));
    CHECK_HIP(hipStreamCreate(&stream));

    std::vector<float> A_clean(M * K, 1.0f);
    std::vector<float> B(K * N, 1.0f);
    std::vector<float> A_dirty(M * K, 1.0f);
    A_dirty[0] = std::nanf("");

    for(int i = 1; i <= total; ++i)
    {
        const bool dirty = (i == inject_at);
        one_matmul(handle, stream,
                   dirty ? A_dirty.data() : A_clean.data(),
                   B.data(), i, dirty);
    }

    CHECK_HIP(hipStreamDestroy(stream));
    CHECK_HBL(hipblasLtDestroy(handle));
    return 0;
}
