// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// w4a16: D = alpha * dequant(A) * B + beta * C, where
//   A is HIP_R_4I    -- signed int4 weights, two per byte, element 2n low
//   B is HIP_R_16BF  -- bf16 activations
//   the A scale is a dense [M][ceil(K/G)] bf16 tensor, one scale per G
//   consecutive K elements of a row of A (G = 32 or 128), symmetric,
//   selected with HIPBLASLT_MATMUL_MATRIX_SCALE_VEC{32,128}_EXT.
//
// Standalone (no Runner<> helper): there is no host arithmetic type for int4.

#include <hip/hip_runtime.h>
#include <hip/library_types.h>
#include <hipblaslt/hipblaslt.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

#define CHECK_HIP(x)                                                                       \
    do                                                                                     \
    {                                                                                      \
        hipError_t e = (x);                                                                \
        if(e != hipSuccess)                                                                \
        {                                                                                  \
            std::cerr << __FILE__ << ":" << __LINE__ << " " << hipGetErrorString(e) << "\n"; \
            return 1;                                                                      \
        }                                                                                  \
    } while(0)

#define CHECK_LT(x)                                                                    \
    do                                                                                 \
    {                                                                                  \
        hipblasStatus_t s = (x);                                                       \
        if(s != HIPBLAS_STATUS_SUCCESS)                                                \
        {                                                                              \
            std::cerr << __FILE__ << ":" << __LINE__ << " hipblaslt status " << s << "\n"; \
            return 1;                                                                  \
        }                                                                              \
    } while(0)

static uint16_t f32_to_bf16(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u); // round to nearest even
    return static_cast<uint16_t>(u >> 16);
}
static float bf16_to_f32(uint16_t h)
{
    uint32_t u = static_cast<uint32_t>(h) << 16;
    float    f;
    std::memcpy(&f, &u, 4);
    return f;
}

int main()
{
    // Column-major TN: A is k x m (lda = k), B is k x n (ldb = k), so A is
    // row-major [M][K] with K contiguous, as the scale layout assumes.
    const int64_t m = 128, n = 128, k = 256;
    const int64_t groupSize = 32;
    const int64_t nGroups   = (k + groupSize - 1) / groupSize;
    const float   alpha = 1.0f, beta = 0.0f;

    std::mt19937                       rng(1234);
    std::uniform_int_distribution<int> q4(-8, 7);
    std::uniform_int_distribution<int> sm(-4, 4);
    std::uniform_int_distribution<int> ex(-2, 2);

    std::vector<int8_t>   aQ(size_t(m) * k);
    std::vector<uint16_t> bH(size_t(n) * k), scaleA(size_t(m) * nGroups);
    for(auto& v : aQ) v = static_cast<int8_t>(q4(rng));
    for(auto& v : bH) v = f32_to_bf16(float(sm(rng)));
    for(auto& v : scaleA) v = f32_to_bf16(std::ldexp(1.0f, ex(rng)));

    std::vector<uint8_t> aPacked(size_t(m) * k / 2);
    for(size_t i = 0; i < aPacked.size(); i++)
        aPacked[i] = uint8_t((aQ[2 * i] & 0xF) | ((aQ[2 * i + 1] & 0xF) << 4));

    void *dA, *dB, *dC, *dD, *dScaleA, *dWs;
    const size_t wsSize = 32 * 1024 * 1024;
    CHECK_HIP(hipMalloc(&dA, aPacked.size()));
    CHECK_HIP(hipMalloc(&dB, bH.size() * 2));
    CHECK_HIP(hipMalloc(&dC, size_t(m) * n * 2));
    CHECK_HIP(hipMalloc(&dD, size_t(m) * n * 2));
    CHECK_HIP(hipMalloc(&dScaleA, scaleA.size() * 2));
    CHECK_HIP(hipMalloc(&dWs, wsSize));
    CHECK_HIP(hipMemcpy(dA, aPacked.data(), aPacked.size(), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dB, bH.data(), bH.size() * 2, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemset(dC, 0, size_t(m) * n * 2));
    CHECK_HIP(hipMemcpy(dScaleA, scaleA.data(), scaleA.size() * 2, hipMemcpyHostToDevice));

    hipblasLtHandle_t handle;
    CHECK_LT(hipblasLtCreate(&handle));

    hipblasLtMatrixLayout_t layoutA, layoutB, layoutC, layoutD;
    // A is transposed (TN), so its layout is k x m.
    CHECK_LT(hipblasLtMatrixLayoutCreate(&layoutA, HIP_R_4I, k, m, k));
    CHECK_LT(hipblasLtMatrixLayoutCreate(&layoutB, HIP_R_16BF, k, n, k));
    CHECK_LT(hipblasLtMatrixLayoutCreate(&layoutC, HIP_R_16BF, m, n, m));
    CHECK_LT(hipblasLtMatrixLayoutCreate(&layoutD, HIP_R_16BF, m, n, m));

    hipblasLtMatmulDesc_t desc;
    CHECK_LT(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    hipblasOperation_t opA = HIPBLAS_OP_T, opB = HIPBLAS_OP_N;
    CHECK_LT(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    CHECK_LT(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));

    // Mode before pointer: a pointer set while the mode is None defaults to Scalar.
    hipblasLtMatmulMatrixScale_t scaleMode = (groupSize == 32)
                                                 ? HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_EXT
                                                 : HIPBLASLT_MATMUL_MATRIX_SCALE_VEC128_EXT;
    CHECK_LT(hipblasLtMatmulDescSetAttribute(
        desc, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
    CHECK_LT(hipblasLtMatmulDescSetAttribute(
        desc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dScaleA, sizeof(dScaleA)));

    hipblasLtMatmulPreference_t pref;
    CHECK_LT(hipblasLtMatmulPreferenceCreate(&pref));
    CHECK_LT(hipblasLtMatmulPreferenceSetAttribute(
        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsSize, sizeof(wsSize)));

    hipblasLtMatmulHeuristicResult_t heuristic[1];
    int                              nFound = 0;
    CHECK_LT(hipblasLtMatmulAlgoGetHeuristic(
        handle, desc, layoutA, layoutB, layoutC, layoutD, pref, 1, heuristic, &nFound));
    if(nFound == 0)
    {
        std::cerr << "no w4a16 solution found for this shape\n";
        return 1;
    }

    hipStream_t stream;
    CHECK_HIP(hipStreamCreate(&stream));
    CHECK_LT(hipblasLtMatmul(handle, desc, &alpha, dA, layoutA, dB, layoutB, &beta, dC, layoutC,
                             dD, layoutD, &heuristic[0].algo, dWs, wsSize, stream));
    CHECK_HIP(hipStreamSynchronize(stream));

    std::vector<uint16_t> got(size_t(m) * n);
    CHECK_HIP(hipMemcpy(got.data(), dD, got.size() * 2, hipMemcpyDeviceToHost));

    // D is bf16, so the expected error is its own rounding: one ulp is 2^-8 of
    // the magnitude. An absolute number would say nothing on its own.
    double maxErr = 0, maxRef = 0, maxUlps = 0;
    for(int64_t j = 0; j < n; j++)
        for(int64_t i = 0; i < m; i++)
        {
            double acc = 0;
            for(int64_t kk = 0; kk < k; kk++)
                acc += double(aQ[size_t(i) * k + kk])
                       * bf16_to_f32(scaleA[size_t(i) * nGroups + kk / groupSize])
                       * bf16_to_f32(bH[size_t(j) * k + kk]);
            const double ref = alpha * acc;
            const double err = std::fabs(bf16_to_f32(got[size_t(j) * m + i]) - ref);
            maxErr = std::max(maxErr, err);
            maxRef = std::max(maxRef, std::fabs(ref));
            if(ref != 0)
                maxUlps = std::max(maxUlps, err / std::ldexp(std::fabs(ref), -8));
        }
    const bool ok = maxUlps <= 2.0;
    std::cout << "w4a16 " << m << "x" << n << "x" << k << " G=" << groupSize
              << " max_abs_err=" << maxErr << " max|ref|=" << maxRef
              << " max_bf16_ulps=" << maxUlps << (ok ? "  PASS\n" : "  FAIL\n");
    if(!ok)
        return 1;

    CHECK_LT(hipblasLtMatmulPreferenceDestroy(pref));
    CHECK_LT(hipblasLtMatmulDescDestroy(desc));
    CHECK_LT(hipblasLtMatrixLayoutDestroy(layoutD));
    CHECK_LT(hipblasLtMatrixLayoutDestroy(layoutC));
    CHECK_LT(hipblasLtMatrixLayoutDestroy(layoutB));
    CHECK_LT(hipblasLtMatrixLayoutDestroy(layoutA));
    CHECK_LT(hipblasLtDestroy(handle));
    std::cout << "done\n";
    return 0;
}
