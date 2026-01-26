/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <iostream>

#include "helper.h"

void simpleGemmF4(hipblasLtHandle_t  handle,
                  hipblasOperation_t trans_a,
                  hipblasOperation_t trans_b,
                  int64_t            m,
                  int64_t            n,
                  int64_t            k,
                  int64_t            batch_count,
                  float&             alpha,
                  float&             beta,
                  void*              d_a,
                  void*              d_b,
                  void*              d_c,
                  void*              d_d,
                  void*              d_workspace,
                  int64_t            max_workspace_size,
                  hipStream_t        stream);

void init(hipblaslt_f4x2* packedData, size_t size)
{
    for(size_t i = 0; i < size; i += 2)
    {
        float x       = rand() % 7 - 3;
        float y       = rand() % 7 - 3;
        packedData[i] = hipblaslt_f4x2(x, y);
    }
}

void UnpackData(float* unpackedData, const hipblaslt_f4x2* packedData, size_t size)
{
    for(size_t i = 0; i < size / 2; i++)
    {
        unpackedData[2 * i] = float(static_cast<hipblaslt_f4x2>((packedData[i].__x & 0x0F)));
        unpackedData[2 * i + 1]
            = float(static_cast<hipblaslt_f4x2>(((packedData[i].__x & 0xF0) >> 4) & 0x0F));
    }
}

void CPUMatMul(float* cpuC, const float* cpuA, const float* cpuB, size_t M, size_t N, size_t K)
{
    for(size_t n = 0; n < N; n++)
    {
        for(size_t m = 0; m < M; m++)
        {
            float accm = 0.0f;
            for(size_t k = 0; k < K; k++)
            {
                float valA = cpuA[m * K + k];
                float valB = cpuB[n * K + k];
                accm += valA * valB;
            }
            cpuC[n * M + m] += accm;
        }
    }
}

void validate(const std::vector<float>& cpuR, const std::vector<float>& gpuR, size_t M, size_t N)
{
    for(size_t n = 0; n < N; n++)
    {
        for(size_t m = 0; m < M; m++)
        {
            float err = std::abs(float(cpuR[m + n * M]) - float(gpuR[m + n * M]));
            if(err > 1e-5)
            {
                std::cout << "Error " << err << " Ref " << float(cpuR[m + n * M]) << " GPU "
                          << gpuR[m + n * M] << std::endl;
                return;
            }
        }
    }
    std::cout << "PASS" << std::endl;
}

int main()
{
    /** This is an example using hipblaslt extension API.
     *  This is a TN example with
     *  a = (k, m). lda = k
     *  b = (k, n). ldb = k
     *  c = d = (m, n). ldc = ldd = m
     */
    constexpr size_t batch_count = 1;
    Runner<hipblaslt_f4x2, hipblaslt_f4x2, float, float, float> runner(
        1024 /*m*/,
        512 /*n*/,
        1024 /*k*/,
        batch_count,
        1.f,
        0.f,
        32 * 1024 * 1024 /*max_workspace_size_in_bytes*/);
    init(static_cast<hipblaslt_f4x2*>(runner.a), runner.m * runner.k);
    init(static_cast<hipblaslt_f4x2*>(runner.b), runner.n * runner.k);

    std::vector<float> cpuR(runner.m * runner.n, 0.f);
    std::vector<float> cpuA(runner.m * runner.k, 0.f);
    std::vector<float> cpuB(runner.k * runner.n, 0.f);
    UnpackData(cpuA.data(), static_cast<hipblaslt_f4x2*>(runner.a), runner.m * runner.k);
    UnpackData(cpuB.data(), static_cast<hipblaslt_f4x2*>(runner.b), runner.n * runner.k);
    CPUMatMul(cpuR.data(), cpuA.data(), cpuB.data(), runner.m, runner.n, runner.k);
    runner.run([&runner] {
        simpleGemmF4(runner.handle,
                   HIPBLAS_OP_T,
                   HIPBLAS_OP_N,
                   runner.m,
                   runner.n,
                   runner.k,
                   runner.batch_count,
                   runner.alpha,
                   runner.beta,
                   runner.d_a,
                   runner.d_b,
                   runner.d_c,
                   runner.d_d,
                   runner.d_workspace,
                   runner.max_workspace_size,
                   runner.stream);
    });

    std::vector<float> gpuR(runner.m * runner.n, 0.f);
    hipMemcpyDtoH(
        gpuR.data(), runner.d_d, runner.batch_count * runner.m * runner.n * sizeof(float));
    validate(cpuR, gpuR, runner.m, runner.n);
    return 0;
}

void simpleGemmF4(hipblasLtHandle_t  handle,
                  hipblasOperation_t trans_a,
                  hipblasOperation_t trans_b,
                  int64_t            m,
                  int64_t            n,
                  int64_t            k,
                  int64_t            batch_count,
                  float&             alpha,
                  float&             beta,
                  void*              d_a,
                  void*              d_b,
                  void*              d_c,
                  void*              d_d,
                  void*              d_workspace,
                  int64_t            max_workspace_size,
                  hipStream_t        stream)
{
    hipblasLtMatrixLayout_t matA, matB, matC, matD;
    CHECK_HIPBLASLT_ERROR(
        hipblasLtMatrixLayoutCreate(&matA, static_cast<hipDataType>(HIP_R_4F_E2M1_EXT), k, m, k));
    CHECK_HIPBLASLT_ERROR(
        hipblasLtMatrixLayoutCreate(&matB, static_cast<hipDataType>(HIP_R_4F_E2M1_EXT), k, n, k));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, m, n, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, m, n, m));

    hipblasLtMatmulDesc_t matmul;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(int32_t)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(int32_t)));

    hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    // Set User Preference attributes
    hipblasLtMatmulPreference_t pref;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceCreate(&pref));
    CHECK_HIPBLASLT_ERROR(
        hipblasLtMatmulPreferenceSetAttribute(pref,
                                              HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &max_workspace_size,
                                              sizeof(max_workspace_size)));

    const int                        request_solutions = 1;
    hipblasLtMatmulHeuristicResult_t heuristicResult[request_solutions];
    int                              returnedAlgoCount = 0;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulAlgoGetHeuristic(handle,
                                                          matmul,
                                                          matA,
                                                          matB,
                                                          matC,
                                                          matD,
                                                          pref,
                                                          request_solutions,
                                                          heuristicResult,
                                                          &returnedAlgoCount));

    if(returnedAlgoCount == 0)
    {
        std::cerr << "No valid solution found!" << std::endl;
    }
    else
    {
        uint64_t workspace_size = 0;
        for(int i = 0; i < returnedAlgoCount; i++)
            workspace_size = max(workspace_size, heuristicResult[i].workspaceSize);
        // In this sample, the workspace is already allocated with max_workspace_size
        // If not, allocate d_workspace here
        // CHECK_HIP_ERROR(hipMalloc(&d_workspace, workspace_size));

        CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(handle,
                                              matmul,
                                              &alpha,
                                              d_a,
                                              matA,
                                              d_b,
                                              matB,
                                              &beta,
                                              d_c,
                                              matC,
                                              d_d,
                                              matD,
                                              &heuristicResult[0].algo,
                                              d_workspace,
                                              workspace_size,
                                              stream));
    }

    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matD));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(pref));
    return;
}

