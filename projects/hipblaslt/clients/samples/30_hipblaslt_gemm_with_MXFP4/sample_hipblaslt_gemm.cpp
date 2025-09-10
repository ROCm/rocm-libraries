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

void dumpBuffer(const char* str, float* data, size_t M, size_t N)
{
    std::cout << "----- " << str << " -----" <<std::endl;
    for(size_t n=0; n<N; n++)
    {
        for(size_t m=0; m<M; m++)
        {
            std::cout << data[m+n*M] << " ";
        }
        std::cout << std::endl;
    }
}

void initF4(hipblaslt_f4x2* packedData, size_t size)
{
    for(size_t i = 0; i < size/2; i++)
    {
        float x       = rand() % 7 - 3;
        float y       = rand() % 7 - 3;
        packedData[i] = hipblaslt_f4x2(x, y);
    }
}

template<typename T>
void initScale(T* data, size_t Z)
{
}

void initScale(hipblaslt_e8* data, size_t Z)
{
    for(size_t z=0; z<Z; z++)
    {
        data[z].data = (rand() % 7 - 3) + 127;
    }
}

void initScale(hipblaslt_f8* data, size_t Z)
{
    for(size_t z=0; z<Z; z++)
    {
        data[z] = rand() % 7;
    }
}

void unpackData(float* unpackedData, const hipblaslt_f4x2* packedData, size_t size)
{
    for(size_t i = 0; i < size / 2; i++)
    {
        unpackedData[2 * i] = packedData[i].castElement(0);
        unpackedData[2 * i + 1] = packedData[i].castElement(1);
    }
}

size_t getIndex(size_t m, size_t k, size_t M, size_t K, bool unrollMajor)
{
    if (unrollMajor)
        return m * K + k;
    else
        return m + k * M;
}

size_t getScaleIndex(size_t m, size_t k, size_t M, size_t K, size_t B, bool unrollMajor)
{
    if (unrollMajor)
        return m * (K / B) + k / B;
    else
        return (k / B) * M + m;
}

template <typename T>
void CPUMatMul(float* cpuC, const float* cpuA, const float* cpuB, const T* sa, const T* sb, size_t M, size_t N, size_t K, size_t B, bool transA, bool transB)
{
    bool unrollMajorA = transA;
    bool unrollMajorB = !transB;

    for(size_t n = 0; n < N; n++)
    {
        for(size_t m = 0; m < M; m++)
        {
            cpuC[n * M + m] = 0.0f;
            for(size_t k = 0; k < K; k+=B)
            {
                float accm = 0.0f;
                for(size_t b=0; b<B; b++)
                {
                    size_t bk = k + b;
                    float valA = cpuA[getIndex(m, bk, M, K, unrollMajorA)];
                    float valB = cpuB[getIndex(n, bk, N, K, unrollMajorB)];
                    accm += valA * valB;
                }
                size_t saIdx = getScaleIndex(m, k, M, K, B, unrollMajorA);
                size_t sbIdx = getScaleIndex(n, k, N, K, B, unrollMajorB);
                cpuC[n * M + m] += (accm * sa[saIdx] * sb[sbIdx]);
            }
        }
    }
}

size_t validate(const std::vector<float>& cpuR, const std::vector<float>& gpuR, size_t M, size_t N)
{
    for(size_t n = 0; n < N; n++)
    {
        for(size_t m = 0; m < M; m++)
        {
            float err = std::abs(float(cpuR[m + n * M]) - float(gpuR[m + n * M]));
            if(err > 1e-5)
            {
                return -1;
            }
        }
    }

    return 0;
}

size_t getDim0(size_t m, size_t k, bool unrollMajor)
{
    return unrollMajor ? k : m;
}

size_t getDim1(size_t m, size_t k, bool unrollMajor)
{
    return unrollMajor ? m : k;
}

template<typename T>
void GPUMatMul(float* d, hipblaslt_f4x2* a, hipblaslt_f4x2* b, T* sa, T* sb, size_t m, size_t n, size_t k, size_t bk, bool transA, bool transB)
{
    bool unrollMajorA = transA;
    bool unrollMajorB = !transB;

    hipStream_t       stream;
    hipblasLtHandle_t handle;
    hipblasOperation_t trans_a = (transA ? HIPBLAS_OP_T : HIPBLAS_OP_N);
    hipblasOperation_t trans_b = (transB ? HIPBLAS_OP_T : HIPBLAS_OP_N);

    void *d_a, *d_b, *d_sa, *d_sb, *d_c, *d_d, *d_workspace; // device
    float alpha = 1;
    float beta  = 0;
    int64_t max_workspace_size = 32*32*1024;

    CHECK_HIP_ERROR(hipStreamCreate(&stream));
    CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));
    CHECK_HIP_ERROR(hipMalloc(&d_a,  m * k * sizeof(hipblaslt_f4x2) / 2));
    CHECK_HIP_ERROR(hipMalloc(&d_b,  n * k * sizeof(hipblaslt_f4x2) / 2));
    CHECK_HIP_ERROR(hipMalloc(&d_sa, m * k / bk * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&d_sb, n * k / bk * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&d_c,  m * n * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_d,  m * n * sizeof(float)));

    CHECK_HIP_ERROR(hipMalloc(&d_workspace, 32*32*1024));

    CHECK_HIP_ERROR(hipMemcpyAsync(d_a,  a, m * k * sizeof(hipblaslt_f4x2) / 2, hipMemcpyHostToDevice, stream));
    CHECK_HIP_ERROR(hipMemcpyAsync(d_b,  b, n * k * sizeof(hipblaslt_f4x2) / 2, hipMemcpyHostToDevice, stream));
    CHECK_HIP_ERROR(hipMemcpyAsync(d_sa, sa, m * k / bk * sizeof(T), hipMemcpyHostToDevice, stream));
    CHECK_HIP_ERROR(hipMemcpyAsync(d_sb, sb, n * k / bk * sizeof(T), hipMemcpyHostToDevice, stream));
    CHECK_HIP_ERROR(hipMemsetAsync(d_c,  0, m * n * sizeof(float), stream));

    hipblasLtMatrixLayout_t matA, matB, matC, matD;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matA, static_cast<hipDataType>(HIP_R_4F_E2M1_EXT), getDim0(m, k, unrollMajorA), getDim1(m, k, unrollMajorA), getDim0(m, k, unrollMajorA)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matB, static_cast<hipDataType>(HIP_R_4F_E2M1_EXT), getDim0(n, k, unrollMajorA), getDim1(n, k, unrollMajorA), getDim0(n, k, unrollMajorB)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, m, n, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, m, n, m));

    hipblasLtMatmulDesc_t matmul;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(int32_t)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(int32_t)));

    auto mode = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    if (std::is_same<T, hipblaslt_e8>::value)
        mode = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    else
        mode = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;

    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(uint32_t)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_sa, sizeof(void*)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(uint32_t)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_sb, sizeof(void*)));

    hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

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

    CHECK_HIP_ERROR(hipMemcpyAsync(d, d_d, m * n * sizeof(float), hipMemcpyDeviceToHost, stream));

    CHECK_HIP_ERROR(hipStreamSynchronize(stream));

    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matD));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(pref));

    CHECK_HIP_ERROR(hipFree(d_workspace));
    CHECK_HIP_ERROR(hipFree(d_a));
    CHECK_HIP_ERROR(hipFree(d_b));
    CHECK_HIP_ERROR(hipFree(d_sa));
    CHECK_HIP_ERROR(hipFree(d_sb));
    CHECK_HIP_ERROR(hipFree(d_c));
    CHECK_HIP_ERROR(hipFree(d_d));
    CHECK_HIPBLASLT_ERROR(hipblasLtDestroy(handle));
    CHECK_HIP_ERROR(hipStreamDestroy(stream));

    return;
}

template<typename T>
void outputResult(size_t M, size_t N, size_t K, size_t B, bool transA, bool transB, size_t ret)
{
    std::cout << "Test MX" << B << (std::is_same<T, hipblaslt_e8>::value ? "E8" : "UF8") << " F4 ";
    std::cout << (transA ? "T" : "N") << (transB ? "T" : "N") << " GEMM";
    std::cout << " M " << M;
    std::cout << " N " << N;
    std::cout << " K " << K;
    std::cout << " : " << ((ret == 0) ? "PASS" : "FAIL");
    std::cout << std::endl;
}

template<typename T>
int MXF4Test(size_t M, size_t N, size_t K, size_t B, bool transA, bool transB)
{
    /** This is an example using hipblaslt extension API.
     *  This is a TN example with
     *  a = (k, m). lda = k
     *  b = (k, n). ldb = k
     *  c = d = (m, n). ldc = ldd = m
     */

    int ret = 0;

    hipblaslt_f4x2* cpuPA = (hipblaslt_f4x2*)malloc(M * K / 2);
    hipblaslt_f4x2* cpuPB = (hipblaslt_f4x2*)malloc(N * K / 2);
    initF4(cpuPA, M * K);
    initF4(cpuPB, N * K);

    std::vector<T> cpuSA(M * K / B);
    std::vector<T> cpuSB(N * K / B);
    initScale(cpuSA.data(), M * K / B);
    initScale(cpuSB.data(), N * K / B);

    std::vector<float> cpuA(M * K, 0.f);
    std::vector<float> cpuB(N * K, 0.f);
    unpackData(cpuA.data(), cpuPA, M * K);
    unpackData(cpuB.data(), cpuPB, N * K );

    std::vector<float> cpuR(M * N, 0.f);
    CPUMatMul(cpuR.data(), cpuA.data(), cpuB.data(), cpuSA.data(), cpuSB.data(), M, N, K, B, transA, transB);

    std::vector<float> cpuD(M * N, 0.f);
    GPUMatMul(cpuD.data(), cpuPA, cpuPB, cpuSA.data(), cpuSB.data(), M, N, K, B, transA, transB);

    ret = validate(cpuR, cpuD, M, N);

    outputResult<T>(M, N, K, B, transA, transB, ret);

    if (ret)
    {
        dumpBuffer("R", cpuR.data(), M, N);
        dumpBuffer("D", cpuD.data(), M, N);
    }

    free(cpuPA);
    free(cpuPB);

    return ret;
}

int main(int argc, char** argv)
{
    size_t M = 16;
    size_t N = 16;
    size_t K = 128;

    if (argc == 4)
    {
        M = std::stoi(argv[1]);
        N = std::stoi(argv[2]);
        K = std::stoi(argv[3]);
    }

    /*----- MX32 E8 F4 Tests ----*/
    MXF4Test<hipblaslt_e8>(M, N, K, 32, true,  false);
    MXF4Test<hipblaslt_e8>(M, N, K, 32, false, false);
    MXF4Test<hipblaslt_e8>(M, N, K, 32, true,  true);
    MXF4Test<hipblaslt_e8>(M, N, K, 32, false, true);

    /*----- MX16 F8 F4 Tests ----*/
    MXF4Test<hipblaslt_f8>(M, N, K, 16, true,  false);
    MXF4Test<hipblaslt_f8>(M, N, K, 16, false, false);
    MXF4Test<hipblaslt_f8>(M, N, K, 16, true,  true);
    MXF4Test<hipblaslt_f8>(M, N, K, 16, false, true);

    return 0;
}
