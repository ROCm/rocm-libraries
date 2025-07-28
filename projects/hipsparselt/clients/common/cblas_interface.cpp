/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022-2025 Advanced Micro Devices, Inc.
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
#include "cblas_interface.hpp"
#include "hipsparselt_vector.hpp"
#include "utility.hpp"
#include <bitset>
#include <omp.h>

CBLAS_TRANSPOSE HIPOperationToCBLASTanspose(hipsparseOperation_t trans)
{
    switch(trans)
    {
    case HIPSPARSE_OPERATION_NON_TRANSPOSE:
        return CblasNoTrans;
    case HIPSPARSE_OPERATION_TRANSPOSE:
        return CblasTrans;
    case HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE:
        return CblasConjTrans;
    }
}

CBLAS_ORDER HIPOrderToCBLASOrder(hipsparseOrder_t order)
{
    switch(order)
    {
    case HIPSPARSE_ORDER_COL:
        return CblasColMajor;
    case HIPSPARSE_ORDER_ROW:
        return CblasRowMajor;
    }
}

// gemm
template <>
void cblas_gemm<hip_bfloat16, hip_bfloat16, float>(hipsparseOrder_t     order,
                                                   hipsparseOperation_t transA,
                                                   hipsparseOperation_t transB,
                                                   int64_t              m,
                                                   int64_t              n,
                                                   int64_t              k,
                                                   float                alpha,
                                                   const hip_bfloat16*  A,
                                                   int64_t              lda,
                                                   int64_t              sizeA,
                                                   const hip_bfloat16*  B,
                                                   int64_t              ldb,
                                                   int64_t              sizeB,
                                                   float                beta,
                                                   hip_bfloat16*        C,
                                                   int64_t              ldc,
                                                   int64_t              sizeC,
                                                   float*               alphaVec,
                                                   bool                 alt)
{
    // cblas does not support hip_bfloat16, so convert to higher precision float
    // This will give more precise result which is acceptable for testing

    host_vector<float> A_float(sizeA), B_float(sizeB), C_float(sizeC);

    for(size_t i = 0; i < sizeA; i++)
        A_float[i] = static_cast<float>(A[i]);
    for(size_t i = 0; i < sizeB; i++)
        B_float[i] = static_cast<float>(B[i]);
    for(size_t i = 0; i < sizeC; i++)
        C_float[i] = static_cast<float>(C[i]);

    if(alphaVec != nullptr)
    {
        host_vector<float> T_float(sizeC);
        memset(T_float, 0, sizeC);
        cblas_sgemm(HIPOrderToCBLASOrder(order),
                    HIPOperationToCBLASTanspose(transA),
                    HIPOperationToCBLASTanspose(transB),
                    m,
                    n,
                    k,
                    static_cast<float>(1),
                    A_float,
                    lda,
                    B_float,
                    ldb,
                    static_cast<float>(0),
                    T_float,
                    ldc);
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < n; j++)
            {
                size_t pos   = order == HIPSPARSE_ORDER_COL ? j * ldc + i : i * ldc + j;
                C_float[pos] = T_float[pos] * alphaVec[i] + C_float[pos] * beta;
            }
        }
    }
    else
    {
        // just directly cast, since transA, transB are integers in the enum
        // printf("transA: hipsparselt =%d, cblas=%d\n", transA, HIPOperationToCBLASTanspose(transA) );
        cblas_sgemm(HIPOrderToCBLASOrder(order),
                    HIPOperationToCBLASTanspose(transA),
                    HIPOperationToCBLASTanspose(transB),
                    m,
                    n,
                    k,
                    alpha,
                    A_float,
                    lda,
                    B_float,
                    ldb,
                    beta,
                    C_float,
                    ldc);
    }

    for(size_t i = 0; i < sizeC; i++)
        C[i] = static_cast<hip_bfloat16>(C_float[i]);
}

template <>
void cblas_gemm<hip_bfloat16, float, float>(hipsparseOrder_t     order,
                                            hipsparseOperation_t transA,
                                            hipsparseOperation_t transB,
                                            int64_t              m,
                                            int64_t              n,
                                            int64_t              k,
                                            float                alpha,
                                            const hip_bfloat16*  A,
                                            int64_t              lda,
                                            int64_t              sizeA,
                                            const hip_bfloat16*  B,
                                            int64_t              ldb,
                                            int64_t              sizeB,
                                            float                beta,
                                            float*               C,
                                            int64_t              ldc,
                                            int64_t              sizeC,
                                            float*               alphaVec,
                                            bool                 alt)
{
    // cblas does not support hip_bfloat16, so convert to higher precision float
    // This will give more precise result which is acceptable for testing

    host_vector<float> A_float(sizeA), B_float(sizeB);

    for(size_t i = 0; i < sizeA; i++)
        A_float[i] = static_cast<float>(A[i]);
    for(size_t i = 0; i < sizeB; i++)
        B_float[i] = static_cast<float>(B[i]);

    if(alphaVec != nullptr)
    {
        host_vector<float> T_float(sizeC);
        memset(T_float, 0, sizeC);
        cblas_sgemm(HIPOrderToCBLASOrder(order),
                    HIPOperationToCBLASTanspose(transA),
                    HIPOperationToCBLASTanspose(transB),
                    m,
                    n,
                    k,
                    static_cast<float>(1),
                    A_float,
                    lda,
                    B_float,
                    ldb,
                    static_cast<float>(0),
                    T_float,
                    ldc);
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < n; j++)
            {
                size_t pos = order == HIPSPARSE_ORDER_COL ? j * ldc + i : i * ldc + j;
                C[pos]     = T_float[pos] * alphaVec[i] + C[pos] * beta;
            }
        }
    }
    else
    {
        // just directly cast, since transA, transB are integers in the enum
        // printf("transA: hipsparselt =%d, cblas=%d\n", transA, HIPOperationToCBLASTanspose(transA) );
        cblas_sgemm(HIPOrderToCBLASOrder(order),
                    HIPOperationToCBLASTanspose(transA),
                    HIPOperationToCBLASTanspose(transB),
                    m,
                    n,
                    k,
                    alpha,
                    A_float,
                    lda,
                    B_float,
                    ldb,
                    beta,
                    C,
                    ldc);
    }
}

template <>
void cblas_gemm<__half, __half, float>(hipsparseOrder_t     order,
                                       hipsparseOperation_t transA,
                                       hipsparseOperation_t transB,
                                       int64_t              m,
                                       int64_t              n,
                                       int64_t              k,
                                       float                alpha,
                                       const __half*        A,
                                       int64_t              lda,
                                       int64_t              sizeA,
                                       const __half*        B,
                                       int64_t              ldb,
                                       int64_t              sizeB,
                                       float                beta,
                                       __half*              C,
                                       int64_t              ldc,
                                       int64_t              sizeC,
                                       float*               alphaVec,
                                       bool                 alt)
{
    // cblas does not support __half, so convert to higher precision float
    // This will give more precise result which is acceptable for testing

    host_vector<float> A_float(sizeA), B_float(sizeB), C_float(sizeC);

    if(alt)
    {
        for(size_t i = 0; i < sizeA; i++)
            A_float[i] = float_to_bfloat16_truncate(float(A[i]));
        for(size_t i = 0; i < sizeB; i++)
            B_float[i] = float_to_bfloat16_truncate(float(B[i]));
        for(size_t i = 0; i < sizeC; i++)
            C_float[i] = float_to_bfloat16_truncate(float(C[i]));
    }
    else
    {
        for(size_t i = 0; i < sizeA; i++)
            A_float[i] = A[i];
        for(size_t i = 0; i < sizeB; i++)
            B_float[i] = B[i];
        for(size_t i = 0; i < sizeC; i++)
            C_float[i] = C[i];
    }

    if(alphaVec != nullptr)
    {
        host_vector<float> T_float(sizeC);
        memset(T_float, 0, sizeC);
        cblas_sgemm(HIPOrderToCBLASOrder(order),
                    HIPOperationToCBLASTanspose(transA),
                    HIPOperationToCBLASTanspose(transB),
                    m,
                    n,
                    k,
                    static_cast<float>(1),
                    A_float,
                    lda,
                    B_float,
                    ldb,
                    static_cast<float>(0),
                    T_float,
                    ldc);
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < n; j++)
            {
                size_t pos   = order == HIPSPARSE_ORDER_COL ? j * ldc + i : i * ldc + j;
                C_float[pos] = T_float[pos] * alphaVec[i] + C_float[pos] * beta;
            }
        }
    }
    else
    {
        // just directly cast, since transA, transB are integers in the enum
        // printf("transA: hipsparselt =%d, cblas=%d\n", transA, HIPOperationToCBLASTanspose(transA) );
        cblas_sgemm(HIPOrderToCBLASOrder(order),
                    HIPOperationToCBLASTanspose(transA),
                    HIPOperationToCBLASTanspose(transB),
                    m,
                    n,
                    k,
                    alpha,
                    A_float,
                    lda,
                    B_float,
                    ldb,
                    beta,
                    C_float,
                    ldc);
    }

    for(size_t i = 0; i < sizeC; i++)
        C[i] = __half(C_float[i]);
}

template <>
void cblas_gemm<__half, float, float>(hipsparseOrder_t     order,
                                      hipsparseOperation_t transA,
                                      hipsparseOperation_t transB,
                                      int64_t              m,
                                      int64_t              n,
                                      int64_t              k,
                                      float                alpha,
                                      const __half*        A,
                                      int64_t              lda,
                                      int64_t              sizeA,
                                      const __half*        B,
                                      int64_t              ldb,
                                      int64_t              sizeB,
                                      float                beta,
                                      float*               C,
                                      int64_t              ldc,
                                      int64_t              sizeC,
                                      float*               alphaVec,
                                      bool                 alt)
{
    // cblas does not support __half, so convert to higher precision float
    // This will give more precise result which is acceptable for testing

    host_vector<float> A_float(sizeA), B_float(sizeB);

    if(alt)
    {
        for(size_t i = 0; i < sizeA; i++)
            A_float[i] = float_to_bfloat16_truncate(float(A[i]));
        for(size_t i = 0; i < sizeB; i++)
            B_float[i] = float_to_bfloat16_truncate(float(B[i]));
    }
    else
    {
        for(size_t i = 0; i < sizeA; i++)
            A_float[i] = A[i];
        for(size_t i = 0; i < sizeB; i++)
            B_float[i] = B[i];
    }

    if(alphaVec != nullptr)
    {
        host_vector<float> T_float(sizeC);
        memset(T_float, 0, sizeC);
        cblas_sgemm(HIPOrderToCBLASOrder(order),
                    HIPOperationToCBLASTanspose(transA),
                    HIPOperationToCBLASTanspose(transB),
                    m,
                    n,
                    k,
                    static_cast<float>(1),
                    A_float,
                    lda,
                    B_float,
                    ldb,
                    static_cast<float>(0),
                    T_float,
                    ldc);
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < n; j++)
            {
                size_t pos = order == HIPSPARSE_ORDER_COL ? j * ldc + i : i * ldc + j;
                C[pos]     = T_float[pos] * alphaVec[i] + C[pos] * beta;
            }
        }
    }
    else
    {

        // just directly cast, since transA, transB are integers in the enum
        // printf("transA: hipsparselt =%d, cblas=%d\n", transA, HIPOperationToCBLASTanspose(transA) );
        cblas_sgemm(HIPOrderToCBLASOrder(order),
                    HIPOperationToCBLASTanspose(transA),
                    HIPOperationToCBLASTanspose(transB),
                    m,
                    n,
                    k,
                    alpha,
                    A_float,
                    lda,
                    B_float,
                    ldb,
                    beta,
                    C,
                    ldc);
    }
}

template <>
void cblas_gemm<int8_t, int8_t, float>(hipsparseOrder_t     order,
                                       hipsparseOperation_t transA,
                                       hipsparseOperation_t transB,
                                       int64_t              m,
                                       int64_t              n,
                                       int64_t              k,
                                       float                alpha,
                                       const int8_t*        A,
                                       int64_t              lda,
                                       int64_t              sizeA,
                                       const int8_t*        B,
                                       int64_t              ldb,
                                       int64_t              sizeB,
                                       float                beta,
                                       int8_t*              C,
                                       int64_t              ldc,
                                       int64_t              sizeC,
                                       float*               alphaVec,
                                       bool                 alt)
{
    // cblas does not support int8_t input / int8_t output, however non-overflowing
    // 32-bit integer operations can be represented accurately with double-precision
    // floats, so convert to doubles and downcast result down to int32_t.
    // NOTE: This will not properly account for 32-bit integer overflow, however
    //       the result should be acceptable for testing.

    host_vector<double> A_double(sizeA);
    host_vector<double> B_double(sizeB);
    host_vector<double> C_double(sizeC);

    for(size_t i = 0; i < sizeA; i++)
        A_double[i] = static_cast<double>(A[i]);
    for(size_t i = 0; i < sizeB; i++)
        B_double[i] = static_cast<double>(B[i]);
    for(size_t i = 0; i < sizeC; i++)
        C_double[i] = static_cast<double>(C[i]);

    if(alphaVec != nullptr)
    {
        host_vector<double> T_double(sizeC);
        memset(T_double, 0, sizeC);
        cblas_dgemm(HIPOrderToCBLASOrder(order),
                    HIPOperationToCBLASTanspose(transA),
                    HIPOperationToCBLASTanspose(transB),
                    m,
                    n,
                    k,
                    static_cast<double>(1),
                    A_double,
                    lda,
                    B_double,
                    ldb,
                    static_cast<double>(0),
                    T_double,
                    ldc);
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < n; j++)
            {
                size_t pos    = order == HIPSPARSE_ORDER_COL ? j * ldc + i : i * ldc + j;
                C_double[pos] = T_double[pos] * static_cast<double>(alphaVec[i])
                                + C_double[pos] * static_cast<double>(beta);
            }
        }
    }
    else
    {
        // just directly cast, since transA, transB are integers in the enum
        cblas_dgemm(HIPOrderToCBLASOrder(order),
                    HIPOperationToCBLASTanspose(transA),
                    HIPOperationToCBLASTanspose(transB),
                    m,
                    n,
                    k,
                    alpha,
                    A_double,
                    lda,
                    B_double,
                    ldb,
                    beta,
                    C_double,
                    ldc);
    }

    auto saturate = [](double val) {
        val = std::nearbyint(val);
        val = val > 127.f ? 127.f : val < -128.f ? -128.f : val;
        return val;
    };

    for(size_t i = 0; i < sizeC; i++)
        C[i] = static_cast<int8_t>(saturate(C_double[i]));
}

template <>
void cblas_gemm<int8_t, float, float>(hipsparseOrder_t     order,
                                      hipsparseOperation_t transA,
                                      hipsparseOperation_t transB,
                                      int64_t              m,
                                      int64_t              n,
                                      int64_t              k,
                                      float                alpha,
                                      const int8_t*        A,
                                      int64_t              lda,
                                      int64_t              sizeA,
                                      const int8_t*        B,
                                      int64_t              ldb,
                                      int64_t              sizeB,
                                      float                beta,
                                      float*               C,
                                      int64_t              ldc,
                                      int64_t              sizeC,
                                      float*               alphaVec,
                                      bool                 alt)
{
    // cblas does not support int8_t input / int8_t output, however non-overflowing
    // 32-bit integer operations can be represented accurately with double-precision
    // floats, so convert to doubles and downcast result down to int32_t.
    // NOTE: This will not properly account for 32-bit integer overflow, however
    //       the result should be acceptable for testing.

    host_vector<double> A_double(sizeA);
    host_vector<double> B_double(sizeB);
    host_vector<double> C_double(sizeC);

    for(size_t i = 0; i < sizeA; i++)
        A_double[i] = static_cast<double>(A[i]);
    for(size_t i = 0; i < sizeB; i++)
        B_double[i] = static_cast<double>(B[i]);
    for(size_t i = 0; i < sizeC; i++)
        C_double[i] = static_cast<double>(C[i]);

    if(alphaVec != nullptr)
    {
        host_vector<double> T_double(sizeC);
        memset(T_double, 0, sizeC);
        cblas_dgemm(HIPOrderToCBLASOrder(order),
                    HIPOperationToCBLASTanspose(transA),
                    HIPOperationToCBLASTanspose(transB),
                    m,
                    n,
                    k,
                    static_cast<double>(1),
                    A_double,
                    lda,
                    B_double,
                    ldb,
                    static_cast<double>(0),
                    T_double,
                    ldc);
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < n; j++)
            {
                size_t pos    = order == HIPSPARSE_ORDER_COL ? j * ldc + i : i * ldc + j;
                C_double[pos] = T_double[pos] * static_cast<double>(alphaVec[i])
                                + C_double[pos] * static_cast<double>(beta);
            }
        }
    }
    else
    {
        // just directly cast, since transA, transB are integers in the enum
        cblas_dgemm(HIPOrderToCBLASOrder(order),
                    HIPOperationToCBLASTanspose(transA),
                    HIPOperationToCBLASTanspose(transB),
                    m,
                    n,
                    k,
                    alpha,
                    A_double,
                    lda,
                    B_double,
                    ldb,
                    beta,
                    C_double,
                    ldc);
    }

    for(size_t i = 0; i < sizeC; i++)
        C[i] = static_cast<float>(C_double[i]);
}

template <>
void cblas_gemm<int8_t, __half, float>(hipsparseOrder_t     order,
                                       hipsparseOperation_t transA,
                                       hipsparseOperation_t transB,
                                       int64_t              m,
                                       int64_t              n,
                                       int64_t              k,
                                       float                alpha,
                                       const int8_t*        A,
                                       int64_t              lda,
                                       int64_t              sizeA,
                                       const int8_t*        B,
                                       int64_t              ldb,
                                       int64_t              sizeB,
                                       float                beta,
                                       __half*              C,
                                       int64_t              ldc,
                                       int64_t              sizeC,
                                       float*               alphaVec,
                                       bool                 alt)
{
    // cblas does not support int8_t input / int8_t output, however non-overflowing
    // 32-bit integer operations can be represented accurately with double-precision
    // floats, so convert to doubles and downcast result down to int32_t.
    // NOTE: This will not properly account for 32-bit integer overflow, however
    //       the result should be acceptable for testing.

    host_vector<double> A_double(sizeA);
    host_vector<double> B_double(sizeB);
    host_vector<double> C_double(sizeC);

    for(size_t i = 0; i < sizeA; i++)
        A_double[i] = static_cast<double>(A[i]);
    for(size_t i = 0; i < sizeB; i++)
        B_double[i] = static_cast<double>(B[i]);
    for(size_t i = 0; i < sizeC; i++)
        C_double[i] = static_cast<double>(C[i]);

    if(alphaVec != nullptr)
    {
        host_vector<double> T_double(sizeC);
        memset(T_double, 0, sizeC);
        cblas_dgemm(HIPOrderToCBLASOrder(order),
                    HIPOperationToCBLASTanspose(transA),
                    HIPOperationToCBLASTanspose(transB),
                    m,
                    n,
                    k,
                    static_cast<double>(1),
                    A_double,
                    lda,
                    B_double,
                    ldb,
                    static_cast<double>(0),
                    T_double,
                    ldc);
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < n; j++)
            {
                size_t pos    = order == HIPSPARSE_ORDER_COL ? j * ldc + i : i * ldc + j;
                C_double[pos] = T_double[pos] * static_cast<double>(alphaVec[i])
                                + C_double[pos] * static_cast<double>(beta);
            }
        }
    }
    else
    {
        // just directly cast, since transA, transB are integers in the enum
        cblas_dgemm(HIPOrderToCBLASOrder(order),
                    HIPOperationToCBLASTanspose(transA),
                    HIPOperationToCBLASTanspose(transB),
                    m,
                    n,
                    k,
                    alpha,
                    A_double,
                    lda,
                    B_double,
                    ldb,
                    beta,
                    C_double,
                    ldc);
    }

    for(size_t i = 0; i < sizeC; i++)
        C[i] = __half(C_double[i]);
}

template <>
void cblas_gemm<int8_t, hip_bfloat16, float>(hipsparseOrder_t     order,
                                             hipsparseOperation_t transA,
                                             hipsparseOperation_t transB,
                                             int64_t              m,
                                             int64_t              n,
                                             int64_t              k,
                                             float                alpha,
                                             const int8_t*        A,
                                             int64_t              lda,
                                             int64_t              sizeA,
                                             const int8_t*        B,
                                             int64_t              ldb,
                                             int64_t              sizeB,
                                             float                beta,
                                             hip_bfloat16*        C,
                                             int64_t              ldc,
                                             int64_t              sizeC,
                                             float*               alphaVec,
                                             bool                 alt)
{
    // cblas does not support int8_t input / int8_t output, however non-overflowing
    // 32-bit integer operations can be represented accurately with double-precision
    // floats, so convert to doubles and downcast result down to int32_t.
    // NOTE: This will not properly account for 32-bit integer overflow, however
    //       the result should be acceptable for testing.

    host_vector<double> A_double(sizeA);
    host_vector<double> B_double(sizeB);
    host_vector<double> C_double(sizeC);

    for(size_t i = 0; i < sizeA; i++)
        A_double[i] = static_cast<double>(A[i]);
    for(size_t i = 0; i < sizeB; i++)
        B_double[i] = static_cast<double>(B[i]);
    for(size_t i = 0; i < sizeC; i++)
        C_double[i] = static_cast<double>(C[i]);

    if(alphaVec != nullptr)
    {
        host_vector<double> T_double(sizeC);
        memset(T_double, 0, sizeC);
        cblas_dgemm(HIPOrderToCBLASOrder(order),
                    HIPOperationToCBLASTanspose(transA),
                    HIPOperationToCBLASTanspose(transB),
                    m,
                    n,
                    k,
                    static_cast<double>(1),
                    A_double,
                    lda,
                    B_double,
                    ldb,
                    static_cast<double>(0),
                    T_double,
                    ldc);
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < n; j++)
            {
                size_t pos    = order == HIPSPARSE_ORDER_COL ? j * ldc + i : i * ldc + j;
                C_double[pos] = T_double[pos] * static_cast<double>(alphaVec[i])
                                + C_double[pos] * static_cast<double>(beta);
            }
        }
    }
    else
    {
        // just directly cast, since transA, transB are integers in the enum
        cblas_dgemm(HIPOrderToCBLASOrder(order),
                    HIPOperationToCBLASTanspose(transA),
                    HIPOperationToCBLASTanspose(transB),
                    m,
                    n,
                    k,
                    alpha,
                    A_double,
                    lda,
                    B_double,
                    ldb,
                    beta,
                    C_double,
                    ldc);
    }

    for(size_t i = 0; i < sizeC; i++)
        C[i] = static_cast<hip_bfloat16>(C_double[i]);
}


#ifdef HIPSPARSELT_CLIENT_ENABLE_FP8_OCP
template <>
void cblas_gemm<hipsparselt_fp8_e4m3, double, float>(hipsparseOrder_t      order,
                                              hipsparseOperation_t  transA,
                                              hipsparseOperation_t  transB,
                                              int64_t               m,
                                              int64_t               n,
                                              int64_t               k,
                                              float                 alpha,
                                              const hipsparselt_fp8_e4m3* A,
                                              int64_t               lda,
                                              int64_t               sizeA,
                                              const hipsparselt_fp8_e4m3* B,
                                              int64_t               ldb,
                                              int64_t               sizeB,
                                              float                 beta,
                                              double*               C,
                                              int64_t               ldc,
                                              int64_t               sizeC,
                                              float*                alphaVec,
                                              bool                  alt)
{
    // cblas does not support int8_t input / int8_t output, however non-overflowing
    // 32-bit integer operations can be represented accurately with double-precision
    // floats, so convert to doubles and downcast result down to int32_t.
    // NOTE: This will not properly account for 32-bit integer overflow, however
    //       the result should be acceptable for testing.

    host_vector<double> A_double(sizeA);
    host_vector<double> B_double(sizeB);

    for(size_t i = 0; i < sizeA; i++)
        A_double[i] = static_cast<double>(A[i]);
    for(size_t i = 0; i < sizeB; i++)
        B_double[i] = static_cast<double>(B[i]);

    if(alphaVec != nullptr)
    {
        host_vector<double> T_double(sizeC);
        memset(T_double, 0, sizeC);
        cblas_dgemm(HIPOrderToCBLASOrder(order),
                    HIPOperationToCBLASTanspose(transA),
                    HIPOperationToCBLASTanspose(transB),
                    m,
                    n,
                    k,
                    static_cast<double>(1),
                    A_double,
                    lda,
                    B_double,
                    ldb,
                    static_cast<double>(0),
                    T_double,
                    ldc);
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < n; j++)
            {
                size_t pos    = order == HIPSPARSE_ORDER_COL ? j * ldc + i : i * ldc + j;
                C[pos]        = T_double[pos] * static_cast<double>(alphaVec[i])
                                + C[pos] * static_cast<double>(beta);
            }
        }
    }
    else
    {
        // just directly cast, since transA, transB are integers in the enum
        cblas_dgemm(HIPOrderToCBLASOrder(order),
                    HIPOperationToCBLASTanspose(transA),
                    HIPOperationToCBLASTanspose(transB),
                    m,
                    n,
                    k,
                    alpha,
                    A_double,
                    lda,
                    B_double,
                    ldb,
                    beta,
                    C,
                    ldc);
    }
}

template <>
void cblas_gemm<hipsparselt_fp8_e5m2, double, float>(hipsparseOrder_t      order,
                                              hipsparseOperation_t  transA,
                                              hipsparseOperation_t  transB,
                                              int64_t               m,
                                              int64_t               n,
                                              int64_t               k,
                                              float                 alpha,
                                              const hipsparselt_fp8_e5m2* A,
                                              int64_t               lda,
                                              int64_t               sizeA,
                                              const hipsparselt_fp8_e5m2* B,
                                              int64_t               ldb,
                                              int64_t               sizeB,
                                              float                 beta,
                                              double*               C,
                                              int64_t               ldc,
                                              int64_t               sizeC,
                                              float*                alphaVec,
                                              bool                  alt)
{
    // cblas does not support int8_t input / int8_t output, however non-overflowing
    // 32-bit integer operations can be represented accurately with double-precision
    // floats, so convert to doubles and downcast result down to int32_t.
    // NOTE: This will not properly account for 32-bit integer overflow, however
    //       the result should be acceptable for testing.

    host_vector<double> A_double(sizeA);
    host_vector<double> B_double(sizeB);

    for(size_t i = 0; i < sizeA; i++)
        A_double[i] = static_cast<double>(A[i]);
    for(size_t i = 0; i < sizeB; i++)
        B_double[i] = static_cast<double>(B[i]);

    if(alphaVec != nullptr)
    {
        host_vector<double> T_double(sizeC);
        memset(T_double, 0, sizeC);
        cblas_dgemm(HIPOrderToCBLASOrder(order),
                    HIPOperationToCBLASTanspose(transA),
                    HIPOperationToCBLASTanspose(transB),
                    m,
                    n,
                    k,
                    static_cast<double>(1),
                    A_double,
                    lda,
                    B_double,
                    ldb,
                    static_cast<double>(0),
                    T_double,
                    ldc);
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < n; j++)
            {
                size_t pos    = order == HIPSPARSE_ORDER_COL ? j * ldc + i : i * ldc + j;
                C[pos]        = T_double[pos] * static_cast<double>(alphaVec[i])
                                + C[pos] * static_cast<double>(beta);
            }
        }
    }
    else
    {
        // just directly cast, since transA, transB are integers in the enum
        cblas_dgemm(HIPOrderToCBLASOrder(order),
                    HIPOperationToCBLASTanspose(transA),
                    HIPOperationToCBLASTanspose(transB),
                    m,
                    n,
                    k,
                    alpha,
                    A_double,
                    lda,
                    B_double,
                    ldb,
                    beta,
                    C,
                    ldc);
    }
}

template <>
void cblas_gemm<hipsparselt_fp8_e4m3, float, float>(hipsparseOrder_t      order,
                                              hipsparseOperation_t  transA,
                                              hipsparseOperation_t  transB,
                                              int64_t               m,
                                              int64_t               n,
                                              int64_t               k,
                                              float                 alpha,
                                              const hipsparselt_fp8_e4m3* A,
                                              int64_t               lda,
                                              int64_t               sizeA,
                                              const hipsparselt_fp8_e4m3* B,
                                              int64_t               ldb,
                                              int64_t               sizeB,
                                              float                 beta,
                                              float*                C,
                                              int64_t               ldc,
                                              int64_t               sizeC,
                                              float*                alphaVec,
                                              bool                  alt)
{
    host_vector<double> C_double(sizeC);
    for(size_t i = 0; i < sizeC; i++)
        C_double[i] = static_cast<double>(C[i]);

    cblas_gemm<hipsparselt_fp8_e4m3, double, float>(order, transA, transB, m, n, k, alpha,
                                                   A, lda, sizeA, B, ldb, sizeB, beta,
                                                   C_double, ldc, sizeC, alphaVec, alt);

    for(size_t i = 0; i < sizeC; i++)
        C[i] = static_cast<float>(C_double[i]);
}

template <>
void cblas_gemm<hipsparselt_fp8_e5m2, float, float>(hipsparseOrder_t      order,
                                              hipsparseOperation_t  transA,
                                              hipsparseOperation_t  transB,
                                              int64_t               m,
                                              int64_t               n,
                                              int64_t               k,
                                              float                 alpha,
                                              const hipsparselt_fp8_e5m2* A,
                                              int64_t               lda,
                                              int64_t               sizeA,
                                              const hipsparselt_fp8_e5m2* B,
                                              int64_t               ldb,
                                              int64_t               sizeB,
                                              float                 beta,
                                              float*                C,
                                              int64_t               ldc,
                                              int64_t               sizeC,
                                              float*                alphaVec,
                                              bool                  alt)
{
    host_vector<double> C_double(sizeC);
    for(size_t i = 0; i < sizeC; i++)
        C_double[i] = static_cast<double>(C[i]);

    cblas_gemm<hipsparselt_fp8_e5m2, double, float>(order, transA, transB, m, n, k, alpha,
                                                   A, lda, sizeA, B, ldb, sizeB, beta,
                                                   C_double, ldc, sizeC, alphaVec, alt);

    for(size_t i = 0; i < sizeC; i++)
        C[i] = static_cast<float>(C_double[i]);
}

#ifdef __HIP_PLATFORM_NVIDIA__

template <>
void cblas_gemm<hipsparselt_fp8_e4m3, __half, float>(hipsparseOrder_t      order,
                                              hipsparseOperation_t  transA,
                                              hipsparseOperation_t  transB,
                                              int64_t               m,
                                              int64_t               n,
                                              int64_t               k,
                                              float                 alpha,
                                              const hipsparselt_fp8_e4m3* A,
                                              int64_t               lda,
                                              int64_t               sizeA,
                                              const hipsparselt_fp8_e4m3* B,
                                              int64_t               ldb,
                                              int64_t               sizeB,
                                              float                 beta,
                                              __half*                C,
                                              int64_t               ldc,
                                              int64_t               sizeC,
                                              float*                alphaVec,
                                              bool                  alt)
{
    host_vector<double> C_double(sizeC);
    for(size_t i = 0; i < sizeC; i++)
        C_double[i] = static_cast<double>(C[i]);

    cblas_gemm<hipsparselt_fp8_e4m3, double, float>(order, transA, transB, m, n, k, alpha,
                                                   A, lda, sizeA, B, ldb, sizeB, beta,
                                                   C_double, ldc, sizeC, alphaVec, alt);

    for(size_t i = 0; i < sizeC; i++)
        C[i] = static_cast<__half>(C_double[i]);
}

template <>
void cblas_gemm<hipsparselt_fp8_e4m3, hip_bfloat16, float>(hipsparseOrder_t      order,
                                              hipsparseOperation_t  transA,
                                              hipsparseOperation_t  transB,
                                              int64_t               m,
                                              int64_t               n,
                                              int64_t               k,
                                              float                 alpha,
                                              const hipsparselt_fp8_e4m3* A,
                                              int64_t               lda,
                                              int64_t               sizeA,
                                              const hipsparselt_fp8_e4m3* B,
                                              int64_t               ldb,
                                              int64_t               sizeB,
                                              float                 beta,
                                              hip_bfloat16*                C,
                                              int64_t               ldc,
                                              int64_t               sizeC,
                                              float*                alphaVec,
                                              bool                  alt)
{
    host_vector<double> C_double(sizeC);
    for(size_t i = 0; i < sizeC; i++)
        C_double[i] = static_cast<double>(C[i]);

    cblas_gemm<hipsparselt_fp8_e4m3, double, float>(order, transA, transB, m, n, k, alpha,
                                                   A, lda, sizeA, B, ldb, sizeB, beta,
                                                   C_double, ldc, sizeC, alphaVec, alt);

    for(size_t i = 0; i < sizeC; i++)
        C[i] = static_cast<hip_bfloat16>(C_double[i]);
}

template <>
void cblas_gemm<hipsparselt_fp8_e5m2, __half, float>(hipsparseOrder_t      order,
                                              hipsparseOperation_t  transA,
                                              hipsparseOperation_t  transB,
                                              int64_t               m,
                                              int64_t               n,
                                              int64_t               k,
                                              float                 alpha,
                                              const hipsparselt_fp8_e5m2* A,
                                              int64_t               lda,
                                              int64_t               sizeA,
                                              const hipsparselt_fp8_e5m2* B,
                                              int64_t               ldb,
                                              int64_t               sizeB,
                                              float                 beta,
                                              __half*                C,
                                              int64_t               ldc,
                                              int64_t               sizeC,
                                              float*                alphaVec,
                                              bool                  alt)
{
    host_vector<double> C_double(sizeC);
    for(size_t i = 0; i < sizeC; i++)
        C_double[i] = static_cast<double>(C[i]);

    cblas_gemm<hipsparselt_fp8_e5m2, double, float>(order, transA, transB, m, n, k, alpha,
                                                   A, lda, sizeA, B, ldb, sizeB, beta,
                                                   C_double, ldc, sizeC, alphaVec, alt);

    for(size_t i = 0; i < sizeC; i++)
        C[i] = static_cast<__half>(C_double[i]);
}
template <>
void cblas_gemm<hipsparselt_fp8_e5m2, hip_bfloat16, float>(hipsparseOrder_t      order,
                                              hipsparseOperation_t  transA,
                                              hipsparseOperation_t  transB,
                                              int64_t               m,
                                              int64_t               n,
                                              int64_t               k,
                                              float                 alpha,
                                              const hipsparselt_fp8_e5m2* A,
                                              int64_t               lda,
                                              int64_t               sizeA,
                                              const hipsparselt_fp8_e5m2* B,
                                              int64_t               ldb,
                                              int64_t               sizeB,
                                              float                 beta,
                                              hip_bfloat16*         C,
                                              int64_t               ldc,
                                              int64_t               sizeC,
                                              float*                alphaVec,
                                              bool                  alt)
{
    host_vector<double> C_double(sizeC);
    for(size_t i = 0; i < sizeC; i++)
        C_double[i] = static_cast<double>(C[i]);

    cblas_gemm<hipsparselt_fp8_e5m2, double, float>(order, transA, transB, m, n, k, alpha,
                                                   A, lda, sizeA, B, ldb, sizeB, beta,
                                                   C_double, ldc, sizeC, alphaVec, alt);

    for(size_t i = 0; i < sizeC; i++)
        C[i] = static_cast<hip_bfloat16>(C_double[i]);
}
#endif
#endif
