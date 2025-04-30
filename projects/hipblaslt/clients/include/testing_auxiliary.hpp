/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
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

#pragma once

#include "flops.hpp"
#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_init.hpp"
#include "hipblaslt_math.hpp"
#include "hipblaslt_random.hpp"
#include "hipblaslt_test.hpp"
#include "hipblaslt_vector.hpp"
#include "unit.hpp"
#include "utility.hpp"
#include <hipblaslt/hipblaslt-ext.hpp> // Add check for hipblaslt-ext
#include <hipblaslt/hipblaslt.h>

void testing_aux_handle_init_bad_arg(const Arguments& arg)
{
    EXPECT_HIPBLAS_STATUS(hipblasLtCreate(nullptr), HIPBLAS_STATUS_INVALID_VALUE);
}

void testing_aux_handle_destroy_bad_arg(const Arguments& arg)
{
    EXPECT_HIPBLAS_STATUS(hipblasLtDestroy(nullptr), HIPBLAS_STATUS_INVALID_VALUE);
}

void testing_aux_handle(const Arguments& arg)
{
    hipblasLtHandle_t handle;
    EXPECT_HIPBLAS_STATUS(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
    EXPECT_HIPBLAS_STATUS(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}

void testing_aux_mat_init_bad_arg(const Arguments& arg)
{
    const int64_t row = 128;
    const int64_t col = 128;
    const int64_t ld  = 128;

    hipblasLtMatrixLayout_t m_descr;

    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutCreate(nullptr, arg.a_type, row, col, ld),
                          HIPBLAS_STATUS_INVALID_VALUE);
}

void testing_aux_mat_init(const Arguments& arg)
{
    const int64_t row = 128;
    const int64_t col = 128;
    const int64_t ld  = 128;

    hipblaslt_local_handle        handle{arg};
    hipblaslt_local_matrix_layout mat(row, col, ld, arg.a_type);
    EXPECT_HIPBLAS_STATUS(mat.status(), HIPBLAS_STATUS_SUCCESS);
}

void testing_aux_mat_destroy_bad_arg(const Arguments& arg)
{
    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutDestroy(nullptr), HIPBLAS_STATUS_INVALID_VALUE);
}

void testing_aux_mat_set_attr_bad_arg(const Arguments& arg)
{
    const int64_t row = 128;
    const int64_t col = 128;
    const int64_t ld  = 128;

    int     data;
    int64_t data64;

    hipblaslt_local_matrix_layout mat(row, col, ld, arg.a_type);
    EXPECT_HIPBLAS_STATUS(mat.status(), HIPBLAS_STATUS_SUCCESS);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutSetAttribute(
                              mat, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, nullptr, sizeof(int)),
                          HIPBLAS_STATUS_INVALID_VALUE);

    data = 1;
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatrixLayoutSetAttribute(mat, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &data, 1),
        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatrixLayoutSetAttribute(
            mat, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, nullptr, sizeof(int64_t)),
        HIPBLAS_STATUS_INVALID_VALUE);

    data64 = ld * col;
    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutSetAttribute(
                              mat, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &data64, 1),
                          HIPBLAS_STATUS_INVALID_VALUE);
}

void testing_aux_mat_get_attr_bad_arg(const Arguments& arg)
{
    const int64_t row = 128;
    const int64_t col = 128;
    const int64_t ld  = 128;

    hipblaslt_local_matrix_layout mat(row, col, ld, arg.a_type);
    EXPECT_HIPBLAS_STATUS(mat.status(), HIPBLAS_STATUS_SUCCESS);

    int     data;
    int64_t data64;
    size_t  sizeWritten;

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatrixLayoutGetAttribute(
            nullptr, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &data, sizeof(int), &sizeWritten),
        HIPBLAS_STATUS_NOT_INITIALIZED);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatrixLayoutGetAttribute(
            mat, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, nullptr, sizeof(int), &sizeWritten),
        HIPBLAS_STATUS_INVALID_VALUE);
    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutGetAttribute(
                              mat, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &data, 1, &sizeWritten),
                          HIPBLAS_STATUS_INVALID_VALUE);
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatrixLayoutGetAttribute(mat,
                                          HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                          nullptr,
                                          sizeof(int64_t),
                                          &sizeWritten),
        HIPBLAS_STATUS_INVALID_VALUE);
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatrixLayoutGetAttribute(
            mat, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &data64, sizeof(int), &sizeWritten),
        HIPBLAS_STATUS_INVALID_VALUE);
    // test sizeWritten is nullptr, and the return state should be success
    data = 0;
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatrixLayoutGetAttribute(
            mat, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &data, sizeof(int), nullptr),
        HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(data, 1);
}

void testing_aux_mat_set_get_attr(const Arguments& arg)
{
    const int64_t row = 128;
    const int64_t col = 128;
    const int64_t ld  = 128;

    hipblaslt_local_matrix_layout mat(row, col, ld, arg.a_type);
    EXPECT_HIPBLAS_STATUS(mat.status(), HIPBLAS_STATUS_SUCCESS);

    int32_t data, data_r;
    size_t  sizeWritten;

    data = 2;
    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutSetAttribute(
                              mat, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &data, sizeof(data)),
                          HIPBLAS_STATUS_SUCCESS);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatrixLayoutGetAttribute(
            mat, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &data_r, sizeof(data), &sizeWritten),
        HIPBLAS_STATUS_SUCCESS);
    ASSERT_TRUE(data_r == data);

    std::vector<int64_t> data64_v = {0, ld * col};
    int64_t              data64_r = 0;
    for(int64_t data64 : data64_v)
    {
        EXPECT_HIPBLAS_STATUS(
            hipblasLtMatrixLayoutSetAttribute(
                mat, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &data64, sizeof(int64_t)),
            HIPBLAS_STATUS_SUCCESS);

        EXPECT_HIPBLAS_STATUS(
            hipblasLtMatrixLayoutGetAttribute(mat,
                                              HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                              &data64_r,
                                              sizeof(int64_t),
                                              &sizeWritten),
            HIPBLAS_STATUS_SUCCESS);
        ASSERT_TRUE(data64_r == data64);
    }
}

void testing_aux_matmul_init_bad_arg(const Arguments& arg)
{
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescCreate(nullptr, arg.compute_type, arg.scale_type),
                          HIPBLAS_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_init(const Arguments& arg)
{
    const hipblasOperation_t opA = HIPBLAS_OP_T;
    const hipblasOperation_t opB = HIPBLAS_OP_N;

    hipblaslt_local_matmul_descr matmul(opA, opB, arg.compute_type, arg.scale_type);
    EXPECT_HIPBLAS_STATUS(matmul.status(), HIPBLAS_STATUS_SUCCESS);
}

void testing_aux_matmul_set_attr_bad_arg(const Arguments& arg)
{
    const hipblasOperation_t opA = HIPBLAS_OP_T;
    const hipblasOperation_t opB = HIPBLAS_OP_N;

    hipblaslt_local_matmul_descr matmul(opA, opB, arg.compute_type, arg.scale_type);
    EXPECT_HIPBLAS_STATUS(matmul.status(), HIPBLAS_STATUS_SUCCESS);

    hipblasLtEpilogue_t data = HIPBLASLT_EPILOGUE_RELU;
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                              nullptr, HIPBLASLT_MATMUL_DESC_EPILOGUE, &data, sizeof(data)),
                          HIPBLAS_STATUS_NOT_INITIALIZED);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                              matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, nullptr, sizeof(data)),
                          HIPBLAS_STATUS_INVALID_VALUE);
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &data, 1),
        HIPBLAS_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_get_attr_bad_arg(const Arguments& arg)
{
    const hipblasOperation_t opA = HIPBLAS_OP_T;
    const hipblasOperation_t opB = HIPBLAS_OP_N;

    hipblaslt_local_matmul_descr matmul(opA, opB, arg.compute_type, arg.scale_type);
    EXPECT_HIPBLAS_STATUS(matmul.status(), HIPBLAS_STATUS_SUCCESS);

    hipblasLtEpilogue_t data = HIPBLASLT_EPILOGUE_RELU;
    size_t              sizeWritten;
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescGetAttribute(
            nullptr, HIPBLASLT_MATMUL_DESC_EPILOGUE, &data, sizeof(data), &sizeWritten),
        HIPBLAS_STATUS_NOT_INITIALIZED);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescGetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, nullptr, sizeof(data), &sizeWritten),
        HIPBLAS_STATUS_INVALID_VALUE);
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                              matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &data, 1, &sizeWritten),
                          HIPBLAS_STATUS_INVALID_VALUE);

    void* dBias = nullptr;
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                              matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &dBias, 4, &sizeWritten),
                          HIPBLAS_STATUS_INVALID_VALUE);
    // test sizeWritten is nullptr, and the return state should be success
    data = HIPBLASLT_EPILOGUE_RELU;
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                              matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &data, sizeof(data), nullptr),
                          HIPBLAS_STATUS_SUCCESS);
    // test return buffer value equals HIPBLASLT_EPILOGUE_DEFAULT
    EXPECT_EQ(data, HIPBLASLT_EPILOGUE_DEFAULT);
}

void testing_aux_matmul_set_get_attr(const Arguments& arg)
{
    const hipblasOperation_t opA = HIPBLAS_OP_T;
    const hipblasOperation_t opB = HIPBLAS_OP_N;

    hipblaslt_local_matmul_descr matmul(opA, opB, arg.compute_type, arg.scale_type);
    EXPECT_HIPBLAS_STATUS(matmul.status(), HIPBLAS_STATUS_SUCCESS);

    hipblasLtEpilogue_t data   = HIPBLASLT_EPILOGUE_DEFAULT;
    hipblasLtEpilogue_t data_r = HIPBLASLT_EPILOGUE_RELU;
    size_t              sizeWritten;
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                              matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &data, sizeof(data)),
                          HIPBLAS_STATUS_SUCCESS);
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescGetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &data_r, sizeof(data_r), &sizeWritten),
        HIPBLAS_STATUS_SUCCESS);
    ASSERT_TRUE(data == data_r);
}

void testing_aux_matmul_pref_get_attr_bad_arg(const Arguments& arg)
{
    hipblaslt_local_preference pref;
    EXPECT_HIPBLAS_STATUS(pref.status(), HIPBLAS_STATUS_SUCCESS);

    uint64_t data;
    size_t   sizeWritten;

    // Test with null preference (should be INVALID_VALUE)
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulPreferenceGetAttribute(
            nullptr, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &data, sizeof(data), &sizeWritten),
        HIPBLAS_STATUS_INVALID_VALUE);

    // Test with sizeInBytes = 0 and sizeWritten = NULL (should be INVALID_VALUE)
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulPreferenceGetAttribute(
                              pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &data, 0, nullptr),
                          HIPBLAS_STATUS_INVALID_VALUE);

    // Test with non-zero sizeInBytes and buf = NULL (should be INVALID_VALUE)
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulPreferenceGetAttribute(pref,
                                              HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              nullptr,
                                              sizeof(uint64_t),
                                              &sizeWritten),
        HIPBLAS_STATUS_INVALID_VALUE);

    // Test with correct size (should be SUCCESS)
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulPreferenceGetAttribute(
            pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &data, sizeof(uint64_t), &sizeWritten),
        HIPBLAS_STATUS_SUCCESS);
}

void testing_aux_matmul_pref_get_attr(const Arguments& arg)
{
    hipblaslt_local_preference pref;
    EXPECT_HIPBLAS_STATUS(pref.status(), HIPBLAS_STATUS_SUCCESS);

    uint64_t data_set = 1024;
    uint64_t data_get = 0;
    size_t   sizeWritten;

    // Set the attribute
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulPreferenceSetAttribute(
            pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &data_set, sizeof(data_set)),
        HIPBLAS_STATUS_SUCCESS);

    // Get the attribute
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulPreferenceGetAttribute(pref,
                                              HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &data_get,
                                              sizeof(data_get),
                                              &sizeWritten),
        HIPBLAS_STATUS_SUCCESS);

    // Verify that the get value matches the set value
    ASSERT_TRUE(data_get == data_set);
    ASSERT_TRUE(sizeWritten == sizeof(data_get));

    // Test getting other attributes (assuming they have default values)
    int32_t search_mode;
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulPreferenceGetAttribute(pref,
                                                                HIPBLASLT_MATMUL_PREF_SEARCH_MODE,
                                                                &search_mode,
                                                                sizeof(search_mode),
                                                                &sizeWritten),
                          HIPBLAS_STATUS_SUCCESS);

    // You might want to add more attribute tests here
}

void testing_aux_matmul_alg_init_bad_arg(const Arguments& arg) {}

void testing_aux_matmul_alg_init(const Arguments& arg) {}

void testing_aux_get_sol_with_null_biasaddr(const Arguments& arg)
{
    using InTypeA   = hipblasLtHalf;
    using InTypeB   = hipblasLtHalf;
    using OutType   = hipblasLtHalf;
    using AlphaType = hipblasLtFloat;
    using BetaType  = hipblasLtFloat;

    hipStream_t        stream;
    hipblasLtHandle_t  handle;
    hipblasOperation_t trans_a     = HIPBLAS_OP_N;
    hipblasOperation_t trans_b     = HIPBLAS_OP_T;
    int64_t            m           = arg.M[0];
    int64_t            n           = arg.N[0];
    int64_t            k           = arg.K[0];
    int64_t            batch_count = 1;
    float              alpha       = arg.alpha;
    float              beta        = arg.beta;
    void*              d_a;
    void*              d_b;
    void*              d_c;
    void*              d_d;
    void*              a;
    void*              b;
    void*              c;
    void*              d;

    CHECK_HIP_ERROR(hipStreamCreate(&stream));
    CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));
    CHECK_HIP_ERROR(hipMalloc(&d_a, m * k * batch_count * sizeof(InTypeA)));
    CHECK_HIP_ERROR(hipMalloc(&d_b, n * k * batch_count * sizeof(InTypeB)));
    CHECK_HIP_ERROR(hipMalloc(&d_c, m * n * batch_count * sizeof(OutType)));
    CHECK_HIP_ERROR(hipMalloc(&d_d, m * n * batch_count * sizeof(OutType)));
    CHECK_HIP_ERROR(hipHostMalloc(&a, m * k * batch_count * sizeof(InTypeA)));
    CHECK_HIP_ERROR(hipHostMalloc(&b, n * k * batch_count * sizeof(InTypeB)));
    CHECK_HIP_ERROR(hipHostMalloc(&c, m * n * batch_count * sizeof(OutType)));
    CHECK_HIP_ERROR(hipHostMalloc(&d, m * n * batch_count * sizeof(OutType)));

    CHECK_HIP_ERROR(hipMemcpyAsync(
        d_a, a, m * k * batch_count * sizeof(InTypeA), hipMemcpyHostToDevice, stream));
    CHECK_HIP_ERROR(hipMemcpyAsync(
        d_b, b, n * k * batch_count * sizeof(InTypeB), hipMemcpyHostToDevice, stream));
    CHECK_HIP_ERROR(hipMemcpyAsync(
        d_c, c, m * n * batch_count * sizeof(OutType), hipMemcpyHostToDevice, stream));

    hipblasLtMatrixLayout_t matA, matB, matC, matD;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matA, arg.a_type, m, k, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matB, arg.a_type, k, n, k));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matC, arg.a_type, m, n, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matD, arg.a_type, m, n, m));

    hipblasLtMatmulDesc_t matmul;
    hipblasLtEpilogue_t   epilogue = HIPBLASLT_EPILOGUE_BIAS;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmul, arg.compute_type, arg.scale_type));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(int32_t)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(int32_t)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    hipblasLtMatmulPreference_t pref;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceCreate(&pref));
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

    CHECK_SOLUTION_FOUND(returnedAlgoCount);

    CHECK_HIP_ERROR(hipFree(a));
    CHECK_HIP_ERROR(hipFree(b));
    CHECK_HIP_ERROR(hipFree(c));
    CHECK_HIP_ERROR(hipFree(d));
    CHECK_HIP_ERROR(hipFree(d_a));
    CHECK_HIP_ERROR(hipFree(d_b));
    CHECK_HIP_ERROR(hipFree(d_c));
    CHECK_HIP_ERROR(hipFree(d_d));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matD));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(pref));
    CHECK_HIPBLASLT_ERROR(hipblasLtDestroy(handle));
    CHECK_HIP_ERROR(hipStreamDestroy(stream));
}

// hipBLASLt API: For testing case of (alpha=0 && (A=NULL || B=NULL))
void testing_aux_get_sol_with_zero_alpha_null_a_b(const Arguments& arg)
{
    using InTypeA   = hipblasLtHalf;
    using InTypeB   = hipblasLtHalf;
    using OutType   = hipblasLtHalf;
    using AlphaType = hipblasLtFloat;
    using BetaType  = hipblasLtFloat;

    hipStream_t        stream;
    hipblasLtHandle_t  handle;
    hipblasOperation_t trans_a     = HIPBLAS_OP_N;
    hipblasOperation_t trans_b     = HIPBLAS_OP_T;
    int64_t            m           = arg.M[0];
    int64_t            n           = arg.N[0];
    int64_t            k           = arg.K[0];
    int64_t            batch_count = 1;
    // Setting alpha = 0.
    float alpha = 0;
    float beta  = arg.beta;
    // Setting d_a, d_b as nullptr.
    void* d_a = NULL;
    void* d_b = NULL;
    void* d_c;
    void* d_d;

    CHECK_HIP_ERROR(hipStreamCreate(&stream));
    CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));
    CHECK_HIP_ERROR(hipMalloc(&d_c, m * n * batch_count * sizeof(OutType)));
    CHECK_HIP_ERROR(hipMalloc(&d_d, m * n * batch_count * sizeof(OutType)));

    hipblasLtMatrixLayout_t matA, matB, matC, matD;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matA, arg.a_type, m, k, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matB, arg.a_type, k, n, k));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matC, arg.a_type, m, n, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matD, arg.a_type, m, n, m));

    hipblasLtMatmulDesc_t matmul;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmul, arg.compute_type, arg.scale_type));
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

    CHECK_SOLUTION_FOUND(returnedAlgoCount);

    // Validation for solution running.
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
                                          nullptr,
                                          0,
                                          stream));

    CHECK_HIP_ERROR(hipFree(d_c));
    CHECK_HIP_ERROR(hipFree(d_d));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matD));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(pref));
    CHECK_HIPBLASLT_ERROR(hipblasLtDestroy(handle));
    CHECK_HIP_ERROR(hipStreamDestroy(stream));
}

// hipBLASLtExt API: For testing case of (alpha=0 && (A=NULL || B=NULL))
void testing_aux_get_sol_with_zero_alpha_null_a_b_ext(const Arguments& arg)
{
    using InTypeA   = hipblasLtHalf;
    using InTypeB   = hipblasLtHalf;
    using OutType   = hipblasLtHalf;
    using AlphaType = hipblasLtFloat;
    using BetaType  = hipblasLtFloat;

    hipStream_t        stream;
    hipblasLtHandle_t  handle;
    hipblasOperation_t trans_a     = HIPBLAS_OP_N;
    hipblasOperation_t trans_b     = HIPBLAS_OP_T;
    int64_t            m           = arg.M[0];
    int64_t            n           = arg.N[0];
    int64_t            k           = arg.K[0];
    int64_t            batch_count = 1;
    // Setting alpha = 0.
    float alpha = 0;
    float beta  = arg.beta;
    // Setting d_a, d_b as nullptr.
    void* d_a = NULL;
    void* d_b = NULL;
    void* d_c;
    void* d_d;

    CHECK_HIP_ERROR(hipStreamCreate(&stream));
    CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));
    CHECK_HIP_ERROR(hipMalloc(&d_c, m * n * batch_count * sizeof(OutType)));
    CHECK_HIP_ERROR(hipMalloc(&d_d, m * n * batch_count * sizeof(OutType)));

    hipblaslt_ext::GemmPreferenceV2 gemmPref;
    hipblaslt_ext::Gemm             gemm(
        handle, trans_a, trans_b, arg.a_type, arg.a_type, arg.a_type, arg.a_type, arg.compute_type);

    hipblaslt_ext::GemmEpilogue
        epilogue; // No action needed, default is HIPBLASLT_EPILOGUE_DEFAULT. (Gemm only)
    hipblaslt_ext::GemmInputs inputs;
    inputs.a     = d_a;
    inputs.b     = d_b;
    inputs.c     = d_c;
    inputs.d     = d_d;
    inputs.alpha = &alpha;
    inputs.beta  = &beta;
    gemm.setProblem(m, n, k, batch_count, epilogue, inputs);

    const int                                     request_solutions = 1;
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;
    CHECK_HIPBLASLT_ERROR(gemm.algoGetHeuristic(request_solutions, gemmPref, heuristicResult));
    CHECK_SOLUTION_FOUND(heuristicResult.size());

    // Make sure to initialize every time when algo changes
    CHECK_HIPBLASLT_ERROR(gemm.initialize(heuristicResult[0].algo, nullptr));
    // Validation for solution running.
    CHECK_HIPBLASLT_ERROR(gemm.run(stream));

    CHECK_HIP_ERROR(hipFree(d_c));
    CHECK_HIP_ERROR(hipFree(d_d));
    CHECK_HIPBLASLT_ERROR(hipblasLtDestroy(handle));
    CHECK_HIP_ERROR(hipStreamDestroy(stream));
}

void testing_aux_matmul_alg_get_attr_bad_arg(const Arguments& arg) {}

void testing_aux_matmul_alg_null_matmul(const Arguments& arg)
{
    using InTypeA   = hipblasLtHalf;
    using InTypeB   = hipblasLtHalf;
    using OutType   = hipblasLtHalf;
    using AlphaType = hipblasLtFloat;
    using BetaType  = hipblasLtFloat;

    hipStream_t        stream;
    hipblasLtHandle_t  handle;
    hipblasOperation_t trans_a     = HIPBLAS_OP_N;
    hipblasOperation_t trans_b     = HIPBLAS_OP_T;
    int64_t            m           = arg.M[0];
    int64_t            n           = arg.N[0];
    int64_t            k           = arg.K[0];
    int64_t            batch_count = 1;
    float              alpha       = arg.alpha;
    float              beta        = arg.beta;
    void*              d_a;
    void*              d_b;
    void*              d_c;
    void*              d_d;
    void*              a;
    void*              b;
    void*              c;
    void*              d;

    CHECK_HIP_ERROR(hipStreamCreate(&stream));
    CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));
    CHECK_HIP_ERROR(hipMalloc(&d_a, m * k * batch_count * sizeof(InTypeA)));
    CHECK_HIP_ERROR(hipMalloc(&d_b, n * k * batch_count * sizeof(InTypeB)));
    CHECK_HIP_ERROR(hipMalloc(&d_c, m * n * batch_count * sizeof(OutType)));
    CHECK_HIP_ERROR(hipMalloc(&d_d, m * n * batch_count * sizeof(OutType)));
    CHECK_HIP_ERROR(hipHostMalloc(&a, m * k * batch_count * sizeof(InTypeA)));
    CHECK_HIP_ERROR(hipHostMalloc(&b, n * k * batch_count * sizeof(InTypeB)));
    CHECK_HIP_ERROR(hipHostMalloc(&c, m * n * batch_count * sizeof(OutType)));
    CHECK_HIP_ERROR(hipHostMalloc(&d, m * n * batch_count * sizeof(OutType)));

    CHECK_HIP_ERROR(hipMemcpyAsync(
        d_a, a, m * k * batch_count * sizeof(InTypeA), hipMemcpyHostToDevice, stream));
    CHECK_HIP_ERROR(hipMemcpyAsync(
        d_b, b, n * k * batch_count * sizeof(InTypeB), hipMemcpyHostToDevice, stream));
    CHECK_HIP_ERROR(hipMemcpyAsync(
        d_c, c, m * n * batch_count * sizeof(OutType), hipMemcpyHostToDevice, stream));

    hipblasLtMatrixLayout_t matA, matB, matC, matD;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matA, arg.a_type, m, k, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matB, arg.a_type, k, n, k));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matC, arg.a_type, m, n, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matD, arg.a_type, m, n, m));

    hipblasLtMatmulDesc_t matmul;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmul, arg.compute_type, arg.scale_type));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(int32_t)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(int32_t)));

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
                                          nullptr,
                                          nullptr,
                                          0,
                                          0));

    CHECK_HIP_ERROR(hipFree(a));
    CHECK_HIP_ERROR(hipFree(b));
    CHECK_HIP_ERROR(hipFree(c));
    CHECK_HIP_ERROR(hipFree(d));
    CHECK_HIP_ERROR(hipFree(d_a));
    CHECK_HIP_ERROR(hipFree(d_b));
    CHECK_HIP_ERROR(hipFree(d_c));
    CHECK_HIP_ERROR(hipFree(d_d));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matD));
    CHECK_HIPBLASLT_ERROR(hipblasLtDestroy(handle));
    CHECK_HIP_ERROR(hipStreamDestroy(stream));
}

void testing_aux_matmul_pref_init_bad_arg(const Arguments& arg)
{
    hipblasLtMatmulPreference_t pref;
    size_t                      workspace_size = 0;

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulPreferenceCreate(nullptr), HIPBLAS_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_pref_init(const Arguments& arg)
{
    hipblaslt_local_preference pref;
    EXPECT_HIPBLAS_STATUS(pref.status(), HIPBLAS_STATUS_SUCCESS);
}
