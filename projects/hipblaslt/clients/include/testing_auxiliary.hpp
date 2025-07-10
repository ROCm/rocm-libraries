/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
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

    int32_t data32;
    int64_t data64;
    uint32_t udata32;
    uint64_t udata64;

    hipblaslt_local_matrix_layout mat(row, col, ld, arg.a_type);
    EXPECT_HIPBLAS_STATUS(mat.status(), HIPBLAS_STATUS_SUCCESS);
    // Test with null matrix layout
    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutSetAttribute(
                            nullptr, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &data32, sizeof(int32_t)),
                        HIPBLAS_STATUS_NOT_INITIALIZED);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutSetAttribute(
                            mat, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, nullptr, sizeof(int32_t)),
                        HIPBLAS_STATUS_INVALID_VALUE);
    

    // Test with zero size
    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutSetAttribute(
        mat, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &data32, 0),
    HIPBLAS_STATUS_INVALID_VALUE);

    // Test HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT with insufficient buffer size
    data32 = 1;
    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutSetAttribute(
            mat, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &data32, sizeof(int32_t)/2),
        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutSetAttribute(
            mat, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &data32, 1),
        HIPBLAS_STATUS_INVALID_VALUE);

    // Test HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET with insufficient buffer size
    data64 = ld * col;
    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutSetAttribute(
            mat, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &data64, sizeof(int64_t)/2),
        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutSetAttribute(
            mat, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &data64, 1),
        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutSetAttribute(
            mat, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, nullptr, sizeof(int64_t)),
        HIPBLAS_STATUS_INVALID_VALUE);

    // Test HIPBLASLT_MATRIX_LAYOUT_TYPE with insufficient buffer size
    udata32 = static_cast<uint32_t>(arg.a_type);
    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutSetAttribute(
            mat, HIPBLASLT_MATRIX_LAYOUT_TYPE, &udata32, sizeof(uint32_t)/2),
        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutSetAttribute(
            mat, HIPBLASLT_MATRIX_LAYOUT_TYPE, &udata32, 1),
        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutSetAttribute(
            mat, HIPBLASLT_MATRIX_LAYOUT_TYPE, nullptr, sizeof(uint32_t)),
        HIPBLAS_STATUS_INVALID_VALUE);

    // Test HIPBLASLT_MATRIX_LAYOUT_ORDER with insufficient buffer size
    data32 = HIPBLASLT_ORDER_COL;
    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutSetAttribute(
                            mat, HIPBLASLT_MATRIX_LAYOUT_ORDER, &data32, sizeof(int32_t)/2),
                        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutSetAttribute(
                            mat, HIPBLASLT_MATRIX_LAYOUT_ORDER, &data32, 1),
                        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutSetAttribute(
                            mat, HIPBLASLT_MATRIX_LAYOUT_ORDER, nullptr, sizeof(int32_t)),
                        HIPBLAS_STATUS_INVALID_VALUE);

    // Test HIPBLASLT_MATRIX_LAYOUT_ROWS with insufficient buffer size
    udata64 = static_cast<uint64_t>(row);
    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutSetAttribute(
                            mat, HIPBLASLT_MATRIX_LAYOUT_ROWS, &udata64, sizeof(uint64_t)/2),
                        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutSetAttribute(
                            mat, HIPBLASLT_MATRIX_LAYOUT_ROWS, &udata64, 1),
                        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutSetAttribute(
                            mat, HIPBLASLT_MATRIX_LAYOUT_ROWS, nullptr, sizeof(uint64_t)),
                        HIPBLAS_STATUS_INVALID_VALUE);

    // Test HIPBLASLT_MATRIX_LAYOUT_COLS with insufficient buffer size
    udata64 = static_cast<uint64_t>(col);
    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutSetAttribute(
                            mat, HIPBLASLT_MATRIX_LAYOUT_COLS, &udata64, sizeof(uint64_t)/2),
                        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutSetAttribute(
                            mat, HIPBLASLT_MATRIX_LAYOUT_COLS, &udata64, 1),
                        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutSetAttribute(
                            mat, HIPBLASLT_MATRIX_LAYOUT_COLS, nullptr, sizeof(uint64_t)),
                        HIPBLAS_STATUS_INVALID_VALUE);

    // Test HIPBLASLT_MATRIX_LAYOUT_LD with insufficient buffer size
    data64 = ld;
    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutSetAttribute(
                            mat, HIPBLASLT_MATRIX_LAYOUT_LD, &data64, sizeof(int64_t)/2),
                        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutSetAttribute(
                            mat, HIPBLASLT_MATRIX_LAYOUT_LD, &data64, 1),
                        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutSetAttribute(
                            mat, HIPBLASLT_MATRIX_LAYOUT_LD, nullptr, sizeof(int64_t)),
                        HIPBLAS_STATUS_INVALID_VALUE);

    // Test with invalid attribute (assuming there's a MAX value for bounds checking)
    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutSetAttribute(
                            mat, static_cast<hipblasLtMatrixLayoutAttribute_t>(999), &data32, sizeof(int32_t)),
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

    EXPECT_HIPBLAS_STATUS(
            hipblasLtMatrixLayoutGetAttribute(
                mat, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, nullptr, 0, nullptr),
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
    size_t              sizeWritten;
    hipblasLtMatmulDesc_t matmul;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmul, arg.compute_type, arg.scale_type));
    int64_t m = 1;
    int64_t n = 1;
    int64_t k = 1;
    
    // For HIPBLASLT_MATMUL_DESC_TRANSA
    hipblasOperation_t transA = HIPBLAS_OP_T;
    hipblasOperation_t transA_r = HIPBLAS_OP_N;

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &transA, 0),
                        HIPBLAS_STATUS_INVALID_VALUE); // edge case


    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(int32_t)/2),
                        HIPBLAS_STATUS_INVALID_VALUE); // edge case

    
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(int32_t)),
                            HIPBLAS_STATUS_SUCCESS); // normal case

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &transA_r, 0, nullptr),
                        HIPBLASLT_MATMUL_DESC_BIAS_POINTER); // edge case


    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &transA_r, sizeof(transA_r)/2, &sizeWritten),
                        HIPBLAS_STATUS_INVALID_VALUE); // edge case
    
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &transA_r, sizeof(transA_r), &sizeWritten),
                        HIPBLAS_STATUS_SUCCESS); // normal case
    
    ASSERT_TRUE(transA_r == transA); // validate

    // For HIPBLASLT_MATMUL_DESC_TRANSB
    
    hipblasOperation_t transB = HIPBLAS_OP_N;
    hipblasOperation_t transB_r = HIPBLAS_OP_T;

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(int32_t)/2),
                        HIPBLAS_STATUS_INVALID_VALUE); // edge case

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(int32_t)),
                        HIPBLAS_STATUS_SUCCESS); // normal case             

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &transB_r, sizeof(transB_r)/2, &sizeWritten),
                        HIPBLAS_STATUS_INVALID_VALUE); // edge case

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &transB_r, sizeof(transB_r), &sizeWritten),
                        HIPBLAS_STATUS_SUCCESS); // normal case  

    ASSERT_TRUE(transB_r == transB); // validate

    // for HIPBLASLT_MATMUL_DESC_EPILOGUE
    hipblasLtEpilogue_t data   = HIPBLASLT_EPILOGUE_DEFAULT; 
    hipblasLtEpilogue_t data_r = HIPBLASLT_EPILOGUE_RELU; 
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute( 
                              matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &data, sizeof(data)), 
                          HIPBLAS_STATUS_SUCCESS); 
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &data_r, sizeof(data_r), &sizeWritten),
                        HIPBLAS_STATUS_SUCCESS);
    ASSERT_TRUE(data_r == data);


    // for HIPBLASLT_MATMUL_DESC_BIAS_POINTER set and get
    void*                      d_bias;
    void*                      d_bias_r;
    CHECK_HIP_ERROR(hipMalloc(&d_bias, k * sizeof(hipblasLtHalf)));
    
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &d_bias, sizeof(void*)/2), 
                        HIPBLAS_STATUS_INVALID_VALUE);
    
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &d_bias, sizeof(void*)), 
                        HIPBLAS_STATUS_SUCCESS);
    
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &d_bias_r, sizeof(d_bias_r), &sizeWritten),
                        HIPBLAS_STATUS_SUCCESS);
    
    ASSERT_TRUE(d_bias_r == d_bias);

    // for HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER
    // We create a new matmul_descr becasue we don't want to set matmulDesc->scaleAType as Scalar in advance.
    // If we set it as Scalar, it will skip some cases, which is not what we desire for.
    const hipblasOperation_t opA = HIPBLAS_OP_T;
    const hipblasOperation_t opB = HIPBLAS_OP_N;

    hipblaslt_local_matmul_descr matmul_descr(opA, opB, arg.compute_type, arg.scale_type);
    EXPECT_HIPBLAS_STATUS(matmul_descr.status(), HIPBLAS_STATUS_SUCCESS);
    
    float h_scale_a_for_desc_a_scale_pointer = 2.f; // Example scale value, adjust as needed
    float* d_scale_a;
    float* d_scale_a_r;
    CHECK_HIP_ERROR(hipMalloc(&d_scale_a, sizeof(float)));
    CHECK_HIP_ERROR(hipMemcpy(d_scale_a, &h_scale_a_for_desc_a_scale_pointer, sizeof(float), hipMemcpyHostToDevice));

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                            matmul_descr, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_scale_a, sizeof(float*)/2),
                        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                            matmul_descr, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_scale_a, sizeof(float*)),
                        HIPBLAS_STATUS_SUCCESS);
                        
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                            matmul_descr, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_scale_a_r, sizeof(float*)/2, &sizeWritten),
                        HIPBLAS_STATUS_INVALID_VALUE);
    
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                            matmul_descr, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_scale_a_r, sizeof(float*), &sizeWritten),
                        HIPBLAS_STATUS_SUCCESS);
    ASSERT_TRUE(d_scale_a_r == d_scale_a); // validate

    // HIPBLASLT_MATMUL_DESC_A_SCALE_MODE

    hipblasLtMatmulMatrixScale_t scale_mode_a = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    hipblasLtMatmulMatrixScale_t scale_mode_a_r = HIPBLASLT_MATMUL_MATRIX_SCALE_END;

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode_a_r, sizeof(uint32_t), &sizeWritten),
                        HIPBLAS_STATUS_INVALID_VALUE); // we didn't set it yet, so we get error

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode_a, sizeof(uint32_t)/2), 
                        HIPBLAS_STATUS_INVALID_VALUE);
    
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode_a, sizeof(uint32_t)), 
                        HIPBLAS_STATUS_SUCCESS);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode_a_r, sizeof(uint32_t)/2, &sizeWritten),
                        HIPBLAS_STATUS_INVALID_VALUE);
    
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode_a_r, sizeof(uint32_t), &sizeWritten),
                        HIPBLAS_STATUS_SUCCESS);
    
    ASSERT_TRUE(scale_mode_a_r == scale_mode_a); // validate

    // HIPBLASLT_MATMUL_DESC_B_SCALE_MODE

    hipblasLtMatmulMatrixScale_t scale_mode_b = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    hipblasLtMatmulMatrixScale_t scale_mode_b_r = HIPBLASLT_MATMUL_MATRIX_SCALE_END;

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode_b_r, sizeof(uint32_t), &sizeWritten),
                        HIPBLAS_STATUS_INVALID_VALUE); // we didn't set it yet, so we get error

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode_b, sizeof(uint32_t)/2), 
                        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode_b, sizeof(uint32_t)), 
                        HIPBLAS_STATUS_SUCCESS);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode_b_r, sizeof(uint32_t)/2, &sizeWritten),
                        HIPBLAS_STATUS_INVALID_VALUE);
    
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode_b_r, sizeof(uint32_t), &sizeWritten),
                        HIPBLAS_STATUS_SUCCESS);

    ASSERT_TRUE(scale_mode_b_r == scale_mode_b); // validate

    scale_mode_a = HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    scale_mode_b = HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode_a, sizeof(uint32_t)),
                        HIPBLAS_STATUS_SUCCESS);
    
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode_a_r, sizeof(uint32_t), &sizeWritten),
                        HIPBLAS_STATUS_SUCCESS);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode_b, sizeof(uint32_t)),
                        HIPBLAS_STATUS_SUCCESS);
    
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode_b_r, sizeof(uint32_t), &sizeWritten),
                        HIPBLAS_STATUS_SUCCESS);

    
    ASSERT_TRUE(scale_mode_a_r == scale_mode_a); // validate
    ASSERT_TRUE(scale_mode_b_r == scale_mode_b); // ditto

        
    scale_mode_a = HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
    scale_mode_b = HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode_a, sizeof(uint32_t)),
                            HIPBLAS_STATUS_SUCCESS);
    
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode_a_r, sizeof(uint32_t), &sizeWritten),
                        HIPBLAS_STATUS_SUCCESS);
    
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode_b, sizeof(uint32_t)),
                        HIPBLAS_STATUS_SUCCESS);
    
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode_b_r, sizeof(uint32_t), &sizeWritten),
                        HIPBLAS_STATUS_SUCCESS);
    
                        
    ASSERT_TRUE(scale_mode_a_r == scale_mode_a); // validate

    scale_mode_a = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3; // will not set anything
    scale_mode_b = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3; // ditto

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode_a, sizeof(uint32_t)),
                        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode_a_r, sizeof(uint32_t), &sizeWritten),
                        HIPBLAS_STATUS_SUCCESS);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode_b, sizeof(uint32_t)),
                        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode_b_r, sizeof(uint32_t), &sizeWritten),
                        HIPBLAS_STATUS_SUCCESS);


    ASSERT_TRUE(scale_mode_a_r == HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F); // validate, it's still the previous value as expected
    ASSERT_TRUE(scale_mode_b_r == HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F); // ditto


    hipStream_t        stream;
    CHECK_HIP_ERROR(hipStreamCreate(&stream));

    // for HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER 
    float h_scale_b = 3.f;
    float* d_scale_b;
    float* d_scale_b_r;
    CHECK_HIP_ERROR(hipMalloc(&d_scale_b, sizeof(float)));
    CHECK_HIP_ERROR(hipMemcpyAsync(d_scale_b, &h_scale_b, sizeof(float), hipMemcpyHostToDevice, stream));

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_scale_b, sizeof(float*)/2),
                        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_scale_b, sizeof(float*)),
                        HIPBLAS_STATUS_SUCCESS);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_scale_b_r, sizeof(float*)/2, &sizeWritten),
                        HIPBLAS_STATUS_INVALID_VALUE);
    
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_scale_b_r, sizeof(float*), &sizeWritten),
                        HIPBLAS_STATUS_SUCCESS);

    ASSERT_TRUE(d_scale_b == d_scale_b_r); // validate

    // for HIPBLASLT_MATMUL_DESC_C_SCALE_POINTER & HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER
    float h_scale_c = 3.f;
    float h_scale_d = 3.f;
    float* d_scale_c;
    float* d_scale_d;
    CHECK_HIP_ERROR(hipMalloc(&d_scale_c, sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_scale_d, sizeof(float)));
    CHECK_HIP_ERROR(hipMemcpyAsync(d_scale_c, &h_scale_c, sizeof(float), hipMemcpyHostToDevice, stream));
    CHECK_HIP_ERROR(hipMemcpyAsync(d_scale_d, &h_scale_d, sizeof(float), hipMemcpyHostToDevice, stream));

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_C_SCALE_POINTER, &d_scale_c, sizeof(float*)/2),
                        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_C_SCALE_POINTER, &d_scale_c, sizeof(float*)),
                        HIPBLAS_STATUS_SUCCESS);
    
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER, &d_scale_d, sizeof(float*)/2),
                        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER, &d_scale_d, sizeof(float*)),
                        HIPBLAS_STATUS_SUCCESS);

    
    // for HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER
    float h_scale_e = 3.f;
    float* d_scale_e;
    CHECK_HIP_ERROR(hipMalloc(&d_scale_e, sizeof(float)));
    CHECK_HIP_ERROR(hipMemcpyAsync(d_scale_e, &h_scale_e, sizeof(float), hipMemcpyHostToDevice, stream));

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER, &d_scale_e, sizeof(float*)/2),
                        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER, &d_scale_e, sizeof(float*)),
                        HIPBLAS_STATUS_SUCCESS);
    
    // For HIPBLASLT_MATMUL_DESC_POINTER_MODE
    hipblasLtPointerMode_t pMode = HIPBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST;
    hipblasLtPointerMode_t pMode_r = HIPBLASLT_POINTER_MODE_HOST;
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                                matmul, HIPBLASLT_MATMUL_DESC_POINTER_MODE, &pMode, sizeof(pMode)/2),
                            HIPBLAS_STATUS_INVALID_VALUE);
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                                matmul, HIPBLASLT_MATMUL_DESC_POINTER_MODE, &pMode, sizeof(pMode)),
                            HIPBLAS_STATUS_SUCCESS);
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                                matmul, HIPBLASLT_MATMUL_DESC_POINTER_MODE, &pMode_r, sizeof(pMode_r)/2, &sizeWritten),
                                HIPBLAS_STATUS_INVALID_VALUE);
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                                matmul, HIPBLASLT_MATMUL_DESC_POINTER_MODE, &pMode_r, sizeof(pMode_r), &sizeWritten),
                            HIPBLAS_STATUS_SUCCESS);

    ASSERT_TRUE(pMode_r == pMode); // validate

    // For HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE Set Desc Bias Data Type
    int32_t bias_data_type = HIP_R_16F;
    int32_t bias_data_type_r = HIP_R_32F;
    
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                                matmul, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_data_type, sizeof(bias_data_type)/2),
                            HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                                matmul, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_data_type, sizeof(bias_data_type)),
                            HIPBLAS_STATUS_SUCCESS);


    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                                matmul, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_data_type_r, sizeof(bias_data_type_r)/2, &sizeWritten),
                            HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                                matmul, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_data_type_r, sizeof(bias_data_type_r), &sizeWritten),
                                HIPBLAS_STATUS_SUCCESS);

    ASSERT_TRUE(bias_data_type_r == bias_data_type); // validate


    // For HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER Set auxiliary buffer
    void* d_aux_buffer;
    CHECK_HIP_ERROR(hipMalloc(&d_aux_buffer, m * n * sizeof(hipblasLtHalf)));
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &d_aux_buffer, sizeof(void*)/2),
                        HIPBLAS_STATUS_INVALID_VALUE);
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &d_aux_buffer, sizeof(void*)),
                        HIPBLAS_STATUS_SUCCESS);

    
    // For HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD Set auxiliary leading dimension (ld)
    const int64_t aux_ld = m;
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &aux_ld, sizeof(aux_ld)/2),
                        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &aux_ld, sizeof(aux_ld)),
                        HIPBLAS_STATUS_SUCCESS);
    
    // for HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE & Set Epilogue Aux Batch Stride
    const int64_t aux_batch_stride = m * n;
    EXPECT_HIPBLAS_STATUS( hipblasLtMatmulDescSetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE, &aux_batch_stride, sizeof(aux_batch_stride)/2), 
                        HIPBLAS_STATUS_INVALID_VALUE);
    
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE, &aux_batch_stride, sizeof(aux_batch_stride)),
                        HIPBLAS_STATUS_SUCCESS);

    // for HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER
    void *d_out_amax;
    void *d_out_amax_r;
    CHECK_HIP_ERROR(hipMalloc(&d_out_amax, 1 * sizeof(float)));
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER, &d_out_amax, sizeof(void*)/2),
                        HIPBLAS_STATUS_INVALID_VALUE);
    
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER, &d_out_amax, sizeof(void*)),
                        HIPBLAS_STATUS_SUCCESS);


    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER, &d_out_amax_r, sizeof(d_out_amax_r)/2, &sizeWritten),
                        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER, &d_out_amax_r, sizeof(d_out_amax_r), &sizeWritten),
                        HIPBLAS_STATUS_SUCCESS);

    ASSERT_TRUE(d_out_amax == d_out_amax_r); // validate

    // for HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE
    hipDataType aux_type_r;
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE, &arg.aux_type, sizeof(hipDataType)/2),
                        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE, &arg.aux_type, sizeof(hipDataType)),
                        HIPBLAS_STATUS_SUCCESS);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE, &aux_type_r, sizeof(aux_type_r)/2, &sizeWritten),
                        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE, &aux_type_r, sizeof(aux_type_r), &sizeWritten),
                        HIPBLAS_STATUS_SUCCESS);

    ASSERT_TRUE(arg.aux_type == aux_type_r); // validate

    // for HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT & HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT
    hipDataType computeTypeA = HIP_R_16F;
    hipDataType computeTypeA_r = HIP_R_32F;
    hipDataType computeTypeB = HIP_R_16F;
    hipDataType computeTypeB_r = HIP_R_32F;

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT, &computeTypeA, sizeof(computeTypeA)/2),
                        HIPBLAS_STATUS_INVALID_VALUE);

    
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT, &computeTypeA, sizeof(computeTypeA)),
                        HIPBLAS_STATUS_SUCCESS);

        
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT, &computeTypeB, sizeof(computeTypeB)/2),
                        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT, &computeTypeB, sizeof(computeTypeB)),
                        HIPBLAS_STATUS_SUCCESS);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT, &computeTypeA_r, sizeof(computeTypeA_r)/2, &sizeWritten),
                        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT, &computeTypeA_r, sizeof(computeTypeA_r), &sizeWritten),
                        HIPBLAS_STATUS_SUCCESS);

                        
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT, &computeTypeB_r, sizeof(computeTypeB_r)/2, &sizeWritten),
                        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT, &computeTypeB_r, sizeof(computeTypeB_r), &sizeWritten),
                        HIPBLAS_STATUS_SUCCESS);

    ASSERT_TRUE(computeTypeA_r == computeTypeA);
    ASSERT_TRUE(computeTypeB_r == computeTypeB);

    // for default

    void* default_ptr;
    void* default_ptr_r;
    CHECK_HIP_ERROR(hipMalloc(&default_ptr, sizeof(hipblasLtHalf)));
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_MAX, &default_ptr, sizeof(void*)), 
                        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                            matmul, HIPBLASLT_MATMUL_DESC_MAX, &default_ptr_r, sizeof(default_ptr_r), &sizeWritten),
                        HIPBLAS_STATUS_INVALID_VALUE);
    
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul));
    CHECK_HIP_ERROR(hipStreamDestroy(stream));
    CHECK_HIP_ERROR(hipFree(d_scale_a));
    CHECK_HIP_ERROR(hipFree(d_bias));
    CHECK_HIP_ERROR(hipFree(d_scale_b));
    CHECK_HIP_ERROR(hipFree(d_scale_c));
    CHECK_HIP_ERROR(hipFree(d_scale_d));
    CHECK_HIP_ERROR(hipFree(d_aux_buffer));
    CHECK_HIP_ERROR(hipFree(d_out_amax));
    CHECK_HIP_ERROR(hipFree(default_ptr));
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

    // Test edge cases
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulPreferenceSetAttribute(nullptr, 
                                            HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                            &data_set,
                                            sizeof(data_set)),
        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulPreferenceSetAttribute(pref, 
                                            HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                            &data_set,
                                            0),
        HIPBLAS_STATUS_INVALID_VALUE);


    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulPreferenceGetAttribute(pref,
                                            HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                            nullptr,
                                            0,
                                            &sizeWritten),
        HIPBLAS_STATUS_INVALID_VALUE);
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
    int32_t search_mode = 0;
    int32_t search_mode_r = 1;
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulPreferenceSetAttribute(pref,
                                                                HIPBLASLT_MATMUL_PREF_SEARCH_MODE,
                                                                &search_mode,
                                                                sizeof(search_mode)),
                        HIPBLAS_STATUS_SUCCESS);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulPreferenceGetAttribute(pref,
                                                                HIPBLASLT_MATMUL_PREF_SEARCH_MODE,
                                                                &search_mode_r,
                                                                sizeof(search_mode_r),
                                                                &sizeWritten),
                        HIPBLAS_STATUS_SUCCESS);
    ASSERT_TRUE(data_get == data_set);
    
    // Test default value
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulPreferenceSetAttribute(pref,
                                                                HIPBLASLT_MATMUL_PREF_MAX,
                                                                &search_mode,
                                                                sizeof(search_mode)),
                        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulPreferenceGetAttribute(pref,
                                                                HIPBLASLT_MATMUL_PREF_MAX,
                                                                &search_mode_r,
                                                                sizeof(search_mode_r),
                                                                &sizeWritten),
                        HIPBLAS_STATUS_INVALID_VALUE);

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

    hipblaslt_ext::GemmPreference gemmPref;
    hipblaslt_ext::Gemm           gemm(
        handle, trans_a, trans_b, arg.a_type, arg.a_type, arg.a_type, arg.a_type, arg.compute_type);

    hipblaslt_ext::GemmEpilogue
        epilogue; // No action needed, default is HIPBLASLT_EPILOGUE_DEFAULT. (Gemm only)
    hipblaslt_ext::GemmInputs inputs;
    inputs.setA(d_a);
    inputs.setB(d_b);
    inputs.setC(d_c);
    inputs.setD(d_d);
    inputs.setAlpha(&alpha);
    inputs.setBeta(&beta);
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

void testing_aux_mat_copy(const Arguments& arg)
{
    hipblasLtMatmulDesc_t matmul_src;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmul_src, arg.compute_type, arg.scale_type));
    hipblasLtMatmulDesc_t matmul_dest;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmul_dest, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    
    EXPECT_HIPBLAS_STATUS(hipblaslt_ext::copyMatmul(nullptr, matmul_dest), HIPBLAS_STATUS_INVALID_VALUE);
    EXPECT_HIPBLAS_STATUS(hipblaslt_ext::copyMatmul(matmul_src, nullptr), HIPBLAS_STATUS_INVALID_VALUE);
    EXPECT_HIPBLAS_STATUS(hipblaslt_ext::copyMatmul(matmul_src, matmul_dest), HIPBLAS_STATUS_SUCCESS);
    
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul_src));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul_dest));
}

void testing_aux_auxiliary_func(const Arguments& arg)
{
    // Test gpu_arch_match
    int             deviceId;
    hipDeviceProp_t deviceProperties;
    static_cast<void>(hipGetDevice(&deviceId));
    static_cast<void>(hipGetDeviceProperties(&deviceProperties, deviceId));
    ASSERT_TRUE(gpu_arch_match(deviceProperties.gcnArchName, ""));
    ASSERT_TRUE(gpu_arch_match(deviceProperties.gcnArchName, "\\d"));

    // Test hipblas_status_to_string
    ASSERT_TRUE(hipblas_status_to_string(HIPBLAS_STATUS_SUCCESS) == "HIPBLAS_STATUS_SUCCESS");
    ASSERT_TRUE(hipblas_status_to_string(HIPBLAS_STATUS_NOT_INITIALIZED) == "HIPBLAS_STATUS_NOT_INITIALIZED");
    ASSERT_TRUE(hipblas_status_to_string(HIPBLAS_STATUS_ALLOC_FAILED) == "HIPBLAS_STATUS_ALLOC_FAILED");
    ASSERT_TRUE(hipblas_status_to_string(HIPBLAS_STATUS_INVALID_VALUE) == "HIPBLAS_STATUS_INVALID_VALUE");
    ASSERT_TRUE(hipblas_status_to_string(HIPBLAS_STATUS_MAPPING_ERROR) == "HIPBLAS_STATUS_MAPPING_ERROR");
    ASSERT_TRUE(hipblas_status_to_string(HIPBLAS_STATUS_EXECUTION_FAILED) == "HIPBLAS_STATUS_EXECUTION_FAILED");
    ASSERT_TRUE(hipblas_status_to_string(HIPBLAS_STATUS_INTERNAL_ERROR) == "HIPBLAS_STATUS_INTERNAL_ERROR");
    ASSERT_TRUE(hipblas_status_to_string(HIPBLAS_STATUS_NOT_SUPPORTED) == "HIPBLAS_STATUS_NOT_SUPPORTED");
    ASSERT_TRUE(hipblas_status_to_string(HIPBLAS_STATUS_ARCH_MISMATCH) == "HIPBLAS_STATUS_ARCH_MISMATCH");
    ASSERT_TRUE(hipblas_status_to_string(HIPBLAS_STATUS_INVALID_ENUM) == "HIPBLAS_STATUS_INVALID_ENUM");
    ASSERT_TRUE(hipblas_status_to_string(HIPBLAS_STATUS_UNKNOWN) == "HIPBLAS_STATUS_UNKNOWN");
    ASSERT_TRUE(hipblas_status_to_string(static_cast<hipblasStatus_t>(12)) == "<undefined hipblasStatus_t value>");
    
    // Test hipblas_operation_to_string
    
    ASSERT_TRUE(hipblas_operation_to_string(HIPBLAS_OP_N) == "N");
    ASSERT_TRUE(hipblas_operation_to_string(HIPBLAS_OP_T) == "T");
    ASSERT_TRUE(hipblas_operation_to_string(HIPBLAS_OP_C) == "C");
    ASSERT_TRUE(hipblas_operation_to_string(static_cast<hipblasOperation_t>(114)) == "invalid");

    // Test char_to_hipblas_operation
    ASSERT_TRUE(char_to_hipblas_operation('N') == HIPBLAS_OP_N);
    ASSERT_TRUE(char_to_hipblas_operation('n') == HIPBLAS_OP_N);
    ASSERT_TRUE(char_to_hipblas_operation('T') == HIPBLAS_OP_T);
    ASSERT_TRUE(char_to_hipblas_operation('t') == HIPBLAS_OP_T);
    ASSERT_TRUE(char_to_hipblas_operation('C') == HIPBLAS_OP_C);
    ASSERT_TRUE(char_to_hipblas_operation('c') == HIPBLAS_OP_C);
    ASSERT_TRUE(char_to_hipblas_operation('X') == HIPBLASLT_OPERATION_INVALID);

    
    // Test char_to_hipblas_operation
    ASSERT_TRUE(char_to_hipblas_operation('N') == HIPBLAS_OP_N);
    ASSERT_TRUE(char_to_hipblas_operation('n') == HIPBLAS_OP_N);
    ASSERT_TRUE(char_to_hipblas_operation('T') == HIPBLAS_OP_T);
    ASSERT_TRUE(char_to_hipblas_operation('t') == HIPBLAS_OP_T);
    ASSERT_TRUE(char_to_hipblas_operation('C') == HIPBLAS_OP_C);
    ASSERT_TRUE(char_to_hipblas_operation('c') == HIPBLAS_OP_C);
    
    // Test hip_datatype_to_string
    ASSERT_TRUE(hip_datatype_to_string(HIP_R_32F) == "f32_r");
    ASSERT_TRUE(hip_datatype_to_string(HIP_R_64F) == "f64_r");
    ASSERT_TRUE(hip_datatype_to_string(HIP_R_16F) == "f16_r");
    ASSERT_TRUE(hip_datatype_to_string(HIP_R_16BF) == "bf16_r");
    ASSERT_TRUE(hip_datatype_to_string(HIP_R_8I) == "i8_r");
    ASSERT_TRUE(hip_datatype_to_string(HIP_R_32I) == "i32_r");
    ASSERT_TRUE(hip_datatype_to_string(HIP_R_8F_E4M3_FNUZ) == "f8_fnuz_r");
    ASSERT_TRUE(hip_datatype_to_string(HIP_R_8F_E5M2_FNUZ) == "bf8_fnuz_r");
    ASSERT_TRUE(hip_datatype_to_string(HIP_R_8F_E4M3) == "f8_r");
    ASSERT_TRUE(hip_datatype_to_string(HIP_R_8F_E5M2) == "bf8_r");
    ASSERT_TRUE(hip_datatype_to_string(static_cast<hipDataType>(HIP_R_6F_E2M3_EXT)) == "f6_r");
    ASSERT_TRUE(hip_datatype_to_string(static_cast<hipDataType>(HIP_R_6F_E3M2_EXT)) == "bf6_r");
    ASSERT_TRUE(hip_datatype_to_string(static_cast<hipDataType>(HIP_R_4F_E2M1_EXT)) == "f4_r");


    // Test hipblas_computetype_to_string
    hipblas_computetype_to_string(HIPBLAS_COMPUTE_16F);
    ASSERT_TRUE(hipblas_computetype_to_string(HIPBLAS_COMPUTE_16F) == "f16_r");
    ASSERT_TRUE(hipblas_computetype_to_string(HIPBLAS_COMPUTE_32F) == "f32_r");
    ASSERT_TRUE(hipblas_computetype_to_string(HIPBLAS_COMPUTE_32F_FAST_TF32) == "xf32_r");
    ASSERT_TRUE(hipblas_computetype_to_string(HIPBLAS_COMPUTE_64F) == "f64_r");
    ASSERT_TRUE(hipblas_computetype_to_string(HIPBLAS_COMPUTE_32I) == "i32_r");
    ASSERT_TRUE(hipblas_computetype_to_string(HIPBLAS_COMPUTE_32F_FAST_16F) == "f32_f16_r");
    ASSERT_TRUE(hipblas_computetype_to_string(HIPBLAS_COMPUTE_32F_FAST_16BF) == "f32_bf16_r");
    ASSERT_TRUE(hipblas_computetype_to_string(HIPBLAS_COMPUTE_64F_PEDANTIC) == "non-supported compute type");

    // Test string_to_hip_datatype
    ASSERT_TRUE(string_to_hip_datatype("f8_fnuz_r") == HIP_R_8F_E4M3_FNUZ);
    ASSERT_TRUE(string_to_hip_datatype("bf8_fnuz_r") == HIP_R_8F_E5M2_FNUZ);
    ASSERT_TRUE(string_to_hip_datatype("f8_r") == HIP_R_8F_E4M3);
    ASSERT_TRUE(string_to_hip_datatype("bf8_r") == HIP_R_8F_E5M2);
    ASSERT_TRUE(string_to_hip_datatype("f32_r") == HIP_R_32F);
    ASSERT_TRUE(string_to_hip_datatype("f64_r") == HIP_R_64F);
    ASSERT_TRUE(string_to_hip_datatype("f16_r") == HIP_R_16F);
    ASSERT_TRUE(string_to_hip_datatype("bf16_r") == HIP_R_16BF);
    ASSERT_TRUE(string_to_hip_datatype("i8_r") == HIP_R_8I  );
    ASSERT_TRUE(string_to_hip_datatype("f6_r") == static_cast<hipDataType>(HIP_R_6F_E2M3_EXT));
    ASSERT_TRUE(string_to_hip_datatype("bf6_r") == static_cast<hipDataType>(HIP_R_6F_E3M2_EXT));
    ASSERT_TRUE(string_to_hip_datatype("f4_r") == static_cast<hipDataType>(HIP_R_4F_E2M1_EXT));
    ASSERT_TRUE(string_to_hip_datatype("i32_r") == HIP_R_32I);
    ASSERT_TRUE(string_to_hip_datatype("") == HIPBLASLT_DATATYPE_INVALID);

    // Test string_to_hip_datatype_assert
    string_to_hip_datatype_assert("f8_fnuz_r");

    // Test string_to_hipblas_computetype
    ASSERT_TRUE(string_to_hipblas_computetype("f32_r") == HIPBLAS_COMPUTE_32F);
    ASSERT_TRUE(string_to_hipblas_computetype("xf32_r") == HIPBLAS_COMPUTE_32F_FAST_TF32);
    ASSERT_TRUE(string_to_hipblas_computetype("f64_r") == HIPBLAS_COMPUTE_64F);
    ASSERT_TRUE(string_to_hipblas_computetype("i32_r") == HIPBLAS_COMPUTE_32I);
    ASSERT_TRUE(string_to_hipblas_computetype("f32_f16_r") == HIPBLAS_COMPUTE_32F_FAST_16F);
    ASSERT_TRUE(string_to_hipblas_computetype("f32_bf16_r") == HIPBLAS_COMPUTE_32F_FAST_16BF);
    ASSERT_TRUE(string_to_hipblas_computetype("") == HIPBLASLT_COMPUTE_TYPE_INVALID);

    // Test string_to_hipblas_computetype_assert
    string_to_hipblas_computetype_assert("f32_r");

    // Test string_to_epilogue_type
    ASSERT_TRUE(string_to_epilogue_type("HIPBLASLT_EPILOGUE_RELU") == HIPBLASLT_EPILOGUE_RELU );
    ASSERT_TRUE(string_to_epilogue_type("HIPBLASLT_EPILOGUE_BIAS") == HIPBLASLT_EPILOGUE_BIAS);
    ASSERT_TRUE(string_to_epilogue_type("HIPBLASLT_EPILOGUE_RELU_BIAS") == HIPBLASLT_EPILOGUE_RELU_BIAS);
    ASSERT_TRUE(string_to_epilogue_type("HIPBLASLT_EPILOGUE_GELU") == HIPBLASLT_EPILOGUE_GELU);
    ASSERT_TRUE(string_to_epilogue_type("HIPBLASLT_EPILOGUE_GELU_BIAS") == HIPBLASLT_EPILOGUE_GELU_BIAS);
    ASSERT_TRUE(string_to_epilogue_type("HIPBLASLT_EPILOGUE_GELU_AUX") == HIPBLASLT_EPILOGUE_GELU_AUX);
    ASSERT_TRUE(string_to_epilogue_type("HIPBLASLT_EPILOGUE_GELU_AUX_BIAS") == HIPBLASLT_EPILOGUE_GELU_AUX_BIAS);
    ASSERT_TRUE(string_to_epilogue_type("HIPBLASLT_EPILOGUE_DGELU") == HIPBLASLT_EPILOGUE_DGELU);
    ASSERT_TRUE(string_to_epilogue_type("HIPBLASLT_EPILOGUE_DGELU_BGRAD") == HIPBLASLT_EPILOGUE_DGELU_BGRAD);
    ASSERT_TRUE(string_to_epilogue_type("HIPBLASLT_EPILOGUE_BGRADA") == HIPBLASLT_EPILOGUE_BGRADA);
    ASSERT_TRUE(string_to_epilogue_type("HIPBLASLT_EPILOGUE_BGRADB") == HIPBLASLT_EPILOGUE_BGRADB);
    ASSERT_TRUE(string_to_epilogue_type("HIPBLASLT_EPILOGUE_SWISH_EXT") == HIPBLASLT_EPILOGUE_SWISH_EXT);
    ASSERT_TRUE(string_to_epilogue_type("HIPBLASLT_EPILOGUE_SWISH_BIAS_EXT") == HIPBLASLT_EPILOGUE_SWISH_BIAS_EXT);
    ASSERT_TRUE(string_to_epilogue_type("HIPBLASLT_EPILOGUE_DEFAULT") == HIPBLASLT_EPILOGUE_DEFAULT);
    ASSERT_TRUE(string_to_epilogue_type("test") == static_cast<hipblasLtEpilogue_t>(0));

    // Test string_to_epilogue_type_assert
    ASSERT_TRUE(string_to_epilogue_type_assert("HIPBLASLT_EPILOGUE_RELU") == HIPBLASLT_EPILOGUE_RELU);

    // Test hipblaslt_isnan
    uint32_t other_type_value = 5;
    ASSERT_FALSE(hipblaslt_isnan(other_type_value));

    float non_integral_nan = std::numeric_limits<float>::quiet_NaN();
    ASSERT_TRUE(hipblaslt_isnan(non_integral_nan));

    hipblaslt_f8_fnuz arg_hipblaslt_f8_fnuz;
    arg_hipblaslt_f8_fnuz.__x = 0x80;
    ASSERT_TRUE(hipblaslt_isnan(arg_hipblaslt_f8_fnuz));

    hipblaslt_bf8_fnuz arg_hipblaslt_bf8_fnuz;
    arg_hipblaslt_bf8_fnuz.__x = 0x80;
    ASSERT_TRUE(hipblaslt_isnan(arg_hipblaslt_bf8_fnuz));

    hipblaslt_f8 arg_hipblaslt_f8;
    arg_hipblaslt_f8.__x = 0x80;
    ASSERT_TRUE(hipblaslt_isnan(arg_hipblaslt_f8));

    hipblaslt_bf8 arg_hipblaslt_bf8;
    arg_hipblaslt_bf8.__x = 0x7d;
    ASSERT_TRUE(hipblaslt_isnan(arg_hipblaslt_bf8));

    hipblasLtHalf pos_inf = static_cast<hipblasLtHalf>(INFINITY); 
    ASSERT_TRUE(hipblaslt_isinf(pos_inf));

    hipblasLtHalf nan = static_cast<hipblasLtHalf>(NAN);
    ASSERT_TRUE(hipblaslt_isnan(nan));
}