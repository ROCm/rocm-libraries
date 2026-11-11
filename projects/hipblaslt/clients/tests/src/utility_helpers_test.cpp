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

#include <gtest/gtest.h>
#include <hipblaslt/hipblaslt.h>
#include <hip/hip_runtime.h>

// Test Suite 1: Matrix Layout Tests
// Targets: utility.hpp layout helper functions via public API
class MatrixLayoutTest : public ::testing::Test
{
protected:
    hipblasLtHandle_t handle;
    hipblasLtMatrixLayout_t layout;

    void SetUp() override
    {
        hipblasLtCreate(&handle);
    }

    void TearDown() override
    {
        hipblasLtDestroy(handle);
    }
};

TEST_F(MatrixLayoutTest, AllDataTypes)
{
    // Test all supported data types - exercises type handling in utility.hpp
    std::vector<hipDataType> types = {
        HIP_R_16F, HIP_R_32F, HIP_R_64F, HIP_R_16BF,
        HIP_R_8I, HIP_R_32I, HIP_C_32F, HIP_C_64F,
        HIP_R_8F_E4M3_FNUZ, HIP_R_8F_E5M2_FNUZ
    };

    for(auto type : types)
    {
        EXPECT_EQ(hipblasLtMatrixLayoutCreate(&layout, type, 128, 128, 128),
                  HIPBLAS_STATUS_SUCCESS);
        hipblasLtMatrixLayoutDestroy(layout);
    }
}

TEST_F(MatrixLayoutTest, BatchStridesAndOrder)
{
    // Create batched layout
    ASSERT_EQ(hipblasLtMatrixLayoutCreate(&layout, HIP_R_32F, 128, 128, 128),
              HIPBLAS_STATUS_SUCCESS);

    // Test batch count attribute
    int32_t batch_count = 4;
    EXPECT_EQ(hipblasLtMatrixLayoutSetAttribute(
                  layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                  &batch_count, sizeof(batch_count)),
              HIPBLAS_STATUS_SUCCESS);

    // Test batch stride attribute
    int64_t batch_stride = 128 * 128;
    EXPECT_EQ(hipblasLtMatrixLayoutSetAttribute(
                  layout, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                  &batch_stride, sizeof(batch_stride)),
              HIPBLAS_STATUS_SUCCESS);

    // Verify all attributes
    int32_t read_batch;
    int64_t read_stride;
    size_t sizeWritten;

    EXPECT_EQ(hipblasLtMatrixLayoutGetAttribute(
                  layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                  &read_batch, sizeof(read_batch), &sizeWritten),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(read_batch, batch_count);

    EXPECT_EQ(hipblasLtMatrixLayoutGetAttribute(
                  layout, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                  &read_stride, sizeof(read_stride), &sizeWritten),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(read_stride, batch_stride);

    hipblasLtMatrixLayoutDestroy(layout);
}

// Test Suite 2: Matmul Descriptor Tests
// Targets: utility.hpp descriptor handling
class MatmulDescriptorTest : public ::testing::Test
{
protected:
    hipblasLtMatmulDesc_t matmul;

    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(MatmulDescriptorTest, AllComputeTypes)
{
    // Test all compute types - exercises compute type conversion in utility.hpp
    ASSERT_EQ(hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F),
              HIPBLAS_STATUS_SUCCESS);
    hipblasLtMatmulDescDestroy(matmul);

    ASSERT_EQ(hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_64F, HIP_R_64F),
              HIPBLAS_STATUS_SUCCESS);
    hipblasLtMatmulDescDestroy(matmul);

    ASSERT_EQ(hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32I, HIP_R_32I),
              HIPBLAS_STATUS_SUCCESS);
    hipblasLtMatmulDescDestroy(matmul);
}

TEST_F(MatmulDescriptorTest, TransposeOperations)
{
    // Test transpose operations - exercises operation conversion in utility.hpp
    ASSERT_EQ(hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F),
              HIPBLAS_STATUS_SUCCESS);

    std::vector<hipblasOperation_t> ops = {HIPBLAS_OP_N, HIPBLAS_OP_T, HIPBLAS_OP_C};

    for(auto op : ops)
    {
        // Set transA
        EXPECT_EQ(hipblasLtMatmulDescSetAttribute(
                      matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &op, sizeof(op)),
                  HIPBLAS_STATUS_SUCCESS);

        // Set transB
        EXPECT_EQ(hipblasLtMatmulDescSetAttribute(
                      matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &op, sizeof(op)),
                  HIPBLAS_STATUS_SUCCESS);

        // Verify
        hipblasOperation_t readA, readB;
        size_t sizeWritten;
        EXPECT_EQ(hipblasLtMatmulDescGetAttribute(
                      matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &readA, sizeof(readA), &sizeWritten),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(readA, op);

        EXPECT_EQ(hipblasLtMatmulDescGetAttribute(
                      matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &readB, sizeof(readB), &sizeWritten),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(readB, op);
    }

    hipblasLtMatmulDescDestroy(matmul);
}

TEST_F(MatmulDescriptorTest, EpilogueAttributes)
{
    // Test epilogue attributes - exercises epilogue handling in utility.hpp
    ASSERT_EQ(hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F),
              HIPBLAS_STATUS_SUCCESS);

    // Test setting epilogue
    hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_RELU;
    EXPECT_EQ(hipblasLtMatmulDescSetAttribute(
                  matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)),
              HIPBLAS_STATUS_SUCCESS);

    // Verify
    hipblasLtEpilogue_t read_epilogue;
    size_t sizeWritten;
    EXPECT_EQ(hipblasLtMatmulDescGetAttribute(
                  matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &read_epilogue, sizeof(read_epilogue), &sizeWritten),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(read_epilogue, epilogue);

    // Test other epilogues
    epilogue = HIPBLASLT_EPILOGUE_GELU;
    EXPECT_EQ(hipblasLtMatmulDescSetAttribute(
                  matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)),
              HIPBLAS_STATUS_SUCCESS);

    hipblasLtMatmulDescDestroy(matmul);
}

// Test Suite 3: Data Validation
// Targets: utility.hpp size calculation and validation
class DataValidationTest : public ::testing::Test
{
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(DataValidationTest, DimensionValidation)
{
    hipblasLtMatrixLayout_t layout;

    // Test various matrix dimensions
    std::vector<uint64_t> sizes = {64, 128, 256, 512, 1024};

    for(auto size : sizes)
    {
        // Create layout with various dimensions - validates that creation succeeds
        EXPECT_EQ(hipblasLtMatrixLayoutCreate(&layout, HIP_R_16F, size, size, size),
                  HIPBLAS_STATUS_SUCCESS);

        // Note: ROWS, COLS, LD are set at creation time but cannot be queried via GetAttribute
        // Only BATCH_COUNT, STRIDED_BATCH_OFFSET, and BATCH_MODE are gettable

        hipblasLtMatrixLayoutDestroy(layout);
    }
}

TEST_F(DataValidationTest, NonSquareMatrices)
{
    hipblasLtMatrixLayout_t layout;

    // Test non-square matrices - validates creation with different M/N/LD values
    EXPECT_EQ(hipblasLtMatrixLayoutCreate(&layout, HIP_R_32F, 128, 256, 128),
              HIPBLAS_STATUS_SUCCESS);

    // Successfully creating the layout validates the dimensions
    // Note: ROWS, COLS cannot be queried via GetAttribute

    hipblasLtMatrixLayoutDestroy(layout);
}
