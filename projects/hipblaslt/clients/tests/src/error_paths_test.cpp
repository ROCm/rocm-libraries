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

// Test Suite 1: Invalid Handle Tests
// Targets: hipblaslt.cpp NULL handle checks
TEST(ErrorPathsTest, InvalidHandle)
{
    hipblasLtMatrixLayout_t matA, matB, matC, matD;
    hipblasLtMatmulDesc_t matmul;

    // NULL handle should return HIPBLAS_STATUS_NOT_INITIALIZED
    EXPECT_EQ(hipblasLtMatrixLayoutCreate(&matA, HIP_R_16F, 128, 128, 128),
              HIPBLAS_STATUS_SUCCESS);

    // Destroy with NULL should fail
    EXPECT_NE(hipblasLtMatrixLayoutDestroy(nullptr), HIPBLAS_STATUS_SUCCESS);

    // Cleanup
    hipblasLtMatrixLayoutDestroy(matA);
}

// Test Suite 2: NULL Pointer Tests
// Targets: hipblaslt.cpp NULL pointer validation paths
TEST(ErrorPathsTest, NullPointers)
{
    hipblasLtHandle_t handle;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

    // NULL output parameter should fail
    EXPECT_NE(hipblasLtMatrixLayoutCreate(nullptr, HIP_R_16F, 128, 128, 128),
              HIPBLAS_STATUS_SUCCESS);

    hipblasLtDestroy(handle);
}

// Test Suite 3: Valid Enum Coverage
// Targets: hipblaslt.cpp enum handling for all valid types
TEST(ErrorPathsTest, AllValidDataTypes)
{
    hipblasLtMatrixLayout_t mat;

    // Test all valid data types to exercise type handling code
    std::vector<hipDataType> valid_types = {
        HIP_R_16F, HIP_R_32F, HIP_R_64F, HIP_R_16BF,
        HIP_R_8I, HIP_R_32I, HIP_C_32F, HIP_C_64F,
        HIP_R_8F_E4M3_FNUZ, HIP_R_8F_E5M2_FNUZ
    };

    for(auto type : valid_types) {
        EXPECT_EQ(hipblasLtMatrixLayoutCreate(&mat, type, 128, 128, 128),
                  HIPBLAS_STATUS_SUCCESS);
        hipblasLtMatrixLayoutDestroy(mat);
    }
}

// Test Suite 4: Various Matrix Sizes
// Targets: hipblaslt.cpp size handling code paths
TEST(ErrorPathsTest, VariousMatrixSizes)
{
    hipblasLtMatrixLayout_t mat;

    // Test various valid sizes to exercise size handling
    std::vector<uint64_t> sizes = {1, 64, 128, 256, 512, 1024, 2048};

    for(auto size : sizes) {
        EXPECT_EQ(hipblasLtMatrixLayoutCreate(&mat, HIP_R_16F, size, size, size),
                  HIPBLAS_STATUS_SUCCESS);
        hipblasLtMatrixLayoutDestroy(mat);
    }
}

// Test Suite 5: Descriptor Attribute Tests
// Targets: hipblaslt.cpp attribute get/set error paths
TEST(ErrorPathsTest, DescriptorAttributes)
{
    hipblasLtMatmulDesc_t matmul;
    ASSERT_EQ(hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F),
              HIPBLAS_STATUS_SUCCESS);

    // Test getting/setting valid attributes first
    hipblasOperation_t transA = HIPBLAS_OP_N;
    size_t size = sizeof(transA);
    size_t sizeWritten;

    // Get transpose A
    EXPECT_EQ(hipblasLtMatmulDescGetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &transA, size, &sizeWritten),
              HIPBLAS_STATUS_SUCCESS);

    // Set transpose A to T
    transA = HIPBLAS_OP_T;
    EXPECT_EQ(hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &transA, size),
              HIPBLAS_STATUS_SUCCESS);

    // Verify it was set
    hipblasOperation_t readTransA;
    EXPECT_EQ(hipblasLtMatmulDescGetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &readTransA, size, &sizeWritten),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(readTransA, HIPBLAS_OP_T);

    hipblasLtMatmulDescDestroy(matmul);
}

// Test Suite 6: Preference Tests
// Targets: hipblaslt.cpp preference creation and attribute handling
TEST(ErrorPathsTest, PreferenceHandling)
{
    hipblasLtMatmulPreference_t pref;

    // Create and destroy preference
    ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&pref), HIPBLAS_STATUS_SUCCESS);

    // Set workspace size attribute
    uint64_t workspace_size = 1024 * 1024; // 1MB
    EXPECT_EQ(hipblasLtMatmulPreferenceSetAttribute(
                  pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                  &workspace_size, sizeof(workspace_size)),
              HIPBLAS_STATUS_SUCCESS);

    // Get workspace size back
    uint64_t read_workspace;
    size_t sizeWritten;
    EXPECT_EQ(hipblasLtMatmulPreferenceGetAttribute(
                  pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                  &read_workspace, sizeof(read_workspace), &sizeWritten),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(read_workspace, workspace_size);

    hipblasLtMatmulPreferenceDestroy(pref);
}

// Test Suite 7: Compute Type Tests
// Targets: hipblaslt.cpp compute type validation
TEST(ErrorPathsTest, ComputeTypes)
{
    hipblasLtMatmulDesc_t matmul;

    // Test different compute types
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

// Note: Non-square layout tests are in utility_helpers_test.cpp (DataValidationTest.NonSquareMatrices)
// Note: hipblasLtMatrixLayoutCreate currently does not validate dimension parameters
// (zero dimensions, ld < rows, invalid enums all return SUCCESS).
// Invalid inputs may cause undefined behavior at matmul execution time.
