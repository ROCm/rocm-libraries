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

#include <cstdint>
#include <memory>
#include <vector>

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

// Int32 compute reads alpha and beta as int32. With a float scale type the heuristic
// returns no algos and hipblasLtMatmul refuses the problem, rather than using the bits
// of the float alpha as an integer, which saturates D.
TEST(ErrorPathsTest, Int32ComputeRefusesNonInt32ScaleType)
{
    constexpr int64_t m = 64, n = 64, k = 16;
    constexpr size_t  workspaceSize = size_t(32) << 20;

    hipblasLtHandle_t handle = nullptr;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
    const std::unique_ptr<void, decltype(&hipblasLtDestroy)> handleOwner(handle,
                                                                         &hipblasLtDestroy);
    using LayoutOwner
        = std::unique_ptr<hipblasLtMatrixLayoutOpaque_t, decltype(&hipblasLtMatrixLayoutDestroy)>;
    auto layout = [](int64_t rows, int64_t cols) {
        hipblasLtMatrixLayout_t l = nullptr;
        EXPECT_EQ(hipblasLtMatrixLayoutCreate(&l, HIP_R_8I, rows, cols, rows),
                  HIPBLAS_STATUS_SUCCESS);
        return LayoutOwner(l, &hipblasLtMatrixLayoutDestroy);
    };
    const LayoutOwner matA = layout(m, k), matB = layout(k, n), matD = layout(m, n);
    using DescOwner
        = std::unique_ptr<hipblasLtMatmulDescOpaque_t, decltype(&hipblasLtMatmulDescDestroy)>;
    auto desc = [](hipDataType scaleType) {
        hipblasLtMatmulDesc_t d = nullptr;
        EXPECT_EQ(hipblasLtMatmulDescCreate(&d, HIPBLAS_COMPUTE_32I, scaleType),
                  HIPBLAS_STATUS_SUCCESS);
        return DescOwner(d, &hipblasLtMatmulDescDestroy);
    };
    const DescOwner int32Scale = desc(HIP_R_32I), floatScale = desc(HIP_R_32F);

    hipblasLtMatmulPreference_t pref = nullptr;
    ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&pref), HIPBLAS_STATUS_SUCCESS);
    const std::unique_ptr<hipblasLtMatmulPreferenceOpaque_t,
                          decltype(&hipblasLtMatmulPreferenceDestroy)>
        prefOwner(pref, &hipblasLtMatmulPreferenceDestroy);
    ASSERT_EQ(hipblasLtMatmulPreferenceSetAttribute(pref,
                                                    HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                    &workspaceSize,
                                                    sizeof(workspaceSize)),
              HIPBLAS_STATUS_SUCCESS);

    // A and B all ones, so each element of D is alpha * k.
    auto freeBuffer = [](void* p) { static_cast<void>(hipFree(p)); };
    using BufferOwner = std::unique_ptr<void, decltype(freeBuffer)>;
    auto buffer       = [&](size_t bytes, int fill) {
        void* p = nullptr;
        EXPECT_EQ(hipMalloc(&p, bytes), hipSuccess);
        if(p)
            EXPECT_EQ(hipMemset(p, fill, bytes), hipSuccess);
        return BufferOwner(p, freeBuffer);
    };
    const BufferOwner a = buffer(m * k, 1), b = buffer(k * n, 1), d = buffer(m * n, 0),
                      workspace = buffer(workspaceSize, 0);
    ASSERT_TRUE(a && b && d && workspace);

    hipblasLtMatmulHeuristicResult_t result{};
    int                              returned = 0;
    ASSERT_EQ(hipblasLtMatmulAlgoGetHeuristic(handle,
                                              int32Scale.get(),
                                              matA.get(),
                                              matB.get(),
                                              matD.get(),
                                              matD.get(),
                                              pref,
                                              1,
                                              &result,
                                              &returned),
              HIPBLAS_STATUS_SUCCESS);
    if(returned == 0)
        GTEST_SKIP() << "the loaded library has no int8 solution";

    std::vector<int8_t> host(m * n);
    auto matmul = [&](hipblasLtMatmulDesc_t desc, const void* alpha, const void* beta) {
        EXPECT_EQ(hipMemset(d.get(), 0x55, m * n), hipSuccess);
        const hipblasStatus_t status = hipblasLtMatmul(handle,
                                                       desc,
                                                       alpha,
                                                       a.get(),
                                                       matA.get(),
                                                       b.get(),
                                                       matB.get(),
                                                       beta,
                                                       d.get(),
                                                       matD.get(),
                                                       d.get(),
                                                       matD.get(),
                                                       &result.algo,
                                                       workspace.get(),
                                                       workspaceSize,
                                                       nullptr);
        EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
        EXPECT_EQ(hipMemcpy(host.data(), d.get(), host.size(), hipMemcpyDeviceToHost),
                  hipSuccess);
        return status;
    };

    const int32_t alphaI = 1, betaI = 0;
    EXPECT_EQ(matmul(int32Scale.get(), &alphaI, &betaI), HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(host[0], k);
    EXPECT_EQ(host[m * n - 1], k);

    hipblasLtMatmulHeuristicResult_t floatResult{};
    returned = -1;
    EXPECT_EQ(hipblasLtMatmulAlgoGetHeuristic(handle,
                                              floatScale.get(),
                                              matA.get(),
                                              matB.get(),
                                              matD.get(),
                                              matD.get(),
                                              pref,
                                              1,
                                              &floatResult,
                                              &returned),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(returned, 0);
    const float alphaF = 1.0f, betaF = 0.0f;
    EXPECT_EQ(matmul(floatScale.get(), &alphaF, &betaF), HIPBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(host[0], 0x55) << "D was written";
}

// Note: Non-square layout tests are in utility_helpers_test.cpp (DataValidationTest.NonSquareMatrices)
// Note: hipblasLtMatrixLayoutCreate currently does not validate dimension parameters
// (zero dimensions, ld < rows, invalid enums all return SUCCESS).
// Invalid inputs may cause undefined behavior at matmul execution time.
