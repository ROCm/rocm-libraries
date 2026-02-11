/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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
#include <miopen/batched_transpose_sol.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/miopen.h>
#include "get_handle.hpp"

// Test that verifies batched transpose is used for NHWC pooling forward operations
TEST(PoolingNHWCTranspose, VerifyBatchedTransposeForwardFP32)
{
    auto&& handle = get_handle();
    auto ctx      = miopen::ExecutionContext{&handle};

    // Test parameters for NHWC pooling (typical ResNet dimensions)
    uint32_t n = 1, c = 64, h = 56, w = 56;

    // Test NCHW->NHWC transpose (input transpose for NHWC pooling)
    auto transpose_sol = miopen::TransposeSolutionDefault2Nhwc(ctx, miopenFloat, n, c, h, w);
    auto kernel_info   = transpose_sol.GetKernelInfo();

    ASSERT_FALSE(kernel_info.kernel_name.empty()) << "Kernel name should not be empty for FP32";

    EXPECT_TRUE(kernel_info.kernel_name.find("batched_transpose_") == 0)
        << "NHWC pooling forward (FP32) should use batched_transpose kernel, got: "
        << kernel_info.kernel_name;
}

TEST(PoolingNHWCTranspose, VerifyBatchedTransposeForwardFP16)
{
    auto&& handle = get_handle();
    auto ctx      = miopen::ExecutionContext{&handle};

    uint32_t n = 1, c = 128, h = 28, w = 28;

    auto transpose_sol = miopen::TransposeSolutionDefault2Nhwc(ctx, miopenHalf, n, c, h, w);
    auto kernel_info   = transpose_sol.GetKernelInfo();

    ASSERT_FALSE(kernel_info.kernel_name.empty()) << "Kernel name should not be empty for FP16";

    EXPECT_TRUE(kernel_info.kernel_name.find("batched_transpose_") == 0)
        << "NHWC pooling forward (FP16) should use batched_transpose kernel, got: "
        << kernel_info.kernel_name;
}

TEST(PoolingNHWCTranspose, VerifyBatchedTransposeForwardBF16)
{
    auto&& handle = get_handle();
    auto ctx      = miopen::ExecutionContext{&handle};

    uint32_t n = 2, c = 256, h = 14, w = 14;

    auto transpose_sol = miopen::TransposeSolutionDefault2Nhwc(ctx, miopenBFloat16, n, c, h, w);
    auto kernel_info   = transpose_sol.GetKernelInfo();

    ASSERT_FALSE(kernel_info.kernel_name.empty()) << "Kernel name should not be empty for BF16";

    EXPECT_TRUE(kernel_info.kernel_name.find("batched_transpose_") == 0)
        << "NHWC pooling forward (BF16) should use batched_transpose kernel, got: "
        << kernel_info.kernel_name;
}

// Test that verifies batched transpose is used for NHWC pooling backward operations
TEST(PoolingNHWCTranspose, VerifyBatchedTransposeBackwardFP32)
{
    auto&& handle = get_handle();
    auto ctx      = miopen::ExecutionContext{&handle};

    // Test parameters for NHWC pooling backward (typical ResNet dimensions)
    uint32_t n = 1, c = 64, h = 56, w = 56;

    // Test NHWC->NCHW transpose (output transpose for NHWC pooling backward)
    auto transpose_sol = miopen::TransposeSolutionNhwc2Default(ctx, miopenFloat, n, c, h, w);
    auto kernel_info   = transpose_sol.GetKernelInfo();

    ASSERT_FALSE(kernel_info.kernel_name.empty())
        << "Kernel name should not be empty for backward FP32";

    EXPECT_TRUE(kernel_info.kernel_name.find("batched_transpose_") == 0)
        << "NHWC pooling backward (FP32) should use batched_transpose kernel, got: "
        << kernel_info.kernel_name;
}

TEST(PoolingNHWCTranspose, VerifyBatchedTransposeBackwardFP16)
{
    auto&& handle = get_handle();
    auto ctx      = miopen::ExecutionContext{&handle};

    uint32_t n = 1, c = 128, h = 28, w = 28;

    auto transpose_sol = miopen::TransposeSolutionNhwc2Default(ctx, miopenHalf, n, c, h, w);
    auto kernel_info   = transpose_sol.GetKernelInfo();

    ASSERT_FALSE(kernel_info.kernel_name.empty())
        << "Kernel name should not be empty for backward FP16";

    EXPECT_TRUE(kernel_info.kernel_name.find("batched_transpose_") == 0)
        << "NHWC pooling backward (FP16) should use batched_transpose kernel, got: "
        << kernel_info.kernel_name;
}

TEST(PoolingNHWCTranspose, VerifyBatchedTransposeBackwardBF16)
{
    auto&& handle = get_handle();
    auto ctx      = miopen::ExecutionContext{&handle};

    uint32_t n = 2, c = 256, h = 14, w = 14;

    auto transpose_sol = miopen::TransposeSolutionNhwc2Default(ctx, miopenBFloat16, n, c, h, w);
    auto kernel_info   = transpose_sol.GetKernelInfo();

    ASSERT_FALSE(kernel_info.kernel_name.empty())
        << "Kernel name should not be empty for backward BF16";

    EXPECT_TRUE(kernel_info.kernel_name.find("batched_transpose_") == 0)
        << "NHWC pooling backward (BF16) should use batched_transpose kernel, got: "
        << kernel_info.kernel_name;
}

// Combined test for multiple data types
TEST(PoolingNHWCTranspose, VerifyBatchedTransposeMultipleTypes)
{
    auto&& handle = get_handle();
    auto ctx      = miopen::ExecutionContext{&handle};

    // Test parameters: typical pooling dimensions
    uint32_t n = 1, c = 64, h = 56, w = 56;

    auto verify_forward = [&](miopenDataType_t data_type, const char* type_name) {
        auto transpose_sol = miopen::TransposeSolutionDefault2Nhwc(ctx, data_type, n, c, h, w);
        auto kernel_info   = transpose_sol.GetKernelInfo();

        ASSERT_FALSE(kernel_info.kernel_name.empty())
            << "Kernel name should not be empty for forward " << type_name;

        EXPECT_TRUE(kernel_info.kernel_name.find("batched_transpose_") == 0)
            << type_name << " forward pooling should use batched_transpose kernel, got: "
            << kernel_info.kernel_name;
    };

    auto verify_backward = [&](miopenDataType_t data_type, const char* type_name) {
        auto transpose_sol = miopen::TransposeSolutionNhwc2Default(ctx, data_type, n, c, h, w);
        auto kernel_info   = transpose_sol.GetKernelInfo();

        ASSERT_FALSE(kernel_info.kernel_name.empty())
            << "Kernel name should not be empty for backward " << type_name;

        EXPECT_TRUE(kernel_info.kernel_name.find("batched_transpose_") == 0)
            << type_name << " backward pooling should use batched_transpose kernel, got: "
            << kernel_info.kernel_name;
    };

    // Verify forward transposes (NCHW->NHWC)
    verify_forward(miopenFloat, "FP32");
    verify_forward(miopenHalf, "FP16");
    verify_forward(miopenBFloat16, "BF16");

    // Verify backward transposes (NHWC->NCHW)
    verify_backward(miopenFloat, "FP32");
    verify_backward(miopenHalf, "FP16");
    verify_backward(miopenBFloat16, "BF16");
}
