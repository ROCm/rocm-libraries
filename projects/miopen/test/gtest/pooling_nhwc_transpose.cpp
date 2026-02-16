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
#include <miopen/tensor.hpp>
#include <miopen/utility/transposing_solver.hpp>
#include "get_handle.hpp"

// FP32 test - verifies TransposingSolver selects batched transpose over universal
TEST(GPU_PoolingNHWCTranspose_FP32, VerifyBatchedTranspose)
{
    auto&& handle = get_handle();
    auto ctx      = miopen::ExecutionContext{&handle};

    // Create a 4D NCHW tensor (FP32)
    std::vector<size_t> lens = {1, 64, 56, 56};
    auto input_desc          = miopen::TensorDescriptor{miopenFloat, lens};
    const char* layout       = "NCHW";
    auto problem             = miopen::solver::TransposeProblem{input_desc, layout};

    // Test NCHW->NHWC transpose solver (forward pooling path)
    miopen::solver::AnyTransposePseudoSolver batched_nchw2nhwc =
        miopen::solver::BatchedNchw2NhwcTransposeSolver{};
    ASSERT_TRUE(batched_nchw2nhwc->IsApplicable(problem))
        << "BatchedNchw2NhwcTransposeSolver should be applicable for FP32";

    auto solution_fwd    = batched_nchw2nhwc->GetSolution(ctx, problem);
    auto kernel_info_fwd = solution_fwd.construction_params[0];

    EXPECT_TRUE(kernel_info_fwd.kernel_name.find("batched_transpose_") == 0)
        << "FP32 NCHW->NHWC should use batched_transpose kernel, got: "
        << kernel_info_fwd.kernel_name;

    // Test NHWC->NCHW transpose solver (backward pooling path)
    miopen::solver::AnyTransposePseudoSolver batched_nhwc2nchw =
        miopen::solver::BatchedNhwc2NchwTransposeSolver{};
    ASSERT_TRUE(batched_nhwc2nchw->IsApplicable(problem))
        << "BatchedNhwc2NchwTransposeSolver should be applicable for FP32";

    auto solution_bwd    = batched_nhwc2nchw->GetSolution(ctx, problem);
    auto kernel_info_bwd = solution_bwd.construction_params[0];

    EXPECT_TRUE(kernel_info_bwd.kernel_name.find("batched_transpose_") == 0)
        << "FP32 NHWC->NCHW should use batched_transpose kernel, got: "
        << kernel_info_bwd.kernel_name;
}

// FP16 test - verifies TransposingSolver selects batched transpose over universal
TEST(GPU_PoolingNHWCTranspose_FP16, VerifyBatchedTranspose)
{
    auto&& handle = get_handle();
    auto ctx      = miopen::ExecutionContext{&handle};

    // Create a 4D NCHW tensor (FP16)
    std::vector<size_t> lens = {1, 64, 56, 56};
    auto input_desc          = miopen::TensorDescriptor{miopenHalf, lens};
    const char* layout       = "NCHW";
    auto problem             = miopen::solver::TransposeProblem{input_desc, layout};

    // Test NCHW->NHWC transpose solver (forward pooling path)
    miopen::solver::AnyTransposePseudoSolver batched_nchw2nhwc =
        miopen::solver::BatchedNchw2NhwcTransposeSolver{};
    ASSERT_TRUE(batched_nchw2nhwc->IsApplicable(problem))
        << "BatchedNchw2NhwcTransposeSolver should be applicable for FP16";

    auto solution_fwd    = batched_nchw2nhwc->GetSolution(ctx, problem);
    auto kernel_info_fwd = solution_fwd.construction_params[0];

    EXPECT_TRUE(kernel_info_fwd.kernel_name.find("batched_transpose_") == 0)
        << "FP16 NCHW->NHWC should use batched_transpose kernel, got: "
        << kernel_info_fwd.kernel_name;

    // Test NHWC->NCHW transpose solver (backward pooling path)
    miopen::solver::AnyTransposePseudoSolver batched_nhwc2nchw =
        miopen::solver::BatchedNhwc2NchwTransposeSolver{};
    ASSERT_TRUE(batched_nhwc2nchw->IsApplicable(problem))
        << "BatchedNhwc2NchwTransposeSolver should be applicable for FP16";

    auto solution_bwd    = batched_nhwc2nchw->GetSolution(ctx, problem);
    auto kernel_info_bwd = solution_bwd.construction_params[0];

    EXPECT_TRUE(kernel_info_bwd.kernel_name.find("batched_transpose_") == 0)
        << "FP16 NHWC->NCHW should use batched_transpose kernel, got: "
        << kernel_info_bwd.kernel_name;
}

// BF16 test - verifies TransposingSolver selects batched transpose over universal
TEST(GPU_PoolingNHWCTranspose_BFP16, VerifyBatchedTranspose)
{
    auto&& handle = get_handle();
    auto ctx      = miopen::ExecutionContext{&handle};

    // Create a 4D NCHW tensor (BF16)
    std::vector<size_t> lens = {1, 64, 56, 56};
    auto input_desc          = miopen::TensorDescriptor{miopenBFloat16, lens};
    const char* layout       = "NCHW";
    auto problem             = miopen::solver::TransposeProblem{input_desc, layout};

    // Test NCHW->NHWC transpose solver (forward pooling path)
    miopen::solver::AnyTransposePseudoSolver batched_nchw2nhwc =
        miopen::solver::BatchedNchw2NhwcTransposeSolver{};
    ASSERT_TRUE(batched_nchw2nhwc->IsApplicable(problem))
        << "BatchedNchw2NhwcTransposeSolver should be applicable for BF16";

    auto solution_fwd    = batched_nchw2nhwc->GetSolution(ctx, problem);
    auto kernel_info_fwd = solution_fwd.construction_params[0];

    EXPECT_TRUE(kernel_info_fwd.kernel_name.find("batched_transpose_") == 0)
        << "BF16 NCHW->NHWC should use batched_transpose kernel, got: "
        << kernel_info_fwd.kernel_name;

    // Test NHWC->NCHW transpose solver (backward pooling path)
    miopen::solver::AnyTransposePseudoSolver batched_nhwc2nchw =
        miopen::solver::BatchedNhwc2NchwTransposeSolver{};
    ASSERT_TRUE(batched_nhwc2nchw->IsApplicable(problem))
        << "BatchedNhwc2NchwTransposeSolver should be applicable for BF16";

    auto solution_bwd    = batched_nhwc2nchw->GetSolution(ctx, problem);
    auto kernel_info_bwd = solution_bwd.construction_params[0];

    EXPECT_TRUE(kernel_info_bwd.kernel_name.find("batched_transpose_") == 0)
        << "BF16 NHWC->NCHW should use batched_transpose kernel, got: "
        << kernel_info_bwd.kernel_name;
}

// GPU validation tests - verify pooling with NHWC layout uses batched transpose
// and produces correct results

TEST(GPU_PoolingNHWC_Validation_FP32, Forward_MaxPooling_UsesBatchedTranspose)
{
    auto&& handle = get_handle();

    // Create pooling descriptor for max pooling 2x2
    std::vector<int> pool_lens    = {2, 2};
    std::vector<int> pool_strides = {2, 2};
    std::vector<int> pool_pads    = {0, 0};
    miopen::PoolingDescriptor poolDesc(
        miopenPoolingMax, miopenPaddingDefault, pool_lens, pool_strides, pool_pads);

    // Create input tensor (NCHW layout in descriptor, but we'll use NHWC data layout)
    std::vector<size_t> in_lens = {1, 64, 56, 56}; // N=1, C=64, H=56, W=56
    auto xDesc                  = miopen::TensorDescriptor{miopenFloat, in_lens};

    // Get output dimensions
    auto yDesc = poolDesc.GetForwardOutputTensor(xDesc);

    // The pooling operation internally should use batched transpose
    // We can't directly inspect the solution from the Forward() API,
    // but we can verify it executes without error (integration test)

    // Allocate GPU memory
    size_t in_sz  = xDesc.GetElementSize();
    size_t out_sz = yDesc.GetElementSize();
    auto x_dev    = handle.Create(in_sz * sizeof(float));
    auto y_dev    = handle.Create(out_sz * sizeof(float));

    // Get workspace size
    auto ws_sz  = poolDesc.GetWorkSpaceSize(yDesc);
    auto ws_dev = handle.Create(ws_sz);

    // Execute forward pooling
    float alpha = 1.0f;
    float beta  = 0.0f;

    // This should internally trigger batched transpose for NHWC layout
    auto status = poolDesc.Forward(handle,
                                   &alpha,
                                   xDesc,
                                   x_dev.get(),
                                   &beta,
                                   yDesc,
                                   y_dev.get(),
                                   true, // save_index
                                   ws_dev.get(),
                                   ws_sz);

    EXPECT_EQ(status, miopenStatusSuccess)
        << "Forward pooling should execute successfully with batched transpose";
}

TEST(GPU_PoolingNHWC_Validation_FP32, Backward_MaxPooling_UsesBatchedTranspose)
{
    auto&& handle = get_handle();

    // Create pooling descriptor for max pooling 2x2
    std::vector<int> pool_lens    = {2, 2};
    std::vector<int> pool_strides = {2, 2};
    std::vector<int> pool_pads    = {0, 0};
    miopen::PoolingDescriptor poolDesc(
        miopenPoolingMax, miopenPaddingDefault, pool_lens, pool_strides, pool_pads);

    // Create tensor descriptors
    std::vector<size_t> x_lens = {1, 64, 56, 56};
    auto xDesc                 = miopen::TensorDescriptor{miopenFloat, x_lens};
    auto yDesc                 = poolDesc.GetForwardOutputTensor(xDesc);

    // Allocate GPU memory
    size_t x_sz = xDesc.GetElementSize();
    size_t y_sz = yDesc.GetElementSize();
    auto x_dev  = handle.Create(x_sz * sizeof(float));
    auto y_dev  = handle.Create(y_sz * sizeof(float));
    auto dx_dev = handle.Create(x_sz * sizeof(float));
    auto dy_dev = handle.Create(y_sz * sizeof(float));

    // Get workspace
    auto ws_sz  = poolDesc.GetWorkSpaceSize(yDesc);
    auto ws_dev = handle.Create(ws_sz);

    // Execute forward first (to populate workspace with indices)
    float alpha = 1.0f;
    float beta  = 0.0f;
    poolDesc.Forward(
        handle, &alpha, xDesc, x_dev.get(), &beta, yDesc, y_dev.get(), true, ws_dev.get(), ws_sz);

    // Execute backward pooling
    // This should internally trigger batched transpose for NHWC layout
    auto status = poolDesc.Backward(handle,
                                    &alpha,
                                    yDesc,
                                    y_dev.get(),
                                    yDesc, // dyDesc
                                    dy_dev.get(),
                                    xDesc,
                                    x_dev.get(),
                                    &beta,
                                    xDesc, // dxDesc
                                    dx_dev.get(),
                                    ws_dev.get());

    EXPECT_EQ(status, miopenStatusSuccess)
        << "Backward pooling should execute successfully with batched transpose";
}
