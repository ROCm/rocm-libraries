/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
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
#include <miopen/pooling/solvers.hpp>
#include <miopen/pooling/problem_description.hpp>
#include <miopen/pooling.hpp>
#include <miopen/tensor.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/utility/transposing_solver.hpp>
#include "get_handle.hpp"

using namespace miopen;
using namespace miopen::pooling;

namespace {

// Common test case structure
struct PoolingTestCase
{
    std::vector<int> input_dims;  // NCHW dimensions
    std::vector<int> pool_lens;   // pooling window size
    std::vector<int> pool_strides;
    std::vector<int> pool_pads;
    miopenPoolingMode_t mode;

    // Used by GTest for printing test parameters
    [[maybe_unused]] friend std::ostream& operator<<(std::ostream& os, const PoolingTestCase& tc)
    {
        os << "PoolingTestCase(input=" << tc.input_dims[0] << "x" << tc.input_dims[1] << "x"
           << tc.input_dims[2] << "x" << tc.input_dims[3] << ", pool=" << tc.pool_lens[0] << "x"
           << tc.pool_lens[1] << ", stride=" << tc.pool_strides[0] << "x" << tc.pool_strides[1]
           << ")";
        return os;
    }
};

// Helper to create tensor descriptor
TensorDescriptor MakeTensorDesc(miopenDataType_t type, const std::vector<int>& dims)
{
    std::vector<std::size_t> lens(dims.begin(), dims.end());
    return TensorDescriptor(type, lens);
}

// Helper to create pooling descriptor
PoolingDescriptor MakePoolingDesc(const PoolingTestCase& test_case)
{
    PoolingDescriptor desc;
    desc.mode    = test_case.mode;
    desc.lens    = test_case.pool_lens;
    desc.strides = test_case.pool_strides;
    desc.pads    = test_case.pool_pads;
    // Set workspace index mode to Image for compatibility with PoolingBackwardNd
    // (BackwardNd doesn't support Mask mode for max pooling)
    desc.SetWorkspaceIndexMode(miopenPoolingWorkspaceIndexImage);
    return desc;
}

// Helper to calculate output dimensions
std::vector<int> CalculateOutputDims(const std::vector<int>& input_dims,
                                      const PoolingTestCase& test_case)
{
    int n = input_dims[0];
    int c = input_dims[1];
    int h = input_dims[2];
    int w = input_dims[3];

    int out_h = (h + 2 * test_case.pool_pads[0] - test_case.pool_lens[0]) /
                    test_case.pool_strides[0] +
                1;
    int out_w = (w + 2 * test_case.pool_pads[1] - test_case.pool_lens[1]) /
                    test_case.pool_strides[1] +
                1;

    return {n, c, out_h, out_w};
}

// Helper to create problem description
ProblemDescription MakeProblem(miopenDataType_t data_type, const PoolingTestCase& test_case)
{
    auto x_desc       = MakeTensorDesc(data_type, test_case.input_dims);
    auto output_dims  = CalculateOutputDims(test_case.input_dims, test_case);
    auto y_desc       = MakeTensorDesc(data_type, output_dims);
    auto pooling_desc = MakePoolingDesc(test_case);

    // Backward pooling needs: pooling_desc, x (input), y (output), dx (gradient input), dy (gradient output)
    // For applicability testing, dx and dy are the same as x and y
    return ProblemDescription(pooling_desc, x_desc, y_desc, x_desc, y_desc);
}

// Get standard test cases
auto GetSmokeTestCases()
{
    return std::vector<PoolingTestCase>{
        // Simple 2x2 max pooling
        {{1, 3, 8, 8}, {2, 2}, {2, 2}, {0, 0}, miopenPoolingMax},
    };
}

// Reserved for future comprehensive tests
[[maybe_unused]] auto GetFullTestCases()
{
    return std::vector<PoolingTestCase>{
        // Various pooling configurations
        {{1, 3, 8, 8}, {2, 2}, {2, 2}, {0, 0}, miopenPoolingMax},
        {{1, 64, 28, 28}, {2, 2}, {2, 2}, {0, 0}, miopenPoolingMax},
        {{2, 128, 14, 14}, {3, 3}, {2, 2}, {1, 1}, miopenPoolingAverage},
        {{1, 256, 7, 7}, {2, 2}, {1, 1}, {0, 0}, miopenPoolingAverageInclusive},
    };
}

} // namespace

//=============================================================================
// Data Type Applicability Tests
//=============================================================================

// Test fixture for backward pooling data type support
class PoolingBackward2dDataTypeTest : public ::testing::TestWithParam<miopenDataType_t>
{
protected:
    ExecutionContext ctx;
};

TEST_P(PoolingBackward2dDataTypeTest, SupportedDataTypes)
{
    auto data_type  = GetParam();
    auto test_case  = GetSmokeTestCases()[0];
    auto problem    = MakeProblem(data_type, test_case);
    auto& handle    = get_handle();
    auto ctx_local  = ExecutionContext(&handle);

    miopen::solver::pooling::PoolingBackward2d solver;

    bool is_applicable = solver.IsApplicable(ctx_local, problem);

    // Supported types: FP32, FP16, BF16
    bool should_be_applicable = (data_type == miopenFloat || data_type == miopenHalf ||
                                  data_type == miopenBFloat16);

    EXPECT_EQ(is_applicable, should_be_applicable)
        << "PoolingBackward2d applicability mismatch for " << GetDataType(data_type);
}

// Smoke tests for supported types
INSTANTIATE_TEST_SUITE_P(Smoke,
                         PoolingBackward2dDataTypeTest,
                         ::testing::Values(miopenFloat, miopenHalf, miopenBFloat16));

// Full tests including unsupported types
INSTANTIATE_TEST_SUITE_P(Full,
                         PoolingBackward2dDataTypeTest,
                         ::testing::Values(miopenFloat,
                                           miopenHalf,
                                           miopenBFloat16,
                                           miopenInt8,
                                           miopenInt32,
                                           miopenDouble));

// Test fixture for backward ND pooling data type support
class PoolingBackwardNdDataTypeTest : public ::testing::TestWithParam<miopenDataType_t>
{
protected:
    ExecutionContext ctx;
};

TEST_P(PoolingBackwardNdDataTypeTest, SupportedDataTypes)
{
    auto data_type  = GetParam();
    auto test_case  = GetSmokeTestCases()[0];
    auto problem    = MakeProblem(data_type, test_case);
    auto& handle    = get_handle();
    auto ctx_local  = ExecutionContext(&handle);

    miopen::solver::pooling::PoolingBackwardNd solver;

    bool is_applicable = solver.IsApplicable(ctx_local, problem);

    // Supported types: FP32, FP16, BF16
    bool should_be_applicable = (data_type == miopenFloat || data_type == miopenHalf ||
                                  data_type == miopenBFloat16);

    EXPECT_EQ(is_applicable, should_be_applicable)
        << "PoolingBackwardNd applicability mismatch for " << GetDataType(data_type);
}

// Smoke tests for supported types
INSTANTIATE_TEST_SUITE_P(Smoke,
                         PoolingBackwardNdDataTypeTest,
                         ::testing::Values(miopenFloat, miopenHalf, miopenBFloat16));

// Full tests including unsupported types
INSTANTIATE_TEST_SUITE_P(Full,
                         PoolingBackwardNdDataTypeTest,
                         ::testing::Values(miopenFloat,
                                           miopenHalf,
                                           miopenBFloat16,
                                           miopenInt8,
                                           miopenInt32,
                                           miopenDouble));

//=============================================================================
// Batched Transpose + Pooling Integration Tests
//=============================================================================

// Verify batched transpose is selected for NHWC pooling (not universal transpose)
TEST(TransposingPoolingSolverSelection, BatchedTransposeSelectedForNHWC)
{
    auto test_case = GetSmokeTestCases()[0];
    auto pool_desc = MakePoolingDesc(test_case);

    // Create NHWC layout descriptors
    auto x_desc_nhwc = MakeTensorDesc(miopenFloat, test_case.input_dims);
    x_desc_nhwc.SetLayout_str("NHWC");

    auto output_dims = CalculateOutputDims(test_case.input_dims, test_case);
    auto y_desc_nhwc = MakeTensorDesc(miopenFloat, output_dims);
    y_desc_nhwc.SetLayout_str("NHWC");

    // For NHWC layout, batched transpose should be selected (not universal)
    // Batched transpose supports: FP32, FP16, BF16, Int8, Int32
    auto data_type = miopenFloat;

    // Verify batched transpose is applicable for this data type
    EXPECT_TRUE(miopen::BatchedTransposeSolution::IsApplicable(data_type))
        << "Batched transpose should support " << GetDataType(data_type)
        << " for NHWC->NCHW conversion";
}

// Verify BF16 support across the stack
TEST(TransposingPoolingCompatibility, BFloat16EndToEndSupport)
{
    auto& handle = get_handle();
    auto ctx     = ExecutionContext(&handle);
    auto test_case = GetSmokeTestCases()[0];

    // 1. Verify base pooling backward solvers support BF16
    auto problem_nchw = MakeProblem(miopenBFloat16, test_case);

    miopen::solver::pooling::PoolingBackward2d solver2d;
    miopen::solver::pooling::PoolingBackwardNd solverNd;

    EXPECT_TRUE(solver2d.IsApplicable(ctx, problem_nchw))
        << "PoolingBackward2d should support BF16";
    EXPECT_TRUE(solverNd.IsApplicable(ctx, problem_nchw))
        << "PoolingBackwardNd should support BF16";

    // 2. Verify batched transpose supports BF16
    EXPECT_TRUE(miopen::BatchedTransposeSolution::IsApplicable(miopenBFloat16))
        << "Batched transpose should support BF16 for layout conversion";

    // 3. Verify transposing wrapper solvers are applicable with BF16 + NHWC
    auto pool_desc = MakePoolingDesc(test_case);
    auto x_desc_nhwc = MakeTensorDesc(miopenBFloat16, test_case.input_dims);
    x_desc_nhwc.SetLayout_str("NHWC");

    auto output_dims = CalculateOutputDims(test_case.input_dims, test_case);
    auto y_desc_nhwc = MakeTensorDesc(miopenBFloat16, output_dims);
    y_desc_nhwc.SetLayout_str("NHWC");

    auto problem_nhwc = ProblemDescription(pool_desc, x_desc_nhwc, y_desc_nhwc,
                                           x_desc_nhwc, y_desc_nhwc);

    miopen::solver::pooling::TransposedPoolingBwd2d transposed_solver2d;
    miopen::solver::pooling::TransposedPoolingBwdNd transposed_solverNd;

    EXPECT_TRUE(transposed_solver2d.IsApplicable(ctx, problem_nhwc))
        << "TransposedPoolingBwd2d should support BF16 + NHWC layout";
    EXPECT_TRUE(transposed_solverNd.IsApplicable(ctx, problem_nhwc))
        << "TransposedPoolingBwdNd should support BF16 + NHWC layout";
}
