// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// API-level applicability sweep for 2D grouped CK xdlops solvers on shapes
// whose total element-strides bracket / exceed INT_MAX. Shapes were chosen to
// bracket the 2^31 element-stride boundary for x = (1, 96, H, W) with
// weight (32, 96, 3, 3).
//
// These tests verify that MIOpen and CK find applicable kernels for the given
// shapes in order to test a wide range of shapes quickly.

#include "conv_api_solution_count_large_stride_common.hpp"
#include <vector>

namespace {

using miopen_test_large_stride::Descriptors;
using miopen_test_large_stride::RunCompileBwdData;
using miopen_test_large_stride::RunCompileFwd;
using miopen_test_large_stride::RunCompileWrw;
using miopen_test_large_stride::SetupDescriptorsImpl;

struct Shape2D
{
    int n, c, h, w;
};

// Shapes bracketing the INT_MAX element-stride boundary for x = (1, 96, H, W).
// Element count = 96 * H * W; INT_MAX ≈ 2.147 GB.
std::vector<Shape2D> ReproducerShapes()
{
    return {
        {1, 96, 4096, 4096},   // 1.61 GB (just below INT_MAX)
        {1, 96, 4608, 4608},   // 2.04 GB (just below INT_MAX)
        {1, 96, 4736, 4736},   // 2.15 GB (just above INT_MAX)
        {1, 96, 5120, 5120},   // 2.52 GB
        {1, 96, 5632, 5632},   // 3.05 GB
        {1, 96, 6144, 6144},   // 3.62 GB
        {1, 96, 8192, 8192},   // 6.44 GB
        {1, 96, 9216, 9216},   // 8.16 GB
        {1, 96, 10240, 10240}, // 10.07 GB
        {1, 96, 11264, 11264}, // 12.18 GB
        {1, 96, 12288, 12288}, // 14.50 GB
        {1, 96, 14336, 14336}, // 19.73 GB
        {1, 96, 16384, 16384}, // 25.77 GB
        {2, 96, 4096, 4096},   // 3.22 GB (smallest applicable BwdData >INT_MAX)
        {1, 96, 8192, 4096},   // 3.22 GB (non-square)
        {1, 96, 4096, 8192},   // 3.22 GB (non-square)
    };
}

::testing::AssertionResult
SetupDescriptors2D(const Shape2D& s, miopenDataType_t dtype, Descriptors& d)
{
    const int x_dims[4] = {s.n, s.c, s.h, s.w};
    const int w_dims[4] = {32, 96, 3, 3};
    return SetupDescriptorsImpl<2>(x_dims, w_dims, dtype, d);
}

void RunFwd(const Shape2D& s, miopenDataType_t dtype)
{
    RunCompileFwd(s, dtype, SetupDescriptors2D, "ConvHipImplicitGemmGroupFwdXdlops");
}
void RunBwdData(const Shape2D& s, miopenDataType_t dtype)
{
    RunCompileBwdData(s, dtype, SetupDescriptors2D, "ConvHipImplicitGemmGroupBwdXdlops");
}
void RunWrw(const Shape2D& s, miopenDataType_t dtype)
{
    RunCompileWrw(s, dtype, SetupDescriptors2D, "ConvHipImplicitGemmGroupWrwXdlops");
}

class GPU_ConvApi_SolutionCount2DLargeStride_FP16 : public ::testing::TestWithParam<Shape2D>
{
};
class GPU_ConvApi_SolutionCount2DLargeStride_FP32 : public ::testing::TestWithParam<Shape2D>
{
};
class GPU_ConvApi_SolutionCount2DLargeStride_BFP16 : public ::testing::TestWithParam<Shape2D>
{
};

} // namespace

TEST_P(GPU_ConvApi_SolutionCount2DLargeStride_FP16, FwdNonZeroAndIncludesCK)
{
    RunFwd(GetParam(), miopenHalf);
}
TEST_P(GPU_ConvApi_SolutionCount2DLargeStride_FP16, BwdDataNonZeroAndIncludesCK)
{
    RunBwdData(GetParam(), miopenHalf);
}
TEST_P(GPU_ConvApi_SolutionCount2DLargeStride_FP16, WrwNonZeroAndIncludesCK)
{
    RunWrw(GetParam(), miopenHalf);
}

TEST_P(GPU_ConvApi_SolutionCount2DLargeStride_FP32, FwdNonZeroAndIncludesCK)
{
    RunFwd(GetParam(), miopenFloat);
}
TEST_P(GPU_ConvApi_SolutionCount2DLargeStride_FP32, BwdDataNonZeroAndIncludesCK)
{
    RunBwdData(GetParam(), miopenFloat);
}
TEST_P(GPU_ConvApi_SolutionCount2DLargeStride_FP32, WrwNonZeroAndIncludesCK)
{
    RunWrw(GetParam(), miopenFloat);
}

TEST_P(GPU_ConvApi_SolutionCount2DLargeStride_BFP16, FwdNonZeroAndIncludesCK)
{
    RunFwd(GetParam(), miopenBFloat16);
}
TEST_P(GPU_ConvApi_SolutionCount2DLargeStride_BFP16, BwdDataNonZeroAndIncludesCK)
{
    RunBwdData(GetParam(), miopenBFloat16);
}
TEST_P(GPU_ConvApi_SolutionCount2DLargeStride_BFP16, WrwNonZeroAndIncludesCK)
{
    RunWrw(GetParam(), miopenBFloat16);
}

INSTANTIATE_TEST_SUITE_P(Standard,
                         GPU_ConvApi_SolutionCount2DLargeStride_FP16,
                         ::testing::ValuesIn(ReproducerShapes()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         GPU_ConvApi_SolutionCount2DLargeStride_FP32,
                         ::testing::ValuesIn(ReproducerShapes()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         GPU_ConvApi_SolutionCount2DLargeStride_BFP16,
                         ::testing::ValuesIn(ReproducerShapes()));
