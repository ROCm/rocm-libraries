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

// Known CK applicability gaps for 2D large-stride shapes (gfx942).
// Two failure modes are merged into these lists:
//
//   * Sub-INT_MAX shapes where CK has a tile-selection miss for a particular
//     dtype/shape (pre-existing CK gap, unrelated to large-tensor work).
//   * Strides >INT_MAX shapes where the filter (correctly) restricts CK
//     selection to *_Large_Tensor instances. For Fwd, registered Fwd
//     large-tensor instances cover most shapes but miss the very largest
//     (16384x16384) across all dtypes. For BwdData and Wrw, no large-tensor
//     instances are registered yet, so the filter returns an empty candidate
//     set for every >INT_MAX shape.
//
// Each Run* helper skips before calling CompileSolution when is_known_failing
// returns true (see conv_api_solution_count_large_stride_common.hpp for the
// rationale). Stale entries here produce only SKIPPED lines in the run
// summary, never failures, so CK integration is non-blocking -- trim
// opportunistically as CK adds large-tensor instances.
bool IsFwdKnownFailing2D(miopenDataType_t dtype, const Shape2D& s)
{
    if(s.c != 96 || s.h != s.w || s.n != 1)
        return false;
    // Square 14336 fails for all dtypes (existing); square 16384 newly fails
    // for all dtypes after the filter narrowed CK selection to large_tensor
    // instances and no registered tile config covers it.
    if(s.h == 14336 || s.h == 16384)
        return true;
    if(dtype == miopenFloat && s.h >= 10240 && s.h <= 14336)
        return true;
    return false;
}

bool IsBwdDataKnownFailing2D(miopenDataType_t /*dtype*/, const Shape2D& s)
{
    if(s.n == 1 && s.c == 96 &&
       ((s.h == 4096 && s.w == 8192) || (s.h == 8192 && s.w == 4096)))
        return true;
    if(s.c != 96 || s.h != s.w)
        return false;
    // Square c=96 shapes: pre-existing 4096..14336 tile-gap range, extended
    // through 16384 because the large-tensor filter blocks all >INT_MAX
    // BwdData shapes (no Bwd large-tensor CK instances registered yet).
    return s.h >= 4096 && s.h <= 16384;
}

bool IsWrwKnownFailing2D(miopenDataType_t dtype, const Shape2D& s)
{
    if(s.c != 96)
        return false;
    // Large-tensor filter blocks Wrw at every >INT_MAX shape across all
    // dtypes. Squares with h >= 4736 are all >INT_MAX (96*h*w > 2^31).
    if(s.h == s.w && s.h >= 4736 && s.h <= 16384)
        return true;
    // Same for the non-square {1,96,8192,4096} and {1,96,4096,8192} (both >INT_MAX).
    if(s.n == 1 && ((s.h == 8192 && s.w == 4096) || (s.h == 4096 && s.w == 8192)))
        return true;
    // FP32 has additional pre-existing tile gaps at sub-INT_MAX squares
    // (4096, 4608) — these would pass for FP16/BFP16.
    if(dtype == miopenFloat && s.h == s.w && s.h >= 4096 && s.h < 4736)
        return true;
    return false;
}

void RunFwd(const Shape2D& s, miopenDataType_t dtype)
{
    RunCompileFwd(s, dtype, SetupDescriptors2D, IsFwdKnownFailing2D,
                  "ConvHipImplicitGemmGroupFwdXdlops");
}
void RunBwdData(const Shape2D& s, miopenDataType_t dtype)
{
    RunCompileBwdData(s, dtype, SetupDescriptors2D, IsBwdDataKnownFailing2D,
                      "ConvHipImplicitGemmGroupBwdXdlops");
}
void RunWrw(const Shape2D& s, miopenDataType_t dtype)
{
    RunCompileWrw(s, dtype, SetupDescriptors2D, IsWrwKnownFailing2D,
                  "ConvHipImplicitGemmGroupWrwXdlops");
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
