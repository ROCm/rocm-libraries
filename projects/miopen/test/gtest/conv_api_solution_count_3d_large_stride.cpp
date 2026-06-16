// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// API-level applicability sweep for 3D grouped CK xdlops solvers on shapes
// whose total element-strides exceed INT_MAX, using reproducer family:
// torch.nn.Conv3d(96, 32, kernel_size=3, padding=1) on x = (1, 96, Nx, Ny, Z)).
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

struct Shape3D
{
    int n, c, d, h, w;
};

// Mirrors the PyTorch reproducer in ROCM-23997.
//   torch.nn.Conv3d(96, 32, kernel_size=3, padding=1)
//   x = torch.empty((1, 96, Nx, Ny, Z))
// Boundary-bracketing set rather than a full spatial_xy×z cross-product: the
// element count 96*nxy^2*z determines whether x's strides cross INT_MAX
// (2,147,483,647), so most cross-product cells are redundant. This list keeps a
// below/above control at each spatial size plus the fine bracket at nxy=512
// {84(below),86,88(above)} where the boundary actually falls.
std::vector<Shape3D> ReproducerShapes()
{
    return {
        // {n, c, d, h, w}; boundary = 96*nxy^2*z vs INT_MAX (2,147,483,647)
        {1, 96, 64, 64, 64},      // below  (control, small spatial)
        {1, 96, 128, 128, 128},   // below  (control)
        {1, 96, 256, 256, 64},    // below  (control, mid spatial)
        {1, 96, 256, 256, 512},   // above
        {1, 96, 256, 256, 1024},  // above
        {1, 96, 512, 512, 84},    // below  } fine boundary
        {1, 96, 512, 512, 86},    // above  } bracket at
        {1, 96, 512, 512, 88},    // above  } nxy=512
        {1, 96, 1024, 1024, 16},  // below  (large spatial control)
        {1, 96, 1024, 1024, 32},  // above
        {1, 96, 1024, 1024, 1024} // above  (largest)
    };
}

::testing::AssertionResult
SetupDescriptors3D(const Shape3D& s, miopenDataType_t dtype, Descriptors& d)
{
    const int x_dims[5] = {s.n, s.c, s.d, s.h, s.w};
    const int w_dims[5] = {32, 96, 3, 3, 3};
    return SetupDescriptorsImpl<3>(x_dims, w_dims, dtype, d);
}

void RunFwd(const Shape3D& s, miopenDataType_t dtype)
{
    RunCompileFwd(s, dtype, SetupDescriptors3D, "ConvHipImplicitGemm3DGroupFwdXdlops");
}
void RunBwdData(const Shape3D& s, miopenDataType_t dtype)
{
    RunCompileBwdData(s, dtype, SetupDescriptors3D, "ConvHipImplicitGemm3DGroupBwdXdlops");
}
void RunWrw(const Shape3D& s, miopenDataType_t dtype)
{
    RunCompileWrw(s, dtype, SetupDescriptors3D, "ConvHipImplicitGemm3DGroupWrwXdlops");
}

class GPU_ConvApi_SolutionCount3DLargeStride_FP16 : public ::testing::TestWithParam<Shape3D>
{
};
class GPU_ConvApi_SolutionCount3DLargeStride_FP32 : public ::testing::TestWithParam<Shape3D>
{
};
class GPU_ConvApi_SolutionCount3DLargeStride_BFP16 : public ::testing::TestWithParam<Shape3D>
{
};

} // namespace

TEST_P(GPU_ConvApi_SolutionCount3DLargeStride_FP16, FwdNonZeroAndIncludesCK)
{
    RunFwd(GetParam(), miopenHalf);
}
TEST_P(GPU_ConvApi_SolutionCount3DLargeStride_FP16, BwdDataNonZeroAndIncludesCK)
{
    RunBwdData(GetParam(), miopenHalf);
}
TEST_P(GPU_ConvApi_SolutionCount3DLargeStride_FP16, WrwNonZeroAndIncludesCK)
{
    RunWrw(GetParam(), miopenHalf);
}

TEST_P(GPU_ConvApi_SolutionCount3DLargeStride_FP32, FwdNonZeroAndIncludesCK)
{
    RunFwd(GetParam(), miopenFloat);
}
TEST_P(GPU_ConvApi_SolutionCount3DLargeStride_FP32, BwdDataNonZeroAndIncludesCK)
{
    RunBwdData(GetParam(), miopenFloat);
}
TEST_P(GPU_ConvApi_SolutionCount3DLargeStride_FP32, WrwNonZeroAndIncludesCK)
{
    RunWrw(GetParam(), miopenFloat);
}

TEST_P(GPU_ConvApi_SolutionCount3DLargeStride_BFP16, FwdNonZeroAndIncludesCK)
{
    RunFwd(GetParam(), miopenBFloat16);
}
TEST_P(GPU_ConvApi_SolutionCount3DLargeStride_BFP16, BwdDataNonZeroAndIncludesCK)
{
    RunBwdData(GetParam(), miopenBFloat16);
}
TEST_P(GPU_ConvApi_SolutionCount3DLargeStride_BFP16, WrwNonZeroAndIncludesCK)
{
    RunWrw(GetParam(), miopenBFloat16);
}

INSTANTIATE_TEST_SUITE_P(Standard,
                         GPU_ConvApi_SolutionCount3DLargeStride_FP16,
                         ::testing::ValuesIn(ReproducerShapes()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         GPU_ConvApi_SolutionCount3DLargeStride_FP32,
                         ::testing::ValuesIn(ReproducerShapes()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         GPU_ConvApi_SolutionCount3DLargeStride_BFP16,
                         ::testing::ValuesIn(ReproducerShapes()));
