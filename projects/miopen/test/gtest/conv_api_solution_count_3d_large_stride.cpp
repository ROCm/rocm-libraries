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
#include <array>
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
std::vector<Shape3D> ReproducerShapes()
{
    constexpr std::array<int, 5> spatial_xy = {64, 128, 256, 512, 1024};
    constexpr std::array<int, 10> z_values  = {16, 32, 64, 84, 86, 88, 128, 256, 512, 1024};
    std::vector<Shape3D> out;
    out.reserve(spatial_xy.size() * z_values.size());
    for(int nxy : spatial_xy)
        for(int z : z_values)
            out.push_back({1, 96, nxy, nxy, z});
    return out;
}

::testing::AssertionResult
SetupDescriptors3D(const Shape3D& s, miopenDataType_t dtype, Descriptors& d)
{
    const int x_dims[5] = {s.n, s.c, s.d, s.h, s.w};
    const int w_dims[5] = {32, 96, 3, 3, 3};
    return SetupDescriptorsImpl<3>(x_dims, w_dims, dtype, d);
}

// Known CK applicability gaps for 3D large-stride shapes (gfx942).
// Two failure modes are merged into these lists:
//
//   * Sub-INT_MAX shapes where CK has a tile-selection miss for a particular
//     dtype/shape (pre-existing CK gap, unrelated to large-tensor work).
//   * Strides >INT_MAX shapes where the filter (correctly) restricts CK
//     selection to *_Large_Tensor instances. For Fwd, registered Fwd
//     large-tensor instances cover most shapes but miss a handful of the very
//     largest (notably 1024-cubed tail). For BwdData and Wrw, no large-tensor
//     instances are registered yet, so the filter returns an empty candidate
//     set for every >INT_MAX shape (96*D*H*W > 2^31).
//
// Each Run* helper skips before calling CompileSolution when is_known_failing
// returns true (see conv_api_solution_count_large_stride_common.hpp for the
// rationale). Stale entries here produce only SKIPPED lines in the run
// summary, never failures, so CK integration is non-blocking -- trim
// opportunistically as CK adds large-tensor instances.

template <std::size_t N>
bool MatchesDhw(const Shape3D& s, const std::array<std::array<int, 3>, N>& dhw)
{
    for(const auto& t : dhw)
        if(t[0] == s.d && t[1] == s.h && t[2] == s.w)
            return true;
    return false;
}

bool IsFwdKnownFailing3D(miopenDataType_t dtype, const Shape3D& s)
{
    // Shapes the CK 3D grouped fwd xdlops solver rejects across all dtypes —
    // these are the >INT_MAX tails that fall outside the registered Fwd
    // large-tensor tile coverage.
    static constexpr std::array<std::array<int, 3>, 4> baseline = {{
        {512, 512, 1024},
        {1024, 1024, 256},
        {1024, 1024, 512},
        {1024, 1024, 1024},
    }};
    if(MatchesDhw(s, baseline))
        return true;
    if(dtype != miopenFloat) // FP16/BFP16 only have the baseline above.
        return false;
    // FP32 has additional Fwd gaps at the 1024-cubed sub-INT_MAX tail and
    // 512-cubed >INT_MAX shapes the large-tensor instances don't yet cover.
    static constexpr std::array<std::array<int, 3>, 5> fp32_extra = {{
        {512, 512, 512},
        {1024, 1024, 84},
        {1024, 1024, 86},
        {1024, 1024, 88},
        {1024, 1024, 128},
    }};
    return MatchesDhw(s, fp32_extra);
}

bool IsBwdDataKnownFailing3D(miopenDataType_t dtype, const Shape3D& s)
{
    // Shapes the CK 3D grouped bwd-data xdlops solver rejects across all dtypes.
    // Mix of pre-existing CK tile-selection gaps at sub-INT_MAX shapes and
    // every >INT_MAX shape (the filter blocks all of these because no Bwd
    // large-tensor CK instances are registered yet).
    static constexpr std::array<std::array<int, 3>, 22> baseline = {{
        {128, 128, 1024},  {256, 256, 256},    {256, 256, 512},   {256, 256, 1024},
        {512, 512, 64},    {512, 512, 84},     {512, 512, 86},    {512, 512, 88},
        {512, 512, 128},   {512, 512, 256},    {512, 512, 512},   {512, 512, 1024},
        {1024, 1024, 16},  {1024, 1024, 32},   {1024, 1024, 64},  {1024, 1024, 84},
        {1024, 1024, 86},  {1024, 1024, 88},   {1024, 1024, 128}, {1024, 1024, 256},
        {1024, 1024, 512}, {1024, 1024, 1024},
    }};
    if(MatchesDhw(s, baseline))
        return true;
    if(dtype != miopenFloat) // Only FP32 has additional bwd-data gaps.
        return false;
    static constexpr std::array<std::array<int, 3>, 5> fp32_extra = {{
        {128, 128, 512},
        {256, 256, 86},
        {256, 256, 88},
        {256, 256, 128},
        {512, 512, 32},
    }};
    return MatchesDhw(s, fp32_extra);
}

bool IsWrwKnownFailing3D(miopenDataType_t dtype, const Shape3D& s)
{
    // Common >INT_MAX tail blocked across all dtypes — large-tensor filter
    // returns an empty candidate set because no Wrw large-tensor CK instances
    // are registered yet. (96*512*512*512 and beyond all exceed 2^31.)
    static constexpr std::array<std::array<int, 3>, 6> common = {{
        {512, 512, 512},
        {512, 512, 1024},
        {1024, 1024, 128},
        {1024, 1024, 256},
        {1024, 1024, 512},
        {1024, 1024, 1024},
    }};
    if(MatchesDhw(s, common))
        return true;
    if(dtype == miopenFloat)
    {
        // FP32 has additional pre-existing CK tile gaps at sub-INT_MAX squares
        // (256x256x{86,88,128}, 512x512x{32,64,..,256}, 1024x1024x{16,..,88}).
        static constexpr std::array<std::array<int, 3>, 21> fp32_extra = {{
            {128, 128, 512},  {128, 128, 1024}, {256, 256, 86},   {256, 256, 88},
            {256, 256, 128},  {256, 256, 256},  {256, 256, 512},  {256, 256, 1024},
            {512, 512, 32},   {512, 512, 64},   {512, 512, 84},   {512, 512, 86},
            {512, 512, 88},   {512, 512, 128},  {512, 512, 256},  {1024, 1024, 16},
            {1024, 1024, 32}, {1024, 1024, 64}, {1024, 1024, 84}, {1024, 1024, 86},
            {1024, 1024, 88},
        }};
        return MatchesDhw(s, fp32_extra);
    }
    // FP16/BFP16: every >INT_MAX shape the Wrw filter cuts (96*D*H*W > 2^31).
    static constexpr std::array<std::array<int, 3>, 11> fp16_bf16_extra = {{
        {256, 256, 512},
        {256, 256, 1024},
        {512, 512, 86},
        {512, 512, 88},
        {512, 512, 128},
        {512, 512, 256},
        {1024, 1024, 32},
        {1024, 1024, 64},
        {1024, 1024, 84},
        {1024, 1024, 86},
        {1024, 1024, 88},
    }};
    return MatchesDhw(s, fp16_bf16_extra);
}

void RunFwd(const Shape3D& s, miopenDataType_t dtype)
{
    RunCompileFwd(
        s, dtype, SetupDescriptors3D, IsFwdKnownFailing3D, "ConvHipImplicitGemm3DGroupFwdXdlops");
}
void RunBwdData(const Shape3D& s, miopenDataType_t dtype)
{
    RunCompileBwdData(s,
                      dtype,
                      SetupDescriptors3D,
                      IsBwdDataKnownFailing3D,
                      "ConvHipImplicitGemm3DGroupBwdXdlops");
}
void RunWrw(const Shape3D& s, miopenDataType_t dtype)
{
    RunCompileWrw(
        s, dtype, SetupDescriptors3D, IsWrwKnownFailing3D, "ConvHipImplicitGemm3DGroupWrwXdlops");
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
