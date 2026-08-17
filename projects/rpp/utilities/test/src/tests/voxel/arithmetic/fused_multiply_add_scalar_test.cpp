/*
MIT License

Copyright (c) 2026 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <gtest/gtest.h>
#include <rpp/rpp.h>

#include <string>
#include <vector>

#include "framework/backend_memory.hpp"
#include "framework/config_param.hpp"
#include "framework/generic_tensor_setup.hpp"
#include "framework/tensor_setup.hpp"
#include "framework/voxel_tensor_setup.hpp"
#include "reference/arithmetic_scalar_ref.hpp"

using namespace rpptest;

namespace {

struct FmaddScalarParams {
    float mul, add;
    std::string name() const { return "mul" + num_token(mul) + "_add" + num_token(add); }
};

// The golden accumulates in double and stores float; a hardware FMA and a separate
// multiply-then-add differ by far less than this.
double abs_tolerance(DType) { return 1e-5; }

double rel_tolerance(DType) { return 1e-6; }

template <typename T>
void run_fused_multiply_add_scalar(const VoxelConfig& cfg, const FmaddScalarParams& p) {
    GenericDescriptor desc(cfg.backend, voxel_dims(cfg.size, cfg.layout), cfg.dtype,
                           to_rpp_layout_3d(cfg.layout));
    const std::size_t count = generic_element_count(*desc);
    const std::size_t bytes = generic_byte_size(*desc, cfg.dtype);

    std::vector<T> input(count);
    fill_input_nd<T>(input.data(), *desc, cfg.dtype, 0);

    // Distinct multiplier and addend per sample, so a kernel that applied param[0] to the whole
    // batch is caught.
    PinnedArray<Rpp32f> mul(cfg.backend, cfg.size.n);
    PinnedArray<Rpp32f> add(cfg.backend, cfg.size.n);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) {
        mul[i] = p.mul * static_cast<Rpp32f>(i + 1);
        add[i] = p.add * static_cast<Rpp32f>(i + 1);
    }

    // The golden and the comparator read roiHost, never the pinned copy: the HIP op rewrites the
    // caller's ROI tensor in place.
    const std::vector<RpptROI3D> roiHost = make_voxel_roi(cfg.size, cfg.roi, cfg.roiType);
    PinnedArray<RpptROI3D> roi(cfg.backend, cfg.size.n);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) roi[i] = roiHost[i];

    // (1) Host golden model. Both sides start at the sentinel so a voxel the op never writes stays
    // obviously unwritten rather than plausibly zero.
    std::vector<T> golden(count, nd_slack_poison<T>(cfg.dtype));
    std::vector<T> actual = golden;
    arithmetic_scalar_reference<T>(input.data(), golden.data(), *desc, roiHost.data(), cfg.roiType,
                                   ScalarArithmeticOp::FusedMultiplyAdd, mul.data(), add.data());

    // (2) Run RPP on the configured backend.
    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(actual.data(), bytes);

    RppHandle handle(cfg.backend, cfg.size.n);
    ASSERT_EQ(rppt_fused_multiply_add_scalar(src.ptr(), desc.get(), dst.ptr(), desc.get(),
                                             mul.data(), add.data(), roi.data(),
                                             to_rpp_roi3d_type(cfg.roiType), handle.get(),
                                             cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    // (3) Compare the voxels inside the ROI box.
    EXPECT_TRUE(compare_voxel_roi<T>(actual.data(), golden.data(), *desc, roiHost.data(),
                                     cfg.roiType, abs_tolerance(cfg.dtype),
                                     rel_tolerance(cfg.dtype)));
}

}  // namespace

// Full name:
// Voxel_Arithmetic/FusedMultiplyAddScalarTest.Correctness/<Backend>_F32toF32_<Layout>_<Roi>_<Roi3DType>_<Shape>_<Mul>_<Add>
class FusedMultiplyAddScalarTest
    : public ::testing::TestWithParam<VoxelWithParams<FmaddScalarParams>> {};

TEST_P(FusedMultiplyAddScalarTest, Correctness) {
    const auto& p = GetParam();
    ASSERT_EQ(p.cfg.dtype, DType::F32) << "fused_multiply_add_scalar is F32 only";
    run_fused_multiply_add_scalar<Rpp32f>(p.cfg, p.op);
}

// f32 -> f32 is the op's only documented conversion.
//
// mul 80 / add 5 are the values the legacy voxel harness uses. With a [0, 1] fill the two terms are
// far apart in magnitude, so a kernel that swapped mulTensor and addTensor, or dropped either, is
// caught; and every result leaves [0, 1], so a clamp to the image-intensity range would show.
INSTANTIATE_TEST_SUITE_P(Voxel_Arithmetic, FusedMultiplyAddScalarTest,
                         ::testing::ValuesIn(voxel_with_params<FmaddScalarParams>(
                             make_voxel_configs({DType::F32},
                                                {VoxelLayout::NCDHW1, VoxelLayout::NCDHW3,
                                                 VoxelLayout::NDHWC3},
                                                {Roi::Full, Roi::Partial},
                                                {Roi3D::XYZWHD, Roi3D::LTFRBB}),
                             {{80.0f, 5.0f}})),
                         voxel_op_config_name<FmaddScalarParams>);
