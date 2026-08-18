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
#include "framework/skip_list.hpp"
#include "framework/tensor_setup.hpp"
#include "framework/voxel_tensor_setup.hpp"
#include "reference/arithmetic_scalar_ref.hpp"

using namespace rpptest;

namespace {

struct SubtractScalarParams {
    float sub;
    std::string name() const { return "sub" + num_token(sub); }
};

// Subtraction is exact; the only per-element error is the golden accumulating in double and
// storing float.
double abs_tolerance(DType) { return 1e-5; }

double rel_tolerance(DType) { return 1e-6; }

template <typename T>
void run_subtract_scalar(const VoxelConfig& cfg, const SubtractScalarParams& p) {
    // Dense strides, as the legacy voxel harness uses.
    GenericDescriptor desc(cfg.backend, voxel_dims(cfg.size, cfg.layout), cfg.dtype,
                           to_rpp_layout_3d(cfg.layout));
    const std::size_t count = generic_element_count(*desc);
    const std::size_t bytes = generic_byte_size(*desc, cfg.dtype);

    std::vector<T> input(count);
    fill_input_nd<T>(input.data(), *desc, cfg.dtype, 0);

    // Scaled per sample, so a kernel that applies param[0] to the whole batch is caught.
    PinnedArray<Rpp32f> sub(cfg.backend, cfg.size.n);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) sub[i] = p.sub * static_cast<Rpp32f>(i + 1);

    // The golden and the comparator read roiHost, never the pinned copy: the HIP op rewrites the
    // caller's ROI tensor in place.
    const std::vector<RpptROI3D> roiHost = make_voxel_roi(cfg.size, cfg.roi, cfg.roiType);
    PinnedArray<RpptROI3D> roi(cfg.backend, cfg.size.n);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) roi[i] = roiHost[i];

    // (1) Host golden model. Both sides start at the sentinel so a voxel the op never wrote stays
    // obviously unwritten instead of plausibly zero; only the ROI box is overwritten.
    std::vector<T> golden(count, nd_slack_poison<T>(cfg.dtype));
    std::vector<T> actual = golden;
    arithmetic_scalar_reference<T>(input.data(), golden.data(), *desc, roiHost.data(), cfg.roiType,
                                   ScalarArithmeticOp::Subtract, sub.data());

    // (2) Run RPP on the configured backend.
    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(actual.data(), bytes);

    RppHandle handle(cfg.backend, cfg.size.n);
    ASSERT_EQ(rppt_subtract_scalar(src.ptr(), desc.get(), dst.ptr(), desc.get(), sub.data(),
                                   roi.data(), to_rpp_roi3d_type(cfg.roiType), handle.get(),
                                   cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    // (3) Compare inside the ROI box.
    EXPECT_TRUE(compare_voxel_roi<T>(actual.data(), golden.data(), *desc, roiHost.data(),
                                     cfg.roiType, abs_tolerance(cfg.dtype),
                                     rel_tolerance(cfg.dtype)));
}

}  // namespace

// Full name:
// Voxel_Arithmetic/SubtractScalarTest.Correctness/<Backend>_F32toF32_<Layout>_<Roi>_<Roi3DType>_<Shape>_<Sub>
class SubtractScalarTest : public SkipListTest<VoxelWithParams<SubtractScalarParams>> {
};

TEST_P(SubtractScalarTest, Correctness) {
    const auto& p = GetParam();
    ASSERT_EQ(p.cfg.dtype, DType::F32) << "subtract_scalar is F32 only";
    run_subtract_scalar<Rpp32f>(p.cfg, p.op);
}

// F32 only, per the API header. The fill spans [0, 1] and the subtrahend is 40 (the value the
// legacy voxel harness drives this op with), so every result is strongly negative: a kernel that
// clamped at 0 would show up on every voxel.
INSTANTIATE_TEST_SUITE_P(Voxel_Arithmetic, SubtractScalarTest,
                         ::testing::ValuesIn(voxel_with_params<SubtractScalarParams>(
                             make_voxel_configs({DType::F32},
                                                {VoxelLayout::NCDHW1, VoxelLayout::NCDHW3,
                                                 VoxelLayout::NDHWC3},
                                                {Roi::Full, Roi::Partial},
                                                {Roi3D::XYZWHD, Roi3D::LTFRBB}),
                             {{40.0f}})),
                         voxel_op_config_name<SubtractScalarParams>);
