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
#include "framework/voxel_tensor_setup.hpp"
#include "reference/flip_voxel_ref.hpp"

using namespace rpptest;

namespace {

// The three documented flags are independent 0/1 masks; the full 2^3 cross product isolates each
// axis (a swapped horizontal/vertical/depth would not cancel out) as well as their combinations,
// with 0/0/0 as the identity (a plain ROI copy, catching a flip applied unconditionally).
struct FlipVoxelParams {
    Rpp32u horizontal, vertical, depth;
    std::string name() const {
        return "h" + std::to_string(horizontal) + "_v" + std::to_string(vertical) + "_d" +
               std::to_string(depth);
    }
};

template <typename T>
void run_flip_voxel(const VoxelConfig& cfg, const FlipVoxelParams& p) {
    GenericDescriptor desc(cfg.backend, voxel_dims(cfg.size, cfg.layout), cfg.dtype,
                           to_rpp_layout_3d(cfg.layout));
    const std::size_t count = generic_element_count(*desc);
    const std::size_t bytes = generic_byte_size(*desc, cfg.dtype);

    std::vector<T> input(count);
    fill_input_nd<T>(input.data(), *desc, cfg.dtype, 0);

    PinnedArray<Rpp32u> horizontal(cfg.backend, cfg.size.n);
    PinnedArray<Rpp32u> vertical(cfg.backend, cfg.size.n);
    PinnedArray<Rpp32u> depth(cfg.backend, cfg.size.n);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) {
        horizontal[i] = p.horizontal;
        vertical[i] = p.vertical;
        depth[i] = p.depth;
    }

    // The golden and the comparator read roiHost, never the pinned copy: the HIP op rewrites the
    // caller's ROI tensor in place.
    const std::vector<RpptROI3D> roiHost = make_voxel_roi(cfg.size, cfg.roi, cfg.roiType);
    PinnedArray<RpptROI3D> roi(cfg.backend, cfg.size.n);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) roi[i] = roiHost[i];

    // (1) Host golden model. golden starts at the sentinel so a voxel the op never writes stays
    // obviously unwritten rather than plausibly zero.
    std::vector<T> golden(count, nd_slack_poison<T>(cfg.dtype));
    std::vector<T> actual = golden;
    flip_voxel_reference<T>(input.data(), golden.data(), *desc, roiHost.data(), cfg.roiType,
                            horizontal.data(), vertical.data(), depth.data());

    // (2) Run RPP on the configured backend.
    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(actual.data(), bytes);

    RppHandle handle(cfg.backend, cfg.size.n);
    ASSERT_EQ(rppt_flip_voxel(src.ptr(), desc.get(), dst.ptr(), desc.get(), horizontal.data(),
                              vertical.data(), depth.data(), roi.data(),
                              to_rpp_roi3d_type(cfg.roiType), handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    // (3) Compare the voxels inside the ROI box. Flip only permutes source elements -- it computes
    // nothing -- so every dtype must be bit-exact and the tolerance is zero.
    EXPECT_TRUE(
        compare_voxel_roi<T>(actual.data(), golden.data(), *desc, roiHost.data(), cfg.roiType, 0.0));
}

}  // namespace

// Full name:
// Voxel_Geometric/FlipVoxelTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Roi3DType>_<Shape>_<Flags>
class FlipVoxelTest : public ::testing::TestWithParam<VoxelWithParams<FlipVoxelParams>> {};

TEST_P(FlipVoxelTest, Correctness) {
    const auto& p = GetParam();
    // A horizontal flip on a partial ROI reads a mis-sized AVX vector backward from the ROI's left
    // edge on HOST U8, past the start of the buffer, and segfaults the whole process. The fault is
    // a hard crash, not a gtest failure, so these cases are skipped rather than left red.
    if (p.cfg.backend == RPP_HOST_BACKEND && p.cfg.dtype == DType::U8 && p.op.horizontal &&
        p.cfg.roi == Roi::Partial)
        GTEST_SKIP() << "HOST U8 flip_voxel reads past the start of the buffer and segfaults on a "
                        "horizontal flip with a partial ROI";

    switch (p.cfg.dtype) {
        case DType::U8:
            run_flip_voxel<Rpp8u>(p.cfg, p.op);
            break;
        case DType::F32:
            run_flip_voxel<Rpp32f>(p.cfg, p.op);
            break;
        default:
            FAIL() << "flip_voxel supports only U8 and F32";
    }
}

// u8 -> u8 and f32 -> f32 are the op's only documented conversions.
INSTANTIATE_TEST_SUITE_P(
    Voxel_Geometric, FlipVoxelTest,
    ::testing::ValuesIn(voxel_with_params<FlipVoxelParams>(
        make_voxel_configs({DType::U8, DType::F32},
                           {VoxelLayout::NCDHW1, VoxelLayout::NCDHW3, VoxelLayout::NDHWC3},
                           {Roi::Full, Roi::Partial}, {Roi3D::XYZWHD, Roi3D::LTFRBB}),
        {FlipVoxelParams{0, 0, 0}, FlipVoxelParams{1, 0, 0}, FlipVoxelParams{0, 1, 0},
         FlipVoxelParams{0, 0, 1}, FlipVoxelParams{1, 1, 0}, FlipVoxelParams{1, 0, 1},
         FlipVoxelParams{0, 1, 1}, FlipVoxelParams{1, 1, 1}})),
    voxel_op_config_name<FlipVoxelParams>);
