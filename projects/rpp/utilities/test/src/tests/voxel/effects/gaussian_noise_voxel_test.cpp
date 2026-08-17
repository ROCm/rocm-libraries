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
#include "reference/gaussian_noise_voxel_ref.hpp"

using namespace rpptest;

namespace {

// Scoped exactly like the image-domain gaussian_noise: the Correctness intent pins
// (mean 0, stdDev 0), where N(mean, stdDev) collapses to a point mass at 0 and the additive noise
// degenerates to a passthrough -- bit-exact and seed-independent (see gaussian_noise_voxel_ref.hpp)
// -- and the Negative intent covers the documented parameter contract. No golden is possible away
// from that corner; the kernel's Box-Muller stream is not described by the public API.
constexpr Rpp32u kSeed = 42u;

template <typename T>
void run_gaussian_noise_voxel(const VoxelConfig& cfg) {
    GenericDescriptor desc(cfg.backend, voxel_dims(cfg.size, cfg.layout), cfg.dtype,
                           to_rpp_layout_3d(cfg.layout));
    const std::size_t count = generic_element_count(*desc);
    const std::size_t bytes = generic_byte_size(*desc, cfg.dtype);

    std::vector<T> input(count);
    fill_input_nd<T>(input.data(), *desc, cfg.dtype, 0);

    PinnedArray<Rpp32f> mean(cfg.backend, cfg.size.n), stdDev(cfg.backend, cfg.size.n);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) {
        mean[i] = 0.0f;  // the RNG-free identity corner
        stdDev[i] = 0.0f;
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
    gaussian_noise_voxel_identity_reference<T>(input.data(), golden.data(), *desc, roiHost.data(),
                                               cfg.roiType);

    // (2) Run RPP on the configured backend.
    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(actual.data(), bytes);

    RppHandle handle(cfg.backend, cfg.size.n);
    ASSERT_EQ(rppt_gaussian_noise_voxel(src.ptr(), desc.get(), dst.ptr(), desc.get(), mean.data(),
                                        stdDev.data(), kSeed, roi.data(),
                                        to_rpp_roi3d_type(cfg.roiType), handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    // (3) Compare the voxels inside the ROI box, bit-exactly: the corner adds literal zero.
    EXPECT_TRUE(compare_voxel_roi<T>(actual.data(), golden.data(), *desc, roiHost.data(),
                                     cfg.roiType, 0.0));
}

// One out-of-contract parameter pair per case; the header requires meanTensor[i] >= 0 and
// stdDevTensor[i] >= 0.
struct GaussianNoiseVoxelNegativeParams {
    float mean;
    float stdDev;
    std::string name() const { return "m" + num_token(mean) + "_s" + num_token(stdDev); }
};

// A negative mean or standard deviation is not a legal call and must be reported rather than
// silently producing an undefined volume. Only the status is asserted: the header does not name
// which RPP_ERROR* an out-of-range scalar maps to.
template <typename T>
void run_gaussian_noise_voxel_negative(const VoxelConfig& cfg,
                                       const GaussianNoiseVoxelNegativeParams& op) {
    GenericDescriptor desc(cfg.backend, voxel_dims(cfg.size, cfg.layout), cfg.dtype,
                           to_rpp_layout_3d(cfg.layout));
    const std::size_t count = generic_element_count(*desc);
    const std::size_t bytes = generic_byte_size(*desc, cfg.dtype);

    std::vector<T> input(count);
    fill_input_nd<T>(input.data(), *desc, cfg.dtype, 0);

    PinnedArray<Rpp32f> mean(cfg.backend, cfg.size.n), stdDev(cfg.backend, cfg.size.n);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) {
        mean[i] = op.mean;
        stdDev[i] = op.stdDev;
    }

    const std::vector<RpptROI3D> roiHost = make_voxel_roi(cfg.size, cfg.roi, cfg.roiType);
    PinnedArray<RpptROI3D> roi(cfg.backend, cfg.size.n);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) roi[i] = roiHost[i];

    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);

    RppHandle handle(cfg.backend, cfg.size.n);
    const RppStatus status = rppt_gaussian_noise_voxel(
        src.ptr(), desc.get(), dst.ptr(), desc.get(), mean.data(), stdDev.data(), kSeed, roi.data(),
        to_rpp_roi3d_type(cfg.roiType), handle.get(), cfg.backend);
    handle.sync();
    EXPECT_NE(status, RPP_SUCCESS)
        << "gaussian_noise_voxel accepted out-of-contract parameters (mean " << op.mean
        << ", stdDev " << op.stdDev << ")";
}

}  // namespace

// Full name:
// Voxel_Effects/GaussianNoiseVoxelTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Roi3DType>_<Shape>
class GaussianNoiseVoxelTest : public ::testing::TestWithParam<VoxelConfig> {};

TEST_P(GaussianNoiseVoxelTest, Correctness) {
    const VoxelConfig& cfg = GetParam();
    if (cfg.dtype == DType::U8)
        run_gaussian_noise_voxel<Rpp8u>(cfg);
    else
        run_gaussian_noise_voxel<Rpp32f>(cfg);
}

// u8 -> u8 and f32 -> f32 are the op's only documented conversions.
INSTANTIATE_TEST_SUITE_P(Voxel_Effects, GaussianNoiseVoxelTest,
                         ::testing::ValuesIn(make_voxel_configs(
                             {DType::U8, DType::F32},
                             {VoxelLayout::NCDHW1, VoxelLayout::NCDHW3, VoxelLayout::NDHWC3},
                             {Roi::Full, Roi::Partial}, {Roi3D::XYZWHD, Roi3D::LTFRBB})),
                         voxel_config_param_name);

// Full name:
// Voxel_Effects/GaussianNoiseVoxelNegativeTest.Negative/<Backend>_..._m<M>_s<S>
class GaussianNoiseVoxelNegativeTest
    : public ::testing::TestWithParam<VoxelWithParams<GaussianNoiseVoxelNegativeParams>> {};

TEST_P(GaussianNoiseVoxelNegativeTest, Negative) {
    const auto& p = GetParam();
    run_gaussian_noise_voxel_negative<Rpp32f>(p.cfg, p.op);
}

INSTANTIATE_TEST_SUITE_P(Voxel_Effects, GaussianNoiseVoxelNegativeTest,
                         ::testing::ValuesIn(voxel_with_params<GaussianNoiseVoxelNegativeParams>(
                             make_voxel_configs({DType::F32}, {VoxelLayout::NCDHW1}, {Roi::Full},
                                                {Roi3D::XYZWHD}),
                             {GaussianNoiseVoxelNegativeParams{-0.5f, 1.0f},
                              GaussianNoiseVoxelNegativeParams{0.5f, -1.0f}})),
                         voxel_op_config_name<GaussianNoiseVoxelNegativeParams>);
