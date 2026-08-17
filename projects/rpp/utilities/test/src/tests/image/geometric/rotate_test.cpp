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
#include "framework/compare_tensor.hpp"
#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/rotate_ref.hpp"

using namespace rpptest;

namespace {

std::string interp_token(RpptInterpolationType i) {
    return i == NEAREST_NEIGHBOR ? "NN" : "BILINEAR";
}

// A rotation angle (degrees, positive = anticlockwise) plus the interpolation. Angles are restricted
// to the cardinal set {0,90,180,270}: those map every output pixel to an integer source coordinate,
// so the golden is bit-exact and independent of the double-vs-float coordinate pipeline. General
// angles are deferred -- with fractional coords the golden's double
// maths diverges from the kernel's float pipeline at texel boundaries, producing diffs that cannot
// be cleanly attributed to a kernel defect. The shared bilinear 4-tap blend is already validated by
// warp_affine's halfshift case.
struct RotateParams {
    float angle;
    RpptInterpolationType interp;
    std::string name() const { return "a" + num_token(angle) + "_" + interp_token(interp); }
};

// Tolerances are set from legitimate numeric error only; they are NOT loosened to hide the real warp
// kernel defects this test surfaces (shared machinery): HOST
// bilinear last col/row off-by-one, I8 out-of-bounds fill = 0 (not -128), and HIP partial-ROI
// placement divergence.
double rotate_tolerance(DType dt, RpptInterpolationType interp) {
    // Nearest-neighbour copies a texel verbatim; bit-exact at cardinal angles (integer coords).
    if (interp == NEAREST_NEIGHBOR) return 0.0;
    // Bilinear: cos/sin of 90/270 deg are not exactly 0 in float, leaving a sub-LSB blend; allow
    // only that genuine rounding error. Integer types round to nearest.
    switch (dt) {
        case DType::U8: return 1.0;
        case DType::I8: return 1.0;
        case DType::F32: return 2e-3;
        case DType::F16: return 5e-3;
        default: return 0.0;
    }
}

template <typename T>
void run_rotate(const TestConfig& cfg, const RotateParams& op) {
    const Rpp32u c = static_cast<Rpp32u>(channels_of(cfg.layout));
    const TensorShape shape{cfg.size.n, c, cfg.size.h, cfg.size.w};
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layout);  // RPP takes a non-const ptr
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    // Per-image angle and ROI in host-accessible (pinned for HIP) memory.
    PinnedArray<Rpp32f> angle(cfg.backend, shape.n);
    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < shape.n; ++i) {
        angle[i] = op.angle;
        roi[i] = roiVec[i];
    }

    // (1) Host golden model. golden starts as a copy of the input so the untouched
    // (outside-ROI) region is defined; only the ROI-sized output region is overwritten.
    std::vector<T> input(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    golden = input;
    rotate_reference<T>(input.data(), golden.data(), desc, cfg.dtype, roi.data(), XYWH, angle.data(),
                        op.interp);

    // (2) Run RPP on the configured backend.
    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, shape.n);
    ASSERT_EQ(rppt_rotate(src.ptr(), &desc, dst.ptr(), &desc, angle.data(), op.interp, roi.data(),
                          XYWH, handle.get(), cfg.backend),
              RPP_SUCCESS);

    // (3) Retrieve the result on the host (no-op copy for HOST, device->host for HIP).
    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    // (4) Compare within tolerance over the ROI-sized output region at the destination origin.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), desc, roi.data(), XYWH,
                               rotate_tolerance(cfg.dtype, op.interp)));
}

}  // namespace

// Full name: Image_Geometric/RotateTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<Angle>_<Interp>
class RotateTest : public ::testing::TestWithParam<WithParams<RotateParams>> {};

TEST_P(RotateTest, Correctness) {
    const auto& p = GetParam();
    switch (p.cfg.dtype) {
        case DType::U8:
            run_rotate<Rpp8u>(p.cfg, p.op);
            break;
        case DType::F16:
            run_rotate<Rpp16f>(p.cfg, p.op);
            break;
        case DType::F32:
            run_rotate<Rpp32f>(p.cfg, p.op);
            break;
        case DType::I8:
            run_rotate<Rpp8s>(p.cfg, p.op);
            break;
        default:
            FAIL() << "unsupported dtype for rotate";
    }
}

// Cardinal angles: 0 (identity), 90/270 (both rotation directions), 180. All map to integer source
// coordinates, so the transform math is validated deterministically.
INSTANTIATE_TEST_SUITE_P(
    Image_Geometric, RotateTest,
    ::testing::ValuesIn(with_params<RotateParams>(
        make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                     {Layout::PKD3, Layout::PLN3, Layout::PLN1}, {Roi::Full, Roi::Partial}),
        {RotateParams{0.0f, NEAREST_NEIGHBOR}, RotateParams{0.0f, BILINEAR},
         RotateParams{90.0f, NEAREST_NEIGHBOR}, RotateParams{90.0f, BILINEAR},
         RotateParams{180.0f, NEAREST_NEIGHBOR}, RotateParams{180.0f, BILINEAR},
         RotateParams{270.0f, NEAREST_NEIGHBOR}, RotateParams{270.0f, BILINEAR}})),
    op_config_name<RotateParams>);
