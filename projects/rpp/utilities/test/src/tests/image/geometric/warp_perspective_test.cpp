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

#include <array>
#include <string>
#include <vector>

#include "framework/backend_memory.hpp"
#include "framework/compare_tensor.hpp"
#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/warp_perspective_ref.hpp"

using namespace rpptest;

namespace {

std::string interp_token(RpptInterpolationType i) {
    return i == NEAREST_NEIGHBOR ? "NN" : "BILINEAR";
}

// A 3x3 projective matrix (row-major, destination->source) plus the interpolation. The matrices
// keep the perspective row {m6,m7,m8}={0,0,1} (w==1), so the source coord is invariant to the
// undocumented mapping-origin convention (matching warp_affine). Genuine-perspective transforms
// (w!=1) are deferred: the golden's double-precision coordinate math diverges from the kernel's
// float pipeline at texel boundaries, and the kernel additionally blackens in-bounds perspective
// pixels (e.g. output(0,0) with w==1) in a way that needs dedicated convention characterization
// before its diffs can be classified as findings vs. numerical artifacts.
struct WarpPerspectiveParams {
    std::array<float, 9> m;
    RpptInterpolationType interp;
    std::string tag;
    std::string name() const { return tag + "_" + interp_token(interp); }
};

// Tolerances are set from legitimate numeric error only; they are NOT loosened to hide the real
// warp_perspective kernel defects this test surfaces (shared with warp_affine
// findings #15-#17): HOST bilinear last col/row off-by-one, I8 out-of-bounds fill = 0 (not -128),
// and HIP partial-ROI placement divergence.
double warp_perspective_tolerance(DType dt, RpptInterpolationType interp) {
    // Nearest-neighbour copies a texel verbatim; bit-exact for the w==1 (integer-coord) cases.
    if (interp == NEAREST_NEIGHBOR) return 0.0;
    // Bilinear blends in float; allow only genuine rounding error. Integer types round to nearest.
    switch (dt) {
        case DType::U8: return 1.0;
        case DType::I8: return 1.0;
        case DType::F32: return 2e-3;
        case DType::F16: return 5e-3;
        default: return 0.0;
    }
}

template <typename T>
void run_warp_perspective(const TestConfig& cfg, const WarpPerspectiveParams& op) {
    const Rpp32u c = static_cast<Rpp32u>(channels_of(cfg.layout));
    const TensorShape shape{cfg.size.n, c, cfg.size.h, cfg.size.w};
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layout);  // RPP takes a non-const ptr
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    // Per-image 3x3 matrices (9 each) and ROIs in host-accessible (pinned for HIP) memory.
    PinnedArray<Rpp32f> persp(cfg.backend, shape.n * 9);
    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < shape.n; ++i) {
        for (int k = 0; k < 9; ++k) persp[i * 9 + k] = op.m[k];
        roi[i] = roiVec[i];
    }

    // (1) Host golden model. golden starts as a copy of the input so the untouched
    // (outside-ROI) region is defined; only the ROI-sized output region is overwritten.
    std::vector<T> input(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    golden = input;
    warp_perspective_reference<T>(input.data(), golden.data(), desc, cfg.dtype, roi.data(), XYWH,
                                  persp.data(), op.interp);

    // (2) Run RPP on the configured backend.
    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, shape.n);
    ASSERT_EQ(rppt_warp_perspective(src.ptr(), &desc, dst.ptr(), &desc, persp.data(), op.interp,
                                    roi.data(), XYWH, handle.get(), cfg.backend),
              RPP_SUCCESS);

    // (3) Retrieve the result on the host (no-op copy for HOST, device->host for HIP).
    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    // (4) Compare within tolerance over the ROI-sized output region at the destination origin.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), desc, roi.data(), XYWH,
                               warp_perspective_tolerance(cfg.dtype, op.interp)));
}

}  // namespace

// Full name: Image_Geometric/WarpPerspectiveTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<Matrix>_<Interp>
class WarpPerspectiveTest : public ::testing::TestWithParam<WithParams<WarpPerspectiveParams>> {};

TEST_P(WarpPerspectiveTest, Correctness) {
    const auto& p = GetParam();
    switch (p.cfg.dtype) {
        case DType::U8:
            run_warp_perspective<Rpp8u>(p.cfg, p.op);
            break;
        case DType::F16:
            run_warp_perspective<Rpp16f>(p.cfg, p.op);
            break;
        case DType::F32:
            run_warp_perspective<Rpp32f>(p.cfg, p.op);
            break;
        case DType::I8:
            run_warp_perspective<Rpp8s>(p.cfg, p.op);
            break;
        default:
            FAIL() << "unsupported dtype for warp_perspective";
    }
}

// identity: output == source. shift: integer translation (w==1). halfshift: half-pixel translation,
// bilinear only, exercises the 4-tap blend.
INSTANTIATE_TEST_SUITE_P(
    Image_Geometric, WarpPerspectiveTest,
    ::testing::ValuesIn(with_params<WarpPerspectiveParams>(
        make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                     {Layout::PKD3, Layout::PLN3, Layout::PLN1}, {Roi::Full, Roi::Partial}),
        {WarpPerspectiveParams{{1, 0, 0, 0, 1, 0, 0, 0, 1}, NEAREST_NEIGHBOR, "identity"},
         WarpPerspectiveParams{{1, 0, 0, 0, 1, 0, 0, 0, 1}, BILINEAR, "identity"},
         WarpPerspectiveParams{{1, 0, 5, 0, 1, -3, 0, 0, 1}, NEAREST_NEIGHBOR, "shift"},
         WarpPerspectiveParams{{1, 0, 5, 0, 1, -3, 0, 0, 1}, BILINEAR, "shift"},
         WarpPerspectiveParams{{1, 0, 5.5f, 0, 1, -2.5f, 0, 0, 1}, BILINEAR, "halfshift"}})),
    op_config_name<WarpPerspectiveParams>);
