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
#include "framework/compare.hpp"
#include "framework/config_param.hpp"
#include "framework/dtype.hpp"
#include "framework/skip_list.hpp"
#include "framework/tensor_setup.hpp"
#include "framework/tolerance.hpp"
#include "reference/warp_affine_ref.hpp"

using namespace rpptest;

namespace {

std::string interp_token(RpptInterpolationType i) {
    return i == NEAREST_NEIGHBOR ? "NN" : "BILINEAR";
}

// A 2x3 affine matrix (forward source->destination) plus the interpolation to sample with. The
// identity and pure-translation matrices are invariant to the mapping-origin convention, so they
// pin the direction unambiguously; scale2 and rot30 have a non-identity linear part and are the
// cases that exercise the centre the matrix acts about (see warp_affine_ref).
struct WarpAffineParams {
    std::array<float, 6> m;
    RpptInterpolationType interp;
    std::string tag;
    std::string name() const {
        return tag + "_" + interp_token(interp);
    }
};

// Tolerances are set from legitimate numeric error only; they are deliberately NOT loosened to
// hide the real warp_affine kernel defects this test surfaces:
//   #15 HOST bilinear reads the last column/row one texel short (all HOST_*_BILINEAR red)
//   #16 I8 out-of-bounds fill is 0, not black -128 (I8 shift/halfshift red on both backends)
//   #17 HIP partial-ROI placement diverges from HOST (all HIP_*_PartialRoi red)
double warp_affine_tolerance(DType dt, RpptInterpolationType interp) {
    // Nearest-neighbour copies a texel verbatim, so it is bit-exact for every dtype.
    if (interp == NEAREST_NEIGHBOR) return 0.0;
    // Bilinear blends in float; allow only genuine rounding error. Integer types round to nearest.
    return kRoundingTolerance(dt);
}

template <typename T>
void run_warp_affine(const TestConfig& cfg, const WarpAffineParams& op) {
    RpptDesc srcDesc = make_src_descriptor(cfg);  // RPP takes a non-const ptr
    RpptDesc dstDesc = make_dst_descriptor(cfg);
    const std::size_t count = element_count(srcDesc);
    const std::size_t bytes = byte_size(srcDesc, cfg.dtype);

    // Per-image affine matrices (6 each) and ROIs in host-accessible (pinned for HIP) memory.
    PinnedArray<Rpp32f> affine(cfg.backend, cfg.size.n * 6);
    PinnedArray<RpptROI> roi(cfg.backend, cfg.size.n);
    const std::vector<RpptROI> roiVec = make_roi(srcDesc, cfg.roi);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) {
        for (int k = 0; k < 6; ++k) affine[i * 6 + k] = op.m[k];
        roi[i] = roiVec[i];
    }

    // (1) Host golden model. golden starts as a copy of the input so the untouched
    // (outside-ROI) region is defined; only the ROI-sized output region is overwritten.
    std::vector<T> input(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    golden = input;
    warp_affine_reference<T>(input.data(), srcDesc, golden.data(), dstDesc, cfg.dtype, roi.data(),
                             XYWH, affine.data(), op.interp);

    // (2) Run RPP on the configured backend.
    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, cfg.size.n);
    ASSERT_EQ(rppt_warp_affine(src.ptr(), &srcDesc, dst.ptr(), &dstDesc, affine.data(), op.interp,
                               roi.data(), XYWH, handle.get(), cfg.backend),
              RPP_SUCCESS);

    // (3) Retrieve the result on the host (no-op copy for HOST, device->host for HIP).
    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    // (4) Compare within tolerance over the ROI-sized output region at the destination origin.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), dstDesc, roi.data(), XYWH,
                               warp_affine_tolerance(cfg.dtype, op.interp)));
}

}  // namespace

// Full name:
// Image_Geometric/WarpAffineTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<Matrix>_<Interp>
class WarpAffineTest : public SkipListTest<WithParams<WarpAffineParams>> {};

TEST_P(WarpAffineTest, Correctness) {
    const auto& p = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(
        p.cfg.dtype, [&](auto tag) { run_warp_affine<Element<decltype(tag)>>(p.cfg, p.op); });
}

// identity: output == source. shift: integer translation (exercises the border fill). halfshift:
// half-pixel translation, bilinear only, exercises the 4-tap blend deterministically.
INSTANTIATE_TEST_SUITE_P(
    Image_Geometric, WarpAffineTest,
    ::testing::ValuesIn(with_params<WarpAffineParams>(
        make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                     {{Layout::PKD3, Layout::PKD3},
                      {Layout::PLN3, Layout::PLN3},
                      {Layout::PLN1, Layout::PLN1},
                      {Layout::PKD3, Layout::PLN3},
                      {Layout::PLN3, Layout::PKD3}},
                     {Roi::Full, Roi::Partial}, {presets::kDefaultSize, presets::kTailWidthSize}),
        {WarpAffineParams{{1, 0, 0, 0, 1, 0}, NEAREST_NEIGHBOR, "identity"},
         WarpAffineParams{{1, 0, 0, 0, 1, 0}, BILINEAR, "identity"},
         WarpAffineParams{{1, 0, 5, 0, 1, -3}, NEAREST_NEIGHBOR, "shift"},
         WarpAffineParams{{1, 0, 5, 0, 1, -3}, BILINEAR, "shift"},
         WarpAffineParams{{1, 0, 5.5f, 0, 1, -2.5f}, BILINEAR, "halfshift"},
         WarpAffineParams{{2, 0, 0, 0, 2, 0}, NEAREST_NEIGHBOR, "scale2"},
         WarpAffineParams{{2, 0, 0, 0, 2, 0}, BILINEAR, "scale2"},
         WarpAffineParams{{0.8660254f, -0.5f, 0, 0.5f, 0.8660254f, 0}, NEAREST_NEIGHBOR, "rot30"},
         WarpAffineParams{{0.8660254f, -0.5f, 0, 0.5f, 0.8660254f, 0}, BILINEAR, "rot30"}})),
    op_config_name<WarpAffineParams>);
