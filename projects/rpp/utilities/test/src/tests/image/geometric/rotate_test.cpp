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
#include "framework/compare.hpp"
#include "framework/config_param.hpp"
#include "framework/dtype.hpp"
#include "framework/skip_list.hpp"
#include "framework/tensor_setup.hpp"
#include "framework/tolerance.hpp"
#include "reference/rotate_ref.hpp"

using namespace rpptest;

namespace {

std::string interp_token(RpptInterpolationType i) {
    return i == NEAREST_NEIGHBOR ? "NN" : "BILINEAR";
}

// Angle in degrees, positive = anticlockwise.
struct RotateParams {
    float angle;
    RpptInterpolationType interp;
    std::string name() const {
        return "a" + num_token(angle) + "_" + interp_token(interp);
    }
};

// NN copies texels verbatim; bilinear allows rounding error only. Not loosened for the known kernel
// defects: HOST bilinear reads the last column/row one texel short, I8 out-of-bounds fill is 0
// rather than -128, and partial ROIs ignore the ROI origin (with HIP placement also diverging).
double rotate_tolerance(DType dt, RpptInterpolationType interp) {
    if (interp == NEAREST_NEIGHBOR) return 0.0;
    // cos/sin of 90/270 deg are not exactly 0 in float, leaving a sub-LSB blend.
    return kRoundingTolerance(dt);
}

template <typename T>
void run_rotate(const TestConfig& cfg, const RotateParams& op) {
    const Rpp32u c = static_cast<Rpp32u>(channels_of(cfg.layoutIn));
    const TensorShape shape{cfg.size.n, c, cfg.size.h, cfg.size.w};
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layoutIn);  // RPP takes a non-const ptr
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    PinnedArray<Rpp32f> angle(cfg.backend, shape.n);
    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < shape.n; ++i) {
        angle[i] = op.angle;
        roi[i] = roiVec[i];
    }

    std::vector<T> input(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    golden = input;
    rotate_reference<T>(input.data(), golden.data(), desc, cfg.dtype, roi.data(), XYWH,
                        angle.data(), op.interp);

    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, shape.n);
    ASSERT_EQ(rppt_rotate(src.ptr(), &desc, dst.ptr(), &desc, angle.data(), op.interp, roi.data(),
                          XYWH, handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    // Uses the test's own ROI copy: HIP rewrites the op's ROI to LTRB in place.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), desc, roiVec.data(), XYWH,
                               rotate_tolerance(cfg.dtype, op.interp)));
}

}  // namespace

// Full name:
// Image_Geometric/RotateTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<Angle>_<Interp>
class RotateTest : public SkipListTest<WithParams<RotateParams>> {};

TEST_P(RotateTest, Correctness) {
    const auto& p = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(
        p.cfg.dtype, [&](auto tag) { run_rotate<Element<decltype(tag)>>(p.cfg, p.op); });
}

// Cardinal angles map to integer source coordinates; 45 lands samples between source pixels and
// sends the corners off the frame.
INSTANTIATE_TEST_SUITE_P(
    Image_Geometric, RotateTest,
    ::testing::ValuesIn(with_params<RotateParams>(
        make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                     {Layout::PKD3, Layout::PLN3, Layout::PLN1}, {Roi::Full, Roi::Partial},
                     {presets::kDefaultSize, presets::kTailWidthSize}),
        {RotateParams{0.0f, NEAREST_NEIGHBOR}, RotateParams{0.0f, BILINEAR},
         RotateParams{90.0f, NEAREST_NEIGHBOR}, RotateParams{90.0f, BILINEAR},
         RotateParams{180.0f, NEAREST_NEIGHBOR}, RotateParams{180.0f, BILINEAR},
         RotateParams{270.0f, NEAREST_NEIGHBOR}, RotateParams{270.0f, BILINEAR},
         RotateParams{45.0f, NEAREST_NEIGHBOR}, RotateParams{45.0f, BILINEAR}})),
    op_config_name<RotateParams>);
