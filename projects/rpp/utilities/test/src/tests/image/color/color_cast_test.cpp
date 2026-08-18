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

#include <vector>

#include "framework/backend_memory.hpp"
#include "framework/compare_tensor.hpp"
#include "framework/config_param.hpp"
#include "framework/dtype_dispatch.hpp"
#include "framework/skip_list.hpp"
#include "framework/tensor_setup.hpp"
#include "framework/tolerance.hpp"
#include "reference/color_cast_ref.hpp"

using namespace rpptest;

namespace {

// alpha within the documented range alpha >= 0; R/G/B cast constants within 0 <= rgb <= 255.
// alpha is chosen not exactly representable so blended results never land on an X.5 boundary,
// keeping the I8 diff purely truncation rather than a rounding-mode ambiguity.
struct ColorCastParams {
    float alpha;
    Rpp8u r, g, b;
    std::string name() const {
        return "a" + num_token(alpha) + "_r" + num_token(r) + "_g" + num_token(g) + "_b" +
               num_token(b);
    }
};

// I8 kept sub-LSB to surface the systemic I8 round-vs-truncate defect (RPP truncates the
// I8 result instead of rounding).
constexpr Tolerance kColorCastTolerance = tolerance(1.0, 2e-3, 5e-3).with_i8(0.5);

template <typename T>
void run_color_cast(const TestConfig& cfg, const ColorCastParams& op) {
    const TensorShape shape{cfg.size.n, static_cast<Rpp32u>(channels_of(cfg.layout)), cfg.size.h,
                            cfg.size.w};
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layout);  // RPP takes a non-const ptr
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    // Parameters live in host-accessible (pinned for HIP) memory.
    PinnedArray<RpptRGB> rgb(cfg.backend, shape.n);
    PinnedArray<Rpp32f> alpha(cfg.backend, shape.n);
    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < shape.n; ++i) {
        rgb[i] = RpptRGB{op.r, op.g, op.b};
        alpha[i] = op.alpha;
        roi[i] = roiVec[i];
    }
    const double rgbD[3] = {static_cast<double>(op.r), static_cast<double>(op.g),
                            static_cast<double>(op.b)};

    // (1) Host golden model. golden starts as a copy of the input so the untouched
    // (outside-ROI) region is defined; only the ROI is overwritten by the reference.
    std::vector<T> input(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    golden = input;
    color_cast_reference<T>(input.data(), golden.data(), desc, cfg.dtype, roi.data(), XYWH,
                            op.alpha, rgbD);

    // (2) Run RPP on the configured backend.
    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, shape.n);
    ASSERT_EQ(rppt_color_cast(src.ptr(), &desc, dst.ptr(), &desc, rgb.data(), alpha.data(),
                              roi.data(), XYWH, handle.get(), cfg.backend),
              RPP_SUCCESS);

    // (3) Retrieve the result on the host (no-op copy for HOST, device->host for HIP).
    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    // (4) Compare within tolerance over the ROI.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), desc, roi.data(), XYWH,
                               kColorCastTolerance(cfg.dtype)));
}

}  // namespace

// Full name:
// Image_Color/ColorCastTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<Alpha>_<RGB>
class ColorCastTest : public SkipListTest<WithParams<ColorCastParams>> {};

TEST_P(ColorCastTest, Correctness) {
    const auto& p = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(p.cfg.dtype, [&](auto tag) {
        run_color_cast<Element<decltype(tag)>>(p.cfg, p.op);
    });
}

// Restricted to the 3-channel layouts: color_cast applies distinct R/G/B constants, and the
// kernel rejects 1-channel input with an error despite the header documenting c = 1/3 (a
// doc/kernel discrepancy).
INSTANTIATE_TEST_SUITE_P(Image_Color, ColorCastTest,
                         ::testing::ValuesIn(with_params<ColorCastParams>(
                             make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                                          {Layout::PKD3, Layout::PLN3, Layout::PLN1},
                                          {Roi::Full, Roi::Partial}),
                             {ColorCastParams{0.6f, 30, 90, 150}})),
                         op_config_name<ColorCastParams>);
