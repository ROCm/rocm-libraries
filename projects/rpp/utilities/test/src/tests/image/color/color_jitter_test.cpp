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

#include <algorithm>
#include <string>
#include <vector>

#include "framework/backend_memory.hpp"
#include "framework/compare_tensor.hpp"
#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/color_jitter_ref.hpp"

using namespace rpptest;

namespace {

// color_jitter's colour transform is built from four scalars, one value per axis shared by every
// pixel of an image. The neutral setting is brightness 0 (an additive translation), contrast 0 (the
// kernel scales by contrast + 1), hue 0 degrees and saturation 1 -- not the ranges the header
// documents, which belong to color_twist; see color_jitter_ref.hpp.
struct ColorJitterParams {
    float brightness, contrast, hue, saturation;
    std::string name() const {
        return "b" + num_token(brightness) + "_c" + num_token(contrast) + "_h" + num_token(hue) +
               "_s" + num_token(saturation);
    }
};

// The op is deterministic and pointwise -- one matrix-vector product per pixel -- so nothing here
// needs slack for accumulated error. The integer dtypes are bit-exact; the floats get only what the
// store itself costs, the golden computing in double and narrowing once where the kernel computes
// in float. A 1 LSB integer allowance would swallow the rounding defects this suite exists to
// catch, so the reds below are left to the goldens rather than absorbed here.
double color_jitter_tolerance(DType dt) {
    switch (dt) {
        case DType::U8:
        case DType::I8:
            return 0.0;
        case DType::F32:
            return 1e-6;
        case DType::F16:
            return 5e-4;  // one half-precision ulp near 1.0
        default:
            return 0.0;
    }
}

template <typename T>
void run_color_jitter(const TestConfig& cfg, const ColorJitterParams& op) {
    const TensorShape shape{cfg.size.n, static_cast<Rpp32u>(channels_of(cfg.layout)), cfg.size.h,
                            cfg.size.w};
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layout);  // RPP takes a non-const ptr
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    // Parameters live in host-accessible (pinned for HIP) memory.
    PinnedArray<Rpp32f> brightness(cfg.backend, shape.n);
    PinnedArray<Rpp32f> contrast(cfg.backend, shape.n);
    PinnedArray<Rpp32f> hue(cfg.backend, shape.n);
    PinnedArray<Rpp32f> saturation(cfg.backend, shape.n);
    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < shape.n; ++i) {
        brightness[i] = op.brightness;
        contrast[i] = op.contrast;
        hue[i] = op.hue;
        saturation[i] = op.saturation;
        roi[i] = roiVec[i];
    }

    // (1) Host golden model. Unlike most ops here the destination is pre-seeded with a *different*
    // pattern than the input rather than a copy of it. The identity parameter set makes the golden
    // equal the input, so seeding dst from the input would report a destination the op never
    // touched as a pass; a distinct pattern makes an unwritten output a mismatch instead. The
    // reference writes the whole destination region, so the seed only defines what lies outside it.
    std::vector<T> input(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    fill_input<T>(golden.data(), count, cfg.dtype, /*salt=*/23);
    const std::vector<T> dstSeed = golden;
    color_jitter_reference<T>(input.data(), golden.data(), desc, cfg.dtype, roi.data(), XYWH,
                              op.brightness, op.contrast, op.hue, op.saturation);

    // (2) Run RPP on the configured backend.
    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(dstSeed.data(), bytes);

    RppHandle handle(cfg.backend, shape.n);
    ASSERT_EQ(rppt_color_jitter(src.ptr(), &desc, dst.ptr(), &desc, brightness.data(),
                                contrast.data(), hue.data(), saturation.data(), roi.data(), XYWH,
                                handle.get(), cfg.backend),
              RPP_SUCCESS);

    // (3) Retrieve the result on the host (no-op copy for HOST, device->host for HIP).
    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    // (4) Compare within tolerance over the ROI.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), desc, roi.data(), XYWH,
                               color_jitter_tolerance(cfg.dtype)));
}

// rppt_color_jitter is documented as a HOST-backend op -- the header brief says so and every
// pointer parameter is specified as HOST memory -- and the HIP path returns
// RPP_ERROR_NOT_IMPLEMENTED (-6). The backend axis is therefore pinned rather than taken from
// available_backends().
std::vector<TestConfig> host_configs(std::vector<TestConfig> configs) {
    configs.erase(std::remove_if(configs.begin(), configs.end(),
                                 [](const TestConfig& c) { return c.backend != RPP_HOST_BACKEND; }),
                  configs.end());
    return configs;
}

}  // namespace

// Full name:
// Image_Color/ColorJitterTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<Params>
class ColorJitterTest : public ::testing::TestWithParam<WithParams<ColorJitterParams>> {};

TEST_P(ColorJitterTest, Correctness) {
    const auto& p = GetParam();
    switch (p.cfg.dtype) {
        case DType::U8:
            run_color_jitter<Rpp8u>(p.cfg, p.op);
            break;
        case DType::F16:
            run_color_jitter<Rpp16f>(p.cfg, p.op);
            break;
        case DType::F32:
            run_color_jitter<Rpp32f>(p.cfg, p.op);
            break;
        case DType::I8:
            run_color_jitter<Rpp8s>(p.cfg, p.op);
            break;
        default:
            FAIL() << "unsupported dtype for color_jitter";
    }
}

// One parameter set per axis rather than a combined one, so a failure names the axis that broke
// instead of blending four of them. In order: the neutral setting, which must be an exact identity;
// full desaturation, which must collapse every pixel to its luma grey; a pure hue rotation; a pure
// contrast scale; and a pure brightness translation.
//
// PLN1 is in the grid because 1-channel input is a documented shape that rppt_color_jitter accepts
// -- it returns RPP_SUCCESS (unlike rppt_color_twist, which rejects it) without writing anything.
INSTANTIATE_TEST_SUITE_P(
    Image_Color, ColorJitterTest,
    ::testing::ValuesIn(with_params<ColorJitterParams>(
        host_configs(make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                                  {Layout::PKD3, Layout::PLN3, Layout::PLN1},
                                  {Roi::Full, Roi::Partial})),
        {ColorJitterParams{0.0f, 0.0f, 0.0f, 1.0f},     // neutral
         ColorJitterParams{0.0f, 0.0f, 0.0f, 0.0f},     // desaturate to grey
         ColorJitterParams{0.0f, 0.0f, 90.0f, 1.0f},    // hue rotation
         ColorJitterParams{0.0f, 0.25f, 0.0f, 1.0f},    // contrast scale
         ColorJitterParams{0.25f, 0.0f, 0.0f, 1.0f}})),  // brightness translation
    op_config_name<ColorJitterParams>);
