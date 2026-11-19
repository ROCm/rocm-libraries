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
#include "framework/compare.hpp"
#include "framework/config_param.hpp"
#include "framework/dtype.hpp"
#include "framework/skip_list.hpp"
#include "framework/tensor_setup.hpp"
#include "framework/tolerance.hpp"
#include "reference/color_jitter_ref.hpp"

using namespace rpptest;

namespace {

// Neutral setting is brightness 0, contrast 0 (the kernel scales by contrast + 1), hue 0 and
// saturation 1; the header's documented ranges belong to color_twist (see color_jitter_ref.hpp).
struct ColorJitterParams {
    float brightness, contrast, hue, saturation;
    std::string name() const {
        return "b" + num_token(brightness) + "_c" + num_token(contrast) + "_h" + num_token(hue) +
               "_s" + num_token(saturation);
    }
};

// One matrix-vector product per pixel: integers are bit-exact, floats allow only the store
// rounding (F16 one half-precision ulp near 1.0).
constexpr Tolerance kColorJitterTolerance = tolerance(0.0, 1e-6, 5e-4);

template <typename T>
void run_color_jitter(const TestConfig& cfg, const ColorJitterParams& op) {
    RpptDesc srcDesc = make_src_descriptor(cfg);  // RPP takes a non-const ptr
    RpptDesc dstDesc = make_dst_descriptor(cfg);
    const std::size_t count = element_count(srcDesc);
    const std::size_t bytes = byte_size(srcDesc, cfg.dtype);

    PinnedArray<Rpp32f> brightness(cfg.backend, cfg.size.n);
    PinnedArray<Rpp32f> contrast(cfg.backend, cfg.size.n);
    PinnedArray<Rpp32f> hue(cfg.backend, cfg.size.n);
    PinnedArray<Rpp32f> saturation(cfg.backend, cfg.size.n);
    PinnedArray<RpptROI> roi(cfg.backend, cfg.size.n);
    const std::vector<RpptROI> roiVec = make_roi(srcDesc, cfg.roi);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) {
        brightness[i] = op.brightness;
        contrast[i] = op.contrast;
        hue[i] = op.hue;
        saturation[i] = op.saturation;
        roi[i] = roiVec[i];
    }

    // dst is seeded with a pattern distinct from the input: the neutral parameter set makes the
    // golden equal the input, so an input-seeded dst would pass even if the op wrote nothing.
    std::vector<T> input(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    fill_input<T>(golden.data(), count, cfg.dtype, /*salt=*/23);
    const std::vector<T> dstSeed = golden;
    color_jitter_reference<T>(input.data(), srcDesc, golden.data(), dstDesc, cfg.dtype, roi.data(),
                              XYWH, op.brightness, op.contrast, op.hue, op.saturation);

    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(dstSeed.data(), bytes);

    RppHandle handle(cfg.backend, cfg.size.n);
    ASSERT_EQ(rppt_color_jitter(src.ptr(), &srcDesc, dst.ptr(), &dstDesc, brightness.data(),
                                contrast.data(), hue.data(), saturation.data(), roi.data(), XYWH,
                                handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), dstDesc, roi.data(), XYWH,
                               kColorJitterTolerance(cfg.dtype)));
}

// rppt_color_jitter is documented as HOST-only and the HIP path returns RPP_ERROR_NOT_IMPLEMENTED,
// so the backend axis is pinned to HOST.
std::vector<TestConfig> host_configs(std::vector<TestConfig> configs) {
    configs.erase(std::remove_if(configs.begin(), configs.end(),
                                 [](const TestConfig& c) { return c.backend != RPP_HOST_BACKEND; }),
                  configs.end());
    return configs;
}

}  // namespace

// Full name:
// Image_Color/ColorJitterTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<Params>
class ColorJitterTest : public SkipListTest<WithParams<ColorJitterParams>> {};

TEST_P(ColorJitterTest, Correctness) {
    const auto& p = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(
        p.cfg.dtype, [&](auto tag) { run_color_jitter<Element<decltype(tag)>>(p.cfg, p.op); });
}

// One parameter set per axis so a failure names the axis that broke. PLN1 is included because the
// op accepts 1-channel input (returns RPP_SUCCESS without writing anything).
INSTANTIATE_TEST_SUITE_P(
    Image_Color, ColorJitterTest,
    ::testing::ValuesIn(with_params<ColorJitterParams>(
        host_configs(concat_configs({
            make_configs(presets::kDefaultDTypes, presets::kLayoutsFullConv,
                         {Roi::Full, Roi::Partial}, {presets::kTailWidthSize}),
            make_configs(presets::kDefaultDTypes, presets::kLayoutsFull, {Roi::Full, Roi::Partial},
                         {presets::kDefaultSize, presets::kSubVectorSize}),
        })),
        {ColorJitterParams{0.0f, 0.0f, 0.0f, 1.0f},      // neutral
         ColorJitterParams{0.0f, 0.0f, 0.0f, 0.0f},      // desaturate to grey
         ColorJitterParams{0.0f, 0.0f, 90.0f, 1.0f},     // hue rotation
         ColorJitterParams{0.0f, 0.25f, 0.0f, 1.0f},     // contrast scale
         ColorJitterParams{0.25f, 0.0f, 0.0f, 1.0f}})),  // brightness translation
    op_config_name<ColorJitterParams>);
