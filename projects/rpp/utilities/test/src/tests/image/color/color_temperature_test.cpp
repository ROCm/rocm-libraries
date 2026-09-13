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
#include "framework/compare.hpp"
#include "framework/config_param.hpp"
#include "framework/dtype.hpp"
#include "framework/skip_list.hpp"
#include "framework/tensor_setup.hpp"
#include "framework/tolerance.hpp"
#include "reference/color_temperature_ref.hpp"

using namespace rpptest;

namespace {

// adjustment is an integer pixel offset within the documented range -100 <= adjustment <= 100.
struct ColorTemperatureParams {
    int adjustment;
    std::string name() const {
        return "adj" + num_token(static_cast<float>(adjustment));
    }
};

// Pure integer additive offset: no rounding, so I8 is bit-exact too (the systemic
// I8 round-vs-truncate defect does not apply to an integer add).
constexpr Tolerance kColorTemperatureTolerance = tolerance(0.0, 2e-3, 5e-3);

template <typename T>
void run_color_temperature(const TestConfig& cfg, const ColorTemperatureParams& op) {
    RpptDesc srcDesc = make_src_descriptor(cfg);  // RPP takes a non-const ptr
    RpptDesc dstDesc = make_dst_descriptor(cfg);
    const std::size_t count = element_count(srcDesc);
    const std::size_t bytes = byte_size(srcDesc, cfg.dtype);

    // Parameters live in host-accessible (pinned for HIP) memory. adjustment is Rpp32s.
    PinnedArray<Rpp32s> adjustment(cfg.backend, cfg.size.n);
    PinnedArray<RpptROI> roi(cfg.backend, cfg.size.n);
    const std::vector<RpptROI> roiVec = make_roi(srcDesc, cfg.roi);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) {
        adjustment[i] = op.adjustment;
        roi[i] = roiVec[i];
    }

    // (1) Host golden model. golden starts as a copy of the input so the untouched
    // (outside-ROI) region is defined; only the ROI is overwritten by the reference.
    std::vector<T> input(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    golden = input;
    color_temperature_reference<T>(input.data(), srcDesc, golden.data(), dstDesc, cfg.dtype,
                                   roi.data(), XYWH, op.adjustment);

    // (2) Run RPP on the configured backend.
    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, cfg.size.n);
    ASSERT_EQ(rppt_color_temperature(src.ptr(), &srcDesc, dst.ptr(), &dstDesc, adjustment.data(),
                                     roi.data(), XYWH, handle.get(), cfg.backend),
              RPP_SUCCESS);

    // (3) Retrieve the result on the host (no-op copy for HOST, device->host for HIP).
    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    // (4) Compare within tolerance over the ROI.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), dstDesc, roi.data(), XYWH,
                               kColorTemperatureTolerance(cfg.dtype)));
}

}  // namespace

// color_temperature operates on RGB only (3 channels), so PLN1 is not instantiated.
// Full name:
// Image_Color/ColorTemperatureTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<Adj>
class ColorTemperatureTest : public SkipListTest<WithParams<ColorTemperatureParams>> {};

TEST_P(ColorTemperatureTest, Correctness) {
    const auto& p = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(
        p.cfg.dtype, [&](auto tag) { run_color_temperature<Element<decltype(tag)>>(p.cfg, p.op); });
}

// Same-layout cases plus both directions of the fused output-layout conversion.
INSTANTIATE_TEST_SUITE_P(Image_Color, ColorTemperatureTest,
                         ::testing::ValuesIn(with_params<ColorTemperatureParams>(
                             concat_configs({
                                 make_configs(presets::kDefaultDTypes, presets::kLayouts3ChConv,
                                              {Roi::Full, Roi::Partial}, {presets::kTailWidthSize}),
                                 make_configs(presets::kDefaultDTypes, presets::kLayouts3Ch,
                                              {Roi::Full, Roi::Partial},
                                              {presets::kDefaultSize, presets::kSubVectorSize}),
                             }),
                             {ColorTemperatureParams{40}})),
                         op_config_name<ColorTemperatureParams>);
