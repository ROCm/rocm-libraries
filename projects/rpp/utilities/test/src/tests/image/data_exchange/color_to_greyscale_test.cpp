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
#include "reference/color_to_greyscale_ref.hpp"

using namespace rpptest;

namespace {

// The source channel order the op is told to assume.
struct ColorToGreyscaleParams {
    RpptSubpixelLayout subpixel;
    std::string name() const {
        return subpixel == BGRtype ? "BGRtype" : "RGBtype";
    }
};

// A single 3-term dot product, so the only legitimate error is float-vs-double accumulation.
// The 1 LSB integer allowance also hides the kernels truncating where the golden rounds.
constexpr Tolerance kColorToGreyscaleTolerance = kRoundingTolerance;

template <typename T>
void run_color_to_greyscale(const TestConfig& cfg, const ColorToGreyscaleParams& op) {
    // 3-channel source, single-channel planar destination (PKD3/PLN3 -> PLN1 in the layout axis).
    RpptDesc srcDesc = make_src_descriptor(cfg);
    RpptDesc dstDesc = make_dst_descriptor(cfg);
    const std::size_t srcCount = element_count(srcDesc), dstCount = element_count(dstDesc);
    const std::size_t srcBytes = byte_size(srcDesc, cfg.dtype);
    const std::size_t dstBytes = byte_size(dstDesc, cfg.dtype);

    // The op takes no ROI; this one only drives the golden, so it stays host-side.
    const std::vector<RpptROI> roi = make_roi(srcDesc, cfg.roi);

    // dst and golden share a seed distinct from the input so a no-op kernel would be caught.
    std::vector<T> input(srcCount), dstInit(dstCount), golden(dstCount), actual(dstCount);
    fill_input<T>(input.data(), srcCount, cfg.dtype);
    fill_input<T>(dstInit.data(), dstCount, cfg.dtype, /*salt=*/1);
    golden = dstInit;
    color_to_greyscale_reference<T>(input.data(), srcDesc, golden.data(), dstDesc, cfg.dtype,
                                    roi.data(), XYWH, op.subpixel);

    DeviceTensor src(cfg.backend, srcBytes), dst(cfg.backend, dstBytes);
    src.write(input.data(), srcBytes);
    dst.write(dstInit.data(), dstBytes);

    RppHandle handle(cfg.backend, cfg.size.n);
    ASSERT_EQ(rppt_color_to_greyscale(src.ptr(), &srcDesc, dst.ptr(), &dstDesc, op.subpixel,
                                      handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), dstBytes);

    const std::vector<RpptROI> dstRoi = make_roi(dstDesc, Roi::Full);
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), dstDesc, dstRoi.data(), XYWH,
                               kColorToGreyscaleTolerance(cfg.dtype)));
}

}  // namespace

// Full name:
// Image_DataExchange/ColorToGreyscaleTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<Subpixel>
class ColorToGreyscaleTest : public SkipListTest<WithParams<ColorToGreyscaleParams>> {};

TEST_P(ColorToGreyscaleTest, Correctness) {
    const auto& p = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(p.cfg.dtype, [&](auto tag) {
        run_color_to_greyscale<Element<decltype(tag)>>(p.cfg, p.op);
    });
}

// The op takes no ROI argument (Roi::Full only). Both 3-channel sources produce a PLN1 destination.
INSTANTIATE_TEST_SUITE_P(
    Image_DataExchange, ColorToGreyscaleTest,
    ::testing::ValuesIn(with_params<ColorToGreyscaleParams>(
        make_configs(presets::kDefaultDTypes,
                     {{Layout::PKD3, Layout::PLN1}, {Layout::PLN3, Layout::PLN1}}, {Roi::Full},
                     {presets::kDefaultSize, presets::kTailWidthSize, presets::kSubVectorSize}),
        {ColorToGreyscaleParams{RGBtype}, ColorToGreyscaleParams{BGRtype}})),
    op_config_name<ColorToGreyscaleParams>);
