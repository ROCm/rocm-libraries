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
#include "framework/dtype_dispatch.hpp"
#include "framework/skip_list.hpp"
#include "framework/tensor_setup.hpp"
#include "framework/tolerance.hpp"
#include "reference/color_to_greyscale_ref.hpp"

using namespace rpptest;

namespace {

// The source channel order the op is told to assume; both values are exercised.
struct ColorToGreyscaleParams {
    RpptSubpixelLayout subpixel;
    std::string name() const { return subpixel == BGRtype ? "BGRtype" : "RGBtype"; }
};

// A single 3-term dot product, so the only legitimate error is float-vs-double accumulation.
// Note this leaves the systemic integer round-vs-truncate difference (the kernels truncate where
// the goldens round) unsurfaced at <= 1 LSB; the integer tolerances are not loosened past 1 LSB.
constexpr Tolerance kColorToGreyscaleTolerance = kRoundingTolerance;

template <typename T>
void run_color_to_greyscale(const TestConfig& cfg, const ColorToGreyscaleParams& op) {
    // The op requires a 3-channel source (PKD3/PLN3) and produces a single-channel planar output,
    // so the two operands need separate descriptors.
    const TensorShape srcShape{cfg.size.n, static_cast<Rpp32u>(channels_of(cfg.layout)), cfg.size.h,
                               cfg.size.w};
    const TensorShape dstShape{cfg.size.n, 1, cfg.size.h, cfg.size.w};
    RpptDesc srcDesc = make_descriptor(srcShape, cfg.dtype, cfg.layout);
    RpptDesc dstDesc = make_descriptor(dstShape, cfg.dtype, Layout::PLN1);
    const std::size_t srcCount = element_count(srcDesc), dstCount = element_count(dstDesc);
    const std::size_t srcBytes = byte_size(srcDesc, cfg.dtype);
    const std::size_t dstBytes = byte_size(dstDesc, cfg.dtype);

    // color_to_greyscale takes no ROI argument; the full-frame ROI exists only to drive the
    // golden's traversal (Roi::Full only), so it never has to be device-visible.
    const std::vector<RpptROI> roi = make_roi(srcDesc, cfg.roi);

    // (1) Host golden model. golden starts as the destination's own pattern so the untouched
    // (row-padding) slack is defined; only the frame is overwritten by the reference.
    std::vector<T> input(srcCount), dstInit(dstCount), golden(dstCount), actual(dstCount);
    fill_input<T>(input.data(), srcCount, cfg.dtype);
    fill_input<T>(dstInit.data(), dstCount, cfg.dtype, /*salt=*/1);
    golden = dstInit;
    color_to_greyscale_reference<T>(input.data(), srcDesc, golden.data(), dstDesc, cfg.dtype,
                                    roi.data(), XYWH, op.subpixel);

    // (2) Run RPP on the configured backend. dst is pre-filled with a distinct pattern so a
    // no-op kernel would be caught.
    DeviceTensor src(cfg.backend, srcBytes), dst(cfg.backend, dstBytes);
    src.write(input.data(), srcBytes);
    dst.write(dstInit.data(), dstBytes);

    RppHandle handle(cfg.backend, srcShape.n);
    ASSERT_EQ(rppt_color_to_greyscale(src.ptr(), &srcDesc, dst.ptr(), &dstDesc, op.subpixel,
                                      handle.get(), cfg.backend),
              RPP_SUCCESS);

    // (3) Retrieve the result on the host (no-op copy for HOST, device->host for HIP).
    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), dstBytes);

    // (4) Compare over the single-channel destination frame.
    const std::vector<RpptROI> dstRoi = make_roi(dstDesc, Roi::Full);
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), dstDesc, dstRoi.data(), XYWH,
                               kColorToGreyscaleTolerance(cfg.dtype)));
}

}  // namespace

// Full name: Image_DataExchange/ColorToGreyscaleTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<Subpixel>
class ColorToGreyscaleTest : public SkipListTest<WithParams<ColorToGreyscaleParams>> {};

TEST_P(ColorToGreyscaleTest, Correctness) {
    const auto& p = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(p.cfg.dtype, [&](auto tag) {
        run_color_to_greyscale<Element<decltype(tag)>>(p.cfg, p.op);
    });
}

// The source must be 3-channel (no PLN1) and the op takes no ROI argument (Roi::Full only).
INSTANTIATE_TEST_SUITE_P(
    Image_DataExchange, ColorToGreyscaleTest,
    ::testing::ValuesIn(with_params<ColorToGreyscaleParams>(
        make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                     {Layout::PKD3, Layout::PLN3}, {Roi::Full}),
        {ColorToGreyscaleParams{RGBtype}, ColorToGreyscaleParams{BGRtype}})),
    op_config_name<ColorToGreyscaleParams>);
