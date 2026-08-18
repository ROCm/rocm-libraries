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
#include "framework/tensor_setup.hpp"
#include "framework/tolerance.hpp"
#include "reference/pixelate_ref.hpp"

using namespace rpptest;

namespace {

// pixelationPercentage in [0, 100]: the fraction of resolution thrown away before the image is
// scaled back up, so a larger value means coarser blocks. 87.5 is the value the legacy harness
// exercises; 50 is a second point on the curve, so a block-size disagreement shows a different
// diff at each rather than one unexplained failure.
struct PixelateParams {
    float percentage;
    std::string name() const { return "p" + num_token(percentage); }
};

// pixelate is a bilinear downscale followed by a nearest-neighbour upscale. NN copies a texel
// verbatim, so all the numeric error comes from the single bilinear pass -- these are the resize
// bilinear tolerances. They are NOT loosened to cover the two defects this test surfaces -- the
// resize-family trailing-edge short read, and the intermediate downscale ignoring the ROI offset --
// so those cases stay red.
constexpr Tolerance kPixelateTolerance = kRoundingTolerance;

template <typename T>
void run_pixelate(const TestConfig& cfg, const PixelateParams& op) {
    const Rpp32u c = static_cast<Rpp32u>(channels_of(cfg.layout));
    const TensorShape shape{cfg.size.n, c, cfg.size.h, cfg.size.w};
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layout);
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < shape.n; ++i) roi[i] = roiVec[i];

    // (1) Host golden model. The op writes only the ROI-sized region at the destination origin, so
    // golden and the device buffer start from the same distinct pattern and only that region is
    // compared.
    std::vector<T> input(count), dstInit(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    fill_input<T>(dstInit.data(), count, cfg.dtype, /*salt=*/1);
    golden = dstInit;
    pixelate_reference<T>(input.data(), golden.data(), desc, cfg.dtype, roi.data(), XYWH,
                          op.percentage);

    // (2) Run RPP on the configured backend. pixelate needs a caller-provided scratch buffer for
    // its downscaled intermediate; the header requires n * strides.nStride * sizeof(Rpp32f) bytes.
    // It is seeded with a poison pattern rather than left uninitialized: the op writes the
    // intermediate itself before reading it back, so any poison reaching the output means the op
    // read scratch it never wrote -- a defect, and one whose error surface stays reproducible.
    const std::size_t scratchBytes =
        static_cast<std::size_t>(shape.n) * desc.strides.nStride * sizeof(Rpp32f);
    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    DeviceTensor scratch(cfg.backend, scratchBytes);
    const std::vector<Rpp8u> scratchPoison(scratchBytes, 0xA5);
    scratch.write(scratchPoison.data(), scratchBytes);
    src.write(input.data(), bytes);
    dst.write(dstInit.data(), bytes);

    RppHandle handle(cfg.backend, shape.n);
    ASSERT_EQ(rppt_pixelate(src.ptr(), &desc, dst.ptr(), &desc, scratch.ptr(), op.percentage,
                            roi.data(), XYWH, handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();
    dst.read(actual.data(), bytes);

    // (3) Compare the ROI-sized region written at the destination origin.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), desc, roi.data(), XYWH,
                               kPixelateTolerance(cfg.dtype)));
}

}  // namespace

// Full name:
// Image_Effects/PixelateTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_p<Pct>
class PixelateTest : public ::testing::TestWithParam<WithParams<PixelateParams>> {};

TEST_P(PixelateTest, Correctness) {
    const auto& p = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(p.cfg.dtype, [&](auto tag) {
        run_pixelate<Element<decltype(tag)>>(p.cfg, p.op);
    });
}

INSTANTIATE_TEST_SUITE_P(
    Image_Effects, PixelateTest,
    ::testing::ValuesIn(with_params<PixelateParams>(
        make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                     {Layout::PKD3, Layout::PLN3, Layout::PLN1}, {Roi::Full, Roi::Partial}),
        {PixelateParams{87.5f}, PixelateParams{50.0f}})),
    op_config_name<PixelateParams>);
