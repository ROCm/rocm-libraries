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
#include "reference/pixelate_ref.hpp"

using namespace rpptest;

namespace {

// pixelationPercentage in [0, 100]: larger means coarser blocks.
struct PixelateParams {
    float percentage;
    std::string name() const {
        return "p" + num_token(percentage);
    }
};

// Bilinear downscale then nearest-neighbour upscale, so all error comes from the bilinear pass. Not
// loosened to cover the kernel's trailing-edge short read or its downscale ignoring the ROI offset.
constexpr Tolerance kPixelateTolerance = kRoundingTolerance;

template <typename T>
void run_pixelate(const TestConfig& cfg, const PixelateParams& op) {
    const Rpp32u c = static_cast<Rpp32u>(channels_of(cfg.layoutIn));
    const TensorShape shape{cfg.size.n, c, cfg.size.h, cfg.size.w};
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layoutIn);
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < shape.n; ++i) roi[i] = roiVec[i];

    // The op writes the ROI-sized region at the destination origin; only that region is compared.
    std::vector<T> input(count), dstInit(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    fill_input<T>(dstInit.data(), count, cfg.dtype, /*salt=*/1);
    golden = dstInit;
    pixelate_reference<T>(input.data(), golden.data(), desc, cfg.dtype, roi.data(), XYWH,
                          op.percentage);

    // Scratch for the downscaled intermediate (n * nStride * sizeof(Rpp32f) bytes per the header),
    // poisoned so a read of unwritten scratch is reproducible.
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

    // Uses the test's own ROI copy: the HIP resize path rewrites the op's ROI to LTRB in place.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), desc, roiVec.data(), XYWH,
                               kPixelateTolerance(cfg.dtype)));
}

}  // namespace

// Full name:
// Image_Effects/PixelateTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_p<Pct>
class PixelateTest : public SkipListTest<WithParams<PixelateParams>> {};

TEST_P(PixelateTest, Correctness) {
    const auto& p = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(
        p.cfg.dtype, [&](auto tag) { run_pixelate<Element<decltype(tag)>>(p.cfg, p.op); });
}

INSTANTIATE_TEST_SUITE_P(Image_Effects, PixelateTest,
                         ::testing::ValuesIn(with_params<PixelateParams>(
                             make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                                          {Layout::PKD3, Layout::PLN3, Layout::PLN1},
                                          {Roi::Full, Roi::Partial},
                                          {presets::kDefaultSize, presets::kTailWidthSize}),
                             {PixelateParams{87.5f}, PixelateParams{50.0f}})),
                         op_config_name<PixelateParams>);
