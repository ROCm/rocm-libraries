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
#include "reference/brightness_ref.hpp"

using namespace rpptest;

namespace {

// alpha within the documented range 0 <= alpha <= 20; beta within 0 <= beta <= 255.
struct BrightnessParams {
    float alpha, beta;
    std::string name() const {
        return "a" + num_token(alpha) + "_b" + num_token(beta);
    }
};

template <typename T>
void run_brightness(const TestConfig& cfg, const BrightnessParams& op) {
    // Non-const because RPP takes a non-const pointer. The two differ only for a layout-converting
    // config; brightness keeps the channel count either way, so both span the same element count.
    RpptDesc srcDesc = make_src_descriptor(cfg);
    RpptDesc dstDesc = make_dst_descriptor(cfg);
    const std::size_t count = element_count(srcDesc);
    const std::size_t bytes = byte_size(srcDesc, cfg.dtype);

    // Parameters live in host-accessible (pinned for HIP) memory.
    PinnedArray<Rpp32f> alpha(cfg.backend, cfg.size.n);
    PinnedArray<Rpp32f> beta(cfg.backend, cfg.size.n);
    PinnedArray<RpptROI> roi(cfg.backend, cfg.size.n);
    const std::vector<RpptROI> roiVec = make_roi(srcDesc, cfg.roi);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) {
        alpha[i] = op.alpha;
        beta[i] = op.beta;
        roi[i] = roiVec[i];
    }

    // (1) Host golden model. golden and the device destination are seeded identically so the
    // region the op leaves untouched (outside the ROI) is defined and compares equal; only the
    // ROI is overwritten, by the reference here and by the op below.
    std::vector<T> input(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    golden = input;
    brightness_reference<T>(input.data(), srcDesc, golden.data(), dstDesc, cfg.dtype, roi.data(),
                            XYWH, op.alpha, op.beta);

    // (2) Run RPP on the configured backend.
    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, cfg.size.n);
    ASSERT_EQ(rppt_brightness(src.ptr(), &srcDesc, dst.ptr(), &dstDesc, alpha.data(), beta.data(),
                              roi.data(), XYWH, handle.get(), cfg.backend),
              RPP_SUCCESS);

    // (3) Retrieve the result on the host (no-op copy for HOST, device->host for HIP).
    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    // (4) Compare within tolerance over the ROI, addressed through the destination descriptor.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), dstDesc, roi.data(), XYWH,
                               kRoundingTolerance(cfg.dtype)));
}

}  // namespace

// Full name:
// Image_Color/BrightnessTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<Alpha>_<Beta>
class BrightnessTest : public SkipListTest<WithParams<BrightnessParams>> {};

TEST_P(BrightnessTest, Correctness) {
    const auto& p = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(
        p.cfg.dtype, [&](auto tag) { run_brightness<Element<decltype(tag)>>(p.cfg, p.op); });
}

INSTANTIATE_TEST_SUITE_P(
    Image_Color, BrightnessTest,
    ::testing::ValuesIn(with_params<BrightnessParams>(
        concat_configs({
            make_configs(presets::kDefaultDTypes, presets::kLayoutsFullConv,
                         {Roi::Full, Roi::Partial}, {presets::kTailWidthSize}),
            make_configs(presets::kDefaultDTypes, presets::kLayoutsFull, {Roi::Full, Roi::Partial},
                         {presets::kDefaultSize, presets::kSubVectorSize}),
        }),
        {BrightnessParams{1.75f, 50.0f}})),
    op_config_name<BrightnessParams>);
