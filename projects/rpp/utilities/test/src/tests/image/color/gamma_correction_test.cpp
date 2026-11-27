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
#include "reference/gamma_correction_ref.hpp"

using namespace rpptest;

namespace {

// gamma within the documented range gamma >= 0.
struct GammaCorrectionParams {
    float gamma;
    std::string name() const {
        return "g" + num_token(gamma);
    }
};

template <typename T>
void run_gamma_correction(const TestConfig& cfg, const GammaCorrectionParams& op) {
    RpptDesc srcDesc = make_src_descriptor(cfg);  // RPP takes a non-const ptr
    RpptDesc dstDesc = make_dst_descriptor(cfg);
    const std::size_t count = element_count(srcDesc);
    const std::size_t bytes = byte_size(srcDesc, cfg.dtype);

    PinnedArray<Rpp32f> gamma(cfg.backend, cfg.size.n);
    PinnedArray<RpptROI> roi(cfg.backend, cfg.size.n);
    const std::vector<RpptROI> roiVec = make_roi(srcDesc, cfg.roi);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) {
        gamma[i] = op.gamma;
        roi[i] = roiVec[i];
    }

    std::vector<T> input(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    golden = input;
    gamma_correction_reference<T>(input.data(), srcDesc, golden.data(), dstDesc, cfg.dtype,
                                  roi.data(), XYWH, op.gamma);

    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, cfg.size.n);
    ASSERT_EQ(rppt_gamma_correction(src.ptr(), &srcDesc, dst.ptr(), &dstDesc, gamma.data(),
                                    roi.data(), XYWH, handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), dstDesc, roi.data(), XYWH,
                               kRoundingTolerance(cfg.dtype)));
}

// Every other case in this suite creates the handle with the descriptor's own
// batch, so the two can never disagree and a kernel that takes the batch from
// the handle instead of the descriptor still looks correct. This case sizes the
// handle larger than the descriptor on purpose: a grid built from the handle
// then spans threads with no image behind them, while a grid built from the
// descriptor is unaffected. The expected values come from the same reference
// model, so the comparison stays the ordinary full-ROI one.
template <typename T>
void run_gamma_correction_mismatched_handle_batch(const TestConfig& cfg,
                                                  const GammaCorrectionParams& op) {
    RpptDesc srcDesc = make_src_descriptor(cfg);
    RpptDesc dstDesc = make_dst_descriptor(cfg);
    const std::size_t count = element_count(srcDesc);
    const std::size_t bytes = byte_size(srcDesc, cfg.dtype);

    PinnedArray<Rpp32f> gamma(cfg.backend, cfg.size.n);
    PinnedArray<RpptROI> roi(cfg.backend, cfg.size.n);
    const std::vector<RpptROI> roiVec = make_roi(srcDesc, cfg.roi);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) {
        gamma[i] = op.gamma;
        roi[i] = roiVec[i];
    }

    std::vector<T> input(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    golden = input;
    gamma_correction_reference<T>(input.data(), srcDesc, golden.data(), dstDesc, cfg.dtype,
                                  roi.data(), XYWH, op.gamma);

    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);  // define outside-ROI dst to mirror the golden

    // The handle batch is larger than the descriptor batch on purpose.
    constexpr Rpp32u kHandleBatchSlack = 4;
    RppHandle handle(cfg.backend, cfg.size.n + kHandleBatchSlack);
    ASSERT_EQ(rppt_gamma_correction(src.ptr(), &srcDesc, dst.ptr(), &dstDesc, gamma.data(),
                                    roi.data(), XYWH, handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), dstDesc, roi.data(), XYWH,
                               kRoundingTolerance(cfg.dtype)));
}

}  // namespace

// Full name:
// Image_Color/GammaCorrectionTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<Gamma>
class GammaCorrectionTest : public SkipListTest<WithParams<GammaCorrectionParams>> {};

TEST_P(GammaCorrectionTest, Correctness) {
    const auto& p = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(
        p.cfg.dtype, [&](auto tag) { run_gamma_correction<Element<decltype(tag)>>(p.cfg, p.op); });
}

// The handle and the descriptor carry different batch sizes here, which no other
// case in the suite does. See the helper's comment for why that matters.
class GammaCorrectionMismatchedHandleBatchTest
    : public SkipListTest<WithParams<GammaCorrectionParams>> {};

TEST_P(GammaCorrectionMismatchedHandleBatchTest, Correctness) {
    const auto& p = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(p.cfg.dtype, [&](auto tag) {
        run_gamma_correction_mismatched_handle_batch<Element<decltype(tag)>>(p.cfg, p.op);
    });
}

INSTANTIATE_TEST_SUITE_P(Image_Color, GammaCorrectionMismatchedHandleBatchTest,
                         ::testing::ValuesIn(with_params<GammaCorrectionParams>(
                             concat_configs({
                                 make_configs(presets::kDefaultDTypes, presets::kLayoutsFullConv,
                                              {Roi::Full}, {presets::kDefaultSize}),
                             }),
                             {GammaCorrectionParams{2.2f}})),
                         op_config_name<GammaCorrectionParams>);

// Same-layout cases plus both directions of the fused output-layout conversion.
INSTANTIATE_TEST_SUITE_P(Image_Color, GammaCorrectionTest,
                         ::testing::ValuesIn(with_params<GammaCorrectionParams>(
                             concat_configs({
                                 make_configs(presets::kDefaultDTypes, presets::kLayoutsFullConv,
                                              {Roi::Full, Roi::Partial}, {presets::kTailWidthSize}),
                                 make_configs(presets::kDefaultDTypes, presets::kLayoutsFull,
                                              {Roi::Full, Roi::Partial},
                                              {presets::kDefaultSize, presets::kSubVectorSize}),
                             }),
                             {GammaCorrectionParams{2.2f}})),
                         op_config_name<GammaCorrectionParams>);
