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

#include <cmath>
#include <string>
#include <vector>

#include "framework/backend_memory.hpp"
#include "framework/config_param.hpp"
#include "framework/dtype.hpp"
#include "framework/intensity.hpp"
#include "framework/skip_list.hpp"
#include "framework/tensor_setup.hpp"

using namespace rpptest;

namespace {

// fog has no golden model: the output blends the image with a fog layer baked into the library
// (src/include/tensor/fog_mask.hpp) that the API does not describe, and intensityFactor = 0 is not
// an identity. Correctness checks storable, changed output; Negative checks the parameter ranges.

// Mid-range values inside the documented ranges (intensityFactor in [0,0.5], greyFactor in [0,1]).
constexpr float kIntensity = 0.4f;
constexpr float kGrey = 0.6f;

// Asserts the ROI changed and every output element is storable for its dtype (U8 [0,255],
// I8 [-128,127], F16/F32 [0,1]).
template <typename T>
void run_fog(const TestConfig& cfg) {
    RpptDesc srcDesc = make_src_descriptor(cfg);  // RPP takes a non-const ptr
    RpptDesc dstDesc = make_dst_descriptor(cfg);
    const std::size_t count = element_count(srcDesc);
    const std::size_t bytes = byte_size(srcDesc, cfg.dtype);

    PinnedArray<Rpp32f> intensity(cfg.backend, cfg.size.n), grey(cfg.backend, cfg.size.n);
    PinnedArray<RpptROI> roi(cfg.backend, cfg.size.n);
    const std::vector<RpptROI> roiVec = make_roi(srcDesc, cfg.roi);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) {
        intensity[i] = kIntensity;
        grey[i] = kGrey;
        roi[i] = roiVec[i];
    }

    // Seed each ROI dst element with its own source pixel so "unchanged" means a dead kernel; a raw
    // byte copy would not line up under a partial ROI or layout toggle.
    std::vector<T> input(count), dstInit(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    fill_input<T>(dstInit.data(), count, cfg.dtype, /*salt=*/1);  // outside-ROI: defined, unread
    for_each_roi_io(srcDesc, dstDesc, roiVec.data(), XYWH,
                    [&](Rpp32u, Rpp32u, Rpp32u, Rpp32u, std::size_t srcIdx, std::size_t dstIdx) {
                        dstInit[dstIdx] = input[srcIdx];
                    });

    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(dstInit.data(), bytes);

    RppHandle handle(cfg.backend, cfg.size.n);
    ASSERT_EQ(rppt_fog(src.ptr(), &srcDesc, dst.ptr(), &dstDesc, intensity.data(), grey.data(),
                       roi.data(), XYWH, handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    // Uses the test's own ROI copy in case the backend rewrites the op's ROI to LTRB in place.
    bool changed = false, storable = true;
    double firstBad = 0.0;
    for_each_roi_io(srcDesc, dstDesc, roiVec.data(), XYWH,
                    [&](Rpp32u, Rpp32u, Rpp32u, Rpp32u, std::size_t srcIdx, std::size_t dstIdx) {
                        const double v = to_double(actual[dstIdx]);
                        if (to_double(input[srcIdx]) != v) changed = true;
                        if (storable && (!std::isfinite(v) ||
                                         std::fabs(v - quantize_stored(v, cfg.dtype)) > 1e-6)) {
                            storable = false;
                            firstBad = v;
                        }
                    });
    EXPECT_TRUE(changed) << "fog left the ROI unchanged at intensityFactor " << kIntensity
                         << " / greyFactor " << kGrey;
    EXPECT_TRUE(storable) << "fog produced " << firstBad
                          << ", outside the storable range for dtype " << dtype_name(cfg.dtype);
}

// One out-of-range parameter pair per case, covering both bounds of both ranges.
struct FogNegativeParams {
    float intensity;
    float grey;
    std::string name() const {
        return "i" + num_token(intensity) + "_g" + num_token(grey);
    }
};

// Only non-success is asserted; the header does not name the RPP_ERROR* for an out-of-range scalar.
template <typename T>
void run_fog_negative(const TestConfig& cfg, const FogNegativeParams& op) {
    RpptDesc desc = make_src_descriptor(cfg);
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    PinnedArray<Rpp32f> intensity(cfg.backend, cfg.size.n), grey(cfg.backend, cfg.size.n);
    PinnedArray<RpptROI> roi(cfg.backend, cfg.size.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) {
        intensity[i] = op.intensity;
        grey[i] = op.grey;
        roi[i] = roiVec[i];
    }

    std::vector<T> input(count);
    fill_input<T>(input.data(), count, cfg.dtype);

    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);

    RppHandle handle(cfg.backend, cfg.size.n);
    const RppStatus status = rppt_fog(src.ptr(), &desc, dst.ptr(), &desc, intensity.data(),
                                      grey.data(), roi.data(), XYWH, handle.get(), cfg.backend);
    handle.sync();
    EXPECT_NE(status, RPP_SUCCESS) << "fog accepted out-of-range parameters (intensityFactor "
                                   << op.intensity << ", greyFactor " << op.grey << ")";
}

}  // namespace

// Full name: Image_Effects/FogTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>
class FogTest : public SkipListTest<TestConfig> {};

TEST_P(FogTest, Correctness) {
    const TestConfig& cfg = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(
        cfg.dtype, [&](auto tag) { run_fog<Element<decltype(tag)>>(cfg); });
}

// Same-layout cases plus both directions of the fused output-layout conversion.
INSTANTIATE_TEST_SUITE_P(
    Image_Effects, FogTest,
    ::testing::ValuesIn(concat_configs({
        make_configs({DType::U8, DType::F16, DType::F32, DType::I8}, presets::kLayoutsFullConv,
                     {Roi::Full, Roi::Partial}, {presets::kTailWidthSize}),
        make_configs({DType::U8, DType::F16, DType::F32, DType::I8}, presets::kLayoutsFull,
                     {Roi::Full, Roi::Partial}, {presets::kDefaultSize, presets::kSubVectorSize}),
    })),
    config_param_name);

// Full name:
// Image_Effects/FogNegativeTest.Negative/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_i<I>_g<G>
class FogNegativeTest : public SkipListTest<WithParams<FogNegativeParams>> {};

TEST_P(FogNegativeTest, Negative) {
    const auto& p = GetParam();
    run_fog_negative<Rpp8u>(p.cfg, p.op);
}

// Argument validation is dtype- and layout-independent, so it runs on one representative config.
INSTANTIATE_TEST_SUITE_P(Image_Effects, FogNegativeTest,
                         ::testing::ValuesIn(with_params<FogNegativeParams>(
                             make_configs({DType::U8}, {Layout::PKD3}, {Roi::Full}),
                             {FogNegativeParams{-0.1f, 0.5f}, FogNegativeParams{0.6f, 0.5f},
                              FogNegativeParams{0.25f, -0.1f}, FogNegativeParams{0.25f, 1.1f}})),
                         op_config_name<FogNegativeParams>);
