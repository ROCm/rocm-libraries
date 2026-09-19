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
#include "reference/rain_ref.hpp"

using namespace rpptest;

namespace {

// rain's rain-layer generator is seeded with std::random_device, so any rainPercentage > 0 is
// non-deterministic and cannot be golden-compared. The test pins rainPercentage = 0 (no drops):
// the op then reduces to a deterministic alpha blend toward the constant rain-layer background,
// which is the only reproducible slice. See rain_ref.hpp. alpha in [0, 1].
struct RainParams {
    float alpha;
    std::string name() const {
        return "a" + num_token(alpha);
    }
};

// The full-ROI SIMD path fuses the blend (fmadd) while the scalar path the reference
// mirrors does a separate multiply+add, so a rounded integer result can differ by 1 LSB.
constexpr Tolerance kRainTolerance = tolerance(1.0, 1e-4, 4e-3);

template <typename T>
void run_rain(const TestConfig& cfg, const RainParams& op) {
    RpptDesc srcDesc = make_src_descriptor(cfg);  // RPP takes a non-const ptr
    RpptDesc dstDesc = make_dst_descriptor(cfg);
    const std::size_t count = element_count(srcDesc);
    const std::size_t bytes = byte_size(srcDesc, cfg.dtype);

    // rainPercentage = 0 => no drops => deterministic; the other geometry params are irrelevant.
    const Rpp32f rainPercentage = 0.0f;
    const Rpp32u rainWidth = 1;
    const Rpp32u rainHeight = 6;
    const Rpp32f slantAngle = 0.0f;

    PinnedArray<Rpp32f> alpha(cfg.backend, cfg.size.n);
    PinnedArray<RpptROI> roi(cfg.backend, cfg.size.n);
    const std::vector<RpptROI> roiVec = make_roi(srcDesc, cfg.roi);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) {
        alpha[i] = op.alpha;
        roi[i] = roiVec[i];
    }

    std::vector<T> input(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    golden = input;
    rain_reference<T>(input.data(), srcDesc, golden.data(), dstDesc, cfg.dtype, roi.data(), XYWH,
                      op.alpha);

    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, cfg.size.n);
    ASSERT_EQ(
        rppt_rain(src.ptr(), &srcDesc, dst.ptr(), &dstDesc, rainPercentage, rainWidth, rainHeight,
                  slantAngle, alpha.data(), roi.data(), XYWH, handle.get(), cfg.backend),
        RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), dstDesc, roi.data(), XYWH,
                               kRainTolerance(cfg.dtype)));
}

}  // namespace

// Full name:
// Image_Effects/RainTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<Alpha>
class RainTest : public SkipListTest<WithParams<RainParams>> {};

TEST_P(RainTest, Correctness) {
    const auto& p = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(
        p.cfg.dtype, [&](auto tag) { run_rain<Element<decltype(tag)>>(p.cfg, p.op); });
}

// Same-layout cases plus both directions of the fused output-layout conversion.
INSTANTIATE_TEST_SUITE_P(Image_Effects, RainTest,
                         ::testing::ValuesIn(with_params<RainParams>(
                             concat_configs({
                                 make_configs({DType::U8, DType::F16, DType::F32},
                                              presets::kLayoutsFullConv, {Roi::Full, Roi::Partial},
                                              {presets::kTailWidthSize}),
                                 make_configs({DType::U8, DType::F16, DType::F32},
                                              presets::kLayoutsFull, {Roi::Full, Roi::Partial},
                                              {presets::kDefaultSize, presets::kSubVectorSize}),
                             }),
                             {RainParams{0.4f}})),
                         op_config_name<RainParams>);
