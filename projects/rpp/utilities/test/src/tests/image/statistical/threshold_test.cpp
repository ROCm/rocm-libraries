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
#include "reference/threshold_ref.hpp"

using namespace rpptest;

namespace {

// Cutoffs in [0,255] units. Half-integer bounds keep integer-valued pixels off the cutoffs, so the
// mask is bit-exact. Channel c uses [rawMin + c*step, rawMax + c*step].
struct ThresholdParams {
    float rawMin, rawMax, step;
    std::string name() const {
        const std::string base = "min" + num_token(rawMin) + "_max" + num_token(rawMax);
        return step == 0.0f ? base : base + "_step" + num_token(step);
    }
};

// Converts a [0,255]-unit cutoff into the dtype's own units.
float cutoff_in_dtype(float raw, DType dt) {
    switch (dt) {
        case DType::U8:
            return raw;
        case DType::I8:
            return raw - 128.0f;
        case DType::F16:
        case DType::F32:
            return raw / 255.0f;
        default:
            return raw;
    }
}

template <typename T>
void run_threshold(const TestConfig& cfg, const ThresholdParams& op) {
    const Rpp32u c = static_cast<Rpp32u>(channels_of(cfg.layoutIn));
    RpptDesc srcDesc = make_src_descriptor(cfg);  // RPP takes a non-const ptr
    RpptDesc dstDesc = make_dst_descriptor(cfg);
    const std::size_t count = element_count(srcDesc);
    const std::size_t bytes = byte_size(srcDesc, cfg.dtype);

    PinnedArray<Rpp32f> minTensor(cfg.backend, cfg.size.n * c);
    PinnedArray<Rpp32f> maxTensor(cfg.backend, cfg.size.n * c);
    PinnedArray<RpptROI> roi(cfg.backend, cfg.size.n);
    const std::vector<RpptROI> roiVec = make_roi(srcDesc, cfg.roi);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) {
        for (Rpp32u ch = 0; ch < c; ++ch) {
            const float shift = op.step * static_cast<float>(ch);
            minTensor[i * c + ch] = cutoff_in_dtype(op.rawMin + shift, cfg.dtype);
            maxTensor[i * c + ch] = cutoff_in_dtype(op.rawMax + shift, cfg.dtype);
        }
        roi[i] = roiVec[i];
    }

    std::vector<T> input(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    golden = input;
    threshold_reference<T>(input.data(), srcDesc, golden.data(), dstDesc, cfg.dtype, roi.data(),
                           XYWH, minTensor.data(), maxTensor.data());

    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, cfg.size.n);
    ASSERT_EQ(rppt_threshold(src.ptr(), &srcDesc, dst.ptr(), &dstDesc, minTensor.data(),
                             maxTensor.data(), roi.data(), XYWH, handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    EXPECT_TRUE(
        compare_roi<T>(actual.data(), golden.data(), dstDesc, roi.data(), XYWH, kExact(cfg.dtype)));
}

}  // namespace

// Full name:
// Image_Statistical/ThresholdTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<Min>_<Max>
class ThresholdTest : public SkipListTest<WithParams<ThresholdParams>> {};

TEST_P(ThresholdTest, Correctness) {
    const auto& p = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(
        p.cfg.dtype, [&](auto tag) { run_threshold<Element<decltype(tag)>>(p.cfg, p.op); });
}

// Same-layout cases plus both directions of the fused output-layout conversion.
INSTANTIATE_TEST_SUITE_P(
    Image_Statistical, ThresholdTest,
    ::testing::ValuesIn(with_params<ThresholdParams>(
        concat_configs({
            make_configs({DType::U8, DType::F16, DType::F32, DType::I8}, presets::kLayoutsFullConv,
                         {Roi::Full, Roi::Partial}, {presets::kTailWidthSize}),
            make_configs({DType::U8, DType::F16, DType::F32, DType::I8}, presets::kLayoutsFull,
                         {Roi::Full, Roi::Partial},
                         {presets::kDefaultSize, presets::kSubVectorSize}),
        }),
        {ThresholdParams{29.5f, 100.5f, 0.0f}, ThresholdParams{100.5f, 29.5f, 0.0f},
         ThresholdParams{29.5f, 100.5f, 40.0f}})),
    op_config_name<ThresholdParams>);
