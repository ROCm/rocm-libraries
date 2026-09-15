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

// Raw cutoffs in [0,255] units, converted to each dtype's units at run time. Half-integer
// bounds are chosen deliberately: fill_input emits integer-valued pixels, so no pixel ever
// lands exactly on a cutoff -- the in-range classification (and thus the binary mask) is
// unambiguous in every dtype and comparison space, keeping the output bit-exact.
// step is added to both cutoffs once per channel, so channel c is tested against
// [rawMin + c*step, rawMax + c*step]; at step 0 every channel shares one pair.
struct ThresholdParams {
    float rawMin, rawMax, step;
    std::string name() const {
        const std::string base = "min" + num_token(rawMin) + "_max" + num_token(rawMax);
        return step == 0.0f ? base : base + "_step" + num_token(step);
    }
};

// Converts a raw [0,255]-unit cutoff into the dtype's own units (the exact value the op
// receives): U8 raw, I8 shifted by -128, F16/F32 normalized to [0,1].
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

// The mask is exactly 0/255, -128/127, or 0.0/1.0 -- all exactly representable, so the
// comparison is bit-exact for every dtype.

template <typename T>
void run_threshold(const TestConfig& cfg, const ThresholdParams& op) {
    const Rpp32u c = static_cast<Rpp32u>(channels_of(cfg.layoutIn));
    RpptDesc srcDesc = make_src_descriptor(cfg);  // RPP takes a non-const ptr
    RpptDesc dstDesc = make_dst_descriptor(cfg);
    const std::size_t count = element_count(srcDesc);
    const std::size_t bytes = byte_size(srcDesc, cfg.dtype);

    // Per-image, per-channel cutoffs in host-accessible (pinned for HIP) memory.
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

    // (1) Host golden model. golden starts as a copy of the input so the untouched
    // (outside-ROI) region is defined; only the ROI is overwritten by the reference.
    std::vector<T> input(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    golden = input;
    threshold_reference<T>(input.data(), srcDesc, golden.data(), dstDesc, cfg.dtype, roi.data(),
                           XYWH, minTensor.data(), maxTensor.data());

    // (2) Run RPP on the configured backend.
    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, cfg.size.n);
    ASSERT_EQ(rppt_threshold(src.ptr(), &srcDesc, dst.ptr(), &dstDesc, minTensor.data(),
                             maxTensor.data(), roi.data(), XYWH, handle.get(), cfg.backend),
              RPP_SUCCESS);

    // (3) Retrieve the result on the host (no-op copy for HOST, device->host for HIP).
    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    // (4) Compare within tolerance over the ROI.
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
