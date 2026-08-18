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
#include "framework/compare_tensor.hpp"
#include "framework/config_param.hpp"
#include "framework/dtype_dispatch.hpp"
#include "framework/tensor_setup.hpp"
#include "framework/tolerance.hpp"
#include "reference/threshold_ref.hpp"

using namespace rpptest;

namespace {

// Raw cutoffs in [0,255] units, converted to each dtype's units at run time. Half-integer
// bounds are chosen deliberately: fill_input emits integer-valued pixels, so no pixel ever
// lands exactly on a cutoff -- the in-range classification (and thus the binary mask) is
// unambiguous in every dtype and comparison space, keeping the output bit-exact.
struct ThresholdParams {
    float rawMin, rawMax;
    std::string name() const { return "min" + num_token(rawMin) + "_max" + num_token(rawMax); }
};

// Converts a raw [0,255]-unit cutoff into the dtype's own units (the exact value the op
// receives): U8 raw, I8 shifted by -128, F16/F32 normalized to [0,1].
float cutoff_in_dtype(float raw, DType dt) {
    switch (dt) {
        case DType::U8: return raw;
        case DType::I8: return raw - 128.0f;
        case DType::F16:
        case DType::F32: return raw / 255.0f;
        default: return raw;
    }
}

// The mask is exactly 0/255, -128/127, or 0.0/1.0 -- all exactly representable, so the
// comparison is bit-exact for every dtype.

template <typename T>
void run_threshold(const TestConfig& cfg, const ThresholdParams& op) {
    const Rpp32u c = static_cast<Rpp32u>(channels_of(cfg.layout));
    const TensorShape shape{cfg.size.n, c, cfg.size.h, cfg.size.w};
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layout);  // RPP takes a non-const ptr
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    // Per-image, per-channel cutoffs in host-accessible (pinned for HIP) memory.
    PinnedArray<Rpp32f> minTensor(cfg.backend, shape.n * c);
    PinnedArray<Rpp32f> maxTensor(cfg.backend, shape.n * c);
    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < shape.n; ++i) {
        for (Rpp32u ch = 0; ch < c; ++ch) {
            minTensor[i * c + ch] = cutoff_in_dtype(op.rawMin, cfg.dtype);
            maxTensor[i * c + ch] = cutoff_in_dtype(op.rawMax, cfg.dtype);
        }
        roi[i] = roiVec[i];
    }

    // (1) Host golden model. golden starts as a copy of the input so the untouched
    // (outside-ROI) region is defined; only the ROI is overwritten by the reference.
    std::vector<T> input(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    golden = input;
    threshold_reference<T>(input.data(), golden.data(), desc, cfg.dtype, roi.data(), XYWH,
                           minTensor.data(), maxTensor.data());

    // (2) Run RPP on the configured backend.
    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, shape.n);
    ASSERT_EQ(rppt_threshold(src.ptr(), &desc, dst.ptr(), &desc, minTensor.data(), maxTensor.data(),
                             roi.data(), XYWH, handle.get(), cfg.backend),
              RPP_SUCCESS);

    // (3) Retrieve the result on the host (no-op copy for HOST, device->host for HIP).
    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    // (4) Compare within tolerance over the ROI.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), desc, roi.data(), XYWH,
                               kExact(cfg.dtype)));
}

}  // namespace

// Full name: Image_Statistical/ThresholdTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<Min>_<Max>
class ThresholdTest : public ::testing::TestWithParam<WithParams<ThresholdParams>> {};

TEST_P(ThresholdTest, Correctness) {
    const auto& p = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(p.cfg.dtype, [&](auto tag) {
        run_threshold<Element<decltype(tag)>>(p.cfg, p.op);
    });
}

INSTANTIATE_TEST_SUITE_P(
    Image_Statistical, ThresholdTest,
    ::testing::ValuesIn(with_params<ThresholdParams>(
        make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                     {Layout::PKD3, Layout::PLN3, Layout::PLN1}, {Roi::Full, Roi::Partial}),
        {ThresholdParams{29.5f, 100.5f}})),
    op_config_name<ThresholdParams>);
