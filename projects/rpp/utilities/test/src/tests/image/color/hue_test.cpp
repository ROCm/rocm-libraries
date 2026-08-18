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
#include "framework/skip_list.hpp"
#include "framework/tensor_setup.hpp"
#include "framework/tolerance.hpp"
#include "reference/hue_ref.hpp"

using namespace rpptest;

namespace {

// hue shift in degrees, within the documented range 0 <= hue <= 359.
struct HueParams {
    float hue;
    std::string name() const {
        return "h" + num_token(hue);
    }
};

template <typename T>
void run_hue(const TestConfig& cfg, const HueParams& op) {
    const TensorShape shape{cfg.size.n, static_cast<Rpp32u>(channels_of(cfg.layout)), cfg.size.h,
                            cfg.size.w};
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layout);  // RPP takes a non-const ptr
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    // Parameters live in host-accessible (pinned for HIP) memory.
    PinnedArray<Rpp32f> hue(cfg.backend, shape.n);
    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < shape.n; ++i) {
        hue[i] = op.hue;
        roi[i] = roiVec[i];
    }

    // (1) Host golden model. golden starts as a copy of the input so the untouched
    // (outside-ROI) region is defined; only the ROI is overwritten by the reference.
    std::vector<T> input(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    golden = input;
    hue_reference<T>(input.data(), golden.data(), desc, cfg.dtype, roi.data(), XYWH, op.hue);

    // (2) Run RPP on the configured backend.
    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, shape.n);
    ASSERT_EQ(rppt_hue(src.ptr(), &desc, dst.ptr(), &desc, hue.data(), roi.data(), XYWH,
                       handle.get(), cfg.backend),
              RPP_SUCCESS);

    // (3) Retrieve the result on the host (no-op copy for HOST, device->host for HIP).
    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    // (4) Compare within tolerance over the ROI.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), desc, roi.data(), XYWH,
                               kRoundingTolerance(cfg.dtype)));
}

}  // namespace

// Full name: Image_Color/HueTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<Hue>
class HueTest : public SkipListTest<WithParams<HueParams>> {};

TEST_P(HueTest, Correctness) {
    const auto& p = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(p.cfg.dtype, [&](auto tag) {
        run_hue<Element<decltype(tag)>>(p.cfg, p.op);
    });
}

// Restricted to the 3-channel layouts: hue is an RGB (c = 3) op.
INSTANTIATE_TEST_SUITE_P(Image_Color, HueTest,
                         ::testing::ValuesIn(with_params<HueParams>(
                             make_configs({DType::U8, DType::F16, DType::F32},
                                          {Layout::PKD3, Layout::PLN3}, {Roi::Full, Roi::Partial}),
                             {HueParams{90.0f}})),
                         op_config_name<HueParams>);
