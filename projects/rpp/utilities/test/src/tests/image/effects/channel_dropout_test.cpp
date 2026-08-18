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

#include <array>
#include <string>
#include <vector>

#include "framework/backend_memory.hpp"
#include "framework/compare_tensor.hpp"
#include "framework/config_param.hpp"
#include "framework/dtype_dispatch.hpp"
#include "framework/skip_list.hpp"
#include "framework/tensor_setup.hpp"
#include "framework/tolerance.hpp"
#include "reference/channel_dropout_ref.hpp"

using namespace rpptest;

namespace {

// Per-channel keep(1)/drop(0) mask. PLN1 (1 channel) uses mask[0] only; PKD3/PLN3 use all
// three. Instantiated with two patterns so PLN1 gets both a drop and a keep case, and the
// 3-channel layouts exercise mixed keep/drop across channels.
struct ChannelDropoutParams {
    std::array<Rpp8u, 3> mask;
    std::string name() const {
        std::string s = "m";
        for (Rpp8u v : mask) s += (v ? '1' : '0');
        return s;
    }
};

// Channel dropout is a pure keep/erase: kept channels are copied bit-exact, dropped channels
// are set to an exact constant. No arithmetic, so every dtype is bit-exact.

template <typename T>
void run_channel_dropout(const TestConfig& cfg, const ChannelDropoutParams& op) {
    const Rpp32u channels = static_cast<Rpp32u>(channels_of(cfg.layout));
    const TensorShape shape{cfg.size.n, channels, cfg.size.h, cfg.size.w};
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layout);  // RPP takes a non-const ptr
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    // dropoutTensor is a 1D Rpp8u tensor of size batchSize * channels ([image * channels + c]).
    PinnedArray<Rpp8u> dropout(cfg.backend, static_cast<std::size_t>(shape.n) * channels);
    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < shape.n; ++i) {
        for (Rpp32u c = 0; c < channels; ++c) dropout[i * channels + c] = op.mask[c];
        roi[i] = roiVec[i];
    }

    // (1) Host golden model. golden starts as a copy of the input so the untouched
    // (outside-ROI) region is defined; only the ROI is overwritten by the reference.
    std::vector<T> input(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    golden = input;
    channel_dropout_reference<T>(input.data(), golden.data(), desc, cfg.dtype, roi.data(), XYWH,
                                 dropout.data());

    // (2) Run RPP on the configured backend.
    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, shape.n);
    ASSERT_EQ(rppt_channel_dropout(src.ptr(), &desc, dst.ptr(), &desc, dropout.data(), roi.data(),
                                   XYWH, handle.get(), cfg.backend),
              RPP_SUCCESS);

    // (3) Retrieve the result on the host (no-op copy for HOST, device->host for HIP).
    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    // (4) Compare within tolerance over the ROI.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), desc, roi.data(), XYWH,
                               kExact(cfg.dtype)));
}

}  // namespace

// Full name:
// Image_Effects/ChannelDropoutTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<Mask>
class ChannelDropoutTest : public SkipListTest<WithParams<ChannelDropoutParams>> {};

TEST_P(ChannelDropoutTest, Correctness) {
    const auto& p = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(p.cfg.dtype, [&](auto tag) {
        run_channel_dropout<Element<decltype(tag)>>(p.cfg, p.op);
    });
}

INSTANTIATE_TEST_SUITE_P(
    Image_Effects, ChannelDropoutTest,
    ::testing::ValuesIn(with_params<ChannelDropoutParams>(
        make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                     {Layout::PKD3, Layout::PLN3, Layout::PLN1}, {Roi::Full, Roi::Partial}),
        {ChannelDropoutParams{{0, 1, 1}}, ChannelDropoutParams{{1, 0, 1}}})),
    op_config_name<ChannelDropoutParams>);
