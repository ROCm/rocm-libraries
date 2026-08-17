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
#include "framework/tensor_setup.hpp"
#include "reference/bitwise_binary_ref.hpp"

using namespace rpptest;

namespace {

template <typename T>
void run_bitwise_and(const TestConfig& cfg) {
    const TensorShape shape{cfg.size.n, static_cast<Rpp32u>(channels_of(cfg.layout)), cfg.size.h,
                            cfg.size.w};
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layout);  // RPP takes a non-const ptr
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    // ROIs live in host-accessible (pinned for HIP) memory.
    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < shape.n; ++i) roi[i] = roiVec[i];

    // (1) Host golden model. Two distinct operands so the op is exercised on differing bit
    // patterns. golden starts as a copy of input1 so the untouched (outside-ROI) region is
    // defined; only the ROI is overwritten by the reference.
    std::vector<T> input1(count), input2(count), golden(count), actual(count);
    fill_input<T>(input1.data(), count, cfg.dtype);
    for (std::size_t i = 0; i < count; ++i)
        input2[i] = from_double<T>(static_cast<double>((i * 53u + 97u) & 0xFFu));
    golden = input1;
    bitwise_binary_reference<T>(input1.data(), input2.data(), golden.data(), desc, roi.data(), XYWH,
                                BitwiseOp::And);

    // (2) Run RPP on the configured backend.
    DeviceTensor src1(cfg.backend, bytes), src2(cfg.backend, bytes), dst(cfg.backend, bytes);
    src1.write(input1.data(), bytes);
    src2.write(input2.data(), bytes);
    dst.write(input1.data(), bytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, shape.n);
    ASSERT_EQ(rppt_bitwise_and(src1.ptr(), src2.ptr(), &desc, dst.ptr(), &desc, roi.data(), XYWH,
                               handle.get(), cfg.backend),
              RPP_SUCCESS);

    // (3) Retrieve the result on the host (no-op copy for HOST, device->host for HIP).
    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    // (4) Compare over the ROI. bitwise_and is bit-exact, so the tolerance is zero.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), desc, roi.data(), XYWH, 0.0));
}

}  // namespace

// Full name: Image_Bitwise/BitwiseAndTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>
class BitwiseAndTest : public ::testing::TestWithParam<TestConfig> {};

TEST_P(BitwiseAndTest, Correctness) {
    const TestConfig cfg = GetParam();
    // rppt_bitwise_and only supports U8; the grid below is U8-only.
    run_bitwise_and<Rpp8u>(cfg);
}

INSTANTIATE_TEST_SUITE_P(Image_Bitwise, BitwiseAndTest,
                         ::testing::ValuesIn(make_configs({DType::U8},
                                                          {Layout::PKD3, Layout::PLN3, Layout::PLN1},
                                                          {Roi::Full, Roi::Partial})),
                         config_param_name);
