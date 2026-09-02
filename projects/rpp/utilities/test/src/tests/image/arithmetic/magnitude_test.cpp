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
#include "reference/magnitude_ref.hpp"

using namespace rpptest;

namespace {

// I8 kept sub-LSB to surface a real kernel bug: the HIP I8 kernel truncates instead of
// rounding (HOST rounds correctly).
constexpr Tolerance kMagnitudeTolerance = kRoundingTolerance.with_i8(0.5);

template <typename T>
void run_magnitude(const TestConfig& cfg) {
    RpptDesc srcDesc = make_src_descriptor(cfg);  // RPP takes a non-const ptr
    RpptDesc dstDesc = make_dst_descriptor(cfg);
    const std::size_t count = element_count(srcDesc);
    const std::size_t bytes = byte_size(srcDesc, cfg.dtype);

    PinnedArray<RpptROI> roi(cfg.backend, cfg.size.n);
    const std::vector<RpptROI> roiVec = make_roi(srcDesc, cfg.roi);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) roi[i] = roiVec[i];

    // Two distinct operands (salt shifts the second) so the magnitude is non-degenerate.
    std::vector<T> input1(count), input2(count), golden(count), actual(count);
    fill_input<T>(input1.data(), count, cfg.dtype);
    fill_input<T>(input2.data(), count, cfg.dtype, 1);
    golden = input1;
    magnitude_reference<T>(input1.data(), input2.data(), srcDesc, golden.data(), dstDesc, cfg.dtype,
                           roi.data(), XYWH);

    DeviceTensor src1(cfg.backend, bytes), src2(cfg.backend, bytes), dst(cfg.backend, bytes);
    src1.write(input1.data(), bytes);
    src2.write(input2.data(), bytes);
    dst.write(input1.data(), bytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, cfg.size.n);
    ASSERT_EQ(rppt_magnitude(src1.ptr(), src2.ptr(), &srcDesc, dst.ptr(), &dstDesc, roi.data(),
                             XYWH, handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back

    dst.read(actual.data(), bytes);

    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), dstDesc, roi.data(), XYWH,
                               kMagnitudeTolerance(cfg.dtype)));
}

}  // namespace

// Full name: Image_Arithmetic/MagnitudeTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>
class MagnitudeTest : public SkipListTest<TestConfig> {};

TEST_P(MagnitudeTest, Correctness) {
    const TestConfig cfg = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(cfg.dtype, [&](auto tag) {
        run_magnitude<Element<decltype(tag)>>(cfg);
    });
}

// Same-layout cases plus both directions of the fused output-layout conversion.
INSTANTIATE_TEST_SUITE_P(
    Image_Arithmetic, MagnitudeTest,
    ::testing::ValuesIn(concat_configs({
        make_configs({DType::U8, DType::F16, DType::F32}, presets::kLayoutsFullConv,
                     {Roi::Full, Roi::Partial},
                     {presets::kTailWidthSize}),
        make_configs({DType::U8, DType::F16, DType::F32}, presets::kLayoutsFull,
                     {Roi::Full, Roi::Partial},
                     {presets::kDefaultSize, presets::kSubVectorSize}),
    })),
    config_param_name);
