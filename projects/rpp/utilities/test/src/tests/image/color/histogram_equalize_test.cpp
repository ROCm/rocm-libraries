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
#include "reference/histogram_equalize_ref.hpp"

using namespace rpptest;

namespace {

// U8-only op. Grayscale equalization is exact integer; the 3-channel YCbCr round-trip picks up a
// little fp rounding, so 1 LSB is allowed (not enough to hide a convention/clamp bug).
double histogram_equalize_tolerance(DType) { return 1.0; }

template <typename T>
void run_histogram_equalize(const TestConfig& cfg) {
    RpptDesc srcDesc = make_src_descriptor(cfg);  // RPP takes a non-const ptr
    RpptDesc dstDesc = make_dst_descriptor(cfg);
    const std::size_t count = element_count(srcDesc);
    const std::size_t bytes = byte_size(srcDesc, cfg.dtype);

    PinnedArray<RpptROI> roi(cfg.backend, cfg.size.n);
    const std::vector<RpptROI> roiVec = make_roi(srcDesc, cfg.roi);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) roi[i] = roiVec[i];

    std::vector<T> input(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    golden = input;
    histogram_equalize_reference<T>(input.data(), srcDesc, golden.data(), dstDesc, roi.data(),
                                    XYWH);

    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, cfg.size.n);
    ASSERT_EQ(rppt_histogram_equalize(src.ptr(), &srcDesc, dst.ptr(), &dstDesc, roi.data(), XYWH,
                                      handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back

    dst.read(actual.data(), bytes);

    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), dstDesc, roi.data(), XYWH,
                               histogram_equalize_tolerance(cfg.dtype)));
}

}  // namespace

// Full name: Image_Color/HistogramEqualizeTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>
//
// HOST is green on the full grid. The two HIP 3-channel PartialRoi cases (PKD3, PLN3) are documented
// reds: the HIP 3-channel partial-ROI path is broken (grayscale-partial and 3-channel-full pass).
// The golden matches HOST and stays correct.
class HistogramEqualizeTest : public SkipListTest<TestConfig> {};

TEST_P(HistogramEqualizeTest, Correctness) {
    const TestConfig cfg = GetParam();
    ASSERT_EQ(cfg.dtype, DType::U8) << "histogram_equalize is U8 only";
    run_histogram_equalize<Rpp8u>(cfg);
}

// Same-layout cases plus both directions of the fused output-layout conversion.
INSTANTIATE_TEST_SUITE_P(
    Image_Color, HistogramEqualizeTest,
    ::testing::ValuesIn(concat_configs({
        make_configs({DType::U8}, presets::kLayoutsFullConv, {Roi::Full, Roi::Partial},
                     {presets::kTailWidthSize}),
        make_configs({DType::U8}, presets::kLayoutsFull, {Roi::Full, Roi::Partial},
                     {presets::kDefaultSize, presets::kSubVectorSize}),
    })),
    config_param_name);
