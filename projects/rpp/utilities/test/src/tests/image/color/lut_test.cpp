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
#include "reference/lut_ref.hpp"

using namespace rpptest;

namespace {

// One table (documented length 65536) shared by the batch. Only the first 256 entries matter for
// the integer dtypes; the rest are zeroed so no read is indeterminate.
template <typename T>
void fill_lut(T* lut, DType dt) {
    for (std::size_t j = 0; j < 65536; ++j) {
        int value = 0;
        if (j < 256) value = 255 - static_cast<int>(j);  // inverted intensity in [0,255]
        if (dt == DType::I8) value -= 128;               // shift into I8's signed range
        lut[j] = static_cast<T>(value);
    }
}

template <typename T>
void run_lut(const TestConfig& cfg) {
    RpptDesc srcDesc = make_src_descriptor(cfg);  // RPP takes a non-const ptr
    RpptDesc dstDesc = make_dst_descriptor(cfg);
    const std::size_t count = element_count(srcDesc);
    const std::size_t bytes = byte_size(srcDesc, cfg.dtype);

    // The table is host/pinned memory even for HIP (the API keeps lutPtr host-side).
    PinnedArray<T> lut(cfg.backend, 65536);
    fill_lut<T>(lut.data(), cfg.dtype);

    PinnedArray<RpptROI> roi(cfg.backend, cfg.size.n);
    const std::vector<RpptROI> roiVec = make_roi(srcDesc, cfg.roi);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) roi[i] = roiVec[i];

    std::vector<T> input(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    golden = input;
    lut_reference<T>(input.data(), srcDesc, golden.data(), dstDesc, cfg.dtype, roi.data(), XYWH,
                     lut.data());

    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, cfg.size.n);
    ASSERT_EQ(rppt_lut(src.ptr(), &srcDesc, dst.ptr(), &dstDesc, lut.data(), roi.data(), XYWH,
                       handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    // A look-up is bit-exact, so zero tolerance.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), dstDesc, roi.data(), XYWH, 0.0));
}

}  // namespace

// Full name: Image_Color/LutTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>
class LutTest : public SkipListTest<TestConfig> {};

TEST_P(LutTest, Correctness) {
    const auto& cfg = GetParam();
    dispatch_dtype<DType::U8, DType::I8>(cfg.dtype,
                                         [&](auto tag) { run_lut<Element<decltype(tag)>>(cfg); });
}

// Restricted to the integer dtypes (U8/I8): the 256-entry intensity index is unambiguous
// there, whereas the float dtypes' table-index semantics are not defined by the public API.
// Same-layout cases plus both directions of the fused output-layout conversion.
INSTANTIATE_TEST_SUITE_P(Image_Color, LutTest,
                         ::testing::ValuesIn(concat_configs({
                             make_configs({DType::U8, DType::I8}, presets::kLayoutsFullConv,
                                          {Roi::Full, Roi::Partial}, {presets::kTailWidthSize}),
                             make_configs({DType::U8, DType::I8}, presets::kLayoutsFull,
                                          {Roi::Full, Roi::Partial},
                                          {presets::kDefaultSize, presets::kSubVectorSize}),
                         })),
                         config_param_name);
