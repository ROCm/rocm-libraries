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
#include "reference/copy_ref.hpp"

using namespace rpptest;

namespace {

template <typename T>
void run_copy(const TestConfig& cfg) {
    RpptDesc srcDesc = make_src_descriptor(cfg);  // RPP takes a non-const ptr
    RpptDesc dstDesc = make_dst_descriptor(cfg);
    const std::size_t count = element_count(srcDesc);
    const std::size_t bytes = byte_size(srcDesc, cfg.dtype);

    // copy takes no ROI; the golden and compare walk the full frame (Roi::Full only).
    PinnedArray<RpptROI> roi(cfg.backend, cfg.size.n);
    const std::vector<RpptROI> roiVec = make_roi(srcDesc, cfg.roi);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) roi[i] = roiVec[i];

    // (1) Host golden model. golden starts as a copy of the input; the reference overwrites
    // the whole frame with src.
    std::vector<T> input(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    golden = input;
    copy_reference<T>(input.data(), srcDesc, golden.data(), dstDesc, roi.data(), XYWH);

    // (2) Run RPP on the configured backend. dst is pre-filled with a distinct pattern so a
    // no-op kernel would be caught.
    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    std::vector<T> dstInit(count);
    fill_input<T>(dstInit.data(), count, cfg.dtype, /*salt=*/1);
    dst.write(dstInit.data(), bytes);

    RppHandle handle(cfg.backend, cfg.size.n);
    ASSERT_EQ(rppt_copy(src.ptr(), &srcDesc, dst.ptr(), &dstDesc, handle.get(), cfg.backend),
              RPP_SUCCESS);

    // (3) Retrieve the result on the host (no-op copy for HOST, device->host for HIP).
    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    // (4) Compare over the frame. copy is bit-exact, so the tolerance is zero.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), dstDesc, roi.data(), XYWH, 0.0));
}

}  // namespace

// Full name: Image_DataExchange/CopyTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>
class CopyTest : public SkipListTest<TestConfig> {};

TEST_P(CopyTest, Correctness) {
    const TestConfig cfg = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(cfg.dtype, [&](auto tag) {
        run_copy<Element<decltype(tag)>>(cfg);
    });
}

// copy has no ROI argument (it copies the whole buffer), so only Roi::Full is instantiated.
// Same-layout cases plus both directions of the fused output-layout conversion.
INSTANTIATE_TEST_SUITE_P(
    Image_DataExchange, CopyTest,
    ::testing::ValuesIn(concat_configs({
        make_configs(presets::kDefaultDTypes, presets::kLayoutsFullConv, {Roi::Full},
                     {presets::kTailWidthSize}),
        make_configs(presets::kDefaultDTypes, presets::kLayoutsFull, {Roi::Full},
                     {presets::kDefaultSize, presets::kSubVectorSize}),
    })),
    config_param_name);
