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
#include "reference/crop_ref.hpp"

using namespace rpptest;

namespace {

template <typename T>
void run_crop(const TestConfig& cfg) {
    RpptDesc srcDesc = make_src_descriptor(cfg);  // RPP takes a non-const ptr
    RpptDesc dstDesc = make_dst_descriptor(cfg);
    const std::size_t count = element_count(srcDesc);
    const std::size_t bytes = byte_size(srcDesc, cfg.dtype);

    PinnedArray<RpptROI> roi(cfg.backend, cfg.size.n);
    const std::vector<RpptROI> roiVec = make_roi(srcDesc, cfg.roi);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) roi[i] = roiVec[i];

    // (1) Host golden model. golden starts as the dst init pattern (below) so the untouched
    // (outside-ROI) region is defined; only the cropped region is overwritten by the reference.
    std::vector<T> input(count), dstInit(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    fill_input<T>(dstInit.data(), count, cfg.dtype, /*salt=*/1);
    golden = dstInit;
    crop_reference<T>(input.data(), srcDesc, golden.data(), dstDesc, roi.data(), XYWH);

    // (2) Run RPP on the configured backend. dst is pre-filled with a distinct pattern so a
    // no-op kernel would be caught within the cropped region.
    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(dstInit.data(), bytes);

    RppHandle handle(cfg.backend, cfg.size.n);
    ASSERT_EQ(rppt_crop(src.ptr(), &srcDesc, dst.ptr(), &dstDesc, roi.data(), XYWH, handle.get(),
                        cfg.backend),
              RPP_SUCCESS);

    // (3) Retrieve the result on the host (no-op copy for HOST, device->host for HIP).
    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    // (4) Compare over the cropped region. crop is a verbatim copy, so the tolerance is zero.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), dstDesc, roi.data(), XYWH, 0.0));
}

}  // namespace

// Full name: Image_Geometric/CropTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>
class CropTest : public SkipListTest<TestConfig> {};

TEST_P(CropTest, Correctness) {
    const TestConfig cfg = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(cfg.dtype, [&](auto tag) {
        run_crop<Element<decltype(tag)>>(cfg);
    });
}

// Same-layout cases plus both directions of the fused output-layout conversion.
INSTANTIATE_TEST_SUITE_P(
    Image_Geometric, CropTest,
    ::testing::ValuesIn(concat_configs({
        make_configs({DType::U8, DType::F16, DType::F32, DType::I8}, presets::kLayoutsFullConv,
                     {Roi::Full, Roi::Partial},
                     {presets::kTailWidthSize}),
        make_configs({DType::U8, DType::F16, DType::F32, DType::I8}, presets::kLayoutsFull,
                     {Roi::Full, Roi::Partial},
                     {presets::kDefaultSize, presets::kSubVectorSize}),
    })),
    config_param_name);
