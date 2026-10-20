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

#include <algorithm>
#include <vector>

#include "framework/backend_memory.hpp"
#include "framework/compare.hpp"
#include "framework/config_param.hpp"
#include "framework/dtype.hpp"
#include "framework/skip_list.hpp"
#include "framework/tensor_setup.hpp"
#include "framework/tolerance.hpp"
#include "reference/coarse_dropout_ref.hpp"

using namespace rpptest;

namespace {

template <typename T>
void run_coarse_dropout(const TestConfig& cfg) {
    RpptDesc srcDesc = make_src_descriptor(cfg);  // RPP takes a non-const ptr
    RpptDesc dstDesc = make_dst_descriptor(cfg);
    const std::size_t count = element_count(srcDesc);
    const std::size_t bytes = byte_size(srcDesc, cfg.dtype);

    const Rpp32u maxBoxesPerImage = 3;

    // anchorBoxInfoTensor stride is maxBoxesPerImage; numBoxesTensor holds active counts.
    PinnedArray<RpptRoiLtrb> boxes(cfg.backend,
                                   static_cast<std::size_t>(cfg.size.n) * maxBoxesPerImage);
    PinnedArray<Rpp32u> numBoxes(cfg.backend, cfg.size.n);
    PinnedArray<RpptROI> roi(cfg.backend, cfg.size.n);
    const std::vector<RpptROI> roiVec = make_roi(srcDesc, cfg.roi);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) roi[i] = roiVec[i];

    // Three non-overlapping boxes per image in absolute coordinates, clamped inside the ROI.
    // numBoxes exercises stride vs active count: image 0 uses 3, image 1 uses 2.
    for (Rpp32u n = 0; n < cfg.size.n; ++n) {
        const RoiBounds rb = roi_bounds(roiVec[n], XYWH);
        const int x0 = static_cast<int>(rb.x0);
        const int y0 = static_cast<int>(rb.y0);
        const int xmax = static_cast<int>(rb.x0 + rb.w) - 1;  // inclusive right edge
        const int ymax = static_cast<int>(rb.y0 + rb.h) - 1;  // inclusive bottom edge
        const int bw = std::max(1, static_cast<int>(rb.w) / 5);
        const int bh = std::max(1, static_cast<int>(rb.h) / 5);
        const int lx = x0;
        const int rx = std::max(lx + bw, xmax - bw + 1);  // right block start (past left block)
        const int ty = y0;
        const int by = std::max(ty + bh, ymax - bh + 1);  // bottom block start (past top blocks)

        auto make_box = [&](int bx0, int by0) {
            RpptRoiLtrb box{};
            box.lt.x = std::min(bx0, xmax);
            box.lt.y = std::min(by0, ymax);
            box.rb.x = std::min(bx0 + bw - 1, xmax);
            box.rb.y = std::min(by0 + bh - 1, ymax);
            return box;
        };

        RpptRoiLtrb* imgBoxes = &boxes[n * maxBoxesPerImage];
        imgBoxes[0] = make_box(lx, ty);  // upper-left
        imgBoxes[1] = make_box(rx, ty);  // upper-right
        imgBoxes[2] = make_box(lx, by);  // lower-left
    }
    numBoxes[0] = 3;
    if (cfg.size.n > 1) numBoxes[1] = 2;
    for (Rpp32u n = 2; n < cfg.size.n; ++n) numBoxes[n] = 3;

    std::vector<T> input(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    golden = input;
    coarse_dropout_reference<T>(input.data(), srcDesc, golden.data(), dstDesc, cfg.dtype,
                                roi.data(), XYWH, boxes.data(), numBoxes.data(), maxBoxesPerImage);

    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, cfg.size.n);
    ASSERT_EQ(
        rppt_coarse_dropout(src.ptr(), &srcDesc, dst.ptr(), &dstDesc, boxes.data(), numBoxes.data(),
                            maxBoxesPerImage, roi.data(), XYWH, handle.get(), cfg.backend),
        RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    EXPECT_TRUE(
        compare_roi<T>(actual.data(), golden.data(), dstDesc, roi.data(), XYWH, kExact(cfg.dtype)));
}

}  // namespace

// Full name:
// Image_Effects/CoarseDropoutTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_2x36x48
class CoarseDropoutTest : public SkipListTest<TestConfig> {};

TEST_P(CoarseDropoutTest, Correctness) {
    const TestConfig cfg = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(
        cfg.dtype, [&](auto tag) { run_coarse_dropout<Element<decltype(tag)>>(cfg); });
}

// HIP PartialRoi is red: the kernel leaves the erased region unwritten under a non-full ROI. Do not
// weaken the golden to force green.
INSTANTIATE_TEST_SUITE_P(
    Image_Effects, CoarseDropoutTest,
    ::testing::ValuesIn(concat_configs({
        make_configs({DType::U8, DType::F16, DType::F32, DType::I8}, presets::kLayoutsFullConv,
                     {Roi::Full, Roi::Partial}, {presets::kTailWidthSize}),
        make_configs({DType::U8, DType::F16, DType::F32, DType::I8}, presets::kLayoutsFull,
                     {Roi::Full, Roi::Partial}, {presets::kDefaultSize, presets::kSubVectorSize}),
    })),
    config_param_name);
