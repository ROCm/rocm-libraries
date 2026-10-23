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
#include "reference/grid_dropout_ref.hpp"

using namespace rpptest;

namespace {

// A regular grid of half-cell holes built from each image's ROI rectangle.
constexpr Rpp32u kGridW = 3;
constexpr Rpp32u kGridH = 3;
constexpr Rpp32u kBoxesInEachImage = kGridW * kGridH;  // 9

template <typename T>
void run_grid_dropout(const TestConfig& cfg) {
    RpptDesc srcDesc = make_src_descriptor(cfg);  // RPP takes a non-const ptr
    RpptDesc dstDesc = make_dst_descriptor(cfg);
    const std::size_t count = element_count(srcDesc);
    const std::size_t bytes = byte_size(srcDesc, cfg.dtype);

    // Grid holes per image, laid out [n * stride + k].
    PinnedArray<RpptRoiLtrb> boxes(cfg.backend,
                                   static_cast<std::size_t>(cfg.size.n) * kBoxesInEachImage);
    PinnedArray<RpptROI> roi(cfg.backend, cfg.size.n);
    const std::vector<RpptROI> roiVec = make_roi(srcDesc, cfg.roi);

    // maxHoleW/maxHoleH are passed to the op; the golden ignores them.
    Rpp32u maxHoleW = 0, maxHoleH = 0;
    for (Rpp32u n = 0; n < cfg.size.n; ++n) {
        roi[n] = roiVec[n];
        const RoiBounds rb = roi_bounds(roiVec[n], XYWH);
        const Rpp32u cellW = std::max(1u, rb.w / kGridW);
        const Rpp32u cellH = std::max(1u, rb.h / kGridH);
        const Rpp32u holeW = std::max(1u, cellW / 2);
        const Rpp32u holeH = std::max(1u, cellH / 2);
        maxHoleW = std::max(maxHoleW, holeW);
        maxHoleH = std::max(maxHoleH, holeH);
        for (Rpp32u row = 0; row < kGridH; ++row) {
            for (Rpp32u col = 0; col < kGridW; ++col) {
                const int x1 = static_cast<int>(rb.x0 + col * cellW);
                const int y1 = static_cast<int>(rb.y0 + row * cellH);
                const int x2 =
                    std::min(x1 + static_cast<int>(holeW) - 1, static_cast<int>(rb.x0 + rb.w - 1));
                const int y2 =
                    std::min(y1 + static_cast<int>(holeH) - 1, static_cast<int>(rb.y0 + rb.h - 1));
                boxes[n * kBoxesInEachImage + (row * kGridW + col)] =
                    RpptRoiLtrb{{x1, y1}, {x2, y2}};
            }
        }
    }

    std::vector<T> input(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    golden = input;
    grid_dropout_reference<T>(input.data(), srcDesc, golden.data(), dstDesc, cfg.dtype, roi.data(),
                              XYWH, boxes.data(), kBoxesInEachImage);

    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, cfg.size.n);
    ASSERT_EQ(
        rppt_grid_dropout(src.ptr(), &srcDesc, dst.ptr(), &dstDesc, boxes.data(), kBoxesInEachImage,
                          maxHoleW, maxHoleH, roi.data(), XYWH, handle.get(), cfg.backend),
        RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    EXPECT_TRUE(
        compare_roi<T>(actual.data(), golden.data(), dstDesc, roi.data(), XYWH, kExact(cfg.dtype)));
}

}  // namespace

// Full name:
// Image_Effects/GridDropoutTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>
class GridDropoutTest : public SkipListTest<TestConfig> {};

TEST_P(GridDropoutTest, Correctness) {
    const TestConfig cfg = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(
        cfg.dtype, [&](auto tag) { run_grid_dropout<Element<decltype(tag)>>(cfg); });
}

// PartialRoi is red on both backends: the kernel copies the ROI but does not erase the holes. Do
// not weaken the golden to force green.
INSTANTIATE_TEST_SUITE_P(
    Image_Effects, GridDropoutTest,
    ::testing::ValuesIn(concat_configs({
        make_configs({DType::U8, DType::F16, DType::F32, DType::I8}, presets::kLayoutsFullConv,
                     {Roi::Full, Roi::Partial}, {presets::kTailWidthSize}),
        make_configs({DType::U8, DType::F16, DType::F32, DType::I8}, presets::kLayoutsFull,
                     {Roi::Full, Roi::Partial}, {presets::kDefaultSize, presets::kSubVectorSize}),
    })),
    config_param_name);
