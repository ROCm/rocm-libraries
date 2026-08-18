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
#include "framework/compare_tensor.hpp"
#include "framework/config_param.hpp"
#include "framework/dtype_dispatch.hpp"
#include "framework/skip_list.hpp"
#include "framework/tensor_setup.hpp"
#include "framework/tolerance.hpp"
#include "reference/coarse_dropout_ref.hpp"

using namespace rpptest;

namespace {

// coarse_dropout is a pure passthrough with select rectangular regions erased to an exact
// constant (black). No arithmetic, so every dtype is bit-exact.

template <typename T>
void run_coarse_dropout(const TestConfig& cfg) {
    const Rpp32u channels = static_cast<Rpp32u>(channels_of(cfg.layout));
    const TensorShape shape{cfg.size.n, channels, cfg.size.h, cfg.size.w};
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layout);  // RPP takes a non-const ptr
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    const Rpp32u maxBoxesPerImage = 3;

    // anchorBoxInfoTensor stride is maxBoxesPerImage; numBoxesTensor holds active counts.
    PinnedArray<RpptRoiLtrb> boxes(cfg.backend, static_cast<std::size_t>(shape.n) * maxBoxesPerImage);
    PinnedArray<Rpp32u> numBoxes(cfg.backend, shape.n);
    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < shape.n; ++i) roi[i] = roiVec[i];

    // Build three mutually non-overlapping erase-regions per image, placed in distinct quadrants
    // (upper-left, upper-right, lower-left) of that image's ROI rectangle, in absolute image
    // coordinates and clamped to stay inside the ROI so they hold for both Full and Partial ROIs.
    // The same boxes feed the golden and the kernel. numBoxes exercises the stride vs active count:
    // image 0 uses all 3, image 1 uses 2 -- but every slot is still a valid in-bounds box.
    for (Rpp32u n = 0; n < shape.n; ++n) {
        const RoiBounds rb = roi_bounds(roiVec[n], XYWH);
        const int x0 = static_cast<int>(rb.x0);
        const int y0 = static_cast<int>(rb.y0);
        const int xmax = static_cast<int>(rb.x0 + rb.w) - 1;  // inclusive right edge
        const int ymax = static_cast<int>(rb.y0 + rb.h) - 1;  // inclusive bottom edge
        const int bw = std::max(1, static_cast<int>(rb.w) / 5);
        const int bh = std::max(1, static_cast<int>(rb.h) / 5);
        // Column anchors: left block near x0, right block near the right edge.
        const int lx = x0;
        const int rx = std::max(lx + bw, xmax - bw + 1);  // right block start (past left block)
        // Row anchors: top blocks near y0, bottom block near the bottom edge.
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
    if (shape.n > 1) numBoxes[1] = 2;
    for (Rpp32u n = 2; n < shape.n; ++n) numBoxes[n] = 3;

    // (1) Host golden model. golden starts as a copy of the input so the untouched
    // (outside-ROI) region is defined; only the ROI is overwritten by the reference.
    std::vector<T> input(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    golden = input;
    coarse_dropout_reference<T>(input.data(), golden.data(), desc, cfg.dtype, roi.data(), XYWH,
                                boxes.data(), numBoxes.data(), maxBoxesPerImage);

    // (2) Run RPP on the configured backend.
    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, shape.n);
    ASSERT_EQ(rppt_coarse_dropout(src.ptr(), &desc, dst.ptr(), &desc, boxes.data(), numBoxes.data(),
                                  maxBoxesPerImage, roi.data(), XYWH, handle.get(), cfg.backend),
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
// Image_Effects/CoarseDropoutTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_2x36x48
class CoarseDropoutTest : public SkipListTest<TestConfig> {};

TEST_P(CoarseDropoutTest, Correctness) {
    const TestConfig cfg = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(cfg.dtype, [&](auto tag) {
        run_coarse_dropout<Element<decltype(tag)>>(cfg);
    });
}

// FullRoi (both backends) and HOST PartialRoi pass the full grid -- this end-to-end validates the
// shared packed-origin + absolute-frame box + I8-black golden model (the same model cutout/grid use).
// The 12 HIP PartialRoi cases are red as a documented kernel finding
// HIP leaves the erased region unwritten under a non-full ROI (same class as the warp/rotate/remap
// HIP partial-ROI placement bug). Do not weaken
// the golden to force green.
INSTANTIATE_TEST_SUITE_P(
    Image_Effects, CoarseDropoutTest,
    ::testing::ValuesIn(make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                                     {Layout::PKD3, Layout::PLN3, Layout::PLN1},
                                     {Roi::Full, Roi::Partial})),
    config_param_name);
