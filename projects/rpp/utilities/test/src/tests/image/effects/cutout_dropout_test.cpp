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
#include <vector>

#include "framework/backend_memory.hpp"
#include "framework/compare_tensor.hpp"
#include "framework/config_param.hpp"
#include "framework/dtype_dispatch.hpp"
#include "framework/skip_list.hpp"
#include "framework/tensor_setup.hpp"
#include "framework/tolerance.hpp"
#include "reference/cutout_dropout_ref.hpp"

using namespace rpptest;

namespace {

// cutout_dropout overwrites rectangular boxes with caller-supplied solid colors: a direct store,
// no arithmetic, so every dtype is bit-exact.

// Per-image box stride and count. cutout_dropout has no maxBoxesPerImage parameter -- the kernel
// infers the anchor/color stride as max(numBoxesTensor). The ragged case (an image whose count is
// below that max) has no documented array-packing contract and was observed to leave that image
// un-erased even on FullRoi, so every image uses the same count == the stride, which is
// unambiguous regardless of the packing convention.
constexpr Rpp32u kMaxBoxesPerImage = 2;

// Per-(box, channel) fill intensity in [0,255]. Distinct colors verify channel handling; PLN1 uses
// index 0 only.
constexpr std::array<std::array<double, 3>, 2> kBoxIntensity = {{
    {{200.0, 60.0, 120.0}},   // box 0
    {{30.0, 220.0, 90.0}},    // box 1
}};

template <typename T>
void run_cutout_dropout(const TestConfig& cfg) {
    const Rpp32u channels = static_cast<Rpp32u>(channels_of(cfg.layout));
    const TensorShape shape{cfg.size.n, channels, cfg.size.h, cfg.size.w};
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layout);  // RPP takes a non-const ptr
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u n = 0; n < shape.n; ++n) roi[n] = roiVec[n];

    // Box geometry / colors / counts. Boxes are laid out per image at [n * stride + k] and built
    // relative to each image's ROI rectangle so they stay inside the ROI for both Full and Partial.
    // The SAME tensors feed the golden and the kernel.
    const std::size_t boxCount = static_cast<std::size_t>(shape.n) * kMaxBoxesPerImage;
    PinnedArray<RpptRoiLtrb> boxes(cfg.backend, boxCount);
    PinnedArray<Rpp32u> numBoxes(cfg.backend, shape.n);
    PinnedArray<T> colors(cfg.backend, boxCount * channels);

    for (Rpp32u n = 0; n < shape.n; ++n) {
        const RoiBounds rb = roi_bounds(roiVec[n], XYWH);
        // box 0: quarter-size block in the upper-left region of the ROI.
        RpptRoiLtrb b0{};
        b0.lt.x = static_cast<int>(rb.x0 + rb.w / 8);
        b0.lt.y = static_cast<int>(rb.y0 + rb.h / 8);
        b0.rb.x = b0.lt.x + static_cast<int>(rb.w / 4) - 1;
        b0.rb.y = b0.lt.y + static_cast<int>(rb.h / 4) - 1;
        // box 1: non-overlapping quarter-size block in the lower-right region of the ROI.
        RpptRoiLtrb b1{};
        b1.lt.x = static_cast<int>(rb.x0 + rb.w * 5 / 8);
        b1.lt.y = static_cast<int>(rb.y0 + rb.h * 5 / 8);
        b1.rb.x = b1.lt.x + static_cast<int>(rb.w / 4) - 1;
        b1.rb.y = b1.lt.y + static_cast<int>(rb.h / 4) - 1;

        boxes[n * kMaxBoxesPerImage + 0] = b0;
        boxes[n * kMaxBoxesPerImage + 1] = b1;

        for (Rpp32u k = 0; k < kMaxBoxesPerImage; ++k)
            for (Rpp32u c = 0; c < channels; ++c)
                colors[(n * kMaxBoxesPerImage + k) * channels + c] =
                    from_double<T>(from_unit(kBoxIntensity[k][c] / 255.0, cfg.dtype));
    }
    // Every image uses the full stride of boxes (uniform count == stride; see kMaxBoxesPerImage).
    for (Rpp32u n = 0; n < shape.n; ++n) numBoxes[n] = kMaxBoxesPerImage;

    // (1) Host golden model. golden starts as a copy of the input so the untouched (outside-ROI)
    // region is defined; only the ROI is overwritten by the reference.
    std::vector<T> input(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    golden = input;
    cutout_dropout_reference<T>(input.data(), golden.data(), desc, cfg.dtype, roi.data(), XYWH,
                               boxes.data(), numBoxes.data(), kMaxBoxesPerImage, colors.data());

    // (2) Run RPP on the configured backend.
    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, shape.n);
    ASSERT_EQ(rppt_cutout_dropout(src.ptr(), &desc, dst.ptr(), &desc, boxes.data(),
                                  static_cast<void*>(colors.data()), numBoxes.data(), roi.data(),
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
// Image_Effects/CutoutDropoutTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>
class CutoutDropoutTest : public SkipListTest<TestConfig> {};

TEST_P(CutoutDropoutTest, Correctness) {
    const auto& cfg = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(cfg.dtype, [&](auto tag) {
        run_cutout_dropout<Element<decltype(tag)>>(cfg);
    });
}

// FullRoi passes the full grid on both backends (validates the erase + per-channel color + I8
// black semantics). Every PartialRoi case is red (both backends) as a documented kernel finding
// under a non-full ROI: the kernel produces the correct packed-origin copy but does NOT apply
// the erase boxes. The golden holds to the absolute-frame box semantics
// (validated by coarse_dropout's passing HOST PartialRoi) -- do not weaken it to force green.
INSTANTIATE_TEST_SUITE_P(
    Image_Effects, CutoutDropoutTest,
    ::testing::ValuesIn(make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                                     {Layout::PKD3, Layout::PLN3, Layout::PLN1},
                                     {Roi::Full, Roi::Partial})),
    config_param_name);
