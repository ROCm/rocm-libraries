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
#include "framework/tensor_setup.hpp"
#include "reference/grid_dropout_ref.hpp"

using namespace rpptest;

namespace {

// A regular gridW x gridH grid of half-cell holes is built deterministically from each image's
// ROI rectangle (no RNG), so the golden and the kernel see identical box geometry.
constexpr Rpp32u kGridW = 3;
constexpr Rpp32u kGridH = 3;
constexpr Rpp32u kBoxesInEachImage = kGridW * kGridH;  // 9

// Grid dropout is a pure copy/erase: kept pixels are copied bit-exact, holes are set to an
// exact constant (black). No arithmetic, so every dtype is bit-exact.
double grid_dropout_tolerance(DType) { return 0.0; }

template <typename T>
void run_grid_dropout(const TestConfig& cfg) {
    const TensorShape shape{cfg.size.n, static_cast<Rpp32u>(channels_of(cfg.layout)), cfg.size.h,
                            cfg.size.w};
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layout);  // RPP takes a non-const ptr
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    // anchorBoxInfoTensor: boxesInEachImage grid holes per image, laid out [n * stride + k].
    PinnedArray<RpptRoiLtrb> boxes(cfg.backend,
                                   static_cast<std::size_t>(shape.n) * kBoxesInEachImage);
    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);

    // Build the grid of holes deterministically from each image's ROI rectangle. maxHoleW/maxHoleH
    // are the max hole extents across all boxes (passed to the op; the golden ignores them).
    Rpp32u maxHoleW = 0, maxHoleH = 0;
    for (Rpp32u n = 0; n < shape.n; ++n) {
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
                const int x2 = std::min(x1 + static_cast<int>(holeW) - 1,
                                        static_cast<int>(rb.x0 + rb.w - 1));
                const int y2 = std::min(y1 + static_cast<int>(holeH) - 1,
                                        static_cast<int>(rb.y0 + rb.h - 1));
                boxes[n * kBoxesInEachImage + (row * kGridW + col)] =
                    RpptRoiLtrb{{x1, y1}, {x2, y2}};
            }
        }
    }

    // (1) Host golden model. golden starts as a copy of the input so the untouched
    // (outside-ROI) region is defined; only the ROI is overwritten by the reference.
    std::vector<T> input(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    golden = input;
    grid_dropout_reference<T>(input.data(), golden.data(), desc, cfg.dtype, roi.data(), XYWH,
                              boxes.data(), kBoxesInEachImage);

    // (2) Run RPP on the configured backend.
    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, shape.n);
    ASSERT_EQ(rppt_grid_dropout(src.ptr(), &desc, dst.ptr(), &desc, boxes.data(),
                                kBoxesInEachImage, maxHoleW, maxHoleH, roi.data(), XYWH,
                                handle.get(), cfg.backend),
              RPP_SUCCESS);

    // (3) Retrieve the result on the host (no-op copy for HOST, device->host for HIP).
    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    // (4) Compare within tolerance over the ROI.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), desc, roi.data(), XYWH,
                               grid_dropout_tolerance(cfg.dtype)));
}

}  // namespace

// Full name: Image_Effects/GridDropoutTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>
class GridDropoutTest : public ::testing::TestWithParam<TestConfig> {};

TEST_P(GridDropoutTest, Correctness) {
    const TestConfig cfg = GetParam();
    switch (cfg.dtype) {
        case DType::U8:
            run_grid_dropout<Rpp8u>(cfg);
            break;
        case DType::F16:
            run_grid_dropout<Rpp16f>(cfg);
            break;
        case DType::F32:
            run_grid_dropout<Rpp32f>(cfg);
            break;
        case DType::I8:
            run_grid_dropout<Rpp8s>(cfg);
            break;
        default:
            FAIL() << "unsupported dtype for grid_dropout";
    }
}

// FullRoi passes the full grid on both backends (validates the hole-erase + I8 black semantics).
// Every PartialRoi case is red (both backends) as a documented kernel finding
// under a non-full ROI: the kernel produces the correct packed-origin copy but does NOT erase
// the holes. The golden holds to the absolute-frame hole semantics (validated by coarse_dropout's
// passing HOST PartialRoi) -- do not weaken it to force green.
INSTANTIATE_TEST_SUITE_P(
    Image_Effects, GridDropoutTest,
    ::testing::ValuesIn(make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                                     {Layout::PKD3, Layout::PLN3, Layout::PLN1},
                                     {Roi::Full, Roi::Partial})),
    config_param_name);
