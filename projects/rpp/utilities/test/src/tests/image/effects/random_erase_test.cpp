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
#include "framework/compare_tensor.hpp"
#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/random_erase_ref.hpp"

using namespace rpptest;

namespace {

// random_erase fills its single erase-region with a direct noiseBuffer lookup: a tiled store, no
// arithmetic, so every dtype is bit-exact.
double random_erase_tolerance(DType) { return 0.0; }

template <typename T>
void run_random_erase(const TestConfig& cfg) {
    const Rpp32u channels = static_cast<Rpp32u>(channels_of(cfg.layout));
    const TensorShape shape{cfg.size.n, channels, cfg.size.h, cfg.size.w};
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layout);  // RPP takes a non-const ptr
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u n = 0; n < shape.n; ++n) roi[n] = roiVec[n];

    // One box per image, built relative to each image's ROI rectangle so it stays inside the ROI
    // for both Full and Partial. The box's absolute origin is generally not a multiple of 255, so
    // the mod-255 tiling phase is exercised even though the box itself is smaller than the tile.
    PinnedArray<RpptRoiLtrb> boxes(cfg.backend, shape.n);
    for (Rpp32u n = 0; n < shape.n; ++n) {
        const RoiBounds rb = roi_bounds(roiVec[n], XYWH);
        RpptRoiLtrb b0{};
        b0.lt.x = static_cast<int>(rb.x0 + rb.w / 8);
        b0.lt.y = static_cast<int>(rb.y0 + rb.h / 8);
        b0.rb.x = b0.lt.x + static_cast<int>(rb.w / 4) - 1;
        b0.rb.y = b0.lt.y + static_cast<int>(rb.h / 4) - 1;
        boxes[n] = b0;
    }

    // noiseBuffer: 255*255*channels elements, same dtype/range as the image. Filled with a
    // deterministic non-constant pattern (distinct salt from the image fill) so a wrong tile
    // offset/addressing produces a visible diff.
    const std::size_t noiseCount = static_cast<std::size_t>(255) * 255 * channels;
    PinnedArray<T> noiseBuffer(cfg.backend, noiseCount);
    fill_input<T>(noiseBuffer.data(), noiseCount, cfg.dtype, /*salt=*/17);

    // (1) Host golden model. golden starts as a copy of the input so the untouched (outside-ROI)
    // region is defined; only the box is overwritten by the reference.
    std::vector<T> input(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    golden = input;
    random_erase_reference<T>(input.data(), golden.data(), desc, cfg.dtype, roi.data(), XYWH,
                              boxes.data(), noiseBuffer.data());

    // (2) Run RPP on the configured backend.
    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, shape.n);
    ASSERT_EQ(rppt_random_erase(src.ptr(), &desc, dst.ptr(), &desc, boxes.data(),
                                static_cast<void*>(noiseBuffer.data()), roi.data(), XYWH,
                                handle.get(), cfg.backend),
              RPP_SUCCESS);

    // (3) Retrieve the result on the host (no-op copy for HOST, device->host for HIP).
    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    // (4) Compare within tolerance over the ROI.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), desc, roi.data(), XYWH,
                               random_erase_tolerance(cfg.dtype)));
}

}  // namespace

// Full name: Image_Effects/RandomEraseTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>
//
// There is no seed parameter anywhere in this API: box placement and noiseBuffer contents are both
// supplied by the caller, so the golden is fully bit-exact and deterministic (unlike jitter /
// noise_shot, which sample their own RNG state internally).
class RandomEraseTest : public ::testing::TestWithParam<TestConfig> {};

TEST_P(RandomEraseTest, Correctness) {
    const auto& cfg = GetParam();
    switch (cfg.dtype) {
        case DType::U8:
            run_random_erase<Rpp8u>(cfg);
            break;
        case DType::F16:
            run_random_erase<Rpp16f>(cfg);
            break;
        case DType::F32:
            run_random_erase<Rpp32f>(cfg);
            break;
        case DType::I8:
            run_random_erase<Rpp8s>(cfg);
            break;
        default:
            FAIL() << "unsupported dtype for random_erase";
    }
}

INSTANTIATE_TEST_SUITE_P(
    Image_Effects, RandomEraseTest,
    ::testing::ValuesIn(make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                                     {Layout::PKD3, Layout::PLN3, Layout::PLN1},
                                     {Roi::Full, Roi::Partial})),
    config_param_name);
