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
#include "reference/fisheye_ref.hpp"

using namespace rpptest;

namespace {

// Nearest-neighbour sampling copies a source texel verbatim and the op has no arithmetic, so every
// dtype is bit-exact. A diff is a disagreement about the map or the interpolation mode, which no
// tolerance could absorb anyway.
constexpr double kTol = 0.0;

template <typename T>
void run_fisheye(const TestConfig& cfg) {
    RpptDesc srcDesc = make_src_descriptor(cfg);
    RpptDesc dstDesc = make_dst_descriptor(cfg);
    const std::size_t count = element_count(srcDesc);
    const std::size_t bytes = byte_size(srcDesc, cfg.dtype);

    PinnedArray<RpptROI> roi(cfg.backend, cfg.size.n);
    const std::vector<RpptROI> roiVec = make_roi(srcDesc, cfg.roi);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) roi[i] = roiVec[i];

    // The op writes the ROI-sized region at the destination origin, so golden and device buffer
    // start from the same distinct pattern and only that region is compared.
    std::vector<T> input(count), dstInit(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    fill_input<T>(dstInit.data(), count, cfg.dtype, /*salt=*/1);
    golden = dstInit;
    fisheye_reference<T>(input.data(), srcDesc, golden.data(), dstDesc, cfg.dtype, roi.data(),
                         XYWH);

    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(dstInit.data(), bytes);

    RppHandle handle(cfg.backend, cfg.size.n);
    ASSERT_EQ(rppt_fisheye(src.ptr(), &srcDesc, dst.ptr(), &dstDesc, roi.data(), XYWH, handle.get(),
                           cfg.backend),
              RPP_SUCCESS);

    handle.sync();
    dst.read(actual.data(), bytes);

    // Compared against the caller's own ROI copy, not the tensor handed to the op: the HIP paths
    // rewrite that tensor from XYWH to LTRB in place, which would make the comparison walk a
    // different region than the golden wrote.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), dstDesc, roiVec.data(), XYWH, kTol));
}

}  // namespace

// Full name:
// Image_Geometric/FisheyeTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>
class FisheyeTest : public SkipListTest<TestConfig> {};

TEST_P(FisheyeTest, Correctness) {
    const TestConfig cfg = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(
        cfg.dtype, [&](auto tag) { run_fisheye<Element<decltype(tag)>>(cfg); });
}

// I8 is off the grid for now, pending the suite-wide decision on whether the image ops need it.
// Same-layout cases plus both directions of the fused output-layout conversion.
INSTANTIATE_TEST_SUITE_P(Image_Geometric, FisheyeTest,
                         ::testing::ValuesIn(make_configs({DType::U8, DType::F16, DType::F32},
                                                          {{Layout::PKD3, Layout::PKD3},
                                                           {Layout::PLN3, Layout::PLN3},
                                                           {Layout::PLN1, Layout::PLN1},
                                                           {Layout::PKD3, Layout::PLN3},
                                                           {Layout::PLN3, Layout::PKD3}},
                                                          {Roi::Full, Roi::Partial},
                                                          {presets::kDefaultSize,
                                                           presets::kTailWidthSize})),
                         config_param_name);
