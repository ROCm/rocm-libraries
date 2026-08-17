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
#include "reference/fisheye_ref.hpp"

using namespace rpptest;

namespace {

// Nearest-neighbour sampling copies a source texel verbatim and the op has no arithmetic, so every
// dtype is bit-exact. A diff is a disagreement about the map or the interpolation mode, which no
// tolerance could absorb anyway.
constexpr double kTol = 0.0;

template <typename T>
void run_fisheye(const TestConfig& cfg) {
    const TensorShape shape{cfg.size.n, static_cast<Rpp32u>(channels_of(cfg.layout)), cfg.size.h,
                            cfg.size.w};
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layout);  // src == dst
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < shape.n; ++i) roi[i] = roiVec[i];

    // The op writes the ROI-sized region at the destination origin, so golden and device buffer
    // start from the same distinct pattern and only that region is compared.
    std::vector<T> input(count), dstInit(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    fill_input<T>(dstInit.data(), count, cfg.dtype, /*salt=*/1);
    golden = dstInit;
    fisheye_reference<T>(input.data(), golden.data(), desc, cfg.dtype, roi.data(), XYWH);

    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(dstInit.data(), bytes);

    RppHandle handle(cfg.backend, shape.n);
    ASSERT_EQ(rppt_fisheye(src.ptr(), &desc, dst.ptr(), &desc, roi.data(), XYWH, handle.get(),
                           cfg.backend),
              RPP_SUCCESS);

    handle.sync();
    dst.read(actual.data(), bytes);

    // Compared against the caller's own ROI copy, not the tensor handed to the op: the HIP paths
    // rewrite that tensor from XYWH to LTRB in place, which would make the comparison walk a
    // different region than the golden wrote.
    EXPECT_TRUE(
        compare_roi<T>(actual.data(), golden.data(), desc, roiVec.data(), XYWH, kTol));
}

}  // namespace

// Full name: Image_Geometric/FisheyeTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>
class FisheyeTest : public ::testing::TestWithParam<TestConfig> {};

TEST_P(FisheyeTest, Correctness) {
    const TestConfig cfg = GetParam();
    switch (cfg.dtype) {
        case DType::U8:
            run_fisheye<Rpp8u>(cfg);
            break;
        case DType::F16:
            run_fisheye<Rpp16f>(cfg);
            break;
        case DType::F32:
            run_fisheye<Rpp32f>(cfg);
            break;
        case DType::I8:
            run_fisheye<Rpp8s>(cfg);
            break;
        default:
            FAIL() << "unsupported dtype for fisheye";
            break;
    }
}

INSTANTIATE_TEST_SUITE_P(
    Image_Geometric, FisheyeTest,
    // I8 is off the grid for now, pending the suite-wide decision on whether the image ops need it.
    ::testing::ValuesIn(make_configs({DType::U8, DType::F16, DType::F32},
                                     {Layout::PKD3, Layout::PLN3, Layout::PLN1},
                                     {Roi::Full, Roi::Partial})),
    config_param_name);
