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
#include "reference/lut_ref.hpp"

using namespace rpptest;

namespace {

// The kernel takes a single integer look-up table (documented length 65536) shared by the
// whole batch. Only the 256 intensity slots are meaningful for the integer dtypes; the rest
// are defined to avoid any indeterminate reads. An inverting table (intensity j -> 255 - j)
// is used so the mapping is easy to reason about.
template <typename T>
void fill_lut(T* lut, DType dt) {
    for (std::size_t j = 0; j < 65536; ++j) {
        int value = 0;
        if (j < 256) value = 255 - static_cast<int>(j);          // inverted intensity in [0,255]
        if (dt == DType::I8) value -= 128;                       // shift into I8's signed range
        lut[j] = static_cast<T>(value);
    }
}

template <typename T>
void run_lut(const TestConfig& cfg) {
    const TensorShape shape{cfg.size.n, static_cast<Rpp32u>(channels_of(cfg.layout)), cfg.size.h,
                            cfg.size.w};
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layout);  // RPP takes a non-const ptr
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    // The table is host/pinned memory even for HIP (the API keeps lutPtr host-side).
    PinnedArray<T> lut(cfg.backend, 65536);
    fill_lut<T>(lut.data(), cfg.dtype);

    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < shape.n; ++i) roi[i] = roiVec[i];

    // (1) Host golden model. golden starts as a copy of the input so the untouched
    // (outside-ROI) region is defined; only the ROI is overwritten by the reference.
    std::vector<T> input(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    golden = input;
    lut_reference<T>(input.data(), golden.data(), desc, cfg.dtype, roi.data(), XYWH, lut.data());

    // (2) Run RPP on the configured backend.
    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, shape.n);
    ASSERT_EQ(rppt_lut(src.ptr(), &desc, dst.ptr(), &desc, lut.data(), roi.data(), XYWH,
                       handle.get(), cfg.backend),
              RPP_SUCCESS);

    // (3) Retrieve the result on the host (no-op copy for HOST, device->host for HIP).
    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    // (4) Compare over the ROI. A look-up is a bit-exact copy of a table entry, so tolerance 0.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), desc, roi.data(), XYWH, 0.0));
}

}  // namespace

// Full name: Image_Color/LutTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>
class LutTest : public ::testing::TestWithParam<TestConfig> {};

TEST_P(LutTest, Correctness) {
    const auto& cfg = GetParam();
    switch (cfg.dtype) {
        case DType::U8:
            run_lut<Rpp8u>(cfg);
            break;
        case DType::I8:
            run_lut<Rpp8s>(cfg);
            break;
        default:
            FAIL() << "unsupported dtype for lut";
    }
}

// Restricted to the integer dtypes (U8/I8): the 256-entry intensity index is unambiguous
// there, whereas the float dtypes' table-index semantics are not defined by the public API.
// All layouts (c = 1/3) are supported.
INSTANTIATE_TEST_SUITE_P(
    Image_Color, LutTest,
    ::testing::ValuesIn(make_configs({DType::U8, DType::I8},
                                     {Layout::PKD3, Layout::PLN3, Layout::PLN1},
                                     {Roi::Full, Roi::Partial})),
    config_param_name);
