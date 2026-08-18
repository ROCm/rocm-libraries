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
#include "framework/config_param.hpp"
#include "framework/dtype_dispatch.hpp"
#include "framework/reduction.hpp"
#include "framework/tensor_setup.hpp"
#include "framework/tolerance.hpp"
#include "reference/tensor_mean_ref.hpp"

using namespace rpptest;

namespace {

// mean is sum/N in float; the tolerance covers accumulation/division rounding. Integer-space
// means (U8 [0,255], I8 [-128,127]) carry a larger magnitude than the [0,1] float means, hence
// the wider absolute tolerance.
constexpr Tolerance kTensorMeanTolerance = tolerance(1e-2, 1e-4, 1e-3);

// mean always outputs Rpp32f regardless of the source dtype (per the API contract).
template <typename Tin>
void run_tensor_mean(const TestConfig& cfg) {
    const TensorShape shape{cfg.size.n, static_cast<Rpp32u>(channels_of(cfg.layout)), cfg.size.h,
                            cfg.size.w};
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layout);  // RPP takes a non-const ptr
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);
    const std::size_t outLen = reduction_length(desc);

    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < shape.n; ++i) roi[i] = roiVec[i];

    // (1) Host golden model.
    std::vector<Tin> input(count);
    fill_input<Tin>(input.data(), count, cfg.dtype);
    const std::vector<double> golden =
        tensor_mean_reference<Tin>(input.data(), desc, roi.data(), XYWH);

    // (2) Run RPP on the configured backend. The result array is host-accessible (pinned for HIP).
    DeviceTensor src(cfg.backend, bytes);
    src.write(input.data(), bytes);
    PinnedArray<Rpp32f> out(cfg.backend, outLen);
    for (std::size_t i = 0; i < outLen; ++i) out[i] = 0.0f;

    RppHandle handle(cfg.backend, shape.n);
    ASSERT_EQ(rppt_tensor_mean(src.ptr(), &desc, out.data(), static_cast<Rpp32u>(outLen),
                               roi.data(), XYWH, handle.get(), cfg.backend),
              RPP_SUCCESS);

    // (3) Drain the op's stream; pinned output is then directly readable on the host.
    handle.sync();

    // (4) Compare within tolerance.
    EXPECT_TRUE(compare_reduction<Rpp32f>(out.data(), golden, kTensorMeanTolerance(cfg.dtype)));
}

}  // namespace

// Full name:
// Image_Statistical/TensorMeanTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>
class TensorMeanTest : public ::testing::TestWithParam<TestConfig> {};

TEST_P(TensorMeanTest, Correctness) {
    const TestConfig cfg = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(cfg.dtype, [&](auto tag) {
        run_tensor_mean<Element<decltype(tag)>>(cfg);
    });
}

INSTANTIATE_TEST_SUITE_P(
    Image_Statistical, TensorMeanTest,
    ::testing::ValuesIn(make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                                     {Layout::PKD3, Layout::PLN3, Layout::PLN1},
                                     {Roi::Full, Roi::Partial},
                                     // Kept in step with tensor_min/max's larger size so the
                                     // reduction ops share one grid.
                                     {{2, 160, 160}})),
    config_param_name);
