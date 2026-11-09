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
#include "framework/reduction.hpp"
#include "framework/skip_list.hpp"
#include "framework/tensor_setup.hpp"
#include "framework/tolerance.hpp"
#include "reference/tensor_mean_ref.hpp"
#include "reference/tensor_stddev_ref.hpp"

using namespace rpptest;

namespace {

// Covers accumulation, division and sqrt rounding; integer-space stddevs are larger, hence the
// wider tolerance.
constexpr Tolerance kTensorStddevTolerance = tolerance(1e-1, 1e-3, 1e-2);

// stddev outputs Rpp32f and takes an n*4 meanTensor ordered [MeanR,MeanG,MeanB,MeanImage].
template <typename Tin>
void run_tensor_stddev(const TestConfig& cfg) {
    RpptDesc desc = make_src_descriptor(cfg);  // RPP takes a non-const ptr
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);
    const std::size_t stride = reduction_stride(desc);
    const std::size_t outLen = reduction_length(desc);

    PinnedArray<RpptROI> roi(cfg.backend, cfg.size.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) roi[i] = roiVec[i];

    std::vector<Tin> input(count);
    fill_input<Tin>(input.data(), count, cfg.dtype);

    // Feed the same float-rounded means to the kernel and the reference. A 1-channel image puts
    // its single mean in all four slots.
    const std::vector<double> meanDbl =
        tensor_mean_reference<Tin>(input.data(), desc, roi.data(), XYWH);
    PinnedArray<Rpp32f> mean(cfg.backend, static_cast<std::size_t>(cfg.size.n) * 4);
    std::vector<double> meanRef(outLen);
    for (Rpp32u n = 0; n < cfg.size.n; ++n) {
        if (desc.c == 3) {
            for (int k = 0; k < 4; ++k) {
                mean[n * 4 + k] = static_cast<Rpp32f>(meanDbl[n * stride + k]);
                meanRef[n * stride + k] = static_cast<double>(mean[n * 4 + k]);
            }
        } else {
            const Rpp32f m = static_cast<Rpp32f>(meanDbl[n]);
            for (int k = 0; k < 4; ++k) mean[n * 4 + k] = m;
            meanRef[n] = static_cast<double>(m);
        }
    }

    const std::vector<double> golden =
        tensor_stddev_reference<Tin>(input.data(), desc, roi.data(), XYWH, meanRef);

    DeviceTensor src(cfg.backend, bytes);
    src.write(input.data(), bytes);
    PinnedArray<Rpp32f> out(cfg.backend, outLen);
    for (std::size_t i = 0; i < outLen; ++i) out[i] = 0.0f;

    RppHandle handle(cfg.backend, cfg.size.n);
    ASSERT_EQ(rppt_tensor_stddev(src.ptr(), &desc, out.data(), static_cast<Rpp32u>(outLen),
                                 mean.data(), roi.data(), XYWH, handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // pinned output is then directly readable on the host

    EXPECT_TRUE(compare_reduction<Rpp32f>(out.data(), golden, kTensorStddevTolerance(cfg.dtype)));
}

}  // namespace

// Full name:
// Image_Statistical/TensorStddevTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>
class TensorStddevTest : public SkipListTest<TestConfig> {};

TEST_P(TensorStddevTest, Correctness) {
    const TestConfig cfg = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(
        cfg.dtype, [&](auto tag) { run_tensor_stddev<Element<decltype(tag)>>(cfg); });
}

INSTANTIATE_TEST_SUITE_P(Image_Statistical, TensorStddevTest,
                         ::testing::ValuesIn(make_configs(
                             {DType::U8, DType::F16, DType::F32, DType::I8},
                             {Layout::PKD3, Layout::PLN3, Layout::PLN1}, {Roi::Full, Roi::Partial},
                             // Same shapes as tensor_min/max.
                             {{2, 160, 160}, {2, 137, 167}})),
                         config_param_name);
