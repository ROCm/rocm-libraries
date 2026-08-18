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

#include <string>
#include <vector>

#include "framework/backend_memory.hpp"
#include "framework/compare_tensor.hpp"
#include "framework/config_param.hpp"
#include "framework/dtype_dispatch.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/gaussian_noise_ref.hpp"

using namespace rpptest;

namespace {

// gaussian_noise is covered at its RNG-free corner only. The Correctness intent pins
// (mean 0, stdDev 0), where N(mean, stdDev) collapses to a point mass at 0 and the additive noise
// degenerates to a passthrough -- bit-exact and seed-independent (see gaussian_noise_ref.hpp).
// Away from that corner the output depends on the kernel's Box-Muller stream, which the public API
// does not describe, so no golden is possible; the Negative intent covers the documented parameter
// contract instead. The two take disjoint parameter sets, so they are separate fixtures.
constexpr Rpp32u kSeed = 42u;

template <typename T>
void run_gaussian_noise(const TestConfig& cfg) {
    const TensorShape shape{cfg.size.n, static_cast<Rpp32u>(channels_of(cfg.layout)), cfg.size.h,
                            cfg.size.w};
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layout);  // RPP takes a non-const ptr
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    PinnedArray<Rpp32f> mean(cfg.backend, shape.n), stdDev(cfg.backend, shape.n);
    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < shape.n; ++i) {
        mean[i] = 0.0f;  // the RNG-free identity corner
        stdDev[i] = 0.0f;
        roi[i] = roiVec[i];
    }

    std::vector<T> input(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    golden = input;
    gaussian_noise_identity_reference<T>(input.data(), golden.data(), desc, roi.data(), XYWH);

    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, shape.n);
    ASSERT_EQ(rppt_gaussian_noise(src.ptr(), &desc, dst.ptr(), &desc, mean.data(), stdDev.data(),
                                  kSeed, roi.data(), XYWH, handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    // Compared against the test's own ROI copy, not the tensor handed to the op, in case the
    // backend rewrites it from XYWH to LTRB in place.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), desc, roiVec.data(), XYWH, 0.0));
}

// One out-of-contract parameter pair per case. The header requires meanTensor[i] >= 0 and
// stdDevTensor[i] >= 0; the parameter values are the grid axis here because argument validation is
// dtype- and layout-independent.
struct GaussianNoiseNegativeParams {
    float mean;
    float stdDev;
    std::string name() const { return "m" + num_token(mean) + "_s" + num_token(stdDev); }
};

// A negative mean or standard deviation is not a legal call and must be reported rather than
// silently producing an undefined image -- a negative stdDev in particular has no meaning for
// Box-Muller, which scales by it. Only the status is asserted: the header does not name which
// RPP_ERROR* an out-of-range scalar maps to.
template <typename T>
void run_gaussian_noise_negative(const TestConfig& cfg, const GaussianNoiseNegativeParams& op) {
    const TensorShape shape{cfg.size.n, static_cast<Rpp32u>(channels_of(cfg.layout)), cfg.size.h,
                            cfg.size.w};
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layout);
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    PinnedArray<Rpp32f> mean(cfg.backend, shape.n), stdDev(cfg.backend, shape.n);
    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < shape.n; ++i) {
        mean[i] = op.mean;
        stdDev[i] = op.stdDev;
        roi[i] = roiVec[i];
    }

    std::vector<T> input(count);
    fill_input<T>(input.data(), count, cfg.dtype);

    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);

    RppHandle handle(cfg.backend, shape.n);
    const RppStatus status =
        rppt_gaussian_noise(src.ptr(), &desc, dst.ptr(), &desc, mean.data(), stdDev.data(), kSeed,
                            roi.data(), XYWH, handle.get(), cfg.backend);
    handle.sync();
    EXPECT_NE(status, RPP_SUCCESS) << "gaussian_noise accepted out-of-contract parameters (mean "
                                   << op.mean << ", stdDev " << op.stdDev << ")";
}

}  // namespace

// Full name:
// Image_Effects/GaussianNoiseTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>
class GaussianNoiseTest : public ::testing::TestWithParam<TestConfig> {};

TEST_P(GaussianNoiseTest, Correctness) {
    const TestConfig& cfg = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(cfg.dtype, [&](auto tag) {
        run_gaussian_noise<Element<decltype(tag)>>(cfg);
    });
}

INSTANTIATE_TEST_SUITE_P(Image_Effects, GaussianNoiseTest,
                         ::testing::ValuesIn(make_configs(
                             {DType::U8, DType::F16, DType::F32, DType::I8},
                             {Layout::PKD3, Layout::PLN3, Layout::PLN1},
                             {Roi::Full, Roi::Partial})),
                         config_param_name);

// Full name:
// Image_Effects/GaussianNoiseNegativeTest.Negative/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_m<M>_s<S>
class GaussianNoiseNegativeTest
    : public ::testing::TestWithParam<WithParams<GaussianNoiseNegativeParams>> {};

TEST_P(GaussianNoiseNegativeTest, Negative) {
    const auto& p = GetParam();
    run_gaussian_noise_negative<Rpp8u>(p.cfg, p.op);
}

INSTANTIATE_TEST_SUITE_P(Image_Effects, GaussianNoiseNegativeTest,
                         ::testing::ValuesIn(with_params<GaussianNoiseNegativeParams>(
                             make_configs({DType::U8}, {Layout::PKD3}, {Roi::Full}),
                             {GaussianNoiseNegativeParams{-0.5f, 1.0f},
                              GaussianNoiseNegativeParams{0.5f, -1.0f}})),
                         op_config_name<GaussianNoiseNegativeParams>);
