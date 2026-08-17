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
#include "reference/solarize_ref.hpp"

using namespace rpptest;

namespace {

// threshold 0.5 is the midpoint so both branches (invert / passthrough) are hit.
struct SolarizeParams {
    float threshold;
    std::string name() const { return "t" + num_token(threshold); }
};

double solarize_tolerance(DType dt) {
    switch (dt) {
        case DType::U8:
        case DType::I8:
            return 0.0;  // integer inversion is exact
        case DType::F32:
            return 1e-4;
        case DType::F16:
            return 2e-3;
        default:
            return 0.0;
    }
}

template <typename T>
void run_solarize(const TestConfig& cfg, const SolarizeParams& op) {
    const TensorShape shape{cfg.size.n, static_cast<Rpp32u>(channels_of(cfg.layout)), cfg.size.h,
                            cfg.size.w};
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layout);  // RPP takes a non-const ptr
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    PinnedArray<Rpp32f> threshold(cfg.backend, shape.n);
    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < shape.n; ++i) {
        threshold[i] = op.threshold;
        roi[i] = roiVec[i];
    }

    std::vector<T> input(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    golden = input;
    solarize_reference<T>(input.data(), golden.data(), desc, cfg.dtype, roi.data(), XYWH,
                          op.threshold);

    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, shape.n);
    ASSERT_EQ(rppt_solarize(src.ptr(), &desc, dst.ptr(), &desc, threshold.data(), roi.data(), XYWH,
                            handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back

    dst.read(actual.data(), bytes);

    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), desc, roi.data(), XYWH,
                               solarize_tolerance(cfg.dtype)));
}

}  // namespace

// Full name: Image_Effects/SolarizeTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<Threshold>
class SolarizeTest : public ::testing::TestWithParam<WithParams<SolarizeParams>> {};

TEST_P(SolarizeTest, Correctness) {
    const auto& p = GetParam();
    switch (p.cfg.dtype) {
        case DType::U8:
            run_solarize<Rpp8u>(p.cfg, p.op);
            break;
        case DType::F16:
            run_solarize<Rpp16f>(p.cfg, p.op);
            break;
        case DType::F32:
            run_solarize<Rpp32f>(p.cfg, p.op);
            break;
        case DType::I8:
            run_solarize<Rpp8s>(p.cfg, p.op);
            break;
        default:
            FAIL() << "unsupported dtype for solarize";
    }
}

INSTANTIATE_TEST_SUITE_P(
    Image_Effects, SolarizeTest,
    ::testing::ValuesIn(with_params<SolarizeParams>(
        make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                     {Layout::PKD3, Layout::PLN3, Layout::PLN1}, {Roi::Full, Roi::Partial}),
        {SolarizeParams{0.5f}})),
    op_config_name<SolarizeParams>);
