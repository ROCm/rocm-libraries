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
#include "reference/non_linear_blend_ref.hpp"

using namespace rpptest;

namespace {

// stdDev of the gaussian blend weight (>= 0). 15 gives a peaked gaussian (edges near src2), 50 a
// flat one (mostly src1) over a 36x48 image -- together they exercise the full weight range.
struct NonLinearBlendParams {
    float stdDev;
    std::string name() const {
        return "s" + num_token(stdDev);
    }
};

double non_linear_blend_tolerance(DType dt) {
    switch (dt) {
        case DType::U8:
            return 1.0;
        case DType::I8:
            return 1.0;
        case DType::F32:
            return 2e-3;
        case DType::F16:
            return 5e-3;
        default:
            return 0.0;
    }
    return 0.0;
}

template <typename T>
void run_non_linear_blend(const TestConfig& cfg, const NonLinearBlendParams& op) {
    const TensorShape shape{cfg.size.n, static_cast<Rpp32u>(channels_of(cfg.layout)), cfg.size.h,
                            cfg.size.w};
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layout);  // RPP takes a non-const ptr
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    PinnedArray<Rpp32f> stdDev(cfg.backend, shape.n);
    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < shape.n; ++i) {
        stdDev[i] = op.stdDev;
        roi[i] = roiVec[i];
    }

    // Two distinct operands (salt shifts the second) so the blend is non-degenerate.
    std::vector<T> input1(count), input2(count), golden(count), actual(count);
    fill_input<T>(input1.data(), count, cfg.dtype);
    fill_input<T>(input2.data(), count, cfg.dtype, 1);
    golden = input1;
    non_linear_blend_reference<T>(input1.data(), input2.data(), golden.data(), desc, cfg.dtype,
                                  roi.data(), XYWH, op.stdDev);

    DeviceTensor src1(cfg.backend, bytes), src2(cfg.backend, bytes), dst(cfg.backend, bytes);
    src1.write(input1.data(), bytes);
    src2.write(input2.data(), bytes);
    dst.write(input1.data(), bytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, shape.n);
    ASSERT_EQ(rppt_non_linear_blend(src1.ptr(), src2.ptr(), &desc, dst.ptr(), &desc, stdDev.data(),
                                    roi.data(), XYWH, handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back

    dst.read(actual.data(), bytes);

    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), desc, roi.data(), XYWH,
                               non_linear_blend_tolerance(cfg.dtype)));
}

}  // namespace

// Full name:
// Image_Effects/NonLinearBlendTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<StdDev>
class NonLinearBlendTest : public ::testing::TestWithParam<WithParams<NonLinearBlendParams>> {};

TEST_P(NonLinearBlendTest, Correctness) {
    const auto& p = GetParam();
    switch (p.cfg.dtype) {
        case DType::U8:
            run_non_linear_blend<Rpp8u>(p.cfg, p.op);
            break;
        case DType::F16:
            run_non_linear_blend<Rpp16f>(p.cfg, p.op);
            break;
        case DType::F32:
            run_non_linear_blend<Rpp32f>(p.cfg, p.op);
            break;
        case DType::I8:
            run_non_linear_blend<Rpp8s>(p.cfg, p.op);
            break;
        default:
            FAIL() << "Unsupported dtype for non_linear_blend";
    }
}

INSTANTIATE_TEST_SUITE_P(Image_Effects, NonLinearBlendTest,
                         ::testing::ValuesIn(with_params<NonLinearBlendParams>(
                             make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                                          {Layout::PKD3, Layout::PLN3, Layout::PLN1},
                                          {Roi::Full, Roi::Partial}),
                             {NonLinearBlendParams{15.0f}, NonLinearBlendParams{50.0f}})),
                         op_config_name<NonLinearBlendParams>);
