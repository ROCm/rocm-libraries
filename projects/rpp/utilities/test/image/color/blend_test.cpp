#include <gtest/gtest.h>
#include <rpp/rpp.h>

#include <vector>

#include "framework/backend_memory.hpp"
#include "framework/compare_tensor.hpp"
#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/blend_ref.hpp"

using namespace rpptest;

namespace {

// alpha 0.75 is exactly representable in float, so integer rounding is unambiguous.
struct BlendParams {
    float alpha;
    std::string name() const { return "a" + num_token(alpha); }
};

double blend_tolerance(DType dt) {
    switch (dt) {
        case DType::U8:
            return 1.0;
        // I8 kept sub-LSB to surface a real kernel bug: RPP truncates the I8 result instead of
        // rounding (HOST scalar remainder + all of HIP), biasing it toward zero. See section 13 of
        // the test-suite-revamp plan.
        case DType::I8:
            return 0.5;
        case DType::F32:
            return 2e-3;
        case DType::F16:
            return 5e-3;
    }
    return 0.0;
}

template <typename T>
void run_blend(const TestConfig& cfg, const BlendParams& op) {
    const TensorShape shape{cfg.size.n, static_cast<Rpp32u>(channels_of(cfg.layout)), cfg.size.h,
                            cfg.size.w};
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layout);  // RPP takes a non-const ptr
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    PinnedArray<Rpp32f> alpha(cfg.backend, shape.n);
    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < shape.n; ++i) {
        alpha[i] = op.alpha;
        roi[i] = roiVec[i];
    }

    // Two distinct operands (salt shifts the second) so alpha is meaningfully exercised.
    std::vector<T> input1(count), input2(count), golden(count), actual(count);
    fill_input<T>(input1.data(), count, cfg.dtype);
    fill_input<T>(input2.data(), count, cfg.dtype, 1);
    golden = input1;
    blend_reference<T>(input1.data(), input2.data(), golden.data(), desc, cfg.dtype, roi.data(),
                       XYWH, op.alpha);

    DeviceTensor src1(cfg.backend, bytes), src2(cfg.backend, bytes), dst(cfg.backend, bytes);
    src1.write(input1.data(), bytes);
    src2.write(input2.data(), bytes);
    dst.write(input1.data(), bytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, shape.n);
    ASSERT_EQ(rppt_blend(src1.ptr(), src2.ptr(), &desc, dst.ptr(), &desc, alpha.data(), roi.data(),
                         XYWH, handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back

    dst.read(actual.data(), bytes);

    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), desc, roi.data(), XYWH,
                               blend_tolerance(cfg.dtype)));
}

}  // namespace

// Full name: Image_Color/BlendTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<Alpha>
class BlendTest : public ::testing::TestWithParam<WithParams<BlendParams>> {};

TEST_P(BlendTest, Correctness) {
    const auto& p = GetParam();
    switch (p.cfg.dtype) {
        case DType::U8:
            run_blend<Rpp8u>(p.cfg, p.op);
            break;
        case DType::F16:
            run_blend<Rpp16f>(p.cfg, p.op);
            break;
        case DType::F32:
            run_blend<Rpp32f>(p.cfg, p.op);
            break;
        case DType::I8:
            run_blend<Rpp8s>(p.cfg, p.op);
            break;
    }
}

INSTANTIATE_TEST_SUITE_P(
    Image_Color, BlendTest,
    ::testing::ValuesIn(with_params<BlendParams>(
        make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                     {Layout::PKD3, Layout::PLN3, Layout::PLN1}, {Roi::Full, Roi::Partial}),
        {BlendParams{0.75f}})),
    op_config_name<BlendParams>);
