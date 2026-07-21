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
    }
    return 0.0;
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
    }
}

INSTANTIATE_TEST_SUITE_P(
    Image_Effects, SolarizeTest,
    ::testing::ValuesIn(with_params<SolarizeParams>(
        make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                     {Layout::PKD3, Layout::PLN3, Layout::PLN1}, {Roi::Full, Roi::Partial}),
        {SolarizeParams{0.5f}})),
    op_config_name<SolarizeParams>);
