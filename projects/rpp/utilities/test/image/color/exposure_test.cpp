#include <gtest/gtest.h>
#include <rpp/rpp.h>

#include <vector>

#include "framework/backend_memory.hpp"
#include "framework/compare_tensor.hpp"
#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/exposure_ref.hpp"

using namespace rpptest;

namespace {

// exposureFactor 0.5 => multiplier 2^0.5 ~= 1.414.
struct ExposureParams {
    float exposureFactor;
    std::string name() const { return "e" + num_token(exposureFactor); }
};

double exposure_tolerance(DType dt) {
    switch (dt) {
        case DType::U8:
            return 1.0;
        // I8 kept sub-LSB to surface a real kernel bug: RPP truncates the I8 result instead of
        // rounding (HOST scalar remainder + all of HIP), biasing it low. See section 13 of the
        // test-suite-revamp plan.
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
void run_exposure(const TestConfig& cfg, const ExposureParams& op) {
    const TensorShape shape{cfg.size.n, static_cast<Rpp32u>(channels_of(cfg.layout)), cfg.size.h,
                            cfg.size.w};
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layout);  // RPP takes a non-const ptr
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    PinnedArray<Rpp32f> exposureFactor(cfg.backend, shape.n);
    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < shape.n; ++i) {
        exposureFactor[i] = op.exposureFactor;
        roi[i] = roiVec[i];
    }

    std::vector<T> input(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    golden = input;
    exposure_reference<T>(input.data(), golden.data(), desc, cfg.dtype, roi.data(), XYWH,
                          op.exposureFactor);

    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, shape.n);
    ASSERT_EQ(rppt_exposure(src.ptr(), &desc, dst.ptr(), &desc, exposureFactor.data(), roi.data(),
                            XYWH, handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back

    dst.read(actual.data(), bytes);

    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), desc, roi.data(), XYWH,
                               exposure_tolerance(cfg.dtype)));
}

}  // namespace

// Full name: Image_Color/ExposureTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<Exposure>
class ExposureTest : public ::testing::TestWithParam<WithParams<ExposureParams>> {};

TEST_P(ExposureTest, Correctness) {
    const auto& p = GetParam();
    switch (p.cfg.dtype) {
        case DType::U8:
            run_exposure<Rpp8u>(p.cfg, p.op);
            break;
        case DType::F16:
            run_exposure<Rpp16f>(p.cfg, p.op);
            break;
        case DType::F32:
            run_exposure<Rpp32f>(p.cfg, p.op);
            break;
        case DType::I8:
            run_exposure<Rpp8s>(p.cfg, p.op);
            break;
    }
}

INSTANTIATE_TEST_SUITE_P(
    Image_Color, ExposureTest,
    ::testing::ValuesIn(with_params<ExposureParams>(
        make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                     {Layout::PKD3, Layout::PLN3, Layout::PLN1}, {Roi::Full, Roi::Partial}),
        {ExposureParams{0.5f}})),
    op_config_name<ExposureParams>);
