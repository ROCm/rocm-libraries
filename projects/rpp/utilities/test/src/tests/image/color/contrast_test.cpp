#include <gtest/gtest.h>
#include <rpp/rpp.h>

#include <vector>

#include "framework/backend_memory.hpp"
#include "framework/compare_tensor.hpp"
#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/contrast_ref.hpp"

using namespace rpptest;

namespace {

// contrastFactor within the documented range (factor > 0); contrastCenter in [0,255] pixel units.
struct ContrastParams {
    float factor, center;
    std::string name() const { return "f" + num_token(factor) + "_c" + num_token(center); }
};

double contrast_tolerance(DType dt) {
    switch (dt) {
        case DType::U8:
            return 1.0;
        // I8 kept sub-LSB to surface the systemic kernel bug: RPP truncates the I8 result instead
        // of rounding (HOST scalar remainder + all of HIP), biasing it low by 1 vs U8.
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
void run_contrast(const TestConfig& cfg, const ContrastParams& op) {
    const TensorShape shape{cfg.size.n, static_cast<Rpp32u>(channels_of(cfg.layout)), cfg.size.h,
                            cfg.size.w};
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layout);  // RPP takes a non-const ptr
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    // Parameters live in host-accessible (pinned for HIP) memory.
    PinnedArray<Rpp32f> factor(cfg.backend, shape.n);
    PinnedArray<Rpp32f> center(cfg.backend, shape.n);
    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < shape.n; ++i) {
        factor[i] = op.factor;
        center[i] = op.center;
        roi[i] = roiVec[i];
    }

    // (1) Host golden model. golden starts as a copy of the input so the untouched
    // (outside-ROI) region is defined; only the ROI is overwritten by the reference.
    std::vector<T> input(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    golden = input;
    contrast_reference<T>(input.data(), golden.data(), desc, cfg.dtype, roi.data(), XYWH, op.factor,
                          op.center);

    // (2) Run RPP on the configured backend.
    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, shape.n);
    ASSERT_EQ(rppt_contrast(src.ptr(), &desc, dst.ptr(), &desc, factor.data(), center.data(),
                            roi.data(), XYWH, handle.get(), cfg.backend),
              RPP_SUCCESS);

    // (3) Retrieve the result on the host (no-op copy for HOST, device->host for HIP).
    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    // (4) Compare within tolerance over the ROI.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), desc, roi.data(), XYWH,
                               contrast_tolerance(cfg.dtype)));
}

}  // namespace

// Full name: Image_Color/ContrastTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<Factor>_<Center>
class ContrastTest : public ::testing::TestWithParam<WithParams<ContrastParams>> {};

TEST_P(ContrastTest, Correctness) {
    const auto& p = GetParam();
    switch (p.cfg.dtype) {
        case DType::U8:
            run_contrast<Rpp8u>(p.cfg, p.op);
            break;
        case DType::F16:
            run_contrast<Rpp16f>(p.cfg, p.op);
            break;
        case DType::F32:
            run_contrast<Rpp32f>(p.cfg, p.op);
            break;
        case DType::I8:
            run_contrast<Rpp8s>(p.cfg, p.op);
            break;
    }
}

INSTANTIATE_TEST_SUITE_P(
    Image_Color, ContrastTest,
    ::testing::ValuesIn(with_params<ContrastParams>(
        make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                     {Layout::PKD3, Layout::PLN3, Layout::PLN1}, {Roi::Full, Roi::Partial}),
        {ContrastParams{1.75f, 128.0f}})),
    op_config_name<ContrastParams>);
