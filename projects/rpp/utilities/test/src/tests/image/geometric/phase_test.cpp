#include <gtest/gtest.h>
#include <rpp/rpp.h>

#include <vector>

#include "framework/backend_memory.hpp"
#include "framework/compare_tensor.hpp"
#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/phase_ref.hpp"

using namespace rpptest;

namespace {

double phase_tolerance(DType dt) {
    switch (dt) {
        case DType::U8:
            return 1.0;
        // I8 kept sub-LSB to surface the systemic I8 round-vs-truncate defect: HIP truncates all I8
        // (actual = ref - 1), HOST truncates the partial-ROI scalar tail (here actual = ref + 1 as
        // phase's negative I8 output truncates toward zero).
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
void run_phase(const TestConfig& cfg) {
    const TensorShape shape{cfg.size.n, static_cast<Rpp32u>(channels_of(cfg.layout)), cfg.size.h,
                            cfg.size.w};
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layout);  // RPP takes a non-const ptr
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < shape.n; ++i) roi[i] = roiVec[i];

    // Two distinct operands (salt shifts the second) so the phase angle is non-degenerate.
    std::vector<T> input1(count), input2(count), golden(count), actual(count);
    fill_input<T>(input1.data(), count, cfg.dtype);
    fill_input<T>(input2.data(), count, cfg.dtype, 1);
    golden = input1;
    phase_reference<T>(input1.data(), input2.data(), golden.data(), desc, cfg.dtype, roi.data(),
                       XYWH);

    DeviceTensor src1(cfg.backend, bytes), src2(cfg.backend, bytes), dst(cfg.backend, bytes);
    src1.write(input1.data(), bytes);
    src2.write(input2.data(), bytes);
    dst.write(input1.data(), bytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, shape.n);
    ASSERT_EQ(rppt_phase(src1.ptr(), src2.ptr(), &desc, dst.ptr(), &desc, roi.data(), XYWH,
                         handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back

    dst.read(actual.data(), bytes);

    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), desc, roi.data(), XYWH,
                               phase_tolerance(cfg.dtype)));
}

}  // namespace

// Full name: Image_Geometric/PhaseTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>
class PhaseTest : public ::testing::TestWithParam<TestConfig> {};

TEST_P(PhaseTest, Correctness) {
    const TestConfig cfg = GetParam();
    switch (cfg.dtype) {
        case DType::U8:
            run_phase<Rpp8u>(cfg);
            break;
        case DType::F16:
            run_phase<Rpp16f>(cfg);
            break;
        case DType::F32:
            run_phase<Rpp32f>(cfg);
            break;
        case DType::I8:
            run_phase<Rpp8s>(cfg);
            break;
    }
}

INSTANTIATE_TEST_SUITE_P(
    Image_Geometric, PhaseTest,
    ::testing::ValuesIn(make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                                     {Layout::PKD3, Layout::PLN3, Layout::PLN1},
                                     {Roi::Full, Roi::Partial})),
    config_param_name);
