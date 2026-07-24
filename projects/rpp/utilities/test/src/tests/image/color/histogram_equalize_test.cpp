#include <gtest/gtest.h>
#include <rpp/rpp.h>

#include <vector>

#include "framework/backend_memory.hpp"
#include "framework/compare_tensor.hpp"
#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/histogram_equalize_ref.hpp"

using namespace rpptest;

namespace {

// U8-only op. Grayscale equalization is exact integer; the 3-channel YCbCr round-trip picks up a
// little fp rounding, so 1 LSB is allowed (not enough to hide a convention/clamp bug).
double histogram_equalize_tolerance(DType) { return 1.0; }

template <typename T>
void run_histogram_equalize(const TestConfig& cfg) {
    const TensorShape shape{cfg.size.n, static_cast<Rpp32u>(channels_of(cfg.layout)), cfg.size.h,
                            cfg.size.w};
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layout);  // RPP takes a non-const ptr
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < shape.n; ++i) roi[i] = roiVec[i];

    std::vector<T> input(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    golden = input;
    histogram_equalize_reference<T>(input.data(), golden.data(), desc, roi.data(), XYWH);

    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, shape.n);
    ASSERT_EQ(rppt_histogram_equalize(src.ptr(), &desc, dst.ptr(), &desc, roi.data(), XYWH,
                                      handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back

    dst.read(actual.data(), bytes);

    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), desc, roi.data(), XYWH,
                               histogram_equalize_tolerance(cfg.dtype)));
}

}  // namespace

// Full name: Image_Color/HistogramEqualizeTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>
//
// HOST is green on the full grid. The two HIP 3-channel PartialRoi cases (PKD3, PLN3) are documented
// reds: the HIP 3-channel partial-ROI path is broken (grayscale-partial and 3-channel-full pass).
// The golden matches HOST and stays correct -- see issues/histogram-equalize-hip-3ch-partial-roi.md.
class HistogramEqualizeTest : public ::testing::TestWithParam<TestConfig> {};

TEST_P(HistogramEqualizeTest, Correctness) {
    const TestConfig cfg = GetParam();
    ASSERT_EQ(cfg.dtype, DType::U8) << "histogram_equalize is U8 only";
    run_histogram_equalize<Rpp8u>(cfg);
}

INSTANTIATE_TEST_SUITE_P(
    Image_Color, HistogramEqualizeTest,
    ::testing::ValuesIn(make_configs({DType::U8},
                                     {Layout::PKD3, Layout::PLN3, Layout::PLN1},
                                     {Roi::Full, Roi::Partial})),
    config_param_name);
