#include <gtest/gtest.h>
#include <rpp/rpp.h>

#include <vector>

#include "framework/backend_memory.hpp"
#include "framework/config_param.hpp"
#include "framework/reduction.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/tensor_mean_ref.hpp"
#include "reference/tensor_stddev_ref.hpp"

using namespace rpptest;

namespace {

// stddev is sqrt(mean of squared deviations) in float; the tolerance covers accumulation, the
// division, and the sqrt. Integer-space stddevs (U8/I8) carry a larger magnitude than the [0,1]
// float stddevs, hence the wider absolute tolerance.
double tensor_stddev_tolerance(DType dt) {
    switch (dt) {
        case DType::U8:
        case DType::I8:
            return 1e-1;
        case DType::F32:
            return 1e-3;
        case DType::F16:
            return 1e-2;
    }
    return 0.0;
}

// stddev always outputs Rpp32f, and takes a meanTensor of size n*4 in [MeanR,MeanG,MeanB,MeanImage]
// order per image (per the API). Tin is the source element type.
template <typename Tin>
void run_tensor_stddev(const TestConfig& cfg) {
    const TensorShape shape{cfg.size.n, static_cast<Rpp32u>(channels_of(cfg.layout)), cfg.size.h,
                            cfg.size.w};
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layout);  // RPP takes a non-const ptr
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);
    const std::size_t stride = reduction_stride(desc);
    const std::size_t outLen = reduction_length(desc);

    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < shape.n; ++i) roi[i] = roiVec[i];

    std::vector<Tin> input(count);
    fill_input<Tin>(input.data(), count, cfg.dtype);

    // The mean the op deviates against. Compute it with the mean reference, then feed the exact
    // float-rounded values to BOTH the kernel (meanTensor, always n*4) and the stddev reference,
    // so neither has an unfair mean. meanTensor layout is [MeanR,MeanG,MeanB,MeanImage] per image;
    // for a 1-channel image all four slots carry that image's single mean.
    const std::vector<double> meanDbl =
        tensor_mean_reference<Tin>(input.data(), desc, roi.data(), XYWH);
    PinnedArray<Rpp32f> mean(cfg.backend, static_cast<std::size_t>(shape.n) * 4);
    std::vector<double> meanRef(outLen);
    for (Rpp32u n = 0; n < shape.n; ++n) {
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

    // (1) Host golden model, deviating against the same (float-rounded) means passed to the op.
    const std::vector<double> golden =
        tensor_stddev_reference<Tin>(input.data(), desc, roi.data(), XYWH, meanRef);

    // (2) Run RPP on the configured backend. The result array is host-accessible (pinned for HIP).
    DeviceTensor src(cfg.backend, bytes);
    src.write(input.data(), bytes);
    PinnedArray<Rpp32f> out(cfg.backend, outLen);
    for (std::size_t i = 0; i < outLen; ++i) out[i] = 0.0f;

    RppHandle handle(cfg.backend, shape.n);
    ASSERT_EQ(rppt_tensor_stddev(src.ptr(), &desc, out.data(), static_cast<Rpp32u>(outLen),
                                 mean.data(), roi.data(), XYWH, handle.get(), cfg.backend),
              RPP_SUCCESS);

    // (3) Drain the op's stream; pinned output is then directly readable on the host.
    handle.sync();

    // (4) Compare within tolerance.
    EXPECT_TRUE(compare_reduction<Rpp32f>(out.data(), golden, tensor_stddev_tolerance(cfg.dtype)));
}

}  // namespace

// Full name:
// Image_Statistical/TensorStddevTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>
class TensorStddevTest : public ::testing::TestWithParam<TestConfig> {};

TEST_P(TensorStddevTest, Correctness) {
    const TestConfig cfg = GetParam();
    switch (cfg.dtype) {
        case DType::U8:
            run_tensor_stddev<Rpp8u>(cfg);
            break;
        case DType::F16:
            run_tensor_stddev<Rpp16f>(cfg);
            break;
        case DType::F32:
            run_tensor_stddev<Rpp32f>(cfg);
            break;
        case DType::I8:
            run_tensor_stddev<Rpp8s>(cfg);
            break;
    }
}

INSTANTIATE_TEST_SUITE_P(
    Image_Statistical, TensorStddevTest,
    ::testing::ValuesIn(make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                                     {Layout::PKD3, Layout::PLN3, Layout::PLN1},
                                     {Roi::Full, Roi::Partial})),
    config_param_name);
