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
#include "reference/gaussian_filter_ref.hpp"

using namespace rpptest;

namespace {

// kernelSize is an odd square window size (3/5/7/9 per the API doc); this suite exercises 3 and 5.
// stdDev is the per-image Gaussian standard deviation (same value fed to golden and kernel).
struct GaussianFilterParams {
    Rpp32u kernelSize;
    float stdDev;
    std::string name() const { return "k" + std::to_string(kernelSize) + "_sd" + num_token(stdDev); }
};

// Tolerances reflect legitimate numeric error only (weighted-sum rounding); U8/I8 allow one LSB for
// round-to-nearest quantization, floats allow accumulation slack.
double gaussian_filter_tolerance(DType dt) {
    switch (dt) {
        case DType::U8:
            return 1.0;
        case DType::I8:
            return 1.0;
        case DType::F32:
            return 1e-3;
        case DType::F16:
            return 1e-2;
        default:
            return 0.0;
    }
}

template <typename T>
void run_gaussian_filter(const TestConfig& cfg, const GaussianFilterParams& op) {
    const TensorShape shape{cfg.size.n, static_cast<Rpp32u>(channels_of(cfg.layout)), cfg.size.h,
                            cfg.size.w};
    // src carries a leading border pad: the HIP filter kernel requires
    // srcDesc.offsetInBytes >= 12 * (kernelSize/2) (read-slack for the KxK window; the border
    // itself is computed by clamping indices to the image, i.e. replicate). dst keeps offset 0,
    // so the golden and comparator index destination elements from 0. Applied on both backends
    // (the HOST path honours offsetInBytes identically) to keep one path.
    RpptDesc srcDesc = make_descriptor(shape, cfg.dtype, cfg.layout);
    RpptDesc dstDesc = srcDesc;  // same dims/strides; only src gets the pad offset
    const std::size_t offsetBytes = 12u * (op.kernelSize / 2);
    const std::size_t offsetElems = offsetBytes / dtype_size(cfg.dtype);
    srcDesc.offsetInBytes = static_cast<Rpp32u>(offsetBytes);

    const std::size_t count = element_count(srcDesc);
    const std::size_t imageBytes = count * dtype_size(cfg.dtype);

    // stdDevTensor is a per-image Rpp32f* in host-accessible (pinned for HIP) memory.
    PinnedArray<Rpp32f> stdDev(cfg.backend, shape.n);
    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(srcDesc, cfg.roi);
    for (Rpp32u i = 0; i < shape.n; ++i) {
        stdDev[i] = op.stdDev;
        roi[i] = roiVec[i];
    }

    // (1) Host golden model. The src buffer is [front pad][image][back pad]; the image base sits
    // at offsetElems. golden/actual are the (pad-free, offset 0) destination. golden starts as a
    // copy of the image so the untouched (outside-ROI) region is defined.
    std::vector<T> src(offsetElems + count + offsetElems), golden(count), actual(count);
    fill_input<T>(src.data(), src.size(), cfg.dtype);
    T* image = src.data() + offsetElems;
    golden.assign(image, image + count);
    gaussian_filter_reference<T>(image, golden.data(), dstDesc, cfg.dtype, roi.data(), XYWH,
                                 op.kernelSize, op.stdDev);

    // (2) Run RPP on the configured backend. gaussian_filter is a single symbol taking a backend
    // arg (no separate _host variant). RPP adds srcDesc.offsetInBytes to the src pointer
    // internally, landing on the image base.
    DeviceTensor srcDev(cfg.backend, src.size() * dtype_size(cfg.dtype));
    DeviceTensor dst(cfg.backend, imageBytes);
    srcDev.write(src.data(), src.size() * dtype_size(cfg.dtype));
    dst.write(image, imageBytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, shape.n);
    ASSERT_EQ(rppt_gaussian_filter(srcDev.ptr(), &srcDesc, dst.ptr(), &dstDesc, stdDev.data(),
                                   op.kernelSize, RpptImageBorderType::REPLICATE, roi.data(), XYWH,
                                   handle.get(), cfg.backend),
              RPP_SUCCESS);

    // (3) Retrieve the result on the host (no-op copy for HOST, device->host for HIP).
    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), imageBytes);

    // (4) Compare within tolerance over the ROI. The golden matches HIP across the whole grid, so it
    // is correct; known HOST-only reds kept red by design: PKD3 k3 diverges at the row edge, and
    // PLN PartialRoi k5 bleeds out-of-ROI neighbors. Both are real kernel defects.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), dstDesc, roi.data(), XYWH,
                               gaussian_filter_tolerance(cfg.dtype)));
}

}  // namespace

// Full name: Image_Filter/GaussianFilterTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_k<KernelSize>_sd<StdDev>
class GaussianFilterTest : public ::testing::TestWithParam<WithParams<GaussianFilterParams>> {};

TEST_P(GaussianFilterTest, Correctness) {
    const auto& p = GetParam();
    switch (p.cfg.dtype) {
        case DType::U8:
            run_gaussian_filter<Rpp8u>(p.cfg, p.op);
            break;
        case DType::F16:
            run_gaussian_filter<Rpp16f>(p.cfg, p.op);
            break;
        case DType::F32:
            run_gaussian_filter<Rpp32f>(p.cfg, p.op);
            break;
        case DType::I8:
            run_gaussian_filter<Rpp8s>(p.cfg, p.op);
            break;
        default:
            FAIL() << "unsupported dtype for gaussian_filter";
    }
}

INSTANTIATE_TEST_SUITE_P(
    Image_Filter, GaussianFilterTest,
    ::testing::ValuesIn(with_params<GaussianFilterParams>(
        make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                     {Layout::PKD3, Layout::PLN3, Layout::PLN1}, {Roi::Full, Roi::Partial}),
        {GaussianFilterParams{3, 1.0f}, GaussianFilterParams{5, 1.0f}})),
    op_config_name<GaussianFilterParams>);
