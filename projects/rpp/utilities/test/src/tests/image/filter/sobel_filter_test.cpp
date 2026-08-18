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

// sobel_filter GTest: PLN1 only, kernelSize = 3 only. sobel_filter's dstDesc is always
// single-channel grayscale (c=1, NCHW); a 3-channel input would require an undocumented
// RGB->grayscale conversion (not independently derivable), and the extended k=5/7 kernels are
// convention-dependent. Both are deferred -- this covers grayscale-in/grayscale-out with the
// canonical 3x3 Sobel operator over {U8,F16,F32,I8} x PLN1 x {Full,Partial} x sobelType {0,1,2}.
#include <gtest/gtest.h>
#include <rpp/rpp.h>

#include <vector>

#include "framework/backend_memory.hpp"
#include "framework/compare_tensor.hpp"
#include "framework/config_param.hpp"
#include "framework/dtype_dispatch.hpp"
#include "framework/skip_list.hpp"
#include "framework/tensor_setup.hpp"
#include "framework/tolerance.hpp"
#include "reference/sobel_filter_ref.hpp"

using namespace rpptest;

namespace {

struct SobelFilterParams {
    Rpp32u sobelType;
    Rpp32u kernelSize;
    std::string name() const {
        const char* g = sobelType == 0 ? "gradX" : sobelType == 1 ? "gradY" : "gradXY";
        return std::string(g) + "_k" + std::to_string(kernelSize);
    }
};

// Legitimate numeric error only (sqrt rounding for the XY case); not loosened to hide
// clamp/offset/truncation defects.
constexpr Tolerance kSobelFilterTolerance = tolerance(1.0, 1e-3, 1e-2);

template <typename T>
void run_sobel_filter(const TestConfig& cfg, const SobelFilterParams& op) {
    const TensorShape shape{cfg.size.n, static_cast<Rpp32u>(channels_of(cfg.layout)), cfg.size.h,
                            cfg.size.w};
    // src carries a leading border pad: the HIP filter kernel requires srcDesc.offsetInBytes >=
    // 12 * (kernelSize/2) (read-slack for the KxK window; the border itself is computed by
    // clamping indices to the image, i.e. replicate). dst keeps offset 0, so the golden and
    // comparator index destination elements from 0. Applied on both backends to keep one path.
    RpptDesc srcDesc = make_descriptor(shape, cfg.dtype, cfg.layout);
    RpptDesc dstDesc = srcDesc;  // PLN1: same c=1 dims/strides; only src gets the pad offset
    const std::size_t offsetBytes = 12u * (op.kernelSize / 2);
    const std::size_t offsetElems = offsetBytes / dtype_size(cfg.dtype);
    srcDesc.offsetInBytes = static_cast<Rpp32u>(offsetBytes);

    const std::size_t count = element_count(srcDesc);
    const std::size_t imageBytes = count * dtype_size(cfg.dtype);

    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(srcDesc, cfg.roi);
    for (Rpp32u i = 0; i < shape.n; ++i) roi[i] = roiVec[i];

    // (1) Host golden model. The src buffer is [front pad][image][back pad]; the image base sits
    // at offsetElems. golden/actual are the (pad-free, offset 0) destination. golden starts as a
    // copy of the image so the untouched (outside-ROI) region is defined.
    std::vector<T> src(offsetElems + count + offsetElems), golden(count), actual(count);
    fill_input<T>(src.data(), src.size(), cfg.dtype);
    T* image = src.data() + offsetElems;
    golden.assign(image, image + count);
    sobel_filter_reference<T>(image, golden.data(), dstDesc, cfg.dtype, roi.data(), XYWH,
                              op.sobelType, op.kernelSize);

    // (2) Run RPP on the configured backend. sobel_filter is a single backend-arg symbol (no
    // separate HOST variant), called unconditionally. RPP adds srcDesc.offsetInBytes to the src
    // pointer internally, landing on the image base.
    DeviceTensor srcDev(cfg.backend, src.size() * dtype_size(cfg.dtype));
    DeviceTensor dst(cfg.backend, imageBytes);
    srcDev.write(src.data(), src.size() * dtype_size(cfg.dtype));
    dst.write(image, imageBytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, shape.n);
    RppStatus status =
        rppt_sobel_filter(srcDev.ptr(), &srcDesc, dst.ptr(), &dstDesc, op.sobelType, op.kernelSize,
                          roi.data(), XYWH, handle.get(), cfg.backend);
    ASSERT_EQ(status, RPP_SUCCESS);

    // (3) Retrieve the result on the host (no-op copy for HOST, device->host for HIP).
    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), imageBytes);

    // (4) Compare over the ROI. gradX/gradY match the kernel for U8/F16/F32 (golden Gx/Gy correct).
    // Known reds kept red by design: sobelType 2 (XY) diverges from the Euclidean gradient
    // magnitude on both backends; I8 gradX/gradY (both backends) and HIP F16 gradX/gradY are not
    // clamped/quantized to the dtype range. All are real kernel defects.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), dstDesc, roi.data(), XYWH,
                               kSobelFilterTolerance(cfg.dtype)));
}

}  // namespace

// Full name: Image_Filter/SobelFilterTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<grad>_k<KernelSize>
class SobelFilterTest : public SkipListTest<WithParams<SobelFilterParams>> {};

TEST_P(SobelFilterTest, Correctness) {
    const auto& p = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(p.cfg.dtype, [&](auto tag) {
        run_sobel_filter<Element<decltype(tag)>>(p.cfg, p.op);
    });
}

INSTANTIATE_TEST_SUITE_P(
    Image_Filter, SobelFilterTest,
    ::testing::ValuesIn(with_params<SobelFilterParams>(
        make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                     {Layout::PLN1}, {Roi::Full, Roi::Partial}),
        {SobelFilterParams{0, 3}, SobelFilterParams{1, 3}, SobelFilterParams{2, 3}})),
    op_config_name<SobelFilterParams>);
