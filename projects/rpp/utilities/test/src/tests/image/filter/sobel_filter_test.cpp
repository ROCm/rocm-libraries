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

// PLN1 and kernelSize 3 only: 3-channel input needs an undocumented RGB->grayscale conversion, and
// the k=5/7 kernels are convention-dependent.
#include <gtest/gtest.h>
#include <rpp/rpp.h>

#include <vector>

#include "framework/backend_memory.hpp"
#include "framework/compare.hpp"
#include "framework/config_param.hpp"
#include "framework/dtype.hpp"
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

// Covers sqrt rounding for the XY case; not loosened to hide clamp/truncation defects.
constexpr Tolerance kSobelFilterTolerance = tolerance(1.0, 1e-3, 1e-2);

template <typename T>
void run_sobel_filter(const TestConfig& cfg, const SobelFilterParams& op) {
    const TensorShape shape{cfg.size.n, static_cast<Rpp32u>(channels_of(cfg.layoutIn)), cfg.size.h,
                            cfg.size.w};
    // The HIP filter kernel requires srcDesc.offsetInBytes >= 12 * (kernelSize/2) as read-slack for
    // the window, so src carries a leading pad. dst keeps offset 0.
    RpptDesc srcDesc = make_descriptor(shape, cfg.dtype, cfg.layoutIn);
    RpptDesc dstDesc = srcDesc;  // only src gets the pad offset
    const std::size_t offsetBytes = 12u * (op.kernelSize / 2);
    const std::size_t offsetElems = offsetBytes / dtype_size(cfg.dtype);
    srcDesc.offsetInBytes = static_cast<Rpp32u>(offsetBytes);

    const std::size_t count = element_count(srcDesc);
    const std::size_t imageBytes = count * dtype_size(cfg.dtype);

    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(srcDesc, cfg.roi);
    for (Rpp32u i = 0; i < shape.n; ++i) roi[i] = roiVec[i];

    // src is [front pad][image][back pad]; the image base sits at offsetElems.
    std::vector<T> src(offsetElems + count + offsetElems), golden(count), actual(count);
    fill_input<T>(src.data(), src.size(), cfg.dtype);
    T* image = src.data() + offsetElems;
    golden.assign(image, image + count);
    sobel_filter_reference<T>(image, golden.data(), dstDesc, cfg.dtype, roi.data(), XYWH,
                              op.sobelType, op.kernelSize);

    DeviceTensor srcDev(cfg.backend, src.size() * dtype_size(cfg.dtype));
    DeviceTensor dst(cfg.backend, imageBytes);
    srcDev.write(src.data(), src.size() * dtype_size(cfg.dtype));
    dst.write(image, imageBytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, shape.n);
    RppStatus status =
        rppt_sobel_filter(srcDev.ptr(), &srcDesc, dst.ptr(), &dstDesc, op.sobelType, op.kernelSize,
                          roi.data(), XYWH, handle.get(), cfg.backend);
    ASSERT_EQ(status, RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), imageBytes);

    // Known kernel reds: sobelType 2 diverges from the Euclidean magnitude; I8 and HIP F16
    // gradX/gradY are not clamped to the dtype range.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), dstDesc, roi.data(), XYWH,
                               kSobelFilterTolerance(cfg.dtype)));
}

}  // namespace

// Full name:
// Image_Filter/SobelFilterTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<grad>_k<KernelSize>
class SobelFilterTest : public SkipListTest<WithParams<SobelFilterParams>> {};

TEST_P(SobelFilterTest, Correctness) {
    const auto& p = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(
        p.cfg.dtype, [&](auto tag) { run_sobel_filter<Element<decltype(tag)>>(p.cfg, p.op); });
}

INSTANTIATE_TEST_SUITE_P(
    Image_Filter, SobelFilterTest,
    ::testing::ValuesIn(with_params<SobelFilterParams>(
        make_configs(presets::kDefaultDTypes, {Layout::PLN1}, {Roi::Full, Roi::Partial},
                     {presets::kDefaultSize, presets::kTailWidthSize, presets::kSubVectorSize}),
        {SobelFilterParams{0, 3}, SobelFilterParams{1, 3}, SobelFilterParams{2, 3}})),
    op_config_name<SobelFilterParams>);
