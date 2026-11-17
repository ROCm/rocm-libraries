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

#include <algorithm>
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
    // The output is always single-channel PLN1; a 3-channel source differs only in srcDesc.
    // The HIP filter kernel requires srcDesc.offsetInBytes >= 12 * (kernelSize/2) as read-slack for
    // the window, so src carries a leading pad. dst keeps offset 0.
    RpptDesc srcDesc = make_src_descriptor(cfg);
    RpptDesc dstDesc = make_dst_descriptor(cfg);
    const std::size_t offsetBytes = 12u * (op.kernelSize / 2);
    const std::size_t offsetElems = offsetBytes / dtype_size(cfg.dtype);
    srcDesc.offsetInBytes = static_cast<Rpp32u>(offsetBytes);

    const std::size_t srcCount = element_count(srcDesc);
    const std::size_t dstCount = element_count(dstDesc);
    const std::size_t dstBytes = dstCount * dtype_size(cfg.dtype);

    PinnedArray<RpptROI> roi(cfg.backend, cfg.size.n);
    const std::vector<RpptROI> roiVec = make_roi(srcDesc, cfg.roi);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) roi[i] = roiVec[i];

    // src is [front pad][image][back pad]; the image base sits at offsetElems.
    std::vector<T> src(offsetElems + srcCount + offsetElems), golden(dstCount), actual(dstCount);
    fill_input<T>(src.data(), src.size(), cfg.dtype);
    T* image = src.data() + offsetElems;
    if (srcDesc.c == 3) {
        // Neutral pixels (R = G = B): any luma weighting that sums to one returns the grey value
        // itself, so the grey plane the gradients see does not hinge on the conversion's rounding.
        std::vector<T> grey(dstCount);
        fill_input_image<T>(grey.data(), dstDesc, cfg.dtype);
        for_each_image_element(srcDesc, [&](Rpp32u n, Rpp32u, Rpp32u y, Rpp32u x, std::size_t idx) {
            image[idx] = grey[plane_index(dstDesc, plane_base(dstDesc, n, 0), y, x)];
        });
    } else {
        std::copy(image, image + dstCount, golden.begin());
    }
    // dst outside the ROI is not compared; dst and golden start from the same content.
    const std::vector<T> dstInit = golden;
    sobel_filter_reference<T>(image, srcDesc, golden.data(), dstDesc, cfg.dtype, roi.data(), XYWH,
                              op.sobelType, op.kernelSize);

    DeviceTensor srcDev(cfg.backend, src.size() * dtype_size(cfg.dtype));
    DeviceTensor dst(cfg.backend, dstBytes);
    srcDev.write(src.data(), src.size() * dtype_size(cfg.dtype));
    dst.write(dstInit.data(), dstBytes);

    RppHandle handle(cfg.backend, cfg.size.n);
    RppStatus status =
        rppt_sobel_filter(srcDev.ptr(), &srcDesc, dst.ptr(), &dstDesc, op.sobelType, op.kernelSize,
                          roi.data(), XYWH, handle.get(), cfg.backend);
    ASSERT_EQ(status, RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), dstBytes);

    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), dstDesc, roi.data(), XYWH,
                               kSobelFilterTolerance(cfg.dtype)));
}

std::vector<TestConfig> hip_configs(std::vector<TestConfig> configs) {
    configs.erase(std::remove_if(configs.begin(), configs.end(),
                                 [](const TestConfig& c) { return c.backend != RPP_HIP_BACKEND; }),
                  configs.end());
    return configs;
}

std::vector<WithParams<SobelFilterParams>> concat_params(
    std::initializer_list<std::vector<WithParams<SobelFilterParams>>> sets) {
    std::vector<WithParams<SobelFilterParams>> out;
    for (const auto& s : sets) out.insert(out.end(), s.begin(), s.end());
    return out;
}

const std::vector<LayoutConv> kRgbToPln1{{Layout::PKD3, Layout::PLN1},
                                         {Layout::PLN3, Layout::PLN1}};
const std::vector<Size> kPln1Sizes{presets::kDefaultSize, presets::kTailWidthSize,
                                   presets::kSubVectorSize};

// Every grid point has pixels with negative gx or gy.
// - gradXY: every dtype on both backends, kernel sizes 3/5/7.
// - gradX/gradY beyond the k3 sweep: only where the store can leave the dtype range (HIP F16 and
//   I8); elsewhere one direction is a single-component clamp the k3 sweep already covers. Larger I8
//   kernels mostly saturate, where the stored value is the same with or without the -128 shift, so
//   I8 is gridded at k5 on the 36-row shapes only.
// - 3-channel input leaves out I8: rppt_color_to_greyscale truncates its I8 result (HIP store, HOST
//   scalar tail), so a neutral pixel can come back one below its value, and the gradient amplifies
//   that past this test's bound.
std::vector<WithParams<SobelFilterParams>> sobel_configs() {
    const std::vector<DType> kRgbDTypes{DType::U8, DType::F16, DType::F32};
    return concat_params({
        with_params<SobelFilterParams>(
            make_configs(presets::kDefaultDTypes, {Layout::PLN1}, {Roi::Full, Roi::Partial},
                         kPln1Sizes),
            {SobelFilterParams{0, 3}, SobelFilterParams{1, 3}, SobelFilterParams{2, 3},
             SobelFilterParams{2, 5}, SobelFilterParams{2, 7}}),
        with_params<SobelFilterParams>(
            make_configs(kRgbDTypes, kRgbToPln1, {Roi::Full, Roi::Partial},
                         {presets::kTailWidthSize}),
            {SobelFilterParams{2, 3}, SobelFilterParams{2, 5}, SobelFilterParams{2, 7}}),
        with_params<SobelFilterParams>(
            hip_configs(
                make_configs({DType::F16}, {Layout::PLN1}, {Roi::Full, Roi::Partial}, kPln1Sizes)),
            {SobelFilterParams{0, 5}, SobelFilterParams{1, 5}, SobelFilterParams{0, 7},
             SobelFilterParams{1, 7}}),
        with_params<SobelFilterParams>(
            hip_configs(make_configs({DType::I8}, {Layout::PLN1}, {Roi::Full, Roi::Partial},
                                     {presets::kDefaultSize, presets::kTailWidthSize})),
            {SobelFilterParams{0, 5}, SobelFilterParams{1, 5}}),
        with_params<SobelFilterParams>(
            hip_configs(make_configs({DType::F16}, kRgbToPln1, {Roi::Full, Roi::Partial},
                                     {presets::kTailWidthSize})),
            {SobelFilterParams{0, 3}, SobelFilterParams{1, 3}, SobelFilterParams{0, 5},
             SobelFilterParams{1, 5}, SobelFilterParams{0, 7}, SobelFilterParams{1, 7}}),
    });
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

INSTANTIATE_TEST_SUITE_P(Image_Filter, SobelFilterTest, ::testing::ValuesIn(sobel_configs()),
                         op_config_name<SobelFilterParams>);
