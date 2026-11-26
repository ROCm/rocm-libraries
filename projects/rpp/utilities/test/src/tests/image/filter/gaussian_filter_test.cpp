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

#include <utility>
#include <vector>

#include "framework/backend_memory.hpp"
#include "framework/compare.hpp"
#include "framework/config_param.hpp"
#include "framework/dtype.hpp"
#include "framework/skip_list.hpp"
#include "framework/tensor_setup.hpp"
#include "framework/tolerance.hpp"
#include "reference/gaussian_filter_ref.hpp"

using namespace rpptest;

namespace {

// Each kernelSize (3/5/7/9) has its own SIMD path. stdDev 0.25 collapses the weights onto the
// centre tap; 5.0 flattens them towards a box.
struct GaussianFilterParams {
    Rpp32u kernelSize;
    float stdDev;
    std::string name() const {
        return "k" + std::to_string(kernelSize) + "_sd" + num_token(stdDev);
    }
};

// U8/I8 allow one LSB for round-to-nearest; floats allow accumulation slack.
constexpr Tolerance kGaussianFilterTolerance = tolerance(1.0, 1e-3, 1e-2);

template <typename T>
void run_gaussian_filter(const TestConfig& cfg, const GaussianFilterParams& op) {
    // The HIP filter kernel requires srcDesc.offsetInBytes >= 12 * (kernelSize/2) as read-slack for
    // the window, so src carries a leading pad. dst keeps offset 0.
    RpptDesc srcDesc = make_src_descriptor(cfg);
    RpptDesc dstDesc = make_dst_descriptor(cfg);
    const std::size_t offsetBytes = 12u * (op.kernelSize / 2);
    const std::size_t offsetElems = offsetBytes / dtype_size(cfg.dtype);
    srcDesc.offsetInBytes = static_cast<Rpp32u>(offsetBytes);

    const std::size_t count = element_count(srcDesc);
    const std::size_t imageBytes = count * dtype_size(cfg.dtype);

    PinnedArray<Rpp32f> stdDev(cfg.backend, cfg.size.n);
    PinnedArray<RpptROI> roi(cfg.backend, cfg.size.n);
    const std::vector<RpptROI> roiVec = make_roi(srcDesc, cfg.roi);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) {
        stdDev[i] = op.stdDev;
        roi[i] = roiVec[i];
    }

    // src is [front pad][image][back pad]; the image base sits at offsetElems.
    std::vector<T> src(offsetElems + count + offsetElems), golden(count), actual(count);
    fill_input<T>(src.data(), src.size(), cfg.dtype);
    T* image = src.data() + offsetElems;
    golden.assign(image, image + count);
    gaussian_filter_reference<T>(image, srcDesc, golden.data(), dstDesc, cfg.dtype, roi.data(),
                                 XYWH, op.kernelSize, op.stdDev);

    DeviceTensor srcDev(cfg.backend, src.size() * dtype_size(cfg.dtype));
    DeviceTensor dst(cfg.backend, imageBytes);
    srcDev.write(src.data(), src.size() * dtype_size(cfg.dtype));
    dst.write(image, imageBytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, cfg.size.n);
    ASSERT_EQ(rppt_gaussian_filter(srcDev.ptr(), &srcDesc, dst.ptr(), &dstDesc, stdDev.data(),
                                   op.kernelSize, RpptImageBorderType::REPLICATE, roi.data(), XYWH,
                                   handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), imageBytes);

    // Known HOST red (kernel defect): PLN PartialRoi k5 reads neighbours from outside the ROI.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), dstDesc, roi.data(), XYWH,
                               kGaussianFilterTolerance(cfg.dtype)));
}

// Same-layout cases plus both directions of the fused output-layout conversion, and HOST
// PKD3-to-PLN3 widths at which the k3, k7 and k9 vector loops used to run past the row end.
// kTailWidthSize does not reach those loop bounds.
std::vector<WithParams<GaussianFilterParams>> gaussian_filter_configs() {
    std::vector<WithParams<GaussianFilterParams>> configs = with_params<GaussianFilterParams>(
        concat_configs({
            make_configs(presets::kDefaultDTypes, presets::kLayoutsFullConv,
                         {Roi::Full, Roi::Partial}, {presets::kTailWidthSize}),
            make_configs(presets::kDefaultDTypes, presets::kLayoutsFull, {Roi::Full, Roi::Partial},
                         {presets::kDefaultSize, presets::kSubVectorSize}),
        }),
        {GaussianFilterParams{3, 1.0f}, GaussianFilterParams{5, 1.0f},
         GaussianFilterParams{7, 1.0f}, GaussianFilterParams{9, 1.0f},
         GaussianFilterParams{5, 0.25f}, GaussianFilterParams{5, 5.0f}});
    for (const auto& [width, kernelSize] : {std::pair{13u, 3u}, {17u, 7u}, {22u, 9u}})
        for (const TestConfig& cfg :
             make_configs(presets::kDefaultDTypes, {LayoutConv{Layout::PKD3, Layout::PLN3}},
                          {Roi::Full}, {Size{1, 11, width}}))
            if (cfg.backend == RPP_HOST_BACKEND)
                configs.push_back({cfg, GaussianFilterParams{kernelSize, 1.0f}});
    return configs;
}

}  // namespace

// Full name:
// Image_Filter/GaussianFilterTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_k<KernelSize>_sd<StdDev>
class GaussianFilterTest : public SkipListTest<WithParams<GaussianFilterParams>> {};

TEST_P(GaussianFilterTest, Correctness) {
    const auto& p = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(
        p.cfg.dtype, [&](auto tag) { run_gaussian_filter<Element<decltype(tag)>>(p.cfg, p.op); });
}

INSTANTIATE_TEST_SUITE_P(Image_Filter, GaussianFilterTest,
                         ::testing::ValuesIn(gaussian_filter_configs()),
                         op_config_name<GaussianFilterParams>);
