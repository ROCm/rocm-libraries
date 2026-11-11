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

#include <string>
#include <vector>

#include "framework/backend_memory.hpp"
#include "framework/compare.hpp"
#include "framework/config_param.hpp"
#include "framework/dtype.hpp"
#include "framework/skip_list.hpp"
#include "framework/tensor_setup.hpp"
#include "framework/tolerance.hpp"
#include "reference/prewitt_filter_ref.hpp"

using namespace rpptest;

namespace {

// prewittType selects the written gradient: 0 -> gx, 1 -> gy, 2 -> magnitude. The window is
// always 3x3, so it is the only op parameter.
struct PrewittFilterParams {
    Rpp32u prewittType;
    std::string name() const {
        return prewittType == 0 ? "gradX" : prewittType == 1 ? "gradY" : "gradXY";
    }
};

// U8/I8 allow one LSB for round-to-nearest (and the sqrt in gradXY); floats allow accumulation
// slack. Not loosened to hide clamp/truncation defects.
constexpr Tolerance kPrewittFilterTolerance = tolerance(1.0, 1e-3, 1e-2);

template <typename T>
void run_prewitt_filter(const TestConfig& cfg, const PrewittFilterParams& op) {
    // src carries the same leading border pad as the other 3x3 filters: the HIP filter kernels
    // require srcDesc.offsetInBytes >= 12 * (kernelSize/2) of read-slack for the window. dst keeps
    // offset 0, so the golden and comparator index destination elements from 0.
    RpptDesc srcDesc = make_src_descriptor(cfg);
    RpptDesc dstDesc = make_dst_descriptor(cfg);
    const std::size_t offsetBytes = 12u * (3 / 2);
    const std::size_t offsetElems = offsetBytes / dtype_size(cfg.dtype);
    srcDesc.offsetInBytes = static_cast<Rpp32u>(offsetBytes);

    const std::size_t count = element_count(srcDesc);
    const std::size_t imageBytes = count * dtype_size(cfg.dtype);

    PinnedArray<RpptROI> roi(cfg.backend, cfg.size.n);
    const std::vector<RpptROI> roiVec = make_roi(srcDesc, cfg.roi);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) roi[i] = roiVec[i];

    // (1) Host golden model. The src buffer is [front pad][image][back pad]; the image base sits
    // at offsetElems. The destination starts from a DIFFERENT pattern than the source (salt = 1),
    // so a kernel that wrote nothing cannot pass by leaving the input in place.
    std::vector<T> src(offsetElems + count + offsetElems), dstInit(count), golden(count),
        actual(count);
    fill_input<T>(src.data(), src.size(), cfg.dtype);
    fill_input<T>(dstInit.data(), count, cfg.dtype, /*salt=*/1);
    T* image = src.data() + offsetElems;
    golden = dstInit;
    prewitt_filter_reference<T>(image, srcDesc, golden.data(), dstDesc, cfg.dtype, roi.data(), XYWH,
                                op.prewittType);

    // (2) Run RPP on the configured backend. RPP adds srcDesc.offsetInBytes to the src pointer
    // internally, landing on the image base.
    DeviceTensor srcDev(cfg.backend, src.size() * dtype_size(cfg.dtype));
    DeviceTensor dst(cfg.backend, imageBytes);
    srcDev.write(src.data(), src.size() * dtype_size(cfg.dtype));
    dst.write(dstInit.data(), imageBytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, cfg.size.n);
    ASSERT_EQ(rppt_prewitt_filter(srcDev.ptr(), &srcDesc, dst.ptr(), &dstDesc, op.prewittType,
                                  RpptImageBorderType::REPLICATE, roi.data(), XYWH, handle.get(),
                                  cfg.backend),
              RPP_SUCCESS);

    // (3) Retrieve the result on the host (no-op copy for HOST, device->host for HIP).
    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), imageBytes);

    // (4) Compare within tolerance over the ROI, against the test's own ROI copy in case the
    // backend rewrites the tensor handed to it.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), dstDesc, roiVec.data(), XYWH,
                               kPrewittFilterTolerance(cfg.dtype)));
}

// One out-of-contract argument per case. Argument validation is dtype- and layout-independent, so
// the arguments are the grid axis here.
struct PrewittFilterNegativeParams {
    Rpp32u prewittType;
    RpptImageBorderType borderType;
    bool singleChannelDst;
    std::string name() const {
        return "type" + std::to_string(prewittType) +
               (borderType == RpptImageBorderType::REPLICATE ? "" : "_constant") +
               (singleChannelDst ? "_dstC1" : "");
    }
};

// An unknown prewittType, a border mode other than REPLICATE and a destination whose channel count
// differs from the source's are not legal calls and must be reported rather than silently
// producing an undefined image. Only the status is asserted: which RPP_ERROR* each maps to is the
// header's to name.
void run_prewitt_filter_negative(const TestConfig& cfg, const PrewittFilterNegativeParams& op) {
    RpptDesc srcDesc = make_src_descriptor(cfg);
    RpptDesc dstDesc = srcDesc;
    if (op.singleChannelDst)
        dstDesc = make_descriptor(TensorShape{cfg.size.n, 1, cfg.size.h, cfg.size.w}, cfg.dtype,
                                  Layout::PLN1);
    const std::size_t count = element_count(srcDesc);
    const std::size_t bytes = byte_size(srcDesc, cfg.dtype);

    PinnedArray<RpptROI> roi(cfg.backend, cfg.size.n);
    const std::vector<RpptROI> roiVec = make_roi(srcDesc, cfg.roi);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) roi[i] = roiVec[i];

    std::vector<Rpp8u> input(count);
    fill_input<Rpp8u>(input.data(), count, cfg.dtype);

    // dst is sized for the source so it is large enough for either destination descriptor.
    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);

    RppHandle handle(cfg.backend, cfg.size.n);
    const RppStatus status =
        rppt_prewitt_filter(src.ptr(), &srcDesc, dst.ptr(), &dstDesc, op.prewittType, op.borderType,
                            roi.data(), XYWH, handle.get(), cfg.backend);
    handle.sync();
    EXPECT_NE(status, RPP_SUCCESS)
        << "prewitt_filter accepted out-of-contract arguments (" << op.name() << ")";
}

}  // namespace

// Full name:
// Image_Filter/PrewittFilterTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<grad>
class PrewittFilterTest : public SkipListTest<WithParams<PrewittFilterParams>> {};

TEST_P(PrewittFilterTest, Correctness) {
    const auto& p = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(
        p.cfg.dtype, [&](auto tag) { run_prewitt_filter<Element<decltype(tag)>>(p.cfg, p.op); });
}

// Same-layout cases plus both directions of the fused output-layout conversion.
INSTANTIATE_TEST_SUITE_P(
    Image_Filter, PrewittFilterTest,
    ::testing::ValuesIn(with_params<PrewittFilterParams>(
        concat_configs({
            make_configs(presets::kDefaultDTypes, presets::kLayoutsFullConv,
                         {Roi::Full, Roi::Partial}, {presets::kTailWidthSize}),
            make_configs(presets::kDefaultDTypes, presets::kLayoutsFull, {Roi::Full, Roi::Partial},
                         {presets::kDefaultSize, presets::kSubVectorSize}),
        }),
        {PrewittFilterParams{0}, PrewittFilterParams{1}, PrewittFilterParams{2}})),
    op_config_name<PrewittFilterParams>);

// Full name:
// Image_Filter/PrewittFilterNegativeTest.Negative/<Backend>_U8toU8_PKD3_FullRoi_<Size>_type<T>[_constant][_dstC1]
class PrewittFilterNegativeTest : public SkipListTest<WithParams<PrewittFilterNegativeParams>> {};

TEST_P(PrewittFilterNegativeTest, Negative) {
    const auto& p = GetParam();
    run_prewitt_filter_negative(p.cfg, p.op);
}

INSTANTIATE_TEST_SUITE_P(Image_Filter, PrewittFilterNegativeTest,
                         ::testing::ValuesIn(with_params<PrewittFilterNegativeParams>(
                             make_configs({DType::U8}, {Layout::PKD3}, {Roi::Full}),
                             {PrewittFilterNegativeParams{3, RpptImageBorderType::REPLICATE, false},
                              PrewittFilterNegativeParams{0, RpptImageBorderType::CONSTANT, false},
                              PrewittFilterNegativeParams{0, RpptImageBorderType::REPLICATE,
                                                          true}})),
                         op_config_name<PrewittFilterNegativeParams>);
