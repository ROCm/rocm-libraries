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

#include <cstdio>
#include <string>
#include <vector>

#include "framework/backend_memory.hpp"
#include "framework/compare.hpp"
#include "framework/config_param.hpp"
#include "framework/dtype.hpp"
#include "framework/skip_list.hpp"
#include "framework/tensor_setup.hpp"
#include "framework/tolerance.hpp"
#include "reference/bilateral_filter_ref.hpp"

using namespace rpptest;

namespace {

// kernelSize is an odd square window size (3/5/7/9); each has its own SIMD path. stdDevSpace is in
// pixels. rawStdDevRange is in [0,255] units and converted to each dtype's stored units at run
// time (see range_in_dtype), the threshold test's pattern, so one parameter set means the same
// smoothing for every dtype.
//
// The parameter table walks the op between its two limits (bilateral_filter_ref.hpp):
//   sr0p1  -> identity: every neighbour that differs from the centre gets ~zero weight.
//   sr10000 -> plain gaussian_filter: the range weight is ~1 everywhere.
//   sr25   -> the edge-preserving regime in between, where the range weight actually varies over
//             the input pattern (adjacent values differ by multiples of 37 in [0,255] units).
//   ss1en20 / sr1en20 -> a tiny but legal sigma, whose 1 / (2 sigma^2) overflows a float; the
//             centre weight must stay exp(0) = 1, so the output is the input.

// num_token prints through std::to_string, which renders 1e-20 as "0" -- the zero sigma the
// negative test rejects. Tiny values keep their exponent instead: 1e-20 -> "1en20".
std::string sigma_token(float v) {
    if (v == 0.0f || v >= 1e-3f) return num_token(v);
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%g", v);
    std::string s = buf;
    for (char& ch : s) {
        if (ch == '.')
            ch = 'p';
        else if (ch == '-')
            ch = 'n';
    }
    return s;
}

struct BilateralFilterParams {
    Rpp32u kernelSize;
    float stdDevSpace;
    float rawStdDevRange;
    std::string name() const {
        return "k" + std::to_string(kernelSize) + "_ss" + sigma_token(stdDevSpace) + "_sr" +
               sigma_token(rawStdDevRange);
    }
};

// A range std-dev is a distance between two pixel values, so the I8 -128 shift does not apply;
// only the float dtypes rescale to [0,1].
float range_in_dtype(float raw, DType dt) {
    return (dt == DType::F16 || dt == DType::F32) ? raw / 255.0f : raw;
}

// Each image of the batch gets its own standard deviations so a kernel that reads index 0 for the
// whole batch fails. Scaling keeps the limiting cases at their limits.
float per_image(float v, Rpp32u n) {
    return v * (1.0f + 0.5f * static_cast<float>(n));
}

// ROIs smaller than the KxK window, where one pixel's window overhangs both opposite edges: 1x1 and
// 2x2, a single row and a single column, 8x8 (one short of K = 9), and 9x9 (exactly K = 9). Full
// ROI only: a partial ROI of a 1-pixel image is empty.
const std::vector<Size> kSmallRoiSizes{{2, 1, 1},  {2, 2, 2}, {2, 1, 20},
                                       {2, 20, 1}, {1, 8, 8}, {1, 9, 9}};

// Tolerances reflect legitimate numeric error only (weighted-mean rounding); U8/I8 allow one LSB
// for round-to-nearest quantization, floats allow accumulation slack.
constexpr Tolerance kBilateralFilterTolerance = tolerance(1.0, 1e-3, 1e-2);

template <typename T>
void run_bilateral_filter(const TestConfig& cfg, const BilateralFilterParams& op) {
    // src carries the same leading border pad as gaussian_filter: the HIP filter kernels require
    // srcDesc.offsetInBytes >= 12 * (kernelSize/2) of read-slack for the KxK window. dst keeps
    // offset 0, so the golden and comparator index destination elements from 0.
    RpptDesc srcDesc = make_src_descriptor(cfg);
    RpptDesc dstDesc = make_dst_descriptor(cfg);
    const std::size_t offsetBytes = 12u * (op.kernelSize / 2);
    const std::size_t offsetElems = offsetBytes / dtype_size(cfg.dtype);
    srcDesc.offsetInBytes = static_cast<Rpp32u>(offsetBytes);

    const std::size_t count = element_count(srcDesc);
    const std::size_t imageBytes = count * dtype_size(cfg.dtype);

    // Both std-dev tensors are per-image Rpp32f* in host-accessible (pinned for HIP) memory.
    PinnedArray<Rpp32f> stdDevSpaceTensor(cfg.backend, cfg.size.n);
    PinnedArray<Rpp32f> stdDevRangeTensor(cfg.backend, cfg.size.n);
    PinnedArray<RpptROI> roi(cfg.backend, cfg.size.n);
    const std::vector<RpptROI> roiVec = make_roi(srcDesc, cfg.roi);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) {
        stdDevSpaceTensor[i] = per_image(op.stdDevSpace, i);
        stdDevRangeTensor[i] = range_in_dtype(per_image(op.rawStdDevRange, i), cfg.dtype);
        roi[i] = roiVec[i];
    }

    // (1) Host golden model. The src buffer is [front pad][image][back pad]; the image base sits
    // at offsetElems. The destination starts from a DIFFERENT pattern than the source (salt = 1):
    // at the sr0p1 identity corner the output equals the input, so seeding dst with the input
    // would let a kernel that wrote nothing pass.
    std::vector<T> src(offsetElems + count + offsetElems), dstInit(count), golden(count),
        actual(count);
    fill_input<T>(src.data(), src.size(), cfg.dtype);
    fill_input<T>(dstInit.data(), count, cfg.dtype, /*salt=*/1);
    T* image = src.data() + offsetElems;
    golden = dstInit;
    bilateral_filter_reference<T>(image, srcDesc, golden.data(), dstDesc, cfg.dtype, roi.data(),
                                  XYWH, op.kernelSize, stdDevSpaceTensor.data(),
                                  stdDevRangeTensor.data());

    // (2) Run RPP on the configured backend. RPP adds srcDesc.offsetInBytes to the src pointer
    // internally, landing on the image base.
    DeviceTensor srcDev(cfg.backend, src.size() * dtype_size(cfg.dtype));
    DeviceTensor dst(cfg.backend, imageBytes);
    srcDev.write(src.data(), src.size() * dtype_size(cfg.dtype));
    dst.write(dstInit.data(), imageBytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, cfg.size.n);
    ASSERT_EQ(rppt_bilateral_filter(srcDev.ptr(), &srcDesc, dst.ptr(), &dstDesc,
                                    stdDevSpaceTensor.data(), stdDevRangeTensor.data(),
                                    op.kernelSize, RpptImageBorderType::REPLICATE, roi.data(), XYWH,
                                    handle.get(), cfg.backend),
              RPP_SUCCESS);

    // (3) Retrieve the result on the host (no-op copy for HOST, device->host for HIP).
    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), imageBytes);

    // (4) Compare within tolerance over the ROI, against the test's own ROI copy in case the
    // backend rewrites the tensor handed to it.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), dstDesc, roiVec.data(), XYWH,
                               kBilateralFilterTolerance(cfg.dtype)));
}

// One out-of-contract argument per case. Argument validation is dtype- and layout-independent, so
// the arguments are the grid axis here.
struct BilateralFilterNegativeParams {
    Rpp32u kernelSize;
    float stdDevSpace;
    float rawStdDevRange;
    RpptImageBorderType borderType;
    std::string name() const {
        const std::string border = borderType == RpptImageBorderType::REPLICATE ? "" : "_constant";
        return "k" + std::to_string(kernelSize) + "_ss" + num_token(stdDevSpace) + "_sr" +
               num_token(rawStdDevRange) + border;
    }
};

// An even or unsupported window, a non-positive standard deviation (a zero sigma divides by zero
// in either exponent) and a border mode other than REPLICATE are not legal calls and must be
// reported rather than silently producing an undefined image. Only the status is asserted: which
// RPP_ERROR* each maps to is the header's to name.
void run_bilateral_filter_negative(const TestConfig& cfg, const BilateralFilterNegativeParams& op) {
    RpptDesc desc = make_src_descriptor(cfg);
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    PinnedArray<Rpp32f> stdDevSpaceTensor(cfg.backend, cfg.size.n);
    PinnedArray<Rpp32f> stdDevRangeTensor(cfg.backend, cfg.size.n);
    PinnedArray<RpptROI> roi(cfg.backend, cfg.size.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) {
        stdDevSpaceTensor[i] = op.stdDevSpace;
        stdDevRangeTensor[i] = range_in_dtype(op.rawStdDevRange, cfg.dtype);
        roi[i] = roiVec[i];
    }

    std::vector<Rpp8u> input(count);
    fill_input<Rpp8u>(input.data(), count, cfg.dtype);

    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);

    RppHandle handle(cfg.backend, cfg.size.n);
    const RppStatus status = rppt_bilateral_filter(
        src.ptr(), &desc, dst.ptr(), &desc, stdDevSpaceTensor.data(), stdDevRangeTensor.data(),
        op.kernelSize, op.borderType, roi.data(), XYWH, handle.get(), cfg.backend);
    handle.sync();
    EXPECT_NE(status, RPP_SUCCESS)
        << "bilateral_filter accepted out-of-contract arguments (" << op.name() << ")";
}

}  // namespace

// Full name:
// Image_Filter/BilateralFilterTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_k<K>_ss<SigmaS>_sr<SigmaR>
class BilateralFilterTest : public SkipListTest<WithParams<BilateralFilterParams>> {};

TEST_P(BilateralFilterTest, Correctness) {
    const auto& p = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(
        p.cfg.dtype, [&](auto tag) { run_bilateral_filter<Element<decltype(tag)>>(p.cfg, p.op); });
}

// Same-layout cases plus both directions of the fused output-layout conversion, then ROIs smaller
// than the window.
INSTANTIATE_TEST_SUITE_P(
    Image_Filter, BilateralFilterTest,
    ::testing::ValuesIn(with_params<BilateralFilterParams>(
        concat_configs({
            make_configs(presets::kDefaultDTypes, presets::kLayoutsFullConv,
                         {Roi::Full, Roi::Partial}, {presets::kTailWidthSize}),
            make_configs(presets::kDefaultDTypes, presets::kLayoutsFull, {Roi::Full, Roi::Partial},
                         {presets::kDefaultSize, presets::kSubVectorSize}),
            make_configs(presets::kDefaultDTypes, presets::kLayoutsFullConv, {Roi::Full},
                         kSmallRoiSizes),
        }),
        {BilateralFilterParams{3, 1.0f, 25.0f}, BilateralFilterParams{5, 1.0f, 25.0f},
         BilateralFilterParams{7, 1.0f, 25.0f}, BilateralFilterParams{9, 1.0f, 25.0f},
         BilateralFilterParams{5, 0.25f, 25.0f}, BilateralFilterParams{5, 5.0f, 25.0f},
         BilateralFilterParams{5, 1.0f, 0.1f}, BilateralFilterParams{5, 1.0f, 10000.0f},
         BilateralFilterParams{5, 1e-20f, 25.0f}, BilateralFilterParams{5, 1.0f, 1e-20f}})),
    op_config_name<BilateralFilterParams>);

// Full name:
// Image_Filter/BilateralFilterNegativeTest.Negative/<Backend>_U8toU8_PKD3_FullRoi_<Size>_k<K>_ss<SigmaS>_sr<SigmaR>[_constant]
class BilateralFilterNegativeTest : public SkipListTest<WithParams<BilateralFilterNegativeParams>> {
};

TEST_P(BilateralFilterNegativeTest, Negative) {
    const auto& p = GetParam();
    run_bilateral_filter_negative(p.cfg, p.op);
}

INSTANTIATE_TEST_SUITE_P(
    Image_Filter, BilateralFilterNegativeTest,
    ::testing::ValuesIn(with_params<BilateralFilterNegativeParams>(
        make_configs({DType::U8}, {Layout::PKD3}, {Roi::Full}),
        {BilateralFilterNegativeParams{4, 1.0f, 25.0f, RpptImageBorderType::REPLICATE},
         BilateralFilterNegativeParams{11, 1.0f, 25.0f, RpptImageBorderType::REPLICATE},
         BilateralFilterNegativeParams{5, 0.0f, 25.0f, RpptImageBorderType::REPLICATE},
         BilateralFilterNegativeParams{5, 1.0f, 0.0f, RpptImageBorderType::REPLICATE},
         BilateralFilterNegativeParams{5, -1.0f, 25.0f, RpptImageBorderType::REPLICATE},
         BilateralFilterNegativeParams{5, 1.0f, -25.0f, RpptImageBorderType::REPLICATE},
         BilateralFilterNegativeParams{5, 1.0f, 25.0f, RpptImageBorderType::CONSTANT}})),
    op_config_name<BilateralFilterNegativeParams>);
