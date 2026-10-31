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
#include "reference/bayer_to_rgb_ref.hpp"

using namespace rpptest;

namespace {

std::string pattern_name(RpptBayerPattern p) {
    static const char* kNames[4] = {"RGGB", "BGGR", "GRBG", "GBRG"};
    return kNames[p];
}

const std::vector<RpptBayerPattern> kPatterns{RpptBayerPattern_RGGB, RpptBayerPattern_BGGR,
                                              RpptBayerPattern_GRBG, RpptBayerPattern_GBRG};

// phaseX/phaseY are the parity of the partial ROI's x/y offset, which decides the colour of every
// pixel since the pattern is anchored at the ROI origin. Full ROIs always have phase 0.
struct BayerToRgbParams {
    RpptBayerPattern pattern;
    int phaseX, phaseY;
    std::string name() const {
        return pattern_name(pattern) + "_ph" + std::to_string(phaseX) + std::to_string(phaseY);
    }
};

// Partial ROIs sit at an even base offset plus the phase; image n is moved down n more rows, so
// the images of a batch decode at different phases and a kernel reading roi[0] fails.
std::vector<RpptROI> bayer_roi(const RpptDesc& d, Roi mode, int phaseX, int phaseY) {
    std::vector<RpptROI> roi = make_roi(d, mode);
    if (mode == Roi::Full) return roi;
    for (Rpp32u n = 0; n < d.n; ++n) {
        roi[n].xywhROI.xy.x = static_cast<int>(2 * (d.w / 8)) + phaseX;
        roi[n].xywhROI.xy.y = static_cast<int>(2 * (d.h / 8) + n) + phaseY;
    }
    return roi;
}

// Means of 2 or 4 values: one LSB for U8/I8 rounding, float accumulation slack otherwise.
constexpr Tolerance kBayerToRgbTolerance = kRoundingTolerance;

template <typename T>
void run_bayer_to_rgb(const TestConfig& cfg, const BayerToRgbParams& op) {
    RpptDesc srcDesc = make_src_descriptor(cfg);
    RpptDesc dstDesc = make_dst_descriptor(cfg);
    const std::size_t srcCount = element_count(srcDesc), dstCount = element_count(dstDesc);
    const std::size_t srcBytes = byte_size(srcDesc, cfg.dtype);
    const std::size_t dstBytes = byte_size(dstDesc, cfg.dtype);

    PinnedArray<RpptROI> roi(cfg.backend, cfg.size.n);
    const std::vector<RpptROI> roiVec = bayer_roi(srcDesc, cfg.roi, op.phaseX, op.phaseY);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) roi[i] = roiVec[i];

    // dst and golden share a seed distinct from the input so a no-op kernel would be caught.
    std::vector<T> input(srcCount), dstInit(dstCount), golden(dstCount), actual(dstCount);
    fill_input<T>(input.data(), srcCount, cfg.dtype);
    fill_input<T>(dstInit.data(), dstCount, cfg.dtype, /*salt=*/1);
    golden = dstInit;
    bayer_to_rgb_reference<T>(input.data(), srcDesc, golden.data(), dstDesc, cfg.dtype,
                              roiVec.data(), XYWH, op.pattern);

    DeviceTensor src(cfg.backend, srcBytes), dst(cfg.backend, dstBytes);
    src.write(input.data(), srcBytes);
    dst.write(dstInit.data(), dstBytes);

    RppHandle handle(cfg.backend, cfg.size.n);
    ASSERT_EQ(rppt_bayer_to_rgb(src.ptr(), &srcDesc, dst.ptr(), &dstDesc, op.pattern,
                                RpptImageBorderType::REFLECT_NO_EDGE, roi.data(), XYWH,
                                handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), dstBytes);

    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), dstDesc, roiVec.data(), XYWH,
                               kBayerToRgbTolerance(cfg.dtype)));
}

std::vector<WithParams<BayerToRgbParams>> bayer_to_rgb_cases() {
    const std::vector<LayoutConv> layouts{{Layout::PLN1, Layout::PKD3},
                                          {Layout::PLN1, Layout::PLN3}};
    const std::vector<Size> sizes{presets::kDefaultSize, presets::kTailWidthSize,
                                  presets::kSubVectorSize};
    std::vector<BayerToRgbParams> full, partial;
    for (RpptBayerPattern p : kPatterns) {
        full.push_back({p, 0, 0});
        for (int py = 0; py < 2; ++py)
            for (int px = 0; px < 2; ++px) partial.push_back({p, px, py});
    }
    auto cases =
        with_params(make_configs(presets::kDefaultDTypes, layouts, {Roi::Full}, sizes), full);
    const auto partialCases =
        with_params(make_configs(presets::kDefaultDTypes, layouts, {Roi::Partial}, sizes), partial);
    cases.insert(cases.end(), partialCases.begin(), partialCases.end());
    return cases;
}

// One illegal argument per case; only the status is asserted.
enum class BayerFault { Pattern, Border, SrcChannels, DstChannels };

struct BayerToRgbNegativeParams {
    BayerFault fault;
    std::string name() const {
        switch (fault) {
            case BayerFault::Pattern:
                return "pattern4";
            case BayerFault::Border:
                return "replicate";
            case BayerFault::SrcChannels:
                return "src3ch";
            case BayerFault::DstChannels:
                return "dst1ch";
        }
        return "unknown";
    }
};

void run_bayer_to_rgb_negative(const TestConfig& cfg, const BayerToRgbNegativeParams& op) {
    TestConfig srcCfg = cfg, dstCfg = cfg;
    if (op.fault == BayerFault::SrcChannels) srcCfg.layoutIn = Layout::PLN3;
    if (op.fault == BayerFault::DstChannels) dstCfg.layoutOut = Layout::PLN1;
    RpptDesc srcDesc = make_src_descriptor(srcCfg);
    RpptDesc dstDesc = make_dst_descriptor(dstCfg);
    const RpptBayerPattern pattern =
        op.fault == BayerFault::Pattern ? static_cast<RpptBayerPattern>(4) : RpptBayerPattern_RGGB;
    const RpptImageBorderType border = op.fault == BayerFault::Border
                                           ? RpptImageBorderType::REPLICATE
                                           : RpptImageBorderType::REFLECT_NO_EDGE;

    PinnedArray<RpptROI> roi(cfg.backend, cfg.size.n);
    const std::vector<RpptROI> roiVec = make_roi(srcDesc, Roi::Full);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) roi[i] = roiVec[i];

    std::vector<Rpp8u> input(element_count(srcDesc));
    fill_input<Rpp8u>(input.data(), input.size(), cfg.dtype);
    DeviceTensor src(cfg.backend, byte_size(srcDesc, cfg.dtype));
    DeviceTensor dst(cfg.backend, byte_size(dstDesc, cfg.dtype));
    src.write(input.data(), byte_size(srcDesc, cfg.dtype));

    RppHandle handle(cfg.backend, cfg.size.n);
    const RppStatus status = rppt_bayer_to_rgb(src.ptr(), &srcDesc, dst.ptr(), &dstDesc, pattern,
                                               border, roi.data(), XYWH, handle.get(), cfg.backend);
    handle.sync();
    EXPECT_NE(status, RPP_SUCCESS)
        << "bayer_to_rgb accepted an illegal argument (" << op.name() << ")";
}

}  // namespace

// Full name:
// Image_DataExchange/BayerToRgbTest.Correctness/<Backend>_<DType>to<DType>_PLN1to<Layout>_<Roi>_<Size>_<Pattern>_ph<X><Y>
class BayerToRgbTest : public SkipListTest<WithParams<BayerToRgbParams>> {};

TEST_P(BayerToRgbTest, Correctness) {
    const auto& p = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(
        p.cfg.dtype, [&](auto tag) { run_bayer_to_rgb<Element<decltype(tag)>>(p.cfg, p.op); });
}

INSTANTIATE_TEST_SUITE_P(Image_DataExchange, BayerToRgbTest,
                         ::testing::ValuesIn(bayer_to_rgb_cases()),
                         op_config_name<BayerToRgbParams>);

// Full name:
// Image_DataExchange/BayerToRgbNegativeTest.Negative/<Backend>_U8toU8_PLN1toPKD3_FullRoi_<Size>_<Fault>
class BayerToRgbNegativeTest : public SkipListTest<WithParams<BayerToRgbNegativeParams>> {};

TEST_P(BayerToRgbNegativeTest, Negative) {
    const auto& p = GetParam();
    run_bayer_to_rgb_negative(p.cfg, p.op);
}

INSTANTIATE_TEST_SUITE_P(Image_DataExchange, BayerToRgbNegativeTest,
                         ::testing::ValuesIn(with_params<BayerToRgbNegativeParams>(
                             make_configs({DType::U8},
                                          std::vector<LayoutConv>{{Layout::PLN1, Layout::PKD3}},
                                          {Roi::Full}),
                             {BayerToRgbNegativeParams{BayerFault::Pattern},
                              BayerToRgbNegativeParams{BayerFault::Border},
                              BayerToRgbNegativeParams{BayerFault::SrcChannels},
                              BayerToRgbNegativeParams{BayerFault::DstChannels}})),
                         op_config_name<BayerToRgbNegativeParams>);
