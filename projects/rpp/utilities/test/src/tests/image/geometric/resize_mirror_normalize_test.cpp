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
#include "reference/resize_mirror_normalize_ref.hpp"

using namespace rpptest;

namespace {

// A 2x downscale from the full ROI; scale 1 (integer source coordinates) from the partial ROI.
constexpr Rpp32u kDstW = 24, kDstH = 18;

// Per-channel mean (in [0,255] intensity units) and stdDev. mean 0 / stdDev 1 is the identity
// normalize under any intensity-space convention.
struct RmnParams {
    double mean[3];
    double stdDev[3];
    Rpp32u mirror;
    RpptInterpolationType interp;
    std::string tag;
    // A width that is not a multiple of 8 also runs the scalar tail.
    Rpp32u dstW = kDstW;
    std::string name() const {
        return tag;
    }
};

double rmn_tolerance(DType dt, const TestConfig& cfg, const RmnParams& op) {
    // Scale 1 with the identity normalize is exact for every dtype.
    const bool exact = op.mean[0] == 0.0 && op.mean[1] == 0.0 && op.mean[2] == 0.0 &&
                       op.stdDev[0] == 1.0 && op.stdDev[1] == 1.0 && op.stdDev[2] == 1.0 &&
                       cfg.roi == Roi::Partial;
    if (exact) return 0.0;
    return kRoundingTolerance(dt);
}

template <typename T>
void run_resize_mirror_normalize(const TestConfig& cfg, const RmnParams& op) {
    const Rpp32u c = static_cast<Rpp32u>(channels_of(cfg.layoutIn));
    const Rpp32u N = cfg.size.n;
    const TensorShape srcShape{N, c, cfg.size.h, cfg.size.w};
    const TensorShape dstShape{N, static_cast<Rpp32u>(channels_of(cfg.layoutOut)), kDstH, op.dstW};
    RpptDesc srcDesc = make_descriptor(srcShape, cfg.dtype, cfg.layoutIn);
    RpptDesc dstDesc = make_descriptor(dstShape, cfg.dtype, cfg.layoutOut);
    const std::size_t srcCount = element_count(srcDesc), dstCount = element_count(dstDesc);
    const std::size_t srcBytes = byte_size(srcDesc, cfg.dtype);
    const std::size_t dstBytes = byte_size(dstDesc, cfg.dtype);

    PinnedArray<RpptROI> roi(cfg.backend, N);
    const std::vector<RpptROI> roiVec = make_roi(srcDesc, cfg.roi);
    for (Rpp32u i = 0; i < N; ++i) roi[i] = roiVec[i];

    PinnedArray<RpptImagePatch> dstSizes(cfg.backend, N);
    PinnedArray<Rpp32f> mean(cfg.backend, static_cast<std::size_t>(N) * c);
    PinnedArray<Rpp32f> stdDev(cfg.backend, static_cast<std::size_t>(N) * c);
    PinnedArray<Rpp32u> mirror(cfg.backend, N);
    for (Rpp32u n = 0; n < N; ++n) {
        dstSizes[n] = RpptImagePatch{op.dstW, kDstH};
        mirror[n] = op.mirror;
        for (Rpp32u ch = 0; ch < c; ++ch) {
            mean[n * c + ch] = static_cast<Rpp32f>(op.mean[ch]);
            stdDev[n * c + ch] = static_cast<Rpp32f>(op.stdDev[ch]);
        }
    }

    std::vector<T> input(srcCount), dstInit(dstCount), golden(dstCount), actual(dstCount);
    fill_input<T>(input.data(), srcCount, cfg.dtype);
    fill_input<T>(dstInit.data(), dstCount, cfg.dtype, /*salt=*/1);
    golden = dstInit;
    resize_mirror_normalize_reference<T>(input.data(), srcDesc, golden.data(), dstDesc, cfg.dtype,
                                         roi.data(), XYWH, dstSizes.data(), mean.data(),
                                         stdDev.data(), mirror.data(), op.interp);

    DeviceTensor src(cfg.backend, srcBytes), dst(cfg.backend, dstBytes);
    src.write(input.data(), srcBytes);
    dst.write(dstInit.data(), dstBytes);

    RppHandle handle(cfg.backend, N);
    ASSERT_EQ(rppt_resize_mirror_normalize(
                  src.ptr(), &srcDesc, dst.ptr(), &dstDesc, dstSizes.data(), op.interp, mean.data(),
                  stdDev.data(), mirror.data(), roi.data(), XYWH, handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();
    dst.read(actual.data(), dstBytes);

    const std::vector<RpptROI> dstRoiVec = make_roi(dstDesc, Roi::Full);
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), dstDesc, dstRoiVec.data(), XYWH,
                               rmn_tolerance(cfg.dtype, cfg, op)));
}

}  // namespace

// Full name:
// Image_Geometric/ResizeMirrorNormalizeTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<Tag>
class ResizeMirrorNormalizeTest : public SkipListTest<WithParams<RmnParams>> {};

TEST_P(ResizeMirrorNormalizeTest, Correctness) {
    const auto& p = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(p.cfg.dtype, [&](auto tag) {
        run_resize_mirror_normalize<Element<decltype(tag)>>(p.cfg, p.op);
    });
}

// The first four sets each add one stage. MeanPerChannelStd uses distinct per-channel stdDev and a
// width of 27, whose 3-column scalar tail hits the HOST U8 PKD3 channel swap.
INSTANTIATE_TEST_SUITE_P(
    Image_Geometric, ResizeMirrorNormalizeTest,
    ::testing::ValuesIn(with_params<RmnParams>(
        make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                     {{Layout::PKD3, Layout::PKD3},
                      {Layout::PLN3, Layout::PLN3},
                      {Layout::PLN1, Layout::PLN1},
                      {Layout::PKD3, Layout::PLN3},
                      {Layout::PLN3, Layout::PKD3}},
                     {Roi::Full, Roi::Partial}, {presets::kDefaultSize, presets::kTailWidthSize}),
        {RmnParams{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, 0, NEAREST_NEIGHBOR, "IdentityNN"},
         RmnParams{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, 1, BILINEAR, "MirrorOnly"},
         RmnParams{{60.0, 80.0, 100.0}, {1.0, 1.0, 1.0}, 0, BILINEAR, "Mean"},
         RmnParams{{60.0, 80.0, 100.0}, {2.0, 2.0, 2.0}, 1, BILINEAR, "MeanStdMirror"},
         RmnParams{{60.0, 80.0, 100.0}, {1.0, 2.0, 4.0}, 0, BILINEAR, "MeanPerChannelStd", 27}})),
    op_config_name<RmnParams>);
