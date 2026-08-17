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
#include "framework/compare_tensor.hpp"
#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/resize_mirror_normalize_ref.hpp"

using namespace rpptest;

namespace {

// The destination size every case resizes to. From the full ROI (48x36) that is an exact 2x
// downscale; from the half-size partial ROI it is scale 1, i.e. a verbatim resize whose source
// coordinates are all integers -- the configuration where interpolation contributes no error at all.
constexpr Rpp32u kDstW = 24, kDstH = 18;

// mean is in [0,255] intensity units (see resize_mirror_normalize_ref.hpp); {60, 80, 100} is what
// the legacy harness passes. Identity/MirrorOnly use mean 0 / stdDev 1, which is the identity
// normalize under ANY intensity-space reading, so those sets validate the resize + mirror + store
// pipeline independently of that assumption. They pair it with nearest-neighbour, which copies a
// texel verbatim, making the partial-ROI cases an exact bit-for-bit control.
struct RmnParams {
    double mean[3];
    double stdDev;
    Rpp32u mirror;
    RpptInterpolationType interp;
    std::string tag;
    std::string name() const { return tag; }
};

double rmn_tolerance(DType dt, const TestConfig& cfg, const RmnParams& op) {
    // Scale 1 (partial ROI -> the same destination size) puts every source coordinate on an
    // integer, so with the identity normalize the whole pipeline is exact for every dtype.
    const bool exact = op.mean[0] == 0.0 && op.mean[1] == 0.0 && op.mean[2] == 0.0 &&
                       op.stdDev == 1.0 && cfg.roi == Roi::Partial;
    if (exact) return 0.0;
    // Otherwise the only legitimate error is fp rounding of the bilinear blend and the divide.
    switch (dt) {
        case DType::U8: return 1.0;
        case DType::I8: return 1.0;
        case DType::F32: return 2e-3;
        case DType::F16: return 5e-3;
        default: return 0.0;
    }
}

template <typename T>
void run_resize_mirror_normalize(const TestConfig& cfg, const RmnParams& op) {
    const Rpp32u c = static_cast<Rpp32u>(channels_of(cfg.layout));
    const Rpp32u N = cfg.size.n;
    const TensorShape srcShape{N, c, cfg.size.h, cfg.size.w};
    const TensorShape dstShape{N, c, kDstH, kDstW};
    RpptDesc srcDesc = make_descriptor(srcShape, cfg.dtype, cfg.layout);
    RpptDesc dstDesc = make_descriptor(dstShape, cfg.dtype, cfg.layout);
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
        dstSizes[n] = RpptImagePatch{kDstW, kDstH};
        mirror[n] = op.mirror;
        for (Rpp32u ch = 0; ch < c; ++ch) {
            mean[n * c + ch] = static_cast<Rpp32f>(op.mean[ch]);
            stdDev[n * c + ch] = static_cast<Rpp32f>(op.stdDev);
        }
    }

    // (1) Host golden model. The op writes the kDstW x kDstH region at the destination origin, so
    // golden and the device buffer start from the same distinct pattern and only that region is
    // compared.
    std::vector<T> input(srcCount), dstInit(dstCount), golden(dstCount), actual(dstCount);
    fill_input<T>(input.data(), srcCount, cfg.dtype);
    fill_input<T>(dstInit.data(), dstCount, cfg.dtype, /*salt=*/1);
    golden = dstInit;
    resize_mirror_normalize_reference<T>(input.data(), srcDesc, golden.data(), dstDesc, cfg.dtype,
                                         roi.data(), XYWH, dstSizes.data(), mean.data(),
                                         stdDev.data(), mirror.data(), op.interp);

    // (2) Run RPP on the configured backend.
    DeviceTensor src(cfg.backend, srcBytes), dst(cfg.backend, dstBytes);
    src.write(input.data(), srcBytes);
    dst.write(dstInit.data(), dstBytes);

    RppHandle handle(cfg.backend, N);
    ASSERT_EQ(rppt_resize_mirror_normalize(src.ptr(), &srcDesc, dst.ptr(), &dstDesc,
                                           dstSizes.data(), op.interp, mean.data(), stdDev.data(),
                                           mirror.data(), roi.data(), XYWH, handle.get(),
                                           cfg.backend),
              RPP_SUCCESS);

    handle.sync();
    dst.read(actual.data(), dstBytes);

    // (3) Compare the whole destination image (packed at the origin). A full-frame ROI over the
    // destination descriptor walks exactly the written kDstW x kDstH region. The ROI handed to the
    // op is not reused here: HIP rewrites that tensor from XYWH to LTRB in place.
    const std::vector<RpptROI> dstRoiVec = make_roi(dstDesc, Roi::Full);
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), dstDesc, dstRoiVec.data(), XYWH,
                               rmn_tolerance(cfg.dtype, cfg, op)));
}

}  // namespace

// Full name:
// Image_Geometric/ResizeMirrorNormalizeTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<Tag>
class ResizeMirrorNormalizeTest : public ::testing::TestWithParam<WithParams<RmnParams>> {};

TEST_P(ResizeMirrorNormalizeTest, Correctness) {
    const auto& p = GetParam();
    switch (p.cfg.dtype) {
        case DType::U8:
            run_resize_mirror_normalize<Rpp8u>(p.cfg, p.op);
            break;
        case DType::F16:
            run_resize_mirror_normalize<Rpp16f>(p.cfg, p.op);
            break;
        case DType::F32:
            run_resize_mirror_normalize<Rpp32f>(p.cfg, p.op);
            break;
        case DType::I8:
            run_resize_mirror_normalize<Rpp8s>(p.cfg, p.op);
            break;
        default:
            FAIL() << "unsupported dtype for resize_mirror_normalize";
    }
}

// Four sets, each isolating one more stage: the plain resize, the mirror, the mean subtraction, and
// finally the stdDev divide on top of the mirror. IdentityNN is expected red on the ASSERT: the op
// returns RPP_ERROR_NOT_IMPLEMENTED (-6) for nearest-neighbour on both backends, exactly as
// resize_crop_mirror does. MirrorOnly is the exact control instead -- under the partial ROI the
// destination size equals the ROI, so the resize is scale 1, every source coordinate is an integer
// and bilinear is exact.
INSTANTIATE_TEST_SUITE_P(
    Image_Geometric, ResizeMirrorNormalizeTest,
    ::testing::ValuesIn(with_params<RmnParams>(
        make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                     {Layout::PKD3, Layout::PLN3, Layout::PLN1}, {Roi::Full, Roi::Partial}),
        {RmnParams{{0.0, 0.0, 0.0}, 1.0, 0, NEAREST_NEIGHBOR, "IdentityNN"},
         RmnParams{{0.0, 0.0, 0.0}, 1.0, 1, BILINEAR, "MirrorOnly"},
         RmnParams{{60.0, 80.0, 100.0}, 1.0, 0, BILINEAR, "Mean"},
         RmnParams{{60.0, 80.0, 100.0}, 2.0, 1, BILINEAR, "MeanStdMirror"}})),
    op_config_name<RmnParams>);
