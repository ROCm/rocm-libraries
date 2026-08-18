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
#include "framework/dtype_dispatch.hpp"
#include "framework/skip_list.hpp"
#include "framework/tensor_setup.hpp"
#include "framework/tolerance.hpp"
#include "reference/resize_ref.hpp"

using namespace rpptest;

namespace {

std::string interp_token(RpptInterpolationType i) {
    return i == NEAREST_NEIGHBOR ? "NN" : "BILINEAR";
}

// A per-image destination size plus the interpolation to sample with. The scale is derived from the
// source ROI, so the same target size is an upscale from the full ROI and a larger upscale from the
// half-size partial ROI; "down" from the partial ROI happens to be scale 1 (a verbatim resize).
struct ResizeParams {
    Rpp32u dstW, dstH;
    RpptInterpolationType interp;
    std::string tag;
    std::string name() const {
        return tag + "_" + std::to_string(dstW) + "x" + std::to_string(dstH) + "_" +
               interp_token(interp);
    }
};

// Tolerances are set from legitimate numeric error only; they are NOT loosened to hide the real
// kernel defect this test surfaces: bilinear samples the last
// column/row one texel short (same root cause as #15). All 96 NN cases and every HOST F16/I8
// bilinear case pass at these tolerances, validating the golden; the red cases are the kernel bug.
double resize_tolerance(DType dt, RpptInterpolationType interp) {
    // Nearest-neighbour copies a texel verbatim, so it is bit-exact for every dtype.
    if (interp == NEAREST_NEIGHBOR) return 0.0;
    // Bilinear blends in float; allow only genuine rounding error. Integer types round to nearest.
    return kRoundingTolerance(dt);
}

template <typename T>
void run_resize(const TestConfig& cfg, const ResizeParams& op) {
    const Rpp32u c = static_cast<Rpp32u>(channels_of(cfg.layout));
    const TensorShape srcShape{cfg.size.n, c, cfg.size.h, cfg.size.w};
    const TensorShape dstShape{cfg.size.n, c, op.dstH, op.dstW};
    RpptDesc srcDesc = make_descriptor(srcShape, cfg.dtype, cfg.layout);
    RpptDesc dstDesc = make_descriptor(dstShape, cfg.dtype, cfg.layout);
    const std::size_t srcCount = element_count(srcDesc), dstCount = element_count(dstDesc);
    const std::size_t srcBytes = byte_size(srcDesc, cfg.dtype);
    const std::size_t dstBytes = byte_size(dstDesc, cfg.dtype);

    // Per-image source ROIs and destination sizes in host-accessible (pinned for HIP) memory.
    PinnedArray<RpptROI> roi(cfg.backend, srcShape.n);
    PinnedArray<RpptImagePatch> dstSizes(cfg.backend, srcShape.n);
    const std::vector<RpptROI> roiVec = make_roi(srcDesc, cfg.roi);
    for (Rpp32u i = 0; i < srcShape.n; ++i) {
        roi[i] = roiVec[i];
        dstSizes[i] = RpptImagePatch{op.dstW, op.dstH};
    }

    // (1) Host golden model. golden starts as a distinct dst pattern so the untouched (outside the
    // written dstW x dstH region) padding is defined; only the destination image is overwritten.
    std::vector<T> input(srcCount), dstInit(dstCount), golden(dstCount), actual(dstCount);
    fill_input<T>(input.data(), srcCount, cfg.dtype);
    fill_input<T>(dstInit.data(), dstCount, cfg.dtype, /*salt=*/1);
    golden = dstInit;
    resize_reference<T>(input.data(), srcDesc, golden.data(), dstDesc, cfg.dtype, roi.data(), XYWH,
                        dstSizes.data(), op.interp);

    // (2) Run RPP on the configured backend.
    DeviceTensor src(cfg.backend, srcBytes), dst(cfg.backend, dstBytes);
    src.write(input.data(), srcBytes);
    dst.write(dstInit.data(), dstBytes);

    RppHandle handle(cfg.backend, srcShape.n);
    ASSERT_EQ(rppt_resize(src.ptr(), &srcDesc, dst.ptr(), &dstDesc, dstSizes.data(), op.interp,
                          roi.data(), XYWH, handle.get(), cfg.backend),
              RPP_SUCCESS);

    // (3) Retrieve the result on the host (no-op copy for HOST, device->host for HIP).
    handle.sync();
    dst.read(actual.data(), dstBytes);

    // (4) Compare over the full destination image (packed at the origin). A full-frame ROI over the
    // destination descriptor walks exactly the dstW x dstH written region.
    std::vector<RpptROI> dstRoiVec = make_roi(dstDesc, Roi::Full);
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), dstDesc, dstRoiVec.data(), XYWH,
                               resize_tolerance(cfg.dtype, op.interp)));
}

}  // namespace

// Full name: Image_Geometric/ResizeTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<Tag>_<DstWxDstH>_<Interp>
class ResizeTest : public SkipListTest<WithParams<ResizeParams>> {};

TEST_P(ResizeTest, Correctness) {
    const auto& p = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(p.cfg.dtype, [&](auto tag) {
        run_resize<Element<decltype(tag)>>(p.cfg, p.op);
    });
}

// up: enlarge the source ROI. down: shrink it (from the partial ROI this is scale 1, a verbatim
// resize). Each exercised with nearest-neighbour and bilinear sampling.
INSTANTIATE_TEST_SUITE_P(
    Image_Geometric, ResizeTest,
    ::testing::ValuesIn(with_params<ResizeParams>(
        make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                     {Layout::PKD3, Layout::PLN3, Layout::PLN1}, {Roi::Full, Roi::Partial}),
        {ResizeParams{72, 54, NEAREST_NEIGHBOR, "up"},
         ResizeParams{72, 54, BILINEAR, "up"},
         ResizeParams{24, 18, NEAREST_NEIGHBOR, "down"},
         ResizeParams{24, 18, BILINEAR, "down"}})),
    op_config_name<ResizeParams>);
