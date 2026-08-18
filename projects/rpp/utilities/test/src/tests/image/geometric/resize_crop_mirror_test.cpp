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
#include "reference/resize_crop_mirror_ref.hpp"

using namespace rpptest;

namespace {

std::string interp_token(RpptInterpolationType i) {
    return i == NEAREST_NEIGHBOR ? "NN" : "BILINEAR";
}

// A per-image destination size, interpolation, and mirror flag. The scale is derived from the source
// ROI (same target size is a bigger upscale from the half-size partial ROI); mirror flips the output
// horizontally.
struct ResizeCropMirrorParams {
    Rpp32u dstW, dstH;
    RpptInterpolationType interp;
    Rpp32u mirror;
    std::string tag;
    std::string name() const {
        return tag + "_" + std::to_string(dstW) + "x" + std::to_string(dstH) + "_" +
               interp_token(interp) + "_m" + std::to_string(mirror);
    }
};

// Tolerances are set from legitimate numeric error only; they are NOT loosened to hide kernel bugs.
// NN copies a texel verbatim (bit-exact); bilinear blends in float. Same sampling as resize, so the
// bilinear last-column/row-short kernel defect surfaces here too.
double resize_crop_mirror_tolerance(DType dt, RpptInterpolationType interp) {
    if (interp == NEAREST_NEIGHBOR) return 0.0;
    return kRoundingTolerance(dt);
}

template <typename T>
void run_resize_crop_mirror(const TestConfig& cfg, const ResizeCropMirrorParams& op) {
    const Rpp32u c = static_cast<Rpp32u>(channels_of(cfg.layout));
    const TensorShape srcShape{cfg.size.n, c, cfg.size.h, cfg.size.w};
    const TensorShape dstShape{cfg.size.n, c, op.dstH, op.dstW};
    RpptDesc srcDesc = make_descriptor(srcShape, cfg.dtype, cfg.layout);
    RpptDesc dstDesc = make_descriptor(dstShape, cfg.dtype, cfg.layout);
    const std::size_t srcCount = element_count(srcDesc), dstCount = element_count(dstDesc);
    const std::size_t srcBytes = byte_size(srcDesc, cfg.dtype);
    const std::size_t dstBytes = byte_size(dstDesc, cfg.dtype);

    PinnedArray<RpptROI> roi(cfg.backend, srcShape.n);
    PinnedArray<RpptImagePatch> dstSizes(cfg.backend, srcShape.n);
    PinnedArray<Rpp32u> mirror(cfg.backend, srcShape.n);
    const std::vector<RpptROI> roiVec = make_roi(srcDesc, cfg.roi);
    for (Rpp32u i = 0; i < srcShape.n; ++i) {
        roi[i] = roiVec[i];
        dstSizes[i] = RpptImagePatch{op.dstW, op.dstH};
        mirror[i] = op.mirror;
    }

    // (1) Host golden model. golden starts as a distinct dst pattern so the untouched padding is
    // defined; only the dstW x dstH destination image is overwritten.
    std::vector<T> input(srcCount), dstInit(dstCount), golden(dstCount), actual(dstCount);
    fill_input<T>(input.data(), srcCount, cfg.dtype);
    fill_input<T>(dstInit.data(), dstCount, cfg.dtype, /*salt=*/1);
    golden = dstInit;
    resize_crop_mirror_reference<T>(input.data(), srcDesc, golden.data(), dstDesc, cfg.dtype,
                                    roi.data(), XYWH, dstSizes.data(), mirror.data(), op.interp);

    // (2) Run RPP on the configured backend.
    DeviceTensor src(cfg.backend, srcBytes), dst(cfg.backend, dstBytes);
    src.write(input.data(), srcBytes);
    dst.write(dstInit.data(), dstBytes);

    RppHandle handle(cfg.backend, srcShape.n);
    ASSERT_EQ(rppt_resize_crop_mirror(src.ptr(), &srcDesc, dst.ptr(), &dstDesc, dstSizes.data(),
                                      op.interp, mirror.data(), roi.data(), XYWH, handle.get(),
                                      cfg.backend),
              RPP_SUCCESS);

    // (3) Retrieve the result on the host.
    handle.sync();
    dst.read(actual.data(), dstBytes);

    // (4) Compare over the full destination image (packed at the origin).
    std::vector<RpptROI> dstRoiVec = make_roi(dstDesc, Roi::Full);
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), dstDesc, dstRoiVec.data(), XYWH,
                               resize_crop_mirror_tolerance(cfg.dtype, op.interp)));
}

}  // namespace

// Full name:
// Image_Geometric/ResizeCropMirrorTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<Tag>_<DstWxDstH>_<Interp>_<Mirror>
//
// Every case is currently red against this (correct) golden, documenting two kernel defects:
//  - NEAREST_NEIGHBOR returns RPP_ERROR_NOT_IMPLEMENTED (both backends) -- the ASSERT on the return
//    value fails.
//  - BILINEAR runs but samples the trailing edge one texel short (interior matches the golden
//    exactly); the same bilinear defect as resize et al.
class ResizeCropMirrorTest : public SkipListTest<WithParams<ResizeCropMirrorParams>> {};

TEST_P(ResizeCropMirrorTest, Correctness) {
    const auto& p = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(p.cfg.dtype, [&](auto tag) {
        run_resize_crop_mirror<Element<decltype(tag)>>(p.cfg, p.op);
    });
}

// Cover the {mirror on/off} x {NN/bilinear} x {up/down} axes: up-NN-nomirror, up-bilinear-mirror,
// down-NN-mirror, down-bilinear-nomirror.
INSTANTIATE_TEST_SUITE_P(
    Image_Geometric, ResizeCropMirrorTest,
    ::testing::ValuesIn(with_params<ResizeCropMirrorParams>(
        make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                     {Layout::PKD3, Layout::PLN3, Layout::PLN1}, {Roi::Full, Roi::Partial}),
        {ResizeCropMirrorParams{72, 54, NEAREST_NEIGHBOR, 0, "up"},
         ResizeCropMirrorParams{72, 54, BILINEAR, 1, "upmir"},
         ResizeCropMirrorParams{24, 18, NEAREST_NEIGHBOR, 1, "downmir"},
         ResizeCropMirrorParams{24, 18, BILINEAR, 0, "down"}})),
    op_config_name<ResizeCropMirrorParams>);
