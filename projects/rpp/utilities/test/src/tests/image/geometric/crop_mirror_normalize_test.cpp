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
#include "framework/tensor_setup.hpp"
#include "framework/tolerance.hpp"
#include "reference/crop_mirror_normalize_ref.hpp"

using namespace rpptest;

namespace {

// The op takes offset/multiplier, but they encode a normalize: offset = -mean/stdDev,
// multiplier = 1/stdDev, so the sets are written in mean/stdDev terms and converted at setup. mean
// is in [0,255] intensity units (see crop_mirror_normalize_ref.hpp).
struct CmnParams {
    double mean[3];
    double stdDev;
    Rpp32u mirror;
    std::string tag;
    std::string name() const { return tag; }
};

// mean 0 / stdDev 1 is the identity normalize, i.e. a pure crop (+mirror): no arithmetic, so those
// sets are bit-exact. The rest carry the fp error of the multiply and the integer round's ties.
double cmn_tolerance(DType dt, const CmnParams& op) {
    if (op.mean[0] == 0.0 && op.mean[1] == 0.0 && op.mean[2] == 0.0 && op.stdDev == 1.0) return 0.0;
    return kRoundingTolerance(dt);
}

template <typename T>
void run_crop_mirror_normalize(const TestConfig& cfg, const CmnParams& op) {
    const Rpp32u c = static_cast<Rpp32u>(channels_of(cfg.layout));
    const TensorShape shape{cfg.size.n, c, cfg.size.h, cfg.size.w};
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layout);  // RPP takes a non-const ptr
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    PinnedArray<Rpp32f> offset(cfg.backend, static_cast<std::size_t>(shape.n) * c);
    PinnedArray<Rpp32f> multiplier(cfg.backend, static_cast<std::size_t>(shape.n) * c);
    PinnedArray<Rpp32u> mirror(cfg.backend, shape.n);
    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u n = 0; n < shape.n; ++n) {
        mirror[n] = op.mirror;
        roi[n] = roiVec[n];
        for (Rpp32u ch = 0; ch < c; ++ch) {
            offset[n * c + ch] = static_cast<Rpp32f>(-op.mean[ch] / op.stdDev);
            multiplier[n * c + ch] = static_cast<Rpp32f>(1.0 / op.stdDev);
        }
    }

    // (1) Host golden model. golden starts as the dst init pattern so the untouched (outside-ROI)
    // region is defined; only the cropped region is overwritten by the reference.
    std::vector<T> input(count), dstInit(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    fill_input<T>(dstInit.data(), count, cfg.dtype, /*salt=*/1);
    golden = dstInit;
    crop_mirror_normalize_reference<T>(input.data(), golden.data(), desc, cfg.dtype, roiVec.data(),
                                       XYWH, offset.data(), multiplier.data(), mirror.data());

    // (2) Run RPP on the configured backend. dst is pre-filled with a distinct pattern so a
    // no-op kernel would be caught even in the identity set.
    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(dstInit.data(), bytes);

    RppHandle handle(cfg.backend, shape.n);
    ASSERT_EQ(rppt_crop_mirror_normalize(src.ptr(), &desc, dst.ptr(), &desc, offset.data(),
                                         multiplier.data(), mirror.data(), roi.data(), XYWH,
                                         handle.get(), cfg.backend),
              RPP_SUCCESS);

    // (3) Retrieve the result on the host (no-op copy for HOST, device->host for HIP).
    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    // (4) Compare over the cropped region, bounded by the pre-call ROI copy: HIP rewrites the
    // caller's XYWH tensor from XYWH to LTRB in place.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), desc, roiVec.data(), XYWH,
                               cmn_tolerance(cfg.dtype, op)));
}

}  // namespace

// Full name:
// Image_Geometric/CropMirrorNormalizeTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<Tag>
class CropMirrorNormalizeTest : public ::testing::TestWithParam<WithParams<CmnParams>> {};

TEST_P(CropMirrorNormalizeTest, Correctness) {
    const auto& p = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(p.cfg.dtype, [&](auto tag) {
        run_crop_mirror_normalize<Element<decltype(tag)>>(p.cfg, p.op);
    });
}

// Four sets, each turning on one more stage of the name. Identity and MirrorOnly are bit-exact
// controls, separating "normalize applied when it should not be" from "mirror on the wrong axis".
// Scale isolates the multiplier: offset 0 and, at 0.5x, no clamping in any dtype. Normalize is the
// legacy harness's own set, the only one where the intensity-space convention is observable.
//
// 30 of the 192 are red against two kernel defects, both left red on purpose: every {F16,F32} x
// Normalize case adds offsetTensor unscaled to a [0,1] pixel, and HOST x I8 x PartialRoi x
// {Scale, Normalize} drops the multiplier in the scalar remainder.
INSTANTIATE_TEST_SUITE_P(
    Image_Geometric, CropMirrorNormalizeTest,
    ::testing::ValuesIn(with_params<CmnParams>(
        make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                     {Layout::PKD3, Layout::PLN3, Layout::PLN1}, {Roi::Full, Roi::Partial}),
        {CmnParams{{0.0, 0.0, 0.0}, 1.0, 0, "Identity"},
         CmnParams{{0.0, 0.0, 0.0}, 1.0, 1, "MirrorOnly"},
         CmnParams{{0.0, 0.0, 0.0}, 2.0, 0, "Scale"},
         CmnParams{{60.0, 80.0, 100.0}, 0.9, 1, "Normalize"}})),
    op_config_name<CmnParams>);
