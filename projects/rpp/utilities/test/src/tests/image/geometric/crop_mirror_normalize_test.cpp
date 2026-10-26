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
#include "reference/crop_mirror_normalize_ref.hpp"

using namespace rpptest;

namespace {

// Written as mean/stdDev and converted to offset = -mean/stdDev, multiplier = 1/stdDev. mean is in
// [0,255] intensity units.
struct CmnParams {
    double mean[3];
    double stdDev;
    Rpp32u mirror;
    std::string tag;
    std::string name() const {
        return tag;
    }
};

// mean 0 / stdDev 1 is a pure crop (+mirror), so those sets are bit-exact.
double cmn_tolerance(DType dt, const CmnParams& op) {
    if (op.mean[0] == 0.0 && op.mean[1] == 0.0 && op.mean[2] == 0.0 && op.stdDev == 1.0) return 0.0;
    return kRoundingTolerance(dt);
}

template <typename T>
void run_crop_mirror_normalize(const TestConfig& cfg, const CmnParams& op) {
    const Rpp32u c = static_cast<Rpp32u>(channels_of(cfg.layoutIn));
    RpptDesc srcDesc = make_src_descriptor(cfg);  // RPP takes a non-const ptr
    RpptDesc dstDesc = make_dst_descriptor(cfg);
    const std::size_t srcCount = element_count(srcDesc);
    const std::size_t dstCount = element_count(dstDesc);
    const std::size_t srcBytes = byte_size(srcDesc, cfg.dtype);
    const std::size_t dstBytes = byte_size(dstDesc, cfg.dtype);

    PinnedArray<Rpp32f> offset(cfg.backend, static_cast<std::size_t>(cfg.size.n) * c);
    PinnedArray<Rpp32f> multiplier(cfg.backend, static_cast<std::size_t>(cfg.size.n) * c);
    PinnedArray<Rpp32u> mirror(cfg.backend, cfg.size.n);
    PinnedArray<RpptROI> roi(cfg.backend, cfg.size.n);
    const std::vector<RpptROI> roiVec = make_roi(srcDesc, cfg.roi);
    for (Rpp32u n = 0; n < cfg.size.n; ++n) {
        mirror[n] = op.mirror;
        roi[n] = roiVec[n];
        for (Rpp32u ch = 0; ch < c; ++ch) {
            offset[n * c + ch] = static_cast<Rpp32f>(-op.mean[ch] / op.stdDev);
            multiplier[n * c + ch] = static_cast<Rpp32f>(1.0 / op.stdDev);
        }
    }

    std::vector<T> input(srcCount), dstInit(dstCount), golden(dstCount), actual(dstCount);
    fill_input<T>(input.data(), srcCount, cfg.dtype);
    fill_input<T>(dstInit.data(), dstCount, cfg.dtype, /*salt=*/1);
    golden = dstInit;
    crop_mirror_normalize_reference<T>(input.data(), srcDesc, golden.data(), dstDesc, cfg.dtype,
                                       roiVec.data(), XYWH, offset.data(), multiplier.data(),
                                       mirror.data());

    // dst is pre-filled with a distinct pattern so a no-op kernel is caught in the identity set.
    DeviceTensor src(cfg.backend, srcBytes), dst(cfg.backend, dstBytes);
    src.write(input.data(), srcBytes);
    dst.write(dstInit.data(), dstBytes);

    RppHandle handle(cfg.backend, cfg.size.n);
    ASSERT_EQ(rppt_crop_mirror_normalize(src.ptr(), &srcDesc, dst.ptr(), &dstDesc, offset.data(),
                                         multiplier.data(), mirror.data(), roi.data(), XYWH,
                                         handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), dstBytes);

    // Uses the test's own ROI copy: HIP rewrites the op's ROI to LTRB in place.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), dstDesc, roiVec.data(), XYWH,
                               cmn_tolerance(cfg.dtype, op)));
}

}  // namespace

// Full name:
// Image_Geometric/CropMirrorNormalizeTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<Tag>
class CropMirrorNormalizeTest : public SkipListTest<WithParams<CmnParams>> {};

TEST_P(CropMirrorNormalizeTest, Correctness) {
    const auto& p = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(p.cfg.dtype, [&](auto tag) {
        run_crop_mirror_normalize<Element<decltype(tag)>>(p.cfg, p.op);
    });
}

// Each set turns on one more stage. Known kernel reds: {F16,F32} Normalize adds offsetTensor
// unscaled to a [0,1] pixel, and HOST I8 PartialRoi {Scale, Normalize} drops the multiplier in the
// scalar remainder.
INSTANTIATE_TEST_SUITE_P(Image_Geometric, CropMirrorNormalizeTest,
                         ::testing::ValuesIn(with_params<CmnParams>(
                             concat_configs({
                                 make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                                              presets::kLayoutsFullConv, {Roi::Full, Roi::Partial},
                                              {presets::kTailWidthSize}),
                                 make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                                              presets::kLayoutsFull, {Roi::Full, Roi::Partial},
                                              {presets::kDefaultSize, presets::kSubVectorSize}),
                             }),
                             {CmnParams{{0.0, 0.0, 0.0}, 1.0, 0, "Identity"},
                              CmnParams{{0.0, 0.0, 0.0}, 1.0, 1, "MirrorOnly"},
                              CmnParams{{0.0, 0.0, 0.0}, 2.0, 0, "Scale"},
                              CmnParams{{60.0, 80.0, 100.0}, 0.9, 1, "Normalize"}})),
                         op_config_name<CmnParams>);
