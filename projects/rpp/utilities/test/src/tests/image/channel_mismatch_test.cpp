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

// Every image op whose destination is documented as "c = same as that of srcDescPtr" must reject
// a src/dst channel mismatch with RPP_ERROR_INVALID_DST_CHANNELS before touching the destination.
// The kernels loop over srcDescPtr->c and address dst through its own strides, so an accepted
// 3->1 call writes two planes past the destination and an accepted 1->3 call leaves two of its
// planes unwritten (#14582).
//
// Both buffers are sized for three channels so that, should the check regress, the overrun lands
// in memory the test owns and is reported as a wrong status rather than corrupting the heap. The
// per-op argument buffers are zeroed scratch: a correct op never reads them, and a regressed one
// fails on its status regardless of what it computed from them.
//
// Ops whose destination channel count differs by design (color_to_greyscale, sobel_filter) are
// not listed.

#include <gtest/gtest.h>
#include <rpp/rpp.h>

#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "framework/backend_memory.hpp"
#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"

using namespace rpptest;

namespace {

constexpr Size kSize{2, 16, 20};
constexpr Rpp8u kSentinel = 0xAA;
constexpr std::size_t kScratchBytes = 64 * 1024;

// Everything an entry point may be handed, owned by one test case.
struct Ctx {
    RppBackend backend;
    RpptDesc srcDesc, dstDesc, tableDesc;
    std::size_t bufBytes;
    DeviceTensor src, src2, dst, devScratch;
    PinnedArray<Rpp8u> hostScratch;
    PinnedArray<RpptROI> roi;
    PinnedArray<RpptImagePatch> patch;
    RppHandle handle;

    Ctx(RppBackend b, const RpptDesc& s, const RpptDesc& d, std::size_t bytes)
        : backend(b),
          srcDesc(s),
          dstDesc(d),
          tableDesc(make_descriptor({kSize.n, 1, kSize.h, kSize.w}, DType::F32, Layout::PLN1)),
          bufBytes(bytes),
          src(b, bytes),
          src2(b, bytes),
          dst(b, bytes),
          devScratch(b, element_count(tableDesc) * sizeof(Rpp32f) + kScratchBytes),
          hostScratch(b, kScratchBytes),
          roi(b, kSize.n),
          patch(b, kSize.n),
          handle(b, kSize.n) {}

    // Zeroed host-side parameter tensor, reinterpreted as whatever the op takes.
    template <typename T>
    T* p() {
        return reinterpret_cast<T*>(hostScratch.data());
    }
    // Zeroed backend-side buffer (LUTs, remap tables, colour/noise/intermediate buffers).
    template <typename T = void>
    T* dev() {
        return static_cast<T*>(devScratch.ptr());
    }
    void* s() {
        return src.ptr();
    }
    void* s2() {
        return src2.ptr();
    }
    void* d() {
        return dst.ptr();
    }
    RpptDesc* sd() {
        return &srcDesc;
    }
    RpptDesc* dd() {
        return &dstDesc;
    }
    RpptROI* r() {
        return roi.data();
    }
    rppHandle_t h() {
        return handle.get();
    }
};

using OpCall = std::function<RppStatus(Ctx&)>;

struct OpCase {
    const char* name;
    OpCall call;
    bool hostOnly = false;
};

constexpr auto B = RpptImageBorderType::REPLICATE;
constexpr auto BL = RpptInterpolationType::BILINEAR;

// clang-format off
const std::vector<OpCase>& op_cases() {
    static const std::vector<OpCase> cases = {
        // arithmetic / bitwise
        {"magnitude", [](Ctx& c) { return rppt_magnitude(c.s(), c.s2(), c.sd(), c.d(), c.dd(), c.r(), XYWH, c.h(), c.backend); }},
        {"bitwise_and", [](Ctx& c) { return rppt_bitwise_and(c.s(), c.s2(), c.sd(), c.d(), c.dd(), c.r(), XYWH, c.h(), c.backend); }},
        {"bitwise_or", [](Ctx& c) { return rppt_bitwise_or(c.s(), c.s2(), c.sd(), c.d(), c.dd(), c.r(), XYWH, c.h(), c.backend); }},
        {"bitwise_xor", [](Ctx& c) { return rppt_bitwise_xor(c.s(), c.s2(), c.sd(), c.d(), c.dd(), c.r(), XYWH, c.h(), c.backend); }},
        {"bitwise_not", [](Ctx& c) { return rppt_bitwise_not(c.s(), c.sd(), c.d(), c.dd(), c.r(), XYWH, c.h(), c.backend); }},
        // color
        {"brightness", [](Ctx& c) { return rppt_brightness(c.s(), c.sd(), c.d(), c.dd(), c.p<Rpp32f>(), c.p<Rpp32f>(), c.r(), XYWH, c.h(), c.backend); }},
        {"gamma_correction", [](Ctx& c) { return rppt_gamma_correction(c.s(), c.sd(), c.d(), c.dd(), c.p<Rpp32f>(), c.r(), XYWH, c.h(), c.backend); }},
        {"blend", [](Ctx& c) { return rppt_blend(c.s(), c.s2(), c.sd(), c.d(), c.dd(), c.p<Rpp32f>(), c.r(), XYWH, c.h(), c.backend); }},
        {"hue", [](Ctx& c) { return rppt_hue(c.s(), c.sd(), c.d(), c.dd(), c.p<Rpp32f>(), c.r(), XYWH, c.h(), c.backend); }},
        {"saturation", [](Ctx& c) { return rppt_saturation(c.s(), c.sd(), c.d(), c.dd(), c.p<Rpp32f>(), c.r(), XYWH, c.h(), c.backend); }},
        {"color_twist", [](Ctx& c) { return rppt_color_twist(c.s(), c.sd(), c.d(), c.dd(), c.p<Rpp32f>(), c.p<Rpp32f>(), c.p<Rpp32f>(), c.p<Rpp32f>(), c.r(), XYWH, c.h(), c.backend); }},
        {"color_jitter", [](Ctx& c) { return rppt_color_jitter(c.s(), c.sd(), c.d(), c.dd(), c.p<Rpp32f>(), c.p<Rpp32f>(), c.p<Rpp32f>(), c.p<Rpp32f>(), c.r(), XYWH, c.h(), c.backend); }},
        {"color_cast", [](Ctx& c) { return rppt_color_cast(c.s(), c.sd(), c.d(), c.dd(), c.p<RpptRGB>(), c.p<Rpp32f>(), c.r(), XYWH, c.h(), c.backend); }},
        {"exposure", [](Ctx& c) { return rppt_exposure(c.s(), c.sd(), c.d(), c.dd(), c.p<Rpp32f>(), c.r(), XYWH, c.h(), c.backend); }},
        {"contrast", [](Ctx& c) { return rppt_contrast(c.s(), c.sd(), c.d(), c.dd(), c.p<Rpp32f>(), c.p<Rpp32f>(), c.r(), XYWH, c.h(), c.backend); }},
        {"lut", [](Ctx& c) { return rppt_lut(c.s(), c.sd(), c.d(), c.dd(), c.dev(), c.r(), XYWH, c.h(), c.backend); }},
        {"color_temperature", [](Ctx& c) { return rppt_color_temperature(c.s(), c.sd(), c.d(), c.dd(), c.p<Rpp32s>(), c.r(), XYWH, c.h(), c.backend); }},
        {"histogram_equalize", [](Ctx& c) { return rppt_histogram_equalize(c.s(), c.sd(), c.d(), c.dd(), c.r(), XYWH, c.h(), c.backend); }},
        {"tone_map", [](Ctx& c) { return rppt_tone_map(c.s(), c.sd(), c.d(), c.dd(), c.p<Rpp32f>(), c.r(), XYWH, c.h(), c.backend); }},
        // data exchange
        {"copy", [](Ctx& c) { return rppt_copy(c.s(), c.sd(), c.d(), c.dd(), c.h(), c.backend); }},
        {"channel_permute", [](Ctx& c) { return rppt_channel_permute(c.s(), c.sd(), c.d(), c.dd(), c.p<Rpp32u>(), c.h(), c.backend); }},
        // effects
        {"gridmask", [](Ctx& c) { return rppt_gridmask(c.s(), c.sd(), c.d(), c.dd(), 4, 0.5f, 0.0f, RpptUintVector2D{0, 0}, c.r(), XYWH, c.h(), c.backend); }},
        {"spatter", [](Ctx& c) { return rppt_spatter(c.s(), c.sd(), c.d(), c.dd(), RpptRGB{0, 0, 0}, c.r(), XYWH, c.h(), c.backend); }},
        {"salt_and_pepper_noise", [](Ctx& c) { return rppt_salt_and_pepper_noise(c.s(), c.sd(), c.d(), c.dd(), c.p<Rpp32f>(), c.p<Rpp32f>(), c.p<Rpp32f>(), c.p<Rpp32f>(), 0, c.r(), XYWH, c.h(), c.backend); }},
        {"shot_noise", [](Ctx& c) { return rppt_shot_noise(c.s(), c.sd(), c.d(), c.dd(), c.p<Rpp32f>(), 0, c.r(), XYWH, c.h(), c.backend); }},
        {"gaussian_noise", [](Ctx& c) { return rppt_gaussian_noise(c.s(), c.sd(), c.d(), c.dd(), c.p<Rpp32f>(), c.p<Rpp32f>(), 0, c.r(), XYWH, c.h(), c.backend); }},
        {"non_linear_blend", [](Ctx& c) { return rppt_non_linear_blend(c.s(), c.s2(), c.sd(), c.d(), c.dd(), c.p<Rpp32f>(), c.r(), XYWH, c.h(), c.backend); }},
        {"water", [](Ctx& c) { return rppt_water(c.s(), c.sd(), c.d(), c.dd(), c.p<Rpp32f>(), c.p<Rpp32f>(), c.p<Rpp32f>(), c.p<Rpp32f>(), c.p<Rpp32f>(), c.p<Rpp32f>(), c.r(), XYWH, c.h(), c.backend); }},
        {"ricap", [](Ctx& c) { return rppt_ricap(c.s(), c.sd(), c.d(), c.dd(), c.p<Rpp32u>(), c.r(), XYWH, c.h(), c.backend); }},
        {"vignette", [](Ctx& c) { return rppt_vignette(c.s(), c.sd(), c.d(), c.dd(), c.p<Rpp32f>(), c.r(), XYWH, c.h(), c.backend); }},
        {"jitter", [](Ctx& c) { return rppt_jitter(c.s(), c.sd(), c.d(), c.dd(), c.p<Rpp32u>(), 0, c.r(), XYWH, c.h(), c.backend); }},
        {"erase", [](Ctx& c) { return rppt_erase(c.s(), c.sd(), c.d(), c.dd(), c.p<RpptRoiLtrb>(), c.dev(), c.p<Rpp32u>(), c.r(), XYWH, c.h(), c.backend); }},
        {"glitch", [](Ctx& c) { return rppt_glitch(c.s(), c.sd(), c.d(), c.dd(), c.p<RpptChannelOffsets>(), c.r(), XYWH, c.h(), c.backend); }},
        {"rain", [](Ctx& c) { return rppt_rain(c.s(), c.sd(), c.d(), c.dd(), 1.0f, 1, 4, 0.0f, c.p<Rpp32f>(), c.r(), XYWH, c.h(), c.backend); }},
        {"pixelate", [](Ctx& c) { return rppt_pixelate(c.s(), c.sd(), c.d(), c.dd(), c.dev(), 50.0f, c.r(), XYWH, c.h(), c.backend); }},
        {"fog", [](Ctx& c) { return rppt_fog(c.s(), c.sd(), c.d(), c.dd(), c.p<Rpp32f>(), c.p<Rpp32f>(), c.r(), XYWH, c.h(), c.backend); }},
        {"posterize", [](Ctx& c) { return rppt_posterize(c.s(), c.sd(), c.d(), c.dd(), c.p<Rpp8u>(), c.r(), XYWH, c.h(), c.backend); }},
        {"solarize", [](Ctx& c) { return rppt_solarize(c.s(), c.sd(), c.d(), c.dd(), c.p<Rpp32f>(), c.r(), XYWH, c.h(), c.backend); }},
        {"snow", [](Ctx& c) { return rppt_snow(c.s(), c.sd(), c.d(), c.dd(), c.p<Rpp32f>(), c.p<Rpp32f>(), c.p<Rpp32s>(), c.r(), XYWH, c.h(), c.backend); }},
        {"channel_dropout", [](Ctx& c) { return rppt_channel_dropout(c.s(), c.sd(), c.d(), c.dd(), c.p<Rpp8u>(), c.r(), XYWH, c.h(), c.backend); }},
        {"cutout_dropout", [](Ctx& c) { return rppt_cutout_dropout(c.s(), c.sd(), c.d(), c.dd(), c.p<RpptRoiLtrb>(), c.dev(), c.p<Rpp32u>(), c.r(), XYWH, c.h(), c.backend); }},
        {"grid_dropout", [](Ctx& c) { return rppt_grid_dropout(c.s(), c.sd(), c.d(), c.dd(), c.p<RpptRoiLtrb>(), 1, 2, 2, c.r(), XYWH, c.h(), c.backend); }},
        {"random_erase", [](Ctx& c) { return rppt_random_erase(c.s(), c.sd(), c.d(), c.dd(), c.p<RpptRoiLtrb>(), c.dev(), c.r(), XYWH, c.h(), c.backend); }},
        {"coarse_dropout", [](Ctx& c) { return rppt_coarse_dropout(c.s(), c.sd(), c.d(), c.dd(), c.p<RpptRoiLtrb>(), c.p<Rpp32u>(), 1, c.r(), XYWH, c.h(), c.backend); }},
        // filter
        {"box_filter", [](Ctx& c) { return rppt_box_filter(c.s(), c.sd(), c.d(), c.dd(), 3, B, c.r(), XYWH, c.h(), c.backend); }},
        {"median_filter", [](Ctx& c) { return rppt_median_filter(c.s(), c.sd(), c.d(), c.dd(), 3, B, c.r(), XYWH, c.h(), c.backend); }},
        {"gaussian_filter", [](Ctx& c) { return rppt_gaussian_filter(c.s(), c.sd(), c.d(), c.dd(), c.p<Rpp32f>(), 3, B, c.r(), XYWH, c.h(), c.backend); }},
        {"emboss", [](Ctx& c) { return rppt_emboss(c.s(), c.sd(), c.d(), c.dd(), c.p<Rpp32f>(), 3, B, c.r(), XYWH, c.h(), c.backend); }},
        {"bilateral_filter", [](Ctx& c) { return rppt_bilateral_filter(c.s(), c.sd(), c.d(), c.dd(), c.p<Rpp32f>(), c.p<Rpp32f>(), 3, B, c.r(), XYWH, c.h(), c.backend); }},
        // geometric
        {"crop", [](Ctx& c) { return rppt_crop(c.s(), c.sd(), c.d(), c.dd(), c.r(), XYWH, c.h(), c.backend); }},
        {"crop_mirror_normalize", [](Ctx& c) { return rppt_crop_mirror_normalize(c.s(), c.sd(), c.d(), c.dd(), c.p<Rpp32f>(), c.p<Rpp32f>(), c.p<Rpp32u>(), c.r(), XYWH, c.h(), c.backend); }},
        {"warp_affine", [](Ctx& c) { return rppt_warp_affine(c.s(), c.sd(), c.d(), c.dd(), c.p<Rpp32f>(), BL, c.r(), XYWH, c.h(), c.backend); }},
        {"flip", [](Ctx& c) { return rppt_flip(c.s(), c.sd(), c.d(), c.dd(), c.p<Rpp32u>(), c.p<Rpp32u>(), c.r(), XYWH, c.h(), c.backend); }},
        {"resize", [](Ctx& c) { return rppt_resize(c.s(), c.sd(), c.d(), c.dd(), c.patch.data(), BL, c.r(), XYWH, c.h(), c.backend); }},
        {"resize_mirror_normalize", [](Ctx& c) { return rppt_resize_mirror_normalize(c.s(), c.sd(), c.d(), c.dd(), c.patch.data(), BL, c.p<Rpp32f>(), c.p<Rpp32f>(), c.p<Rpp32u>(), c.r(), XYWH, c.h(), c.backend); }},
        {"resize_crop_mirror", [](Ctx& c) { return rppt_resize_crop_mirror(c.s(), c.sd(), c.d(), c.dd(), c.patch.data(), BL, c.p<Rpp32u>(), c.r(), XYWH, c.h(), c.backend); }},
        {"rotate", [](Ctx& c) { return rppt_rotate(c.s(), c.sd(), c.d(), c.dd(), c.p<Rpp32f>(), BL, c.r(), XYWH, c.h(), c.backend); }},
        {"phase", [](Ctx& c) { return rppt_phase(c.s(), c.s2(), c.sd(), c.d(), c.dd(), c.r(), XYWH, c.h(), c.backend); }},
        {"crop_and_patch", [](Ctx& c) { return rppt_crop_and_patch(c.s(), c.s2(), c.sd(), c.d(), c.dd(), c.r(), c.r(), c.r(), XYWH, c.h(), c.backend); }},
        {"remap", [](Ctx& c) { return rppt_remap(c.s(), c.sd(), c.d(), c.dd(), c.dev<Rpp32f>(), c.dev<Rpp32f>(), &c.tableDesc, BL, c.r(), XYWH, c.h(), c.backend); }},
        {"lens_correction", [](Ctx& c) { return rppt_lens_correction(c.s(), c.sd(), c.d(), c.dd(), c.dev<Rpp32f>(), c.dev<Rpp32f>(), &c.tableDesc, c.p<Rpp32f>(), c.p<Rpp32f>(), c.r(), XYWH, c.h(), c.backend); }},
        {"warp_perspective", [](Ctx& c) { return rppt_warp_perspective(c.s(), c.sd(), c.d(), c.dd(), c.p<Rpp32f>(), BL, c.r(), XYWH, c.h(), c.backend); }},
        {"jpeg_compression_distortion", [](Ctx& c) { return rppt_jpeg_compression_distortion(c.s(), c.sd(), c.d(), c.dd(), c.p<Rpp32s>(), c.r(), XYWH, c.h(), c.backend); }},
        {"fisheye", [](Ctx& c) { return rppt_fisheye(c.s(), c.sd(), c.d(), c.dd(), c.r(), XYWH, c.h(), c.backend); }},
        // morphological
        {"erode", [](Ctx& c) { return rppt_erode(c.s(), c.sd(), c.d(), c.dd(), 3, c.r(), XYWH, c.h(), c.backend); }},
        {"dilate", [](Ctx& c) { return rppt_dilate(c.s(), c.sd(), c.d(), c.dd(), 3, c.r(), XYWH, c.h(), c.backend); }},
        {"erode_host", [](Ctx& c) { return rppt_erode_host(c.s(), c.sd(), c.d(), c.dd(), 3, c.r(), XYWH, c.h()); }, true},
        {"dilate_host", [](Ctx& c) { return rppt_dilate_host(c.s(), c.sd(), c.d(), c.dd(), 3, c.r(), XYWH, c.h()); }, true},
        // statistical
        {"threshold", [](Ctx& c) { return rppt_threshold(c.s(), c.sd(), c.d(), c.dd(), c.p<Rpp32f>(), c.p<Rpp32f>(), c.r(), XYWH, c.h(), c.backend); }},
    };
    return cases;
}
// clang-format on

struct ChannelMismatchParam {
    std::size_t opIndex;
    RppBackend backend;
    Layout src, dst;
};

std::vector<ChannelMismatchParam> make_params() {
    std::vector<ChannelMismatchParam> out;
    for (std::size_t i = 0; i < op_cases().size(); ++i)
        for (RppBackend backend : available_backends()) {
            if (op_cases()[i].hostOnly && backend != RPP_HOST_BACKEND) continue;
            out.push_back({i, backend, Layout::PLN3, Layout::PLN1});
            out.push_back({i, backend, Layout::PLN1, Layout::PLN3});
        }
    return out;
}

std::string param_name(const ::testing::TestParamInfo<ChannelMismatchParam>& info) {
    const auto& p = info.param;
    return std::string(op_cases()[p.opIndex].name) + "_" + backend_name(p.backend) + "_" +
           layout_name(p.src) + "to" + layout_name(p.dst);
}

inline void PrintTo(const ChannelMismatchParam&, std::ostream*) {}

}  // namespace

// Full name:
// Image_Validation/ChannelMismatchTest.Rejected/<op>_<Backend>_PLN3toPLN1 (and PLN1toPLN3)
class ChannelMismatchTest : public ::testing::TestWithParam<ChannelMismatchParam> {};

TEST_P(ChannelMismatchTest, Rejected) {
    const auto& p = GetParam();
    const OpCase& op = op_cases()[p.opIndex];

    const auto desc = [](Layout l) {
        const Rpp32u c = static_cast<Rpp32u>(channels_of(l));
        return make_descriptor({kSize.n, c, kSize.h, kSize.w}, DType::U8, l);
    };
    const RpptDesc srcDesc = desc(p.src);
    const RpptDesc dstDesc = desc(p.dst);
    // Room for three channels on both sides; see the file comment.
    const RpptDesc wide = make_descriptor({kSize.n, 3, kSize.h, kSize.w}, DType::U8, Layout::PLN3);
    const std::size_t bytes = byte_size(wide, DType::U8);

    Ctx ctx(p.backend, srcDesc, dstDesc, bytes);
    std::memset(ctx.hostScratch.data(), 0, kScratchBytes);
    const std::vector<RpptROI> roiVec = make_roi(srcDesc, Roi::Full);
    for (Rpp32u i = 0; i < kSize.n; ++i) {
        ctx.roi[i] = roiVec[i];
        ctx.patch[i] = RpptImagePatch{kSize.w, kSize.h};
    }

    std::vector<Rpp8u> input(bytes);
    fill_input<Rpp8u>(input.data(), bytes, DType::U8);
    ctx.src.write(input.data(), bytes);
    ctx.src2.write(input.data(), bytes);
    const std::vector<Rpp8u> sentinel(bytes, kSentinel);
    ctx.dst.write(sentinel.data(), bytes);
    const std::vector<Rpp8u> zeros(element_count(ctx.tableDesc) * sizeof(Rpp32f) + kScratchBytes);
    ctx.devScratch.write(zeros.data(), zeros.size());

    const RppStatus status = op.call(ctx);
    ctx.handle.sync();

    EXPECT_EQ(status, RPP_ERROR_INVALID_DST_CHANNELS)
        << op.name << " accepted src c=" << srcDesc.c << " with dst c=" << dstDesc.c;

    // Rejection must come before any work: the destination is byte-for-byte untouched.
    std::vector<Rpp8u> after(bytes);
    ctx.dst.read(after.data(), bytes);
    EXPECT_EQ(after, sentinel) << op.name << " wrote to the destination despite the mismatch";
}

INSTANTIATE_TEST_SUITE_P(Image_Validation, ChannelMismatchTest, ::testing::ValuesIn(make_params()),
                         param_name);
