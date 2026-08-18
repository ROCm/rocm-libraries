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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "framework/backend_memory.hpp"
#include "framework/config_param.hpp"
#include "framework/skip_list.hpp"
#include "framework/generic_tensor_setup.hpp"  // nd_slack_poison
#include "framework/tensor_setup.hpp"
#include "reference/yuv_to_rgb_ref.hpp"

using namespace rpptest;

namespace {

// The three NV12 -> RGB24 ops share one signature; only the vertical chroma upsampler differs.
using YuvToRgbFn = RppStatus (*)(RppPtr_t, RppPtr_t, RpptDescPtr, RppPtr_t, RpptDescPtr, Rpp32u,
                                 Rpp32u, Rpp32u, Rpp32u, Rpp32u, RpptColorStandard, RpptColorRange,
                                 rppHandle_t, RppBackend);

struct YuvParams {
    RpptColorStandard standard;
    RpptColorRange range;
    const char* token;
    std::string name() const { return token; }
};

// NV12 requires even width and height. n = 1: the ops take a single image, not a batch.
constexpr Size kSize{1, 36, 48};

// The matrix axis: both ranges on the two mainstream standards, plus one wide-gamut and one
// legacy matrix so a wr/wb table error cannot hide behind the two common entries.
const std::vector<YuvParams> kYuvParams = {
    {RpptColorStandard_BT709, RpptColorRange_STUDIO, "BT709_STUDIO"},
    {RpptColorStandard_BT709, RpptColorRange_FULL, "BT709_FULL"},
    {RpptColorStandard_BT601, RpptColorRange_STUDIO, "BT601_STUDIO"},
    {RpptColorStandard_BT601, RpptColorRange_FULL, "BT601_FULL"},
    {RpptColorStandard_BT2020_NCL, RpptColorRange_FULL, "BT2020NCL_FULL"},
    {RpptColorStandard_SMPTE240M, RpptColorRange_STUDIO, "SMPTE240M_STUDIO"},
};

// These ops document "RPP_HIP_BACKEND only" and return RPP_ERROR_INCOMPATIBLE_BACKEND otherwise, so
// the grid is built by hand instead of with make_configs() (which yields every available backend).
// Without a HIP build the instantiation is simply empty. The Layout/Roi slots of the label are
// vestigial here -- there is no layout or ROI axis -- but PKD3/FullRoi do describe the fixed
// destination, and keeping the slots keeps the label grammar uniform.
std::vector<WithParams<YuvParams>> yuv_configs() {
    std::vector<WithParams<YuvParams>> configs;
    for (RppBackend backend : available_backends()) {
        if (backend != RPP_HIP_BACKEND) continue;
        for (const YuvParams& op : kYuvParams)
            configs.push_back({TestConfig{backend, DType::U8, Layout::PKD3, Roi::Full, kSize}, op});
    }
    return configs;
}

// Lays the deterministic pattern out row by row at `pitch`; the slack keeps whatever the plane was
// initialized with (poison), so a kernel that reads past the logical row width is caught.
void fill_pitched_plane(std::vector<Rpp8u>& plane, Rpp32u pitch, Rpp32u rowBytes, Rpp32u rows,
                        unsigned salt) {
    std::vector<Rpp8u> pattern(static_cast<std::size_t>(rowBytes) * rows);
    fill_input<Rpp8u>(pattern.data(), pattern.size(), DType::U8, salt);
    for (Rpp32u r = 0; r < rows; ++r) {
        const std::size_t from = static_cast<std::size_t>(r) * rowBytes;
        std::copy(pattern.begin() + from, pattern.begin() + from + rowBytes,
                  plane.begin() + static_cast<std::size_t>(r) * pitch);
    }
}

// compare_roi() addresses elements through an RpptDesc and cannot describe a pitched packed-RGB
// buffer, so this walks height x width x 3 through dstPitch instead. Only the logical width*3 bytes
// of each row are compared; the pitch slack is not data. Message style mirrors compare_roi().
::testing::AssertionResult compare_rgb_pitched(const Rpp8u* actual, const Rpp8u* reference,
                                               Rpp32u width, Rpp32u height, Rpp32u dstPitch,
                                               double tolerance) {
    for (Rpp32u y = 0; y < height; ++y)
        for (Rpp32u x = 0; x < width; ++x)
            for (Rpp32u c = 0; c < 3; ++c) {
                const std::size_t i = static_cast<std::size_t>(y) * dstPitch + 3 * x + c;
                const double a = static_cast<double>(actual[i]);
                const double r = static_cast<double>(reference[i]);
                const double diff = std::fabs(a - r);
                if (diff > tolerance)
                    return ::testing::AssertionFailure()
                           << "mismatch at row=" << y << " col=" << x << " channel=" << c
                           << ": actual=" << a << " reference=" << r << " diff=" << diff
                           << " tolerance=" << tolerance;
            }
    return ::testing::AssertionSuccess();
}

// The conversion is a float matrix multiply rounded to U8, so one LSB.
constexpr double kTolerance = 1.0;

void run_yuv_to_rgb(const TestConfig& cfg, const YuvParams& op, YuvToRgbFn fn,
                    YuvChromaUpsample upsample) {
    const Rpp32u width = cfg.size.w, height = cfg.size.h;
    const Rpp32u chromaHeight = height / 2;
    const Rpp32u uvRowBytes = width;  // width/2 chroma samples, two interleaved bytes each

    // Pitches deliberately exceed the tight row width so a kernel that ignores them is caught.
    const Rpp32u yPitch = width + 16;
    const Rpp32u uvPitch = uvRowBytes + 16;
    const Rpp32u dstPitch = 3 * width + 16;

    const std::size_t yBytes = static_cast<std::size_t>(yPitch) * height;
    const std::size_t uvBytes = static_cast<std::size_t>(uvPitch) * chromaHeight;
    const std::size_t dstBytes = static_cast<std::size_t>(dstPitch) * height;

    // The descriptors are vestigial (only dataType is inspected); addressing is by the explicit
    // byte pitches. Source is a single-plane 8-bit luma frame, destination packed RGB24.
    const Rpp8u poison = nd_slack_poison<Rpp8u>(DType::U8);
    RpptDesc srcDesc = make_descriptor({1, 1, height, width}, DType::U8, Layout::PLN1);
    RpptDesc dstDesc = make_descriptor({1, 3, height, width}, DType::U8, Layout::PKD3);

    // (1) Host golden model. UV uses a different salt so chroma is not a copy of luma. dstInit is
    // all poison, so a kernel that writes nothing (or skips rows) fails rather than coincidentally
    // matching.
    std::vector<Rpp8u> hostY(yBytes, poison), hostUV(uvBytes, poison);
    fill_pitched_plane(hostY, yPitch, width, height, /*salt=*/0);
    fill_pitched_plane(hostUV, uvPitch, uvRowBytes, chromaHeight, /*salt=*/3);

    std::vector<Rpp8u> dstInit(dstBytes, poison), actual(dstBytes, 0);
    std::vector<Rpp8u> golden = dstInit;
    yuv_to_rgb_reference(hostY.data(), hostUV.data(), golden.data(), yPitch, uvPitch, dstPitch,
                         width, height, op.standard, op.range, upsample);

    // (2) Run RPP (HIP only for these ops).
    DeviceTensor srcY(cfg.backend, yBytes), srcUV(cfg.backend, uvBytes),
        dst(cfg.backend, dstBytes);
    srcY.write(hostY.data(), yBytes);
    srcUV.write(hostUV.data(), uvBytes);
    dst.write(dstInit.data(), dstBytes);

    RppHandle handle(cfg.backend, 1);
    ASSERT_EQ(fn(srcY.ptr(), srcUV.ptr(), &srcDesc, dst.ptr(), &dstDesc, yPitch, uvPitch, dstPitch,
                 width, height, op.standard, op.range, handle.get(), cfg.backend),
              RPP_SUCCESS);

    // (3) Retrieve the result on the host.
    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), dstBytes);

    // (4) Compare the logical RGB bytes only.
    EXPECT_TRUE(compare_rgb_pitched(actual.data(), golden.data(), width, height, dstPitch,
                                    kTolerance));
}

}  // namespace

// Full name: Image_DataExchange/<Suite>.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<Standard>_<Range>
class YuvToRgbTest : public SkipListTest<WithParams<YuvParams>> {};
class YuvToRgbCubicVTest : public SkipListTest<WithParams<YuvParams>> {};
class YuvToRgbLinearVTest : public SkipListTest<WithParams<YuvParams>> {};

TEST_P(YuvToRgbTest, Correctness) {
    const auto& p = GetParam();
    run_yuv_to_rgb(p.cfg, p.op, &rppt_yuv_to_rgb, YuvChromaUpsample::Nearest);
}

TEST_P(YuvToRgbCubicVTest, Correctness) {
    const auto& p = GetParam();
    run_yuv_to_rgb(p.cfg, p.op, &rppt_yuv_to_rgb_cubic_v, YuvChromaUpsample::CubicV);
}

TEST_P(YuvToRgbLinearVTest, Correctness) {
    const auto& p = GetParam();
    run_yuv_to_rgb(p.cfg, p.op, &rppt_yuv_to_rgb_linear_v, YuvChromaUpsample::LinearV);
}

INSTANTIATE_TEST_SUITE_P(Image_DataExchange, YuvToRgbTest, ::testing::ValuesIn(yuv_configs()),
                         op_config_name<YuvParams>);
INSTANTIATE_TEST_SUITE_P(Image_DataExchange, YuvToRgbCubicVTest, ::testing::ValuesIn(yuv_configs()),
                         op_config_name<YuvParams>);
INSTANTIATE_TEST_SUITE_P(Image_DataExchange, YuvToRgbLinearVTest,
                         ::testing::ValuesIn(yuv_configs()), op_config_name<YuvParams>);
