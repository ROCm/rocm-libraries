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
#include "reference/water_ref.hpp"

using namespace rpptest;

namespace {

// Nearest-neighbour sampling copies a source texel verbatim and the op has no arithmetic, so every
// dtype is bit-exact. A diff is a disagreement about the displacement or the border rule, which no
// tolerance could absorb anyway.
constexpr double kTol = 0.0;

// Flat: zero amplitudes, so the map is the exact identity and the output must be a verbatim copy of
// the ROI. Separates a plumbing or ROI-placement failure from a wave-math disagreement.
// Wave: the parameters the legacy harness uses. Amplitudes of 2 and 5 pixels push the border
// columns/rows off the ROI, so the black-fill branch is exercised too.
enum class WaterKind { Flat, Wave };

struct WaterParams {
    WaterKind kind;
    std::string name() const { return kind == WaterKind::Flat ? "Flat" : "Wave"; }
    float amplitudeX() const { return kind == WaterKind::Wave ? 2.0f : 0.0f; }
    float amplitudeY() const { return kind == WaterKind::Wave ? 5.0f : 0.0f; }
};

constexpr float kFreqX = 5.8f, kFreqY = 1.2f, kPhaseX = 10.0f, kPhaseY = 15.0f;

template <typename T>
void run_water(const TestConfig& cfg, const WaterParams& op) {
    RpptDesc srcDesc = make_src_descriptor(cfg);
    RpptDesc dstDesc = make_dst_descriptor(cfg);
    const std::size_t count = element_count(srcDesc);
    const std::size_t bytes = byte_size(srcDesc, cfg.dtype);

    PinnedArray<RpptROI> roi(cfg.backend, cfg.size.n);
    const std::vector<RpptROI> roiVec = make_roi(srcDesc, cfg.roi);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) roi[i] = roiVec[i];

    PinnedArray<Rpp32f> amplX(cfg.backend, cfg.size.n), amplY(cfg.backend, cfg.size.n);
    PinnedArray<Rpp32f> freqX(cfg.backend, cfg.size.n), freqY(cfg.backend, cfg.size.n);
    PinnedArray<Rpp32f> phaseX(cfg.backend, cfg.size.n), phaseY(cfg.backend, cfg.size.n);
    for (Rpp32u n = 0; n < cfg.size.n; ++n) {
        amplX[n] = op.amplitudeX();
        amplY[n] = op.amplitudeY();
        freqX[n] = kFreqX;
        freqY[n] = kFreqY;
        phaseX[n] = kPhaseX;
        phaseY[n] = kPhaseY;
    }

    // The op writes the ROI-sized region at the destination origin, so golden and device buffer
    // start from the same distinct pattern and only that region is compared.
    std::vector<T> input(count), dstInit(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    fill_input<T>(dstInit.data(), count, cfg.dtype, /*salt=*/1);
    golden = dstInit;
    water_reference<T>(input.data(), srcDesc, golden.data(), dstDesc, cfg.dtype, roi.data(), XYWH,
                       amplX.data(), amplY.data(), freqX.data(), freqY.data(), phaseX.data(),
                       phaseY.data());

    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(dstInit.data(), bytes);

    RppHandle handle(cfg.backend, cfg.size.n);
    ASSERT_EQ(rppt_water(src.ptr(), &srcDesc, dst.ptr(), &dstDesc, amplX.data(), amplY.data(),
                         freqX.data(), freqY.data(), phaseX.data(), phaseY.data(), roi.data(), XYWH,
                         handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();
    dst.read(actual.data(), bytes);

    // Compared against the caller's own ROI copy, not the tensor handed to the op, in case the
    // backend rewrites it from XYWH to LTRB in place.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), dstDesc, roiVec.data(), XYWH, kTol));
}

}  // namespace

// Full name: Image_Effects/WaterTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<Kind>
class WaterTest : public SkipListTest<WithParams<WaterParams>> {};

TEST_P(WaterTest, Correctness) {
    const auto& p = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32>(p.cfg.dtype, [&](auto tag) {
        run_water<Element<decltype(tag)>>(p.cfg, p.op);
    });
}

// I8 is off the grid for now, pending the suite-wide decision on whether the image ops need it.
INSTANTIATE_TEST_SUITE_P(
    Image_Effects, WaterTest,
    ::testing::ValuesIn(with_params<WaterParams>(
        concat_configs({
            make_configs({DType::U8, DType::F16, DType::F32}, presets::kLayoutsFullConv,
                         {Roi::Full, Roi::Partial},
                         {presets::kTailWidthSize}),
            make_configs({DType::U8, DType::F16, DType::F32}, presets::kLayoutsFull,
                         {Roi::Full, Roi::Partial},
                         {presets::kDefaultSize, presets::kSubVectorSize}),
        }),
        {WaterParams{WaterKind::Flat}, WaterParams{WaterKind::Wave}})),
    op_config_name<WaterParams>);
