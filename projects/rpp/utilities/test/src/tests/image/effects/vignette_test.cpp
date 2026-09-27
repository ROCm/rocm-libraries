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
#include "reference/vignette_ref.hpp"

using namespace rpptest;

namespace {

// The header warns that HOST evaluates the Gaussian with fastexpavx() rather than exp() and that up
// to 5% pixel mismatch is expected. That allowance does not materialise: against the exact exp() in
// the golden, the measured worst case over this grid is 0 on U8, 1 ULP on F32 (1.2e-7) and 1 ULP on
// F16 (4.9e-4), on both backends. These tolerances are sized from that measurement, not from the
// header's blanket 5% -- which would be 13 grey levels and could hide a real defect in the falloff.
// U8 is given 1 rather than 0 only to absorb a rounding tie, not observed error.
constexpr Tolerance kVignetteTolerance = tolerance(1.0, 1e-6, 1e-3);

// 6 is the intensity the legacy harness uses: on a 48-wide ROI it puts sigma at 8 px, so the
// corners are driven essentially to black while the centre is untouched. 1 is the opposite end --
// sigma is the full extent, so the corners keep ~78% of their value and the whole region stays
// close to the source, which separates a wrong falloff shape from a wrong overall scale.
struct VignetteParams {
    float intensity;
    std::string name() const {
        return "i" + num_token(intensity);
    }
};

template <typename T>
void run_vignette(const TestConfig& cfg, const VignetteParams& op) {
    RpptDesc srcDesc = make_src_descriptor(cfg);
    RpptDesc dstDesc = make_dst_descriptor(cfg);
    const std::size_t count = element_count(srcDesc);
    const std::size_t bytes = byte_size(srcDesc, cfg.dtype);

    PinnedArray<RpptROI> roi(cfg.backend, cfg.size.n);
    const std::vector<RpptROI> roiVec = make_roi(srcDesc, cfg.roi);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) roi[i] = roiVec[i];

    PinnedArray<Rpp32f> intensity(cfg.backend, cfg.size.n);
    for (Rpp32u n = 0; n < cfg.size.n; ++n) intensity[n] = op.intensity;

    // The op writes the ROI-sized region at the destination origin, so golden and device buffer
    // start from the same distinct pattern and only that region is compared.
    std::vector<T> input(count), dstInit(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    fill_input<T>(dstInit.data(), count, cfg.dtype, /*salt=*/1);
    golden = dstInit;
    vignette_reference<T>(input.data(), srcDesc, golden.data(), dstDesc, cfg.dtype, roi.data(),
                          XYWH, intensity.data());

    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(dstInit.data(), bytes);

    RppHandle handle(cfg.backend, cfg.size.n);
    ASSERT_EQ(rppt_vignette(src.ptr(), &srcDesc, dst.ptr(), &dstDesc, intensity.data(), roi.data(),
                            XYWH, handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();
    dst.read(actual.data(), bytes);

    // Compared against the caller's own ROI copy, not the tensor handed to the op, in case the
    // backend rewrites it from XYWH to LTRB in place.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), dstDesc, roiVec.data(), XYWH,
                               kVignetteTolerance(cfg.dtype)));
}

}  // namespace

// Full name:
// Image_Effects/VignetteTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<Intensity>
class VignetteTest : public SkipListTest<WithParams<VignetteParams>> {};

TEST_P(VignetteTest, Correctness) {
    const auto& p = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32>(
        p.cfg.dtype, [&](auto tag) { run_vignette<Element<decltype(tag)>>(p.cfg, p.op); });
}

// I8 is off the grid for now, pending the suite-wide decision on whether the image ops need it.
// Same-layout cases plus both directions of the fused output-layout conversion.
INSTANTIATE_TEST_SUITE_P(Image_Effects, VignetteTest,
                         ::testing::ValuesIn(with_params<VignetteParams>(
                             concat_configs({
                                 make_configs({DType::U8, DType::F16, DType::F32},
                                              presets::kLayoutsFullConv, {Roi::Full, Roi::Partial},
                                              {presets::kTailWidthSize}),
                                 make_configs({DType::U8, DType::F16, DType::F32},
                                              presets::kLayoutsFull, {Roi::Full, Roi::Partial},
                                              {presets::kDefaultSize, presets::kSubVectorSize}),
                             }),
                             {VignetteParams{6.0f}, VignetteParams{1.0f}})),
                         op_config_name<VignetteParams>);
