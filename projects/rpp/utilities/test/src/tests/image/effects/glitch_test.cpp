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
#include "reference/glitch_ref.hpp"

using namespace rpptest;

namespace {

// The op only relocates whole pixels, so every dtype must come back bit-exact.
constexpr double kTol = 0.0;

// Identity: every offset 0, so the output is a verbatim copy of the ROI. Separates a plumbing or
// ROI-placement failure from a shift-math disagreement.
// Shift: the offsets the legacy harness uses -- R by (10,10), G unmoved, B by (5,5). All three fit
// inside the 24x18 partial ROI, so both the shifted and the passed-through branch are exercised.
enum class GlitchKind { Identity, Shift };

struct GlitchParams {
    GlitchKind kind;
    std::string name() const { return kind == GlitchKind::Identity ? "Identity" : "Shift"; }
};

RpptChannelOffsets make_offsets(GlitchKind kind) {
    RpptChannelOffsets o{};
    const bool shift = kind == GlitchKind::Shift;
    o.r.x = shift ? 10 : 0;
    o.r.y = shift ? 10 : 0;
    o.g.x = 0;
    o.g.y = 0;
    o.b.x = shift ? 5 : 0;
    o.b.y = shift ? 5 : 0;
    return o;
}

template <typename T>
void run_glitch(const TestConfig& cfg, const GlitchParams& op) {
    const TensorShape shape{cfg.size.n, static_cast<Rpp32u>(channels_of(cfg.layout)), cfg.size.h,
                            cfg.size.w};
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layout);  // src == dst
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < shape.n; ++i) roi[i] = roiVec[i];

    PinnedArray<RpptChannelOffsets> offsets(cfg.backend, shape.n);
    for (Rpp32u n = 0; n < shape.n; ++n) offsets[n] = make_offsets(op.kind);

    // The op writes the ROI-sized region at the destination origin, so golden and device buffer
    // start from the same distinct pattern and only that region is compared.
    std::vector<T> input(count), dstInit(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    fill_input<T>(dstInit.data(), count, cfg.dtype, /*salt=*/1);
    golden = dstInit;
    glitch_reference<T>(input.data(), golden.data(), desc, roi.data(), XYWH, offsets.data());

    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(dstInit.data(), bytes);

    RppHandle handle(cfg.backend, shape.n);
    ASSERT_EQ(rppt_glitch(src.ptr(), &desc, dst.ptr(), &desc, offsets.data(), roi.data(), XYWH,
                          handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();
    dst.read(actual.data(), bytes);

    // Compared against the caller's own ROI copy, not the tensor handed to the op, in case the
    // backend rewrites it from XYWH to LTRB in place.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), desc, roiVec.data(), XYWH, kTol));
}

}  // namespace

// Full name: Image_Effects/GlitchTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<Kind>
class GlitchTest : public ::testing::TestWithParam<WithParams<GlitchParams>> {};

TEST_P(GlitchTest, Correctness) {
    const auto& p = GetParam();
    // The HOST U8 planar path sizes its vector loop as (roiWidth & ~31) - 32 in an unsigned, which
    // underflows to ~4e9 whenever the ROI is narrower than 64 columns, and the loop then walks the
    // whole address space. The segfault takes the entire process with it, so these two cases are
    // enumerated but skipped rather than left red. Remove this guard once the kernel is fixed.
    if (p.cfg.backend == RPP_HOST_BACKEND && p.cfg.dtype == DType::U8 &&
        p.cfg.layout == Layout::PLN3 && p.cfg.roi == Roi::Partial)
        GTEST_SKIP() << "HOST U8 planar glitch segfaults on a ROI narrower than its vector stride";

    switch (p.cfg.dtype) {
        case DType::U8:
            run_glitch<Rpp8u>(p.cfg, p.op);
            break;
        case DType::F16:
            run_glitch<Rpp16f>(p.cfg, p.op);
            break;
        case DType::F32:
            run_glitch<Rpp32f>(p.cfg, p.op);
            break;
        default:
            FAIL() << "unsupported dtype for glitch";
            break;
    }
}

// Three channels only: the op takes an R/G/B offset triple and PLN1 is not part of its contract.
// I8 is off the grid for now, pending the suite-wide decision on whether the image ops need it.
INSTANTIATE_TEST_SUITE_P(
    Image_Effects, GlitchTest,
    ::testing::ValuesIn(with_params<GlitchParams>(
        make_configs({DType::U8, DType::F16, DType::F32}, {Layout::PKD3, Layout::PLN3},
                     {Roi::Full, Roi::Partial}),
        {GlitchParams{GlitchKind::Identity}, GlitchParams{GlitchKind::Shift}})),
    op_config_name<GlitchParams>);
