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
#include "reference/remap_ref.hpp"

using namespace rpptest;

namespace {

std::string interp_token(RpptInterpolationType i) {
    return i == NEAREST_NEIGHBOR ? "NN" : "BILINEAR";
}

// The remap-table pattern to build per image, plus the interpolation to sample with. Identity is a
// verbatim copy of the ROI region; hflip mirrors it horizontally; halfshift probes bilinear with a
// deterministic half-texel column offset.
enum class RemapKind { Identity, Hflip, Halfshift };

struct RemapParams {
    RemapKind kind;
    RpptInterpolationType interp;
    std::string tag;
    std::string name() const { return tag + "_" + interp_token(interp); }
};

// Tolerances are set from legitimate numeric error only; they are NOT loosened to hide integer
// truncation or any other kernel defect this test may surface.
double remap_tolerance(DType dt, RpptInterpolationType interp) {
    // Nearest-neighbour (and the identity/hflip table lookups) copy a texel verbatim: bit-exact.
    if (interp == NEAREST_NEIGHBOR) return 0.0;
    // Bilinear blends in float; allow only genuine rounding error. Integer types round to nearest.
    return kRoundingTolerance(dt);
}

template <typename T>
void run_remap(const TestConfig& cfg, const RemapParams& op) {
    const Rpp32u c = static_cast<Rpp32u>(channels_of(cfg.layout));
    const Rpp32u N = cfg.size.n, imgH = cfg.size.h, imgW = cfg.size.w;
    const TensorShape shape{N, c, imgH, imgW};
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layout);  // src == dst
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    // Remap tables: a SEPARATE F32, single-channel, UNPADDED tensor (row stride is exactly imgW).
    RpptDesc td{};
    td.numDims = 4;
    td.offsetInBytes = 0;
    td.dataType = F32;
    td.layout = NHWC;
    td.n = N;
    td.c = 1;
    td.h = imgH;
    td.w = imgW;
    td.strides.nStride = static_cast<Rpp32u>(imgH) * imgW;  // unpadded
    td.strides.hStride = imgW;
    td.strides.wStride = 1;
    td.strides.cStride = 1;

    // Per-image ROIs in host-accessible (pinned for HIP) memory.
    PinnedArray<RpptROI> roi(cfg.backend, N);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < N; ++i) roi[i] = roiVec[i];

    // Fill both tables with ABSOLUTE source coordinates, packed at the table origin over the
    // [0,roiH) x [0,roiW) output region of each image; the untouched remainder stays zero.
    const std::size_t tblCount = static_cast<std::size_t>(N) * imgH * imgW;
    PinnedArray<Rpp32f> rowT(cfg.backend, tblCount), colT(cfg.backend, tblCount);
    for (std::size_t k = 0; k < tblCount; ++k) {
        rowT[k] = 0.0f;
        colT[k] = 0.0f;
    }
    for (Rpp32u n = 0; n < N; ++n) {
        const RoiBounds b = roi_bounds(roi[n], XYWH);
        const std::size_t base = static_cast<std::size_t>(n) * td.strides.nStride;
        for (Rpp32u j = 0; j < b.h; ++j)
            for (Rpp32u i = 0; i < b.w; ++i) {
                const std::size_t idx = base + static_cast<std::size_t>(j) * td.strides.hStride +
                                        static_cast<std::size_t>(i) * td.strides.wStride;
                float col = 0.0f, row = 0.0f;
                switch (op.kind) {
                    case RemapKind::Identity:
                        col = static_cast<float>(b.x0 + i);
                        row = static_cast<float>(b.y0 + j);
                        break;
                    case RemapKind::Hflip:
                        col = static_cast<float>(b.x0 + (b.w - 1 - i));
                        row = static_cast<float>(b.y0 + j);
                        break;
                    case RemapKind::Halfshift:
                        col = static_cast<float>(b.x0 + i) + 0.5f;
                        row = static_cast<float>(b.y0 + j);
                        break;
                }
                colT[idx] = col;
                rowT[idx] = row;
            }
    }

    // (1) Host golden model. golden starts as a copy of the input so the untouched (outside-ROI)
    // region is defined; only the ROI-sized output region is overwritten.
    std::vector<T> input(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    golden = input;
    remap_reference<T>(input.data(), golden.data(), desc, cfg.dtype, rowT.data(), colT.data(), td,
                       roi.data(), XYWH, op.interp);

    // (2) Run RPP on the configured backend.
    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, N);
    ASSERT_EQ(rppt_remap(src.ptr(), &desc, dst.ptr(), &desc, rowT.data(), colT.data(), &td,
                         op.interp, roi.data(), XYWH, handle.get(), cfg.backend),
              RPP_SUCCESS);

    // (3) Retrieve the result on the host (no-op copy for HOST, device->host for HIP).
    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    // (4) Compare within tolerance over the ROI-sized output region at the destination origin.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), desc, roi.data(), XYWH,
                               remap_tolerance(cfg.dtype, op.interp)));
}

}  // namespace

// Full name: Image_Geometric/RemapTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<Tag>_<Interp>
class RemapTest : public ::testing::TestWithParam<WithParams<RemapParams>> {};

TEST_P(RemapTest, Correctness) {
    const auto& p = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(p.cfg.dtype, [&](auto tag) {
        run_remap<Element<decltype(tag)>>(p.cfg, p.op);
    });
}

// identity: verbatim copy of the ROI region. hflip: horizontal mirror within the ROI. halfshift:
// half-texel column offset, bilinear only, exercises the 4-tap blend (including the black-border tap
// at the right edge) deterministically.
INSTANTIATE_TEST_SUITE_P(
    Image_Geometric, RemapTest,
    ::testing::ValuesIn(with_params<RemapParams>(
        make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                     {Layout::PKD3, Layout::PLN3, Layout::PLN1}, {Roi::Full, Roi::Partial}),
        {RemapParams{RemapKind::Identity, NEAREST_NEIGHBOR, "identity"},
         RemapParams{RemapKind::Hflip, NEAREST_NEIGHBOR, "hflip"},
         RemapParams{RemapKind::Identity, BILINEAR, "identity"},
         RemapParams{RemapKind::Halfshift, BILINEAR, "halfshift"}})),
    op_config_name<RemapParams>);
