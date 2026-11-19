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
#include "reference/remap_ref.hpp"

using namespace rpptest;

namespace {

std::string interp_token(RpptInterpolationType i) {
    return i == NEAREST_NEIGHBOR ? "NN" : "BILINEAR";
}

// The remap-table pattern built per image.
enum class RemapKind { Identity, Hflip, Halfshift };

struct RemapParams {
    RemapKind kind;
    RpptInterpolationType interp;
    std::string tag;
    std::string name() const {
        return tag + "_" + interp_token(interp);
    }
};

// Nearest-neighbour copies texels verbatim; bilinear allows rounding error only.
double remap_tolerance(DType dt, RpptInterpolationType interp) {
    if (interp == NEAREST_NEIGHBOR) return 0.0;
    return kRoundingTolerance(dt);
}

template <typename T>
void run_remap(const TestConfig& cfg, const RemapParams& op) {
    const Rpp32u N = cfg.size.n, imgH = cfg.size.h, imgW = cfg.size.w;
    RpptDesc srcDesc = make_src_descriptor(cfg);
    RpptDesc dstDesc = make_dst_descriptor(cfg);
    const std::size_t count = element_count(srcDesc);
    const std::size_t bytes = byte_size(srcDesc, cfg.dtype);

    // Remap tables: a separate F32, single-channel, unpadded tensor.
    RpptDesc td{};
    td.numDims = 4;
    td.offsetInBytes = 0;
    td.dataType = F32;
    td.layout = NHWC;
    td.n = N;
    td.c = 1;
    td.h = imgH;
    td.w = imgW;
    td.strides.nStride = static_cast<Rpp32u>(imgH) * imgW;
    td.strides.hStride = imgW;
    td.strides.wStride = 1;
    td.strides.cStride = 1;

    PinnedArray<RpptROI> roi(cfg.backend, N);
    const std::vector<RpptROI> roiVec = make_roi(srcDesc, cfg.roi);
    for (Rpp32u i = 0; i < N; ++i) roi[i] = roiVec[i];

    // Tables hold absolute source coordinates, packed at the table origin over the ROI-sized
    // region.
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

    std::vector<T> input(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    golden = input;
    remap_reference<T>(input.data(), srcDesc, golden.data(), dstDesc, cfg.dtype, rowT.data(),
                       colT.data(), td, roi.data(), XYWH, op.interp);

    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, N);
    ASSERT_EQ(rppt_remap(src.ptr(), &srcDesc, dst.ptr(), &dstDesc, rowT.data(), colT.data(), &td,
                         op.interp, roi.data(), XYWH, handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    // Uses the test's own ROI copy: HIP rewrites the op's ROI to LTRB in place.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), dstDesc, roiVec.data(), XYWH,
                               remap_tolerance(cfg.dtype, op.interp)));
}

}  // namespace

// Full name:
// Image_Geometric/RemapTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<Tag>_<Interp>
class RemapTest : public SkipListTest<WithParams<RemapParams>> {};

TEST_P(RemapTest, Correctness) {
    const auto& p = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(
        p.cfg.dtype, [&](auto tag) { run_remap<Element<decltype(tag)>>(p.cfg, p.op); });
}

// halfshift is a half-texel column offset that exercises the bilinear blend, including the
// black-border tap at the right edge.
INSTANTIATE_TEST_SUITE_P(Image_Geometric, RemapTest,
                         ::testing::ValuesIn(with_params<RemapParams>(
                             make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                                          {{Layout::PKD3, Layout::PKD3},
                                           {Layout::PLN3, Layout::PLN3},
                                           {Layout::PLN1, Layout::PLN1},
                                           {Layout::PKD3, Layout::PLN3},
                                           {Layout::PLN3, Layout::PKD3}},
                                          {Roi::Full, Roi::Partial},
                                          {presets::kDefaultSize, presets::kTailWidthSize}),
                             {RemapParams{RemapKind::Identity, NEAREST_NEIGHBOR, "identity"},
                              RemapParams{RemapKind::Hflip, NEAREST_NEIGHBOR, "hflip"},
                              RemapParams{RemapKind::Identity, BILINEAR, "identity"},
                              RemapParams{RemapKind::Halfshift, BILINEAR, "halfshift"}})),
                         op_config_name<RemapParams>);
