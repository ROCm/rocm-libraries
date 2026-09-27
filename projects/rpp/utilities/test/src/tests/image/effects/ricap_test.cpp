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
#include "reference/ricap_ref.hpp"

using namespace rpptest;

namespace {

// The 2x2 mosaic boundary point (w0, h0) plus the source origin of each of the four crops. The
// region extents follow from the boundary (region 0/2 are w0 wide, 1/3 are W-w0; region 0/1 are h0
// high, 2/3 are H-h0), so the four crops tile the output exactly.
struct RicapParams {
    Rpp32u w0, h0;
    Rpp32u cropX[4], cropY[4];
    bool distinctSources;  // true: region k of image n draws from image (n+k); false: all from 0
    std::string tag;
    std::string name() const {
        return tag;
    }
};

template <typename T>
void run_ricap(const TestConfig& cfg, const RicapParams& op) {
    RpptDesc srcDesc = make_src_descriptor(cfg);  // RPP takes a non-const ptr
    RpptDesc dstDesc = make_dst_descriptor(cfg);
    const std::size_t count = element_count(srcDesc);
    const std::size_t bytes = byte_size(srcDesc, cfg.dtype);

    // Exactly 4 crop ROIs shared by the whole batch (not one per image): they are ricap's source
    // rectangles AND the output region extents. Host-accessible / device-visible (pinned on HIP).
    PinnedArray<RpptROI> cropRegion(cfg.backend, 4);
    const Rpp32u regionW[4] = {op.w0, srcDesc.w - op.w0, op.w0, srcDesc.w - op.w0};
    const Rpp32u regionH[4] = {op.h0, op.h0, srcDesc.h - op.h0, srcDesc.h - op.h0};
    for (int k = 0; k < 4; ++k) {
        cropRegion[k].xywhROI.xy.x = static_cast<int>(op.cropX[k]);
        cropRegion[k].xywhROI.xy.y = static_cast<int>(op.cropY[k]);
        cropRegion[k].xywhROI.roiWidth = static_cast<int>(regionW[k]);
        cropRegion[k].xywhROI.roiHeight = static_cast<int>(regionH[k]);
        ASSERT_LE(op.cropX[k] + regionW[k], srcDesc.w) << "crop " << k << " overruns image width";
        ASSERT_LE(op.cropY[k] + regionH[k], srcDesc.h) << "crop " << k << " overruns image height";
    }

    // batchSize * 4 source-image indices: region k of output image n comes from image
    // permutation[n*4 + k].
    PinnedArray<Rpp32u> permutation(cfg.backend, static_cast<std::size_t>(cfg.size.n) * 4);
    for (Rpp32u n = 0; n < cfg.size.n; ++n)
        for (Rpp32u k = 0; k < 4; ++k)
            permutation[n * 4 + k] = op.distinctSources ? (n + k) % cfg.size.n : 0u;

    // (1) Host golden model. One tensor holds the whole batch, and fill_input's pattern advances
    // with the linear element offset, so the images already differ from each other.
    std::vector<T> in(count), golden(count), actual(count);
    fill_input<T>(in.data(), count, cfg.dtype);
    ricap_reference<T>(in.data(), srcDesc, golden.data(), dstDesc, permutation.data(),
                       cropRegion.data(), XYWH);

    // (2) Run RPP on the configured backend.
    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(in.data(), bytes);

    RppHandle handle(cfg.backend, cfg.size.n);
    ASSERT_EQ(rppt_ricap(src.ptr(), &srcDesc, dst.ptr(), &dstDesc, permutation.data(),
                         cropRegion.data(), XYWH, handle.get(), cfg.backend),
              RPP_SUCCESS);

    // (3) Retrieve the result on the host (no-op copy for HOST, device->host for HIP).
    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    // (4) ricap is a verbatim copy for every dtype and it writes the entire destination frame, so
    // compare the full image bit-exact (tolerance 0).
    PinnedArray<RpptROI> fullRoi(cfg.backend, cfg.size.n);
    const std::vector<RpptROI> fullVec = make_roi(dstDesc, Roi::Full);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) fullRoi[i] = fullVec[i];
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), dstDesc, fullRoi.data(), XYWH, 0.0));
}

}  // namespace

// Full name:
// Image_Effects/RicapTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<Case>
class RicapTest : public SkipListTest<WithParams<RicapParams>> {};

TEST_P(RicapTest, Correctness) {
    const auto& p = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(
        p.cfg.dtype, [&](auto tag) { run_ricap<Element<decltype(tag)>>(p.cfg, p.op); });
}

// Only {Roi::Full} on the roi axis: ricap has no standard source-ROI argument (its four crop
// rectangles ARE the ROI) and it always writes the full destination frame, so a Partial-ROI variant
// would not be meaningful -- the same reasoning as crop_and_patch. The 4-image batch lets the four
// regions each draw from a distinct source image (the paper's case); ricap requires every image in
// the batch to have the same dimensions.
//
// Cases (output 36 high x 48 wide):
//   quad   - boundary (20, 16), crops at (2,3)/(10,5)/(6,12)/(14,8), permutation {n, n+1, n+2, n+3}
//            so every output image mixes four different sources. Every crop fits: 2+20, 10+28,
//            6+20, 14+28 <= 48 and 3+16, 5+16, 12+20, 8+20 <= 36.
//   quad8  - the same mix with boundary (24, 16), i.e. both region widths a multiple of 8.
//   single - boundary (24, 18), all crops at (0,0), permutation {0,0,0,0}: the degenerate case
//            where the output is image 0's top-left quadrant tiled 2x2.
//   bound  - single's degenerate crops/permutation with quad's boundary (20, 16): isolates the
//            vertical split as the only variable.
// Same-layout cases plus both directions of the fused output-layout conversion.
INSTANTIATE_TEST_SUITE_P(Image_Effects, RicapTest,
                         ::testing::ValuesIn(with_params<RicapParams>(
                             make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                                          {{Layout::PKD3, Layout::PKD3},
                                           {Layout::PLN3, Layout::PLN3},
                                           {Layout::PLN1, Layout::PLN1},
                                           {Layout::PKD3, Layout::PLN3},
                                           {Layout::PLN3, Layout::PKD3}},
                                          {Roi::Full}, {{4, 36, 48}}),
                             {RicapParams{20, 16, {2, 10, 6, 14}, {3, 5, 12, 8}, true, "quad"},
                              RicapParams{24, 18, {0, 0, 0, 0}, {0, 0, 0, 0}, false, "single"},
                              RicapParams{24, 16, {2, 10, 6, 14}, {3, 5, 12, 8}, true, "quad8"},
                              RicapParams{20, 16, {0, 0, 0, 0}, {0, 0, 0, 0}, false, "bound"}})),
                         op_config_name<RicapParams>);
