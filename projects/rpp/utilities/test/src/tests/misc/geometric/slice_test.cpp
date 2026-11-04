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

#include <cstring>
#include <string>
#include <vector>

#include "framework/backend_memory.hpp"
#include "framework/compare.hpp"
#include "framework/config_param.hpp"
#include "framework/dtype.hpp"
#include "framework/intensity.hpp"
#include "framework/nd_config_param.hpp"
#include "framework/skip_list.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/slice_ref.hpp"

using namespace rpptest;

namespace {

// slice is not rank-agnostic: it dispatches on numDims and on the descriptor's layout, which must
// agree with the rank:
//
//   numDims 3 -> (N,H,W)         layout NCHW   -- no channel axis
//   numDims 4 -> (N,C,H,W)       layout NCHW   -- planar, C must be 1 or 3
//              (N,H,W,C)       layout NHWC   -- packed, C = 3
//   numDims 5 -> (N,C,D,H,W)     layout NCDHW  -- planar, C must be 1 or 3
//              (N,D,H,W,C)     layout NDHWC  -- packed, C = 3
//
// C must be 1 or 3 (get_layout_params only handles those). The kernels apply the anchor to the
// spatial axes only, contrary to the API doc, so the channel axis is passed through whole.
enum class SliceLayout { Planar, Packed };

// Inside: the slice lies within the ROI (no padding). Padded: the slice runs off the trailing end
// of every spatial axis, so the border must be fillValue. PaddedTail: Padded with a 0 width anchor,
// so the width tail is tested independently of the width-anchor offset. DensePadded: Padded with a
// dense destination stride, so the fill must cover the sample exactly with no row slack to absorb
// an off-by-one.
enum class SliceKind { Inside, Padded, PaddedTail, DensePadded };

struct SliceParams {
    SliceLayout layout;
    SliceKind kind;
    std::string name() const {
        static const char* const kKindNames[] = {"_Inside", "_Padded", "_PaddedTail",
                                                 "_DensePadded"};
        return std::string(layout == SliceLayout::Planar ? "Planar" : "Packed") +
               kKindNames[static_cast<int>(kind)];
    }
};

constexpr Rpp32u kChannels = 3;

// Per-sample extents for a rank, in the op's own layout order. Spatial extents are deliberately
// distinct and not all multiples of 8, so a vectorized copy that skips its remainder shows up.
NdDims slice_extents(Rpp32u nDim, SliceLayout layout) {
    const bool planar = layout == SliceLayout::Planar;
    switch (nDim) {
        case 2:
            return {2, 24, 32};  // N,H,W
        case 3:
            return planar ? NdDims{2, kChannels, 12, 20}   // N,C,H,W
                          : NdDims{2, 12, 20, kChannels};  // N,H,W,C
        default:
            return planar ? NdDims{2, kChannels, 4, 10, 20}   // N,C,D,H,W
                          : NdDims{2, 4, 10, 20, kChannels};  // N,D,H,W,C
    }
}

RpptLayout rpp_layout(Rpp32u nDim, SliceLayout layout) {
    if (nDim == 2) return RpptLayout::NCHW;  // (N,H,W): the numDims==2 path takes no channel axis
    if (nDim == 3) return layout == SliceLayout::Planar ? RpptLayout::NCHW : RpptLayout::NHWC;
    return layout == SliceLayout::Planar ? RpptLayout::NCDHW : RpptLayout::NDHWC;
}

// Which per-sample axis holds the channels, or -1 when there is none (rank 2). Planar layouts put
// it first, packed layouts last.
int channel_axis(Rpp32u nDim, SliceLayout layout) {
    if (nDim == 2) return -1;
    return layout == SliceLayout::Planar ? 0 : static_cast<int>(nDim) - 1;
}

// The per-sample width axis: innermost for planar, one in from the end for packed (whose innermost
// axis is the channel axis).
int width_axis(Rpp32u nDim, SliceLayout layout) {
    return static_cast<int>(nDim) - (layout == SliceLayout::Packed && nDim > 2 ? 2 : 1);
}

// Spatial axes (except the PaddedTail width axis) use a non-zero, axis-dependent anchor so a
// dropped or transposed anchor term cannot pass.
Rpp32s anchor_for(Rpp32u axis, int channelAxis, int widthAxis, SliceKind kind) {
    if (static_cast<int>(axis) == channelAxis) return 0;
    if (kind == SliceKind::PaddedTail && static_cast<int>(axis) == widthAxis) return 0;
    return 1 + static_cast<Rpp32s>(axis % 2);
}

Rpp32s shape_for(Rpp32u axis, Rpp32u extent, int channelAxis, int widthAxis, SliceKind kind) {
    if (static_cast<int>(axis) == channelAxis) return static_cast<Rpp32s>(extent);
    const Rpp32s anchor = anchor_for(axis, channelAxis, widthAxis, kind);
    if (kind != SliceKind::Inside) return static_cast<Rpp32s>(extent) - anchor + 2;
    const Rpp32s length = static_cast<Rpp32s>(extent) - anchor - 1;
    return length < 1 ? 1 : length;
}

// The header does not say whether fillValue points to the tensor dtype or to Rpp32f; zero reads the
// same either way.
constexpr double kFillValue = 0.0;

template <typename T>
void run_slice(const NdConfig& cfg, const SliceParams& p) {
    const Rpp32u nDim = cfg.nDim;
    const NdDims srcDims = slice_extents(nDim, p.layout);
    const Rpp32u batch = srcDims[0];
    const RpptLayout layout = rpp_layout(nDim, p.layout);
    const int channelAxis = channel_axis(nDim, p.layout);
    const int widthAxis = width_axis(nDim, p.layout);

    // One destination descriptor for the batch, so every sample gets the same anchor and shape.
    NdDims dstDims(nDim + 1);
    dstDims[0] = batch;
    for (Rpp32u a = 0; a < nDim; ++a)
        dstDims[a + 1] =
            static_cast<Rpp32u>(shape_for(a, srcDims[a + 1], channelAxis, widthAxis, p.kind));

    // Descriptors are device-addressable for HIP: the ND kernels read dims/strides on device.
    // Width is row-padded: slice stores a full 8-element vector for each row tail, so with dense
    // strides the tail store would land in the next row. DensePadded leaves the destination dense.
    const int padAxis = widthAxis + 1;
    GenericDescriptor srcDesc(cfg.backend, srcDims, cfg.dtypeIn, layout, padAxis);
    GenericDescriptor dstDesc(cfg.backend, dstDims, cfg.dtypeIn, layout,
                              p.kind == SliceKind::DensePadded ? -1 : padAxis);

    const std::size_t srcCount = generic_element_count(*srcDesc);
    const std::size_t dstCount = generic_element_count(*dstDesc);
    const std::size_t srcBytes = generic_byte_size(*srcDesc, cfg.dtypeIn);
    const std::size_t dstBytes = generic_byte_size(*dstDesc, cfg.dtypeIn);

    // anchor/shape/roi must be host-accessible: the op reads them on the host to size each launch.
    PinnedArray<Rpp32s> anchor(cfg.backend, static_cast<std::size_t>(batch) * nDim);
    PinnedArray<Rpp32s> shape(cfg.backend, static_cast<std::size_t>(batch) * nDim);
    for (Rpp32u n = 0; n < batch; ++n)
        for (Rpp32u a = 0; a < nDim; ++a) {
            const std::size_t i = static_cast<std::size_t>(n) * nDim + a;
            anchor[i] = anchor_for(a, channelAxis, widthAxis, p.kind);
            shape[i] = static_cast<Rpp32s>(dstDims[a + 1]);
        }
    const std::vector<Rpp32u> roiVec = make_nd_roi_tensor(srcDims);  // the whole source tensor
    PinnedArray<Rpp32u> roi(cfg.backend, roiVec.size());
    for (std::size_t i = 0; i < roiVec.size(); ++i) roi[i] = roiVec[i];

    // fillValue is dereferenced on the device by the HIP fill kernel, so it is pinned too.
    PinnedArray<T> fillValue(cfg.backend, 1);
    fillValue[0] = from_double<T>(kFillValue);

    // actual is seeded with a sentinel so an element the op leaves untouched is a mismatch.
    // fill_input_nd poisons the row slack, so a kernel that reads slack cannot pass.
    std::vector<T> input(srcCount), golden(dstCount), actual(dstCount);
    fill_input_nd<T>(input.data(), *srcDesc, cfg.dtypeIn);
    slice_reference<T>(input.data(), golden.data(), *srcDesc, *dstDesc, anchor.data(), shape.data(),
                       roi.data(), kFillValue);
    std::memset(actual.data(), 0xCD, dstBytes);

    DeviceTensor src(cfg.backend, srcBytes), dst(cfg.backend, dstBytes);
    src.write(input.data(), srcBytes);
    dst.write(actual.data(), dstBytes);  // seed the destination with the sentinel

    RppHandle handle(cfg.backend, batch);
    ASSERT_EQ(rppt_slice(src.ptr(), srcDesc.get(), dst.ptr(), dstDesc.get(), anchor.data(),
                         shape.data(), fillValue.data(), p.kind != SliceKind::Inside, roi.data(),
                         handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), dstBytes);

    EXPECT_TRUE(compare_nd<T>(actual.data(), golden.data(), *dstDesc, 0.0));
}

}  // namespace

// Full name:
// Misc_Geometric/SliceTest.Correctness/<Backend>_<DTypeConv>_<Rank>_<Layout>_<Kind>_<Shape> (the
// shape token is the nominal rank shape, not the extents from slice_extents).
class SliceTest : public SkipListTest<NdWithParams<SliceParams>> {};

// Planar cases are skipped rather than run: the HOST planar paths write past the destination buffer
// and abort the whole test binary. Skipping keeps the cases visible in the listing.
constexpr char kPlanarSkip[] =
    "slice HOST planar writes past the destination buffer (destination channel pointer advanced by "
    "the source channel stride) and aborts the test binary";
constexpr char kHipDenseSkip[] =
    "slice HIP stores whole vectors past the row end, so a dense destination is written out of "
    "bounds";

TEST_P(SliceTest, Correctness) {
    const NdConfig cfg = GetParam().cfg;
    const SliceParams p = GetParam().op;
    if (p.layout == SliceLayout::Planar) GTEST_SKIP() << kPlanarSkip;
    if (p.kind == SliceKind::DensePadded && cfg.backend == RPP_HIP_BACKEND)
        GTEST_SKIP() << kHipDenseSkip;
    dispatch_dtype<DType::U8, DType::F32>(
        cfg.dtypeIn, [&](auto tag) { run_slice<Element<decltype(tag)>>(cfg, p); });
}

// U8 and F32 are the conversions the header documents. Rank 2 has no channel axis, so it is planar
// only. PaddedTail is packed-only.
std::vector<NdWithParams<SliceParams>> slice_configs() {
    std::vector<NdWithParams<SliceParams>> out;
    // slice_extents() supplies the extents, so the shape axis is not varied.
    for (const NdConfig& cfg :
         make_nd_configs({DType::U8, DType::F32}, {2, 3, 4}, {NdShape::VectorAligned}))
        for (SliceLayout layout : {SliceLayout::Planar, SliceLayout::Packed}) {
            if (cfg.nDim == 2 && layout == SliceLayout::Packed) continue;
            for (SliceKind kind : {SliceKind::Inside, SliceKind::Padded, SliceKind::PaddedTail,
                                   SliceKind::DensePadded}) {
                if (kind == SliceKind::PaddedTail && layout != SliceLayout::Packed) continue;
                out.push_back({cfg, SliceParams{layout, kind}});
            }
        }
    return out;
}

INSTANTIATE_TEST_SUITE_P(Misc_Geometric, SliceTest, ::testing::ValuesIn(slice_configs()),
                         nd_op_config_name<SliceParams>);
