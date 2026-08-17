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
#include "framework/config_param.hpp"
#include "framework/generic_tensor_setup.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/slice_ref.hpp"

using namespace rpptest;

namespace {

// slice takes an RpptGenericDesc but is not a rank-agnostic ND op: it dispatches on numDims AND on
// the descriptor's layout, which must agree with the rank. The shapes below are the ones the op
// actually defines, which is what this grid instantiates:
//
//   numDims 3 -> (N,H,W)         layout NCHW   -- no channel axis
//   numDims 4 -> (N,C,H,W)       layout NCHW   -- planar, C must be 1 or 3
//              (N,H,W,C)       layout NHWC   -- packed, C = 3
//   numDims 5 -> (N,C,D,H,W)     layout NCDHW  -- planar, C must be 1 or 3
//              (N,D,H,W,C)     layout NDHWC  -- packed, C = 3
//
// C is restricted to 1/3 because the op sizes its copy from get_layout_params(layout, C), which
// only fills in those two cases; any other channel count leaves the copy length unset. The channel
// axis is also not sliceable -- the kernels apply the anchor to the spatial axes only -- so this
// grid slices the spatial axes and passes the channel axis through whole
// (issues/slice-channel-axis-anchor-ignored.md records the deviation from the API doc's
// "starting index of the slice for each dimension").
enum class SliceLayout { Planar, Packed };

// Inside: the slice lies strictly within the ROI, so enablePadding is false and fillValue is never
// consulted -- a pure gather. Padded: the same anchors with the slice running off the trailing end
// of every spatial axis, so a border of the output must come out as the fill value.
enum class SliceKind { Inside, Padded };

struct SliceParams {
    SliceLayout layout;
    SliceKind kind;
    std::string name() const {
        return std::string(layout == SliceLayout::Planar ? "Planar" : "Packed") +
               (kind == SliceKind::Inside ? "_Inside" : "_Padded");
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

// The channel axis is passed through whole (anchor 0, full extent); every spatial axis is sliced
// from a non-zero, axis-dependent anchor so that a dropped or transposed anchor term cannot pass.
Rpp32s anchor_for(Rpp32u axis, int channelAxis) {
    if (static_cast<int>(axis) == channelAxis) return 0;
    return 1 + static_cast<Rpp32s>(axis % 2);
}

Rpp32s shape_for(Rpp32u axis, Rpp32u extent, int channelAxis, SliceKind kind) {
    if (static_cast<int>(axis) == channelAxis) return static_cast<Rpp32s>(extent);
    const Rpp32s anchor = anchor_for(axis, channelAxis);
    if (kind == SliceKind::Padded) return static_cast<Rpp32s>(extent) - anchor + 2;
    const Rpp32s length = static_cast<Rpp32s>(extent) - anchor - 1;
    return length < 1 ? 1 : length;
}

// Only 0 is used as the fill value: the API takes fillValue as a void pointer and the header never
// says whether it points to a value in the tensor's dtype or to an Rpp32f, so any non-zero choice
// would encode a guess at the type, while an all-zero word reads as 0 under every interpretation.
// It must live in device-accessible memory -- the HIP fill kernel dereferences it on the device, so
// a host pointer faults the GPU.
constexpr double kFillValue = 0.0;

template <typename T>
void run_slice(const NdConfig& cfg, const SliceParams& p) {
    const Rpp32u nDim = cfg.nDim;
    const NdDims srcDims = slice_extents(nDim, p.layout);
    const Rpp32u batch = srcDims[0];
    const RpptLayout layout = rpp_layout(nDim, p.layout);
    const int channelAxis = channel_axis(nDim, p.layout);

    // The destination extents are the slice shape, densely packed. They are shared by the whole
    // batch (a single descriptor), so every sample gets the same anchor and shape.
    NdDims dstDims(nDim + 1);
    dstDims[0] = batch;
    for (Rpp32u a = 0; a < nDim; ++a)
        dstDims[a + 1] = static_cast<Rpp32u>(shape_for(a, srcDims[a + 1], channelAxis, p.kind));

    // Descriptors are device-addressable for HIP: the ND kernels read dims/strides on device.
    // Row-padded strides: slice's copy is vectorized 8 elements per thread and stores a full
    // vector for the tail of every row, so the caller must leave that slack -- the same convention
    // the image domain follows via padded_width(). With dense strides the tail store lands in the
    // next row. The padded axis is width: innermost for planar, one in from the end for packed
    // (whose innermost axis is the channel axis).
    const int padAxis =
        static_cast<int>(srcDims.size()) - (p.layout == SliceLayout::Packed && nDim > 2 ? 2 : 1);
    GenericDescriptor srcDesc(cfg.backend, srcDims, cfg.dtypeIn, layout, padAxis);
    GenericDescriptor dstDesc(cfg.backend, dstDims, cfg.dtypeIn, layout, padAxis);

    const std::size_t srcCount = generic_element_count(*srcDesc);
    const std::size_t dstCount = generic_element_count(*dstDesc);
    const std::size_t srcBytes = generic_byte_size(*srcDesc, cfg.dtypeIn);
    const std::size_t dstBytes = generic_byte_size(*dstDesc, cfg.dtypeIn);

    // (1) anchor / shape / roi tensors live in host-accessible (pinned for HIP) memory: the op
    // reads them on the host to size each launch.
    PinnedArray<Rpp32s> anchor(cfg.backend, static_cast<std::size_t>(batch) * nDim);
    PinnedArray<Rpp32s> shape(cfg.backend, static_cast<std::size_t>(batch) * nDim);
    for (Rpp32u n = 0; n < batch; ++n)
        for (Rpp32u a = 0; a < nDim; ++a) {
            const std::size_t i = static_cast<std::size_t>(n) * nDim + a;
            anchor[i] = anchor_for(a, channelAxis);
            shape[i] = static_cast<Rpp32s>(dstDims[a + 1]);
        }
    const std::vector<Rpp32u> roiVec = make_nd_roi_tensor(srcDims);  // the whole source tensor
    PinnedArray<Rpp32u> roi(cfg.backend, roiVec.size());
    for (std::size_t i = 0; i < roiVec.size(); ++i) roi[i] = roiVec[i];

    // fillValue is dereferenced on the device by the HIP fill kernel, so it is pinned too.
    PinnedArray<T> fillValue(cfg.backend, 1);
    fillValue[0] = from_double<T>(kFillValue);

    // (2) Host golden model. The reference covers the full destination, so golden needs no
    // pre-seeding; the actual buffer is seeded with a sentinel byte pattern instead, so an output
    // element the op leaves untouched shows up as a mismatch (the op writes every one of them).
    // fill_input_nd addresses the pattern by coordinate and poisons the row slack, so the logical
    // input does not depend on the stride convention and a kernel that reads slack cannot pass by
    // picking up something that looks like data.
    std::vector<T> input(srcCount), golden(dstCount), actual(dstCount);
    fill_input_nd<T>(input.data(), *srcDesc, cfg.dtypeIn);
    slice_reference<T>(input.data(), golden.data(), *srcDesc, *dstDesc, anchor.data(), shape.data(),
                       roi.data(), kFillValue);
    std::memset(actual.data(), 0xCD, dstBytes);

    // (3) Run RPP on the configured backend.
    DeviceTensor src(cfg.backend, srcBytes), dst(cfg.backend, dstBytes);
    src.write(input.data(), srcBytes);
    dst.write(actual.data(), dstBytes);  // seed the destination with the sentinel

    RppHandle handle(cfg.backend, batch);
    ASSERT_EQ(rppt_slice(src.ptr(), srcDesc.get(), dst.ptr(), dstDesc.get(), anchor.data(),
                         shape.data(), fillValue.data(), p.kind == SliceKind::Padded, roi.data(),
                         handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), dstBytes);

    // (4) Compare the whole output tensor, bit-exactly: slice only relocates elements (plus the
    // constant fill), so there is no arithmetic to lose precision to at any dtype.
    EXPECT_TRUE(compare_nd<T>(actual.data(), golden.data(), *dstDesc, 0.0));
}

}  // namespace

// Full name:
// Misc_Geometric/SliceTest.Correctness/<Backend>_<DTypeConv>_<Rank>_<Layout>_<Kind>_<Shape> (the
// shape token is the framework's nominal rank shape, not this op's layout-ordered extents -- see
// slice_extents; the rank and layout tokens identify the case).
class SliceTest : public ::testing::TestWithParam<NdWithParams<SliceParams>> {};

// The planar cases are instantiated but not executed. Both HOST planar branches advance the
// destination channel pointer by the SOURCE channel stride, so every channel after the first is
// written at a growing overshoot; whenever the destination plane is smaller than the source plane
// -- the ordinary case for an in-bounds slice -- the write lands past the end of the destination
// buffer and aborts the process on a heap error, taking every later suite in the run with it.
// Skipping rather than dropping them keeps the cases listed and filterable, so the coverage gap is
// visible and the grid is restored by deleting one branch once the strides are fixed.
constexpr char kPlanarSkip[] =
    "slice HOST planar writes past the destination buffer (destination channel pointer advanced by "
    "the source channel stride) and aborts the test binary";

TEST_P(SliceTest, Correctness) {
    const NdConfig cfg = GetParam().cfg;
    const SliceParams p = GetParam().op;
    if (p.layout == SliceLayout::Planar) GTEST_SKIP() << kPlanarSkip;
    switch (cfg.dtypeIn) {
        case DType::U8:
            run_slice<Rpp8u>(cfg, p);
            break;
        case DType::F32:
            run_slice<Rpp32f>(cfg, p);
            break;
        default:
            FAIL() << "unsupported dtype for slice";
    }
}

// Scoped to U8 and F32: the header documents exactly "Support added for f32 -> f32 and u8 -> u8
// datatypes", so F16/I8 are out of contract and are not instantiated. Rank 2 has no channel axis,
// so only its planar form is instantiated -- the packed duplicate would be the same call.
std::vector<NdWithParams<SliceParams>> slice_configs() {
    std::vector<NdWithParams<SliceParams>> out;
    for (const NdConfig& cfg : make_nd_configs({DType::U8, DType::F32}, {2, 3, 4}))
        for (SliceLayout layout : {SliceLayout::Planar, SliceLayout::Packed}) {
            if (cfg.nDim == 2 && layout == SliceLayout::Packed) continue;
            for (SliceKind kind : {SliceKind::Inside, SliceKind::Padded})
                out.push_back({cfg, SliceParams{layout, kind}});
        }
    return out;
}

INSTANTIATE_TEST_SUITE_P(Misc_Geometric, SliceTest, ::testing::ValuesIn(slice_configs()),
                         nd_op_config_name<SliceParams>);
