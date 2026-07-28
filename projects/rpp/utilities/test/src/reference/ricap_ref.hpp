#ifndef RPP_TEST_RICAP_REF_H
#define RPP_TEST_RICAP_REF_H

#include <rpp/rpp.h>

#include <cstddef>

#include "framework/tensor_setup.hpp"

namespace rpptest {

// Independent host golden model for rppt_ricap, derived from the RICAP definition
// (https://arxiv.org/abs/1811.09030) plus the public API doc in rppt_tensor_effects_augmentations.h,
// NOT from the RPP kernel. Used as the reference for both HOST and HIP so kernel bugs surface as
// diffs.
//
// RICAP assembles each output image as a 2x2 mosaic of four crops taken from (possibly) four
// different images of the same batch. With output extent WxH and boundary point (w0, h0), region k
// occupies the output rectangle:
//   k=0 -> origin (0 , 0 ), extent w0     x h0
//   k=1 -> origin (w0, 0 ), extent (W-w0) x h0
//   k=2 -> origin (0 , h0), extent w0     x (H-h0)
//   k=3 -> origin (w0, h0), extent (W-w0) x (H-h0)
// cropRegion[k] supplies both the SOURCE origin (x, y) of region k and its extent
// (roiWidth, roiHeight); the boundary point is therefore implied by the extents themselves, so the
// region output origins below are derived from cropRegion[0]'s extent rather than hard-coded. The
// four extents are expected to tile the output exactly (w0+w1 == W, h0+h2 == H, w0 == w2,
// w1 == w3, h0 == h1, h2 == h3).
//
// Region k of output image n is read from source image permutation[n*4 + k]; all batch images live
// in one tensor, so image p starts at p * d.strides.nStride.
//
// Note: the suite's usual "source read at ROI offset, output written packed at the destination
// origin" rule does not apply here. ricap has no source-ROI argument (the four crop rectangles are
// the ROI) and it writes the ENTIRE destination frame, so the reference walks the full frame with
// explicit placement math instead of for_each_roi_io().
//
// ricap does no arithmetic, rounding, or clamping: every output element is a verbatim copy of a
// source element, so the result is bit-exact for U8/F16/F32/I8 alike (compare at tolerance 0).
template <typename T>
void ricap_reference(const T* src, T* dst, const RpptDesc& d, const Rpp32u* permutation,
                     const RpptROI* cropRegion, RpptRoiType roiType) {
    RoiBounds crop[4];
    for (int k = 0; k < 4; ++k) crop[k] = roi_bounds(cropRegion[k], roiType);

    // Region output origins from the cumulative crop extents (the boundary point).
    const Rpp32u originX[4] = {0, crop[0].w, 0, crop[0].w};
    const Rpp32u originY[4] = {0, 0, crop[0].h, crop[0].h};

    for (Rpp32u n = 0; n < d.n; ++n)
        for (int k = 0; k < 4; ++k) {
            const Rpp32u p = permutation[n * 4 + k];
            for (Rpp32u c = 0; c < d.c; ++c) {
                const std::size_t dstBase = static_cast<std::size_t>(n) * d.strides.nStride +
                                            static_cast<std::size_t>(c) * d.strides.cStride;
                const std::size_t srcBase = static_cast<std::size_t>(p) * d.strides.nStride +
                                            static_cast<std::size_t>(c) * d.strides.cStride;
                for (Rpp32u j = 0; j < crop[k].h; ++j)
                    for (Rpp32u i = 0; i < crop[k].w; ++i) {
                        const Rpp32u dy = originY[k] + j, dx = originX[k] + i;
                        const Rpp32u sy = crop[k].y0 + j, sx = crop[k].x0 + i;
                        if (dy >= d.h || dx >= d.w || sy >= d.h || sx >= d.w) continue;
                        dst[dstBase + dy * d.strides.hStride + dx * d.strides.wStride] =
                            src[srcBase + sy * d.strides.hStride + sx * d.strides.wStride];
                    }
            }
        }
}

}  // namespace rpptest

#endif  // RPP_TEST_RICAP_REF_H
