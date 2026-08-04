#ifndef RPP_TEST_FLIP_REF_H
#define RPP_TEST_FLIP_REF_H

#include <rpp/rpp.h>

#include <cstddef>

#include "framework/tensor_setup.hpp"

namespace rpptest {

// Independent host golden model for rppt_flip, derived from the op's definition (a
// mask-controlled mirror of the ROI region about its vertical and/or horizontal axis), NOT from
// the RPP kernel. Used as the reference for both backends so kernel bugs surface as diffs.
//
// Flip is a pure permutation of source elements -- no arithmetic, so no rounding or clamping and
// every dtype is bit-exact. The source is read at the ROI offset and the output written packed at
// the destination origin (the placement every RPP op uses), so output element (j, i) comes from
// source (y0 + [vertical ? h-1-j : j], x0 + [horizontal ? w-1-i : i]).
template <typename T>
void flip_reference(const T* src, T* dst, const RpptDesc& d, const RpptROI* roi,
                    RpptRoiType roiType, Rpp32u horizontal, Rpp32u vertical) {
    for_each_roi_plane(d, roi, roiType, [&](Rpp32u, const RoiBounds& b, Rpp32u, std::size_t base) {
        for (Rpp32u j = 0; j < b.h; ++j)
            for (Rpp32u i = 0; i < b.w; ++i) {
                const Rpp32u srcRow = b.y0 + (vertical ? (b.h - 1 - j) : j);
                const Rpp32u srcCol = b.x0 + (horizontal ? (b.w - 1 - i) : i);
                dst[plane_index(d, base, j, i)] = src[plane_index(d, base, srcRow, srcCol)];
            }
    });
}

}  // namespace rpptest

#endif  // RPP_TEST_FLIP_REF_H
