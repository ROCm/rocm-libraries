#ifndef RPP_TEST_DILATE_REF_H
#define RPP_TEST_DILATE_REF_H

#include <rpp/rpp.h>

#include <algorithm>
#include <cstddef>

#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Independent host golden model for rppt_dilate, derived from the op's definition
// (grayscale morphological dilation: per-channel MAX over a KxK flat square window,
// clamp-to-edge border), NOT from the RPP kernel. Used as the reference for both
// backends so kernel bugs surface as diffs.
//
// The window for output-local pixel (j,i) samples ROI-local neighbours (j+dy, i+dx)
// for dy,dx in [-r, r] (r = kernelSize/2), each neighbour coordinate clamped to the
// ROI bounds [0, roiH-1] x [0, roiW-1]. The source/destination element mapping mirrors
// for_each_roi_io (framework/tensor_setup.hpp): source read at the ROI offset, output
// written packed at the destination origin. dilation only selects an existing pixel
// value (no arithmetic), so to_double/from_double round-trips exactly for every dtype
// here and the result is bit-exact (tolerance 0).
template <typename T>
void dilate_reference(const T* src, T* dst, const RpptDesc& d, DType /*dt*/, const RpptROI* roi,
                      RpptRoiType type, Rpp32u kernelSize) {
    const int r = static_cast<int>(kernelSize / 2);
    for (Rpp32u n = 0; n < d.n; ++n) {
        const RoiBounds b = roi_bounds(roi[n], type);
        const int roiH = static_cast<int>(b.h);
        const int roiW = static_cast<int>(b.w);
        for (Rpp32u c = 0; c < d.c; ++c) {
            const std::size_t base = static_cast<std::size_t>(n) * d.strides.nStride +
                                     static_cast<std::size_t>(c) * d.strides.cStride;
            for (int j = 0; j < roiH; ++j)
                for (int i = 0; i < roiW; ++i) {
                    double maxVal = -1e300;
                    for (int dy = -r; dy <= r; ++dy)
                        for (int dx = -r; dx <= r; ++dx) {
                            const int sy = std::min(std::max(j + dy, 0), roiH - 1);
                            const int sx = std::min(std::max(i + dx, 0), roiW - 1);
                            const std::size_t srcIdx =
                                base + static_cast<std::size_t>(b.y0 + sy) * d.strides.hStride +
                                static_cast<std::size_t>(b.x0 + sx) * d.strides.wStride;
                            maxVal = std::max(maxVal, to_double(src[srcIdx]));
                        }
                    const std::size_t dstIdx = base +
                                               static_cast<std::size_t>(j) * d.strides.hStride +
                                               static_cast<std::size_t>(i) * d.strides.wStride;
                    dst[dstIdx] = from_double<T>(maxVal);
                }
        }
    }
}

}  // namespace rpptest

#endif  // RPP_TEST_DILATE_REF_H
