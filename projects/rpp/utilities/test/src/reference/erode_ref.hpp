#ifndef RPP_TEST_ERODE_REF_H
#define RPP_TEST_ERODE_REF_H

#include <rpp/rpp.h>

#include <algorithm>
#include <cstddef>

#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Independent host golden model for rppt_erode, derived from the op's definition
// (grayscale morphological erosion: per-channel MIN over a flat KxK square window centered
// on each pixel, with clamp-to-edge border), NOT from the RPP kernel. Used as the reference
// for both HOST and HIP backends so kernel bugs surface as diffs.
//
// The min selects an existing source value (no arithmetic), so the result is bit-exact for
// every dtype; to_double/from_double round-trip exactly for the integer and float dtypes here.
//
// Index math mirrors for_each_roi_io (framework/tensor_setup.hpp): source is read at the ROI
// offset and output written packed at the destination origin. For output-local (j,i) the KxK
// window samples ROI-local neighbors (j+dy, i+dx), each clamped to the ROI bounds
// [0, roiH-1] x [0, roiW-1], then mapped to the real source element.
template <typename T>
void erode_reference(const T* src, T* dst, const RpptDesc& d, DType /*dt*/, const RpptROI* roi,
                     RpptRoiType type, Rpp32u kernelSize) {
    const int r = static_cast<int>(kernelSize / 2);
    for (Rpp32u n = 0; n < d.n; ++n) {
        const RoiBounds b = roi_bounds(roi[n], type);
        for (Rpp32u c = 0; c < d.c; ++c) {
            const std::size_t base = static_cast<std::size_t>(n) * d.strides.nStride +
                                     static_cast<std::size_t>(c) * d.strides.cStride;
            for (Rpp32u j = 0; j < b.h; ++j)
                for (Rpp32u i = 0; i < b.w; ++i) {
                    double minVal = 0.0;
                    bool first = true;
                    for (int dy = -r; dy <= r; ++dy)
                        for (int dx = -r; dx <= r; ++dx) {
                            const int cr = std::min(std::max(static_cast<int>(j) + dy, 0),
                                                    static_cast<int>(b.h) - 1);
                            const int cc = std::min(std::max(static_cast<int>(i) + dx, 0),
                                                    static_cast<int>(b.w) - 1);
                            const std::size_t srcIdx = base + (b.y0 + cr) * d.strides.hStride +
                                                       (b.x0 + cc) * d.strides.wStride;
                            const double v = to_double(src[srcIdx]);
                            if (first || v < minVal) {
                                minVal = v;
                                first = false;
                            }
                        }
                    const std::size_t dstIdx =
                        base + j * d.strides.hStride + i * d.strides.wStride;
                    dst[dstIdx] = from_double<T>(minVal);
                }
        }
    }
}

}  // namespace rpptest

#endif  // RPP_TEST_ERODE_REF_H
