#ifndef RPP_TEST_SOBEL_FILTER_REF_H
#define RPP_TEST_SOBEL_FILTER_REF_H

#include <rpp/rpp.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/filter_common.hpp"

namespace rpptest {

// Independent host golden model for rppt_sobel_filter, derived from the canonical 3x3 Sobel
// operator definition (Gx/Gy gradient kernels, gradient magnitude for the XY case, REPLICATE
// border), NOT from the RPP kernel. Used as the reference for BOTH backends so kernel bugs
// surface as diffs.
//
// Scope: PLN1 only, kernelSize = 3 only. sobel_filter's dstDesc is always single-channel
// grayscale (c=1, NCHW), so a 3-channel input would require an undocumented RGB->grayscale
// conversion (not independently derivable), and the extended k=5/7 kernels are
// convention-dependent (varying coefficient conventions). Both are deferred here; this model
// covers grayscale-in/grayscale-out with the universally-defined 3x3 Sobel operator.
//
// Per output-local pixel (j,i) the 3x3 window samples ROI-local neighbours (j+dy, i+dx) for
// dy,dx in [-r, r] (r = kernelSize/2), each neighbour clamped to the ROI bounds (REPLICATE),
// via gather_roi_window (reference/filter_common.hpp). Sobel needs the raw gx/gy before
// quantization (for the magnitude case), so this writes its own loop instead of using
// convolve_reference. The source/destination element mapping mirrors for_each_roi_io: source
// read at the ROI offset, output written packed at the destination origin.
//
// Kernels (row-major, dy=-1..1 outer, dx=-1..1 inner):
//   Gx = [-1,0,1, -2,0,2, -1,0,1]      Gy = [-1,-2,-1, 0,0,0, 1,2,1]
// gx = sum Gx[k]*w[k], gy = sum Gy[k]*w[k].  sobelType 0 -> gx, 1 -> gy, 2 -> sqrt(gx^2+gy^2).
// The result is quantized back to the dtype via quantize_stored (U8 round+clamp[0,255],
// I8 round+clamp[-128,127], F16/F32 clamp[0,1]) -- gradients can be negative / out of range, so
// clamping is the intended "same depth as src" behavior; any resulting diff is a finding, not a
// reference bug.
template <typename T>
void sobel_filter_reference(const T* src, T* dst, const RpptDesc& d, DType dt, const RpptROI* roi,
                            RpptRoiType type, Rpp32u sobelType, Rpp32u kernelSize) {
    static const double Gx[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    static const double Gy[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    const int r = static_cast<int>(kernelSize / 2);
    double window[9];
    for (Rpp32u n = 0; n < d.n; ++n) {
        const RoiBounds b = roi_bounds(roi[n], type);
        for (Rpp32u c = 0; c < d.c; ++c) {
            const std::size_t base = static_cast<std::size_t>(n) * d.strides.nStride +
                                     static_cast<std::size_t>(c) * d.strides.cStride;
            for (int j = 0; j < static_cast<int>(b.h); ++j)
                for (int i = 0; i < static_cast<int>(b.w); ++i) {
                    gather_roi_window(src, d, b, base, j, i, r, window);
                    double gx = 0.0, gy = 0.0;
                    for (int k = 0; k < 9; ++k) {
                        gx += Gx[k] * window[k];
                        gy += Gy[k] * window[k];
                    }
                    const double result = sobelType == 0   ? gx
                                          : sobelType == 1 ? gy
                                                           : std::sqrt(gx * gx + gy * gy);
                    const std::size_t dstIdx =
                        base + static_cast<std::size_t>(j) * d.strides.hStride +
                        static_cast<std::size_t>(i) * d.strides.wStride;
                    dst[dstIdx] = from_double<T>(quantize_stored(result, dt));
                }
        }
    }
}

}  // namespace rpptest

#endif  // RPP_TEST_SOBEL_FILTER_REF_H
