#ifndef RPP_TEST_FILTER_COMMON_REF_H
#define RPP_TEST_FILTER_COMMON_REF_H

#include <rpp/rpp.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Shared primitives for the KxK spatial-filter golden models (box / gaussian / median / sobel).
// These are derived from the filters' definitions (a KxK window sampled per output pixel with a
// clamp-to-edge / REPLICATE border), NOT from the RPP kernels, so kernel bugs surface as diffs.
//
// Border & ROI model: each filter treats the ROI as its working image. For output-local pixel
// (j,i) the KxK window samples ROI-local neighbours (j+dy, i+dx) for dy,dx in [-r, r]
// (r = kernelSize/2), each neighbour coordinate clamped to the ROI bounds [0, roiH-1] x
// [0, roiW-1] (REPLICATE). The source/destination element mapping mirrors for_each_roi_io
// (framework/tensor_setup.hpp): source read at the ROI offset, output written packed at the
// destination origin.

// Gathers the KxK neighbourhood of output-local pixel (j,i) for the channel plane whose element
// base offset is `base`, clamping each neighbour to the ROI bounds (REPLICATE border), into
// window[0 .. kernelSize*kernelSize-1] in row-major order (dy = -r..r outer, dx = -r..r inner).
template <typename T>
inline void gather_roi_window(const T* src, const RpptDesc& d, const RoiBounds& b,
                              std::size_t base, int j, int i, int r, double* window) {
    const int roiH = static_cast<int>(b.h);
    const int roiW = static_cast<int>(b.w);
    int k = 0;
    for (int dy = -r; dy <= r; ++dy)
        for (int dx = -r; dx <= r; ++dx) {
            const int sy = std::min(std::max(j + dy, 0), roiH - 1);
            const int sx = std::min(std::max(i + dx, 0), roiW - 1);
            const std::size_t srcIdx =
                base + static_cast<std::size_t>(b.y0 + sy) * d.strides.hStride +
                static_cast<std::size_t>(b.x0 + sx) * d.strides.wStride;
            window[k++] = to_double(src[srcIdx]);
        }
}

// Applies a KxK linear filter (kernel row-major, length kernelSize*kernelSize, same dy/dx order as
// gather_roi_window) per channel over each image's ROI with the clamp-to-ROI border above. The
// weighted sum is quantized back to the dtype via quantize_stored -- integers round to nearest and
// clamp, floats clamp to [0,1] -- which is the intended semantics (round-to-nearest, not truncate;
// see the systemic I8 round-vs-truncate finding). Used by box_filter and gaussian_filter.
template <typename T>
void convolve_reference(const T* src, T* dst, const RpptDesc& d, DType dt, const RpptROI* roi,
                        RpptRoiType type, Rpp32u kernelSize, const std::vector<double>& kernel) {
    const int r = static_cast<int>(kernelSize / 2);
    const int kk = static_cast<int>(kernelSize * kernelSize);
    std::vector<double> window(kk);
    for (Rpp32u n = 0; n < d.n; ++n) {
        const RoiBounds b = roi_bounds(roi[n], type);
        for (Rpp32u c = 0; c < d.c; ++c) {
            const std::size_t base = static_cast<std::size_t>(n) * d.strides.nStride +
                                     static_cast<std::size_t>(c) * d.strides.cStride;
            for (int j = 0; j < static_cast<int>(b.h); ++j)
                for (int i = 0; i < static_cast<int>(b.w); ++i) {
                    gather_roi_window(src, d, b, base, j, i, r, window.data());
                    double acc = 0.0;
                    for (int k = 0; k < kk; ++k) acc += kernel[k] * window[k];
                    const std::size_t dstIdx =
                        base + static_cast<std::size_t>(j) * d.strides.hStride +
                        static_cast<std::size_t>(i) * d.strides.wStride;
                    dst[dstIdx] = from_double<T>(quantize_stored(acc, dt));
                }
        }
    }
}

}  // namespace rpptest

#endif  // RPP_TEST_FILTER_COMMON_REF_H
