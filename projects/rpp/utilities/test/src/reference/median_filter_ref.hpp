#ifndef RPP_TEST_MEDIAN_FILTER_REF_H
#define RPP_TEST_MEDIAN_FILTER_REF_H

#include <rpp/rpp.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/filter_common.hpp"

namespace rpptest {

// Independent host golden model for rppt_median_filter, derived from the median-filter definition
// (each output pixel is the middle element of its sorted KxK neighbourhood, per channel, REPLICATE
// border), NOT from the RPP kernel. Used as the reference for both backends so kernel bugs surface
// as diffs.
//
// Median is a rank filter (not a convolution): the window is gathered via gather_roi_window (same
// clamp-to-ROI REPLICATE border as the other filters), sorted, and the middle element selected.
// kernelSize is odd so kernelSize*kernelSize is odd and the median is an EXISTING pixel value (no
// arithmetic), so to_double/from_double round-trips exactly for every dtype and the result is
// bit-exact (tolerance 0). The source/destination element mapping mirrors for_each_roi_io: source
// read at the ROI offset, output written packed at the destination origin.
template <typename T>
void median_filter_reference(const T* src, T* dst, const RpptDesc& d, DType /*dt*/,
                             const RpptROI* roi, RpptRoiType type, Rpp32u kernelSize) {
    const int r = static_cast<int>(kernelSize / 2);
    const int kk = static_cast<int>(kernelSize * kernelSize);
    std::vector<double> window(kk);
    for (Rpp32u n = 0; n < d.n; ++n) {
        const RoiBounds b = roi_bounds(roi[n], type);
        const int roiH = static_cast<int>(b.h);
        const int roiW = static_cast<int>(b.w);
        for (Rpp32u c = 0; c < d.c; ++c) {
            const std::size_t base = static_cast<std::size_t>(n) * d.strides.nStride +
                                     static_cast<std::size_t>(c) * d.strides.cStride;
            for (int j = 0; j < roiH; ++j)
                for (int i = 0; i < roiW; ++i) {
                    gather_roi_window(src, d, b, base, j, i, r, window.data());
                    std::sort(window.begin(), window.end());
                    const std::size_t dstIdx =
                        base + static_cast<std::size_t>(j) * d.strides.hStride +
                        static_cast<std::size_t>(i) * d.strides.wStride;
                    dst[dstIdx] = from_double<T>(window[kk / 2]);
                }
        }
    }
}

}  // namespace rpptest

#endif  // RPP_TEST_MEDIAN_FILTER_REF_H
