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
// Median is a rank filter (not a convolution): the window filter_reference gathers (same
// clamp-to-ROI REPLICATE border as the other filters) is sorted and the middle element selected.
// kernelSize is odd so kernelSize*kernelSize is odd and the median is an EXISTING pixel value (no
// arithmetic), so to_double/from_double round-trips exactly for every dtype and the result is
// bit-exact (tolerance 0).
template <typename T>
void median_filter_reference(const T* src, T* dst, const RpptDesc& d, DType /*dt*/,
                             const RpptROI* roi, RpptRoiType type, Rpp32u kernelSize) {
    filter_reference<T>(src, dst, d, roi, type, kernelSize, [](double* w, int kk) {
        std::sort(w, w + kk);
        return w[kk / 2];
    });
}

}  // namespace rpptest

#endif  // RPP_TEST_MEDIAN_FILTER_REF_H
