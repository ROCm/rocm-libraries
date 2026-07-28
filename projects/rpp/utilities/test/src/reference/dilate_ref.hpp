#ifndef RPP_TEST_DILATE_REF_H
#define RPP_TEST_DILATE_REF_H

#include <rpp/rpp.h>

#include <algorithm>

#include "framework/config_param.hpp"
#include "reference/filter_common.hpp"

namespace rpptest {

// Independent host golden model for rppt_dilate, derived from the op's definition
// (grayscale morphological dilation: per-channel MAX over a KxK flat square window,
// clamp-to-edge border), NOT from the RPP kernel. Used as the reference for both
// backends so kernel bugs surface as diffs.
//
// The window, its REPLICATE border and the src-at-ROI-offset / dst-at-origin placement come from
// filter_reference (reference/filter_common.hpp); dilation supplies only the MAX reduction. The max
// selects an existing source value (no arithmetic), so the result is bit-exact (tolerance 0).
template <typename T>
void dilate_reference(const T* src, T* dst, const RpptDesc& d, DType /*dt*/, const RpptROI* roi,
                      RpptRoiType type, Rpp32u kernelSize) {
    filter_reference<T>(src, dst, d, roi, type, kernelSize,
                        [](const double* w, int kk) { return *std::max_element(w, w + kk); });
}

}  // namespace rpptest

#endif  // RPP_TEST_DILATE_REF_H
