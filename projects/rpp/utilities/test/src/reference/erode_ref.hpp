#ifndef RPP_TEST_ERODE_REF_H
#define RPP_TEST_ERODE_REF_H

#include <rpp/rpp.h>

#include <algorithm>

#include "framework/config_param.hpp"
#include "reference/filter_common.hpp"

namespace rpptest {

// Independent host golden model for rppt_erode, derived from the op's definition
// (grayscale morphological erosion: per-channel MIN over a flat KxK square window centered
// on each pixel, with clamp-to-edge border), NOT from the RPP kernel. Used as the reference
// for both HOST and HIP backends so kernel bugs surface as diffs.
//
// The window, its REPLICATE border and the src-at-ROI-offset / dst-at-origin placement come from
// filter_reference (reference/filter_common.hpp); erosion supplies only the MIN reduction. The min
// selects an existing source value (no arithmetic), so the result is bit-exact for every dtype.
template <typename T>
void erode_reference(const T* src, T* dst, const RpptDesc& d, DType /*dt*/, const RpptROI* roi,
                     RpptRoiType type, Rpp32u kernelSize) {
    filter_reference<T>(src, dst, d, roi, type, kernelSize,
                        [](const double* w, int kk) { return *std::min_element(w, w + kk); });
}

}  // namespace rpptest

#endif  // RPP_TEST_ERODE_REF_H
