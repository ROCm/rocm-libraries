#ifndef RPP_TEST_CROP_REF_H
#define RPP_TEST_CROP_REF_H

#include <rpp/rpp.h>

#include <cstddef>

#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Independent host golden model for rppt_crop, derived from the op's definition (crop each image
// to its ROI: the destination is the source region at the ROI offset, written packed at the
// destination origin), NOT from the RPP kernel. Used as the reference for both backends so kernel
// bugs surface as diffs.
//
// crop has no scalar params and does no arithmetic, rounding, or clamping: each cropped element is
// copied verbatim, so the result is bit-exact for U8/I8/F16/F32 alike. for_each_roi_io() encodes
// crop's src-at-ROI-offset -> dst-at-origin placement (the same mapping every ROI op shares).
template <typename T>
void crop_reference(const T* src, T* dst, const RpptDesc& d, const RpptROI* roi,
                    RpptRoiType roiType) {
    for_each_roi_io(d, roi, roiType,
                    [&](Rpp32u, Rpp32u, Rpp32u, Rpp32u, std::size_t srcIdx, std::size_t dstIdx) {
                        dst[dstIdx] = src[srcIdx];
                    });
}

}  // namespace rpptest

#endif  // RPP_TEST_CROP_REF_H
