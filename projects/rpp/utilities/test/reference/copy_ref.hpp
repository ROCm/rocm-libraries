#ifndef RPP_TEST_COPY_REF_H
#define RPP_TEST_COPY_REF_H

#include <rpp/rpp.h>

#include <cstddef>

#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Independent host golden model for rppt_copy, derived from the op's definition (dst = src, a
// bit-exact buffer copy of every element), NOT from the RPP kernel. Used as the reference for
// both backends so kernel bugs surface as diffs.
//
// copy takes no ROI and no scalar params: every dtype is copied verbatim with no arithmetic,
// rounding, or clamping, so the result is bit-exact for U8/I8/F16/F32 alike.
template <typename T>
void copy_reference(const T* src, T* dst, const RpptDesc& d, const RpptROI* roi,
                    RpptRoiType roiType) {
    for_each_roi_io(d, roi, roiType,
                    [&](Rpp32u, Rpp32u, Rpp32u, Rpp32u, std::size_t srcIdx, std::size_t dstIdx) {
                        dst[dstIdx] = src[srcIdx];
                    });
}

}  // namespace rpptest

#endif  // RPP_TEST_COPY_REF_H
