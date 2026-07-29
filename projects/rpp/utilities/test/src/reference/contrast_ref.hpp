#ifndef RPP_TEST_CONTRAST_REF_H
#define RPP_TEST_CONTRAST_REF_H

#include <rpp/rpp.h>

#include <cmath>

#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Independent host golden model for rppt_contrast, derived from the op's definition
// (out = (pixel - center) * factor + center), NOT from the RPP kernel. Used as the
// reference for both backends so kernel bugs surface as diffs.
//
// center is expressed in [0,255] pixel units. Integer types work in [0,255] intensity
// space and round to nearest; I8 pixels are the same intensities shifted by -128:
//   U8  : clamp[0,255]  ( round( (v - center) * factor + center ) )
//   I8  : clamp[-128,127]( round( ((v + 128) - center) * factor + center ) - 128 )
//   F32 : clamp[0,1]    ( (v - center/255) * factor + center/255 )
//   F16 : same as F32, stored as half
inline double contrast_scalar(double v, DType dt, double factor, double center) {
    switch (dt) {
        case DType::U8:
            return clampd(std::nearbyint((v - center) * factor + center), 0.0, 255.0);
        case DType::I8:
            return clampd(std::nearbyint(((v + 128.0) - center) * factor + center) - 128.0, -128.0,
                          127.0);
        case DType::F16:
        case DType::F32: {
            const double c = center / 255.0;
            return clampd((v - c) * factor + c, 0.0, 1.0);
        }
        default: return v;
    }
}

// Writes the contrast result into dst, reading the source at the ROI offset and writing
// packed at the destination origin (matching the region and placement the RPP op uses).
// dst outside the written region is left as the caller initialized it.
template <typename T>
void contrast_reference(const T* src, T* dst, const RpptDesc& d, DType dt, const RpptROI* roi,
                        RpptRoiType roiType, double factor, double center) {
    for_each_roi_io(d, roi, roiType,
                    [&](Rpp32u, Rpp32u, Rpp32u, Rpp32u, std::size_t srcIdx, std::size_t dstIdx) {
                        dst[dstIdx] = from_double<T>(
                            contrast_scalar(to_double(src[srcIdx]), dt, factor, center));
                    });
}

}  // namespace rpptest

#endif  // RPP_TEST_CONTRAST_REF_H
