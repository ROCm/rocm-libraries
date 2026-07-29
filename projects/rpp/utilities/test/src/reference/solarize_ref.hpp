#ifndef RPP_TEST_SOLARIZE_REF_H
#define RPP_TEST_SOLARIZE_REF_H

#include <rpp/rpp.h>

#include <cmath>

#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Independent host golden model for rppt_solarize, derived from the op's definition (pixels at or
// above the threshold are inverted about the intensity range), NOT from the kernel. The inversion
// and threshold compare are exact integer operations, so no rounding of a product is involved.
//
// The normalized [0,1] threshold is scaled to the integer intensity range; I8 pixels invert in
// [0,255] intensity space (i = v + 128, inverted intensity 255 - i, back to signed = -1 - v):
//   U8      : T = round(threshold * 255); out = (v >= T)       ? (255 - v) : v
//   I8      : T = round(threshold * 255); out = (v + 128 >= T) ? (-1 - v)  : v
//   F16/F32 : out = (v >= threshold) ? (1 - v) : v
inline double solarize_scalar(double v, DType dt, double threshold) {
    switch (dt) {
        case DType::U8: {
            const double t = std::round(threshold * 255.0);
            return (v >= t) ? (255.0 - v) : v;
        }
        case DType::I8: {
            const double t = std::round(threshold * 255.0);
            return ((v + 128.0) >= t) ? (-1.0 - v) : v;
        }
        case DType::F16:
        case DType::F32:
            return (v >= threshold) ? (1.0 - v) : v;
        default:
            return v;
    }
}

template <typename T>
void solarize_reference(const T* src, T* dst, const RpptDesc& d, DType dt, const RpptROI* roi,
                        RpptRoiType roiType, double threshold) {
    for_each_roi_io(d, roi, roiType,
                    [&](Rpp32u, Rpp32u, Rpp32u, Rpp32u, std::size_t srcIdx, std::size_t dstIdx) {
                        dst[dstIdx] =
                            from_double<T>(solarize_scalar(to_double(src[srcIdx]), dt, threshold));
                    });
}

}  // namespace rpptest

#endif  // RPP_TEST_SOLARIZE_REF_H
