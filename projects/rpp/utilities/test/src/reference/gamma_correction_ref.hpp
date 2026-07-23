#ifndef RPP_TEST_GAMMA_CORRECTION_REF_H
#define RPP_TEST_GAMMA_CORRECTION_REF_H

#include <rpp/rpp.h>

#include <cmath>

#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Independent host golden model for rppt_gamma_correction, derived from the op's definition
// (out = (pixel / max)^gamma * max), NOT from the RPP kernel. Used as the reference for both
// backends so kernel bugs surface as diffs.
//
// Gamma is applied in normalized [0,1] intensity space (to_unit -> pow -> from_unit): integer
// types round to nearest, I8 pixels are the same intensities shifted by -128:
//   U8  : clamp[0,255]  ( round( (v/255)^gamma * 255 ) )
//   I8  : clamp[-128,127]( round( ((v+128)/255)^gamma * 255 ) - 128 )
//   F32 : clamp[0,1]    ( v^gamma )
//   F16 : same as F32, stored as half
inline double gamma_correction_scalar(double v, DType dt, double gamma) {
    return from_unit(std::pow(to_unit(v, dt), gamma), dt);
}

// Writes the gamma-correction result into dst, reading the source at the ROI offset and
// writing packed at the destination origin (matching the region and placement the RPP op
// uses). dst outside the written region is left as the caller initialized it.
template <typename T>
void gamma_correction_reference(const T* src, T* dst, const RpptDesc& d, DType dt,
                                const RpptROI* roi, RpptRoiType roiType, double gamma) {
    for_each_roi_io(d, roi, roiType,
                    [&](Rpp32u, Rpp32u, Rpp32u, Rpp32u, std::size_t srcIdx, std::size_t dstIdx) {
                        dst[dstIdx] =
                            from_double<T>(gamma_correction_scalar(to_double(src[srcIdx]), dt, gamma));
                    });
}

}  // namespace rpptest

#endif  // RPP_TEST_GAMMA_CORRECTION_REF_H
