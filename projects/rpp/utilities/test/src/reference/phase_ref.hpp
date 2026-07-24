#ifndef RPP_TEST_PHASE_REF_H
#define RPP_TEST_PHASE_REF_H

#include <rpp/rpp.h>

#include <cmath>

#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Independent host golden model for rppt_phase (two-source, no params), derived from the op's
// definition (the phase angle of the vector (src2, src1) = atan2(src1, src2)), NOT from the kernel.
//
// The angle is normalized so a first-quadrant angle [0, pi/2] fills the whole output range: the
// stored result is (2/pi) * atan2(src1, src2) expressed in [0,1] unit intensity, then quantized to
// the dtype (integers round to nearest, I8 pixels are the same intensities shifted by -128). atan2
// is scale-invariant in a common positive factor, so integers may be combined in stored [0,255]
// space directly; I8 is shifted into that space first:
//   U8      : round( (2/pi) * atan2(a, b)            * 255 ),          clamp[0,255]
//   I8      : round( (2/pi) * atan2(a+128, b+128)    * 255 ) - 128,    clamp[-128,127]
//   F16/F32 :        (2/pi) * atan2(a, b),                            clamp[0,1]
inline double phase_scalar(double a, double b, DType dt) {
    constexpr double kTwoOverPi = 0.63661977236758134308;  // 2 / pi
    switch (dt) {
        case DType::U8:
            return clampd(std::nearbyint(kTwoOverPi * std::atan2(a, b) * 255.0), 0.0, 255.0);
        case DType::I8:
            return clampd(std::nearbyint(kTwoOverPi * std::atan2(a + 128.0, b + 128.0) * 255.0), 0.0,
                          255.0) -
                   128.0;
        case DType::F16:
        case DType::F32:
            return clampd(kTwoOverPi * std::atan2(a, b), 0.0, 1.0);
    }
    return 0.0;
}

template <typename T>
void phase_reference(const T* src1, const T* src2, T* dst, const RpptDesc& d, DType dt,
                     const RpptROI* roi, RpptRoiType roiType) {
    for_each_roi_io(d, roi, roiType,
                    [&](Rpp32u, Rpp32u, Rpp32u, Rpp32u, std::size_t srcIdx, std::size_t dstIdx) {
                        dst[dstIdx] = from_double<T>(
                            phase_scalar(to_double(src1[srcIdx]), to_double(src2[srcIdx]), dt));
                    });
}

}  // namespace rpptest

#endif  // RPP_TEST_PHASE_REF_H
