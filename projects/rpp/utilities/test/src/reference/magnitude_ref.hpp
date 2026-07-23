#ifndef RPP_TEST_MAGNITUDE_REF_H
#define RPP_TEST_MAGNITUDE_REF_H

#include <rpp/rpp.h>

#include <cmath>

#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Independent host golden model for rppt_magnitude (two-source, no params), derived from the op's
// definition (out = sqrt(src1^2 + src2^2)), NOT from the kernel.
//
// Integer types round to nearest and combine in [0,255] intensity space; I8 pixels are the same
// intensities shifted by -128:
//   U8      : clamp[0,255]  ( round(sqrt(a^2 + b^2)) )
//   I8      : clamp[-128,127]( round(sqrt((a+128)^2 + (b+128)^2)) - 128 )
//   F16/F32 : clamp[0,1]    ( sqrt(a^2 + b^2) )
inline double magnitude_scalar(double a, double b, DType dt) {
    switch (dt) {
        case DType::U8:
            return clampd(std::nearbyint(std::sqrt(a * a + b * b)), 0.0, 255.0);
        case DType::I8: {
            const double a1 = a + 128.0, b1 = b + 128.0;
            return clampd(std::nearbyint(std::sqrt(a1 * a1 + b1 * b1)), 0.0, 255.0) - 128.0;
        }
        case DType::F16:
        case DType::F32:
            return clampd(std::sqrt(a * a + b * b), 0.0, 1.0);
    }
    return 0.0;
}

template <typename T>
void magnitude_reference(const T* src1, const T* src2, T* dst, const RpptDesc& d, DType dt,
                         const RpptROI* roi, RpptRoiType roiType) {
    for_each_roi_io(d, roi, roiType,
                    [&](Rpp32u, Rpp32u, Rpp32u, Rpp32u, std::size_t srcIdx, std::size_t dstIdx) {
                        dst[dstIdx] = from_double<T>(
                            magnitude_scalar(to_double(src1[srcIdx]), to_double(src2[srcIdx]), dt));
                    });
}

}  // namespace rpptest

#endif  // RPP_TEST_MAGNITUDE_REF_H
