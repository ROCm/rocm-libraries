/*
MIT License

Copyright (c) 2026 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef RPP_TEST_PHASE_REF_H
#define RPP_TEST_PHASE_REF_H

#include <rpp/rpp.h>

#include <cmath>

#include "framework/config_param.hpp"
#include "framework/intensity.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

/*
Reference model: phase

RPP op
  rppt_phase   (Image / Geometric augmentation)

Description
  Element-wise phase angle of the vector (src2, src1) formed by two co-located
  sources. Takes no parameters. The angle is normalized so that a
  first-quadrant angle [0, pi/2] fills the whole output range.

Expression
  dst(x, y, c) = (2/pi) * atan2( src1, src2 )

Per-type form
  atan2 is scale-invariant in a common positive factor, so integers may be
  combined in stored [0,255] space directly; I8 is shifted into that space
  first.

    U8      clamp[0,255]   ( round( (2/pi) * atan2(a, b)         * 255 ) )
    I8      clamp[-128,127]( round( (2/pi) * atan2(a+128, b+128) * 255 ) - 128 )
    F16/F32 clamp[0,1]     (        (2/pi) * atan2(a, b) )
*/
inline double phase_scalar(double a, double b, DType dt) {
    constexpr double kTwoOverPi = 0.63661977236758134308;  // 2 / pi
    switch (dt) {
        case DType::U8:
            return clampd(std::nearbyint(kTwoOverPi * std::atan2(a, b) * 255.0), 0.0, 255.0);
        case DType::I8:
            return clampd(std::nearbyint(kTwoOverPi * std::atan2(a + 128.0, b + 128.0) * 255.0),
                          0.0, 255.0) -
                   128.0;
        case DType::F16:
        case DType::F32:
            return clampd(kTwoOverPi * std::atan2(a, b), 0.0, 1.0);
        default:
            return 0.0;
    }
}

template <typename T>
void phase_reference(const T* src1, const T* src2, const RpptDesc& sd, T* dst, const RpptDesc& dd,
                     DType dt, const RpptROI* roi, RpptRoiType roiType) {
    for_each_roi_io(sd, dd, roi, roiType,
                    [&](Rpp32u, Rpp32u, Rpp32u, Rpp32u, std::size_t srcIdx, std::size_t dstIdx) {
                        dst[dstIdx] = from_double<T>(
                            phase_scalar(to_double(src1[srcIdx]), to_double(src2[srcIdx]), dt));
                    });
}

}  // namespace rpptest

#endif  // RPP_TEST_PHASE_REF_H
