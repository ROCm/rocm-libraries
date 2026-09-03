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

#ifndef RPP_TEST_BLEND_REF_H
#define RPP_TEST_BLEND_REF_H

#include <rpp/rpp.h>

#include <cmath>

#include "framework/config_param.hpp"
#include "framework/intensity.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

/*
Reference model: blend

RPP op
  rppt_blend   (Image / Color augmentation)

Description
  Pointwise linear cross-fade between two co-located sources. Each output
  element is the interpolation of the two source elements at the same
  position, weighted by a per-image alpha: alpha = 1 yields src1, alpha = 0
  yields src2.

Expression
  dst(x, y, c) = clamp( alpha * src1 + (1 - alpha) * src2 )
               = clamp( (src1 - src2) * alpha + src2 )

Per-type form
  Integer types round to nearest. For I8 the +128 intensity offsets cancel in
  (src1 - src2), so the interpolation is identical in signed space.

    U8      clamp[0,255]   ( round((src1 - src2) * alpha + src2) )
    I8      clamp[-128,127]( round((src1 - src2) * alpha + src2) )
    F16/F32 clamp[0,1]     ( (src1 - src2) * alpha + src2 )
*/
inline double blend_scalar(double s1, double s2, DType dt, double alpha) {
    const double v = (s1 - s2) * alpha + s2;
    switch (dt) {
        case DType::U8:
            return clampd(std::nearbyint(v), 0.0, 255.0);
        case DType::I8:
            return clampd(std::nearbyint(v), -128.0, 127.0);
        case DType::F16:
        case DType::F32:
            return clampd(v, 0.0, 1.0);
        default:
            return v;
    }
}

template <typename T>
void blend_reference(const T* src1, const T* src2, const RpptDesc& sd, T* dst, const RpptDesc& dd,
                     DType dt, const RpptROI* roi, RpptRoiType roiType, double alpha) {
    for_each_roi_io(sd, dd, roi, roiType,
                    [&](Rpp32u, Rpp32u, Rpp32u, Rpp32u, std::size_t srcIdx, std::size_t dstIdx) {
                        dst[dstIdx] = from_double<T>(blend_scalar(
                            to_double(src1[srcIdx]), to_double(src2[srcIdx]), dt, alpha));
                    });
}

}  // namespace rpptest

#endif  // RPP_TEST_BLEND_REF_H
