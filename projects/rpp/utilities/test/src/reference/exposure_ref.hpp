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

#ifndef RPP_TEST_EXPOSURE_REF_H
#define RPP_TEST_EXPOSURE_REF_H

#include <rpp/rpp.h>

#include <cmath>

#include "framework/config_param.hpp"
#include "framework/intensity.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

/*
Reference model: exposure

RPP op
  rppt_exposure   (Image / Color augmentation)

Description
  Pointwise exposure adjustment in stops. Each channel of each pixel is scaled
  by 2^exposureFactor, so a factor of +1 doubles the intensity and -1 halves
  it, then clamped to the range of the pixel type.

Expression
  dst(x, y, c) = clamp( src(x, y, c) * 2^exposureFactor )

Per-type form
  Integer types work in [0,255] intensity space and round to nearest; I8 is
  the same intensity shifted -128. mult = 2^exposureFactor.

    U8      clamp[0,255]   ( round(v * mult) )
    I8      clamp[-128,127]( round((v + 128) * mult) - 128 )
    F16/F32 clamp[0,1]     ( v * mult )
*/
inline double exposure_scalar(double v, DType dt, double mult) {
    switch (dt) {
        case DType::U8:
            return clampd(std::nearbyint(v * mult), 0.0, 255.0);
        case DType::I8:
            return clampd(std::nearbyint((v + 128.0) * mult), 0.0, 255.0) - 128.0;
        case DType::F16:
        case DType::F32:
            return clampd(v * mult, 0.0, 1.0);
        default:
            return v;
    }
}

template <typename T>
void exposure_reference(const T* src, const RpptDesc& sd, T* dst, const RpptDesc& dd, DType dt,
                        const RpptROI* roi, RpptRoiType roiType, double exposureFactor) {
    const double mult = std::pow(2.0, exposureFactor);
    for_each_roi_io(sd, dd, roi, roiType,
                    [&](Rpp32u, Rpp32u, Rpp32u, Rpp32u, std::size_t srcIdx, std::size_t dstIdx) {
                        dst[dstIdx] =
                            from_double<T>(exposure_scalar(to_double(src[srcIdx]), dt, mult));
                    });
}

}  // namespace rpptest

#endif  // RPP_TEST_EXPOSURE_REF_H
