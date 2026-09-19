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

#ifndef RPP_TEST_CONTRAST_REF_H
#define RPP_TEST_CONTRAST_REF_H

#include <rpp/rpp.h>

#include <cmath>

#include "framework/config_param.hpp"
#include "framework/intensity.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

/*
Reference model: contrast

RPP op
  rppt_contrast   (Image / Color augmentation)

Description
  Pointwise contrast stretch about a fixed pivot. Each channel's distance from
  the contrast centre is scaled by factor and added back to the centre, so
  factor > 1 pushes intensities away from the centre and factor < 1 pulls them
  toward it. The centre itself is unchanged.

Expression
  dst(x, y, c) = clamp( (src(x, y, c) - centre) * factor + centre )

Per-type form
  centre is in [0,255] pixel units for every type. Integer types work in
  [0,255] intensity space and round to nearest; I8 is the same intensity
  shifted -128.

    U8    clamp[0,255]   ( round( (v - centre) * factor + centre ) )
    I8    clamp[-128,127]( round( ((v + 128) - centre) * factor + centre ) - 128 )
    F32   clamp[0,1]     ( (v - centre/255) * factor + centre/255 )
    F16   as F32, stored as half
*/
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
        default:
            return v;
    }
}

template <typename T>
void contrast_reference(const T* src, const RpptDesc& sd, T* dst, const RpptDesc& dd, DType dt,
                        const RpptROI* roi, RpptRoiType roiType, double factor, double center) {
    for_each_roi_io(sd, dd, roi, roiType,
                    [&](Rpp32u, Rpp32u, Rpp32u, Rpp32u, std::size_t srcIdx, std::size_t dstIdx) {
                        dst[dstIdx] = from_double<T>(
                            contrast_scalar(to_double(src[srcIdx]), dt, factor, center));
                    });
}

}  // namespace rpptest

#endif  // RPP_TEST_CONTRAST_REF_H
