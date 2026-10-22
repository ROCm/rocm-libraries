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

#ifndef RPP_TEST_COLOR_TEMPERATURE_REF_H
#define RPP_TEST_COLOR_TEMPERATURE_REF_H

#include <rpp/rpp.h>

#include "framework/config_param.hpp"
#include "framework/intensity.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

/*
Reference model: color_temperature

RPP op
  rppt_color_temperature   (Image / Color augmentation)

Description
  Pointwise warm/cool shift. Red is raised and blue is lowered by the same
  adjustment while green is left alone, so a positive adjustment warms the
  image and a negative one cools it. Channel order is RGB: c=0 Red, c=1
  Green, c=2 Blue.

Expression
  delta        = +adjustment (c=0), 0 (c=1), -adjustment (c=2)
  dst(x, y, c) = clamp( src(x, y, c) + delta )

Per-type form
  adjustment is an integer pixel adjustment in [0,255] intensity units (API
  range [-100,100]). It is a pure additive offset, so no rounding is involved
  for the integer types; I8 shares the U8 intensity scale, so the -128 shift
  cancels.

    U8    clamp[0,255]   ( v + delta )
    I8    clamp[-128,127]( v + delta )
    F32   clamp[0,1]     ( v + delta/255 )
    F16   as F32, stored as half
*/
inline double color_temperature_scalar(double v, DType dt, double delta) {
    switch (dt) {
        case DType::U8:
            return clampd(v + delta, 0.0, 255.0);
        case DType::I8:
            return clampd(v + delta, -128.0, 127.0);
        case DType::F16:
        case DType::F32:
            return clampd(v + delta / 255.0, 0.0, 1.0);
        default:
            return v;
    }
}

template <typename T>
void color_temperature_reference(const T* src, const RpptDesc& sd, T* dst, const RpptDesc& dd,
                                 DType dt, const RpptROI* roi, RpptRoiType roiType,
                                 int adjustment) {
    for_each_roi_io(
        sd, dd, roi, roiType,
        [&](Rpp32u, Rpp32u c, Rpp32u, Rpp32u, std::size_t srcIdx, std::size_t dstIdx) {
            const double delta = c == 0 ? static_cast<double>(adjustment)
                                        : (c == 2 ? -static_cast<double>(adjustment) : 0.0);
            dst[dstIdx] =
                from_double<T>(color_temperature_scalar(to_double(src[srcIdx]), dt, delta));
        });
}

}  // namespace rpptest

#endif  // RPP_TEST_COLOR_TEMPERATURE_REF_H
