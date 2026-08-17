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
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Independent host golden model for rppt_color_temperature, derived from the op's definition
// (a per-channel pixel adjustment: Red += adjustment, Blue -= adjustment, Green unchanged),
// NOT from the RPP kernel. Used as the reference for both backends so kernel bugs surface as
// diffs.
//
// adjustment is an integer "pixel adjustment value" in [0,255] intensity units (API range
// [-100,100]). It is a pure additive offset, so no rounding is involved for integer dtypes;
// I8 shares the same intensity scale as U8 (offset by -128), so the shift cancels:
//   U8  : clamp[0,255]  ( v + delta )
//   I8  : clamp[-128,127]( v + delta )
//   F32 : clamp[0,1]    ( v + delta/255 )
//   F16 : same as F32, stored as half
// where delta = +adjustment on Red (c==0), -adjustment on Blue (c==2), 0 on Green (c==1).
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

// Writes the color-temperature result into dst, reading the source at the ROI offset and
// writing packed at the destination origin (matching the region and placement the RPP op
// uses). dst outside the written region is left as the caller initialized it. Channel order
// is RGB: c==0 Red, c==1 Green, c==2 Blue.
template <typename T>
void color_temperature_reference(const T* src, T* dst, const RpptDesc& d, DType dt,
                                 const RpptROI* roi, RpptRoiType roiType, int adjustment) {
    for_each_roi_io(
        d, roi, roiType,
        [&](Rpp32u, Rpp32u c, Rpp32u, Rpp32u, std::size_t srcIdx, std::size_t dstIdx) {
            const double delta = c == 0 ? static_cast<double>(adjustment)
                                        : (c == 2 ? -static_cast<double>(adjustment) : 0.0);
            dst[dstIdx] = from_double<T>(color_temperature_scalar(to_double(src[srcIdx]), dt, delta));
        });
}

}  // namespace rpptest

#endif  // RPP_TEST_COLOR_TEMPERATURE_REF_H
