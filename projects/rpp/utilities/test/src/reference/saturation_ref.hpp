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

#ifndef RPP_TEST_SATURATION_REF_H
#define RPP_TEST_SATURATION_REF_H

#include <rpp/rpp.h>

#include <cmath>

#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/color_hsv.hpp"

namespace rpptest {

// Independent host golden model for rppt_saturation, derived from the op's definition (a
// saturation scale in HSV space: RGB -> HSV, S = clamp(S * factor, 0, 1), HSV -> RGB), NOT
// from the RPP kernel. Used as the reference for both backends so kernel bugs surface as diffs.
//
// saturation is a 3-channel (RGB, ch0=R/ch1=G/ch2=B) op. The scale is computed on normalized
// [0,1] RGB; per dtype the pixel is normalized into [0,1], scaled, then written back:
//   U8  : v/255       -> scale -> clamp[0,255]  ( round(v'*255) )
//   I8  : (v+128)/255 -> scale -> clamp[-128,127]( round(v'*255) - 128 )
//   F32 : v           -> scale -> clamp[0,1]
//   F16 : same as F32, stored as half
// Grey pixels (R==G==B) have S==0 and are invariant to the scale.

// Writes the saturation result into dst, reading each source pixel's three channels at the ROI
// offset and writing packed at the destination origin (matching the region and placement the
// RPP op uses). dst outside the written region is left as the caller initialized it.
template <typename T>
void saturation_reference(const T* src, T* dst, const RpptDesc& d, DType dt, const RpptROI* roi,
                          RpptRoiType roiType, double factor) {
    for_each_roi_pixel(d, roi, roiType,
                       [&](Rpp32u, Rpp32u, Rpp32u, std::size_t srcPix, std::size_t dstPix) {
        double rgb[3];
        for (int c = 0; c < 3; ++c)
            rgb[c] = to_unit(to_double(src[channel_index(d, srcPix, c)]), dt);
        saturation_scale_rgb(rgb[0], rgb[1], rgb[2], factor);
        for (int c = 0; c < 3; ++c)
            dst[channel_index(d, dstPix, c)] = from_double<T>(from_unit(rgb[c], dt));
    });
}

}  // namespace rpptest

#endif  // RPP_TEST_SATURATION_REF_H
