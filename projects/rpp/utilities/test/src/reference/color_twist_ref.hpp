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

#ifndef RPP_TEST_COLOR_TWIST_REF_H
#define RPP_TEST_COLOR_TWIST_REF_H

#include <rpp/rpp.h>

#include <cmath>

#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/color_hsv.hpp"

namespace rpptest {

// Independent host golden model for rppt_color_twist, derived from the op's intended fused
// definition (hue rotate -> saturation scale -> brightness/contrast affine, all in normalized
// [0,1] continuous space with a single quantization at the end), NOT from the RPP kernel. Used
// as the reference for both backends so kernel bugs surface as diffs.
//
// Per dtype each channel is normalized into [0,1]: U8 v/255, I8 (v+128)/255, F16/F32 as-is.
// The three stages are carried through as doubles (no clamp/quantize between them):
//   Stage 1 (3-channel):    RGB->HSV, H = (H + hueDeg) mod 360, HSV->RGB
//   Stage 2 (3-channel):    RGB->HSV, S = clamp(S * satFactor, 0, 1), HSV->RGB
//   Stage 3 (per channel):  x = brightness * x + contrast/255
// A single quantization closes the pipeline: U8 clamp[0,255](round(x*255)),
// I8 clamp[-128,127](round(x*255)-128), F32/F16 clamp[0,1](x). For a 1-channel (PLN1) image
// hue/saturation are no-ops (S==0, no hue), so only the brightness/contrast affine is applied.

// Writes the color_twist result into dst, reading each source pixel at the ROI offset and
// writing packed at the destination origin (matching the region and placement the RPP op uses).
// dst outside the written region is left as the caller initialized it.
template <typename T>
void color_twist_reference(const T* src, T* dst, const RpptDesc& d, DType dt, const RpptROI* roi,
                           RpptRoiType roiType, double brightness, double contrast, double hueDeg,
                           double satFactor) {
    const double beta = contrast / 255.0;
    for_each_roi_pixel(d, roi, roiType,
                       [&](Rpp32u, Rpp32u, Rpp32u, std::size_t srcPix, std::size_t dstPix) {
        if (d.c == 3) {
            double rgb[3];
            for (int c = 0; c < 3; ++c)
                rgb[c] = to_unit(to_double(src[channel_index(d, srcPix, c)]), dt);
            hue_rotate_rgb(rgb[0], rgb[1], rgb[2], hueDeg);                    // Stage 1
            saturation_scale_rgb(rgb[0], rgb[1], rgb[2], satFactor);           // Stage 2
            for (int c = 0; c < 3; ++c) rgb[c] = brightness * rgb[c] + beta;  // Stage 3
            for (int c = 0; c < 3; ++c)
                dst[channel_index(d, dstPix, c)] = from_double<T>(from_unit(rgb[c], dt));
        } else {  // 1-channel: only the Stage 3 affine (hue/saturation are no-ops)
            const double x = brightness * to_unit(to_double(src[srcPix]), dt) + beta;
            dst[dstPix] = from_double<T>(from_unit(x, dt));
        }
    });
}

}  // namespace rpptest

#endif  // RPP_TEST_COLOR_TWIST_REF_H
