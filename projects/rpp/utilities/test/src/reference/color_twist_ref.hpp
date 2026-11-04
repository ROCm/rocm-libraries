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
#include "framework/intensity.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/color_hsv.hpp"

namespace rpptest {

/*
Reference model: color_twist

RPP op
  rppt_color_twist   (Image / Color augmentation)

Description
  Four colour adjustments fused into one pass: a hue rotation, a saturation
  scale, and a brightness/contrast affine, applied in that order. The whole
  pipeline is carried in continuous normalized space with no intermediate
  clamp or quantize, so only the final write rounds. For a 1-channel (PLN1)
  image the hue and saturation stages are no-ops (S == 0, no hue), leaving
  just the affine.

Expression
  Stage 1 (3-channel)   RGB->HSV, H = (H + hueDeg) mod 360, HSV->RGB
  Stage 2 (3-channel)   RGB->HSV, S = clamp(S * satFactor, 0, 1), HSV->RGB
  Stage 3 (per channel) x = brightness * x + contrast/255

Per-type form
  Each channel is normalized into [0,1] on the way in (U8 v/255, I8
  (v+128)/255, F16/F32 as-is), and a single quantization closes the pipeline.

    U8    clamp[0,255]   ( round(x*255) )
    I8    clamp[-128,127]( round(x*255) - 128 )
    F32   clamp[0,1]     ( x )
    F16   as F32, stored as half
*/
template <typename T>
void color_twist_reference(const T* src, const RpptDesc& sd, T* dst, const RpptDesc& dd, DType dt,
                           const RpptROI* roi, RpptRoiType roiType, double brightness,
                           double contrast, double hueDeg, double satFactor) {
    const double beta = contrast / 255.0;
    for_each_roi_pixel(
        sd, dd, roi, roiType, [&](Rpp32u, Rpp32u, Rpp32u, std::size_t srcPix, std::size_t dstPix) {
            if (sd.c == 3) {
                double rgb[3];
                for (int c = 0; c < 3; ++c)
                    rgb[c] = to_unit(to_double(src[channel_index(sd, srcPix, c)]), dt);
                hue_rotate_rgb(rgb[0], rgb[1], rgb[2], hueDeg);                   // Stage 1
                saturation_scale_rgb(rgb[0], rgb[1], rgb[2], satFactor);          // Stage 2
                for (int c = 0; c < 3; ++c) rgb[c] = brightness * rgb[c] + beta;  // Stage 3
                for (int c = 0; c < 3; ++c)
                    dst[channel_index(dd, dstPix, c)] = from_double<T>(from_unit(rgb[c], dt));
            } else {  // 1-channel: only the Stage 3 affine (hue/saturation are no-ops)
                const double x = brightness * to_unit(to_double(src[srcPix]), dt) + beta;
                dst[dstPix] = from_double<T>(from_unit(x, dt));
            }
        });
}

}  // namespace rpptest

#endif  // RPP_TEST_COLOR_TWIST_REF_H
