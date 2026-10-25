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

#ifndef RPP_TEST_COLOR_JITTER_REF_H
#define RPP_TEST_COLOR_JITTER_REF_H

#include <rpp/rpp.h>

#include <cmath>
#include <cstddef>

#include "framework/config_param.hpp"
#include "framework/intensity.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

/*
Reference model: color_jitter

RPP op
  rppt_color_jitter   (Image / Color augmentation; HOST only)

Description
  A colour transform matrix (CTM) applying hue rotation, saturation, contrast
  and brightness in a single 3x3 product.

Expression
  With sch = saturation*cos(hue), ssh = saturation*sin(hue), L the luma matrix
  (every row = the Rec.601 weights) and B Haeberli's rotation generator:

  hueSat = L + sch*(I - L) + ssh*B
  dst    = (contrast + 1) * (hueSat . rgb) + brightness

  The (contrast + 1) scale follows the kernel, so the neutral contrast is 0
  even though the header documents 0 < contrast <= 255.

Per-type form
  Computed in normalized [0,1] intensity space with a single quantization at
  the end.

    U8    clamp[0,255]   ( round(x*255) )
    I8    clamp[-128,127]( round(x*255) - 128 )
    F16   clamp[0,1]
    F32   clamp[0,1]

Notes
  The header's description is copied from rppt_color_twist, a different
  algorithm, so this model follows src/modules/tensor/cpu/kernel/color_jitter.cpp
  except for three kernel defects it deliberately does not reproduce:

  1. The kernel applies the hue/saturation matrix transposed, so saturation 0
     does not desaturate to the luma grey.
  2. The kernel's saturation basis uses Haeberli's rounded table instead of
     (I - L), so even the neutral setting is off by 1e-3.
  3. The kernel writes the brightness translation into a row the product
     ignores, so brightness has no effect. This model applies it.
*/

namespace color_jitter_detail {

constexpr double kWr = 0.299, kWg = 0.587, kWb = 0.114;  // RGB_TO_GREY_WEIGHT_{RED,GREEN,BLUE}

// Haeberli's hue-rotation generator (published constants).
constexpr double kRot[9] = {0.168,  0.330,  -0.497,  //
                            -0.328, 0.035,  0.292,   //
                            1.250,  -1.050, -0.203};

// Row-major 3x3: L + sch*(I - L) + ssh*kRot. Row c of the result produces output channel c.
inline void hue_saturation_matrix(double hueDeg, double satFactor, double m[9]) {
    const double rad = hueDeg * kPi / 180.0;
    const double sch = satFactor * std::cos(rad);
    const double ssh = satFactor * std::sin(rad);
    const double luma[9] = {kWr, kWg, kWb, kWr, kWg, kWb, kWr, kWg, kWb};
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) {
            const int i = r * 3 + c;
            const double identity = (r == c) ? 1.0 : 0.0;
            m[i] = luma[i] + sch * (identity - luma[i]) + ssh * kRot[i];
        }
}

}  // namespace color_jitter_detail

// The full colour transform: `m` is the 3x3 applied to RGB, `translation` the additive term.
inline void color_jitter_matrix(double brightness, double contrast, double hueDeg, double satFactor,
                                double m[9], double& translation) {
    color_jitter_detail::hue_saturation_matrix(hueDeg, satFactor, m);
    const double scale = contrast + 1.0;
    for (int i = 0; i < 9; ++i) m[i] *= scale;
    translation = brightness;
}

template <typename T>
void color_jitter_reference(const T* src, const RpptDesc& sd, T* dst, const RpptDesc& dd, DType dt,
                            const RpptROI* roi, RpptRoiType roiType, double brightness,
                            double contrast, double hueDeg, double satFactor) {
    double m[9], translation;
    color_jitter_matrix(brightness, contrast, hueDeg, satFactor, m, translation);
    for_each_roi_pixel(
        sd, dd, roi, roiType, [&](Rpp32u, Rpp32u, Rpp32u, std::size_t srcPix, std::size_t dstPix) {
            if (sd.c == 3) {
                double rgb[3];
                for (int c = 0; c < 3; ++c)
                    rgb[c] = to_unit(to_double(src[channel_index(sd, srcPix, c)]), dt);
                for (int c = 0; c < 3; ++c) {
                    const double x = m[c * 3] * rgb[0] + m[c * 3 + 1] * rgb[1] +
                                     m[c * 3 + 2] * rgb[2] + translation;
                    dst[channel_index(dd, dstPix, c)] = from_double<T>(from_unit(x, dt));
                }
            } else {
                // 1-channel: every hue/saturation matrix row sums to 1, so only contrast and
                // brightness apply.
                const double x =
                    (contrast + 1.0) * to_unit(to_double(src[srcPix]), dt) + translation;
                dst[dstPix] = from_double<T>(from_unit(x, dt));
            }
        });
}

}  // namespace rpptest

#endif  // RPP_TEST_COLOR_JITTER_REF_H
