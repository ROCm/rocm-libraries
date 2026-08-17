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
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Kernel-derived REGRESSION golden for rppt_color_jitter. Used for both backends (the op is HOST
// only today) and deterministic, so this is an exact model, not a property check.
//
// The public header cannot be used to derive this one. Its description and parameter ranges are
// copied verbatim from rppt_color_twist, which is a different algorithm; color_jitter is a colour
// transform matrix (CTM). That mismatch is a filed documentation defect, so the algorithm here is
// transcribed from the RPP kernel (src/modules/tensor/cpu/kernel/color_jitter.cpp) instead, with
// the user's explicit authorization. The transcription was validated numerically against the
// kernel across eight parameter settings: it reproduces every pixel to 4.7e-7 in F32.
//
// What this model deliberately does NOT reproduce are the three confirmed defects, so they stay
// red rather than being locked in:
//
//   1. The kernel applies the hue/saturation matrix TRANSPOSED. A saturation matrix is pinned by
//      two properties: at (saturation 1, hue 0) it must be the identity, and at saturation 0 it
//      must desaturate to the luma grey. The transposed form fails the second outright -- it maps
//      every pixel to (Wr, Wg, Wb) * (R+G+B), which depends only on the channel sum. This model
//      uses the correct orientation, which satisfies both exactly.
//   2. The kernel's saturation basis is Haeberli's published table, whose third row is rounded to
//      -0.300 / -0.588 rather than the -Wr / -Wg that would cancel the luma weights, so even the
//      neutral setting is off by 1e-3. This model derives that basis exactly as (I - L).
//   3. The kernel writes the brightness translation into the matrix's fourth row, where the
//      product annihilates it, so the parameter is dead. This model applies it.
//
// Semantics. With sch = saturation*cos(hue), ssh = saturation*sin(hue), L the luma matrix (every
// row = the Rec.601 weights) and B Haeberli's rotation generator:
//   hueSat = L + sch*(I - L) + ssh*B
//   out    = (contrast + 1) * (hueSat . rgb) + brightness
// The (contrast + 1) scale is the kernel's own convention, kept because it is the op's design
// intent -- that its neutral value is 0 while the header documents 0 < contrast <= 255 is the
// documentation defect, not something for the golden to reinterpret.
//
// Everything runs in normalized [0,1] intensity space (to_unit / from_unit), matching the rest of
// the suite. The 3x3 part is homogeneous, so this is identical to the kernel's per-dtype domain
// (raw 0..255 for U8, +128-shifted for I8, [0,1] for F16/F32); only the brightness translation is
// scale-dependent, and the kernel's translation is dead, so no test outcome rests on that choice.
// A single quantization closes the pipeline: U8 clamp[0,255](round(x*255)),
// I8 clamp[-128,127](round(x*255)-128), F16/F32 clamp[0,1](x).

namespace color_jitter_detail {

constexpr double kWr = 0.299, kWg = 0.587, kWb = 0.114;  // RGB_TO_GREY_WEIGHT_{RED,GREEN,BLUE}

// Haeberli's hue-rotation generator. Unlike the saturation basis this has no simpler closed form
// in the luma weights, so the published constants stand.
constexpr double kRot[9] = {0.168,  0.330, -0.497,   //
                            -0.328, 0.035, 0.292,    //
                            1.250,  -1.050, -0.203};

// Row-major 3x3: L + sch*(I - L) + ssh*kRot. Row c of the result produces output channel c.
inline void hue_saturation_matrix(double hueDeg, double satFactor, double m[9]) {
    const double rad = hueDeg * M_PI / 180.0;
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
inline void color_jitter_matrix(double brightness, double contrast, double hueDeg,
                                double satFactor, double m[9], double& translation) {
    color_jitter_detail::hue_saturation_matrix(hueDeg, satFactor, m);
    const double scale = contrast + 1.0;
    for (int i = 0; i < 9; ++i) m[i] *= scale;
    translation = brightness;
}

template <typename T>
void color_jitter_reference(const T* src, T* dst, const RpptDesc& d, DType dt, const RpptROI* roi,
                            RpptRoiType roiType, double brightness, double contrast, double hueDeg,
                            double satFactor) {
    double m[9], translation;
    color_jitter_matrix(brightness, contrast, hueDeg, satFactor, m, translation);
    for_each_roi_pixel(d, roi, roiType,
                       [&](Rpp32u, Rpp32u, Rpp32u, std::size_t srcPix, std::size_t dstPix) {
        if (d.c == 3) {
            double rgb[3];
            for (int c = 0; c < 3; ++c)
                rgb[c] = to_unit(to_double(src[channel_index(d, srcPix, c)]), dt);
            for (int c = 0; c < 3; ++c) {
                const double x =
                    m[c * 3] * rgb[0] + m[c * 3 + 1] * rgb[1] + m[c * 3 + 2] * rgb[2] + translation;
                dst[channel_index(d, dstPix, c)] = from_double<T>(from_unit(x, dt));
            }
        } else {
            // 1-channel: hue and saturation have no meaning on a single channel, and every row of
            // the hue/saturation matrix sums to 1, so a grey pixel reduces to the contrast scale
            // plus the translation whatever hue/saturation are set to.
            const double x =
                (contrast + 1.0) * to_unit(to_double(src[srcPix]), dt) + translation;
            dst[dstPix] = from_double<T>(from_unit(x, dt));
        }
    });
}

}  // namespace rpptest

#endif  // RPP_TEST_COLOR_JITTER_REF_H
