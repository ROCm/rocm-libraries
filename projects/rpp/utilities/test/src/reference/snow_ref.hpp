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

#ifndef RPP_TEST_SNOW_REF_H
#define RPP_TEST_SNOW_REF_H

#include <rpp/rpp.h>

#include <algorithm>
#include <cmath>

#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Kernel-derived REGRESSION golden for rppt_snow.
//
// Unlike the other references in this suite, snow's exact per-element semantics are NOT
// documented in the public API header (it describes only the parameters), and there is no
// spec. This model is therefore transcribed from the RPP snow kernel
// (src/modules/tensor/cpu/kernel/snow.cpp) with the user's explicit authorization. It LOCKS
// current behavior (a regression test) rather than encoding independently-derived intent,
// and deliberately mirrors the kernel's quirks -- notably that the I8 store truncates toward
// zero (no round-to-nearest) -- so the grid stays green until behavior changes. snow is fully
// deterministic (no RNG), so the same model serves both HOST and HIP.
//
// Semantics (per pixel, in normalized [0,1] intensity space):
//   snowCoefficient = fmaf(snowThreshold, 0.5, 0.333333333)
//   3-channel: RGB -> HSL; (darkMode) boost L in [0,0.39215686]; if L <= snowCoefficient and
//     the pixel is NOT in the excluded blue/white band (hue in [0.514,0.63], sat >= 0.196,
//     L >= 0.196), scale L by brightnessCoefficient; HSL -> RGB.
//   1-channel: treat the intensity directly as L and apply the same L adjustments (no band).
// Per dtype: U8/I8 normalize to [0,1] on load ( U8: v/255, I8: (v+128)/255 ) and denormalize
//   on store ( U8: round(x*255) clamp[0,255]; I8: trunc(x*255-128) clamp[-128,127] ); F16/F32
//   compute in place and clamp to [0,1].

namespace snow_detail {

constexpr float kOneOver255 = 0.00392156862745f;  // matches the kernel's ONE_OVER_255
constexpr float kHueLower = 0.514f;               // SNOW_HUE_LOWER_BOUND
constexpr float kHueUpper = 0.63f;                // SNOW_HUE_UPPER_BOUND
constexpr float kSatThreshold = 0.196f;           // SNOW_SAT_THRESHOLD
constexpr float kLightThreshold = 0.196f;         // SNOW_LIGHTNESS_THRESHOLD
constexpr float kUpperThreshold = 0.39215686f;    // dark-mode lightness upper bound (100/255)
constexpr float kBrightnessFactor = 2.5f;         // dark-mode brightness factor

// 3-channel snow: mirrors compute_snow_host().
inline void snow_rgb(float& R, float& G, float& B, float brightnessCoefficient,
                     float snowCoefficient, int darkMode) {
    const float rf = R, gf = G, bf = B;
    const float cmax = std::max(rf, std::max(gf, bf));
    const float cmin = std::min(rf, std::min(gf, bf));
    const float delta = cmax - cmin;
    float hue = 0.0f, sat = 0.0f, add = 0.0f;
    float l = (cmax + cmin) * 0.5f;
    if (delta != 0.0f) {
        sat = (l <= 0.5f) ? delta / (cmax + cmin) : delta / (2.0f - (cmax + cmin));
        if (cmax == rf) {
            hue = gf - bf;
            add = 0.0f;
        } else if (cmax == gf) {
            hue = bf - rf;
            add = 2.0f;
        } else {
            hue = rf - gf;
            add = 4.0f;
        }
        hue /= delta;
        hue += add;
        hue /= 6.0f;
    }
    // Modify lightness
    if (l >= 0.0f && l <= kUpperThreshold && darkMode == 1)
        l = l * std::fmaf(-l / kUpperThreshold, kBrightnessFactor - 1.0f, kBrightnessFactor);
    if (l <= snowCoefficient && !((hue >= kHueLower && hue <= kHueUpper) &&
                                  (sat >= kSatThreshold) && (l >= kLightThreshold)))
        l = l * brightnessCoefficient;

    // HSL -> RGB with brightness/contrast adjustment
    float hc[3];
    hc[0] = 6.0f * (hue - TWO_OVER_3);
    hc[1] = 0.0f;
    hc[2] = 6.0f * (1.0f - hue);
    if (hue < TWO_OVER_3) {
        hc[0] = 0.0f;
        hc[1] = 6.0f * (TWO_OVER_3 - hue);
        hc[2] = 6.0f * (hue - ONE_OVER_3);
    }
    if (hue < ONE_OVER_3) {
        hc[0] = 6.0f * (ONE_OVER_3 - hue);
        hc[1] = 6.0f * hue;
        hc[2] = 0.0f;
    }
    hc[0] = std::min(hc[0], 1.0f);
    hc[1] = std::min(hc[1], 1.0f);
    hc[2] = std::min(hc[2], 1.0f);

    const float sat2 = 2.0f * sat;
    const float satInv = 1.0f - sat;
    const float lumInv = 1.0f - l;
    const float lum2m1 = (2.0f * l) - 1.0f;
    hc[0] = (sat2 * hc[0]) + satInv;
    hc[1] = (sat2 * hc[1]) + satInv;
    hc[2] = (sat2 * hc[2]) + satInv;
    if (l >= 0.5f) {
        hc[0] = (lumInv * hc[0]) + lum2m1;
        hc[1] = (lumInv * hc[1]) + lum2m1;
        hc[2] = (lumInv * hc[2]) + lum2m1;
    } else {
        hc[0] *= l;
        hc[1] *= l;
        hc[2] *= l;
    }
    R = hc[0];
    G = hc[1];
    B = hc[2];
}

// 1-channel snow: mirrors compute_snow_host_gray().
inline void snow_gray(float& pixel, float brightnessCoefficient, float snowCoefficient,
                      int darkMode) {
    if (darkMode == 1 && pixel >= 0.0f && pixel <= kUpperThreshold)
        pixel *= std::fmaf(-pixel / kUpperThreshold, kBrightnessFactor - 1.0f, kBrightnessFactor);
    if (pixel <= snowCoefficient) pixel *= brightnessCoefficient;
}

inline float load_norm(double v, DType dt) {
    switch (dt) {
        case DType::U8: return static_cast<float>(v) * kOneOver255;
        case DType::I8: return (static_cast<float>(v) + 128.0f) * kOneOver255;
        default:        return static_cast<float>(v);  // F16/F32 already [0,1]
    }
}

inline double store_denorm(float x, DType dt) {
    switch (dt) {
        case DType::U8: return clampd(std::nearbyint(static_cast<double>(x) * 255.0), 0.0, 255.0);
        // I8 truncates toward zero on the static_cast (no round) -- the kernel's behavior.
        case DType::I8: return clampd(static_cast<double>(x * 255.0f - 128.0f), -128.0, 127.0);
        default:        return clampd(static_cast<double>(x), 0.0, 1.0);  // F16/F32
    }
}

}  // namespace snow_detail

// snowThreshold is the raw API parameter (mapped to snowCoefficient internally).
template <typename T>
void snow_reference(const T* src, T* dst, const RpptDesc& d, DType dt, const RpptROI* roi,
                    RpptRoiType roiType, double brightnessCoefficient, double snowThreshold,
                    int darkMode) {
    const float bc = static_cast<float>(brightnessCoefficient);
    const float snowCoefficient =
        std::fmaf(static_cast<float>(snowThreshold), 0.5f, 0.333333333f);

    if (d.c == 3) {
        for_each_roi_pixel(
            d, roi, roiType, [&](Rpp32u, Rpp32u, Rpp32u, std::size_t srcPix, std::size_t dstPix) {
                float rgb[3];
                for (Rpp32u c = 0; c < 3; ++c)
                    rgb[c] = snow_detail::load_norm(to_double(src[channel_index(d, srcPix, c)]), dt);
                snow_detail::snow_rgb(rgb[0], rgb[1], rgb[2], bc, snowCoefficient, darkMode);
                for (Rpp32u c = 0; c < 3; ++c)
                    dst[channel_index(d, dstPix, c)] =
                        from_double<T>(snow_detail::store_denorm(rgb[c], dt));
            });
    } else {  // PLN1
        for_each_roi_io(
            d, roi, roiType,
            [&](Rpp32u, Rpp32u, Rpp32u, Rpp32u, std::size_t srcIdx, std::size_t dstIdx) {
                float p = snow_detail::load_norm(to_double(src[srcIdx]), dt);
                snow_detail::snow_gray(p, bc, snowCoefficient, darkMode);
                dst[dstIdx] = from_double<T>(snow_detail::store_denorm(p, dt));
            });
    }
}

}  // namespace rpptest

#endif  // RPP_TEST_SNOW_REF_H
