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

#ifndef RPP_TEST_COLOR_HSV_H
#define RPP_TEST_COLOR_HSV_H

#include <algorithm>
#include <cmath>

#include "framework/tensor_setup.hpp"

namespace rpptest {

// Shared RGB<->HSV building blocks for the color golden models (hue, saturation, color_twist).
// These encode the standard sextant conversion on normalized [0,1] channels, derived from the
// HSV definition (NOT any RPP kernel), so every op that manipulates hue/saturation rounds the
// same way and a single reviewed copy backs them all.

// Standard RGB->HSV sextant conversion on normalized [0,1] channels. H in [0,360), S,V in [0,1].
inline void rgb_to_hsv(double r, double g, double b, double& h, double& s, double& v) {
    const double cmax = std::max({r, g, b});
    const double cmin = std::min({r, g, b});
    const double delta = cmax - cmin;

    if (delta <= 0.0)   h = 0.0;
    else if (cmax == r) h = 60.0 * std::fmod((g - b) / delta, 6.0);
    else if (cmax == g) h = 60.0 * (((b - r) / delta) + 2.0);
    else                h = 60.0 * (((r - g) / delta) + 4.0);
    if (h < 0.0) h += 360.0;

    s = (cmax <= 0.0) ? 0.0 : delta / cmax;
    v = cmax;
}

// Standard HSV->RGB sextant conversion, the inverse of rgb_to_hsv.
inline void hsv_to_rgb(double h, double s, double v, double& r, double& g, double& b) {
    const double c = v * s;
    const double hp = h / 60.0;
    const double x = c * (1.0 - std::fabs(std::fmod(hp, 2.0) - 1.0));
    const double m = v - c;
    double rp, gp, bp;
    switch (static_cast<int>(hp)) {
        case 0:  rp = c; gp = x; bp = 0.0; break;
        case 1:  rp = x; gp = c; bp = 0.0; break;
        case 2:  rp = 0.0; gp = c; bp = x; break;
        case 3:  rp = 0.0; gp = x; bp = c; break;
        case 4:  rp = x; gp = 0.0; bp = c; break;
        default: rp = c; gp = 0.0; bp = x; break;  // hp in [5,6)
    }
    r = rp + m;
    g = gp + m;
    b = bp + m;
}

// Rotates the hue of a normalized [0,1] RGB triplet by hueDeg degrees (in-place).
inline void hue_rotate_rgb(double& r, double& g, double& b, double hueDeg) {
    double h, s, v;
    rgb_to_hsv(r, g, b, h, s, v);
    h = std::fmod(h + hueDeg, 360.0);
    if (h < 0.0) h += 360.0;
    hsv_to_rgb(h, s, v, r, g, b);
}

// Scales the saturation of a normalized [0,1] RGB triplet by factor, clamped to [0,1] (in-place).
inline void saturation_scale_rgb(double& r, double& g, double& b, double factor) {
    double h, s, v;
    rgb_to_hsv(r, g, b, h, s, v);
    s = clampd(s * factor, 0.0, 1.0);
    hsv_to_rgb(h, s, v, r, g, b);
}

}  // namespace rpptest

#endif  // RPP_TEST_COLOR_HSV_H
