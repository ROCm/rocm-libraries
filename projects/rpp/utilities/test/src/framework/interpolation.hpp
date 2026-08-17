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

#ifndef RPP_TEST_INTERPOLATION_H
#define RPP_TEST_INTERPOLATION_H

#include <rpp/rpp.h>

#include <cassert>
#include <cmath>
#include <cstddef>

#include "framework/tensor_setup.hpp"

namespace rpptest {

// Shared, op-agnostic source sampling for the geometric golden models. Every "move pixels around"
// op inverse-maps an output coordinate to a source coordinate and samples the source there; this
// header is the single implementation of that sampling (nearest / bilinear) so no two op
// references reimplement it and diverge.
//
// Coordinates are absolute within one image-channel plane whose element (0,0) sits at `base`. The
// valid source region is the half-open rectangle [x0,x1) x [y0,y1) (for ROI ops this is the ROI
// rectangle); samples outside it return `border` (the dtype's black, in stored units).

template <typename T>
inline double src_texel(const T* src, const RpptDesc& d, std::size_t base, int x, int y, int x0,
                        int y0, int x1, int y1, double border) {
    if (x < x0 || y < y0 || x >= x1 || y >= y1) return border;
    return to_double(src[plane_index(d, base, static_cast<std::size_t>(y),
                                     static_cast<std::size_t>(x))]);
}

// Samples the plane at fractional (x,y) (texel centers at integer coords) using `interp`.
// Interpolation runs directly on stored values: it is affine, so it commutes with the U8/I8
// intensity offset and needs no unit conversion. Only NEAREST_NEIGHBOR and BILINEAR are
// implemented; the filtered modes (BICUBIC/LANCZOS/GAUSSIAN/TRIANGULAR) are added when an op that
// needs them is ported.
template <typename T>
inline double sample(const T* src, const RpptDesc& d, std::size_t base, double x, double y, int x0,
                     int y0, int x1, int y1, RpptInterpolationType interp, double border) {
    switch (interp) {
        case NEAREST_NEIGHBOR: {
            const int xi = static_cast<int>(std::floor(x + 0.5));
            const int yi = static_cast<int>(std::floor(y + 0.5));
            return src_texel(src, d, base, xi, yi, x0, y0, x1, y1, border);
        }
        case BILINEAR: {
            const double fx = std::floor(x), fy = std::floor(y);
            const int xa = static_cast<int>(fx), ya = static_cast<int>(fy);
            const double dx = x - fx, dy = y - fy;
            const double v00 = src_texel(src, d, base, xa, ya, x0, y0, x1, y1, border);
            const double v01 = src_texel(src, d, base, xa + 1, ya, x0, y0, x1, y1, border);
            const double v10 = src_texel(src, d, base, xa, ya + 1, x0, y0, x1, y1, border);
            const double v11 = src_texel(src, d, base, xa + 1, ya + 1, x0, y0, x1, y1, border);
            return v00 * (1.0 - dx) * (1.0 - dy) + v01 * dx * (1.0 - dy) +
                   v10 * (1.0 - dx) * dy + v11 * dx * dy;
        }
        default:
            assert(false && "interpolation mode not implemented in the test sampler");
            return border;
    }
}

}  // namespace rpptest

#endif  // RPP_TEST_INTERPOLATION_H
