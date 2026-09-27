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

#include "framework/intensity.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Shared, op-agnostic source sampling for the geometric golden models. Every "move pixels around"
// op inverse-maps an output coordinate to a source coordinate and samples the source there; this
// header is the single implementation of that sampling (nearest / bilinear) so no two op
// references reimplement it and diverge.
//
// Coordinates are absolute within one image-channel plane whose element (0,0) sits at `base`. The
// valid source region is the half-open rectangle [x0,x1) x [y0,y1) (for ROI ops this is the ROI
// rectangle); what happens outside it is the caller's Border policy.

// ---- border handling -------------------------------------------------------

// How a sample outside the valid rectangle is resolved. Constant substitutes a fixed value; the
// other four remap the coordinate back inside, so every tap reads real source data.
//
// For valid indices a b c d (extent n = 4), the sequences either side of the rectangle are:
//
//     Constant     VVV|abcd|VVV     (V = the constant)
//     Replicate    aaa|abcd|ddd
//     Wrap         bcd|abcd|abc
//     Reflect      cba|abcd|dcb     the edge sample IS repeated across the boundary
//     Reflect101   dcb|abcd|cba     the edge sample is NOT repeated
//
// The names and the semantics match OpenCV's BORDER_CONSTANT / _REPLICATE / _WRAP / _REFLECT /
// _REFLECT_101, so a golden cross-checked against cv2 means the same thing by the same word.
// Reflect and Reflect101 differ only in whether the edge texel appears twice: one step outside
// they agree, two steps outside they do not.
enum class BorderMode { Constant, Replicate, Wrap, Reflect, Reflect101 };

// The border policy for one sampling call. Implicitly constructible from a bare double, which
// means Constant with that value -- the same idiom Bound uses in tolerance.hpp, and what keeps
// every existing `..., dtype_black(dt))` call site meaning exactly what it meant before.
struct Border {
    BorderMode mode = BorderMode::Constant;
    double value = 0.0;  // read only when mode == Constant

    constexpr Border(double constantValue = 0.0)
        : mode(BorderMode::Constant), value(constantValue) {}
    constexpr Border(BorderMode m, double constantValue = 0.0) : mode(m), value(constantValue) {}
};

// Maps a zero-based index r into [0, n) under the given mode. Identity for an r already in range,
// so a caller may apply it unconditionally. Not meaningful for Constant, which has no in-range
// answer -- src_texel() resolves that case before reaching here.
//
// Every mode is defined for an arbitrary distance outside, not just one step: a warp can map a
// destination pixel far beyond the ROI, and a rule that only handled the first period would be
// wrong there in a way that a +/-1 test could not see.
inline int border_index(int r, int n, BorderMode mode) {
    assert(n >= 1 && "border remapping needs a non-empty range");
    switch (mode) {
        case BorderMode::Replicate:
            return r < 0 ? 0 : (r >= n ? n - 1 : r);
        case BorderMode::Wrap:
            return ((r % n) + n) % n;
        case BorderMode::Reflect: {
            // Period 2n: the range followed by its mirror, edge texel included in both halves.
            const int period = 2 * n;
            const int p = ((r % period) + period) % period;
            return p < n ? p : (period - 1 - p);
        }
        case BorderMode::Reflect101: {
            // Period 2n-2: the mirror shares its endpoints with the range, so the edge texel
            // appears once. Degenerate at n == 1, where that period would be zero.
            if (n == 1) return 0;
            const int period = 2 * n - 2;
            const int p = ((r % period) + period) % period;
            return p < n ? p : (period - p);
        }
        case BorderMode::Constant:
            break;
    }
    assert(false && "border_index called for BorderMode::Constant");
    return r;
}

// ---- sampling --------------------------------------------------------------

template <typename T>
inline double src_texel(const T* src, const RpptDesc& d, std::size_t base, int x, int y, int x0,
                        int y0, int x1, int y1, Border border) {
    int sx = x, sy = y;
    if (x < x0 || y < y0 || x >= x1 || y >= y1) {
        if (border.mode == BorderMode::Constant) return border.value;
        // Each axis is resolved independently, as in OpenCV. Remapping an already-in-range
        // coordinate is the identity, so the axis that was in bounds stays where it was.
        sx = x0 + border_index(x - x0, x1 - x0, border.mode);
        sy = y0 + border_index(y - y0, y1 - y0, border.mode);
    }
    return to_double(
        src[plane_index(d, base, static_cast<std::size_t>(sy), static_cast<std::size_t>(sx))]);
}

// Samples the plane at fractional (x,y) (texel centers at integer coords) using `interp`.
// Interpolation runs directly on stored values: it is affine, so it commutes with the U8/I8
// intensity offset and needs no unit conversion. Only NEAREST_NEIGHBOR and BILINEAR are
// implemented; the filtered modes (BICUBIC/LANCZOS/GAUSSIAN/TRIANGULAR) are added when an op that
// needs them is ported.
//
// The border policy applies per tap, so under a remapping mode BILINEAR blends four real texels
// and introduces no border colour at all -- which is what a replicating kernel does.
template <typename T>
inline double sample(const T* src, const RpptDesc& d, std::size_t base, double x, double y, int x0,
                     int y0, int x1, int y1, RpptInterpolationType interp, Border border) {
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
            return v00 * (1.0 - dx) * (1.0 - dy) + v01 * dx * (1.0 - dy) + v10 * (1.0 - dx) * dy +
                   v11 * dx * dy;
        }
        default:
            assert(false && "interpolation mode not implemented in the test sampler");
            return border.value;
    }
}

}  // namespace rpptest

#endif  // RPP_TEST_INTERPOLATION_H
