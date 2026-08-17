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

#ifndef RPP_TEST_FISHEYE_REF_H
#define RPP_TEST_FISHEYE_REF_H

#include <rpp/rpp.h>

#include <cmath>
#include <cstring>

#include "framework/config_param.hpp"
#include "framework/geometric.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Independent host golden model for rppt_fisheye, derived from the definition of the fisheye
// augmentation (a radial magnification about the frame centre, the classic "fish eye" filter) and
// NOT from the RPP kernel. Used as the reference for both backends so kernel bugs surface as diffs.
//
// The op takes no parameters, so the whole of its behaviour is the map below. Destination pixel
// (i, j) of the ROI-sized output region is placed on the centred unit square, its radius pulled in
// by the fisheye profile, and the source read back at the rescaled point:
//
//     nx = 2 i / w - 1 ,  ny = 2 j / h - 1 ,  r = hypot(nx, ny)      // r approximate, see below
//     r > 1                        -> black (the destination corners fall outside the unit disc)
//     r' = (r + 1 - sqrt(1 - r^2)) / 2                       // radial profile, r' <= r
//     theta = atan2(ny, nx)
//     srcX = x0 + (r' cos(theta) + 1) w / 2
//     srcY = y0 + (r' sin(theta) + 1) h / 2
//
// r' <= r everywhere and r'(1) = 1, so the disc is magnified towards its centre and its rim is
// fixed. The profile is the identity only at r = 0 and r = 1, where the map reproduces the source
// pixel exactly -- which is what makes the sub-pixel conventions below self-consistent rather than
// arbitrary.
//
// NOTE (semantics assumptions): the public header documents no formula, so a kernel that chose
// differently shows up as a diff -- a finding, not a reference bug.
//   - Sampling is NEAREST_NEIGHBOR: the op exposes no interpolationType, and a fisheye is a plain
//     resampling of existing texels. Bilinear would disagree on nearly every pixel.
//   - Coordinates are frame-corner based (nx = 2i/w - 1, no +0.5 pixel-centre term), which is what
//     makes r' = r round-trip to exactly srcX = i. The inverse of that choice is that the map is
//     asymmetric by half a texel about the centre.
//   - The normalisation extent and the destination region are the ROI's, and the source is read at
//     the ROI offset, matching the ROI convention of the suite's other warps
//     (framework/geometric.hpp): the ROI both sizes the output and bounds the valid source
//     rectangle.
//   - The radius is the fast approximation below rather than an exact sqrt, and the whole map runs
//     in single precision. That is a deliberate accuracy-for-speed trade of this augmentation, so
//     the model has to make it too: an exact radius disagrees on the disc rim (where r = 1 exactly
//     is pushed outside the disc) and on the ~2% of pixels whose mapped coordinate sits within the
//     approximation's error of a half-integer, which nearest-neighbour then rounds the other way.

// A destination coordinate outside every ROI rectangle, so the shared driver fills it with the
// dtype's black. Used for the pixels that fall outside the unit disc.
constexpr double kFisheyeOutside = -1.0;

// Lomont's fast inverse square root with one Newton round -- the sanctioned approximation, ~0.17%
// relative error, always low. Written from the published algorithm (Lomont, C., 2003, "Fast Inverse
// Square Root"), which is the accuracy contract the op is held to.
inline float fisheye_inverse_sqrt(float x) {
    const float xHalf = 0.5f * x;
    int i;
    std::memcpy(&i, &x, sizeof(i));
    i = 0x5f3759df - (i >> 1);
    std::memcpy(&x, &i, sizeof(x));
    return x * (1.5f - xHalf * x * x);
}

// Destination pixel (outX, outY) within a w x h region whose source origin is (x0, y0) -> the
// source coordinate to sample, in the absolute image frame.
inline void fisheye_map(double outX, double outY, double w, double h, double x0, double y0,
                        double& srcX, double& srcY) {
    const float nx = static_cast<float>(2.0 * outX / w - 1.0);
    const float ny = static_cast<float>(2.0 * outY / h - 1.0);
    const float r = 1.0f / fisheye_inverse_sqrt(nx * nx + ny * ny);
    if (!(r >= 0.0f && r <= 1.0f)) {
        srcX = srcY = kFisheyeOutside;
        return;
    }
    const float rSrc = (r + 1.0f - std::sqrt(1.0f - r * r)) * 0.5f;
    const float theta = std::atan2(ny, nx);
    srcX = x0 + (rSrc * std::cos(theta) + 1.0f) * static_cast<float>(w) * 0.5f;
    srcY = y0 + (rSrc * std::sin(theta) + 1.0f) * static_cast<float>(h) * 0.5f;
}

template <typename T>
void fisheye_reference(const T* src, T* dst, const RpptDesc& d, DType dt, const RpptROI* roi,
                       RpptRoiType roiType) {
    geometric_reference<T>(src, dst, d, dt, roi, roiType, roi_out_sizes(d, roi, roiType),
                           NEAREST_NEIGHBOR,
                           [&](Rpp32u n, double ox, double oy, double& sx, double& sy) {
                               const RoiBounds b = roi_bounds(roi[n], roiType);
                               fisheye_map(ox, oy, b.w, b.h, b.x0, b.y0, sx, sy);
                           });
}

}  // namespace rpptest

#endif  // RPP_TEST_FISHEYE_REF_H
