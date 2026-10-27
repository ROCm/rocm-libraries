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

/*
Reference model: fisheye

RPP op
  rppt_fisheye   (Image / Effects augmentation)

Description
  Radial magnification about the frame centre. Destination pixel (i, j) of the
  ROI-sized output is placed on the centred unit square, its radius pulled in
  by the fisheye profile, and the source read back at the rescaled point. The
  rim (r = 1) and centre (r = 0) are fixed.

Expression
  nx    = 2 i / w - 1,  ny = 2 j / h - 1,  r = hypot(nx, ny)
  r > 1 -> black (the destination corners fall outside the unit disc)
  r'    = (r + 1 - sqrt(1 - r^2)) / 2          radial profile, r' <= r
  theta = atan2(ny, nx)
  srcX  = x0 + (r' cos(theta) + 1) w / 2
  srcY  = y0 + (r' sin(theta) + 1) h / 2

Notes
  The public header documents no formula; the model mirrors the kernel.

  - Sampling is NEAREST_NEIGHBOR; the op exposes no interpolationType.
  - Coordinates are frame-corner based (no +0.5 pixel-centre term).
  - The normalisation extent and output region are the ROI's, and the source
    is read at the ROI offset (framework/geometric.hpp).
  - The radius uses the fast inverse sqrt below and the map runs in single
    precision, as in the kernel. An exact radius disagrees on the disc rim and
    on pixels whose mapped coordinate is near a half-integer.
*/

// A destination coordinate outside every ROI rectangle, so the shared driver fills it with the
// dtype's black. Used for the pixels that fall outside the unit disc.
constexpr double kFisheyeOutside = -1.0;

// Fast inverse square root with one Newton round (Lomont, C., 2003, "Fast Inverse Square Root"),
// ~0.17% relative error, always low.
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
void fisheye_reference(const T* src, const RpptDesc& sd, T* dst, const RpptDesc& dd, DType dt,
                       const RpptROI* roi, RpptRoiType roiType) {
    geometric_reference<T>(src, sd, dst, dd, dt, roi, roiType, roi_out_sizes(sd, roi, roiType),
                           NEAREST_NEIGHBOR,
                           [&](Rpp32u n, double ox, double oy, double& sx, double& sy) {
                               const RoiBounds b = roi_bounds(roi[n], roiType);
                               fisheye_map(ox, oy, b.w, b.h, b.x0, b.y0, sx, sy);
                           });
}

}  // namespace rpptest

#endif  // RPP_TEST_FISHEYE_REF_H
