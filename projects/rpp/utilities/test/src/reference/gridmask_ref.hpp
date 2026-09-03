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

#ifndef RPP_TEST_GRIDMASK_REF_H
#define RPP_TEST_GRIDMASK_REF_H

#include <rpp/rpp.h>

#include <cmath>

#include "framework/config_param.hpp"
#include "framework/intensity.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

/*
Reference model: gridmask

RPP op
  rppt_gridmask   (Image / Effects augmentation)

Description
  GridMask Data Augmentation (https://arxiv.org/abs/2001.04086). The image is
  multiplied by a binary periodic mask, so each pixel is either copied verbatim
  or forced to black.

  The mask is a grid of period d == tileWidth ("width of black square +
  spacing until next black square") whose black square has edge
  l == gridRatio * tileWidth ("black square width / tileWidth"), rotated by
  gridAngle radians and shifted by translateVector.

  The grid is a rigid pattern laid over the processed region. A positive
  translateVector moves the black square to [t, t+l) within its cell -- the
  translation is subtracted, not added. Image coordinates are y-DOWN, so it is
  R(+gridAngle), not the y-up R(-gridAngle), that rotates the pattern
  counter-clockwise on screen. Coordinates are ROI-RELATIVE: p = (i, j) with
  the ROI's top-left as the origin, matching the suite's convention for
  spatially generated patterns under a ROI (non_linear_blend centres its
  gaussian on the region, not on the absolute image frame).

Expression
  (gx, gy) = R(gridAngle) * (p - translateVector)
  masked   = (gx mod d) < l  &&  (gy mod d) < l
  dst      = masked ? black : src

  The modulo is a floating-point one corrected for negative operands (rotation
  can push a grid coordinate negative), so the tiling stays periodic on both
  sides of the origin.

Per-type form
  "Black" is 0 intensity in the suite's shared intensity model
  (from_unit(0.0, dt)).

    U8  0        I8  -128 (0 intensity shifted by -128)
    F16 0.0      F32 0.0

Notes
  The API doc states neither the rotation handedness nor the rotate/translate
  order. The gridAngle == 0 parameter sets are independent of both, and the
  rotated set uses translateVector == {0,0} so that the order cannot matter
  there either.
*/

// True iff the ROI-relative pixel (x, y) falls inside a black square of the grid.
inline bool gridmask_masked(int x, int y, Rpp32u tileWidth, double gridRatio, double gridAngle,
                            Rpp32u translateX, Rpp32u translateY) {
    const double d = static_cast<double>(tileWidth);
    if (d <= 0.0) return false;
    const double l = gridRatio * d;
    if (l <= 0.0) return false;

    const double c = std::cos(gridAngle), s = std::sin(gridAngle);
    const double px = static_cast<double>(x) - static_cast<double>(translateX);
    const double py = static_cast<double>(y) - static_cast<double>(translateY);
    double gx = px * c - py * s;
    double gy = px * s + py * c;

    gx = std::fmod(gx, d);
    if (gx < 0.0) gx += d;
    gy = std::fmod(gy, d);
    if (gy < 0.0) gy += d;

    return gx < l && gy < l;
}

inline double gridmask_scalar(double v, DType dt, bool masked) {
    return masked ? from_unit(0.0, dt) : v;
}

// Mask membership is tested against the ROI-relative coordinate (i, j).
template <typename T>
void gridmask_reference(const T* src, const RpptDesc& sd, T* dst, const RpptDesc& dd, DType dt,
                        const RpptROI* roi, RpptRoiType roiType, Rpp32u tileWidth, double gridRatio,
                        double gridAngle, Rpp32u translateX, Rpp32u translateY) {
    for_each_roi_io(
        sd, dd, roi, roiType,
        [&](Rpp32u, Rpp32u, Rpp32u j, Rpp32u i, std::size_t srcIdx, std::size_t dstIdx) {
            const bool masked = gridmask_masked(static_cast<int>(i), static_cast<int>(j), tileWidth,
                                                gridRatio, gridAngle, translateX, translateY);
            dst[dstIdx] = from_double<T>(gridmask_scalar(to_double(src[srcIdx]), dt, masked));
        });
}

}  // namespace rpptest

#endif  // RPP_TEST_GRIDMASK_REF_H
