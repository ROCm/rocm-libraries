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

#ifndef RPP_TEST_WARP_PERSPECTIVE_REF_H
#define RPP_TEST_WARP_PERSPECTIVE_REF_H

#include <rpp/rpp.h>

#include "framework/config_param.hpp"
#include "framework/geometric.hpp"

namespace rpptest {

/*
Reference model: warp_perspective

RPP op
  rppt_warp_perspective   (Image / Geometric augmentation)

Description
  Projective image warp. perspectiveTensor holds 9 values per image, the
  row-major 3x3 matrix M = [m0 m1 m2; m3 m4 m5; m6 m7 m8], which maps each
  output pixel to the source coordinate it samples, in homogeneous form.
  Centring, ROI offset, border and sampling are as in warp_affine.

Expression
  cx = floor(roiW/2), cy = floor(roiH/2), dx = outX - cx, dy = outY - cy

  w    =  m6*dx + m7*dy + m8
  srcX = x0 + (m0*dx + m1*dy + m2) / w + cx
  srcY = y0 + (m3*dx + m4*dy + m5) / w + cy

Notes
  The public header does not document the matrix direction or the mapping
  frame; the model follows warp_affine, and shares its ROI-origin defect, so
  partial-ROI cases are on the skip list.
*/
template <typename T>
void warp_perspective_reference(const T* src, const RpptDesc& sd, T* dst, const RpptDesc& dd,
                                DType dt, const RpptROI* roi, RpptRoiType roiType,
                                const Rpp32f* perspectiveTensor, RpptInterpolationType interp) {
    geometric_reference<T>(src, sd, dst, dd, dt, roi, roiType, roi_out_sizes(sd, roi, roiType),
                           interp, [&](Rpp32u n, double ox, double oy, double& sx, double& sy) {
                               const Rpp32f* m =
                                   perspectiveTensor + static_cast<std::size_t>(n) * 9;
                               const RoiBounds b = roi_bounds(roi[n], roiType);
                               const double cx = static_cast<double>(b.w / 2);
                               const double cy = static_cast<double>(b.h / 2);
                               const double dx = ox - cx, dy = oy - cy;
                               const double w = m[6] * dx + m[7] * dy + m[8];
                               sx = b.x0 + (m[0] * dx + m[1] * dy + m[2]) / w + cx;
                               sy = b.y0 + (m[3] * dx + m[4] * dy + m[5]) / w + cy;
                           });
}

}  // namespace rpptest

#endif  // RPP_TEST_WARP_PERSPECTIVE_REF_H
