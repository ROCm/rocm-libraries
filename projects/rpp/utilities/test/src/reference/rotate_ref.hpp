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

#ifndef RPP_TEST_ROTATE_REF_H
#define RPP_TEST_ROTATE_REF_H

#include <rpp/rpp.h>

#include <cmath>

#include "framework/config_param.hpp"
#include "framework/geometric.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

/*
Reference model: rotate

RPP op
  rppt_rotate   (Image / Geometric augmentation)

Description
  Rotates the image about its centre by a per-image angle in degrees, positive
  being anticlockwise. Each output pixel inverse-maps to the source by a
  rotation about the ROI centre, and the ROI origin (x0,y0) is added to reach
  the absolute frame. The output is ROI-sized and samples outside the ROI are
  black. Sampling and quantization are handled by geometric_reference().

Expression
  With dx = outX - cx, dy = outY - cy and R(theta) = [cos -sin; sin cos]:

  srcX = x0 + cx + dx*cos(theta) - dy*sin(theta)
  srcY = y0 + cy + dx*sin(theta) + dy*cos(theta)

Notes
  The public header documents neither the centre of rotation nor the sign. The
  model matches the op: the centre is (roiWidth/2, roiHeight/2) truncated to
  whole pixels, as the warp_affine kernel computes it (roiWidth >> 1).

  Cardinal angles (0/90/180/270) map to integer source coordinates, so the
  result is bit-exact there.

  rotate shares warp_affine's ROI-origin defect (the kernels omit +x0/+y0), so
  partial-ROI cases are on the skip list; see warp_affine_ref.hpp.
*/
template <typename T>
void rotate_reference(const T* src, T* dst, const RpptDesc& d, DType dt, const RpptROI* roi,
                      RpptRoiType roiType, const Rpp32f* angleDeg, RpptInterpolationType interp) {
    geometric_reference<T>(src, d, dst, d, dt, roi, roiType, roi_out_sizes(d, roi, roiType), interp,
                           [&](Rpp32u n, double ox, double oy, double& sx, double& sy) {
                               const RoiBounds b = roi_bounds(roi[n], roiType);
                               // integer division by 2 is on purpose
                               const double cx = static_cast<double>(b.w / 2);
                               const double cy = static_cast<double>(b.h / 2);
                               const double theta = static_cast<double>(angleDeg[n]) * kPi / 180.0;
                               const double ct = std::cos(theta), st = std::sin(theta);
                               const double dx = ox - cx, dy = oy - cy;
                               sx = b.x0 + cx + dx * ct - dy * st;
                               sy = b.y0 + cy + dx * st + dy * ct;
                           });
}

}  // namespace rpptest

#endif  // RPP_TEST_ROTATE_REF_H
