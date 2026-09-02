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
  being anticlockwise. Being a warp, it inverse-maps: for each output pixel the
  source coordinate is the output coordinate rotated about the centre.

  The output index is origin-based and the source coordinate is absolute
  (full-image frame). Like warp_affine and warp_perspective, RPP sizes the
  output from the ROI and bounds valid samples to the ROI rectangle, outside
  which the sample is black. Sampling, interpolation, border and quantize are
  handled by geometric_reference(), shared with the other warps.

Expression
  With dx = outX - cx, dy = outY - cy and R(theta) = [cos -sin; sin cos]:

  srcX = cx + dx*cos(theta) - dy*sin(theta)
  srcY = cy + dx*sin(theta) + dy*cos(theta)

Notes
  The public header documents neither the centre of rotation nor the exact
  sign. The centre = (roiWidth/2, roiHeight/2) and the source = R(+theta)*d
  mapping above were confirmed against the op on FullRoi by black-box probing:
  180 deg gives srcCol = W - outCol, so the centre is W/2 and not (W-1)/2, and
  +10 deg fixes the sign. The halves are truncated to whole pixels, as the
  shared warp_affine kernel computes them (roiWidth >> 1); only an odd extent
  tells the truncated half from the geometric one.

  Cardinal angles (0/90/180/270) map to integer source coordinates, so the
  golden is bit-exact there.
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
                               const double theta = static_cast<double>(angleDeg[n]) * M_PI / 180.0;
                               const double ct = std::cos(theta), st = std::sin(theta);
                               const double dx = ox - cx, dy = oy - cy;
                               sx = cx + dx * ct - dy * st;
                               sy = cy + dx * st + dy * ct;
                           });
}

}  // namespace rpptest

#endif  // RPP_TEST_ROTATE_REF_H
