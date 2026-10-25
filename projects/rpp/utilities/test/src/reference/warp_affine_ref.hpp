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

#ifndef RPP_TEST_WARP_AFFINE_REF_H
#define RPP_TEST_WARP_AFFINE_REF_H

#include <rpp/rpp.h>

#include "framework/config_param.hpp"
#include "framework/geometric.hpp"

namespace rpptest {

/*
Reference model: warp_affine

RPP op
  rppt_warp_affine   (Image / Geometric augmentation)

Description
  Affine image warp. affineTensor holds 6 values per image, the row-major 2x3
  matrix M = [m0 m1 m2; m3 m4 m5], which maps each output pixel to the source
  coordinate it samples. The matrix acts about the ROI centre (truncated half
  extents) and the ROI origin (x0,y0) is added to reach the absolute frame.
  Out-of-ROI samples are the type's black. Sampling and quantization are
  handled by geometric_reference().

Expression
  cx = floor(roiW/2), cy = floor(roiH/2), dx = outX - cx, dy = outY - cy

  srcX = x0 + m0*dx + m1*dy + m2 + cx
  srcY = y0 + m3*dx + m4*dy + m5 + cy

Notes
  The public header does not document the matrix direction or the mapping
  frame; the model uses destination->source, as the op does.

  The +x0/+y0 follows the ROI convention of other RPP ops (crop, resize, flip)
  and makes the identity matrix reproduce the ROI. The kernels omit it while
  still clipping against the absolute ROI, so an identity warp over a partial
  ROI returns a part-black image (and HOST and HIP disagree). See the
  *_PartialRoi_* entries in skip_list.hpp.
*/
template <typename T>
void warp_affine_reference(const T* src, const RpptDesc& sd, T* dst, const RpptDesc& dd, DType dt,
                           const RpptROI* roi, RpptRoiType roiType, const Rpp32f* affineTensor,
                           RpptInterpolationType interp) {
    geometric_reference<T>(src, sd, dst, dd, dt, roi, roiType, roi_out_sizes(sd, roi, roiType),
                           interp, [&](Rpp32u n, double ox, double oy, double& sx, double& sy) {
                               const Rpp32f* m = affineTensor + static_cast<std::size_t>(n) * 6;
                               const RoiBounds b = roi_bounds(roi[n], roiType);
                               const double cx = static_cast<double>(b.w / 2),
                                            cy = static_cast<double>(b.h / 2);
                               const double dx = ox - cx, dy = oy - cy;
                               sx = b.x0 + m[0] * dx + m[1] * dy + m[2] + cx;
                               sy = b.y0 + m[3] * dx + m[4] * dy + m[5] + cy;
                           });
}

}  // namespace rpptest

#endif  // RPP_TEST_WARP_AFFINE_REF_H
