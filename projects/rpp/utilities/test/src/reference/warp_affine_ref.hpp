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
  matrix M = [m0 m1 m2; m3 m4 m5]. A warp is an inverse mapping: for each
  output pixel the matrix gives the SOURCE coordinate to sample, which is the
  parametric form of the remap definition
  output(x,y) = input(mapx(x,y), mapy(x,y)).

  The output index (outX, outY) is origin-based within the ROI-sized output
  region, and the matrix acts about the centre of the ROI, that centre being
  the truncated half of each extent. The resulting source coordinate is
  ROI-relative, so the ROI origin (x0,y0) is added to put it in the absolute
  (full-image) frame that geometric_reference() samples in. Output pixel
  centres are at integer indices and out-of-ROI samples are the type's black.
  Sampling, interpolation, border and quantize are handled by
  geometric_reference().

Expression
  cx = floor(roiW/2), cy = floor(roiH/2), dx = outX - cx, dy = outY - cy

  srcX = x0 + m0*dx + m1*dy + m2 + cx
  srcY = y0 + m3*dx + m4*dy + m5 + cy

Notes
  The public header does not document the matrix direction or the mapping
  frame. The destination->source direction above is confirmed against the op
  via the pure-translation cases and is consistent with the remap contract.
  The centring cancels for the identity and for a pure translation, so only a
  matrix with a non-identity linear part distinguishes it.

  The +x0/+y0 is the ROI convention every other ROI-taking op in RPP follows
  (crop, resize and flip all offset the source pointer by the ROI origin), and
  it is what makes the identity matrix reproduce the ROI. The kernels do NOT
  do this: both backends map the origin-based output index straight to a
  source coordinate while still clipping against the absolute ROI rectangle,
  so an identity warp over a partial ROI returns a part-black image. Mixing
  the two frames is the defect, not a convention -- and with them mixed the
  backends do not agree with each other either, HIP placing the result
  differently again. See the *_PartialRoi_* entries in skip_list.hpp.
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
