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

  The output index (outX, outY) is origin-based and the source coordinate is
  absolute (full-image frame): RPP's warp ignores the ROI offset for the
  mapping and uses the ROI only to size the output. The matrix acts about the
  centre of the ROI, not its origin, and that centre is the truncated half of
  each extent. Output pixel centres are at integer indices and out-of-image
  samples are the type's black. Sampling, interpolation, border and quantize
  are handled by geometric_reference().

Expression
  cx = floor(roiW/2), cy = floor(roiH/2), dx = outX - cx, dy = outY - cy

  srcX = m0*dx + m1*dy + m2 + cx
  srcY = m3*dx + m4*dy + m5 + cy

Notes
  The public header does not document the matrix direction or the mapping
  frame. The destination->source direction above (confirmed against the op via
  the pure-translation cases, and consistent with the remap contract) and the
  absolute-frame / origin-based output placement are assumed. The centring
  cancels for the identity and for a pure translation, so only a matrix with a
  non-identity linear part distinguishes it.
*/
template <typename T>
void warp_affine_reference(const T* src, const RpptDesc& sd, T* dst, const RpptDesc& dd, DType dt,
                           const RpptROI* roi, RpptRoiType roiType, const Rpp32f* affineTensor,
                           RpptInterpolationType interp) {
    geometric_reference<T>(
        src, sd, dst, dd, dt, roi, roiType, roi_out_sizes(sd, roi, roiType), interp,
        [&](Rpp32u n, double ox, double oy, double& sx, double& sy) {
            const Rpp32f* m = affineTensor + static_cast<std::size_t>(n) * 6;
            const RoiBounds b = roi_bounds(roi[n], roiType);
            const double cx = static_cast<double>(b.w / 2), cy = static_cast<double>(b.h / 2);
            const double dx = ox - cx, dy = oy - cy;
            sx = m[0] * dx + m[1] * dy + m[2] + cx;
            sy = m[3] * dx + m[4] * dy + m[5] + cy;
        });
}

}  // namespace rpptest

#endif  // RPP_TEST_WARP_AFFINE_REF_H
