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

// Independent host golden model for rppt_warp_perspective, derived from the op's definition (a
// projective image warp), NOT from the RPP kernel. Used as the reference for both backends so
// kernel bugs surface as diffs.
//
// perspectiveTensor holds 9 values per image, the row-major 3x3 matrix M = [m0 m1 m2; m3 m4 m5;
// m6 m7 m8]. A warp is an inverse mapping: for each output pixel the matrix gives the SOURCE
// coordinate to sample, in homogeneous form
//     w    = m6*outX + m7*outY + m8
//     srcX = (m0*outX + m1*outY + m2) / w
//     srcY = (m3*outX + m4*outY + m5) / w
// the projective generalization of the remap contract output(x,y) = input(mapx(x,y), mapy(x,y)).
// The output index (outX,outY) is origin-based and the source coordinate is absolute (full-image
// frame): RPP's warp ignores the ROI offset for the mapping and uses the ROI only to size the
// output and bound the valid-source rectangle [x0,x0+w) x [y0,y0+h), outside which the sample is
// black. Output pixel centers are at integer indices; sampling/interpolation/border/quantize are
// handled by geometric_reference() (shared with warp_affine).
//
// NOTE (semantics assumption): the public header does not document the matrix direction or the
// mapping frame. The destination->source direction and absolute-frame / origin-based output
// placement are assumed, matching rppt_warp_affine (whose conventions were confirmed against the op
// via pure-translation cases) since the two share the same geometric machinery.
template <typename T>
void warp_perspective_reference(const T* src, T* dst, const RpptDesc& d, DType dt,
                                const RpptROI* roi, RpptRoiType roiType,
                                const Rpp32f* perspectiveTensor, RpptInterpolationType interp) {
    geometric_reference<T>(src, dst, d, dt, roi, roiType, roi_out_sizes(d, roi, roiType), interp,
                           [&](Rpp32u n, double ox, double oy, double& sx, double& sy) {
                               const Rpp32f* m =
                                   perspectiveTensor + static_cast<std::size_t>(n) * 9;
                               const double w = m[6] * ox + m[7] * oy + m[8];
                               sx = (m[0] * ox + m[1] * oy + m[2]) / w;
                               sy = (m[3] * ox + m[4] * oy + m[5]) / w;
                           });
}

}  // namespace rpptest

#endif  // RPP_TEST_WARP_PERSPECTIVE_REF_H
