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
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Independent host golden model for rppt_warp_affine, derived from the op's definition (an affine
// image warp), NOT from the RPP kernel. Used as the reference for both backends so kernel bugs
// surface as diffs.
//
// affineTensor holds 6 values per image, the row-major 2x3 matrix M = [m0 m1 m2; m3 m4 m5]. A warp
// is an inverse mapping: for each output pixel the matrix gives the SOURCE coordinate to sample,
//     srcX = m0*outX + m1*outY + m2
//     srcY = m3*outX + m4*outY + m5
// which is the parametric form of the remap definition output(x,y) = input(mapx(x,y), mapy(x,y)).
// The output index (outX,outY) is origin-based and the source coordinate is absolute (full-image
// frame): RPP's warp ignores the ROI offset for the mapping and uses the ROI only to size the
// output. Output pixel centers are at integer indices; out-of-image samples are the dtype's black.
// Sampling/interpolation/border/quantize are handled by geometric_reference().
//
// NOTE (semantics assumption): the public header does not document the matrix direction or the
// mapping frame. The destination->source direction above (confirmed against the op via the
// pure-translation cases, and consistent with the remap contract) and the absolute-frame /
// origin-based output placement are assumed.
template <typename T>
void warp_affine_reference(const T* src, T* dst, const RpptDesc& d, DType dt, const RpptROI* roi,
                           RpptRoiType roiType, const Rpp32f* affineTensor,
                           RpptInterpolationType interp) {
    geometric_reference<T>(
        src, dst, d, dt, roi, roiType, roi_out_sizes(d, roi, roiType), interp,
        [&](Rpp32u n, double ox, double oy, double& sx, double& sy) {
            const Rpp32f* m = affineTensor + static_cast<std::size_t>(n) * 6;
            sx = m[0] * ox + m[1] * oy + m[2];
            sy = m[3] * ox + m[4] * oy + m[5];
        });
}

}  // namespace rpptest

#endif  // RPP_TEST_WARP_AFFINE_REF_H
