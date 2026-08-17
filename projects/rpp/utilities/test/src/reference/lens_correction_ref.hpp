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

#ifndef RPP_TEST_LENS_CORRECTION_REF_H
#define RPP_TEST_LENS_CORRECTION_REF_H

#include <rpp/rpp.h>

#include "framework/config_param.hpp"
#include "framework/geometric.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Independent host golden model for rppt_lens_correction, derived from the op's definition
// ("compensate barrel lens distortion", parameterized by a 9-element camera matrix and 8 distortion
// coefficients) and the standard Brown-Conrady camera model those two arguments name, NOT from the
// RPP kernel. Used as the reference for both backends so kernel bugs surface as diffs.
//
// The op is an undistortion remap: the caller's rowRemapTable / colRemapTable are scratch (the
// legacy harness allocates them zeroed and never fills them for this op), so the op derives the map
// itself from the intrinsics and then samples the source through it. With 9 + 8 parameters and a
// documented "black image if the camera matrix determinant is 0" (i.e. the matrix is inverted),
// this is the textbook pinhole + radial/tangential model, coefficients in the universal
// (k1, k2, p1, p2, k3, k4, k5, k6) order:
//
//     x  = (outX - cx) / fx ,  y = (outY - cy) / fy          // pixel -> normalized
//     r2 = x^2 + y^2
//     radial = (1 + k1 r2 + k2 r2^2 + k3 r2^3) / (1 + k4 r2 + k5 r2^2 + k6 r2^3)
//     x' = x*radial + 2 p1 x y + p2 (r2 + 2 x^2)             // apply distortion
//     y' = y*radial + p1 (r2 + 2 y^2) + 2 p2 x y
//     srcX = fx x' + cx ,  srcY = fy y' + cy                 // normalized -> pixel
//
// The source is then sampled at (srcX, srcY) and the result quantized per dtype -- both handled by
// geometric_reference() / interpolation.hpp, shared with the other warps, so the sampling model is
// the suite's own and not the kernel's.
//
// NOTE (semantics assumptions): the public header documents neither of the following, so a kernel
// that chose differently shows up as a diff -- a finding, not a reference bug.
//   - Interpolation is taken to be BILINEAR (the op exposes no interpolationType, and bilinear is
//     the convention for an undistortion remap). The zero-coefficient parameter set below makes the
//     map the exact identity, where bilinear and nearest-neighbour agree, so that set validates the
//     model independently of this choice.
//   - The output index is origin-based, matching the ROI convention RPP's other warps use (see
//     framework/geometric.hpp): the ROI sizes the output and bounds the valid source rectangle, and
//     the mapping itself ignores the ROI offset. Under a full ROI the two readings coincide.
//
// The camera matrix is read in its standard row-major form [fx 0 cx; 0 fy cy; 0 0 1]; skew
// (element 1) is assumed 0, as it is in the intrinsics the API's own sample uses. A singular matrix
// (fx or fy == 0) is documented to yield a black image and is not exercised.

// Destination pixel (outX, outY) -> the source coordinate to sample, in the absolute image frame.
// cameraMatrix is 9 elements row-major, distortionCoeffs 8 as (k1, k2, p1, p2, k3, k4, k5, k6).
inline void lens_correction_map(double outX, double outY, const Rpp32f* cameraMatrix,
                                const Rpp32f* distortionCoeffs, double& srcX, double& srcY) {
    const double fx = cameraMatrix[0], cx = cameraMatrix[2];
    const double fy = cameraMatrix[4], cy = cameraMatrix[5];
    const double k1 = distortionCoeffs[0], k2 = distortionCoeffs[1];
    const double p1 = distortionCoeffs[2], p2 = distortionCoeffs[3];
    const double k3 = distortionCoeffs[4], k4 = distortionCoeffs[5];
    const double k5 = distortionCoeffs[6], k6 = distortionCoeffs[7];

    const double x = (outX - cx) / fx;
    const double y = (outY - cy) / fy;
    const double r2 = x * x + y * y;
    const double r4 = r2 * r2, r6 = r4 * r2;
    const double radial = (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / (1.0 + k4 * r2 + k5 * r4 + k6 * r6);

    const double xD = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
    const double yD = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;

    srcX = fx * xD + cx;
    srcY = fy * yD + cy;
}

// cameraMatrixTensor holds 9 values per image, distortionCoeffsTensor 8 per image.
template <typename T>
void lens_correction_reference(const T* src, T* dst, const RpptDesc& d, DType dt,
                               const RpptROI* roi, RpptRoiType roiType,
                               const Rpp32f* cameraMatrixTensor,
                               const Rpp32f* distortionCoeffsTensor,
                               RpptInterpolationType interp) {
    geometric_reference<T>(src, dst, d, dt, roi, roiType, roi_out_sizes(d, roi, roiType), interp,
                           [&](Rpp32u n, double ox, double oy, double& sx, double& sy) {
                               lens_correction_map(ox, oy,
                                                   cameraMatrixTensor + static_cast<std::size_t>(n) * 9,
                                                   distortionCoeffsTensor +
                                                       static_cast<std::size_t>(n) * 8,
                                                   sx, sy);
                           });
}

}  // namespace rpptest

#endif  // RPP_TEST_LENS_CORRECTION_REF_H
