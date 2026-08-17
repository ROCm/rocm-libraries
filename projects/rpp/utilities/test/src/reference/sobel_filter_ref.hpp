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

#ifndef RPP_TEST_SOBEL_FILTER_REF_H
#define RPP_TEST_SOBEL_FILTER_REF_H

#include <rpp/rpp.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/filter_common.hpp"

namespace rpptest {

// Independent host golden model for rppt_sobel_filter, derived from the canonical 3x3 Sobel
// operator definition (Gx/Gy gradient kernels, gradient magnitude for the XY case, REPLICATE
// border), NOT from the RPP kernel. Used as the reference for BOTH backends so kernel bugs
// surface as diffs.
//
// Scope: PLN1 only, kernelSize = 3 only. sobel_filter's dstDesc is always single-channel
// grayscale (c=1, NCHW), so a 3-channel input would require an undocumented RGB->grayscale
// conversion (not independently derivable), and the extended k=5/7 kernels are
// convention-dependent (varying coefficient conventions). Both are deferred here; this model
// covers grayscale-in/grayscale-out with the universally-defined 3x3 Sobel operator.
//
// The window comes from filter_reference (reference/filter_common.hpp), which owns the REPLICATE
// border and the placement; sobel supplies only the reduction, since it needs the raw gx/gy before
// quantization (for the magnitude case) rather than a single convolution.
//
// Kernels (row-major, dy=-1..1 outer, dx=-1..1 inner):
//   Gx = [-1,0,1, -2,0,2, -1,0,1]      Gy = [-1,-2,-1, 0,0,0, 1,2,1]
// gx = sum Gx[k]*w[k], gy = sum Gy[k]*w[k].  sobelType 0 -> gx, 1 -> gy, 2 -> sqrt(gx^2+gy^2).
// The result is quantized back to the dtype via quantize_stored (U8 round+clamp[0,255],
// I8 round+clamp[-128,127], F16/F32 clamp[0,1]) -- gradients can be negative / out of range, so
// clamping is the intended "same depth as src" behavior; any resulting diff is a finding, not a
// reference bug.
template <typename T>
void sobel_filter_reference(const T* src, T* dst, const RpptDesc& d, DType dt, const RpptROI* roi,
                            RpptRoiType type, Rpp32u sobelType, Rpp32u kernelSize) {
    static const double Gx[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    static const double Gy[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    filter_reference<T>(src, dst, d, roi, type, kernelSize, [&](const double* w, int kk) {
        double gx = 0.0, gy = 0.0;
        for (int k = 0; k < kk; ++k) {
            gx += Gx[k] * w[k];
            gy += Gy[k] * w[k];
        }
        const double result =
            sobelType == 0 ? gx : sobelType == 1 ? gy : std::sqrt(gx * gx + gy * gy);
        return quantize_stored(result, dt);
    });
}

}  // namespace rpptest

#endif  // RPP_TEST_SOBEL_FILTER_REF_H
