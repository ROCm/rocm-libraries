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

#ifndef RPP_TEST_EMBOSS_REF_H
#define RPP_TEST_EMBOSS_REF_H

#include <rpp/rpp.h>

#include <vector>

#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/filter_common.hpp"

namespace rpptest {

// Kernel-derived REGRESSION golden for rppt_emboss.
//
// Unlike most references in this suite, emboss's per-element semantics are NOT documented anywhere:
// the public API header describes only the parameters and shows sample images, and there is no
// spec. Emboss is also not recoverable from first principles -- several "standard" 3x3 emboss
// kernels exist, the strength and bias conventions are unstated, and there is no canonical NxN
// emboss kernel at all for the larger sizes. The weight tables below are therefore transcribed from
// the RPP emboss kernel with the user's explicit authorization. This model LOCKS current behavior
// (a regression test) rather than encoding independently-derived intent, so it will not catch a
// wrong-by-design kernel -- only a change in it. emboss is fully deterministic, so the same model
// serves both HOST and HIP.
//
// Everything OUTSIDE the weight table is the suite's own shared machinery, not the kernel's: the
// KxK window, the clamp-to-ROI REPLICATE border (the only border type the op supports), the
// source-at-ROI-offset / destination-at-origin placement and the per-dtype quantization all come
// from filter_reference() in reference/filter_common.hpp, shared with box/gaussian/median/sobel. So
// a border or placement defect still surfaces as a diff; only the coefficients are locked.
//
// The taps are applied as a CORRELATION (no kernel flip), matching every other filter in the suite.
// This matters here and nowhere else: box/gaussian/median are symmetric, so their tests cannot tell
// correlation from convolution, whereas the emboss table is deliberately anti-symmetric.
//
// Both tables sum to 1, so the filter is DC-preserving at strength 1 and needs no mid-grey bias
// (the classic +128 emboss offset is absent here). strength scales every tap, so the DC gain is the
// strength itself, and the kernel clamps strength at 2.0 from above (also undocumented, also
// transcribed).

// The base emboss taps, row-major, top-left to bottom-right, in the same dy/dx order
// gather_roi_window() produces. Only the sizes the test grids are provided.
inline std::vector<double> emboss_kernel(Rpp32u kernelSize, double strength) {
    static const double k3[9] = {2, 1, 0, 1, 1, -1, 0, -1, -2};
    static const double k5[25] = {3, 3, 2,  1,  0,  3, 2, 1,  0,  -1, 2, 1, 1,
                                  -1, -2, 1, 0, -1, -2, -3, 0, -1, -2, -3, -3};
    const double* base = (kernelSize == 3) ? k3 : k5;
    const std::size_t count = static_cast<std::size_t>(kernelSize) * kernelSize;
    const double scale = (strength > 2.0) ? 2.0 : strength;  // clamped from above only
    std::vector<double> kernel(count);
    for (std::size_t i = 0; i < count; ++i) kernel[i] = base[i] * scale;
    return kernel;
}

// I8 stores intensities shifted by -128, so the window is lifted into [0,255] intensity space
// before the dot product and dropped back afterwards. For the suite's other linear filters this
// shift cancels (their weights sum to 1, so it contributes exactly +128 and the store removes it)
// and stored-space arithmetic is equivalent -- which is why convolve_reference() ignores it. Here
// the weights sum to `strength`, so the shift contributes 128*strength and does NOT cancel: at
// strength 2 an I8 result computed in stored space would be off by 128. Hence the explicit lift.
template <typename T>
void emboss_reference(const T* src, T* dst, const RpptDesc& d, DType dt, const RpptROI* roi,
                      RpptRoiType type, double strength, Rpp32u kernelSize) {
    const std::vector<double> kernel = emboss_kernel(kernelSize, strength);
    const double shift = (dt == DType::I8) ? 128.0 : 0.0;
    filter_reference<T>(src, dst, d, roi, type, kernelSize, [&](const double* w, int kk) {
        double acc = 0.0;
        for (int k = 0; k < kk; ++k) acc += kernel[k] * (w[k] + shift);
        return quantize_stored(acc - shift, dt);
    });
}

}  // namespace rpptest

#endif  // RPP_TEST_EMBOSS_REF_H
