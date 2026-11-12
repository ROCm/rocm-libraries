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

#include <stdexcept>
#include <string>
#include <vector>

#include "framework/config_param.hpp"
#include "framework/intensity.hpp"
#include "reference/filter_common.hpp"

namespace rpptest {

/*
Reference model: emboss

RPP op
  rppt_emboss   (Image / Filter augmentation)

Description
  Directional relief filter: an anti-symmetric KxK tap table that lights one
  edge orientation and darkens the opposite. strength scales every tap and is
  clamped at 2.0 from above. Each table sums to 1, so there is no mid-grey
  bias and the DC gain equals strength.

  The taps are applied as a correlation (no kernel flip), like every other
  filter in the suite; this is observable only because the table is
  anti-symmetric.

Expression
  dst(j, i) = sum{ strength * tap[k] * src(j+dy, i+dx) }

Per-type form
  The weights sum to strength rather than 1, so the I8 -128 shift does not
  cancel: the window is lifted into [0,255] intensity space before the dot
  product and dropped back afterwards.

Notes
  The header does not specify the tap tables or the strength convention, so
  the tables mirror the RPP kernel. The window, REPLICATE border, placement
  and quantization come from filter_reference().
*/

// Row-major taps in gather_roi_window() order, one table per documented size (3/5/7/9).
inline std::vector<double> emboss_kernel(Rpp32u kernelSize, double strength) {
    static const double k3[9] = {2, 1, 0, 1, 1, -1, 0, -1, -2};
    static const double k5[25] = {3,  3,  2, 1, 0,  3,  2,  1, 0,  -1, 2,  1, 1,
                                  -1, -2, 1, 0, -1, -2, -3, 0, -1, -2, -3, -3};
    static const double k7[49] = {4,  5, 4,  3,  2,  1,  0,  5,  3,  3,  2,  1,  0,  -1, 4,  3,  2,
                                  1,  0, -1, -2, 3,  2,  1,  1,  -1, -2, -3, 2,  1,  0,  -1, -2, -3,
                                  -4, 1, 0,  -1, -2, -3, -3, -5, 0,  -1, -2, -3, -4, -5, -4};
    static const double k9[81] = {5,  7,  6,  5,  4, 3,  2,  1,  0,  7,  4,  5,  4, 3, 2,  1,  0,
                                  -1, 6,  5,  3,  3, 2,  1,  0,  -1, -2, 5,  4,  3, 2, 1,  0,  -1,
                                  -2, -3, 4,  3,  2, 1,  1,  -1, -2, -3, -4, 3,  2, 1, 0,  -1, -2,
                                  -3, -4, -5, 2,  1, 0,  -1, -2, -3, -3, -5, -6, 1, 0, -1, -2, -3,
                                  -4, -5, -4, -7, 0, -1, -2, -3, -4, -5, -6, -7, -5};
    const double* base = nullptr;
    switch (kernelSize) {
        case 3:
            base = k3;
            break;
        case 5:
            base = k5;
            break;
        case 7:
            base = k7;
            break;
        case 9:
            base = k9;
            break;
        default:
            throw std::invalid_argument("emboss_reference has no tap table for kernelSize " +
                                        std::to_string(kernelSize));
    }
    const std::size_t count = static_cast<std::size_t>(kernelSize) * kernelSize;
    const double scale = (strength > 2.0) ? 2.0 : strength;  // clamped from above only
    std::vector<double> kernel(count);
    for (std::size_t i = 0; i < count; ++i) kernel[i] = base[i] * scale;
    return kernel;
}

template <typename T>
void emboss_reference(const T* src, const RpptDesc& sd, T* dst, const RpptDesc& dd, DType dt,
                      const RpptROI* roi, RpptRoiType type, double strength, Rpp32u kernelSize) {
    const std::vector<double> kernel = emboss_kernel(kernelSize, strength);
    const double shift = (dt == DType::I8) ? 128.0 : 0.0;
    filter_reference<T>(src, sd, dst, dd, roi, type, kernelSize, [&](const double* w, int kk) {
        double acc = 0.0;
        for (int k = 0; k < kk; ++k) acc += kernel[k] * (w[k] + shift);
        return quantize_stored(acc - shift, dt);
    });
}

}  // namespace rpptest

#endif  // RPP_TEST_EMBOSS_REF_H
