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

#ifndef RPP_TEST_GAUSSIAN_FILTER_REF_H
#define RPP_TEST_GAUSSIAN_FILTER_REF_H

#include <rpp/rpp.h>

#include <cmath>
#include <vector>

#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/filter_common.hpp"

namespace rpptest {

// Independent host golden model for rppt_gaussian_filter, derived from the Gaussian-blur definition
// (each output pixel is a normalized Gaussian-weighted sum of its KxK neighbourhood, per channel,
// REPLICATE border), NOT from the RPP kernel. Used as the reference for both backends so kernel bugs
// surface as diffs.
//
// The weight at offset (dy,dx) relative to the window centre is exp(-(dy*dy + dx*dx)/(2*sigma*sigma))
// for dy,dx in [-r, r] (r = kernelSize/2), and the whole KxK kernel is normalized so its weights sum
// to 1.0 (preserves DC). Weights are laid out row-major (dy = -r..r outer, dx = -r..r inner) to match
// gather_roi_window; convolve_reference applies the window/border/quantization.
template <typename T>
void gaussian_filter_reference(const T* src, T* dst, const RpptDesc& d, DType dt, const RpptROI* roi,
                               RpptRoiType type, Rpp32u kernelSize, double stdDev) {
    const int r = static_cast<int>(kernelSize / 2);
    const double twoSigmaSq = 2.0 * stdDev * stdDev;
    std::vector<double> kernel;
    kernel.reserve(kernelSize * kernelSize);
    double sum = 0.0;
    for (int dy = -r; dy <= r; ++dy)
        for (int dx = -r; dx <= r; ++dx) {
            const double w = std::exp(-static_cast<double>(dy * dy + dx * dx) / twoSigmaSq);
            kernel.push_back(w);
            sum += w;
        }
    for (double& w : kernel) w /= sum;
    convolve_reference<T>(src, dst, d, dt, roi, type, kernelSize, kernel);
}

}  // namespace rpptest

#endif  // RPP_TEST_GAUSSIAN_FILTER_REF_H
