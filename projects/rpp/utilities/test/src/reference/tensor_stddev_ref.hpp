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

#ifndef RPP_TEST_TENSOR_STDDEV_REF_H
#define RPP_TEST_TENSOR_STDDEV_REF_H

#include <rpp/rpp.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "framework/config_param.hpp"
#include "framework/reduction.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Independent host golden model for rppt_tensor_stddev, derived from the op's definition (the
// channel-wise standard deviation R/G/B and the whole-image standard deviation, per image, over
// the ROI, computed with respect to the provided mean tensor), NOT from the RPP kernel. Used as
// the reference for both backends so kernel bugs surface as diffs.
//
// stddev is the population standard deviation (divide by the element count, not count-1), in the
// same stored intensity space as the data (U8 [0,255], I8 [-128,127], F16/F32 [0,1]). Per image,
// with N = roiWidth*roiHeight pixels per channel and mean in the same [R,G,B,total] layout as the
// output:
//   out[R/G/B] = sqrt( sum_channel( (x - mean_channel)^2 ) / N )
//   out[total] = sqrt( sum_allchannels( (x - mean_total)^2 ) / (3*N) )
// For a 1-channel image the single result is sqrt( sum( (x - mean)^2 ) / N ). `mean` holds the
// exact values handed to the kernel (float-rounded), so the reference deviates against the same
// mean the op uses.
template <typename T>
std::vector<double> tensor_stddev_reference(const T* src, const RpptDesc& d, const RpptROI* roi,
                                            RpptRoiType type, const std::vector<double>& mean) {
    const std::size_t stride = reduction_stride(d);
    const std::vector<std::size_t> N = roi_pixel_counts(d, roi, type);
    std::vector<double> sq(reduction_length(d), 0.0);  // per-channel sum of squared deviations
    std::vector<double> sqImg(d.n, 0.0);               // whole-image sum of squared deviations
    for_each_roi_value(src, d, roi, type, [&](Rpp32u n, Rpp32u c, double v) {
        const double dc = v - mean[n * stride + c];
        sq[n * stride + c] += dc * dc;
        if (d.c == 3) {
            const double di = v - mean[n * stride + 3];
            sqImg[n] += di * di;
        }
    });

    std::vector<double> out(reduction_length(d), 0.0);
    for (Rpp32u n = 0; n < d.n; ++n) {
        for (Rpp32u c = 0; c < d.c; ++c)
            out[n * stride + c] = std::sqrt(sq[n * stride + c] / static_cast<double>(N[n]));
        if (d.c == 3)
            out[n * stride + 3] = std::sqrt(sqImg[n] / (3.0 * static_cast<double>(N[n])));
    }
    return out;
}

}  // namespace rpptest

#endif  // RPP_TEST_TENSOR_STDDEV_REF_H
