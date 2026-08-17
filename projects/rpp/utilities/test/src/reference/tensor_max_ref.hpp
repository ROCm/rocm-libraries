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

#ifndef RPP_TEST_TENSOR_MAX_REF_H
#define RPP_TEST_TENSOR_MAX_REF_H

#include <rpp/rpp.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "framework/config_param.hpp"
#include "framework/reduction.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Independent host golden model for rppt_tensor_max, derived from the op's definition (the
// channel-wise maximum R/G/B and the overall maximum across all channels, per image, over the
// ROI), NOT from the RPP kernel. Used as the reference for both backends so kernel bugs surface
// as diffs.
//
// Per image: out[R/G/B] = max of that channel, out[total] = max over all three channels. For a
// 1-channel image the single result is that channel's maximum. max selects an existing element,
// so there is no arithmetic and the result is exact for every dtype.
template <typename T>
std::vector<double> tensor_max_reference(const T* src, const RpptDesc& d, const RpptROI* roi,
                                         RpptRoiType type) {
    const std::size_t stride = reduction_stride(d);
    std::vector<double> out(reduction_length(d), -std::numeric_limits<double>::infinity());
    for_each_roi_value(src, d, roi, type, [&](Rpp32u n, Rpp32u c, double v) {
        out[n * stride + c] = std::max(out[n * stride + c], v);
    });
    if (d.c == 3)
        for (Rpp32u n = 0; n < d.n; ++n)
            out[n * stride + 3] = std::max({out[n * stride + 0], out[n * stride + 1],
                                            out[n * stride + 2]});
    return out;
}

}  // namespace rpptest

#endif  // RPP_TEST_TENSOR_MAX_REF_H
