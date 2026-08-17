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

#ifndef RPP_TEST_TENSOR_SUM_REF_H
#define RPP_TEST_TENSOR_SUM_REF_H

#include <rpp/rpp.h>

#include <cstddef>
#include <vector>

#include "framework/config_param.hpp"
#include "framework/reduction.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Independent host golden model for rppt_tensor_sum, derived from the op's definition (the
// channel-wise sum R/G/B and the total sum, per image, over the ROI in raw intensity space),
// NOT from the RPP kernel. Used as the reference for both backends so kernel bugs surface as
// diffs.
//
// Per image: out[R]=sum(R), out[G]=sum(G), out[B]=sum(B), out[total]=sum(R)+sum(G)+sum(B).
// For a 1-channel image the single result is that channel's sum. Sums are exact in the stored
// intensity space (U8 [0,255], I8 [-128,127], F16/F32 [0,1]); the golden accumulates in double.
template <typename T>
std::vector<double> tensor_sum_reference(const T* src, const RpptDesc& d, const RpptROI* roi,
                                         RpptRoiType type) {
    const std::size_t stride = reduction_stride(d);
    std::vector<double> out(reduction_length(d), 0.0);
    for_each_roi_value(src, d, roi, type,
                       [&](Rpp32u n, Rpp32u c, double v) { out[n * stride + c] += v; });
    if (d.c == 3)
        for (Rpp32u n = 0; n < d.n; ++n)
            out[n * stride + 3] =
                out[n * stride + 0] + out[n * stride + 1] + out[n * stride + 2];
    return out;
}

}  // namespace rpptest

#endif  // RPP_TEST_TENSOR_SUM_REF_H
