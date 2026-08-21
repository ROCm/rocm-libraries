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

#ifndef RPP_TEST_MEDIAN_FILTER_REF_H
#define RPP_TEST_MEDIAN_FILTER_REF_H

#include <rpp/rpp.h>

#include <algorithm>

#include "framework/config_param.hpp"
#include "reference/filter_common.hpp"

namespace rpptest {

/*
Reference model: median_filter

RPP op
  rppt_median_filter   (Image / Filter augmentation)

Description
  Per-channel median over a KxK window -- a rank filter, not a convolution.
  The window filter_reference gathers (same clamp-to-ROI REPLICATE border as
  the other filters) is sorted and the middle element selected.

Expression
  dst(j, i) = median{ src(j+dy, i+dx) : dy, dx in [-r, r] }

Per-type form
  kernelSize is odd, so K^2 is odd and the median is an EXISTING pixel value
  rather than an average of two. No arithmetic is performed, so the
  to_double/from_double round-trip is exact for every type and the result is
  bit-exact (tolerance 0).
*/
template <typename T>
void median_filter_reference(const T* src, T* dst, const RpptDesc& d, DType /*dt*/,
                             const RpptROI* roi, RpptRoiType type, Rpp32u kernelSize) {
    filter_reference<T>(src, dst, d, roi, type, kernelSize, [](double* w, int kk) {
        std::sort(w, w + kk);
        return w[kk / 2];
    });
}

}  // namespace rpptest

#endif  // RPP_TEST_MEDIAN_FILTER_REF_H
