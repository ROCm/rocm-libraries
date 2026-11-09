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

#ifndef RPP_TEST_REDUCTION_H
#define RPP_TEST_REDUCTION_H

#include <rpp/rpp.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "framework/intensity.hpp"
#include "framework/tensor_setup.hpp"

// Helpers for the reduction ops (tensor_sum/min/max/mean/stddev), which emit per-image statistics
// rather than an image.
//
// Result-array layout (from the API docs): [R, G, B, total] per 3-channel image, one value per
// 1-channel image.
namespace rpptest {

inline std::size_t reduction_stride(const RpptDesc& d) {
    return d.c == 1 ? 1 : 4;
}

inline std::size_t reduction_length(const RpptDesc& d) {
    return static_cast<std::size_t>(d.n) * reduction_stride(d);
}

// Per-image ROI pixel count for a single channel (roiWidth * roiHeight).
inline std::vector<std::size_t> roi_pixel_counts(const RpptDesc& d, const RpptROI* roi,
                                                 RpptRoiType type) {
    std::vector<std::size_t> counts(d.n);
    for (Rpp32u i = 0; i < d.n; ++i) {
        const RoiBounds b = roi_bounds(roi[i], type);
        counts[i] = static_cast<std::size_t>(b.w) * b.h;
    }
    return counts;
}

// Invokes acc(n, c, value) for every source element of each image's ROI, value being the stored
// pixel as a double.
template <typename T, typename Acc>
void for_each_roi_value(const T* src, const RpptDesc& d, const RpptROI* roi, RpptRoiType type,
                        Acc acc) {
    for_each_roi_io(d, roi, type,
                    [&](Rpp32u n, Rpp32u c, Rpp32u, Rpp32u, std::size_t srcIdx, std::size_t) {
                        acc(n, c, to_double(src[srcIdx]));
                    });
}

}  // namespace rpptest

#endif  // RPP_TEST_REDUCTION_H
