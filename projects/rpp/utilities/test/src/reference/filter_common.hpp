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

#ifndef RPP_TEST_FILTER_COMMON_REF_H
#define RPP_TEST_FILTER_COMMON_REF_H

#include <rpp/rpp.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Shared primitives for the KxK spatial-window golden models (box / gaussian / median / sobel
// filters, erode / dilate morphology). These are derived from the operators' definitions (a KxK
// window sampled per output pixel with a clamp-to-edge / REPLICATE border), NOT from the RPP
// kernels, so kernel bugs surface as diffs.
//
// Border & ROI model: each op treats the ROI as its working image. For output-local pixel (j,i)
// the KxK window samples ROI-local neighbours (j+dy, i+dx) for dy,dx in [-r, r] (r = kernelSize/2),
// each neighbour coordinate clamped to the ROI bounds [0, roiH-1] x [0, roiW-1] (REPLICATE). The
// source/destination element mapping mirrors for_each_roi_io (framework/tensor_setup.hpp): source
// read at the ROI offset, output written packed at the destination origin.

// Gathers the KxK neighbourhood of output-local pixel (j,i) for the channel plane whose element
// base offset is `base`, clamping each neighbour to the ROI bounds (REPLICATE border), into
// window[0 .. kernelSize*kernelSize-1] in row-major order (dy = -r..r outer, dx = -r..r inner).
template <typename T>
inline void gather_roi_window(const T* src, const RpptDesc& d, const RoiBounds& b,
                              std::size_t base, int j, int i, int r, double* window) {
    const int roiH = static_cast<int>(b.h);
    const int roiW = static_cast<int>(b.w);
    int k = 0;
    for (int dy = -r; dy <= r; ++dy)
        for (int dx = -r; dx <= r; ++dx) {
            const int sy = std::min(std::max(j + dy, 0), roiH - 1);
            const int sx = std::min(std::max(i + dx, 0), roiW - 1);
            window[k++] = to_double(src[plane_index(d, base, b.y0 + sy, b.x0 + sx)]);
        }
}

// The one window-filter driver: per channel over each image's ROI it gathers the KxK window and
// stores reduce(window, kk), which returns the output already in stored units. Every KxK op in the
// suite goes through this, so the window, the border and the placement have a single definition.
template <typename T, typename Reduce>
void filter_reference(const T* src, T* dst, const RpptDesc& d, const RpptROI* roi, RpptRoiType type,
                      Rpp32u kernelSize, Reduce reduce) {
    const int r = static_cast<int>(kernelSize / 2);
    const int kk = static_cast<int>(kernelSize * kernelSize);
    std::vector<double> window(kk);
    for_each_roi_plane(d, roi, type, [&](Rpp32u, const RoiBounds& b, Rpp32u, std::size_t base) {
        for (int j = 0; j < static_cast<int>(b.h); ++j)
            for (int i = 0; i < static_cast<int>(b.w); ++i) {
                gather_roi_window(src, d, b, base, j, i, r, window.data());
                dst[plane_index(d, base, j, i)] = from_double<T>(reduce(window.data(), kk));
            }
    });
}

// Applies a KxK linear filter (kernel row-major, length kernelSize*kernelSize, same dy/dx order as
// gather_roi_window). The weighted sum is quantized back to the dtype via quantize_stored --
// integers round to nearest and clamp, floats clamp to [0,1] -- which is the intended semantics
// (round-to-nearest, not truncate; see the systemic I8 round-vs-truncate finding). Used by
// box_filter and gaussian_filter.
template <typename T>
void convolve_reference(const T* src, T* dst, const RpptDesc& d, DType dt, const RpptROI* roi,
                        RpptRoiType type, Rpp32u kernelSize, const std::vector<double>& kernel) {
    filter_reference<T>(src, dst, d, roi, type, kernelSize, [&](const double* w, int kk) {
        double acc = 0.0;
        for (int k = 0; k < kk; ++k) acc += kernel[k] * w[k];
        return quantize_stored(acc, dt);
    });
}

}  // namespace rpptest

#endif  // RPP_TEST_FILTER_COMMON_REF_H
