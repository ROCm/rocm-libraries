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

#include <gtest/gtest.h>
#include <rpp/rpp.h>

#include <cmath>
#include <cstddef>
#include <sstream>
#include <vector>

#include "framework/tensor_setup.hpp"

// Shared helpers for the statistical reduction ops (tensor_sum/min/max/mean/stddev). These ops
// are reductions: they emit a small result array of per-image statistics, not a full image, so
// they need their own harness rather than the pointwise for_each_roi_io / compare_roi path.
//
// Result-array layout (from the API docs): a 3-channel image produces 4 values per image in the
// order [R, G, B, total]; a 1-channel image produces 1 value per image. So the array length is
// n*4 for c==3 and n for c==1, and the per-image stride is 4 or 1.
namespace rpptest {

inline std::size_t reduction_stride(const RpptDesc& d) {
    return d.c == 1 ? 1 : 4;
}

inline std::size_t reduction_length(const RpptDesc& d) {
    return static_cast<std::size_t>(d.n) * reduction_stride(d);
}

// Per-image ROI pixel count (single channel), N = roiWidth * roiHeight. Reductions average /
// normalize over these counts (per channel), or over 3*N for the whole-image statistic.
inline std::vector<std::size_t> roi_pixel_counts(const RpptDesc& d, const RpptROI* roi,
                                                 RpptRoiType type) {
    std::vector<std::size_t> counts(d.n);
    for (Rpp32u i = 0; i < d.n; ++i) {
        const RoiBounds b = roi_bounds(roi[i], type);
        counts[i] = static_cast<std::size_t>(b.w) * b.h;
    }
    return counts;
}

// Walks each image's ROI and invokes acc(n, c, value) for every source element, value being the
// stored pixel as a double (raw intensity space: U8 [0,255], I8 [-128,127], F16/F32 [0,1]). The
// shared accumulation primitive the reduction references build on; it reuses for_each_roi_io so
// the ROI walk matches the rest of the suite.
template <typename T, typename Acc>
void for_each_roi_value(const T* src, const RpptDesc& d, const RpptROI* roi, RpptRoiType type,
                        Acc acc) {
    for_each_roi_io(d, roi, type,
                    [&](Rpp32u n, Rpp32u c, Rpp32u, Rpp32u, std::size_t srcIdx, std::size_t) {
                        acc(n, c, to_double(src[srcIdx]));
                    });
}

// Compares a typed reduction output array against a double-valued golden within tolerance.
// Returns a rich GTest AssertionResult: on failure it reports the total mismatch count and lists
// the first few offending slots (flat index, actual, golden, diff, tol) so the failure is
// self-describing in both the console and the XML report. TOut is the op's output element type
// (e.g. Rpp64u for U8 sum, Rpp8u for U8 min, Rpp32f for mean/stddev).
template <typename TOut>
::testing::AssertionResult compare_reduction(const TOut* actual, const std::vector<double>& golden,
                                             double tol) {
    constexpr std::size_t maxShown = 10;
    std::size_t mismatches = 0;
    std::ostringstream details;
    for (std::size_t i = 0; i < golden.size(); ++i) {
        const double a = to_double(actual[i]);
        const double diff = std::fabs(a - golden[i]);
        if (diff <= tol) continue;
        if (mismatches < maxShown)
            details << "\n  [" << i << "] actual=" << a << " golden=" << golden[i]
                    << " diff=" << diff << " tol=" << tol;
        ++mismatches;
    }
    if (mismatches == 0) return ::testing::AssertionSuccess();
    if (mismatches > maxShown) details << "\n  ... (" << (mismatches - maxShown) << " more)";
    return ::testing::AssertionFailure() << mismatches << " of " << golden.size()
                                         << " reduction values exceeded tolerance:" << details.str();
}

}  // namespace rpptest

#endif  // RPP_TEST_REDUCTION_H
