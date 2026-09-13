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

#ifndef RPP_TEST_COMPARE_H
#define RPP_TEST_COMPARE_H

#include <gtest/gtest.h>
#include <rpp/rpp.h>

#include <cmath>
#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

#include "framework/generic_tensor_setup.hpp"
#include "framework/intensity.hpp"
#include "framework/nd_config_param.hpp"
#include "framework/tensor_setup.hpp"
#include "framework/tolerance.hpp"
#include "framework/voxel_config_param.hpp"
#include "framework/voxel_tensor_setup.hpp"

// Every comparison the suite makes goes through this header: the image ROI walk, the ND tensor
// walk, the voxel box walk, and the flat reduction array. They differ only in how they enumerate
// elements and how they name a coordinate -- the tolerance rule, the mismatch accounting and the
// failure message are shared, so a change to any of those lands in one place and every domain
// reports the same way.

namespace rpptest {

// Mismatches beyond this many are counted but not listed; the count is always exact.
inline constexpr std::size_t kMaxReportedMismatches = 10;

// Accumulates the out-of-tolerance elements of one comparison and renders the verdict. The walk
// is never cut short at the first mismatch: "142 of 5184 exceeded tolerance" separates a
// systematic break from a single bad lane, which is the first thing you want to know, and the
// listed sample shows where it starts.
class Mismatches {
   public:
    explicit Mismatches(Bound bound) : bound_(bound) {}

    // `where` names the element, in whatever coordinate system the caller walks.
    void check(double actual, double reference, const std::string& where) {
        ++visited_;
        const double diff = std::fabs(actual - reference);
        const double tolerance = bound_(reference);
        if (diff <= tolerance) return;
        if (count_ < kMaxReportedMismatches)
            detail_ << "\n  at " << where << ": actual=" << actual << " reference=" << reference
                    << " diff=" << diff << " tolerance=" << tolerance;
        ++count_;
    }

    ::testing::AssertionResult result() const {
        if (count_ == 0) return ::testing::AssertionSuccess();
        ::testing::AssertionResult failure = ::testing::AssertionFailure();
        failure << count_ << " of " << visited_ << " values exceeded tolerance:" << detail_.str();
        if (count_ > kMaxReportedMismatches)
            failure << "\n  ... (" << (count_ - kMaxReportedMismatches) << " more)";
        return failure;
    }

   private:
    Bound bound_;
    std::size_t count_ = 0;
    std::size_t visited_ = 0;
    std::ostringstream detail_;
};

// ---- coordinate rendering --------------------------------------------------

// Named axes, e.g. "n=1 c=2 row=3 col=4". Anonymous axes use bracket_coord below.
inline std::string named_coord(std::initializer_list<std::pair<const char*, std::size_t>> axes) {
    std::ostringstream out;
    const char* sep = "";
    for (const auto& [name, value] : axes) {
        out << sep << name << "=" << value;
        sep = " ";
    }
    return out.str();
}

// Positional axes, e.g. "[1,0,4,7]".
inline std::string bracket_coord(const NdDims& coord) {
    std::ostringstream out;
    out << "[";
    for (std::size_t axis = 0; axis < coord.size(); ++axis) out << (axis ? "," : "") << coord[axis];
    out << "]";
    return out.str();
}

// ---- image domain ----------------------------------------------------------

// Element-wise comparison over the destination ROI region only -- the area RPP actually writes;
// what it leaves outside is not documented, so the suite does not assert it.
template <typename T>
::testing::AssertionResult compare_roi(const T* actual, const T* reference, const RpptDesc& d,
                                       const RpptROI* roi, RpptRoiType roiType, Bound bound) {
    Mismatches mismatches(bound);
    for_each_roi_io(d, roi, roiType,
                    [&](Rpp32u n, Rpp32u c, Rpp32u j, Rpp32u i, std::size_t, std::size_t dstIdx) {
                        mismatches.check(to_double(actual[dstIdx]), to_double(reference[dstIdx]),
                                         named_coord({{"n", n}, {"c", c}, {"row", j}, {"col", i}}));
                    });
    return mismatches.result();
}

// ---- ND (generic tensor) domain --------------------------------------------

// Compares the logical elements only -- padding slack is not data.
template <typename T>
::testing::AssertionResult compare_nd(const T* actual, const T* reference, const RpptGenericDesc& d,
                                      Bound bound) {
    Mismatches mismatches(bound);
    for_each_nd_coord(d, [&](const NdDims& coord) {
        const std::size_t i = nd_offset(d, coord);
        mismatches.check(to_double(actual[i]), to_double(reference[i]), bracket_coord(coord));
    });
    return mismatches.result();
}

// ---- voxel domain ----------------------------------------------------------

// Compares the ROI box's output region only -- the destination-origin block the op fills.
template <typename T>
::testing::AssertionResult compare_voxel_roi(const T* actual, const T* reference,
                                             const RpptGenericDesc& desc, const RpptROI3D* roi,
                                             Roi3D type, Bound bound) {
    Mismatches mismatches(bound);
    for_each_voxel_roi_io(
        desc, roi, type,
        [&](Rpp32u n, Rpp32u c, Rpp32u z, Rpp32u y, Rpp32u x, std::size_t, std::size_t idx) {
            mismatches.check(to_double(actual[idx]), to_double(reference[idx]),
                             named_coord({{"n", n}, {"c", c}, {"z", z}, {"y", y}, {"x", x}}));
        });
    return mismatches.result();
}

// ---- reductions ------------------------------------------------------------

// Compares a typed reduction output array against a double-valued golden. TOut is the op's output
// element type (e.g. Rpp64u for U8 sum, Rpp8u for U8 min, Rpp32f for mean/stddev).
template <typename TOut>
::testing::AssertionResult compare_reduction(const TOut* actual, const std::vector<double>& golden,
                                             Bound bound) {
    Mismatches mismatches(bound);
    for (std::size_t i = 0; i < golden.size(); ++i)
        mismatches.check(to_double(actual[i]), golden[i], "[" + std::to_string(i) + "]");
    return mismatches.result();
}

}  // namespace rpptest

#endif  // RPP_TEST_COMPARE_H
