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

#ifndef RPP_TEST_COMPARE_TENSOR_H
#define RPP_TEST_COMPARE_TENSOR_H

#include <gtest/gtest.h>
#include <rpp/rpp.h>

#include <cmath>

#include "framework/tensor_setup.hpp"

namespace rpptest {

// Element-wise tolerance comparison over the destination ROI region only (the area RPP
// actually writes; the rest is intentionally not compared). Returns a rich GTest
// AssertionResult naming the first offending output coordinate.
template <typename T>
::testing::AssertionResult compare_roi(const T* actual, const T* reference, const RpptDesc& d,
                                       const RpptROI* roi, RpptRoiType roiType, double tolerance) {
    ::testing::AssertionResult result = ::testing::AssertionSuccess();
    bool failed = false;
    for_each_roi_io(d, roi, roiType,
                    [&](Rpp32u n, Rpp32u c, Rpp32u j, Rpp32u i, std::size_t, std::size_t dstIdx) {
                        if (failed) return;
                        const double a = to_double(actual[dstIdx]);
                        const double r = to_double(reference[dstIdx]);
                        const double diff = std::fabs(a - r);
                        if (diff > tolerance) {
                            failed = true;
                            result = ::testing::AssertionFailure()
                                     << "mismatch at n=" << n << " c=" << c << " row=" << j
                                     << " col=" << i << ": actual=" << a << " reference=" << r
                                     << " diff=" << diff << " tolerance=" << tolerance;
                        }
                    });
    return result;
}

}  // namespace rpptest

#endif  // RPP_TEST_COMPARE_TENSOR_H
