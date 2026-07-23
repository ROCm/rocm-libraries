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
