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

#ifndef RPP_TEST_NON_SILENT_REGION_DETECTION_REF_H
#define RPP_TEST_NON_SILENT_REGION_DETECTION_REF_H

#include <rpp/rpp.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace rpptest {

/*
Reference model: non-silent region detection

RPP op
  rppt_non_silent_region_detection   (Audio / Statistical)

Description
  Finds the region of each sample whose short-term power reaches a cut-off,
  and returns its start index and length. Mono F32 input. Computed in
  double.

Expression
  mms[i]  = (1 / W) * sum of x[j]^2 for j in [i - W + 1, i], j >= 0
  cutOff  = ref * 10 ^ (cutOffDB / 10)
            ref = referencePower, or max(mms) if it is 0
  first   = first i with mms[i] >= cutOff
  last    = last i with mms[i] >= cutOff
  begin   = max(first - (W - 1), 0)
  length  = last - begin + 1

  If no mms value reaches the cut-off, begin = length = 0.

Notes
  - mms[i] averages the window that ends at i, so a region is detected up
    to W - 1 samples after it starts; begin moves back by W - 1 to make up
    for that. The end is not adjusted.
  - resetInterval only controls how often the kernels recompute their
    running sum to limit float drift. It does not change the result, so it
    is not modelled.
*/

struct NonSilentRegion {
    Rpp32s begin;
    Rpp32s length;
};

inline std::vector<double> moving_mean_square(const Rpp32f* x, Rpp32s length, Rpp32s windowLength) {
    std::vector<double> mms(length);
    for (Rpp32s i = 0; i < length; ++i) {
        double sum = 0.0;
        for (Rpp32s j = std::max(i - windowLength + 1, 0); j <= i; ++j)
            sum += static_cast<double>(x[j]) * x[j];
        mms[i] = sum / windowLength;
    }
    return mms;
}

inline double non_silent_cutoff(const std::vector<double>& mms, Rpp32f cutOffDB,
                                Rpp32f referencePower) {
    const double ref =
        referencePower == 0.0f ? *std::max_element(mms.begin(), mms.end()) : referencePower;
    return ref * std::pow(10.0, cutOffDB / 10.0);
}

inline NonSilentRegion non_silent_region_reference(const Rpp32f* x, Rpp32s length, Rpp32f cutOffDB,
                                                   Rpp32s windowLength, Rpp32f referencePower) {
    const std::vector<double> mms = moving_mean_square(x, length, windowLength);
    const double cutOff = non_silent_cutoff(mms, cutOffDB, referencePower);

    Rpp32s first = -1, last = -1;
    for (Rpp32s i = 0; i < length; ++i)
        if (mms[i] >= cutOff) {
            if (first < 0) first = i;
            last = i;
        }
    if (first < 0) return {0, 0};

    const Rpp32s begin = std::max(first - (windowLength - 1), 0);
    return {begin, last - begin + 1};
}

}  // namespace rpptest

#endif  // RPP_TEST_NON_SILENT_REGION_DETECTION_REF_H
