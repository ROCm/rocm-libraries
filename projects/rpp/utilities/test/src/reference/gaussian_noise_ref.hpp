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

#ifndef RPP_TEST_GAUSSIAN_NOISE_REF_H
#define RPP_TEST_GAUSSIAN_NOISE_REF_H

#include <rpp/rpp.h>

#include "framework/tensor_setup.hpp"

namespace rpptest {

/*
Reference model: gaussian_noise   (degenerate corner only)

RPP op
  rppt_gaussian_noise   (Image / Effects augmentation)

Description
  Adds Gaussian noise drawn from N(mean, stdDev), per the public API doc, with
  meanTensor[i] >= 0 and stdDevTensor[i] >= 0.

Expression
  dst = src + N(mean, stdDev)

Scope
  Only the (mean = 0, stdDev = 0) corner is modelled, where the op is a
  passthrough independent of the seed. The general case depends on the
  kernel's Box-Muller stream, which the public API does not describe.
*/
template <typename T>
void gaussian_noise_identity_reference(const T* src, T* dst, const RpptDesc& d, const RpptROI* roi,
                                       RpptRoiType roiType) {
    for_each_roi_io(d, roi, roiType,
                    [&](Rpp32u, Rpp32u, Rpp32u, Rpp32u, std::size_t srcIdx, std::size_t dstIdx) {
                        dst[dstIdx] = src[srcIdx];
                    });
}

}  // namespace rpptest

#endif  // RPP_TEST_GAUSSIAN_NOISE_REF_H
