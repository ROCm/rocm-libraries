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
  This reference covers only the (mean = 0, stdDev = 0) corner. At stdDev = 0
  the distribution collapses to a point mass at mean, so with mean = 0 every
  sample is exactly 0 and the op degenerates to a passthrough with no RNG
  involved -- bit-exact and reproducible whatever the seed.

  Away from that corner the output depends on the kernel's Box-Muller stream,
  whose exact per-element consumption order is not described by the public
  API, so the general case is deliberately left unmodelled rather than
  guessed. It is covered by parameter-validation checks in the test file, not
  by a golden.
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
