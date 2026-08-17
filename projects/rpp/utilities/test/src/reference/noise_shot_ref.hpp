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

#ifndef RPP_TEST_NOISE_SHOT_REF_H
#define RPP_TEST_NOISE_SHOT_REF_H

#include <rpp/rpp.h>

#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Independent host golden model for rppt_shot_noise, derived from the op's public API doc
// ("adds Poisson/shot noise based on a user defined shotNoiseFactor ... shotNoiseFactorTensor[i]
// >= 0"), NOT from the RPP kernel.
//
// This reference covers only the shotNoiseFactor = 0 corner: shot noise is conventionally
// modeled by scaling a pixel's intensity into a photon count by 1/factor, drawing a Poisson
// sample, then scaling back by factor -- a formula that is undefined (division by zero) at
// factor = 0 and therefore degenerates to a no-op passthrough at that corner. The general-case
// Poisson scaling for factor > 0 is not derivable from the public API doc comment alone (the
// exact photon-count normalization is unspecified), so it is deliberately left unmodeled here;
// factor > 0 is instead covered by runtime invariants in the test file (valid-range sanity and
// seed determinism), not by this golden. Used as the reference for both backends so kernel bugs
// at the factor = 0 corner surface as diffs.
template <typename T>
void noise_shot_identity_reference(const T* src, T* dst, const RpptDesc& d, const RpptROI* roi,
                                   RpptRoiType roiType) {
    for_each_roi_io(d, roi, roiType,
                    [&](Rpp32u, Rpp32u, Rpp32u, Rpp32u, std::size_t srcIdx, std::size_t dstIdx) {
                        dst[dstIdx] = src[srcIdx];
                    });
}

}  // namespace rpptest

#endif  // RPP_TEST_NOISE_SHOT_REF_H
