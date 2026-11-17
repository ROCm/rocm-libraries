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

#ifndef RPP_TEST_DOWN_MIXING_REF_H
#define RPP_TEST_DOWN_MIXING_REF_H

#include <rpp/rpp.h>

#include <vector>

#include "framework/audio_setup.hpp"

namespace rpptest {

/*
Reference model: down mixing

RPP op
  rppt_down_mixing   (Audio / DataExchange)

Description
  Mixes an interleaved multi-channel sample down to mono by averaging its
  channels. F32 only. Computed in double and stored once.

Expression
  dst[t] = (1 / C) * sum_c src[t, c]

Notes
  - Every channel has the same weight, 1 / C, so normalizeWeights (scale the
    weights to sum to 1) does not change the result.
  - A mono source is copied unchanged.
  - The destination is mono: dstDesc has one channel and the same lengths.
*/

inline void down_mixing_reference(const Rpp32f* src, const RpptDesc& srcDesc, Rpp32f* dst,
                                  const RpptDesc& dstDesc, const std::vector<Rpp32s>& lengths) {
    for (Rpp32u n = 0; n < srcDesc.n; ++n)
        for (Rpp32u t = 0; t < static_cast<Rpp32u>(lengths[n]); ++t) {
            double sum = 0.0;
            for (Rpp32u c = 0; c < srcDesc.c; ++c) sum += src[audio_index(srcDesc, n, t, c)];
            dst[audio_index(dstDesc, n, t, 0)] = static_cast<Rpp32f>(sum / srcDesc.c);
        }
}

}  // namespace rpptest

#endif  // RPP_TEST_DOWN_MIXING_REF_H
