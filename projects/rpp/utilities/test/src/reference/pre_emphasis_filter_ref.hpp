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

#ifndef RPP_TEST_PRE_EMPHASIS_FILTER_REF_H
#define RPP_TEST_PRE_EMPHASIS_FILTER_REF_H

#include <rpp/rpp.h>

#include <vector>

#include "framework/audio_setup.hpp"

namespace rpptest {

/*
Reference model: pre-emphasis filter

RPP op
  rppt_pre_emphasis_filter   (Audio / Filter)

Description
  First-order high-pass filter over each sample, with a per-sample
  coefficient. F32 only. Computed in double and stored once.

Expression
  dst[i] = src[i] - coeff[n] * src[i - 1]

  where src[-1], the value before the first, comes from the border type:
    ZERO     0
    CLAMP    src[0]
    REFLECT  src[1]

Notes
  - The filter runs over srcLengthTensor contiguous floats from the start of
    the sample, with no notion of channels. The HIP backend accepts only
    numDims = 2 (mono), so only mono is modelled.
  - Only each sample's length is written here; the destination past it is
    not modelled.
*/

inline double pre_emphasis_border(const Rpp32f* sample, RpptAudioBorderType border) {
    switch (border) {
        case RpptAudioBorderType::CLAMP:
            return sample[0];
        case RpptAudioBorderType::REFLECT:
            return sample[1];
        case RpptAudioBorderType::ZERO:
        default:
            return 0.0;
    }
}

inline void pre_emphasis_filter_reference(const Rpp32f* src, Rpp32f* dst, const RpptDesc& desc,
                                          const std::vector<Rpp32s>& lengths,
                                          const std::vector<Rpp32f>& coeffs,
                                          RpptAudioBorderType border) {
    for (Rpp32u n = 0; n < desc.n; ++n) {
        const Rpp32f* in = src + static_cast<std::size_t>(n) * desc.strides.nStride;
        Rpp32f* out = dst + static_cast<std::size_t>(n) * desc.strides.nStride;
        const double coeff = coeffs[n];
        for (Rpp32s i = 0; i < lengths[n]; ++i) {
            const double previous = i == 0 ? pre_emphasis_border(in, border) : in[i - 1];
            out[i] = static_cast<Rpp32f>(in[i] - coeff * previous);
        }
    }
}

}  // namespace rpptest

#endif  // RPP_TEST_PRE_EMPHASIS_FILTER_REF_H
