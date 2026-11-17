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

#ifndef RPP_TEST_TO_DECIBELS_REF_H
#define RPP_TEST_TO_DECIBELS_REF_H

#include <rpp/rpp.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "framework/audio_setup.hpp"

namespace rpptest {

/*
Reference model: to decibels

RPP op
  rppt_to_decibels   (Audio / Arithmetic)

Description
  Converts magnitudes to decibels relative to a reference magnitude, with
  the ratio floored at a cut-off. F32 only. Computed in double and stored
  once.

Expression
  ref      = referenceMagnitude, or the sample's maximum if it is 0
  minRatio = 10 ^ (cutOffDB / multiplier)
  dst      = multiplier * log10(max(minRatio, src / ref))

Notes
  - A sample whose maximum is 0 uses ref = 1, so its values are not divided
    by zero.
  - Only the 1D form is modelled: a mono signal, srcDims = (width 1,
    height length). The 2D form (a spectrogram, srcDims = (width, height))
    is not.
*/

inline void to_decibels_reference(const Rpp32f* src, Rpp32f* dst, const RpptDesc& desc,
                                  const std::vector<Rpp32s>& lengths, Rpp32f cutOffDB,
                                  Rpp32f multiplier, Rpp32f referenceMagnitude) {
    const double minRatio = std::pow(10.0, static_cast<double>(cutOffDB) / multiplier);
    for (Rpp32u n = 0; n < desc.n; ++n) {
        const Rpp32f* in = src + static_cast<std::size_t>(n) * desc.strides.nStride;
        Rpp32f* out = dst + static_cast<std::size_t>(n) * desc.strides.nStride;

        double ref = referenceMagnitude;
        if (ref == 0.0) {
            ref = *std::max_element(in, in + lengths[n]);
            if (ref == 0.0) ref = 1.0;
        }
        for (Rpp32s i = 0; i < lengths[n]; ++i)
            out[i] = static_cast<Rpp32f>(multiplier * std::log10(std::max(minRatio, in[i] / ref)));
    }
}

}  // namespace rpptest

#endif  // RPP_TEST_TO_DECIBELS_REF_H
