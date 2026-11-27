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

#ifndef RPP_TEST_RESAMPLE_REF_H
#define RPP_TEST_RESAMPLE_REF_H

#include <rpp/rpp.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "framework/audio_setup.hpp"

namespace rpptest {

/*
Reference model: resample

RPP op
  rppt_resample   (Audio / Filter)

Description
  Changes each sample's rate from inRate[n] to outRate[n] by convolving it with
  the caller's resampling window. F32 only. Computed in double and stored once.

Expression
  outLength = ceil(length * outRate / inRate)
  pos(o)    = o * inRate / outRate                  (output frame o in input frames)
  dst[o, c] = sum over input frames i with -lobes <= i - pos(o) < lobes, 0 <= i < length
                of src[i, c] * window(i - pos(o))
  window(x) = lookup linearly interpolated at x * scale + center

Notes
  - When inRate == outRate the sample is copied unchanged.
  - Frames past the input's ends contribute nothing (zero padding).
  - The window is an input: lookup, scale, center and lobes come from the
    RpptResamplingWindow the test builds.
*/

inline Rpp32s resampled_length(Rpp32s length, Rpp32f inRate, Rpp32f outRate) {
    return static_cast<Rpp32s>(std::ceil(static_cast<double>(length) * outRate / inRate));
}

inline double resampling_window_weight(const RpptResamplingWindow& window, double x) {
    const double pos = x * window.scale + window.center;
    const Rpp32s i = static_cast<Rpp32s>(std::floor(pos));
    const double frac = pos - i;
    return window.lookup[i] + frac * (window.lookup[i + 1] - window.lookup[i]);
}

inline void resample_reference(const Rpp32f* src, const RpptDesc& srcDesc, Rpp32f* dst,
                               const RpptDesc& dstDesc, const std::vector<Rpp32s>& lengths,
                               const std::vector<Rpp32f>& inRates,
                               const std::vector<Rpp32f>& outRates,
                               const RpptResamplingWindow& window) {
    for (Rpp32u n = 0; n < srcDesc.n; ++n) {
        const Rpp32s outLength = resampled_length(lengths[n], inRates[n], outRates[n]);
        const double step = static_cast<double>(inRates[n]) / outRates[n];
        for (Rpp32s o = 0; o < outLength; ++o)
            for (Rpp32u c = 0; c < srcDesc.c; ++c) {
                double sum = 0.0;
                if (inRates[n] == outRates[n]) {
                    sum = src[audio_index(srcDesc, n, o, c)];
                } else {
                    // the frames i with -lobes <= i - pos < lobes, clipped to the sample
                    const double pos = o * step;
                    const Rpp32s center = static_cast<Rpp32s>(std::ceil(pos));
                    const Rpp32s first = std::max(center - window.lobes, 0);
                    const Rpp32s last = std::min(center + window.lobes, lengths[n]);
                    for (Rpp32s i = first; i < last; ++i)
                        sum += src[audio_index(srcDesc, n, i, c)] *
                               resampling_window_weight(window, i - pos);
                }
                dst[audio_index(dstDesc, n, o, c)] = static_cast<Rpp32f>(sum);
            }
    }
}

}  // namespace rpptest

#endif  // RPP_TEST_RESAMPLE_REF_H
