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

#ifndef RPP_TEST_MEL_FILTER_BANK_REF_H
#define RPP_TEST_MEL_FILTER_BANK_REF_H

#include <rpp/rpp.h>

#include <cmath>
#include <vector>

#include "framework/audio_setup.hpp"

namespace rpptest {

/*
Reference model: mel filter bank

RPP op
  rppt_mel_filter_bank   (Audio / Spectrogram)

Description
  Maps each frame of an NFT spectrogram (numBins frequency rows) onto numFilter
  triangular filters spaced evenly on the mel scale. F32 only. Computed in
  double and stored once.

Expression
  nfft    = 2 * (numBins - 1),  bin k is at frequency f_k = k * sampleRate / nfft
  mel_j   = mel(minFreq) + j * (mel(maxFreq) - mel(minFreq)) / (numFilter + 1),
            f_j = hz(mel_j),  0 <= j <= numFilter + 1
  filter m rises over [f_m, f_m+1) and falls over [f_m+1, f_m+2):
    w_m(f) = (f - f_m) / (f_m+1 - f_m)        f_m   <= f < f_m+1
           = (f_m+2 - f) / (f_m+2 - f_m+1)    f_m+1 <= f < f_m+2
           = 0                                otherwise
  dst[m, t] = scale_m * sum over bins with minFreq <= f_k < maxFreq of w_m(f_k) * src[k, t]
  scale_m   = normalize ? 2 / (f_m+2 - f_m) : 1      (equal area per filter)

  HTK:    mel(f) = 1127 ln(1 + f / 700)
  Slaney: mel(f) = 3 f / 200 below 1000 Hz, 15 + 27 ln(f / 1000) / ln(6.4) above

Notes
  - maxFreq = 0 means sampleRate / 2.
*/

inline double hz_to_mel(double hz, RpptMelScaleFormula formula) {
    if (formula == RpptMelScaleFormula::HTK) return 1127.0 * std::log(1.0 + hz / 700.0);
    if (hz < 1000.0) return 3.0 * hz / 200.0;
    return 15.0 + 27.0 * std::log(hz / 1000.0) / std::log(6.4);
}

inline double mel_to_hz(double mel, RpptMelScaleFormula formula) {
    if (formula == RpptMelScaleFormula::HTK) return 700.0 * (std::exp(mel / 1127.0) - 1.0);
    if (mel < 15.0) return 200.0 * mel / 3.0;
    return 1000.0 * std::exp((mel - 15.0) * std::log(6.4) / 27.0);
}

struct MelFilterBankArgs {
    Rpp32f minFreq;
    Rpp32f maxFreq;  // 0: sampleRate / 2
    RpptMelScaleFormula formula;
    Rpp32s numFilter;
    Rpp32f sampleRate;
    bool normalize;
};

// src and dst are NFT planes: srcPlanes[n] = {numBins, numFrames}, dst has numFilter rows.
inline void mel_filter_bank_reference(const Rpp32f* src, const RpptDesc& srcDesc, Rpp32f* dst,
                                      const RpptDesc& dstDesc,
                                      const std::vector<AudioPlane>& srcPlanes,
                                      const MelFilterBankArgs& a) {
    const double maxFreq = a.maxFreq == 0 ? a.sampleRate / 2.0 : a.maxFreq;
    const double melLow = hz_to_mel(a.minFreq, a.formula);
    const double melHigh = hz_to_mel(maxFreq, a.formula);
    std::vector<double> edges(a.numFilter + 2);  // f_j
    for (Rpp32s j = 0; j < a.numFilter + 2; ++j)
        edges[j] = mel_to_hz(melLow + j * (melHigh - melLow) / (a.numFilter + 1), a.formula);

    for (Rpp32u n = 0; n < srcDesc.n; ++n) {
        const Rpp32u numBins = srcPlanes[n].rows;
        const Rpp32u numFrames = srcPlanes[n].cols;
        const double binHz = static_cast<double>(a.sampleRate) / (2 * (numBins - 1));
        for (Rpp32s m = 0; m < a.numFilter; ++m) {
            const double left = edges[m], center = edges[m + 1], right = edges[m + 2];
            const double scale = a.normalize ? 2.0 / (right - left) : 1.0;
            for (Rpp32u t = 0; t < numFrames; ++t) {
                double sum = 0.0;
                for (Rpp32u k = 0; k < numBins; ++k) {
                    const double f = k * binHz;
                    if (f < a.minFreq || f >= maxFreq) continue;
                    double weight = 0.0;
                    if (f >= left && f < center)
                        weight = (f - left) / (center - left);
                    else if (f >= center && f < right)
                        weight = (right - f) / (right - center);
                    sum += weight * src[audio_plane_index(srcDesc, n, k, t)];
                }
                dst[audio_plane_index(dstDesc, n, m, t)] = static_cast<Rpp32f>(scale * sum);
            }
        }
    }
}

}  // namespace rpptest

#endif  // RPP_TEST_MEL_FILTER_BANK_REF_H
