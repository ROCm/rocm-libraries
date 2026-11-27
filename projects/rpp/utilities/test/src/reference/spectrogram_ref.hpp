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

#ifndef RPP_TEST_SPECTROGRAM_REF_H
#define RPP_TEST_SPECTROGRAM_REF_H

#include <rpp/rpp.h>

#include <cmath>
#include <vector>

#include "framework/audio_setup.hpp"

namespace rpptest {

/*
Reference model: spectrogram

RPP op
  rppt_spectrogram   (Audio / Spectrogram)

Description
  Short-time Fourier transform of a mono sample: slides a window along the
  signal, multiplies each frame by the window function and takes the magnitude
  (power 1) or power (power 2) of its DFT. F32 only. Computed in double with a
  direct DFT and stored once.

Expression
  offset     = centerWindows ? windowLength / 2 : 0
  numWindows = (centerWindows ? length : length - windowLength) / windowStep + 1
  frame w:     x[j] = window[j] * signal(w * windowStep - offset + j),  0 <= j < windowLength
  X[k]       = sum_j x[j] * exp(-2 pi i k j / nfft),  0 <= k <= nfft / 2
  out[k, w]  = |X[k]| ^ power        (NFT: frequency rows; NTF: out[w, k])

  signal(i) outside [0, length) is the reflection without repeating the edge
  (-1 -> 1, length -> length - 2) when reflectPadding, otherwise 0.

Notes
  - window is the caller's windowFunction, or when that is null the periodic
    Hann window 0.5 * (1 - cos(2 pi (j + 0.5) / windowLength)).
  - nfft = 0 means nfft = windowLength. When nfft > windowLength the frame is
    zero-padded; where the frame sits within nfft only changes X's phase, so
    the reference puts it at 0.
*/

inline Rpp32s spectrogram_num_windows(Rpp32s length, Rpp32s windowLength, Rpp32s windowStep,
                                      bool centerWindows) {
    return (centerWindows ? length : length - windowLength) / windowStep + 1;
}

inline std::vector<double> spectrogram_hann_window(Rpp32s windowLength) {
    std::vector<double> window(windowLength);
    for (Rpp32s j = 0; j < windowLength; ++j)
        window[j] = 0.5 * (1.0 - std::cos(2.0 * M_PI * (j + 0.5) / windowLength));
    return window;
}

// Index of i reflected into [0, length) without repeating the edge.
inline Rpp32s reflect_index(Rpp32s i, Rpp32s length) {
    if (length < 2) return length - 1;
    while (i < 0 || i >= length) i = i < 0 ? -i : 2 * (length - 1) - i;
    return i;
}

struct SpectrogramArgs {
    bool centerWindows;
    bool reflectPadding;
    std::vector<double> window;  // windowLength values
    Rpp32s nfft;                 // already resolved: never 0
    Rpp32s power;
    Rpp32s windowStep;
};

inline void spectrogram_reference(const Rpp32f* src, const RpptDesc& srcDesc, Rpp32f* dst,
                                  const RpptDesc& dstDesc, const std::vector<Rpp32s>& lengths,
                                  const SpectrogramArgs& a) {
    const Rpp32s windowLength = static_cast<Rpp32s>(a.window.size());
    const Rpp32s numBins = a.nfft / 2 + 1;
    const Rpp32s offset = a.centerWindows ? windowLength / 2 : 0;
    const bool freqRows = dstDesc.layout == NFT;

    // cos and sin of 2 pi m / nfft: the DFT's angles, reduced mod nfft
    std::vector<double> cosTable(a.nfft), sinTable(a.nfft);
    for (Rpp32s m = 0; m < a.nfft; ++m) {
        cosTable[m] = std::cos(2.0 * M_PI * m / a.nfft);
        sinTable[m] = std::sin(2.0 * M_PI * m / a.nfft);
    }

    std::vector<double> frame(windowLength);
    for (Rpp32u n = 0; n < srcDesc.n; ++n) {
        const Rpp32s length = lengths[n];
        const Rpp32s numWindows =
            spectrogram_num_windows(length, windowLength, a.windowStep, a.centerWindows);
        for (Rpp32s w = 0; w < numWindows; ++w) {
            for (Rpp32s j = 0; j < windowLength; ++j) {
                Rpp32s i = w * a.windowStep - offset + j;
                double value = 0.0;
                if (i >= 0 && i < length)
                    value = src[audio_index(srcDesc, n, i, 0)];
                else if (a.reflectPadding)
                    value = src[audio_index(srcDesc, n, reflect_index(i, length), 0)];
                frame[j] = a.window[j] * value;
            }
            for (Rpp32s k = 0; k < numBins; ++k) {
                double re = 0.0, im = 0.0;
                for (Rpp32s j = 0; j < windowLength; ++j) {
                    const Rpp32s m = (k * j) % a.nfft;
                    re += frame[j] * cosTable[m];
                    im -= frame[j] * sinTable[m];
                }
                const double powerValue = re * re + im * im;
                const double out = a.power == 2 ? powerValue : std::sqrt(powerValue);
                const std::size_t idx = freqRows ? audio_plane_index(dstDesc, n, k, w)
                                                 : audio_plane_index(dstDesc, n, w, k);
                dst[idx] = static_cast<Rpp32f>(out);
            }
        }
    }
}

}  // namespace rpptest

#endif  // RPP_TEST_SPECTROGRAM_REF_H
