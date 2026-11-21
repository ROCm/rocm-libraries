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

#ifndef RPP_TEST_AUDIO_SIGNAL_H
#define RPP_TEST_AUDIO_SIGNAL_H

#include <rpp/rpp.h>

#include <cmath>
#include <cstdint>

// Deterministic test signals in [-1, 1]. Each is a callable (n, t, c) -> Rpp32f of sample, frame
// and channel, so a test picks the content its op needs and make_audio_input() lays it out. The
// content depends on n and c, so a kernel that mixes up samples or channels changes the output.

namespace rpptest {

// Two sines, peak amplitude 0.9. Frequencies are in cycles per frame.
struct Tones {
    Rpp32f operator()(Rpp32u n, Rpp32u t, Rpp32u c) const {
        const double twoPi = 2.0 * M_PI;
        const double f1 = 0.011 + 0.003 * n + 0.005 * c;
        const double f2 = 0.13 + 0.007 * n;
        return static_cast<Rpp32f>(0.5 * std::sin(twoPi * f1 * t) +
                                   0.4 * std::sin(twoPi * f2 * t + 0.3 * c));
    }
};

// Uniform white noise in [-1, 1). A hash of (seed, n, t, c), not a std:: distribution: those are
// implementation-defined, and this keeps the value independent of the order elements are visited.
struct Noise {
    std::uint64_t seed = 0;

    Rpp32f operator()(Rpp32u n, Rpp32u t, Rpp32u c) const {
        std::uint64_t x = seed ^ (static_cast<std::uint64_t>(n) << 48) ^
                          (static_cast<std::uint64_t>(c) << 40) ^ t;
        // SplitMix64 finalizer
        x += 0x9E3779B97F4A7C15ull;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
        x ^= x >> 31;
        const double unit = static_cast<double>(x >> 11) / static_cast<double>(1ull << 53);
        return static_cast<Rpp32f>(2.0 * unit - 1.0);
    }
};

}  // namespace rpptest

#endif  // RPP_TEST_AUDIO_SIGNAL_H
