/*
MIT License

Copyright (c) 2019 - 2026 Advanced Micro Devices, Inc.

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

#ifndef RPP_CPU_FFT_HPP
#define RPP_CPU_FFT_HPP

// In-house forward FFT (float32, unnormalized) used as a drop-in replacement for the FFTS
// library in the CPU spectrogram kernel. It computes the forward DFT
//     X[k] = sum_{n=0}^{N-1} x[n] * exp(-2*pi*i*k*n/N)
// with the same (unnormalized) scaling FFTS uses, so downstream magnitude/power output matches.
//
// Arbitrary transform sizes are supported:
//   - power-of-two N: iterative radix-2 Cooley-Tukey
//   - all other N:    Bluestein's chirp-z algorithm (reduces to a power-of-two convolution)
//
// A plan holds only read-only precomputed tables, so a single plan may be shared across OpenMP
// threads; all mutable work is done through caller-supplied scratch buffers (one per thread).

#include <cmath>
#include <complex>
#include <utility>
#include <vector>

#include "rppdefs.h"

typedef std::complex<Rpp32f> RppFftComplex;

struct RppCpuFftPlan {
    Rpp32s n = 0;  // transform size (nfft)
    bool isPow2 = false;

    // radix-2 tables for size n (used when isPow2 == true)
    std::vector<Rpp32u> bitRev;
    std::vector<RppFftComplex> twiddles;  // twiddles[k] = exp(-2*pi*i*k/n), k in [0, n/2)

    // Bluestein tables (used when isPow2 == false)
    Rpp32s m = 0;  // convolution length = next power of two >= 2n-1
    std::vector<Rpp32u> bitRevM;
    std::vector<RppFftComplex> twiddlesM;  // twiddles[k] = exp(-2*pi*i*k/m), k in [0, m/2)
    std::vector<RppFftComplex> chirp;      // chirp[k] = exp(-pi*i*k^2/n), k in [0, n)
    std::vector<RppFftComplex> kernelFft;  // FFT_m(padded conj(chirp)), size m
};

inline bool rpp_fft_is_pow2(Rpp32s n) {
    return (n > 0) && ((n & (n - 1)) == 0);
}

inline Rpp32s rpp_fft_next_pow2(Rpp32s n) {
    Rpp32s p = 1;
    while (p < n) p <<= 1;
    return p;
}

// Bit-reversal permutation table for a power-of-two length
inline void rpp_fft_build_bitrev(std::vector<Rpp32u>& bitRev, Rpp32s len) {
    bitRev.resize(len);
    Rpp32s logN = 0;
    while ((1 << logN) < len) logN++;
    for (Rpp32s i = 0; i < len; i++) {
        Rpp32u r = 0;
        for (Rpp32s j = 0; j < logN; j++)
            if (i & (1 << j)) r |= (1u << (logN - 1 - j));
        bitRev[i] = r;
    }
}

// Forward twiddle factors twiddles[k] = exp(-2*pi*i*k/len) for k in [0, len/2)
inline void rpp_fft_build_twiddles(std::vector<RppFftComplex>& tw, Rpp32s len) {
    tw.resize(len / 2);
    for (Rpp32s k = 0; k < len / 2; k++) {
        Rpp64f ang = -2.0 * M_PI * (Rpp64f)k / (Rpp64f)len;
        tw[k] = RppFftComplex((Rpp32f)std::cos(ang), (Rpp32f)std::sin(ang));
    }
}

// In-place forward radix-2 DIT FFT (unnormalized), using precomputed bit-reversal and twiddles.
inline void rpp_fft_radix2_forward(RppFftComplex* data, Rpp32s len,
                                   const std::vector<Rpp32u>& bitRev,
                                   const std::vector<RppFftComplex>& tw) {
    for (Rpp32s i = 0; i < len; i++) {
        Rpp32u j = bitRev[i];
        if ((Rpp32u)i < j) std::swap(data[i], data[j]);
    }
    for (Rpp32s size = 2; size <= len; size <<= 1) {
        Rpp32s half = size >> 1;
        Rpp32s step = len / size;  // twiddle stride into tw (length len/2)
        for (Rpp32s i = 0; i < len; i += size) {
            Rpp32s k = 0;
            for (Rpp32s j = i; j < i + half; j++, k += step) {
                RppFftComplex t = tw[k] * data[j + half];
                data[j + half] = data[j] - t;
                data[j] = data[j] + t;
            }
        }
    }
}

// In-place inverse FFT (normalized by 1/len) via the conjugation identity:
//     ifft(x) = conj(fft(conj(x))) / len
inline void rpp_fft_radix2_inverse(RppFftComplex* data, Rpp32s len,
                                   const std::vector<Rpp32u>& bitRev,
                                   const std::vector<RppFftComplex>& tw) {
    for (Rpp32s i = 0; i < len; i++) data[i] = std::conj(data[i]);
    rpp_fft_radix2_forward(data, len, bitRev, tw);
    Rpp32f inv = 1.0f / (Rpp32f)len;
    for (Rpp32s i = 0; i < len; i++) data[i] = std::conj(data[i]) * inv;
}

// Largest supported transform size. Guards against integer overflow in the Bluestein
// setup (2*nfft-1 and the next-power-of-two search) and against pathological allocation.
constexpr Rpp32s RPP_CPU_FFT_MAX_NFFT = 1 << 24;  // 16,777,216

// Precompute all read-only tables for a forward transform of size nfft.
// Returns false (leaving plan unmodified) if nfft is outside [1, RPP_CPU_FFT_MAX_NFFT].
inline bool rpp_cpu_fft_plan_init(RppCpuFftPlan& plan, Rpp32s nfft) {
    if (nfft <= 0 || nfft > RPP_CPU_FFT_MAX_NFFT) return false;
    plan.n = nfft;
    plan.isPow2 = rpp_fft_is_pow2(nfft);
    if (plan.isPow2) {
        rpp_fft_build_bitrev(plan.bitRev, nfft);
        rpp_fft_build_twiddles(plan.twiddles, nfft);
        return true;
    }

    // Bluestein setup
    plan.m = rpp_fft_next_pow2(2 * nfft - 1);
    rpp_fft_build_bitrev(plan.bitRevM, plan.m);
    rpp_fft_build_twiddles(plan.twiddlesM, plan.m);

    plan.chirp.resize(nfft);
    for (Rpp32s k = 0; k < nfft; k++) {
        // exp(-pi*i*k^2/n); reduce k^2 mod 2n to keep the argument small and precise
        Rpp64s k2mod = ((Rpp64s)k * (Rpp64s)k) % (2LL * (Rpp64s)nfft);
        Rpp64f ang = -M_PI * (Rpp64f)k2mod / (Rpp64f)nfft;
        plan.chirp[k] = RppFftComplex((Rpp32f)std::cos(ang), (Rpp32f)std::sin(ang));
    }

    // Convolution kernel b[j] = conj(chirp[j]) = exp(+pi*i*j^2/n), even in j, wrapped into m.
    plan.kernelFft.assign(plan.m, RppFftComplex(0.0f, 0.0f));
    plan.kernelFft[0] = std::conj(plan.chirp[0]);
    for (Rpp32s k = 1; k < nfft; k++) {
        RppFftComplex v = std::conj(plan.chirp[k]);
        plan.kernelFft[k] = v;
        plan.kernelFft[plan.m - k] = v;
    }
    rpp_fft_radix2_forward(plan.kernelFft.data(), plan.m, plan.bitRevM, plan.twiddlesM);
    return true;
}

// Forward FFT of a real-valued signal (imaginary part assumed zero). Writes the first
// numBins = n/2 + 1 complex bins to out. scratch must hold at least:
//     n  elements when plan.isPow2 == true
//     m  elements otherwise
// scratch and out are caller-owned (use one set per thread for concurrent execution).
inline void rpp_cpu_fft_forward_real(const RppCpuFftPlan& plan, const Rpp32f* in, Rpp32s inLen,
                                     RppFftComplex* out, RppFftComplex* scratch) {
    const Rpp32s n = plan.n;
    const Rpp32s numBins = n / 2 + 1;

    if (plan.isPow2) {
        for (Rpp32s i = 0; i < n; i++) scratch[i] = RppFftComplex((i < inLen) ? in[i] : 0.0f, 0.0f);
        rpp_fft_radix2_forward(scratch, n, plan.bitRev, plan.twiddles);
        for (Rpp32s i = 0; i < numBins; i++) out[i] = scratch[i];
        return;
    }

    // Bluestein: X[k] = chirp[k] * (a (*) b)[k], with a[n] = x[n]*chirp[n], b = conj(chirp)
    const Rpp32s m = plan.m;
    for (Rpp32s i = 0; i < m; i++) scratch[i] = RppFftComplex(0.0f, 0.0f);
    for (Rpp32s i = 0; i < n; i++) {
        Rpp32f x = (i < inLen) ? in[i] : 0.0f;
        scratch[i] = RppFftComplex(x, 0.0f) * plan.chirp[i];
    }
    rpp_fft_radix2_forward(scratch, m, plan.bitRevM, plan.twiddlesM);
    for (Rpp32s i = 0; i < m; i++) scratch[i] *= plan.kernelFft[i];
    rpp_fft_radix2_inverse(scratch, m, plan.bitRevM, plan.twiddlesM);
    for (Rpp32s k = 0; k < numBins; k++) out[k] = scratch[k] * plan.chirp[k];
}

#endif  // RPP_CPU_FFT_HPP
