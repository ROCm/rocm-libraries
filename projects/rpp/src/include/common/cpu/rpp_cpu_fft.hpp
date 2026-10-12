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

#ifndef RPP_CPU_FFT_HPP
#define RPP_CPU_FFT_HPP

// In-house forward FFT (unnormalized) for the CPU spectrogram kernel.
// Computes the forward DFT:
//     X[k] = sum_{n=0}^{N-1} x[n] * exp(-2*pi*i*k*n/N)
//
// Arbitrary transform sizes are supported:
//   - power-of-two N: iterative radix-2 Cooley-Tukey
//   - all other N:    Bluestein's chirp-z algorithm (reduces to a power-of-two convolution)
//
// A plan holds only read-only precomputed tables, so a single plan may be shared across OpenMP
// threads; all mutable work is done through caller-supplied scratch buffers (one per thread).
//
// Precision: the input samples are float32, but twiddles, butterflies and the output bins are
// all double. A float32 radix-2 FFT accumulates roughly sqrt(log2(N)) ULPs of error over its
// stages (~2e-6 relative at N = 512), which is both larger than a float32 ULP and dependent on
// how the compiler vectorizes and contracts the butterflies - so the last bits of the result
// vary with toolchain, -march and ISA. Accumulating in double keeps the error ~1e-16 relative,
// far below a float32 ULP, so narrowing to float32 at the end yields the correctly rounded
// float32 value on any toolchain. That makes the kernel both more accurate than a float32 FFT
// and bitwise reproducible, which is what the audio QA suite's exact-match checks require.

#include <cmath>
#include <complex>
#include <utility>
#include <vector>

#include "rppdefs.h"

constexpr Rpp64f RPP_PI = 3.14159265358979323846;
constexpr Rpp64f RPP_2PI = 2.0 * RPP_PI;

// Working type for twiddles, scratch and output bins. Double precision - see the note above.
typedef std::complex<Rpp64f> RppFftComplex;

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
        Rpp64f ang = -RPP_2PI * (Rpp64f)k / (Rpp64f)len;
        tw[k] = RppFftComplex(std::cos(ang), std::sin(ang));
    }
}

// Textbook complex multiply on the real/imaginary parts.
//
// operator* on std::complex is deliberately avoided in the inner loops: the standard requires the
// C99 Annex G NaN/infinity fixup, so GCC compiles every complex multiply into an out-of-line
// libgcc call (__muldc3) rather than inline arithmetic. That call, not the arithmetic, dominates
// the butterfly. FFT inputs here are always finite, so the fixup is dead weight.
inline RppFftComplex rpp_fft_cmul(const RppFftComplex& a, const RppFftComplex& b) {
    const Rpp64f ar = a.real(), ai = a.imag(), br = b.real(), bi = b.imag();
    return RppFftComplex(ar * br - ai * bi, ar * bi + ai * br);
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
                const Rpp64f wr = tw[k].real(), wi = tw[k].imag();
                const Rpp64f xr = data[j + half].real(), xi = data[j + half].imag();
                const Rpp64f tr = wr * xr - wi * xi;
                const Rpp64f ti = wr * xi + wi * xr;
                const Rpp64f ur = data[j].real(), ui = data[j].imag();
                data[j + half] = RppFftComplex(ur - tr, ui - ti);
                data[j] = RppFftComplex(ur + tr, ui + ti);
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
    Rpp64f inv = 1.0 / (Rpp64f)len;
    for (Rpp32s i = 0; i < len; i++)
        data[i] = RppFftComplex(data[i].real() * inv, -data[i].imag() * inv);
}

// Precompute all read-only tables for a forward transform of size nfft.
inline void rpp_cpu_fft_plan_init(RppCpuFftPlan& plan, Rpp32s nfft) {
    plan.n = nfft;
    plan.isPow2 = rpp_fft_is_pow2(nfft);
    if (plan.isPow2) {
        rpp_fft_build_bitrev(plan.bitRev, nfft);
        rpp_fft_build_twiddles(plan.twiddles, nfft);
        return;
    }

    // Bluestein setup
    plan.m = rpp_fft_next_pow2(2 * nfft - 1);
    rpp_fft_build_bitrev(plan.bitRevM, plan.m);
    rpp_fft_build_twiddles(plan.twiddlesM, plan.m);

    plan.chirp.resize(nfft);
    for (Rpp32s k = 0; k < nfft; k++) {
        // exp(-pi*i*k^2/n); reduce k^2 mod 2n to keep the argument small and precise
        Rpp64s k2mod = ((Rpp64s)k * (Rpp64s)k) % (2LL * (Rpp64s)nfft);
        Rpp64f ang = -RPP_PI * (Rpp64f)k2mod / (Rpp64f)nfft;
        plan.chirp[k] = RppFftComplex(std::cos(ang), std::sin(ang));
    }

    // Convolution kernel b[j] = conj(chirp[j]) = exp(+pi*i*j^2/n), even in j, wrapped into m.
    plan.kernelFft.assign(plan.m, RppFftComplex(0.0, 0.0));
    plan.kernelFft[0] = std::conj(plan.chirp[0]);
    for (Rpp32s k = 1; k < nfft; k++) {
        RppFftComplex v = std::conj(plan.chirp[k]);
        plan.kernelFft[k] = v;
        plan.kernelFft[plan.m - k] = v;
    }
    rpp_fft_radix2_forward(plan.kernelFft.data(), plan.m, plan.bitRevM, plan.twiddlesM);
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
        for (Rpp32s i = 0; i < n; i++)
            scratch[i] = RppFftComplex((i < inLen) ? (Rpp64f)in[i] : 0.0, 0.0);
        rpp_fft_radix2_forward(scratch, n, plan.bitRev, plan.twiddles);
        for (Rpp32s i = 0; i < numBins; i++) out[i] = scratch[i];
        return;
    }

    // Bluestein: X[k] = chirp[k] * (a (*) b)[k], with a[n] = x[n]*chirp[n], b = conj(chirp)
    const Rpp32s m = plan.m;
    for (Rpp32s i = 0; i < m; i++) scratch[i] = RppFftComplex(0.0, 0.0);
    for (Rpp32s i = 0; i < n; i++) {
        Rpp64f x = (i < inLen) ? (Rpp64f)in[i] : 0.0;
        scratch[i] = rpp_fft_cmul(RppFftComplex(x, 0.0), plan.chirp[i]);
    }
    rpp_fft_radix2_forward(scratch, m, plan.bitRevM, plan.twiddlesM);
    for (Rpp32s i = 0; i < m; i++) scratch[i] = rpp_fft_cmul(scratch[i], plan.kernelFft[i]);
    rpp_fft_radix2_inverse(scratch, m, plan.bitRevM, plan.twiddlesM);
    for (Rpp32s k = 0; k < numBins; k++) out[k] = rpp_fft_cmul(scratch[k], plan.chirp[k]);
}

#endif  // RPP_CPU_FFT_HPP
