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

// Brings in the portable SIMD intrinsics: <immintrin.h> on x86, SIMDE with native aliases
// elsewhere. Including <immintrin.h> directly here would break the non-x86 HOST build.
#include "rpp_cpu_simd_load_store.hpp"
#include "rppdefs.h"

constexpr Rpp64f RPP_PI = 3.14159265358979323846;
constexpr Rpp64f RPP_2PI = 2.0 * RPP_PI;

// Working type for twiddles, scratch and output bins. Double precision - see the note above.
typedef std::complex<Rpp64f> RppFftComplex;

struct RppCpuFftPlan {
    Rpp32s n = 0;  // transform size (nfft)
    bool isPow2 = false;

    // radix-2 tables (used when isPow2 == true). A real signal of length n is transformed as
    // a complex FFT of length n/2, so the permutation and inner twiddles are sized for n/2;
    // `twiddles` is the n-point table used to recombine the halves.
    std::vector<Rpp32u> bitRev;               // bit-reversal for length n/2
    std::vector<RppFftComplex> twiddlesHalf;  // exp(-2*pi*i*k/(n/2)), k in [0, n/4)
    std::vector<RppFftComplex> twiddles;      // exp(-2*pi*i*k/n), k in [0, n/2)

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

// Complex multiply on raw parts: std::complex's operator* compiles to an out-of-line libgcc call.
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
        rpp_fft_build_bitrev(plan.bitRev, nfft / 2);
        rpp_fft_build_twiddles(plan.twiddlesHalf, nfft / 2);
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
//     n/2 elements when plan.isPow2 == true
//     m   elements otherwise
// scratch and out are caller-owned, must not alias, and one set is needed per thread.
inline void rpp_cpu_fft_forward_real(const RppCpuFftPlan& plan, const Rpp32f* in, Rpp32s inLen,
                                     RppFftComplex* out, RppFftComplex* scratch) {
    const Rpp32s n = plan.n;
    const Rpp32s numBins = n / 2 + 1;

    if (plan.isPow2) {
        if (n == 1) {
            out[0] = RppFftComplex((inLen > 0) ? (Rpp64f)in[0] : 0.0, 0.0);
            return;
        }

        // Real-input transform. Pack the n real samples into n/2 complex ones,
        // z[k] = x[2k] + i*x[2k+1], transform those, then recombine. This halves the
        // transform length versus running a full complex FFT over a zero imaginary part.
        const Rpp32s h = n / 2;
        for (Rpp32s k = 0; k < h; k++) {
            const Rpp32s e = 2 * k, o = e + 1;
            const Rpp64f re = (e < inLen) ? (Rpp64f)in[e] : 0.0;
            const Rpp64f im = (o < inLen) ? (Rpp64f)in[o] : 0.0;
            scratch[k] = RppFftComplex(re, im);
        }
        rpp_fft_radix2_forward(scratch, h, plan.bitRev, plan.twiddlesHalf);

        // Z[0] holds both purely real end bins: X[0] = Re+Im and X[n/2] = Re-Im.
        const Rpp64f z0r = scratch[0].real(), z0i = scratch[0].imag();
        out[0] = RppFftComplex(z0r + z0i, 0.0);
        out[h] = RppFftComplex(z0r - z0i, 0.0);

        // For the rest, split Z into the spectra of the even and odd samples and recombine:
        //     B  = conj(Z[h-k])
        //     Ze = (Z[k] + B) / 2            (even samples)
        //     Zo = -i * (Z[k] - B) / 2       (odd samples)
        //     X[k] = Ze + exp(-2*pi*i*k/n) * Zo
        for (Rpp32s k = 1; k < h; k++) {
            const Rpp64f ar = scratch[k].real(), ai = scratch[k].imag();
            const Rpp64f br = scratch[h - k].real(), bi = -scratch[h - k].imag();
            const Rpp64f zer = 0.5 * (ar + br), zei = 0.5 * (ai + bi);
            const Rpp64f dr = 0.5 * (ar - br), di = 0.5 * (ai - bi);
            const Rpp64f zor = di, zoi = -dr;
            const Rpp64f wr = plan.twiddles[k].real(), wi = plan.twiddles[k].imag();
            out[k] = RppFftComplex(zer + wr * zor - wi * zoi, zei + wr * zoi + wi * zor);
        }
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

// Complex multiply matching the scalar form: cr = ar*br - ai*bi, ci = ar*bi + ai*br.
inline void rpp_simd_fft_cmul(__m256d ar, __m256d ai, __m256d br, __m256d bi, __m256d& cr,
                              __m256d& ci) {
    cr = _mm256_sub_pd(_mm256_mul_pd(ar, br), _mm256_mul_pd(ai, bi));
    ci = _mm256_add_pd(_mm256_mul_pd(ar, bi), _mm256_mul_pd(ai, br));
}

// Radix-4 form of the four-lane butterfly. Each pass fuses what were two consecutive radix-2
// stages, so the intermediate results stay in registers instead of being written out and read
// back. The arithmetic and its order are unchanged, only the memory traffic halves. Fusing
// stage pairs also means the existing base-2 bit-reversal still applies - a textbook radix-4
// would need base-4 digit reversal. When log2(len) is odd one radix-2 stage runs first, and its
// twiddles are all tw[0] == 1 so it carries no multiply.
inline void rpp_simd_fft_radix4_forward(Rpp64f* re, Rpp64f* im, Rpp32s len,
                                        const std::vector<Rpp32u>& bitRev,
                                        const std::vector<RppFftComplex>& tw) {
    for (Rpp32s i = 0; i < len; i++) {
        const Rpp32u j = bitRev[i];
        if ((Rpp32u)i < j) {
            const __m256d ar = _mm256_loadu_pd(re + i * 4), br = _mm256_loadu_pd(re + j * 4);
            const __m256d ai = _mm256_loadu_pd(im + i * 4), bi = _mm256_loadu_pd(im + j * 4);
            _mm256_storeu_pd(re + i * 4, br);
            _mm256_storeu_pd(re + j * 4, ar);
            _mm256_storeu_pd(im + i * 4, bi);
            _mm256_storeu_pd(im + j * 4, ai);
        }
    }

    Rpp32s stages = 0;
    while ((1 << stages) < len) stages++;

    Rpp32s H = 1;
    if (stages & 1) {
        for (Rpp32s i = 0; i < len; i += 2) {
            Rpp64f* aRe = re + i * 4;
            Rpp64f* aIm = im + i * 4;
            Rpp64f* bRe = re + (i + 1) * 4;
            Rpp64f* bIm = im + (i + 1) * 4;
            const __m256d x0r = _mm256_loadu_pd(aRe), x0i = _mm256_loadu_pd(aIm);
            const __m256d x1r = _mm256_loadu_pd(bRe), x1i = _mm256_loadu_pd(bIm);
            _mm256_storeu_pd(aRe, _mm256_add_pd(x0r, x1r));
            _mm256_storeu_pd(aIm, _mm256_add_pd(x0i, x1i));
            _mm256_storeu_pd(bRe, _mm256_sub_pd(x0r, x1r));
            _mm256_storeu_pd(bIm, _mm256_sub_pd(x0i, x1i));
        }
        H = 2;
    }

    for (; 4 * H <= len; H <<= 2) {
        const Rpp32s stepA = len / (2 * H);
        const Rpp32s stepB = len / (4 * H);
        for (Rpp32s i = 0; i < len; i += 4 * H) {
            for (Rpp32s t = 0; t < H; t++) {
                const __m256d wAr = _mm256_set1_pd(tw[t * stepA].real());
                const __m256d wAi = _mm256_set1_pd(tw[t * stepA].imag());
                const __m256d wB0r = _mm256_set1_pd(tw[t * stepB].real());
                const __m256d wB0i = _mm256_set1_pd(tw[t * stepB].imag());
                const __m256d wB1r = _mm256_set1_pd(tw[(t + H) * stepB].real());
                const __m256d wB1i = _mm256_set1_pd(tw[(t + H) * stepB].imag());

                Rpp64f* p0r = re + (i + t) * 4;
                Rpp64f* p0i = im + (i + t) * 4;
                Rpp64f* p1r = re + (i + t + H) * 4;
                Rpp64f* p1i = im + (i + t + H) * 4;
                Rpp64f* p2r = re + (i + t + 2 * H) * 4;
                Rpp64f* p2i = im + (i + t + 2 * H) * 4;
                Rpp64f* p3r = re + (i + t + 3 * H) * 4;
                Rpp64f* p3i = im + (i + t + 3 * H) * 4;

                const __m256d x0r = _mm256_loadu_pd(p0r), x0i = _mm256_loadu_pd(p0i);
                const __m256d x1r = _mm256_loadu_pd(p1r), x1i = _mm256_loadu_pd(p1i);
                const __m256d x2r = _mm256_loadu_pd(p2r), x2i = _mm256_loadu_pd(p2i);
                const __m256d x3r = _mm256_loadu_pd(p3r), x3i = _mm256_loadu_pd(p3i);

                __m256d t1r, t1i, t3r, t3i;
                rpp_simd_fft_cmul(wAr, wAi, x1r, x1i, t1r, t1i);
                rpp_simd_fft_cmul(wAr, wAi, x3r, x3i, t3r, t3i);
                const __m256d u0r = _mm256_add_pd(x0r, t1r), u0i = _mm256_add_pd(x0i, t1i);
                const __m256d u1r = _mm256_sub_pd(x0r, t1r), u1i = _mm256_sub_pd(x0i, t1i);
                const __m256d u2r = _mm256_add_pd(x2r, t3r), u2i = _mm256_add_pd(x2i, t3i);
                const __m256d u3r = _mm256_sub_pd(x2r, t3r), u3i = _mm256_sub_pd(x2i, t3i);

                __m256d v2r, v2i, v3r, v3i;
                rpp_simd_fft_cmul(wB0r, wB0i, u2r, u2i, v2r, v2i);
                rpp_simd_fft_cmul(wB1r, wB1i, u3r, u3i, v3r, v3i);

                _mm256_storeu_pd(p0r, _mm256_add_pd(u0r, v2r));
                _mm256_storeu_pd(p0i, _mm256_add_pd(u0i, v2i));
                _mm256_storeu_pd(p2r, _mm256_sub_pd(u0r, v2r));
                _mm256_storeu_pd(p2i, _mm256_sub_pd(u0i, v2i));
                _mm256_storeu_pd(p1r, _mm256_add_pd(u1r, v3r));
                _mm256_storeu_pd(p1i, _mm256_add_pd(u1i, v3i));
                _mm256_storeu_pd(p3r, _mm256_sub_pd(u1r, v3r));
                _mm256_storeu_pd(p3i, _mm256_sub_pd(u1i, v3i));
            }
        }
    }
}

// One output bin of the real-input recombination: X = Ze + W * Zo, where Ze = (a + b) / 2 and
// Zo = -i * (a - b) / 2. The caller passes b already conjugated.
inline void rpp_simd_fft_recombine(const RppFftComplex& w, __m256d ar, __m256d ai, __m256d br,
                                   __m256d bi, Rpp64f* outRe, Rpp64f* outIm) {
    const __m256d zero = _mm256_setzero_pd();
    const __m256d pointFive = _mm256_set1_pd(0.5);
    const __m256d wr = _mm256_set1_pd(w.real());
    const __m256d wi = _mm256_set1_pd(w.imag());
    const __m256d zer = _mm256_mul_pd(pointFive, _mm256_add_pd(ar, br));
    const __m256d zei = _mm256_mul_pd(pointFive, _mm256_add_pd(ai, bi));
    const __m256d dr = _mm256_mul_pd(pointFive, _mm256_sub_pd(ar, br));
    const __m256d di = _mm256_mul_pd(pointFive, _mm256_sub_pd(ai, bi));
    const __m256d zor = di;  // Zo = -i * d
    const __m256d zoi = _mm256_sub_pd(zero, dr);
    _mm256_storeu_pd(
        outRe, _mm256_sub_pd(_mm256_add_pd(zer, _mm256_mul_pd(wr, zor)), _mm256_mul_pd(wi, zoi)));
    _mm256_storeu_pd(
        outIm, _mm256_add_pd(_mm256_add_pd(zei, _mm256_mul_pd(wr, zoi)), _mm256_mul_pd(wi, zor)));
}

// SIMD counterpart of rpp_cpu_fft_forward_real: transforms four real signals of length n at
// once, in[l] being signal l. Bin k of transform l is written to outRe/outIm[k * 4 + l], which
// keeps the lanes contiguous for whatever the caller does with them next. Power-of-two n only;
// other sizes, and any group of fewer than four signals, use the scalar entry point.
// scratchRe and scratchIm each need (n/2) * 4 elements and must not alias outRe/outIm.
inline void rpp_simd_fft_forward_real(const RppCpuFftPlan& plan, const Rpp32f* const* in,
                                      Rpp32s inLen, Rpp64f* outRe, Rpp64f* outIm, Rpp64f* scratchRe,
                                      Rpp64f* scratchIm) {
    const Rpp32s n = plan.n;
    if (n == 1) {
        for (Rpp32s l = 0; l < 4; l++) {
            outRe[l] = (inLen > 0) ? (Rpp64f)in[l][0] : 0.0;
            outIm[l] = 0.0;
        }
        return;
    }

    const Rpp32s h = n / 2;
    // Packing is a 4 x h transpose with an even/odd split: lane l reads consecutive pairs from
    // in[l] and they land interleaved across lanes. Four k at a time, so each lane contributes
    // one 256-bit vector and a 4x4 transpose puts them in place. Widening float to double is
    // exact, and the shuffles only move data, so this packs the same bits as the scalar form.
    Rpp32s k = 0;
    if (inLen >= n) {
        for (; k + 4 <= h; k += 4) {
            __m256d evenD[4], oddD[4];
            for (Rpp32s l = 0; l < 4; l++) {
                const __m128 v0 = _mm_loadu_ps(in[l] + 2 * k);
                const __m128 v1 = _mm_loadu_ps(in[l] + 2 * k + 4);
                evenD[l] = _mm256_cvtps_pd(_mm_shuffle_ps(v0, v1, _MM_SHUFFLE(2, 0, 2, 0)));
                oddD[l] = _mm256_cvtps_pd(_mm_shuffle_ps(v0, v1, _MM_SHUFFLE(3, 1, 3, 1)));
            }
            for (Rpp32s pass = 0; pass < 2; pass++) {
                const __m256d* src = pass ? oddD : evenD;
                Rpp64f* dst = (pass ? scratchIm : scratchRe) + k * 4;
                const __m256d t0 = _mm256_unpacklo_pd(src[0], src[1]);
                const __m256d t1 = _mm256_unpackhi_pd(src[0], src[1]);
                const __m256d t2 = _mm256_unpacklo_pd(src[2], src[3]);
                const __m256d t3 = _mm256_unpackhi_pd(src[2], src[3]);
                _mm256_storeu_pd(dst, _mm256_permute2f128_pd(t0, t2, 0x20));
                _mm256_storeu_pd(dst + 4, _mm256_permute2f128_pd(t1, t3, 0x20));
                _mm256_storeu_pd(dst + 8, _mm256_permute2f128_pd(t0, t2, 0x31));
                _mm256_storeu_pd(dst + 12, _mm256_permute2f128_pd(t1, t3, 0x31));
            }
        }
    }
    for (; k < h; k++) {
        const Rpp32s e = 2 * k, o = e + 1;
        for (Rpp32s l = 0; l < 4; l++) {
            scratchRe[k * 4 + l] = (e < inLen) ? (Rpp64f)in[l][e] : 0.0;
            scratchIm[k * 4 + l] = (o < inLen) ? (Rpp64f)in[l][o] : 0.0;
        }
    }
    rpp_simd_fft_radix4_forward(scratchRe, scratchIm, h, plan.bitRev, plan.twiddlesHalf);

    // Z[0] carries both purely real end bins: X[0] = Re+Im and X[n/2] = Re-Im.
    const __m256d zero = _mm256_setzero_pd();
    const __m256d z0r = _mm256_loadu_pd(scratchRe), z0i = _mm256_loadu_pd(scratchIm);
    _mm256_storeu_pd(outRe, _mm256_add_pd(z0r, z0i));
    _mm256_storeu_pd(outIm, zero);
    _mm256_storeu_pd(outRe + h * 4, _mm256_sub_pd(z0r, z0i));
    _mm256_storeu_pd(outIm + h * 4, zero);

    // Bins k and h - k are built from the same two scratch entries, so pairing them halves the
    // loads. Each bin keeps its own twiddle and its own arithmetic, so the results are unchanged.
    // Sequential walk over bins, loading both scratch entries each time. Pairing k with h - k
    // halves the loads but splits the stores into two opposing streams, which measured slower.
    for (k = 1; k < h; k++) {
        const __m256d ar = _mm256_loadu_pd(scratchRe + k * 4);
        const __m256d ai = _mm256_loadu_pd(scratchIm + k * 4);
        const __m256d br = _mm256_loadu_pd(scratchRe + (h - k) * 4);
        const __m256d bi = _mm256_sub_pd(zero, _mm256_loadu_pd(scratchIm + (h - k) * 4));
        rpp_simd_fft_recombine(plan.twiddles[k], ar, ai, br, bi, outRe + k * 4, outIm + k * 4);
    }
}

#endif  // RPP_CPU_FFT_HPP
