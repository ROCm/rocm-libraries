// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef HIPFFTW_H_
#define HIPFFTW_H_

#include "hipfft-export.h"
#include <cstddef>
#include <cstdlib>

#ifdef __cplusplus
extern "C" {
#endif

// FFTW can use standard complex type
#if defined(_Complex_I) && defined(complex) && defined(I)
typedef double complex fftw_complex;
typedef float complex  fftwf_complex;
#else
typedef double fftw_complex[2];
typedef float  fftwf_complex[2];
#endif

// Planning flags and constants
#define FFTW_MEASURE (0U)
#define FFTW_DESTROY_INPUT (1U << 0)
#define FFTW_UNALIGNED (1U << 1)
#define FFTW_CONSERVE_MEMORY (1U << 2)
#define FFTW_EXHAUSTIVE (1U << 3)
#define FFTW_PRESERVE_INPUT (1U << 4)
#define FFTW_PATIENT (1U << 5)
#define FFTW_ESTIMATE (1U << 6)
#define FFTW_WISDOM_ONLY (1U << 21)
#define FFTW_FORWARD (-1)
#define FFTW_BACKWARD (1)

// Buffer management
HIPFFT_EXPORT void*   fftw_malloc(size_t n);
HIPFFT_EXPORT void*   fftwf_malloc(size_t n);
HIPFFT_EXPORT double* fftw_alloc_real(size_t n);
HIPFFT_EXPORT float*  fftwf_alloc_real(size_t n);
HIPFFT_EXPORT fftw_complex* fftw_alloc_complex(size_t n);
HIPFFT_EXPORT fftwf_complex* fftwf_alloc_complex(size_t n);

HIPFFT_EXPORT void fftw_free(void* p);
HIPFFT_EXPORT void fftwf_free(void* p);

// Plan usage
typedef struct fftw_plan_s*  fftw_plan;
typedef struct fftwf_plan_s* fftwf_plan;

HIPFFT_EXPORT void fftw_destroy_plan(fftw_plan plan);
HIPFFT_EXPORT void fftwf_destroy_plan(fftwf_plan plan);
HIPFFT_EXPORT void fftw_cleanup();
HIPFFT_EXPORT void fftwf_cleanup();
HIPFFT_EXPORT void fftw_execute(const fftw_plan plan);
HIPFFT_EXPORT void fftwf_execute(const fftwf_plan plan);
HIPFFT_EXPORT void fftw_execute_dft(const fftw_plan plan, fftw_complex* in, fftw_complex* out);
HIPFFT_EXPORT void fftwf_execute_dft(const fftwf_plan plan, fftwf_complex* in, fftwf_complex* out);
HIPFFT_EXPORT void fftw_execute_dft_r2c(const fftw_plan plan, double* in, fftw_complex* out);
HIPFFT_EXPORT void fftwf_execute_dft_r2c(const fftwf_plan plan, float* in, fftwf_complex* out);
HIPFFT_EXPORT void fftw_execute_dft_c2r(const fftw_plan plan, fftw_complex* in, double* out);
HIPFFT_EXPORT void fftwf_execute_dft_c2r(const fftwf_plan plan, fftwf_complex* in, float* out);

// Basic plans
HIPFFT_EXPORT fftw_plan
    fftw_plan_dft_1d(int n, fftw_complex* in, fftw_complex* out, int sign, unsigned flags);
HIPFFT_EXPORT fftwf_plan
    fftwf_plan_dft_1d(int n, fftwf_complex* in, fftwf_complex* out, int sign, unsigned flags);
HIPFFT_EXPORT fftw_plan
    fftw_plan_dft_2d(int n0, int n1, fftw_complex* in, fftw_complex* out, int sign, unsigned flags);
HIPFFT_EXPORT fftwf_plan fftwf_plan_dft_2d(
    int n0, int n1, fftwf_complex* in, fftwf_complex* out, int sign, unsigned flags);
HIPFFT_EXPORT fftw_plan fftw_plan_dft_3d(
    int n0, int n1, int n2, fftw_complex* in, fftw_complex* out, int sign, unsigned flags);
HIPFFT_EXPORT fftwf_plan fftwf_plan_dft_3d(
    int n0, int n1, int n2, fftwf_complex* in, fftwf_complex* out, int sign, unsigned flags);
HIPFFT_EXPORT fftw_plan fftw_plan_dft(
    int rank, const int* n, fftw_complex* in, fftw_complex* out, int sign, unsigned flags);
HIPFFT_EXPORT fftwf_plan fftwf_plan_dft(
    int rank, const int* n, fftwf_complex* in, fftwf_complex* out, int sign, unsigned flags);
HIPFFT_EXPORT fftw_plan  fftw_plan_dft_r2c_1d(int n, double* in, fftw_complex* out, unsigned flags);
HIPFFT_EXPORT fftwf_plan fftwf_plan_dft_r2c_1d(int            n,
                                               float*         in,
                                               fftwf_complex* out,
                                               unsigned       flags);
HIPFFT_EXPORT fftw_plan
    fftw_plan_dft_r2c_2d(int n0, int n1, double* in, fftw_complex* out, unsigned flags);
HIPFFT_EXPORT fftwf_plan
    fftwf_plan_dft_r2c_2d(int n0, int n1, float* in, fftwf_complex* out, unsigned flags);
HIPFFT_EXPORT fftw_plan
    fftw_plan_dft_r2c_3d(int n0, int n1, int n2, double* in, fftw_complex* out, unsigned flags);
HIPFFT_EXPORT fftwf_plan
    fftwf_plan_dft_r2c_3d(int n0, int n1, int n2, float* in, fftwf_complex* out, unsigned flags);
HIPFFT_EXPORT fftw_plan
    fftw_plan_dft_r2c(int rank, const int* n, double* in, fftw_complex* out, unsigned flags);
HIPFFT_EXPORT fftwf_plan
    fftwf_plan_dft_r2c(int rank, const int* n, float* in, fftwf_complex* out, unsigned flags);
HIPFFT_EXPORT fftw_plan  fftw_plan_dft_c2r_1d(int n, fftw_complex* in, double* out, unsigned flags);
HIPFFT_EXPORT fftwf_plan fftwf_plan_dft_c2r_1d(int            n,
                                               fftwf_complex* in,
                                               float*         out,
                                               unsigned       flags);
HIPFFT_EXPORT fftw_plan
    fftw_plan_dft_c2r_2d(int n0, int n1, fftw_complex* in, double* out, unsigned flags);
HIPFFT_EXPORT fftwf_plan
    fftwf_plan_dft_c2r_2d(int n0, int n1, fftwf_complex* in, float* out, unsigned flags);
HIPFFT_EXPORT fftw_plan
    fftw_plan_dft_c2r_3d(int n0, int n1, int n2, fftw_complex* in, double* out, unsigned flags);
HIPFFT_EXPORT fftwf_plan
    fftwf_plan_dft_c2r_3d(int n0, int n1, int n2, fftwf_complex* in, float* out, unsigned flags);
HIPFFT_EXPORT fftw_plan
    fftw_plan_dft_c2r(int rank, const int* n, fftw_complex* in, double* out, unsigned flags);
HIPFFT_EXPORT fftwf_plan
    fftwf_plan_dft_c2r(int rank, const int* n, fftwf_complex* in, float* out, unsigned flags);

// Non-functional utility APIs
HIPFFT_EXPORT void   fftw_print_plan(const fftw_plan);
HIPFFT_EXPORT void   fftwf_print_plan(const fftwf_plan);
HIPFFT_EXPORT void   fftw_set_timelimit(double);
HIPFFT_EXPORT void   fftwf_set_timelimit(double);
HIPFFT_EXPORT double fftw_cost(const fftw_plan);
HIPFFT_EXPORT double fftwf_cost(const fftw_plan);
HIPFFT_EXPORT void   fftw_flops(const fftw_plan, double*, double*, double*);
HIPFFT_EXPORT void   fftwf_flops(const fftw_plan, double*, double*, double*);

#ifdef __cplusplus
}
#endif

#endif
