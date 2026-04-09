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
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Data types and constant values

/**
 * @brief Type for double-precision complex floating-point values.
 * \note `fftw_complex` is defined equivalent to `double complex` in applications
 * that include the standard `complex.h` prior to `hipfft/hipfftw.h`.
 */
#if defined(_Complex_I) && defined(complex) && defined(I)
typedef double complex fftw_complex;
#else
typedef double fftw_complex[2];
#endif

/**
 * @brief Single-precision equivalent of \ref fftw_complex.
 */
#if defined(_Complex_I) && defined(complex) && defined(I)
typedef float complex fftwf_complex;
#else
typedef float  fftwf_complex[2];
#endif

/**
 * @brief Type for double-precision hipFFTW plans.
 */
typedef struct fftw_plan_s* fftw_plan;

/**
 * @brief Single-precision equivalent of \ref fftw_plan;
 */
typedef struct fftwf_plan_s* fftwf_plan;

typedef struct
{
    int n;
    int is;
    int os;
} hipfftw_iodim;
typedef struct
{
    ptrdiff_t n;
    ptrdiff_t is;
    ptrdiff_t os;
} hipfftw_iodim64;

/**
 * @brief Structure type describing lengths (or batch sizes) and input/output
 * strides (or distances) along a dimension for generalized data layouts.
 * Structure members are:
 * - ``n`` length (or batch size);
 * - ``is`` input stride (or distance);
 * - ``os`` output stride (or distance).
 *
 * All the above members are of type ``int``.
 */
typedef hipfftw_iodim fftw_iodim;

/**
 * @brief This type is strictly equivalent to ``fftw_iodim``.
 */
typedef hipfftw_iodim fftwf_iodim;

/**
 * @brief This structure type is the equivalent of ``fftw_iodim`` using
 * ``ptrdiff_t`` for members' type instead of ``int``.
 */
typedef hipfftw_iodim64 fftw_iodim64;

/**
 * @brief This type is strictly equivalent to ``fftw_iodim64``.
 */
typedef hipfftw_iodim64 fftwf_iodim64;

/**
 * @brief Flag value allowing hipFFTW to compute (possibly many) FFTs at plan creation
 * to find the optimal configuration, using the input and output data buffers set at plan
 * creation (hence possibly overwriting data therein).
 */
#define FFTW_MEASURE (0U)

/**
 * @brief Flag value allowing an out-of-place hipFFTW plan to overwrite its input buffer
 * data at execution.
 */
#define FFTW_DESTROY_INPUT (1U << 0)

/**
 * @brief Flag value ruling out any alignment assumption for the input and output buffers
 * to be used at plan execution.
 */
#define FFTW_UNALIGNED (1U << 1)

/**
 * @brief Flag value instructing plans to prefer configurations minimizing their memory footprint.
 */
#define FFTW_CONSERVE_MEMORY (1U << 2)

/**
 * @brief Flag value equivalent to \ref FFTW_PATIENT, enabling the largest possible set of plan
 * configurations to be considered in the measurements.
 */
#define FFTW_EXHAUSTIVE (1U << 3)

/**
 * @brief Flag value forbidding an out-of-place hipFFTW plan to overwrite its input buffer
 * data at execution.
 * \note This flag value is not supported for out-of-place multidimensional real inverse tranforms.
 */
#define FFTW_PRESERVE_INPUT (1U << 4)

/**
 * @brief Flag value equivalent to \ref FFTW_MEASURE, enabling a larger-than-default set of plan
 * configurations to be considered in the measurements.
 */
#define FFTW_PATIENT (1U << 5)

/**
 * @brief Flag value enforcing hipFFTW to use a heuristric when selecting the plan configuration,
 * thereby ruling out measurements and leaving input and output buffers untouched.
 */
#define FFTW_ESTIMATE (1U << 6)

/**
 * @brief Flag instructing hipFFTW to query the requested plan from an on-disk database file ("wisdom").
 * If not found therein, no plan is created.
 * \note This flag value is not supported by hipFFTW.
 */
#define FFTW_WISDOM_ONLY (1U << 21)

/**
 * @brief Exponent ``sign`` value to be used for forward discrete Fourier transforms.
 */
#define FFTW_FORWARD (-1)

/**
 * @brief Exponent ``sign`` value to be used for backward (inverse) discrete Fourier transforms.
 */
#define FFTW_BACKWARD (1)

// Buffer management
/**
 * @brief Allocates a data buffer accessible by the host.
 * @param[in] n number of bytes desired for the buffer.
 * @return a pointer to the base address of the allocated memory block upon success (``nullptr`` otherwise).
 * 
 * @remark The returned base address is at least 64-bit aligned.
 */
HIPFFT_EXPORT void* fftw_malloc(size_t n);
/**
 * @brief This function is strictly equivalent to \ref fftw_malloc
 */
HIPFFT_EXPORT void* fftwf_malloc(size_t n);
/**
 * @brief This function is strictly equivalent to ``(double*) fftw_malloc(n * sizeof(double))``
 */
HIPFFT_EXPORT double* fftw_alloc_real(size_t n);
/**
 * @brief This function is strictly equivalent to ``(float*) fftw_malloc(n * sizeof(float))``
 */
HIPFFT_EXPORT float* fftwf_alloc_real(size_t n);
/**
 * @brief This function is strictly equivalent to ``(fftw_complex*) fftw_malloc(n * sizeof(fftw_complex))``
 */
HIPFFT_EXPORT fftw_complex* fftw_alloc_complex(size_t n);
/**
 * @brief This function is strictly equivalent to ``(fftwf_complex*) fftw_malloc(n * sizeof(fftwf_complex))``
 */
HIPFFT_EXPORT fftwf_complex* fftwf_alloc_complex(size_t n);

/**
 * @brief Frees a buffer previously allocated by any of the allocation functions above.
 * 
 * @param[in] p pointer to the base address of the buffer to be freed.
 */
HIPFFT_EXPORT void fftw_free(void* p);
/**
 * @brief This function is strictly equivalent to \ref fftw_free
 */
HIPFFT_EXPORT void fftwf_free(void* p);

// Plan creation

// Basic plans
/**
 * @brief Creates a basic plan for a one-dimensional, double-precision, complex discrete Fourier transform of length ``n``.
 * 
 * @param[in] n strictly positive length of the transform;
 * @param[in] in pointer to the input buffer for the transform;
 * @param[in] out pointer to the output buffer for the transform;
 * @param[in] sign exponent sign defining the desired complex transform (``FFTW_FORWARD`` or ``FFTW_BACKWARD``);
 * @param[in] flags bitwise OR (``|``) combination of zero or more constant flag values.
 * @return a valid double-precision hipFFTW plan ready for execution upon success (``nullptr`` otherwise).
 */
HIPFFT_EXPORT fftw_plan
    fftw_plan_dft_1d(int n, fftw_complex* in, fftw_complex* out, int sign, unsigned flags);
/**
 * @brief Single-precision equivalent of \ref fftw_plan_dft_1d.
 */
HIPFFT_EXPORT fftwf_plan
    fftwf_plan_dft_1d(int n, fftwf_complex* in, fftwf_complex* out, int sign, unsigned flags);
/**
 * @brief Creates a basic plan for a two-dimensional, double-precision, complex discrete Fourier transform of lengths ``n0 x n1``.
 * 
 * @param[in] n0, n1 strictly positive lengths of the transform;
 * @param[in] in pointer to the input buffer for the transform;
 * @param[in] out pointer to the output buffer for the transform;
 * @param[in] sign exponent sign defining the desired complex transform (``FFTW_FORWARD`` or ``FFTW_BACKWARD``);
 * @param[in] flags bitwise OR (``|``) combination of zero or more constant flag values.
 * @return a valid double-precision hipFFTW plan ready for execution upon success (``nullptr`` otherwise).
 */
HIPFFT_EXPORT fftw_plan
    fftw_plan_dft_2d(int n0, int n1, fftw_complex* in, fftw_complex* out, int sign, unsigned flags);
/**
 * @brief Single-precision equivalent of \ref fftw_plan_dft_2d.
 */
HIPFFT_EXPORT fftwf_plan fftwf_plan_dft_2d(
    int n0, int n1, fftwf_complex* in, fftwf_complex* out, int sign, unsigned flags);
/**
 * @brief Creates a basic plan for a three-dimensional, double-precision, complex discrete Fourier transform of lengths ``n0 x n1 x n2``.
 * 
 * @param[in] n0, n1,n2 strictly positive lengths of the transform;
 * @param[in] in pointer to the input buffer for the transform;
 * @param[in] out pointer to the output buffer for the transform;
 * @param[in] sign exponent sign defining the desired complex transform (``FFTW_FORWARD`` or ``FFTW_BACKWARD``);
 * @param[in] flags bitwise OR (``|``) combination of zero or more constant flag values.
 * @return a valid double-precision hipFFTW plan ready for execution upon success (``nullptr`` otherwise).
 */
HIPFFT_EXPORT fftw_plan fftw_plan_dft_3d(
    int n0, int n1, int n2, fftw_complex* in, fftw_complex* out, int sign, unsigned flags);
/**
 * @brief Single-precision equivalent of \ref fftw_plan_dft_3d.
 */
HIPFFT_EXPORT fftwf_plan fftwf_plan_dft_3d(
    int n0, int n1, int n2, fftwf_complex* in, fftwf_complex* out, int sign, unsigned flags);
/**
 * @brief Creates a basic plan for a multidimensional, double-precision, complex discrete Fourier transform of lengths ``n[0] x n[1] x ... x n[rank-1]``.
 * 
 * @param[in] rank strictly positive rank of the transform;
 * @param[in] n array of strictly positive lengths of the transform (must be of size ``rank``);
 * @param[in] in pointer to the input buffer for the transform;
 * @param[in] out pointer to the output buffer for the transform;
 * @param[in] sign exponent sign defining the desired complex transform (``FFTW_FORWARD`` or ``FFTW_BACKWARD``);
 * @param[in] flags bitwise OR (``|``) combination of zero or more constant flag values.
 * @return a valid double-precision hipFFTW plan ready for execution upon success (``nullptr`` otherwise).
 */
HIPFFT_EXPORT fftw_plan fftw_plan_dft(
    int rank, const int* n, fftw_complex* in, fftw_complex* out, int sign, unsigned flags);
/**
 * @brief Single-precision equivalent of \ref fftw_plan_dft.
 */
HIPFFT_EXPORT fftwf_plan fftwf_plan_dft(
    int rank, const int* n, fftwf_complex* in, fftwf_complex* out, int sign, unsigned flags);
/**
 * @brief Creates a basic plan for a one-dimensional, double-precision, real forward discrete Fourier transform of length ``n``.
 * 
 * @param[in] n strictly positive length of the transform;
 * @param[in] in pointer to the input buffer for the transform;
 * @param[in] out pointer to the output buffer for the transform;
 * @param[in] flags bitwise OR (``|``) combination of zero or more constant flag values.
 * @return a valid double-precision hipFFTW plan ready for execution upon success (``nullptr`` otherwise).
 */
HIPFFT_EXPORT fftw_plan fftw_plan_dft_r2c_1d(int n, double* in, fftw_complex* out, unsigned flags);
/**
 * @brief Single-precision equivalent of \ref fftw_plan_dft_r2c_1d.
 */
HIPFFT_EXPORT fftwf_plan fftwf_plan_dft_r2c_1d(int            n,
                                               float*         in,
                                               fftwf_complex* out,
                                               unsigned       flags);
/**
 * @brief Creates a basic plan for a two-dimensional, double-precision, real forward discrete Fourier transform of lengths ``n0 x n1``.
 * 
 * @param[in] n0, n1 strictly positive lengths of the transform;
 * @param[in] in pointer to the input buffer for the transform;
 * @param[in] out pointer to the output buffer for the transform;
 * @param[in] flags bitwise OR (``|``) combination of zero or more constant flag values.
 * @return a valid double-precision hipFFTW plan ready for execution upon success (``nullptr`` otherwise).
 */
HIPFFT_EXPORT fftw_plan
    fftw_plan_dft_r2c_2d(int n0, int n1, double* in, fftw_complex* out, unsigned flags);
/**
 * @brief Single-precision equivalent of \ref fftw_plan_dft_r2c_2d. 
 */
HIPFFT_EXPORT fftwf_plan
    fftwf_plan_dft_r2c_2d(int n0, int n1, float* in, fftwf_complex* out, unsigned flags);
/**
 * @brief Creates a basic plan for a three-dimensional, double-precision, real forward discrete Fourier transform of lengths ``n0 x n1 x n2``.
 * 
 * @param[in] n0, n1, n2 strictly positive lengths of the transform;
 * @param[in] in pointer to the input buffer for the transform;
 * @param[in] out pointer to the output buffer for the transform;
 * @param[in] flags bitwise OR (``|``) combination of zero or more constant flag values.
 * @return a valid double-precision hipFFTW plan ready for execution upon success (``nullptr`` otherwise).
 */
HIPFFT_EXPORT fftw_plan
    fftw_plan_dft_r2c_3d(int n0, int n1, int n2, double* in, fftw_complex* out, unsigned flags);
/**
 * @brief Single-precision equivalent of \ref fftw_plan_dft_r2c_3d. 
 */
HIPFFT_EXPORT fftwf_plan
    fftwf_plan_dft_r2c_3d(int n0, int n1, int n2, float* in, fftwf_complex* out, unsigned flags);
/**
 * @brief Creates a basic plan for a multidimensional, double-precision, real forward discrete Fourier transform of lengths ``n[0] x n[1] x ... x n[rank-1]``.
 * 
 * @param[in] rank strictly positive rank of the transform;
 * @param[in] n array of strictly positive lengths of the transform (must be of size ``rank``);
 * @param[in] in pointer to the input buffer for the transform;
 * @param[in] out pointer to the output buffer for the transform;
 * @param[in] flags bitwise OR (``|``) combination of zero or more constant flag values.
 * @return a valid double-precision hipFFTW plan ready for execution upon success (``nullptr`` otherwise).
 */
HIPFFT_EXPORT fftw_plan
    fftw_plan_dft_r2c(int rank, const int* n, double* in, fftw_complex* out, unsigned flags);
/**
 * @brief Single-precision equivalent of \ref fftw_plan_dft_r2c. 
 */
HIPFFT_EXPORT fftwf_plan
    fftwf_plan_dft_r2c(int rank, const int* n, float* in, fftwf_complex* out, unsigned flags);
/**
 * @brief Creates a basic plan for a one-dimensional, double-precision, real backward (inverse) discrete Fourier transform of length ``n``.
 * 
 * @param[in] n strictly positive length of the transform;
 * @param[in] in pointer to the input buffer for the transform;
 * @param[in] out pointer to the output buffer for the transform;
 * @param[in] flags bitwise OR (``|``) combination of zero or more constant flag values.
 * @return a valid double-precision hipFFTW plan ready for execution upon success (``nullptr`` otherwise).
 */
HIPFFT_EXPORT fftw_plan fftw_plan_dft_c2r_1d(int n, fftw_complex* in, double* out, unsigned flags);
/**
 * @brief Single-precision equivalent of \ref fftw_plan_dft_c2r_1d.
 */
HIPFFT_EXPORT fftwf_plan fftwf_plan_dft_c2r_1d(int            n,
                                               fftwf_complex* in,
                                               float*         out,
                                               unsigned       flags);
/**
 * @brief Creates a basic plan for a two-dimensional, double-precision, real backward (inverse) discrete Fourier transform of lengths ``n0 x n1``.
 * 
 * @param[in] n0, n1 strictly positive lengths of the transform;
 * @param[in] in pointer to the input buffer for the transform;
 * @param[in] out pointer to the output buffer for the transform;
 * @param[in] flags bitwise OR (``|``) combination of zero or more constant flag values.
 * @return a valid double-precision hipFFTW plan ready for execution upon success (``nullptr`` otherwise).
 */
HIPFFT_EXPORT fftw_plan
    fftw_plan_dft_c2r_2d(int n0, int n1, fftw_complex* in, double* out, unsigned flags);
/**
 * @brief Single-precision equivalent of \ref fftw_plan_dft_c2r_2d.
 */
HIPFFT_EXPORT fftwf_plan
    fftwf_plan_dft_c2r_2d(int n0, int n1, fftwf_complex* in, float* out, unsigned flags);
/**
 * @brief Creates a basic plan for a three-dimensional, double-precision, real backward (inverse) discrete Fourier transform of lengths ``n0 x n1 x n2``.
 * 
 * @param[in] n0, n1, n2 strictly positive lengths of the transform;
 * @param[in] in pointer to the input buffer for the transform;
 * @param[in] out pointer to the output buffer for the transform;
 * @param[in] flags bitwise OR (``|``) combination of zero or more constant flag values.
 * @return a valid double-precision hipFFTW plan ready for execution upon success (``nullptr`` otherwise).
 */
HIPFFT_EXPORT fftw_plan
    fftw_plan_dft_c2r_3d(int n0, int n1, int n2, fftw_complex* in, double* out, unsigned flags);
/**
 * @brief Single-precision equivalent of \ref fftw_plan_dft_c2r_3d.
 */
HIPFFT_EXPORT fftwf_plan
    fftwf_plan_dft_c2r_3d(int n0, int n1, int n2, fftwf_complex* in, float* out, unsigned flags);
/**
 * @brief Creates a basic plan for a multidimensional, double-precision, real backward (inverse) discrete Fourier transform of lengths ``n[0] x n[1] x ... x n[rank-1]``.
 * 
 * @param[in] rank strictly positive rank of the transform;
 * @param[in] n array of strictly positive lengths of the transform (must be of size ``rank``);
 * @param[in] in pointer to the input buffer for the transform;
 * @param[in] out pointer to the output buffer for the transform;
 * @param[in] flags bitwise OR (``|``) combination of zero or more constant flag values.
 * @return a valid double-precision hipFFTW plan ready for execution upon success (``nullptr`` otherwise).
 */
HIPFFT_EXPORT fftw_plan
    fftw_plan_dft_c2r(int rank, const int* n, fftw_complex* in, double* out, unsigned flags);
/**
 * @brief Single-precision equivalent of \ref fftw_plan_dft_c2r.
 */
HIPFFT_EXPORT fftwf_plan
    fftwf_plan_dft_c2r(int rank, const int* n, fftwf_complex* in, float* out, unsigned flags);

// Advanced plans
/**
 * @brief Creates an advanced plan for a multidimensional, double-precision, complex discrete Fourier transform
 * of lengths ``n[0] x n[1] x ... x n[rank-1]`` and batch size ``howmany``.
 * 
 * @param[in] rank strictly positive rank of the transform;
 * @param[in] n array of strictly positive lengths of the transform (must be of size ``rank``);
 * @param[in] howmany strictly positive batch size;
 * @param[in] in pointer to the input buffer for the transform;
 * @param[in] inembed array of strictly positive input-embedding lengths (must be of size ``rank``). Default input-embedding is considered if set to ``NULL``;
 * @param[in] istride strictly positive elementary stride in input data (along the last dimension);
 * @param[in] idist strictly positive distance between consecutive input data sequences in the batch;
 * @param[in] out pointer to the output buffer for the transform;
 * @param[in] onembed array of strictly positive output-embedding lengths (must be of size ``rank``). Default output-embedding is considered if set to ``NULL``;
 * @param[in] ostride strictly positive elementary stride in output data (along the last dimension);
 * @param[in] odist strictly positive distance between consecutive output data sequences in the batch;
 * @param[in] sign exponent sign defining the desired complex transform (``FFTW_FORWARD`` or ``FFTW_BACKWARD``);
 * @param[in] flags bitwise OR (``|``) combination of zero or more constant flag values.
 * @return a valid double-precision hipFFTW plan ready for execution upon success (``nullptr`` otherwise).
 */
HIPFFT_EXPORT fftw_plan fftw_plan_many_dft(int           rank,
                                           const int*    n,
                                           int           howmany,
                                           fftw_complex* in,
                                           const int*    inembed,
                                           int           istride,
                                           int           idist,
                                           fftw_complex* out,
                                           const int*    onembed,
                                           int           ostride,
                                           int           odist,
                                           int           sign,
                                           unsigned      flags);
/**
 * @brief Single-precision equivalent of \ref fftw_plan_many_dft.
 */
HIPFFT_EXPORT fftwf_plan fftwf_plan_many_dft(int            rank,
                                             const int*     n,
                                             int            howmany,
                                             fftwf_complex* in,
                                             const int*     inembed,
                                             int            istride,
                                             int            idist,
                                             fftwf_complex* out,
                                             const int*     onembed,
                                             int            ostride,
                                             int            odist,
                                             int            sign,
                                             unsigned       flags);
/**
 * @brief Creates an advanced plan for a multidimensional, double-precision, real forward discrete Fourier
 * transform of lengths ``n[0] x n[1] x ... x n[rank-1]`` and batch size ``howmany``.
 * 
 * @param[in] rank strictly positive rank of the transform;
 * @param[in] n array of strictly positive lengths of the transform (must be of size ``rank``);
 * @param[in] howmany strictly positive batch size;
 * @param[in] in pointer to the input buffer for the transform;
 * @param[in] inembed array of strictly positive input-embedding lengths (must be of size ``rank``). Default input-embedding is considered if set to ``NULL``;
 * @param[in] istride strictly positive elementary stride in input data (along the last dimension);
 * @param[in] idist strictly positive distance between consecutive input data sequences in the batch;
 * @param[in] out pointer to the output buffer for the transform;
 * @param[in] onembed array of strictly positive output-embedding lengths (must be of size ``rank``). Default output-embedding is considered if set to ``NULL``;
 * @param[in] ostride strictly positive elementary stride in output data (along the last dimension);
 * @param[in] odist strictly positive distance between consecutive output data sequences in the batch;
 * @param[in] flags bitwise OR (``|``) combination of zero or more constant flag values.
 * @return a valid double-precision hipFFTW plan ready for execution upon success (``nullptr`` otherwise).
 */
HIPFFT_EXPORT fftw_plan fftw_plan_many_dft_r2c(int           rank,
                                               const int*    n,
                                               int           howmany,
                                               double*       in,
                                               const int*    inembed,
                                               int           istride,
                                               int           idist,
                                               fftw_complex* out,
                                               const int*    onembed,
                                               int           ostride,
                                               int           odist,
                                               unsigned      flags);
/**
 * @brief Single-precision equivalent of \ref fftw_plan_many_dft_r2c.
 */
HIPFFT_EXPORT fftwf_plan fftwf_plan_many_dft_r2c(int            rank,
                                                 const int*     n,
                                                 int            howmany,
                                                 float*         in,
                                                 const int*     inembed,
                                                 int            istride,
                                                 int            idist,
                                                 fftwf_complex* out,
                                                 const int*     onembed,
                                                 int            ostride,
                                                 int            odist,
                                                 unsigned       flags);
/**
 * @brief Creates an advanced plan for a multidimensional, double-precision, real backward (inverse)
 * discrete Fourier transform of lengths ``n[0] x n[1] x ... x n[rank-1]`` and batch size ``howmany``.
 * 
 * @param[in] rank strictly positive rank of the transform;
 * @param[in] n array of strictly positive lengths of the transform (must be of size ``rank``);
 * @param[in] howmany strictly positive batch size;
 * @param[in] in pointer to the input buffer for the transform;
 * @param[in] inembed array of strictly positive input-embedding lengths (must be of size ``rank``). Default input-embedding is considered if set to ``NULL``;
 * @param[in] istride strictly positive elementary stride in input data (along the last dimension);
 * @param[in] idist strictly positive distance between consecutive input data sequences in the batch;
 * @param[in] out pointer to the output buffer for the transform;
 * @param[in] onembed array of strictly positive output-embedding lengths (must be of size ``rank``). Default output-embedding is considered if set to ``NULL``;
 * @param[in] ostride strictly positive elementary stride in output data (along the last dimension);
 * @param[in] odist strictly positive distance between consecutive output data sequences in the batch;
 * @param[in] flags bitwise OR (``|``) combination of zero or more constant flag values.
 * @return a valid double-precision hipFFTW plan ready for execution upon success (``nullptr`` otherwise).
 */
HIPFFT_EXPORT fftw_plan fftw_plan_many_dft_c2r(int           rank,
                                               const int*    n,
                                               int           howmany,
                                               fftw_complex* in,
                                               const int*    inembed,
                                               int           istride,
                                               int           idist,
                                               double*       out,
                                               const int*    onembed,
                                               int           ostride,
                                               int           odist,
                                               unsigned      flags);
/**
 * @brief Single-precision equivalent of \ref fftw_plan_many_dft_c2r.
 */
HIPFFT_EXPORT fftwf_plan fftwf_plan_many_dft_c2r(int            rank,
                                                 const int*     n,
                                                 int            howmany,
                                                 fftwf_complex* in,
                                                 const int*     inembed,
                                                 int            istride,
                                                 int            idist,
                                                 float*         out,
                                                 const int*     onembed,
                                                 int            ostride,
                                                 int            odist,
                                                 unsigned       flags);
// guru plans
/**
 * @brief Creates an arbitrary plan for a multidimensional, double-precision, complex
 * discrete Fourier transform of lengths ``dims[0].n x dims[1].n x ... x dims[rank-1].n``
 * and batch sizes ``howmany_dims[0].n x howmany_dims[1].n x ... x howmany_dims[howmany_rank-1].n``.
 * 
 * @param[in] rank strictly positive rank of the transform;
 * @param[in] dims array of ``rank`` ``fftw_iodim`` values;
 * @param[in] howmany_rank strictly positive rank of the transform's batch sizes;
 * @param[in] howmany_dims array of ``howmany_rank`` ``fftw_iodim`` values;
 * @param[in] in pointer to the input buffer for the transform;
 * @param[in] out pointer to the output buffer for the transform;
 * @param[in] sign exponent sign defining the desired complex transform (``FFTW_FORWARD`` or ``FFTW_BACKWARD``);
 * @param[in] flags bitwise OR (``|``) combination of zero or more constant flag values.
 * @return a valid double-precision hipFFTW plan ready for execution upon success (``nullptr`` otherwise).
 */
HIPFFT_EXPORT fftw_plan fftw_plan_guru_dft(int               rank,
                                           const fftw_iodim* dims,
                                           int               howmany_rank,
                                           const fftw_iodim* howmany_dims,
                                           fftw_complex*     in,
                                           fftw_complex*     out,
                                           int               sign,
                                           unsigned          flags);
/**
 * @brief Single-precision equivalent of \ref fftw_plan_guru_dft.
 */
HIPFFT_EXPORT fftwf_plan fftwf_plan_guru_dft(int                rank,
                                             const fftwf_iodim* dims,
                                             int                howmany_rank,
                                             const fftwf_iodim* howmany_dims,
                                             fftwf_complex*     in,
                                             fftwf_complex*     out,
                                             int                sign,
                                             unsigned           flags);
/**
 * @brief Creates an arbitrary plan for a multidimensional, double-precision, real forward
 * discrete Fourier transform of lengths ``dims[0].n x dims[1].n x ... x dims[rank-1].n``
 * and batch sizes ``howmany_dims[0].n x howmany_dims[1].n x ... x howmany_dims[howmany_rank-1].n``.
 * 
 * @param[in] rank strictly positive rank of the transform;
 * @param[in] dims array of ``rank`` ``fftw_iodim`` values;
 * @param[in] howmany_rank strictly positive rank of the transform's batch sizes;
 * @param[in] howmany_dims array of ``howmany_rank`` ``fftw_iodim`` values;
 * @param[in] in pointer to the input buffer for the transform;
 * @param[in] out pointer to the output buffer for the transform;
 * @param[in] flags bitwise OR (``|``) combination of zero or more constant flag values.
 * @return a valid double-precision hipFFTW plan ready for execution upon success (``nullptr`` otherwise).
 */
HIPFFT_EXPORT fftw_plan fftw_plan_guru_dft_r2c(int               rank,
                                               const fftw_iodim* dims,
                                               int               howmany_rank,
                                               const fftw_iodim* howmany_dims,
                                               double*           in,
                                               fftw_complex*     out,
                                               unsigned          flags);
/**
 * @brief Single-precision equivalent of \ref fftw_plan_guru_dft_r2c.
 */
HIPFFT_EXPORT fftwf_plan fftwf_plan_guru_dft_r2c(int                rank,
                                                 const fftwf_iodim* dims,
                                                 int                howmany_rank,
                                                 const fftwf_iodim* howmany_dims,
                                                 float*             in,
                                                 fftwf_complex*     out,
                                                 unsigned           flags);
/**
 * @brief Creates an arbitrary plan for a multidimensional, double-precision, real backward
 * (inverse) discrete Fourier transform of lengths ``dims[0].n x dims[1].n x ... x dims[rank-1].n``
 * and batch sizes ``howmany_dims[0].n x howmany_dims[1].n x ... x howmany_dims[howmany_rank-1].n``.
 * 
 * @param[in] rank strictly positive rank of the transform;
 * @param[in] dims array of ``rank`` ``fftw_iodim`` values;
 * @param[in] howmany_rank strictly positive rank of the transform's batch sizes;
 * @param[in] howmany_dims array of ``howmany_rank`` ``fftw_iodim`` values;
 * @param[in] in pointer to the input buffer for the transform;
 * @param[in] out pointer to the output buffer for the transform;
 * @param[in] flags bitwise OR (``|``) combination of zero or more constant flag values.
 * @return a valid double-precision hipFFTW plan ready for execution upon success (``nullptr`` otherwise).
 */
HIPFFT_EXPORT fftw_plan fftw_plan_guru_dft_c2r(int               rank,
                                               const fftw_iodim* dims,
                                               int               howmany_rank,
                                               const fftw_iodim* howmany_dims,
                                               fftw_complex*     in,
                                               double*           out,
                                               unsigned          flags);
/**
 * @brief Single-precision equivalent of \ref fftw_plan_guru_dft_c2r.
 */
HIPFFT_EXPORT fftwf_plan fftwf_plan_guru_dft_c2r(int                rank,
                                                 const fftwf_iodim* dims,
                                                 int                howmany_rank,
                                                 const fftwf_iodim* howmany_dims,
                                                 fftwf_complex*     in,
                                                 float*             out,
                                                 unsigned           flags);
// guru64 plans
/**
 * @brief Equivalent of \ref fftw_plan_guru_dft using layout-describing values of type
 * ``fftw_iodim64`` instead of ``fftw_iodim``.
 */
HIPFFT_EXPORT fftw_plan fftw_plan_guru64_dft(int                 rank,
                                             const fftw_iodim64* dims,
                                             int                 howmany_rank,
                                             const fftw_iodim64* howmany_dims,
                                             fftw_complex*       in,
                                             fftw_complex*       out,
                                             int                 sign,
                                             unsigned            flags);
/**
 * @brief Equivalent of \ref fftwf_plan_guru_dft using layout-describing values of type
 * ``fftwf_iodim64`` instead of ``fftwf_iodim``.
 */
HIPFFT_EXPORT fftwf_plan fftwf_plan_guru64_dft(int                  rank,
                                               const fftwf_iodim64* dims,
                                               int                  howmany_rank,
                                               const fftwf_iodim64* howmany_dims,
                                               fftwf_complex*       in,
                                               fftwf_complex*       out,
                                               int                  sign,
                                               unsigned             flags);
/**
 * @brief Equivalent of \ref fftw_plan_guru_dft_r2c using layout-describing values of type
 * ``fftw_iodim64`` instead of ``fftw_iodim``.
 */
HIPFFT_EXPORT fftw_plan fftw_plan_guru64_dft_r2c(int                 rank,
                                                 const fftw_iodim64* dims,
                                                 int                 howmany_rank,
                                                 const fftw_iodim64* howmany_dims,
                                                 double*             in,
                                                 fftw_complex*       out,
                                                 unsigned            flags);
/**
 * @brief Equivalent of \ref fftwf_plan_guru_dft_r2c using layout-describing values of type
 * ``fftwf_iodim64`` instead of ``fftwf_iodim``.
 */
HIPFFT_EXPORT fftwf_plan fftwf_plan_guru64_dft_r2c(int                  rank,
                                                   const fftwf_iodim64* dims,
                                                   int                  howmany_rank,
                                                   const fftwf_iodim64* howmany_dims,
                                                   float*               in,
                                                   fftwf_complex*       out,
                                                   unsigned             flags);
/**
 * @brief Equivalent of \ref fftw_plan_guru_dft_c2r using layout-describing values of type
 * ``fftw_iodim64`` instead of ``fftw_iodim``.
 */
HIPFFT_EXPORT fftw_plan fftw_plan_guru64_dft_c2r(int                 rank,
                                                 const fftw_iodim64* dims,
                                                 int                 howmany_rank,
                                                 const fftw_iodim64* howmany_dims,
                                                 fftw_complex*       in,
                                                 double*             out,
                                                 unsigned            flags);
/**
 * @brief Equivalent of \ref fftwf_plan_guru_dft_c2r using layout-describing values of type
 * ``fftwf_iodim64`` instead of ``fftwf_iodim``.
 */
HIPFFT_EXPORT fftwf_plan fftwf_plan_guru64_dft_c2r(int                  rank,
                                                   const fftwf_iodim64* dims,
                                                   int                  howmany_rank,
                                                   const fftwf_iodim64* howmany_dims,
                                                   fftwf_complex*       in,
                                                   float*               out,
                                                   unsigned             flags);

// Plan execution
/**
 * @brief Computes the discrete Fourier transform that a double-precision plan captures using
 * the input and output data buffers that were communicated at plan's creation.
 * 
 * @param[in] plan the double-precision plan capturing the transform to compute.
 */
HIPFFT_EXPORT void fftw_execute(const fftw_plan plan);
/**
 * @brief Single-precision equivalent of \ref fftw_execute 
 */
HIPFFT_EXPORT void fftwf_execute(const fftwf_plan plan);
/**
 * @brief Computes the discrete Fourier transform that a double-precision plan captures using new input and output data buffers.
 * The plan must have been created for a complex transform.
 * 
 * @param[in] plan the double-precision plan capturing the complex transform to compute;
 * @param[in] in pointer to a new input buffer for the transform;
 * @param[out] out pointer to a new output buffer for the transform.
 */
HIPFFT_EXPORT void fftw_execute_dft(const fftw_plan plan, fftw_complex* in, fftw_complex* out);
/**
 * @brief Single-precision equivalent of \ref fftw_execute_dft. 
 */
HIPFFT_EXPORT void fftwf_execute_dft(const fftwf_plan plan, fftwf_complex* in, fftwf_complex* out);
/**
 * @brief Computes the discrete Fourier transform that a double-precision plan captures using new input and output data buffers.
 * The plan must have been created for a real forward transform.
 * 
 * @param[in] plan the double-precision plan capturing the real forward transform to compute;
 * @param[in] in pointer to a new input buffer for the transform;
 * @param[out] out pointer to a new output buffer for the transform.
 */
HIPFFT_EXPORT void fftw_execute_dft_r2c(const fftw_plan plan, double* in, fftw_complex* out);
/**
 * @brief Single-precision equivalent of \ref fftw_execute_dft_r2c. 
 */
HIPFFT_EXPORT void fftwf_execute_dft_r2c(const fftwf_plan plan, float* in, fftwf_complex* out);
/**
 * @brief Computes the discrete Fourier transform that a double-precision plan captures using new input and output data buffers.
 * The plan must have been created for a real backward (inverse) transform.
 * 
 * @param[in] plan the double-precision plan capturing the real backward (inverse) transform to compute;
 * @param[in] in pointer to a new input buffer for the transform;
 * @param[out] out pointer to a new output buffer for the transform.
 */
HIPFFT_EXPORT void fftw_execute_dft_c2r(const fftw_plan plan, fftw_complex* in, double* out);
/**
 * @brief Single-precision equivalent of \ref fftw_execute_dft_c2r. 
 */
HIPFFT_EXPORT void fftwf_execute_dft_c2r(const fftwf_plan plan, fftwf_complex* in, float* out);

// Plan destruction
/**
 * @brief Deallocates a double-precision plan and frees all its resources.
 * 
 * @param[in] plan plan to be destroyed.
 */
HIPFFT_EXPORT void fftw_destroy_plan(fftw_plan plan);
/**
 * @brief Single-precision equivalent of \ref fftw_destroy_plan. 
 */
HIPFFT_EXPORT void fftwf_destroy_plan(fftwf_plan plan);

// Non-functional utility APIs
HIPFFT_EXPORT void   fftw_print_plan(const fftw_plan);
HIPFFT_EXPORT void   fftwf_print_plan(const fftwf_plan);
HIPFFT_EXPORT void   fftw_set_timelimit(double);
HIPFFT_EXPORT void   fftwf_set_timelimit(double);
HIPFFT_EXPORT double fftw_cost(const fftw_plan);
HIPFFT_EXPORT double fftwf_cost(const fftw_plan);
HIPFFT_EXPORT void   fftw_flops(const fftw_plan, double*, double*, double*);
HIPFFT_EXPORT void   fftwf_flops(const fftw_plan, double*, double*, double*);
HIPFFT_EXPORT void   fftw_cleanup();
HIPFFT_EXPORT void   fftwf_cleanup();

#ifdef __cplusplus
}
#endif

#endif
