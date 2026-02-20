// Copyright (C) 2016 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once
#ifndef FFTWTRANSFORM_H
#define FFTWTRANSFORM_H

#include "hostbuf.h"
#include "rocfft_complex.h"
#include "test_params.h"
#include <fftw3.h>
#include <vector>

// Function to return maximum error for float and double types.
//
// Following Schatzman (1996; Accuracy of the Discrete Fourier
// Transform and the Fast Fourier Transform), the shape of relative
// l_2 error vs length should look like
//
//   epsilon * sqrt(log2(length)).
//
// The magic epsilon constants below were chosen so that we get a
// reasonable upper bound for (all of) our tests.
//
// For rocFFT, prime lengths result in the highest error.  As such,
// the epsilons below are perhaps too loose for pow2 lengths; but they
// are appropriate for prime lengths.
template <typename Tfloat>
inline double type_epsilon();
template <>
inline double type_epsilon<rocfft_fp16>()
{
    return half_epsilon;
}
template <>
inline double type_epsilon<float>()
{
    return single_epsilon;
}
template <>
inline double type_epsilon<double>()
{
    return double_epsilon;
}

static constexpr double default_half_epsilon()
{
    return 9.77e-4;
}

static constexpr double default_single_epsilon()
{
    return 3.75e-5;
}

static constexpr double default_double_epsilon()
{
    return 1e-15;
}

// C++ traits to translate float->fftwf_complex and
// double->fftw_complex.
// The correct FFTW complex type can be accessed via, for example,
// using complex_t = typename fftw_complex_trait<Tfloat>::complex_t;
template <typename Tfloat>
struct fftw_trait;
template <>
struct fftw_trait<rocfft_fp16>
{
    // fftw does not support half precision, so use single precision and convert
    using fftw_complex_type = fftwf_complex;
    using fftw_plan_type    = fftwf_plan;
};
template <>
struct fftw_trait<float>
{
    using fftw_complex_type = fftwf_complex;
    using fftw_plan_type    = fftwf_plan;
};
template <>
struct fftw_trait<double>
{
    using fftw_complex_type = fftw_complex;
    using fftw_plan_type    = fftw_plan;
};

// Copies the half-precision input buffer to a single-precision
// buffer.  Note that the input buffer is already sized like it's a
// single-precision buffer (but only half of it is filled), because
// we allocate a single-precision buffer for FFTW to plan with.
static hostbuf half_to_single_copy(const hostbuf& in)
{
    auto out      = in.copy();
    auto in_begin = reinterpret_cast<const rocfft_fp16*>(in.data());
    std::copy_n(
        in_begin, in.size() / sizeof(rocfft_fp16) / 2, reinterpret_cast<float*>(out.data()));
    return out;
}

// converts a wider precision buffer to a narrower precision, in-place
template <typename TfloatIn, typename TfloatOut>
void narrow_precision_inplace(hostbuf& in)
{
    // ensure we're actually shrinking the data
    static_assert(sizeof(TfloatIn) > sizeof(TfloatOut));

    auto readPtr  = reinterpret_cast<const TfloatIn*>(in.data());
    auto writePtr = reinterpret_cast<TfloatOut*>(in.data());
    std::copy_n(readPtr, in.size() / sizeof(TfloatIn), writePtr);
    in.shrink(in.size() / (sizeof(TfloatIn) / sizeof(TfloatOut)));
}

static void single_to_half_inplace(hostbuf& in)
{
    narrow_precision_inplace<float, rocfft_fp16>(in);
}

// Template wrappers for real-valued FFTW allocators:
template <typename Tfloat>
inline Tfloat* fftw_alloc_real_type(size_t n);
template <>
inline float* fftw_alloc_real_type<float>(size_t n)
{
    return fftwf_alloc_real(n);
}
template <>
inline double* fftw_alloc_real_type<double>(size_t n)
{
    return fftw_alloc_real(n);
}

// Template wrappers for complex-valued FFTW allocators:
template <typename Tfloat>
inline typename fftw_trait<Tfloat>::fftw_complex_type* fftw_alloc_complex_type(size_t n);
template <>
inline typename fftw_trait<float>::fftw_complex_type* fftw_alloc_complex_type<float>(size_t n)
{
    return fftwf_alloc_complex(n);
}
template <>
inline typename fftw_trait<double>::fftw_complex_type* fftw_alloc_complex_type<double>(size_t n)
{
    return fftw_alloc_complex(n);
}

template <typename fftw_type>
inline fftw_type* fftw_alloc_type(size_t n);
template <>
inline float* fftw_alloc_type<float>(size_t n)
{
    return fftw_alloc_real_type<float>(n);
}
template <>
inline double* fftw_alloc_type<double>(size_t n)
{
    return fftw_alloc_real_type<double>(n);
}
template <>
inline fftwf_complex* fftw_alloc_type<fftwf_complex>(size_t n)
{
    return fftw_alloc_complex_type<float>(n);
}
template <>
inline fftw_complex* fftw_alloc_type<fftw_complex>(size_t n)
{
    return fftw_alloc_complex_type<double>(n);
}
template <>
inline rocfft_complex<float>* fftw_alloc_type<rocfft_complex<float>>(size_t n)
{
    return (rocfft_complex<float>*)fftw_alloc_complex_type<float>(n);
}
template <>
inline rocfft_complex<double>* fftw_alloc_type<rocfft_complex<double>>(size_t n)
{
    return (rocfft_complex<double>*)fftw_alloc_complex_type<double>(n);
}

// Template wrappers for FFTW plan executors:
template <typename Tfloat>
inline void fftw_execute_type(typename fftw_trait<Tfloat>::fftw_plan_type plan);
template <>
inline void fftw_execute_type<float>(typename fftw_trait<float>::fftw_plan_type plan)
{
    return fftwf_execute(plan);
}
template <>
inline void fftw_execute_type<double>(typename fftw_trait<double>::fftw_plan_type plan)
{
    return fftw_execute(plan);
}

// Template wrappers for FFTW plan destroyers:
template <typename Tfftw_plan>
inline void fftw_destroy_plan_type(Tfftw_plan plan);
template <>
inline void fftw_destroy_plan_type<fftwf_plan>(fftwf_plan plan)
{
    return fftwf_destroy_plan(plan);
}
template <>
inline void fftw_destroy_plan_type<fftw_plan>(fftw_plan plan)
{
    return fftw_destroy_plan(plan);
}

// Template wrappers for FFTW c2c planners:
template <typename Tfloat>
inline typename fftw_trait<Tfloat>::fftw_plan_type
    fftw_plan_guru64_dft(int                                             rank,
                         const fftw_iodim64*                             dims,
                         int                                             howmany_rank,
                         const fftw_iodim64*                             howmany_dims,
                         typename fftw_trait<Tfloat>::fftw_complex_type* in,
                         typename fftw_trait<Tfloat>::fftw_complex_type* out,
                         int                                             sign,
                         unsigned                                        flags);

template <>
inline typename fftw_trait<rocfft_fp16>::fftw_plan_type
    fftw_plan_guru64_dft<rocfft_fp16>(int                 rank,
                                      const fftw_iodim64* dims,
                                      int                 howmany_rank,
                                      const fftw_iodim64* howmany_dims,
                                      typename fftw_trait<rocfft_fp16>::fftw_complex_type* in,
                                      typename fftw_trait<rocfft_fp16>::fftw_complex_type* out,
                                      int                                                  sign,
                                      unsigned                                             flags)
{
    return fftwf_plan_guru64_dft(rank, dims, howmany_rank, howmany_dims, in, out, sign, flags);
}

template <>
inline typename fftw_trait<float>::fftw_plan_type
    fftw_plan_guru64_dft<float>(int                                            rank,
                                const fftw_iodim64*                            dims,
                                int                                            howmany_rank,
                                const fftw_iodim64*                            howmany_dims,
                                typename fftw_trait<float>::fftw_complex_type* in,
                                typename fftw_trait<float>::fftw_complex_type* out,
                                int                                            sign,
                                unsigned                                       flags)
{
    return fftwf_plan_guru64_dft(rank, dims, howmany_rank, howmany_dims, in, out, sign, flags);
}

template <>
inline typename fftw_trait<double>::fftw_plan_type
    fftw_plan_guru64_dft<double>(int                                             rank,
                                 const fftw_iodim64*                             dims,
                                 int                                             howmany_rank,
                                 const fftw_iodim64*                             howmany_dims,
                                 typename fftw_trait<double>::fftw_complex_type* in,
                                 typename fftw_trait<double>::fftw_complex_type* out,
                                 int                                             sign,
                                 unsigned                                        flags)
{
    return fftw_plan_guru64_dft(rank, dims, howmany_rank, howmany_dims, in, out, sign, flags);
}

// Template wrappers for FFTW c2c executors:
template <typename Tfloat>
inline void fftw_plan_execute_c2c(typename fftw_trait<Tfloat>::fftw_plan_type plan,
                                  std::vector<hostbuf>&                       in,
                                  std::vector<hostbuf>&                       out);

template <>
inline void
    fftw_plan_execute_c2c<rocfft_fp16>(typename fftw_trait<rocfft_fp16>::fftw_plan_type plan,
                                       std::vector<hostbuf>&                            in,
                                       std::vector<hostbuf>&                            out)
{
    // since FFTW does not natively support half precision, convert
    // input to single, execute, then convert output back to half
    auto in_single = half_to_single_copy(in.front());
    fftwf_execute_dft(plan,
                      reinterpret_cast<fftwf_complex*>(in_single.data()),
                      reinterpret_cast<fftwf_complex*>(out.front().data()));
    single_to_half_inplace(out.front());
}

template <>
inline void fftw_plan_execute_c2c<float>(typename fftw_trait<float>::fftw_plan_type plan,
                                         std::vector<hostbuf>&                      in,
                                         std::vector<hostbuf>&                      out)
{
    fftwf_execute_dft(plan,
                      reinterpret_cast<fftwf_complex*>(in.front().data()),
                      reinterpret_cast<fftwf_complex*>(out.front().data()));
}

template <>
inline void fftw_plan_execute_c2c<double>(typename fftw_trait<double>::fftw_plan_type plan,
                                          std::vector<hostbuf>&                       in,
                                          std::vector<hostbuf>&                       out)
{
    fftw_execute_dft(plan,
                     reinterpret_cast<fftw_complex*>(in.front().data()),
                     reinterpret_cast<fftw_complex*>(out.front().data()));
}

// Template wrappers for FFTW r2c planners:
template <typename Tfloat>
inline typename fftw_trait<Tfloat>::fftw_plan_type
    fftw_plan_guru64_r2c(int                                             rank,
                         const fftw_iodim64*                             dims,
                         int                                             howmany_rank,
                         const fftw_iodim64*                             howmany_dims,
                         Tfloat*                                         in,
                         typename fftw_trait<Tfloat>::fftw_complex_type* out,
                         unsigned                                        flags);
template <>
inline typename fftw_trait<rocfft_fp16>::fftw_plan_type
    fftw_plan_guru64_r2c<rocfft_fp16>(int                 rank,
                                      const fftw_iodim64* dims,
                                      int                 howmany_rank,
                                      const fftw_iodim64* howmany_dims,
                                      rocfft_fp16*        in,
                                      typename fftw_trait<rocfft_fp16>::fftw_complex_type* out,
                                      unsigned                                             flags)
{
    return fftwf_plan_guru64_dft_r2c(
        rank, dims, howmany_rank, howmany_dims, reinterpret_cast<float*>(in), out, flags);
}
template <>
inline typename fftw_trait<float>::fftw_plan_type
    fftw_plan_guru64_r2c<float>(int                                            rank,
                                const fftw_iodim64*                            dims,
                                int                                            howmany_rank,
                                const fftw_iodim64*                            howmany_dims,
                                float*                                         in,
                                typename fftw_trait<float>::fftw_complex_type* out,
                                unsigned                                       flags)
{
    return fftwf_plan_guru64_dft_r2c(rank, dims, howmany_rank, howmany_dims, in, out, flags);
}
template <>
inline typename fftw_trait<double>::fftw_plan_type
    fftw_plan_guru64_r2c<double>(int                                             rank,
                                 const fftw_iodim64*                             dims,
                                 int                                             howmany_rank,
                                 const fftw_iodim64*                             howmany_dims,
                                 double*                                         in,
                                 typename fftw_trait<double>::fftw_complex_type* out,
                                 unsigned                                        flags)
{
    return fftw_plan_guru64_dft_r2c(rank, dims, howmany_rank, howmany_dims, in, out, flags);
}

// Template wrappers for FFTW r2c executors:
template <typename Tfloat>
inline void fftw_plan_execute_r2c(typename fftw_trait<Tfloat>::fftw_plan_type plan,
                                  std::vector<hostbuf>&                       in,
                                  std::vector<hostbuf>&                       out);
template <>
inline void fftw_plan_execute_r2c<rocfft_fp16>(typename fftw_trait<float>::fftw_plan_type plan,
                                               std::vector<hostbuf>&                      in,
                                               std::vector<hostbuf>&                      out)
{
    // since FFTW does not natively support half precision, convert
    // input to single, execute, then convert output back to half
    auto in_single = half_to_single_copy(in.front());
    fftwf_execute_dft_r2c(plan,
                          reinterpret_cast<float*>(in_single.data()),
                          reinterpret_cast<fftwf_complex*>(out.front().data()));
    single_to_half_inplace(out.front());
}
template <>
inline void fftw_plan_execute_r2c<float>(typename fftw_trait<float>::fftw_plan_type plan,
                                         std::vector<hostbuf>&                      in,
                                         std::vector<hostbuf>&                      out)
{
    fftwf_execute_dft_r2c(plan,
                          reinterpret_cast<float*>(in.front().data()),
                          reinterpret_cast<fftwf_complex*>(out.front().data()));
}
template <>
inline void fftw_plan_execute_r2c<double>(typename fftw_trait<double>::fftw_plan_type plan,
                                          std::vector<hostbuf>&                       in,
                                          std::vector<hostbuf>&                       out)
{
    fftw_execute_dft_r2c(plan,
                         reinterpret_cast<double*>(in.front().data()),
                         reinterpret_cast<fftw_complex*>(out.front().data()));
}

// Template wrappers for FFTW c2r planners:
template <typename Tfloat>
inline typename fftw_trait<Tfloat>::fftw_plan_type
    fftw_plan_guru64_c2r(int                                             rank,
                         const fftw_iodim64*                             dims,
                         int                                             howmany_rank,
                         const fftw_iodim64*                             howmany_dims,
                         typename fftw_trait<Tfloat>::fftw_complex_type* in,
                         Tfloat*                                         out,
                         unsigned                                        flags);
template <>
inline typename fftw_trait<rocfft_fp16>::fftw_plan_type
    fftw_plan_guru64_c2r<rocfft_fp16>(int                 rank,
                                      const fftw_iodim64* dims,
                                      int                 howmany_rank,
                                      const fftw_iodim64* howmany_dims,
                                      typename fftw_trait<rocfft_fp16>::fftw_complex_type* in,
                                      rocfft_fp16*                                         out,
                                      unsigned                                             flags)
{
    return fftwf_plan_guru64_dft_c2r(
        rank, dims, howmany_rank, howmany_dims, in, reinterpret_cast<float*>(out), flags);
}
template <>
inline typename fftw_trait<float>::fftw_plan_type
    fftw_plan_guru64_c2r<float>(int                                            rank,
                                const fftw_iodim64*                            dims,
                                int                                            howmany_rank,
                                const fftw_iodim64*                            howmany_dims,
                                typename fftw_trait<float>::fftw_complex_type* in,
                                float*                                         out,
                                unsigned                                       flags)
{
    return fftwf_plan_guru64_dft_c2r(rank, dims, howmany_rank, howmany_dims, in, out, flags);
}
template <>
inline typename fftw_trait<double>::fftw_plan_type
    fftw_plan_guru64_c2r<double>(int                                             rank,
                                 const fftw_iodim64*                             dims,
                                 int                                             howmany_rank,
                                 const fftw_iodim64*                             howmany_dims,
                                 typename fftw_trait<double>::fftw_complex_type* in,
                                 double*                                         out,
                                 unsigned                                        flags)
{
    return fftw_plan_guru64_dft_c2r(rank, dims, howmany_rank, howmany_dims, in, out, flags);
}

// Template wrappers for FFTW c2r executors:
template <typename Tfloat>
inline void fftw_plan_execute_c2r(typename fftw_trait<Tfloat>::fftw_plan_type plan,
                                  std::vector<hostbuf>&                       in,
                                  std::vector<hostbuf>&                       out);
template <>
inline void fftw_plan_execute_c2r<rocfft_fp16>(typename fftw_trait<float>::fftw_plan_type plan,
                                               std::vector<hostbuf>&                      in,
                                               std::vector<hostbuf>&                      out)
{
    // since FFTW does not natively support half precision, convert
    // input to single, execute, then convert output back to half
    auto in_single = half_to_single_copy(in.front());
    fftwf_execute_dft_c2r(plan,
                          reinterpret_cast<fftwf_complex*>(in_single.data()),
                          reinterpret_cast<float*>(out.front().data()));
    single_to_half_inplace(out.front());
}
template <>
inline void fftw_plan_execute_c2r<float>(typename fftw_trait<float>::fftw_plan_type plan,
                                         std::vector<hostbuf>&                      in,
                                         std::vector<hostbuf>&                      out)
{
    fftwf_execute_dft_c2r(plan,
                          reinterpret_cast<fftwf_complex*>(in.front().data()),
                          reinterpret_cast<float*>(out.front().data()));
}
template <>
inline void fftw_plan_execute_c2r<double>(typename fftw_trait<double>::fftw_plan_type plan,
                                          std::vector<hostbuf>&                       in,
                                          std::vector<hostbuf>&                       out)
{
    fftw_execute_dft_c2r(plan,
                         reinterpret_cast<fftw_complex*>(in.front().data()),
                         reinterpret_cast<double*>(out.front().data()));
}

#ifdef FFTW_HAVE_SPRINT_PLAN
// Template wrappers for FFTW print plan:
template <typename Tfloat>
inline char* fftw_sprint_plan(const typename fftw_trait<Tfloat>::fftw_plan_type plan);
template <>
inline char*
    fftw_sprint_plan<rocfft_fp16>(const typename fftw_trait<rocfft_fp16>::fftw_plan_type plan)
{
    return fftwf_sprint_plan(plan);
}
template <>
inline char* fftw_sprint_plan<float>(const typename fftw_trait<float>::fftw_plan_type plan)
{
    return fftwf_sprint_plan(plan);
}
template <>
inline char* fftw_sprint_plan<double>(const typename fftw_trait<double>::fftw_plan_type plan)
{
    return fftw_sprint_plan(plan);
}
#endif

#endif
