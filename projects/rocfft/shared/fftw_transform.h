// Copyright (C) 2016 - 2026 Advanced Micro Devices, Inc. All rights reserved.
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

// C++ traits to simplify type selection for FFTW plans, complex and/or real
// FFTW types, based on the base real type of the floating-point arithmetic of
// interest. The correct FFTW complex type can be accessed via, for example,
// using complex_t = typename fftw_complex_trait<Tfloat>::complex_t;
// NOTE: FFTW does not support half-precision: members of fftw_trait<rocfft_fp16>
// are made equivalent to the members of fftw_trait<float>.
template <typename Tfloat,
          std::enable_if_t<std::is_same<Tfloat, double>::value || std::is_same<Tfloat, float>::value
                               || std::is_same<Tfloat, rocfft_fp16>::value,
                           bool> = true>
struct fftw_trait
{

private:
    using raw_fftw_plan_type = std::conditional_t<std::is_same_v<Tfloat, double>,
                                                  std::remove_pointer<fftw_plan>::type,
                                                  std::remove_pointer<fftwf_plan>::type>;
    static void plan_deleter(raw_fftw_plan_type* plan_ptr)
    {
        static_assert(
            std::is_same<raw_fftw_plan_type, std::remove_pointer<fftw_plan>::type>::value
            || std::is_same<raw_fftw_plan_type, std::remove_pointer<fftwf_plan>::type>::value);
        if constexpr(std::is_same<raw_fftw_plan_type, std::remove_pointer<fftw_plan>::type>::value)
            fftw_destroy_plan(plan_ptr);
        else
            fftwf_destroy_plan(plan_ptr);
    }

public:
    using real_t = std::conditional_t<std::is_same<Tfloat, double>::value, double, float>;
    using complex_t
        = std::conditional_t<std::is_same<Tfloat, double>::value, fftw_complex, fftwf_complex>;
    using plan_wrapper_t = std::unique_ptr<raw_fftw_plan_type, decltype(&plan_deleter)>;

    static plan_wrapper_t make_wrapper(raw_fftw_plan_type* raw_plan_ptr)
    {
        return plan_wrapper_t{raw_plan_ptr, plan_deleter};
    }
};
// External alias templates to avoid verbose syntax thereafter
template <typename T>
using fftw_plan_wrapper_t = typename fftw_trait<T>::plan_wrapper_t;
template <typename T>
using fftw_complex_t = typename fftw_trait<T>::complex_t;
template <typename T>
using fftw_real_t = typename fftw_trait<T>::real_t;

// Copies the half-precision input buffer to a single-precision buffer.
static hostbuf half_to_single_copy(const hostbuf& in)
{
    hostbuf out;
    out.alloc(2 * in.size());
    auto in_begin = reinterpret_cast<const rocfft_fp16*>(in.data());
    std::copy_n(in_begin, in.size() / sizeof(rocfft_fp16), reinterpret_cast<float*>(out.data()));
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
inline fftw_complex_t<Tfloat>* fftw_alloc_complex_type(size_t n);
template <>
inline fftw_complex_t<float>* fftw_alloc_complex_type<float>(size_t n)
{
    return fftwf_alloc_complex(n);
}
template <>
inline fftw_complex_t<double>* fftw_alloc_complex_type<double>(size_t n)
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
inline void fftw_execute_type(const fftw_plan_wrapper_t<Tfloat>& plan);
template <>
inline void fftw_execute_type<float>(const fftw_plan_wrapper_t<float>& plan)
{
    return fftwf_execute(plan.get());
}
template <>
inline void fftw_execute_type<double>(const fftw_plan_wrapper_t<double>& plan)
{
    return fftw_execute(plan.get());
}

// Template wrappers for FFTW c2c planners:
template <typename Tfloat>
inline fftw_plan_wrapper_t<Tfloat> fftw_plan_guru64_dft(int                     rank,
                                                        const fftw_iodim64*     dims,
                                                        int                     howmany_rank,
                                                        const fftw_iodim64*     howmany_dims,
                                                        fftw_complex_t<Tfloat>* in,
                                                        fftw_complex_t<Tfloat>* out,
                                                        int                     sign,
                                                        unsigned                flags);

template <>
inline fftw_plan_wrapper_t<rocfft_fp16>
    fftw_plan_guru64_dft<rocfft_fp16>(int                          rank,
                                      const fftw_iodim64*          dims,
                                      int                          howmany_rank,
                                      const fftw_iodim64*          howmany_dims,
                                      fftw_complex_t<rocfft_fp16>* in,
                                      fftw_complex_t<rocfft_fp16>* out,
                                      int                          sign,
                                      unsigned                     flags)
{
    return fftw_trait<rocfft_fp16>::make_wrapper(
        fftwf_plan_guru64_dft(rank, dims, howmany_rank, howmany_dims, in, out, sign, flags));
}

template <>
inline fftw_plan_wrapper_t<float> fftw_plan_guru64_dft<float>(int                    rank,
                                                              const fftw_iodim64*    dims,
                                                              int                    howmany_rank,
                                                              const fftw_iodim64*    howmany_dims,
                                                              fftw_complex_t<float>* in,
                                                              fftw_complex_t<float>* out,
                                                              int                    sign,
                                                              unsigned               flags)
{
    return fftw_trait<float>::make_wrapper(
        fftwf_plan_guru64_dft(rank, dims, howmany_rank, howmany_dims, in, out, sign, flags));
}

template <>
inline fftw_plan_wrapper_t<double> fftw_plan_guru64_dft<double>(int                 rank,
                                                                const fftw_iodim64* dims,
                                                                int                 howmany_rank,
                                                                const fftw_iodim64* howmany_dims,
                                                                fftw_complex_t<double>* in,
                                                                fftw_complex_t<double>* out,
                                                                int                     sign,
                                                                unsigned                flags)
{
    return fftw_trait<double>::make_wrapper(
        fftw_plan_guru64_dft(rank, dims, howmany_rank, howmany_dims, in, out, sign, flags));
}

// Template wrappers for FFTW c2c executors:
template <typename Tfloat>
inline void fftw_plan_execute_c2c(const fftw_plan_wrapper_t<Tfloat>& plan,
                                  std::vector<hostbuf>&              in,
                                  std::vector<hostbuf>&              out);
template <>
inline void fftw_plan_execute_c2c<rocfft_fp16>(const fftw_plan_wrapper_t<rocfft_fp16>& plan,
                                               std::vector<hostbuf>&                   in,
                                               std::vector<hostbuf>&                   out)
{
    // since FFTW does not natively support half precision, convert
    // input to single, execute, then convert output back to half
    auto in_single = half_to_single_copy(in.front());
    fftwf_execute_dft(plan.get(),
                      reinterpret_cast<fftwf_complex*>(in_single.data()),
                      reinterpret_cast<fftwf_complex*>(out.front().data()));
    single_to_half_inplace(out.front());
}

template <>
inline void fftw_plan_execute_c2c<float>(const fftw_plan_wrapper_t<float>& plan,
                                         std::vector<hostbuf>&             in,
                                         std::vector<hostbuf>&             out)
{
    fftwf_execute_dft(plan.get(),
                      reinterpret_cast<fftwf_complex*>(in.front().data()),
                      reinterpret_cast<fftwf_complex*>(out.front().data()));
}

template <>
inline void fftw_plan_execute_c2c<double>(const fftw_plan_wrapper_t<double>& plan,
                                          std::vector<hostbuf>&              in,
                                          std::vector<hostbuf>&              out)
{
    fftw_execute_dft(plan.get(),
                     reinterpret_cast<fftw_complex*>(in.front().data()),
                     reinterpret_cast<fftw_complex*>(out.front().data()));
}

// Template wrappers for FFTW r2c planners:
template <typename Tfloat>
inline fftw_plan_wrapper_t<Tfloat> fftw_plan_guru64_r2c(int                     rank,
                                                        const fftw_iodim64*     dims,
                                                        int                     howmany_rank,
                                                        const fftw_iodim64*     howmany_dims,
                                                        Tfloat*                 in,
                                                        fftw_complex_t<Tfloat>* out,
                                                        unsigned                flags);
template <>
inline fftw_plan_wrapper_t<rocfft_fp16>
    fftw_plan_guru64_r2c<rocfft_fp16>(int                          rank,
                                      const fftw_iodim64*          dims,
                                      int                          howmany_rank,
                                      const fftw_iodim64*          howmany_dims,
                                      rocfft_fp16*                 in,
                                      fftw_complex_t<rocfft_fp16>* out,
                                      unsigned                     flags)
{
    return fftw_trait<rocfft_fp16>::make_wrapper(fftwf_plan_guru64_dft_r2c(
        rank, dims, howmany_rank, howmany_dims, reinterpret_cast<float*>(in), out, flags));
}
template <>
inline fftw_plan_wrapper_t<float> fftw_plan_guru64_r2c<float>(int                    rank,
                                                              const fftw_iodim64*    dims,
                                                              int                    howmany_rank,
                                                              const fftw_iodim64*    howmany_dims,
                                                              float*                 in,
                                                              fftw_complex_t<float>* out,
                                                              unsigned               flags)
{
    return fftw_trait<float>::make_wrapper(
        fftwf_plan_guru64_dft_r2c(rank, dims, howmany_rank, howmany_dims, in, out, flags));
}
template <>
inline fftw_plan_wrapper_t<double> fftw_plan_guru64_r2c<double>(int                 rank,
                                                                const fftw_iodim64* dims,
                                                                int                 howmany_rank,
                                                                const fftw_iodim64* howmany_dims,
                                                                double*             in,
                                                                fftw_complex_t<double>* out,
                                                                unsigned                flags)
{
    return fftw_trait<double>::make_wrapper(
        fftw_plan_guru64_dft_r2c(rank, dims, howmany_rank, howmany_dims, in, out, flags));
}

// Template wrappers for FFTW r2c executors:
template <typename Tfloat>
inline void fftw_plan_execute_r2c(const fftw_plan_wrapper_t<Tfloat>& plan,
                                  std::vector<hostbuf>&              in,
                                  std::vector<hostbuf>&              out);
template <>
inline void fftw_plan_execute_r2c<rocfft_fp16>(const fftw_plan_wrapper_t<rocfft_fp16>& plan,
                                               std::vector<hostbuf>&                   in,
                                               std::vector<hostbuf>&                   out)
{
    // since FFTW does not natively support half precision, convert
    // input to single, execute, then convert output back to half
    auto in_single = half_to_single_copy(in.front());
    fftwf_execute_dft_r2c(plan.get(),
                          reinterpret_cast<float*>(in_single.data()),
                          reinterpret_cast<fftwf_complex*>(out.front().data()));
    single_to_half_inplace(out.front());
}
template <>
inline void fftw_plan_execute_r2c<float>(const fftw_plan_wrapper_t<float>& plan,
                                         std::vector<hostbuf>&             in,
                                         std::vector<hostbuf>&             out)
{
    fftwf_execute_dft_r2c(plan.get(),
                          reinterpret_cast<float*>(in.front().data()),
                          reinterpret_cast<fftwf_complex*>(out.front().data()));
}
template <>
inline void fftw_plan_execute_r2c<double>(const fftw_plan_wrapper_t<double>& plan,
                                          std::vector<hostbuf>&              in,
                                          std::vector<hostbuf>&              out)
{
    fftw_execute_dft_r2c(plan.get(),
                         reinterpret_cast<double*>(in.front().data()),
                         reinterpret_cast<fftw_complex*>(out.front().data()));
}

// Template wrappers for FFTW c2r planners:
template <typename Tfloat>
inline fftw_plan_wrapper_t<Tfloat> fftw_plan_guru64_c2r(int                     rank,
                                                        const fftw_iodim64*     dims,
                                                        int                     howmany_rank,
                                                        const fftw_iodim64*     howmany_dims,
                                                        fftw_complex_t<Tfloat>* in,
                                                        Tfloat*                 out,
                                                        unsigned                flags);
template <>
inline fftw_plan_wrapper_t<rocfft_fp16>
    fftw_plan_guru64_c2r<rocfft_fp16>(int                          rank,
                                      const fftw_iodim64*          dims,
                                      int                          howmany_rank,
                                      const fftw_iodim64*          howmany_dims,
                                      fftw_complex_t<rocfft_fp16>* in,
                                      rocfft_fp16*                 out,
                                      unsigned                     flags)
{
    return fftw_trait<rocfft_fp16>::make_wrapper(fftwf_plan_guru64_dft_c2r(
        rank, dims, howmany_rank, howmany_dims, in, reinterpret_cast<float*>(out), flags));
}
template <>
inline fftw_plan_wrapper_t<float> fftw_plan_guru64_c2r<float>(int                    rank,
                                                              const fftw_iodim64*    dims,
                                                              int                    howmany_rank,
                                                              const fftw_iodim64*    howmany_dims,
                                                              fftw_complex_t<float>* in,
                                                              float*                 out,
                                                              unsigned               flags)
{
    return fftw_trait<float>::make_wrapper(
        fftwf_plan_guru64_dft_c2r(rank, dims, howmany_rank, howmany_dims, in, out, flags));
}
template <>
inline fftw_plan_wrapper_t<double> fftw_plan_guru64_c2r<double>(int                 rank,
                                                                const fftw_iodim64* dims,
                                                                int                 howmany_rank,
                                                                const fftw_iodim64* howmany_dims,
                                                                fftw_complex_t<double>* in,
                                                                double*                 out,
                                                                unsigned                flags)
{
    return fftw_trait<double>::make_wrapper(
        fftw_plan_guru64_dft_c2r(rank, dims, howmany_rank, howmany_dims, in, out, flags));
}

// Template wrappers for FFTW c2r executors:
template <typename Tfloat>
inline void fftw_plan_execute_c2r(const fftw_plan_wrapper_t<Tfloat>& plan,
                                  std::vector<hostbuf>&              in,
                                  std::vector<hostbuf>&              out);
template <>
inline void fftw_plan_execute_c2r<rocfft_fp16>(const fftw_plan_wrapper_t<rocfft_fp16>& plan,
                                               std::vector<hostbuf>&                   in,
                                               std::vector<hostbuf>&                   out)
{
    // since FFTW does not natively support half precision, convert
    // input to single, execute, then convert output back to half
    auto in_single = half_to_single_copy(in.front());
    fftwf_execute_dft_c2r(plan.get(),
                          reinterpret_cast<fftwf_complex*>(in_single.data()),
                          reinterpret_cast<float*>(out.front().data()));
    single_to_half_inplace(out.front());
}
template <>
inline void fftw_plan_execute_c2r<float>(const fftw_plan_wrapper_t<float>& plan,
                                         std::vector<hostbuf>&             in,
                                         std::vector<hostbuf>&             out)
{
    fftwf_execute_dft_c2r(plan.get(),
                          reinterpret_cast<fftwf_complex*>(in.front().data()),
                          reinterpret_cast<float*>(out.front().data()));
}
template <>
inline void fftw_plan_execute_c2r<double>(const fftw_plan_wrapper_t<double>& plan,
                                          std::vector<hostbuf>&              in,
                                          std::vector<hostbuf>&              out)
{
    fftw_execute_dft_c2r(plan.get(),
                         reinterpret_cast<fftw_complex*>(in.front().data()),
                         reinterpret_cast<double*>(out.front().data()));
}

#ifdef FFTW_HAVE_SPRINT_PLAN
// Template wrappers for FFTW print plan:
template <typename Tfloat>
inline char* fftw_sprint_plan(const fftw_plan_wrapper_t<Tfloat>& plan);
template <>
inline char* fftw_sprint_plan<rocfft_fp16>(const fftw_plan_wrapper_t<rocfft_fp16>& plan)
{
    return fftwf_sprint_plan(plan.get());
}
template <>
inline char* fftw_sprint_plan<float>(const fftw_plan_wrapper_t<float>& plan)
{
    return fftwf_sprint_plan(plan.get());
}
template <>
inline char* fftw_sprint_plan<double>(const fftw_plan_wrapper_t<double>& plan)
{
    return fftw_sprint_plan(plan.get());
}
#endif

#endif
