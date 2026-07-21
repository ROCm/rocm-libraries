/* **************************************************************************
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

ROCSOLVER_BEGIN_NAMESPACE

/*
 * ===========================================================================
 *    common location for functions that are used across several rocSOLVER
 *    routines, excepting device functions and kernels (see
 *    lib_device_helpers.hpp and lapack_device_functions.hpp).
 * ===========================================================================
 */

__device__ __host__ inline int64_t idx2D(const int64_t i, const int64_t j, const int64_t lda)
{
    return j * lda + i;
}

__device__ __host__ inline int64_t
    idx2D(const int64_t i, const int64_t j, const int64_t inca, const int64_t lda)
{
    return j * lda + i * inca;
}

__device__ __host__ inline int32_t ceil(const int32_t n, const int32_t nb)
{
    return ((n - 1) / nb) + 1;
}

__device__ __host__ inline int64_t ceil(const int64_t n, const int64_t nb)
{
    return ((n - 1) / nb) + 1;
}

template <typename T>
T const* cast2constType(T* array)
{
    return array;
}

template <typename T>
T const* const* cast2constType(T* const* array)
{
    return array;
}

template <typename T>
T* cast2constPointer(T* array)
{
    return array;
}

template <typename T>
T* const* cast2constPointer(T** array)
{
    return array;
}

template <typename I>
I get_index(I* intervals, I max, I dim)
{
    I i;

    for(i = 0; i < max; ++i)
    {
        if(dim <= intervals[i])
            break;
    }

    return i;
}

template <typename T, typename I>
static void call_swap(I& n, T& x_in, I& incx, T& y_in, I& incy)
{
    T* const x = &(x_in);
    T* const y = &(y_in);
    for(I i = 0; i < n; i++)
    {
        auto const ix = i * static_cast<int64_t>(incx);
        auto const iy = i * static_cast<int64_t>(incy);

        T const temp = x[ix];
        x[ix] = y[iy];
        y[iy] = temp;
    }
}

static float real_part(float z)
{
    return (z);
}
static float real_part(std::complex<float> z)
{
    return (z.real());
}
static float real_part(rocblas_complex_num<float> z)
{
    return (z.real());
}

static double real_part(double z)
{
    return (z);
}
static double real_part(std::complex<double> z)
{
    return (z.real());
}
static double real_part(rocblas_complex_num<double> z)
{
    return (z.real());
}

static float imag_part(float z)
{
    return (0);
}
static float imag_part(std::complex<float> z)
{
    return (z.imag());
}
static float imag_part(rocblas_complex_num<float> z)
{
    return (z.imag());
}

static double imag_part(double z)
{
    return (0);
}
static double imag_part(std::complex<double> z)
{
    return (z.imag());
}
static double imag_part(rocblas_complex_num<double> z)
{
    return (z.imag());
}

static bool is_device_pointer(void* ptr)
{
    hipPointerAttribute_t dev_attributes;
    if(ptr == nullptr)
        return false;

    auto istat = hipPointerGetAttributes(&dev_attributes, ptr);
    if(istat != hipSuccess)
        fmt::print(stderr, "is_device_pointer: istat = {} {}\n", static_cast<std::int32_t>(istat),
                   hipGetErrorName(istat));

    assert(istat == hipSuccess);
    return (dev_attributes.type == hipMemoryTypeDevice);
}

#ifdef ROCSOLVER_VERIFY_ASSUMPTIONS
// Ensure __assert_fail is declared.
#if !__is_identifier(__assert_fail)
extern "C" [[noreturn]] void __assert_fail(const char* assertion,
                                           const char* file,
                                           unsigned int line,
                                           const char* function) noexcept;
#endif
// ROCSOLVER_FAIL(msg) is called with a string literal to print a message and abort the program.
// By default, it calls __assert_fail, but can be defined to something else.
#ifndef ROCSOLVER_FAIL
#define ROCSOLVER_FAIL(msg) __assert_fail(msg, __FILE__, __LINE__, __PRETTY_FUNCTION__)
#endif
#endif

// ROCSOLVER_UNREACHABLE is an alternative to __builtin_unreachable that verifies that the path is
// actually unreachable if ROCSOLVER_VERIFY_ASSUMPTIONS is defined.
#ifdef ROCSOLVER_VERIFY_ASSUMPTIONS
#define ROCSOLVER_UNREACHABLE() ROCSOLVER_FAIL("unreachable statement")
#else
#define ROCSOLVER_UNREACHABLE() __builtin_unreachable()
#endif

// ROCSOLVER_UNREACHABLE_X is a variant of ROCSOLVER_UNREACHABLE that takes a string as a parameter,
// which should explain why this path is believed to be unreachable.
#ifdef ROCSOLVER_VERIFY_ASSUMPTIONS
#define ROCSOLVER_UNREACHABLE_X(msg) ROCSOLVER_FAIL("unreachable statement (assumed " msg ")")
#else
#define ROCSOLVER_UNREACHABLE_X(msg) __builtin_unreachable()
#endif

// ROCSOLVER_ASSUME is an alternative to __builtin_assume that verifies that the assumption is
// actually true if ROCSOLVER_VERIFY_ASSUMPTIONS is defined.
#ifdef ROCSOLVER_VERIFY_ASSUMPTIONS
#define ROCSOLVER_ASSUME(invariant)     \
    do                                  \
    {                                   \
        if(!(invariant))                \
        {                               \
            ROCSOLVER_FAIL(#invariant); \
        }                               \
    } while(0)
#else
#define ROCSOLVER_ASSUME(invariant) __builtin_assume(invariant)
#endif

// ROCSOLVER_ASSUME_X is a variant of ROCSOLVER_ASSUME that takes a string as a second parameter,
// which should explain why this invariant is believed to be guaranteed.
#ifdef ROCSOLVER_VERIFY_ASSUMPTIONS
#define ROCSOLVER_ASSUME_X(invariant, msg)                   \
    do                                                       \
    {                                                        \
        if(!(invariant))                                     \
        {                                                    \
            ROCSOLVER_FAIL(#invariant " (assumed " msg ")"); \
        }                                                    \
    } while(0)
#else
#define ROCSOLVER_ASSUME_X(invariant, msg) __builtin_assume(invariant)
#endif

#ifndef CHECK_HIP
#define CHECK_HIP(fcn)                  \
    {                                   \
        hipError_t const istat = (fcn); \
        assert(istat == hipSuccess);    \
    }
#endif

ROCSOLVER_END_NAMESPACE
