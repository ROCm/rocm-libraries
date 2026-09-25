/* ************************************************************************
 * Copyright (C) 2018-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

#include "host_alloc.hpp"
#include "rocblas.h"
#include "rocblas_test.hpp"
#include "singletons.hpp"

#if __GLIBC__ < 3 && __GLIBC__MINOR__ < 39
#undef _GLIBCXX_USE_C99_INTTYPES_TR1
#endif
#include <cinttypes>

#define MEM_MAX_GUARD_PAD 8192

//
// Forward declaration of rocblas_init_nan
//
template <typename T>
void rocblas_init_nan(T* A, size_t N);

template <typename T>
inline rocblas_stride align_stride(rocblas_stride stride)
{
    // hipMalloc aligns pointers on 256 byte boundaries (or a multiple of 256)
    // this function is to align stride*sizeof(T) on 256 byte boundaries
    size_t byte_alignment = 256;

    if(byte_alignment % sizeof(T) == 0)
    {
        size_t type_alignment = byte_alignment / sizeof(T);
        return ((stride - 1) / type_alignment + 1) * type_alignment;
    }
    else
    {
        return ((stride - 1) / byte_alignment + 1) * byte_alignment;
    }
}

/* ============================================================================================ */
/*! \brief  base-class to allocate/deallocate device memory */
template <typename T>
class d_vector
{
private:
    size_t m_size;
    size_t m_pad, m_guard_len;
    size_t m_bytes;

    static bool m_init_guard;

public:
    inline size_t nmemb() const noexcept
    {
        return m_size;
    }

public:
    bool use_HMM = false;

public:
    static T m_guard[MEM_MAX_GUARD_PAD];

    // One constructor for every configuration. Whether the guard regions exist is decided
    // by g_DVEC_PAD, a single global defined in singletons.cpp, and not by GOOGLE_TEST.
    //
    // This is not a stylistic preference. Every member of this class template has the same
    // mangled name however the translation unit was compiled, so a body that depends on
    // GOOGLE_TEST gives one symbol two definitions, and the linker keeps whichever it sees
    // first. rocblas-gemm-tune compiles its own sources without GOOGLE_TEST and links the
    // common client libraries, which are built with it, so both definitions really do meet
    // in one binary. A program that wants no guards asks for none at run time instead.
    d_vector(size_t s, bool HMM = false)
        : m_size(s)
        , m_pad(std::min(g_DVEC_PAD, size_t(MEM_MAX_GUARD_PAD)))
        , m_guard_len(m_pad * sizeof(T))
        // Never zero: hipMalloc(0) hands back a null pointer, which every caller reads as
        // a failed allocation. Only an empty vector with no pad reaches the clamp; any pad
        // at all already makes the allocation non-empty.
        , m_bytes(std::max(s + m_pad * 2, size_t(1)) * sizeof(T))
        , use_HMM(HMM)
    {
        // Filled on first construction whatever the pad currently is. Keying this off
        // m_pad would leave m_guard zero for a type whose first d_vector happened to be
        // built while the pad was zero, and the pad is now a run-time setting, so a later
        // guarded allocation of that same type would compare against zeros. The fill is
        // bounded and happens once per type.
        if(!m_init_guard)
        {
            // Initialize m_guard with random data
            rocblas_init_nan(m_guard, MEM_MAX_GUARD_PAD);
            m_init_guard = true;
        }
    }

    T* device_vector_setup()
    {
        T* d = nullptr;

        if(use_HMM)
        {
            if(!host_mem_safe(m_bytes))
            {
                return nullptr; // caller decides on throwing exception
            }
        }

        if(use_HMM ? hipMallocManaged(&d, m_bytes) : (hipMalloc)(&d, m_bytes) != hipSuccess)
        {
            rocblas_cerr << "Warning: hip can't allocate " << m_bytes << " bytes ("
                         << (m_bytes >> 30) << " GB)" << std::endl;

            d = nullptr;
        }
        else if(m_guard_len > 0)
        {
            // Copy m_guard to device memory before allocated memory
            if(hipMemcpy(d, m_guard, m_guard_len, hipMemcpyDefault) != hipSuccess)
                rocblas_cerr << "Error: hipMemcpy pre-guard copy failure." << std::endl;

            // Point to allocated block
            d += m_pad;

            // Copy m_guard to device memory after allocated memory
            if(hipMemcpy(d + m_size, m_guard, m_guard_len, hipMemcpyDefault) != hipSuccess)
                rocblas_cerr << "Error: hipMemcpy post-guard copy failure." << std::endl;
        }

        if(use_HMM)
            alloc_ptr_use(d, m_bytes); // count the same as host memory

        return d;
    }

    void device_vector_check(T* d)
    {
        if(m_pad > 0)
        {
            std::vector<T> host(m_pad);

            // Copy device memory after allocated memory to host
            if(hipMemcpy(host.data(), d + m_size, m_guard_len, hipMemcpyDefault) != hipSuccess)
                rocblas_cerr << "Error: hipMemcpy post-guard copy failure." << std::endl;

            // Make sure no corruption has occurred
            if(memcmp(host.data(), m_guard, m_guard_len) != 0)
                d_vector_report_failure("post-guard overwritten");

            // Point to m_guard before allocated memory
            d -= m_pad;

            // Copy device memory after allocated memory to host
            if(hipMemcpy(host.data(), d, m_guard_len, hipMemcpyDefault) != hipSuccess)
                rocblas_cerr << "Error: hipMemcpy pre-guard copy failure." << std::endl;

            // Make sure no corruption has occurred
            if(memcmp(host.data(), m_guard, m_guard_len) != 0)
                d_vector_report_failure("pre-guard overwritten");
        }
    }

    void device_vector_teardown(T* d)
    {
        if(d != nullptr)
        {
            device_vector_check(d);

            if(m_pad > 0)
                d -= m_pad; // restore to start of alloc

            if(use_HMM)
                free_ptr_use(d); // release count

            // Free device memory. Reported rather than asserted: CHECK_HIP_ERROR expands to
            // a Google Test assertion only under GOOGLE_TEST, which would make this body
            // another definition that depends on the macro. A destructor is also the wrong
            // place for a fatal assertion.
            hipError_t status = (hipFree)(d);
            if(status != hipSuccess)
                d_vector_report_failure(std::string("cannot free the device allocation: ")
                                        + hipGetErrorName(status));
        }
    }
};

template <typename T>
T d_vector<T>::m_guard[MEM_MAX_GUARD_PAD] = {};

template <typename T>
bool d_vector<T>::m_init_guard = false;

#undef MEM_MAX_GUARD_PAD
