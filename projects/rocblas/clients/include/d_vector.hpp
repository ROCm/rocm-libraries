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

#if defined(__GLIBC__) && (__GLIBC__ < 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ < 39))
#undef _GLIBCXX_USE_C99_INTTYPES_TR1
#endif
#include <cinttypes>
#include <memory>
#include <mutex>
#include <new>

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
    const size_t m_size;
    const size_t m_pad;
    const size_t m_guard_len;
    const size_t m_bytes;
    bool         m_guard_written;

    static std::once_flag m_init_flag;

public:
    bool use_HMM = false;

    static T m_guard[MEM_MAX_GUARD_PAD];

    d_vector(const d_vector&) = delete;
    d_vector& operator=(const d_vector&) = delete;
    d_vector(d_vector&&)                 = delete;
    d_vector& operator=(d_vector&&) = delete;

    inline size_t nmemb() const noexcept
    {
        return m_size;
    }

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
        , m_guard_written(false)
        , use_HMM(HMM)
    {
        // Filled on first construction whatever the pad currently is. Keying this off
        // m_pad would leave m_guard zero for a type whose first d_vector happened to be
        // built while the pad was zero, and the pad is now a run-time setting, so a later
        // guarded allocation of that same type would compare against zeros. The fill is
        // bounded and happens once per type, even if multiple d_vector<T> objects are
        // constructed concurrently.
        std::call_once(m_init_flag, [] { rocblas_init_nan(m_guard, MEM_MAX_GUARD_PAD); });
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

        if((use_HMM ? hipMallocManaged(&d, m_bytes) : (hipMalloc)(&d, m_bytes)) != hipSuccess)
        {
            rocblas_cerr << "Warning: hip can't allocate " << m_bytes << " bytes ("
                         << (m_bytes >> 30) << " GB)" << std::endl;

            d = nullptr;
        }
        else if(m_guard_len > 0)
        {
            hipError_t status = hipMemcpy(d, m_guard, m_guard_len, hipMemcpyDefault);
            if(status != hipSuccess)
                d_vector_report_failure(std::string("cannot write the guard before the allocation: ")
                                        + hipGetErrorName(status));

            // Point to allocated block
            d += m_pad;

            if(status == hipSuccess)
            {
                status = hipMemcpy(d + m_size, m_guard, m_guard_len, hipMemcpyDefault);
                if(status != hipSuccess)
                    d_vector_report_failure(std::string("cannot write the guard after the allocation: ")
                                            + hipGetErrorName(status));
            }

            m_guard_written = (status == hipSuccess);
        }

        if(use_HMM)
            alloc_ptr_use(d, m_bytes);

        return d;
    }

    // Reads both guard regions back from the device and reports any mismatch through
    // d_vector_report_failure. Called from device_vector_teardown, so it must not rely on
    // destructor-unsafe fatal assertions.
    void device_vector_check(T* d)
    {
        if(!m_guard_written)
            return;

        if(m_pad > 0)
        {
            std::unique_ptr<unsigned char[]> host_guard(new(std::nothrow) unsigned char[m_guard_len]);
            if(!host_guard)
            {
                d_vector_report_failure("cannot allocate " + std::to_string(m_guard_len)
                                        + " bytes to read the guards back; corruption would go unreported");
                return;
            }

            // Copy device memory after allocated memory to host
            const auto* reference = reinterpret_cast<const unsigned char*>(m_guard);
            hipError_t   status
                = hipMemcpy(host_guard.get(), d + m_size, m_guard_len, hipMemcpyDefault);
            if(status != hipSuccess)
                d_vector_report_failure(std::string("cannot read the guard after the allocation: ")
                                        + hipGetErrorName(status));

            // Make sure no corruption has occurred
            if(status == hipSuccess && memcmp(host_guard.get(), reference, m_guard_len) != 0)
                d_vector_report_failure("post-guard overwritten");

            // Point to m_guard before allocated memory
            d -= m_pad;

            // Copy device memory after allocated memory to host
            status = hipMemcpy(host_guard.get(), d, m_guard_len, hipMemcpyDefault);
            if(status != hipSuccess)
                d_vector_report_failure(std::string("cannot read the guard before the allocation: ")
                                        + hipGetErrorName(status));

            // Make sure no corruption has occurred
            if(status == hipSuccess && memcmp(host_guard.get(), reference, m_guard_len) != 0)
                d_vector_report_failure("pre-guard overwritten");
        }
    }

    void device_vector_teardown(T* d)
    {
        if(d != nullptr)
        {
            device_vector_check(d);

            if(use_HMM)
                free_ptr_use(d);

            if(m_pad > 0)
                d -= m_pad; // restore to start of alloc

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
std::once_flag d_vector<T>::m_init_flag;

#undef MEM_MAX_GUARD_PAD
