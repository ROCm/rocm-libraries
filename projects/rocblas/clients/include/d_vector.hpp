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

// Older than glibc 2.39. Spelled out rather than "major < 3 && minor < 39", which only
// happens to mean the same thing while glibc stays on major 2. Keep in step with the same
// test in rocblas_random.hpp.
#if defined(__GLIBC__) && (__GLIBC__ < 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ < 39))
#undef _GLIBCXX_USE_C99_INTTYPES_TR1
#endif
#include <algorithm>
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
    // Geometry, fixed at construction. const so the offsets that setup applies, check
    // reads and teardown undoes cannot drift apart; all three derive them from m_pad.
    const size_t m_size;
    const size_t m_pad;
    const size_t m_guard_len; // m_pad * sizeof(T)
    const size_t m_bytes;

    // Set only when both guard writes succeeded, so a failed write disables the check
    // rather than reporting corruption against uninitialised memory.
    bool m_guard_written;

    // Guards the one-time fill of m_guard against concurrent construction.
    static std::once_flag m_init_flag;

public:
    bool use_HMM = false;

    static T m_guard[MEM_MAX_GUARD_PAD];

    // Subclasses own a raw device pointer: copying or moving would duplicate it without
    // transferring ownership, and both objects would free it.
    d_vector(const d_vector&) = delete;
    d_vector& operator=(const d_vector&) = delete;
    d_vector(d_vector&&)                 = delete;
    d_vector& operator=(d_vector&&) = delete;

    inline size_t nmemb() const noexcept
    {
        return m_size;
    }

    // One constructor for every configuration: the guards are switched on by g_DVEC_PAD at
    // run time, never by GOOGLE_TEST. See singletons.hpp for why this header must not test
    // that macro.
    d_vector(size_t s, bool HMM = false)
        : m_size(s)
        , m_pad(std::min(g_DVEC_PAD, size_t(MEM_MAX_GUARD_PAD)))
        , m_guard_len(m_pad * sizeof(T))
        // Never zero: hipMalloc(0) returns a null pointer, which callers read as failure.
        , m_bytes(std::max(s + m_pad * 2, size_t(1)) * sizeof(T))
        , m_guard_written(false)
        , use_HMM(HMM)
    {
        // Once per type, whatever the pad is now: keying this off m_pad would leave m_guard
        // zeroed for a type first constructed unguarded, and the pad can change at run time.
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
        else if(m_pad > 0)
        {
            hipError_t status = hipMemcpy(d, m_guard, m_guard_len, hipMemcpyDefault);
            if(status != hipSuccess)
                d_vector_report_failure(
                    std::string("cannot write the guard before the allocation: ")
                    + hipGetErrorName(status));

            // Offset past the pre-guard unconditionally, so d is the pointer teardown will
            // hand to free_ptr_use whether or not the guard writes succeeded.
            d += m_pad;

            if(status == hipSuccess)
            {
                status = hipMemcpy(d + m_size, m_guard, m_guard_len, hipMemcpyDefault);
                if(status != hipSuccess)
                    d_vector_report_failure(
                        std::string("cannot write the guard after the allocation: ")
                        + hipGetErrorName(status));
            }

            // Checked later only if both regions hold the pattern, so a failed write does
            // not turn into a corruption report against uninitialised memory.
            m_guard_written = (status == hipSuccess);
        }

        if(use_HMM)
            alloc_ptr_use(d, m_bytes);

        return d;
    }

    // Reads both guard regions back and reports any mismatch. Runs from a destructor, so
    // nothing here may be a fatal assertion.
    void device_vector_check(T* d)
    {
        if(!m_guard_written)
            return;

        // One buffer, exactly one guard long, reused for both regions. nothrow new because
        // a throwing allocation in a destructor calls std::terminate; no alignment needed
        // because hipMemcpy and memcmp are byte-wise.
        std::unique_ptr<unsigned char[]> host_guard(new(std::nothrow) unsigned char[m_guard_len]);
        if(!host_guard)
        {
            d_vector_report_failure(
                "cannot allocate " + std::to_string(m_guard_len)
                + " bytes to read the guards back; corruption would go unreported");
            return;
        }

        const auto* reference = reinterpret_cast<const unsigned char*>(m_guard);

        // Post-guard first, because d still points at the user allocation. Each comparison
        // is gated on its own copy, so a failed read cannot leave the other region's bytes
        // behind to be compared twice.
        hipError_t status = hipMemcpy(host_guard.get(), d + m_size, m_guard_len, hipMemcpyDefault);
        if(status != hipSuccess)
            d_vector_report_failure(std::string("cannot read the guard after the allocation: ")
                                    + hipGetErrorName(status));
        if(status == hipSuccess && memcmp(host_guard.get(), reference, m_guard_len) != 0)
            d_vector_report_guard_corruption(host_guard.get(), reference, m_guard_len, "post");

        // Pre-guard sits m_pad elements below the user pointer.
        status = hipMemcpy(host_guard.get(), d - m_pad, m_guard_len, hipMemcpyDefault);
        if(status != hipSuccess)
            d_vector_report_failure(std::string("cannot read the guard before the allocation: ")
                                    + hipGetErrorName(status));
        if(status == hipSuccess && memcmp(host_guard.get(), reference, m_guard_len) != 0)
            d_vector_report_guard_corruption(host_guard.get(), reference, m_guard_len, "pre");
    }

    // Checks the guards, releases the HMM accounting entry, then frees. No-op on nullptr.
    void device_vector_teardown(T* d)
    {
        if(d != nullptr)
        {
            device_vector_check(d);

            // Released on the pointer setup counted, which is the offset one. Releasing
            // after restoring the base misses in the tracker's map, and a miss is silent.
            if(use_HMM)
                free_ptr_use(d);

            if(m_pad > 0)
                d -= m_pad; // restore to start of alloc

            // Reported, not asserted: CHECK_HIP_ERROR varies with GOOGLE_TEST, which this
            // header may not depend on, and a destructor is the wrong place to assert.
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
