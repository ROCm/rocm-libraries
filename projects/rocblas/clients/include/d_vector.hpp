/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#if defined(__GLIBC__) && __GLIBC__ < 3 && __GLIBC_MINOR__ < 39
#undef _GLIBCXX_USE_C99_INTTYPES_TR1
#endif
#include <cinttypes>
#ifdef GOOGLE_TEST
#include <mutex>
#endif

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
    size_t m_pad;
    // Byte length of each guard region (m_pad * sizeof(T)). Modified to zero in
    // device_vector_setup when a guard write fails so that device_vector_check
    // does not compare uninitialized device memory against the guard pattern.
    size_t m_guard_len;
    size_t m_bytes;

// Guards one-time initialization of m_guard against concurrent construction.
// Declared only in GOOGLE_TEST builds, where the guard and call_once are used.
#ifdef GOOGLE_TEST
    static std::once_flag m_init_flag;
#endif

public:
    bool use_HMM = false;

    static T m_guard[MEM_MAX_GUARD_PAD];

    // Non-copyable and non-movable: subclasses own a raw device pointer;
    // copying or moving would duplicate it without transferring ownership,
    // causing a double-free when both objects are destroyed.
    d_vector(const d_vector&) = delete;
    d_vector& operator=(const d_vector&) = delete;
    d_vector(d_vector&&)                 = delete;
    d_vector& operator=(d_vector&&) = delete;

    inline size_t nmemb() const noexcept
    {
        return m_size;
    }

#ifdef GOOGLE_TEST
    d_vector(size_t s, bool HMM = false)
        : m_size(s)
        , m_pad(std::min(g_DVEC_PAD, size_t(MEM_MAX_GUARD_PAD)))
        , m_guard_len(m_pad * sizeof(T))
        , m_bytes((s + m_pad * 2) * sizeof(T))
        , use_HMM(HMM)
    {
        // Initialize m_guard with NaN bytes exactly once, even if multiple
        // d_vector<T> objects are constructed concurrently.
        std::call_once(m_init_flag, [] { rocblas_init_nan(m_guard, MEM_MAX_GUARD_PAD); });
    }
#else
    d_vector(size_t s, bool HMM = false)
        : m_size(s)
        , m_pad(0)
        , m_guard_len(0)
        , m_bytes(s ? s * sizeof(T) : sizeof(T)) // minimum one element: hipMalloc(0) is UB
        , use_HMM(HMM)
    {
    }
#endif

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
#ifdef GOOGLE_TEST
        else
        {
            if(m_guard_len > 0)
            {
                // A guard that was not written is a guard that reports corruption later, so
                // the failure has to be raised here where it says what actually went wrong.
                // EXPECT is the only option in a function that returns a pointer.
                hipError_t status = hipMemcpy(d, m_guard, m_guard_len, hipMemcpyHostToDevice);
                EXPECT_EQ(status, hipSuccess)
                    << "cannot write the guard before the allocation: " << hipGetErrorName(status);

                // Point to allocated block — always, even on failure, so that d names the
                // same offset pointer that teardown will pass to free_ptr_use. hipFree
                // receives the base pointer (after d -= m_pad in teardown), not this one.
                // m_guard_len is zeroed on failure to disable checking; m_pad is not touched.
                d += m_pad;

                if(status != hipSuccess)
                {
                    // Pre-guard was never written; disable checking for this allocation so
                    // device_vector_check does not compare uninitialized device memory against
                    // the guard pattern and report corruption that never happened.
                    m_guard_len = 0;
                }
                else
                {
                    status = hipMemcpy(d + m_size, m_guard, m_guard_len, hipMemcpyHostToDevice);
                    EXPECT_EQ(status, hipSuccess) << "cannot write the guard after the allocation: "
                                                  << hipGetErrorName(status);
                    if(status != hipSuccess)
                        m_guard_len = 0;
                }
            }
        }
#endif

        // Key: the pointer handed to the caller (d, past the pre-guard in
        // GOOGLE_TEST builds; the base allocation otherwise). teardown calls
        // free_ptr_use on the same pointer before adjusting it, so the map
        // entry is always found and mem_used returns to its pre-setup value.
        if(use_HMM)
            alloc_ptr_use(d, m_bytes);

        return d;
    }

    // Reads both guard regions from device and compares them against the reference
    // pattern. Reports any mismatch as a non-fatal GTest failure. Called from
    // device_vector_teardown (i.e. from a destructor) so EXPECT not ASSERT is used
    // throughout, and early return is avoided to ensure both guards are checked.
    // No-op when m_guard_len == 0 (guards were never written or pad is zero).
    void device_vector_check(T* d)
    {
#ifdef GOOGLE_TEST
        if(m_guard_len > 0)
        {
            // One stack buffer per guard: a failed read cannot leave the other guard's
            // stale bytes in a shared buffer and produce a false pass or miss on the next
            // comparison. Sized to MEM_MAX_GUARD_PAD (the compile-time cap on m_pad) so
            // no heap allocation is needed — this runs from a destructor, where std::vector
            // construction could throw std::bad_alloc and call std::terminate.
            // Post-guard is checked first because d already points to the user allocation;
            // pre-guard requires d -= m_pad and is checked second.
            // EXPECT (not ASSERT): ASSERT would return from device_vector_check early,
            // skipping the second guard; EXPECT reports and continues.
            alignas(alignof(T)) unsigned char after_bytes[MEM_MAX_GUARD_PAD * sizeof(T)];

            hipError_t status
                = hipMemcpy(after_bytes, d + m_size, m_guard_len, hipMemcpyDeviceToHost);
            EXPECT_EQ(status, hipSuccess)
                << "cannot read the guard after the allocation: " << hipGetErrorName(status);

            if(status == hipSuccess)
                EXPECT_EQ(memcmp(after_bytes, m_guard, m_guard_len), 0) << "post-guard overwritten";

            // Point to the pre-guard region of the device allocation.
            d -= m_pad;

            alignas(alignof(T)) unsigned char before_bytes[MEM_MAX_GUARD_PAD * sizeof(T)];

            status = hipMemcpy(before_bytes, d, m_guard_len, hipMemcpyDeviceToHost);
            EXPECT_EQ(status, hipSuccess)
                << "cannot read the guard before the allocation: " << hipGetErrorName(status);

            if(status == hipSuccess)
                EXPECT_EQ(memcmp(before_bytes, m_guard, m_guard_len), 0) << "pre-guard overwritten";
        }
#endif
    }

    // Checks guards for corruption, releases the HMM accounting entry on the
    // offset pointer (before restoring it to the base), then frees the allocation.
    // Safe to call with d == nullptr (no-op).
    void device_vector_teardown(T* d)
    {
        if(d != nullptr)
        {
            device_vector_check(d);

            // Released on the pointer setup counted, which is the one it handed out, past the
            // guard. Releasing after the pointer moves back to the start of the allocation
            // misses in the tracker's map, and a miss is silent: the count stays up for the
            // life of the process and host_mem_safe starts refusing allocations that would
            // have fit.
            if(use_HMM)
                free_ptr_use(d);

            if(m_pad > 0)
                d -= m_pad; // restore to start of alloc

            // Free device memory
            CHECK_HIP_ERROR((hipFree)(d));
        }
    }
};

template <typename T>
T d_vector<T>::m_guard[MEM_MAX_GUARD_PAD] = {};

#ifdef GOOGLE_TEST
template <typename T>
std::once_flag d_vector<T>::m_init_flag;
#endif

#undef MEM_MAX_GUARD_PAD
