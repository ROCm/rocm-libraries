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

#if defined(__GLIBC__) && __GLIBC__ < 3 && __GLIBC_MINOR__ < 39
#undef _GLIBCXX_USE_C99_INTTYPES_TR1
#endif
#include <cinttypes>
#ifdef GOOGLE_TEST
#include <memory>
#include <mutex>
#include <new>
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
    // Geometry, fixed at construction. const so the guard offsets used by
    // device_vector_setup, device_vector_check and device_vector_teardown cannot drift
    // apart: all three derive their pointer arithmetic from m_pad alone.
    const size_t m_size;
    const size_t m_pad;
    // Byte length of each guard region (m_pad * sizeof(T)).
    const size_t m_guard_len;
    const size_t m_bytes;

#ifdef GOOGLE_TEST
    // Set only once both guard regions actually hold the pattern. Kept separate from
    // m_guard_len so a failed guard write disables checking without touching geometry.
    bool m_guard_written = false;

    // Guards one-time initialization of m_guard against concurrent construction.
    // Declared only in GOOGLE_TEST builds, where the guard and call_once are used.
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
        else if(m_pad > 0)
        {
            // A guard that was not written is a guard that reports corruption later, so the
            // failure has to be raised here, where the message says what actually went wrong.
            // EXPECT is the only option in a function that returns a pointer.
            hipError_t status = hipMemcpy(d, m_guard, m_guard_len, hipMemcpyDefault);
            EXPECT_EQ(status, hipSuccess)
                << "cannot write the guard before the allocation: " << hipGetErrorName(status);

            // Offset past the pre-guard unconditionally, so d names the pointer teardown will
            // hand to free_ptr_use whether or not the guard writes succeeded. hipFree receives
            // the base pointer, recovered in teardown by subtracting the same m_pad.
            d += m_pad;

            if(status == hipSuccess)
            {
                status = hipMemcpy(d + m_size, m_guard, m_guard_len, hipMemcpyDefault);
                EXPECT_EQ(status, hipSuccess)
                    << "cannot write the guard after the allocation: " << hipGetErrorName(status);
            }

            // Checked later only if both regions hold the pattern; otherwise
            // device_vector_check would compare uninitialised device memory against it and
            // report corruption that never happened.
            m_guard_written = (status == hipSuccess);
        }
#endif

        // Key: the pointer handed to the caller (d, past the pre-guard in
        // GOOGLE_TEST builds; the base allocation otherwise). teardown calls
        // free_ptr_use on the same pointer before adjusting it, so the map
        // entry is always found and mem_used returns to its pre-setup value.
        // m_bytes includes guard pad bytes in GOOGLE_TEST builds; alloc and free
        // are symmetric (both use m_bytes) so the ceiling delta is always zero.
        if(use_HMM)
            alloc_ptr_use(d, m_bytes);

        return d;
    }

    // Reads both guard regions back from the device and compares them against the reference
    // pattern, reporting any mismatch as a non-fatal GTest failure. Called from
    // device_vector_teardown (i.e. from a destructor) so EXPECT rather than ASSERT is used
    // throughout: ASSERT would return early and skip the second guard.
    void device_vector_check(T* d)
    {
#ifdef GOOGLE_TEST
        if(!m_guard_written)
            return;

        // One host buffer, exactly one guard long, reused for both regions. Heap rather than
        // an automatic array because m_guard_len is a runtime value, so an array would have to
        // be sized for MEM_MAX_GUARD_PAD: 128 KB of stack for a 16-byte type, to compare at
        // most that much. nothrow new rather than std::vector because a throwing allocation in
        // a destructor calls std::terminate. No alignment needed: hipMemcpy and memcmp are
        // both byte-wise.
        std::unique_ptr<unsigned char[]> host_guard(new(std::nothrow) unsigned char[m_guard_len]);
        if(!host_guard)
        {
            ADD_FAILURE() << "cannot allocate " << m_guard_len
                          << " bytes to read the guards back; corruption would go unreported";
            return;
        }

        // Called only when memcmp detected a difference; scans byte-by-byte to report the
        // count and the index of the first differing byte, without paying the scan cost on
        // the common (clean) path.
        auto report_guard_corruption
            = [](const unsigned char* host, const T* ref, size_t guard_bytes, const char* tag) {
                  const auto* r         = reinterpret_cast<const unsigned char*>(ref);
                  size_t      differing = 0, first = 0;
                  for(size_t i = 0; i < guard_bytes; ++i)
                      if(host[i] != r[i])
                      {
                          if(!differing) // record index of first differing byte only once
                              first = i;
                          ++differing;
                      }
                  // memcmp detected corruption before this lambda was called, so differing
                  // must be > 0 and first is valid. The check defends against any future
                  // caller that violates the precondition.
                  if(differing > 0)
                      ADD_FAILURE()
                          << differing << " " << tag << "-guard byte(s) corrupted; first at byte "
                          << first << " of " << guard_bytes << " (expected 0x" << std::hex
                          << static_cast<unsigned>(r[first]) << ", got 0x"
                          << static_cast<unsigned>(host[first]) << std::dec << ")";
              };

        // Post-guard first, because d still points at the user allocation. Each comparison is
        // gated on its own copy succeeding, so a failed read cannot leave the other region's
        // bytes behind to be compared a second time.
        hipError_t status = hipMemcpy(host_guard.get(), d + m_size, m_guard_len, hipMemcpyDefault);
        EXPECT_EQ(status, hipSuccess)
            << "cannot read the guard after the allocation: " << hipGetErrorName(status);
        if(status == hipSuccess && memcmp(host_guard.get(), m_guard, m_guard_len) != 0)
            report_guard_corruption(host_guard.get(), m_guard, m_guard_len, "post");

        // Pre-guard sits m_pad elements below the user pointer.
        status = hipMemcpy(host_guard.get(), d - m_pad, m_guard_len, hipMemcpyDefault);
        EXPECT_EQ(status, hipSuccess)
            << "cannot read the guard before the allocation: " << hipGetErrorName(status);
        if(status == hipSuccess && memcmp(host_guard.get(), m_guard, m_guard_len) != 0)
            report_guard_corruption(host_guard.get(), m_guard, m_guard_len, "pre");
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
