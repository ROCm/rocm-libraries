/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <algorithm>
#include <hip/hip_runtime.h>
#include <iostream>
#include <memory>
#include <new>

#include "device_matrix.hpp"
#include "device_vector.hpp"
#include "hipblas_arguments.hpp"

// Cache flushing for the timing loops, as in rocblas-bench.
// - With small operands the timing loop measures cache bandwidth, because every call after the
//   first finds its operands in cache. arg.flush_batch_count or arg.flush_memory_size make the
//   loop cycle through copies of the operands so each copy is evicted before it is reused
//   (Whaley and Castaldo, "Achieving accurate and context-sensitive timing for code optimization").
// - Only one of the two can be set. With flush_memory_size the number of copies is
//   flush_memory_size / bytes a call accesses, rounded up. The bytes a call accesses leave out
//   leading-dimension and increment gaps, and each operand counts at least the 256 byte block
//   its copies start on.
// - The copies are capped at nine tenths of the free device memory, counting the whole
//   allocation of each copy, gaps and alignment included.
// - Used only in timing code, not in testing code.

constexpr size_t hipblas_flush_align_bytes = 256;

inline size_t hipblas_flush_align(size_t bytes)
{
    return (bytes + hipblas_flush_align_bytes - 1) / hipblas_flush_align_bytes
           * hipblas_flush_align_bytes;
}

// Bytes a call accesses (the m x n matrix without the gaps up to lda, or the n vector elements
// without the gaps of inc) and bytes one copy allocates (gaps included), for one operand.
template <typename T>
size_t hipblas_flush_accessed_bytes(const device_matrix<T>& A)
{
    return std::min(A.m() * A.n(), A.nmemb()) * sizeof(hipblas_internal_type<T>);
}

template <typename T>
size_t hipblas_flush_accessed_bytes(const device_vector<T>& x)
{
    return std::min(x.n() ? x.n() : 1, x.nmemb()) * sizeof(hipblas_internal_type<T>);
}

template <typename T>
size_t hipblas_flush_allocated_bytes(const device_matrix<T>& A)
{
    return A.nmemb() * sizeof(hipblas_internal_type<T>);
}

template <typename T>
size_t hipblas_flush_allocated_bytes(const device_vector<T>& x)
{
    return x.nmemb() * sizeof(hipblas_internal_type<T>);
}

// Number of copies from the bytes a call accesses (accessed_bytes) and the bytes one copy of
// the operands allocates (allocated_bytes).
inline size_t hipblas_flush_batch_count_for(const Arguments& arg,
                                            size_t           accessed_bytes,
                                            size_t           allocated_bytes)
{
    size_t count = 1;
    if(arg.flush_batch_count > 1 && arg.flush_memory_size)
        std::cerr << "hipBLAS WARNING: cannot set both flush_batch_count and flush_memory_size, "
                     "using flush_batch_count = "
                  << arg.flush_batch_count << std::endl;

    if(arg.flush_batch_count > 1)
        count = arg.flush_batch_count;
    else if(arg.flush_memory_size && accessed_bytes)
        count = 1 + (arg.flush_memory_size - 1) / accessed_bytes;

    if(count > 1 && allocated_bytes)
    {
        // copy 0 is the existing operands; keep a tenth of free memory in reserve
        size_t free_bytes = 0, total_bytes = 0;
        if(hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess)
        {
            size_t fit = 1 + (free_bytes / 10 * 9) / allocated_bytes;
            if(count > fit)
            {
                std::cerr << "hipBLAS WARNING: cache flush needs " << count << " copies of "
                          << allocated_bytes << " bytes, only " << fit << " fit in device memory"
                          << std::endl;
                count = fit;
            }
        }
    }
    return count;
}

// Number of operand copies for a timing loop that rotates the given device_matrix and
// device_vector operands.
template <typename... Operands>
size_t hipblas_flush_batch_count(const Arguments& arg, const Operands&... operands)
{
    size_t accessed_bytes = (hipblas_flush_align(hipblas_flush_accessed_bytes(operands)) + ... + 0);
    size_t allocated_bytes
        = (hipblas_flush_align(hipblas_flush_allocated_bytes(operands)) + ... + 0);
    return hipblas_flush_batch_count_for(arg, accessed_bytes, allocated_bytes);
}

//!
//! @brief count copies of a device buffer for a timing loop to cycle through.
//! Copy 0 is the original buffer, copies 1 to count - 1 are new device memory holding its
//! contents at construction, each starting on a 256 byte boundary.
//!
template <typename T>
class device_rotating_buffer
{
public:
    device_rotating_buffer(hipblas_internal_type<T>* src, size_t nmemb, size_t count)
        : m_src(src)
        , m_stride(hipblas_flush_align(nmemb * sizeof(hipblas_internal_type<T>))
                   / sizeof(hipblas_internal_type<T>))
    {
        // Errors are reported through memcheck(): the CHECK_* macros return from the
        // enclosing function, which a constructor cannot do in the gtest build.
        if(count <= 1)
            return;
        try
        {
            m_copies = std::make_unique<device_vector<T>>((count - 1) * m_stride);
        }
        catch(const std::bad_alloc&)
        {
            m_status = hipErrorOutOfMemory;
            return;
        }
        m_status = m_copies->memcheck();

        // Seed copy 1, then double the initialized copies with each transfer.
        if(m_status == hipSuccess)
            m_status = hipMemcpy(
                (*this)[1], src, nmemb * sizeof(hipblas_internal_type<T>), hipMemcpyDeviceToDevice);
        for(size_t done = 1; done < count - 1 && m_status == hipSuccess; done *= 2)
        {
            size_t copies = std::min(done, count - 1 - done);
            m_status      = hipMemcpy((*this)[1 + done],
                                 (*this)[1],
                                 copies * m_stride * sizeof(hipblas_internal_type<T>),
                                 hipMemcpyDeviceToDevice);
        }
    }

    device_rotating_buffer(const device_rotating_buffer&)            = delete;
    device_rotating_buffer& operator=(const device_rotating_buffer&) = delete;

    hipError_t memcheck() const
    {
        return m_status;
    }

    hipblas_internal_type<T>* operator[](size_t i)
    {
        return i == 0 ? m_src
                      : static_cast<hipblas_internal_type<T>*>(*m_copies) + (i - 1) * m_stride;
    }

private:
    hipblas_internal_type<T>*         m_src;
    size_t                            m_stride;
    std::unique_ptr<device_vector<T>> m_copies;
    hipError_t                        m_status = hipSuccess;
};
