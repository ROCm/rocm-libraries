/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cmath>
#include <type_traits>
#include <vector>

#include "host_alloc.hpp"

//!
//! @brief  Pseudo-vector subclass which uses host memory.
//!
template <typename T>
struct host_vector : std::vector<T, host_memory_allocator<T>>
{
    // Inherit constructors
    using std::vector<T, host_memory_allocator<T>>::vector;

    //!
    //! @brief Constructor.
    //! @param  inc Element index increment. If zero treated as one
    //!
    host_vector(size_t n, int64_t inc = 1)
        : std::vector<T, host_memory_allocator<T>>(calculate_nmemb(n, inc))
        , m_n(n)
        , m_inc(inc ? inc : 1)
    {
    }

    //!
    //! @brief Copy constructor from host_vector of other types convertible to T
    //!
    template <typename U, std::enable_if_t<std::is_convertible<U, T>{}, int> = 0>
    host_vector(const host_vector<U>& x)
        : std::vector<T, host_memory_allocator<T>>(x.size())
        , m_n(x.size())
        , m_inc(1)
    {
        for(size_t i = 0; i < m_n; ++i)
            (*this)[i] = x[i];
    }

    //!
    //! @brief Decay into pointer wherever pointer is expected
    //!
    operator T*()
    {
        return this->data();
    }

    //!
    //! @brief Decay into constant pointer wherever constant pointer is expected
    //!
    operator const T*() const
    {
        return this->data();
    }

    //!
    //! @brief Allow signed indices
    //!
    inline T& operator[](int64_t idx)
    {
        return std::vector<T, host_memory_allocator<T>>::operator[]((size_t)idx);
    }

    inline const T& operator[](int64_t idx) const
    {
        return std::vector<T, host_memory_allocator<T>>::operator[]((size_t)idx);
    }

    //!
    //! @brief Transfer from a device vector.
    //! @param  that That device vector.
    //! @return the hip error.
    //!
    hipError_t transfer_from(const device_vector<T>& that)
    {
        hipError_t hip_err;

        if(that.use_HMM && hipSuccess != (hip_err = hipDeviceSynchronize()))
            return hip_err;

        return hipMemcpy(*this,
                         that,
                         sizeof(T) * this->size(),
                         that.use_HMM ? hipMemcpyHostToHost : hipMemcpyDeviceToHost);
    }

    //!
    //! @brief Returns the length of the vector.
    //!
    size_t n() const
    {
        return m_n;
    }

    //!
    //! @brief Returns the increment of the vector.
    //!
    int64_t inc() const
    {
        return m_inc;
    }

    //!
    //! @brief Returns the batch count (always 1).
    //!
    static constexpr rocblas_int batch_count()
    {
        return 1;
    }

    //!
    //! @brief Returns the stride (out of context, always 0)
    //!
    static constexpr rocblas_stride stride()
    {
        return 0;
    }

    //!
    //! @brief Check if memory exists (out of context, always hipSuccess)
    //!
    static constexpr hipError_t memcheck()
    {
        return hipSuccess;
    }

private:
    size_t  m_n   = 0;
    int64_t m_inc = 0;

    static size_t calculate_nmemb(size_t n, int64_t inc)
    {
        return 1 + ((n ? n : 1) - 1) * std::abs(inc ? inc : 1);
    }
};
