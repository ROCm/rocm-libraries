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

#include "host_alloc.hpp"

//
// Local declaration of the device strided batch vector.
//
template <typename T>
class device_strided_batch_vector;

//!
//! @brief Implementation of a host strided batched vector.
//!
template <typename T>
class host_multiple_strided_batch_vector
{
public:
    //!
    //! @brief Disallow copying.
    //!
    host_multiple_strided_batch_vector(const host_multiple_strided_batch_vector&) = delete;

    //!
    //! @brief Disallow assigning.
    //!
    host_multiple_strided_batch_vector& operator=(const host_multiple_strided_batch_vector&)
        = delete;

    //!
    //! @brief Constructor.
    //! @param n   The length of the vector.
    //! @param inc The increment.
    //! @param stride The stride.
    //! @param batch_count The batch count.
    //! @param multiple_count The multiple count.
    //!
    explicit host_multiple_strided_batch_vector(
        size_t n, int64_t inc, rocblas_stride stride, int64_t batch_count, int64_t multiple_count)
        : m_n(n)
        , m_inc(inc ? inc : 1)
        , m_stride(stride)
        , m_batch_count(batch_count)
        , m_multiple_count(multiple_count)
        , m_multiple_stride(calculate_multiple_stride(n, inc, stride, batch_count))
        , m_nmemb(calculate_nmemb(n, inc, stride, batch_count, multiple_count))
    {
        bool valid_parameters = m_nmemb > 0;
        if(valid_parameters)
        {
            m_data = (T*)host_malloc_throw(m_nmemb, sizeof(T));
        }
    }

    //!
    //! @brief Destructor.
    //!
    ~host_multiple_strided_batch_vector()
    {
        if(nullptr != m_data)
        {
            host_free(m_data);
            m_data = nullptr;
        }
    }

    //!
    //! @brief Returns the data pointer.
    //!
    T* data()
    {
        return m_data;
    }

    //!
    //! @brief Returns the data pointer.
    //!
    const T* data() const
    {
        return m_data;
    }

    //!
    //! @brief Returns the length.
    //!
    size_t n() const
    {
        return m_n;
    }

    //!
    //! @brief Returns the increment.
    //!
    int64_t inc() const
    {
        return m_inc;
    }

    //!
    //! @brief Returns the batch count.
    //!
    int64_t batch_count() const
    {
        return m_batch_count;
    }

    //!
    //! @brief Returns the multiple count.
    //!
    int64_t multiple_count() const
    {
        return m_multiple_count;
    }

    //!
    //! @brief Returns the stride.
    //!
    rocblas_stride stride() const
    {
        return m_stride;
    }

    //!
    //! @brief Returns the multiple stride.
    //!
    rocblas_stride multiple_stride() const
    {
        return m_multiple_stride;
    }

    //!
    //! @brief Returns nmemb
    //!
    rocblas_stride nmemb() const
    {
        return m_nmemb;
    }

    //!
    //! @brief Returns pointer.
    //! @param multiple_index The multiple index.
    //! @return A mutable pointer to the multiple_index'th strided batched vector.
    //!
    T* operator[](int64_t multiple_index)
    {
        return this->m_data + this->m_multiple_stride * multiple_index;
    }

    //!
    //! @brief Returns non-mutable pointer.
    //! @param multiple_index The multiple index.
    //! @return A non-mutable mutable pointer to the multiple_index'th strided batched vector.
    //!
    const T* operator[](int64_t multiple_index) const
    {
        return this->m_data + this->m_multiple_stride * multiple_index;
    }

    //!
    //! @brief Cast operator.
    //! @remark Returns the pointer of the first vector.
    //!
    operator T*()
    {
        return (*this)[0];
    }

    //!
    //! @brief Non-mutable cast operator.
    //! @remark Returns the non-mutable pointer of the first vector.
    //!
    operator const T*() const
    {
        return (*this)[0];
    }

    //!
    //! @brief Tell whether resources allocation failed.
    //!
    explicit operator bool() const
    {
        return nullptr != m_data;
    }

    //!
    //! @brief Copy data from a strided batched vector on host.
    //! @param that That strided batched vector on host.
    //! @return true if successful, false otherwise.
    //!
    bool copy_from(const host_multiple_strided_batch_vector& that)
    {
        if(that.n() == m_n && that.inc() == m_inc && that.stride() == m_stride
           && that.batch_count() == m_batch_count && that.multiple_stride() == m_multiple_stride
           && that.multiple_count() == m_multiple_count)
        {
            memcpy(data(), that.data(), sizeof(T) * m_nmemb);
            return true;
        }
        else
        {
            return false;
        }
    }

    //!
    //! @brief Transfer data from a strided batched vector on device.
    //! @param that That strided batched vector on device.
    //! @return The hip error.
    //!
    hipError_t transfer_from(const device_multiple_strided_batch_vector<T>& that)
    {
        hipError_t hip_err;

        if(that.use_HMM && hipSuccess != (hip_err = hipDeviceSynchronize()))
            return hip_err;

        return hipMemcpy(m_data,
                         that.data(),
                         sizeof(T) * m_nmemb,
                         that.use_HMM ? hipMemcpyHostToHost : hipMemcpyDeviceToHost);
    }

    //!
    //! @brief Check if memory exists.
    //! @return hipSuccess if memory exists, hipErrorOutOfMemory otherwise.
    //!
    hipError_t memcheck() const
    {
        return ((bool)*this) ? hipSuccess : hipErrorOutOfMemory;
    }

private:
    size_t         m_n{};
    int64_t        m_inc{};
    rocblas_stride m_stride{};
    int64_t        m_batch_count{};
    int64_t        m_multiple_count{};
    int64_t        m_multiple_stride{};
    size_t         m_nmemb{};
    T*             m_data{};

    static int64_t
        calculate_multiple_stride(size_t n, int64_t inc, rocblas_stride stride, int64_t batch_count)
    {
        return align_stride<T>((inc > 0 ? inc : -inc) * n
                               + size_t((batch_count > 0 ? batch_count : 1) - 1)
                                     * std::abs(stride));
    }

    static size_t calculate_nmemb(
        size_t n, int64_t inc, rocblas_stride stride, int64_t batch_count, int64_t multiple_count)
    {
        int64_t multiple_stride = calculate_multiple_stride(n, inc, stride, batch_count);
        return multiple_stride * multiple_count;
    }
};
