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

#include "d_vector.hpp"

//
// Local declaration of the host strided batch vector.
//
template <typename T>
class host_strided_batch_vector;

//!
//! @brief Implementation of a strided batched vector on device.
//!
template <typename T>
class device_multiple_strided_batch_vector : public d_vector<T>
{
public:
    //!
    //! @brief Disallow copying.
    //!
    device_multiple_strided_batch_vector(const device_multiple_strided_batch_vector&) = delete;

    //!
    //! @brief Disallow assigning.
    //!
    device_multiple_strided_batch_vector& operator=(const device_multiple_strided_batch_vector&)
        = delete;

    //!
    //! @brief Constructor.
    //! @param n   The length of the vector.
    //! @param inc The increment.
    //! @param stride The stride.
    //! @param batch_count The batch count.
    //! @param multiple_count The multiple count.
    //! @param HMM         HipManagedMemory Flag.
    //!
    explicit device_multiple_strided_batch_vector(size_t         n,
                                                  int64_t        inc,
                                                  rocblas_stride stride,
                                                  int64_t        batch_count,
                                                  int64_t        multiple_count,
                                                  bool           HMM = false)
        : d_vector<T>(calculate_nmemb(n, inc, stride, batch_count, multiple_count), HMM)
        , m_n(n)
        , m_inc(inc ? inc : 1)
        , m_stride(stride)
        , m_batch_count(batch_count)
        , m_multiple_count(multiple_count)
        , m_multiple_stride(calculate_multiple_stride(n, inc, stride, batch_count))
        , m_nmemb(calculate_nmemb(n, inc, stride, batch_count, multiple_count))
    {
        m_data = this->device_vector_setup();
    }

    //!
    //! @brief Destructor.
    //!
    ~device_multiple_strided_batch_vector()
    {
        if(nullptr != m_data)
        {
            this->device_vector_teardown(m_data);
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
    //! @brief Returns the stride value.
    //!
    rocblas_stride stride() const
    {
        return m_stride;
    }

    //!
    //! @brief Returns the number of members.
    //!
    size_t nmemb() const
    {
        return m_nmemb;
    }

    //!
    //! @brief Returns pointer.
    //! @param multiple_index The multiple index.
    //! @return A mutable pointer to the multiple_index'th vector.
    //!
    T* operator[](int64_t multiple_index)
    {
        return m_data + multiple_index * m_multiple_stride;
    }

    //!
    //! @brief Returns non-mutable pointer.
    //! @param multiple_index The multiple index.
    //! @return A non-mutable mutable pointer to the multiple_index'th vector.
    //!
    const T* operator[](int64_t multiple_index) const
    {
        return m_data + multiple_index * m_multiple_stride;
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
    //! @brief Tell whether resource allocation failed.
    //!
    explicit operator bool() const
    {
        return nullptr != m_data;
    }

    //!
    //! @brief Transfer data from a strided batched vector on host.
    //! @param that That strided batched vector on host.
    //! @return The hip error.
    //!
    hipError_t transfer_from(const host_strided_batch_vector<T>& that)
    {
        return hipMemcpy(data(),
                         that.data(),
                         sizeof(T) * this->nmemb(),
                         this->use_HMM ? hipMemcpyHostToHost : hipMemcpyHostToDevice);
    }

    //!
    //! @brief Broadcast data from one strided_batch_vector on host to each batch_count strided_batch_vectors.
    //! @param that That strided_batch_vector on host.
    //! @return The hip error.
    //!
    hipError_t broadcast_one_strided_batch_vector_from(const host_strided_batch_vector<T>& that)
    {
        if(that.nmemb() > this->m_multiple_stride)
        {
            rocblas_cout << "ERROR: size mismatch in broadcast_one_vector_from" << std::endl;
            rocblas_cout << "that.nmemb, m_multiple_strided = " << that.nmemb() << ", "
                         << m_multiple_stride << std::endl;
            return hipErrorInvalidValue;
        }
        hipError_t status = hipSuccess;
        for(int64_t multiple_index = 0; multiple_index < m_multiple_count; multiple_index++)
        {

            status = hipMemcpy(this->data() + (multiple_index * m_multiple_stride),
                               that.data(),
                               sizeof(T) * that.nmemb(),
                               this->use_HMM ? hipMemcpyHostToHost : hipMemcpyHostToDevice);

            if(status != hipSuccess)
            {
                rocblas_cout << "ERROR: status, multiple_index = " << status << ", "
                             << multiple_index << std::endl;
                break;
            }
        }
        return status;
    }

    //!
    //! @brief Check if memory exists.
    //! @return hipSuccess if memory exists, hipErrorOutOfMemory otherwise.
    //!
    hipError_t memcheck() const
    {
        if(*this)
            return hipSuccess;
        else
            return hipErrorOutOfMemory;
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
