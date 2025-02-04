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

#include "d_vector.hpp"

//
// Local declaration of the host strided batch vector.
//
template <typename T>
class host_batch_vector;

//!
//! @brief  pseudo-vector subclass which uses a batch of device memory pointers and
//!  - an array of pointers in host memory
//!  - an array of pointers in device memory
//!
template <typename T>
class device_batch_vector : public d_vector<T>
{
public:
    //!
    //! @brief Disallow copying.
    //!
    device_batch_vector(const device_batch_vector&) = delete;

    //!
    //! @brief Disallow assigning.
    //!
    device_batch_vector& operator=(const device_batch_vector&) = delete;

    //!
    //! @brief Constructor.
    //! @param n           The length of the vector.
    //! @param inc         The increment.
    //! @param batch_count The batch count.
    //! @param HMM         HipManagedMemory Flag.
    //!
    explicit device_batch_vector(size_t n, int64_t inc, int64_t batch_count, bool HMM = false)
        : d_vector<T>(calculate_nmemb(n, inc) * batch_count, HMM)
        // d_vector is a single contiguous block for performance
        , m_n(n)
        , m_inc(inc ? inc : 1)
        , m_nmemb(calculate_nmemb(n, inc))
        , m_batch_count(batch_count)
    {
        if(false == try_initialize_memory())
        {
            free_memory();
        }
    }

    //!
    //! @brief Destructor.
    //!
    ~device_batch_vector()
    {
        free_memory();
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
    //! @brief Returns the value of batch_count.
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
        return 0;
    }

    //!
    //! @brief Access to device data.
    //! @return Pointer to the device data.
    //!
    T** ptr_on_device()
    {
        return m_device_data;
    }

    //!
    //! @brief Const access to device data.
    //! @return Const pointer to the device data.
    //!
    const T* const* ptr_on_device() const
    {
        return m_device_data;
    }

    //!
    //! @brief access to device data.
    //! @return Const pointer to the device data.
    //!
    T* const* const_batch_ptr()
    {
        return m_device_data;
    }

    //!
    //! @brief Random access.
    //! @param batch_index The batch index.
    //! @return Pointer to the array on device.
    //!
    T* operator[](rocblas_int batch_index)
    {

        return m_data[batch_index];
    }

    //!
    //! @brief Constant random access.
    //! @param batch_index The batch index.
    //! @return Constant pointer to the array on device.
    //!
    const T* operator[](rocblas_int batch_index) const
    {

        return m_data[batch_index];
    }

    //!
    //! @brief Const cast of the data on host.
    //!
    operator const T* const *() const
    {
        return m_data;
    }

    //!
    //! @brief Cast of the data on host.
    //!
    // clang-format off
    operator T**()
    // clang-format on
    {
        return m_data;
    }

    //!
    //! @brief Tell whether resource allocation failed.
    //!
    explicit operator bool() const
    {
        return nullptr != m_data;
    }

    //!
    //! @brief Copy from a host batched vector.
    //! @param that The host_batch_vector to copy.
    //!
    hipError_t transfer_from(const host_batch_vector<T>& that)
    {
        hipError_t hip_err;
        //
        // Copy each vector.
        //
        hipMemcpyKind kind = this->use_HMM ? hipMemcpyHostToHost : hipMemcpyHostToDevice;
        if(m_batch_count > 0)
        {
            if(hipSuccess
               != (hip_err
                   = hipMemcpy((*this)[0], that[0], sizeof(T) * m_nmemb * m_batch_count, kind)))
            {
                return hip_err;
            }
        }

        return hipSuccess;
    }

    //!
    //! @brief Broadcast from a host batched vector.
    //! @param that The host_batch_vector to copy.
    //!
    hipError_t broadcast_one_batch_vector_from(const host_batch_vector<T>& that)
    {
        hipError_t hip_err;
        //
        // Copy each vector.
        //

        if(this->m_nmemb != that.nmemb())
        {
            rocblas_cout << "ERROR: m_nmemb mismatch in broadcast_one_batch_vector_from"
                         << std::endl;
            return hipErrorInvalidValue;
        }
        if(that.batch_count() == 0 || this->m_batch_count % that.batch_count() != 0)
        {
            rocblas_cout << "ERROR: batch_count mismatch in broadcast_one_batch_vector_from"
                         << std::endl;
            return hipErrorInvalidValue;
        }

        int64_t flush_batch_count = this->m_batch_count / that.batch_count();

        hipMemcpyKind kind = this->use_HMM ? hipMemcpyHostToHost : hipMemcpyHostToDevice;
        for(uint64_t flush_batch_index = 0; flush_batch_index < flush_batch_count;
            flush_batch_index++)
        {
            if(m_batch_count > 0)
            {
                hip_err = hipMemcpy((*this)[flush_batch_index * uint64_t(that.batch_count())],
                                    that[0],
                                    sizeof(T) * that.nmemb() * that.batch_count(),
                                    kind);
                if(hipSuccess != hip_err)
                    return hip_err;
            }
        }

        return hipSuccess;
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
    size_t  m_n{};
    int64_t m_inc{};
    size_t  m_nmemb{}; // in one batch
    int64_t m_batch_count{};
    T**     m_data{};
    T**     m_device_data{};

    static size_t calculate_nmemb(size_t n, int64_t inc)
    {
        // allocate even for zero n
        return 1 + ((n ? n : 1) - 1) * std::abs(inc ? inc : 1);
    }

    //!
    //! @brief Try to allocate the resources.
    //! @return true if success false otherwise.
    //!
    bool try_initialize_memory()
    {
        bool success = false;

        success
            = (hipSuccess
               == (!this->use_HMM ? (hipMalloc)(&m_device_data, m_batch_count * sizeof(T*))
                                  : hipMallocManaged(&m_device_data, m_batch_count * sizeof(T*))));
        if(success)
        {
            success = (nullptr
                       != (m_data = !this->use_HMM ? (T**)calloc(m_batch_count, sizeof(T*))
                                                   : m_device_data));
            if(success)
            {
                for(int64_t batch_index = 0; batch_index < m_batch_count; ++batch_index)
                {
                    if(batch_index == 0)
                    {
                        success = (nullptr != (m_data[batch_index] = this->device_vector_setup()));
                        if(!success)
                        {
                            break;
                        }
                    }
                    else
                    {
                        m_data[batch_index] = m_data[0] + batch_index * m_nmemb;
                    }
                }

                if(success && !this->use_HMM)
                {
                    success = (hipSuccess
                               == hipMemcpy(m_device_data,
                                            m_data,
                                            sizeof(T*) * m_batch_count,
                                            hipMemcpyHostToDevice));
                }
            }
        }
        return success;
    }

    //!
    //! @brief Free the resources, as much as we can.
    //!
    void free_memory()
    {
        if(nullptr != m_data)
        {
            for(int64_t batch_index = 0; batch_index < m_batch_count; ++batch_index)
            {
                if(batch_index == 0 && nullptr != m_data[batch_index])
                {
                    this->device_vector_teardown(m_data[batch_index]);
                    m_data[batch_index] = nullptr;
                }
                else
                {
                    m_data[batch_index] = nullptr;
                }
            }

            if(!this->use_HMM)
            {
                free(m_data);
            }
            // else this is just a copy of m_device_data

            m_data = nullptr;
        }

        if(nullptr != m_device_data)
        {
            auto tmp_device_data = m_device_data;
            m_device_data        = nullptr;
            CHECK_HIP_ERROR((hipFree)(tmp_device_data));
        }
    }
};
