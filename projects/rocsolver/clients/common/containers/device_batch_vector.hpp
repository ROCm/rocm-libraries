/* **************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include "d_vector.hpp"

//
// Local declaration of the host strided batch vector.
//
template <typename T>
class host_batch_vector;

//!
//! @brief  pseudo-vector subclass which uses a batch of device memory pointers
//! and
//!  - an array of pointers in host memory
//!  - an array of pointers in device memory
//!
template <typename T, size_t PAD = 0, typename U = T>
class device_batch_vector : private d_vector<T, PAD, U>
{
public:
    using value_type = T;

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
    //!
    explicit device_batch_vector(int64_t n, int64_t inc, int64_t batch_count)
        : d_vector<T, PAD, U>(size_t(n) * std::abs(inc))
        , m_n(n)
        , m_inc(inc)
        , m_batch_count(batch_count)
    {
        if(false == this->try_initialize_memory())
        {
            this->free_memory();
        }
    }

    //!
    //! @brief Constructor.
    //! @param n           The length of the vector.
    //! @param inc         The increment.
    //! @param stride      (UNUSED) The stride.
    //! @param batch_count The batch count.
    //!
    explicit device_batch_vector(int64_t n, int64_t inc, rocblas_stride stride, int64_t batch_count)
        : device_batch_vector(n, inc, batch_count)
    {
    }

    //!
    //! @brief Constructor (kept for backward compatibility only, to be removed).
    //! @param batch_count The number of vectors.
    //! @param size_vector The size of each vectors.
    //!
    explicit device_batch_vector(int64_t batch_count, size_t size_vector)
        : device_batch_vector(size_vector, 1, batch_count)
    {
    }

    //!
    //! @brief Destructor.
    //!
    ~device_batch_vector()
    {
        this->free_memory();
    }

    //!
    //! @brief Returns the length of the vector.
    //!
    int64_t n() const
    {
        return this->m_n;
    }

    //!
    //! @brief Returns the increment of the vector.
    //!
    int64_t inc() const
    {
        return this->m_inc;
    }

    //!
    //! @brief Returns the value of batch_count.
    //!
    int64_t batch_count() const
    {
        return this->m_batch_count;
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
        return this->m_device_data;
    }

    //!
    //! @brief Const access to device data.
    //! @return Const pointer to the device data.
    //!
    const T* const* ptr_on_device() const
    {
        return this->m_device_data;
    }

    T* const* data()
    {
        return this->m_device_data;
    }

    const T* const* data() const
    {
        return this->m_device_data;
    }

    //!
    //! @brief Random access.
    //! @param batch_index The batch index.
    //! @return Pointer to the array on device.
    //!
    T* operator[](int64_t batch_index)
    {
        return this->m_data[batch_index];
    }

    //!
    //! @brief Constant random access.
    //! @param batch_index The batch index.
    //! @return Constant pointer to the array on device.
    //!
    const T* operator[](int64_t batch_index) const
    {
        return this->m_data[batch_index];
    }

    //!
    //! @brief Const cast of the data on host.
    //!
    operator const T* const *() const
    {
        return this->m_data;
    }

    // clang-format off
    //!
    //! @brief Cast of the data on host.
    //!
    operator T**()
    {
        return this->m_data;
    }
    // clang-format on

    //!
    //! @brief Tell whether ressources allocation failed.
    //!
    explicit operator bool() const
    {
        return nullptr != this->m_data;
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
        for(int64_t batch_index = 0; batch_index < this->m_batch_count; ++batch_index)
        {
            if(hipSuccess
               != (hip_err = hipMemcpy((*this)[batch_index], that[batch_index],
                                       sizeof(T) * this->nmemb(), hipMemcpyHostToDevice)))
            {
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
    int64_t m_n{};
    int64_t m_inc{};
    int64_t m_batch_count{};
    T** m_data{};
    T** m_device_data{};

    //!
    //! @brief Try to allocate the ressources.
    //! @return true if success false otherwise.
    //!
    bool try_initialize_memory()
    {
        bool success = false;

        success = (hipSuccess == (hipMalloc)(&this->m_device_data, this->m_batch_count * sizeof(T*)));
        if(success)
        {
            success = (nullptr != (this->m_data = (T**)calloc(this->m_batch_count, sizeof(T*))));
            if(success)
            {
                for(int64_t batch_index = 0; batch_index < this->m_batch_count; ++batch_index)
                {
                    success = (nullptr != (this->m_data[batch_index] = this->device_vector_setup()));
                    if(!success)
                    {
                        break;
                    }
                }

                if(success)
                {
                    success = (hipSuccess
                               == hipMemcpy(this->m_device_data, this->m_data,
                                            sizeof(T*) * this->m_batch_count, hipMemcpyHostToDevice));
                }
            }
        }
        return success;
    }

    //!
    //! @brief Free the ressources, as much as we can.
    //!
    void free_memory()
    {
        if(nullptr != this->m_data)
        {
            for(int64_t batch_index = 0; batch_index < this->m_batch_count; ++batch_index)
            {
                if(nullptr != this->m_data[batch_index])
                {
                    this->device_vector_teardown(this->m_data[batch_index]);
                    this->m_data[batch_index] = nullptr;
                }
            }

            free(this->m_data);
            this->m_data = nullptr;
        }

        if(nullptr != this->m_device_data)
        {
            auto tmp_device_data = this->m_device_data;
            this->m_device_data = nullptr;
            CHECK_HIP_ERROR((hipFree)(tmp_device_data));
        }
    }
};
