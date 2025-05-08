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

//
// Local declaration of the host strided batch vector.
//
template <typename T>
class host_strided_batch_vector;

//!
//! @brief Implementation of a strided batched vector on device.
//!
template <typename T, size_t PAD = 0, typename U = T>
class device_strided_batch_vector : public d_vector<T, PAD, U>
{
public:
    using value_type = T;

public:
    //!
    //! @brief The storage type to use.
    //!
    typedef enum class estorage
    {
        block,
        interleave,
    } storage;

    //!
    //! @brief Disallow copying.
    //!
    device_strided_batch_vector(const device_strided_batch_vector&) = delete;

    //!
    //! @brief Disallow assigning.
    //!
    device_strided_batch_vector& operator=(const device_strided_batch_vector&) = delete;

    //!
    //! @brief Constructor.
    //! @param n   The length of the vector.
    //! @param inc The increment.
    //! @param stride The stride.
    //! @param batch_count The batch count.
    //! @param stg The storage format to use.
    //!
    explicit device_strided_batch_vector(int64_t n,
                                         int64_t inc,
                                         rocblas_stride stride,
                                         int64_t batch_count,
                                         storage stg = storage::block)
        : d_vector<T, PAD, U>(calculate_nmemb(n, inc, stride, batch_count, stg))
        , m_storage(stg)
        , m_n(n)
        , m_inc(inc)
        , m_stride(stride)
        , m_batch_count(batch_count)
    {
        bool valid_parameters = true;

        switch(this->m_storage)
        {
        case storage::block:
        {
            if(std::abs(this->m_stride) < this->m_n * std::abs(this->m_inc))
            {
                valid_parameters = false;
            }
            break;
        }
        case storage::interleave:
        {
            if(std::abs(this->m_inc) < std::abs(this->m_stride) * this->m_batch_count)
            {
                valid_parameters = false;
            }
            break;
        }
        }

        if(valid_parameters)
        {
            this->m_data = this->device_vector_setup();
        }
    }

    //!
    //! @brief Destructor.
    //!
    ~device_strided_batch_vector()
    {
        if(nullptr != this->m_data)
        {
            this->device_vector_teardown(this->m_data);
            this->m_data = nullptr;
        }
    }

    //!
    //! @brief Returns the data pointer.
    //!
    T* data()
    {
        return this->m_data;
    }

    //!
    //! @brief Returns the data pointer.
    //!
    const T* data() const
    {
        return this->m_data;
    }

    //!
    //! @brief Returns the length.
    //!
    int64_t n() const
    {
        return this->m_n;
    }

    //!
    //! @brief Returns the increment.
    //!
    int64_t inc() const
    {
        return this->m_inc;
    }

    //!
    //! @brief Returns the batch count.
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
        return this->m_stride;
    }

    //!
    //! @brief Random access.
    //! @param batch_index The batch index.
    //! @return Pointer to the array on device.
    //!
    T* operator[](int64_t batch_index)
    {
        return (this->m_stride >= 0)
            ? this->m_data + batch_index * this->m_stride
            : this->m_data + (batch_index + 1 - this->m_batch_count) * this->m_stride;
    }

    //!
    //! @brief Constant random access.
    //! @param batch_index The batch index.
    //! @return Constant pointer to the array on device.
    //!
    const T* operator[](int64_t batch_index) const
    {
        return (this->m_stride >= 0)
            ? this->m_data + batch_index * this->m_stride
            : this->m_data + (batch_index + 1 - this->m_batch_count) * this->m_stride;
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
    //! @brief Tell whether ressources allocation failed.
    //!
    explicit operator bool() const
    {
        return nullptr != this->m_data;
    }

    //!
    //! @brief Transfer data from a strided batched vector on device.
    //! @param that That strided batched vector on device.
    //! @return The hip error.
    //!
    hipError_t transfer_from(const host_strided_batch_vector<T>& that)
    {
        return hipMemcpy(this->data(), that.data(), sizeof(T) * this->nmemb(), hipMemcpyHostToDevice);
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
    storage m_storage{storage::block};
    int64_t m_n{};
    int64_t m_inc{};
    rocblas_stride m_stride{};
    int64_t m_batch_count{};
    T* m_data{};

    static size_t
        calculate_nmemb(int64_t n, int64_t inc, rocblas_stride stride, int64_t batch_count, storage st)
    {
        switch(st)
        {
        case storage::block: return size_t(std::abs(stride)) * batch_count;
        case storage::interleave: return size_t(n) * std::abs(inc);
        }
        return 0;
    }
};
