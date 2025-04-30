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
 *
 * ************************************************************************ */
#pragma once

//
// Local declaration of the device strided batch vector.
//
template <typename T, size_t PAD, typename U>
class device_strided_batch_vector;

//!
//! @brief Implementation of a host strided batched vector.
//!
template <typename T>
class host_strided_batch_vector
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
        interleave
    } storage;

    //!
    //! @brief Disallow copying.
    //!
    host_strided_batch_vector(const host_strided_batch_vector&) = delete;

    //!
    //! @brief Disallow assigning.
    //!
    host_strided_batch_vector& operator=(const host_strided_batch_vector&) = delete;

    //!
    //! @brief Constructor.
    //! @param n   The length of the vector.
    //! @param inc The increment.
    //! @param stride The stride.
    //! @param batch_count The batch count.
    //! @param stg The storage format to use.
    //!
    explicit host_strided_batch_vector(int64_t        n,
                                       int64_t        inc,
                                       rocblas_stride stride,
                                       int64_t        batch_count,
                                       storage        stg = storage::block)
        : m_storage(stg)
        , m_n(n)
        , m_inc(inc)
        , m_stride(stride)
        , m_batch_count(batch_count)
        , m_nmemb(calculate_nmemb(n, inc, stride, batch_count, stg))
    {
        bool valid_parameters = this->m_nmemb > 0;
        if(valid_parameters)
        {
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
                this->m_data = new T[this->m_nmemb];
            }
        }
    }

    //!
    //! @brief Destructor.
    //!
    ~host_strided_batch_vector()
    {
        if(nullptr != this->m_data)
        {
            delete[] this->m_data;
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
    //! @brief Returns the stride.
    //!
    rocblas_stride stride() const
    {
        return this->m_stride;
    }

    //!
    //! @brief Random access.
    //! @param batch_index The batch index.
    //! @return Pointer to the array on host.
    //!
    T* operator[](int64_t batch_index)
    {
        return (this->m_stride >= 0)
                   ? this->m_data + this->m_stride * batch_index
                   : this->m_data + (batch_index + 1 - this->m_batch_count) * this->m_stride;
    }

    //!
    //! @brief Constant random access.
    //! @param batch_index The batch index.
    //! @return Constant pointer to the array on host.
    //!
    const T* operator[](int64_t batch_index) const
    {
        return (this->m_stride >= 0)
                   ? this->m_data + this->m_stride * batch_index
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
    //! @brief Copy data from a strided batched vector on host.
    //! @param that That strided batched vector on host.
    //! @return The hip error.
    //!
    hipError_t transfer_from(const host_strided_batch_vector& that)
    {
        return hipMemcpy(this->data(), that.data(), sizeof(T) * this->m_nmemb, hipMemcpyHostToHost);
    }

    //!
    //! @brief Transfer data from a strided batched vector on device.
    //! @param that That strided batched vector on device.
    //! @return The hip error.
    //!
    template <size_t PAD, typename U>
    hipError_t transfer_from(const device_strided_batch_vector<T, PAD, U>& that)
    {
        return hipMemcpy(
            this->m_data, that.data(), sizeof(T) * this->m_nmemb, hipMemcpyDeviceToHost);
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
    storage        m_storage{storage::block};
    int64_t        m_n{};
    int64_t        m_inc{};
    rocblas_stride m_stride{};
    int64_t        m_batch_count{};
    size_t         m_nmemb{};
    T*             m_data{};

    static size_t calculate_nmemb(
        int64_t n, int64_t inc, rocblas_stride stride, int64_t batch_count, storage st)
    {
        switch(st)
        {
        case storage::block:
            return size_t(std::abs(stride)) * batch_count;
        case storage::interleave:
            return size_t(n) * std::abs(inc);
        }
        return 0;
    }
};
