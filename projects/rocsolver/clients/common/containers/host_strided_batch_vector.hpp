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

#include <ostream>

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
    explicit host_strided_batch_vector(int64_t n,
                                       int64_t inc,
                                       rocblas_stride stride,
                                       int64_t batch_count,
                                       storage stg = storage::block)
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
                // Value-initialization (`new T{}` or `new T[]{}`) of a non-class type yields zero-initialization
                this->m_data = new T[this->m_nmemb]{};
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
    //! @return true if successful, false otherwise.
    //!
    bool copy_from(const host_strided_batch_vector& that)
    {
        if(that.n() == this->m_n && that.inc() == this->m_inc && that.stride() == this->m_stride
           && that.batch_count() == this->m_batch_count)
        {
            memcpy(this->data(), that.data(), sizeof(T) * this->m_nmemb);
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
    template <size_t PAD, typename U>
    hipError_t transfer_from(const device_strided_batch_vector<T, PAD, U>& that)
    {
        return hipMemcpy(this->m_data, that.data(), sizeof(T) * this->m_nmemb, hipMemcpyDeviceToHost);
    }

    //!
    //! @brief Check if memory exists.
    //! @return hipSuccess if memory exists, hipErrorOutOfMemory otherwise.
    //!
    hipError_t memcheck() const
    {
        return ((bool)*this) ? hipSuccess : hipErrorOutOfMemory;
    }

    //!
    //! @brief Get size of vector
    //! @return number of elements
    //!
    size_t size() const
    {
        return this->m_nmemb;
    }

private:
    storage m_storage{storage::block};
    int64_t m_n{};
    int64_t m_inc{};
    rocblas_stride m_stride{};
    int64_t m_batch_count{};
    size_t m_nmemb{};
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

//!
//! @brief Overload output operator.
//! @param os The ostream.
//! @param that That host strided batch vector.
//!
template <typename T>
std::ostream& operator<<(std::ostream& os, const host_strided_batch_vector<T>& that)
{
    auto n = that.n();
    auto inc = std::abs(that.inc());
    auto batch_count = that.batch_count();

    for(int64_t batch_index = 0; batch_index < batch_count; ++batch_index)
    {
        auto batch_data = that[batch_index];
        os << "[" << batch_index << "] = { " << batch_data[0];
        for(int64_t i = 1; i < n; ++i)
        {
            os << ", " << batch_data[i * inc];
        }
        os << " }" << std::endl;
    }

    return os;
}
