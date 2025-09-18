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

#include <string.h>

#include "host_alloc.hpp"
#include "rocblas_init.hpp"

//
// Local declaration of the device batch vector.
//
template <typename T>
class device_batch_vector;

//!
//! @brief Implementation of the batch vector on host.
//!
template <typename T>
class host_batch_vector
{
public:
    //!
    //! @brief Delete copy constructor.
    //!
    host_batch_vector(const host_batch_vector<T>& that) = delete;

    //!
    //! @brief Delete copy assignment.
    //!
    host_batch_vector& operator=(const host_batch_vector<T>& that) = delete;

    //!
    //! @brief Constructor.
    //! @param n           The length of the vector.
    //! @param inc         The increment.
    //! @param batch_count The batch count.
    //!
    explicit host_batch_vector(size_t n, int64_t inc, int64_t batch_count)
        : m_n(n)
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
    ~host_batch_vector()
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
    //! @brief Returns the batch count.
    //!
    int64_t batch_count() const
    {
        return m_batch_count;
    }

    //!
    //! @brief Returns the nmemb.
    //!
    size_t nmemb() const
    {
        return m_nmemb;
    }

    //!
    //! @brief Returns the stride value.
    //!
    rocblas_stride stride() const
    {
        return 0;
    }

    //!
    //! @brief Random access to the vectors.
    //! @param batch_index the batch index.
    //! @return The mutable pointer.
    //!
    T* operator[](int64_t batch_index)
    {

        return m_data[batch_index];
    }

    //!
    //! @brief Constant random access to the vectors.
    //! @param batch_index the batch index.
    //! @return The non-mutable pointer.
    //!
    const T* operator[](int64_t batch_index) const
    {

        return m_data[batch_index];
    }

    //!
    //! @brief Cast to a double pointer.
    //!
    // clang-format off
    operator T**()
    // clang-format on
    {
        return m_data;
    }

    //!
    //! @brief Constant cast to a double pointer.
    //!
    operator const T* const *()
    {
        return m_data;
    }

    //!
    //! @brief Copy from a host batched vector.
    //! @param that the vector the data is copied from.
    //! @return true if the copy is done successfully, false otherwise.
    //!
    bool copy_from(const host_batch_vector<T>& that)
    {
        if((batch_count() == that.batch_count()) && (n() == that.n()) && (inc() == that.inc()))
        {
            size_t num_bytes = m_nmemb * sizeof(T) * m_batch_count;
            if(m_batch_count > 0)
                memcpy((*this)[0], that[0], num_bytes);
            return true;
        }
        else
        {
            return false;
        }
    }

    //!
    //! @brief Transfer from a device batched vector.
    //! @param that the vector the data is copied from.
    //! @return the hip error.
    //!
    hipError_t transfer_from(const device_batch_vector<T>& that)
    {
        hipError_t hip_err;

        if(that.use_HMM && hipSuccess != (hip_err = hipDeviceSynchronize()))
            return hip_err;

        size_t        num_bytes = m_nmemb * sizeof(T) * m_batch_count;
        hipMemcpyKind kind      = that.use_HMM ? hipMemcpyHostToHost : hipMemcpyDeviceToHost;

        if(m_batch_count > 0)
        {
            if(hipSuccess != (hip_err = hipMemcpy((*this)[0], that[0], num_bytes, kind)))
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
        return (nullptr != m_data) ? hipSuccess : hipErrorOutOfMemory;
    }

private:
    size_t  m_n{}; // This may hold a matrix so using size_t.
    int64_t m_inc{};
    size_t  m_nmemb{};
    int64_t m_batch_count{};
    T**     m_data{};

    static size_t calculate_nmemb(size_t n, int64_t inc)
    {
        // allocate when n is zero
        return 1 + ((n ? n : 1) - 1) * std::abs(inc ? inc : 1);
    }

    bool try_initialize_memory()
    {
        bool success = (nullptr != (m_data = (T**)host_calloc_throw(m_batch_count, sizeof(T*))));
        if(success)
        {
            for(int64_t batch_index = 0; batch_index < m_batch_count; ++batch_index)
            {
                if(batch_index == 0)
                {
                    success = (nullptr
                               != (m_data[batch_index]
                                   = (T*)host_malloc_throw(m_nmemb * m_batch_count, sizeof(T))));
                    if(false == success)
                    {
                        break;
                    }
                }
                else
                {
                    m_data[batch_index] = m_data[0] + batch_index * m_nmemb;
                }
            }
        }
        return success;
    }

    void free_memory()
    {
        if(nullptr != m_data)
        {
            for(int64_t batch_index = 0; batch_index < m_batch_count; ++batch_index)
            {
                if(batch_index == 0 && nullptr != m_data[batch_index])
                {
                    host_free(m_data[batch_index]);
                    m_data[batch_index] = nullptr;
                }
                else
                {
                    m_data[batch_index] = nullptr;
                }
            }

            host_free(m_data);
            m_data = nullptr;
        }
    }
};

//!
//! @brief Overload output operator.
//! @param os The ostream.
//! @param that That host batch vector.
//!
template <typename T>
rocblas_internal_ostream& operator<<(rocblas_internal_ostream& os, const host_batch_vector<T>& that)
{
    auto n           = that.n();
    auto inc         = std::abs(that.inc());
    auto batch_count = that.batch_count();

    for(int64_t batch_index = 0; batch_index < batch_count; ++batch_index)
    {
        auto batch_data = that[batch_index];
        os << "[" << batch_index << "] = { " << batch_data[0];
        for(size_t i = 1; i < n; ++i)
        {
            os << ", " << batch_data[i * inc];
        }
        os << " }" << std::endl;
    }

    return os;
}
