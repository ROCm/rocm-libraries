/* ************************************************************************
 * Copyright (C) 2018-2022 Advanced Micro Devices, Inc. All rights reserved.
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
// Forward declaration of the host matrix.
//
template <typename T>
class host_matrix;

//!
//! @brief pseudo-matrix subclass which uses device memory
//!
template <typename T>
class device_matrix : public d_vector<T>
{

public:
    //!
    //! @brief Disallow copying.
    //!
    device_matrix(const device_matrix&) = delete;

    //!
    //! @brief Disallow assigning
    //!
    device_matrix& operator=(const device_matrix&) = delete;

    //!
    //! @brief Constructor.
    //! @param m           The number of rows of the Matrix.
    //! @param n           The number of cols of the Matrix.
    //! @param lda         The leading dimension of the Matrix.
    //! @param HMM         HipManagedMemory Flag.
    //!
    explicit device_matrix(size_t m, size_t n, size_t lda, bool HMM = false)
        : d_vector<T>{n * lda, HMM}
        , m_m{m}
        , m_n{n}
        , m_lda{lda}
        , m_data{this->device_vector_setup()}
    {
    }

    //!
    //! @brief Destructor.
    //!
    ~device_matrix()
    {
        this->device_vector_teardown(m_data);
        m_data = nullptr;
    }

    //!
    //! @brief Returns the rows of the Matrix.
    //!
    size_t m() const
    {
        return this->m_m;
    }

    //!
    //! @brief Returns the cols of the Matrix.
    //!
    size_t n() const
    {
        return this->m_n;
    }

    //!
    //! @brief Returns the leading dimension of the Matrix.
    //!
    size_t lda() const
    {
        return this->m_lda;
    }

    //!
    //! @brief Returns the batch count (always 1).
    //!
    rocblas_int batch_count() const
    {
        return 1;
    }

    //!
    //! @brief Returns the stride (out of context, always 0)
    //!
    rocblas_stride stride() const
    {
        return 0;
    }

    //!
    //! @brief Decay into pointer wherever pointer is expected.
    //!
    operator T*()
    {
        return m_data;
    }

    //!
    //! @brief Decay into constant pointer wherever pointer is expected.
    //!
    operator const T*() const
    {
        return m_data;
    }

    //!
    //! @brief Transfer data from a host matrix.
    //! @param that The host matrix.
    //! @return the hip error.
    //!
    hipError_t transfer_from(const host_matrix<T>& that)
    {
        return hipMemcpy(m_data,
                         (const T*)that,
                         this->nmemb() * sizeof(T),
                         this->use_HMM ? hipMemcpyHostToHost : hipMemcpyHostToDevice);
    }

    hipError_t memcheck() const
    {
        return !this->nmemb() || m_data ? hipSuccess : hipErrorOutOfMemory;
    }

private:
    size_t m_m   = 0;
    size_t m_n   = 0;
    size_t m_lda = 0;
    T*     m_data{};
};
