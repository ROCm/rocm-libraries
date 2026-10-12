/* ************************************************************************
* Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights Reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*
* ************************************************************************ */

#pragma once
#include "auto_testing_bad_arg.hpp"
#include "display.hpp"
#include "flops.hpp"
#include "gbyte.hpp"
#include "rocsparse.hpp"
#include "rocsparse_check.hpp"
#include "rocsparse_matrix_factory.hpp"
#include "rocsparse_reproducibility.hpp"
#include "rocsparse_reproducibility_test_save.hpp"
#include "rocsparse_traits.hpp"
#include "rocsparse_vector_utils.hpp"
#include "utility.hpp"

namespace rocsparse_clients
{

    template <typename T>
    struct dense_vector_t
    {
    protected:
        host_dense_vector<T>*  m_h;
        device_dense_vector<T> m_d;
        rocsparse_local_dnvec  m_dnvec;

    public:
        operator rocsparse_dnvec_descr&()
        {
            return this->m_dnvec;
        }
        operator const rocsparse_dnvec_descr&() const
        {
            return this->m_dnvec;
        }

        static host_dense_vector<T>* init(int64_t size)
        {
            host_dense_vector<T>* that = new host_dense_vector<T>(size);
            T*                    d    = that->data();
            for(int64_t i = 0; i < size; ++i)
            {
                d[i] = 1;
            }
            return that;
        }

        ~dense_vector_t()
        {
            if(m_h != nullptr)
            {
                delete m_h;
            }
        }
        dense_vector_t(int64_t size)
            : m_h(init(size))
            , m_d(m_h[0])
            , m_dnvec(m_d)
        {
        }
    };

    template <typename T>
    struct dense_matrix_t
    {
    protected:
        host_dense_matrix<T>*  m_h;
        device_dense_matrix<T> m_d;
        rocsparse_local_dnmat  m_dnmat;

    public:
        operator rocsparse_dnmat_descr&()
        {
            return this->m_dnmat;
        }
        operator const rocsparse_dnmat_descr&() const
        {
            return this->m_dnmat;
        }

        static host_dense_matrix<T>* init(int64_t m, int64_t n)
        {
            host_dense_matrix<T>* that = new host_dense_matrix<T>(m, n);
            T*                    d    = that->data();
            for(int64_t i = 0; i < m * n; ++i)
                d[0] = 1;
            return that;
        }
        ~dense_matrix_t()
        {
            if(m_h != nullptr)
                delete m_h;
        }
        dense_matrix_t(int64_t m, int64_t n)
            : m_h(init(m, n))
            , m_d(m_h[0])
            , m_dnmat(m_d)
        {
        }
    };

    template <typename T, typename I, typename J>
    struct csr_tridiag_matrix_t
    {
    protected:
        host_csr_matrix<T, I, J>*  m_h;
        device_csr_matrix<T, I, J> m_d;
        rocsparse_local_spmat      m_spmat;

    public:
        operator rocsparse_spmat_descr&()
        {
            return this->m_spmat;
        }
        operator const rocsparse_spmat_descr&() const
        {
            return this->m_spmat;
        }

        // Tridiagonal matrix with 2 on the diagonal and -1 on the off-diagonals.
        static host_csr_matrix<T, I, J>* init(int64_t size)
        {
            const int64_t nnz  = 3 * size - 2;
            auto that = new host_csr_matrix<T, I, J>(size, size, nnz, rocsparse_index_base_zero);

            int64_t k = 0;
            for(int64_t i = 0; i < size; ++i)
            {
                that->ptr[i] = k;
                if(i > 0)
                {
                    that->ind[k] = i - 1;
                    that->val[k] = -1;
                    ++k;
                }
                that->ind[k] = i;
                that->val[k] = 2;
                ++k;
                if(i < size - 1)
                {
                    that->ind[k] = i + 1;
                    that->val[k] = -1;
                    ++k;
                }
            }
            that->ptr[size] = k;
            return that;
        }

        csr_tridiag_matrix_t(int64_t size)
            : m_h(init(size))
            , m_d(m_h[0])
            , m_spmat(m_d)
        {
        }
    };

}
