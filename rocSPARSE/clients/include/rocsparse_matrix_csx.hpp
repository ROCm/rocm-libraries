/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2023 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef ROCSPARSE_MATRIX_CSX_HPP
#define ROCSPARSE_MATRIX_CSX_HPP

#include "rocsparse_vector.hpp"

template <memory_mode::value_t MODE,
          rocsparse_direction  DIRECTION,
          typename T,
          typename I,
          typename J>
struct csx_matrix
{
    template <typename S>
    using array_t = typename memory_traits<MODE>::template array_t<S>;

    static constexpr rocsparse_direction dir = DIRECTION;
    J                                    m{};
    J                                    n{};
    I                                    nnz{};
    rocsparse_index_base                 base{};
    rocsparse_storage_mode               storage_mode{rocsparse_storage_mode_sorted};
    array_t<I>                           ptr{};
    array_t<J>                           ind{};
    array_t<T>                           val{};

    csx_matrix(){};
    ~csx_matrix(){};

    csx_matrix(J m_, J n_, I nnz_, rocsparse_index_base base_)
        : m(m_)
        , n(n_)
        , nnz(nnz_)
        , base(base_)
        , ptr((rocsparse_direction_row == DIRECTION) ? ((m_ > 0) ? (m_ + 1) : 0)
                                                     : ((n_ > 0) ? (n_ + 1) : 0))
        , ind(nnz_)
        , val(nnz_){};

    explicit csx_matrix(const csx_matrix<MODE, DIRECTION, T, I, J>& that_, bool transfer = true)
        : csx_matrix<MODE, DIRECTION, T, I, J>(that_.m, that_.n, that_.nnz, that_.base)
    {
        if(transfer)
        {
            this->transfer_from(that_);
        }
    }

    template <memory_mode::value_t THAT_MODE>
    explicit csx_matrix(const csx_matrix<THAT_MODE, DIRECTION, T, I, J>& that_,
                        bool                                             transfer = true)
        : csx_matrix<MODE, DIRECTION, T, I, J>(that_.m, that_.n, that_.nnz, that_.base)
    {
        if(transfer)
        {
            this->transfer_from(that_);
        }
    }

    rocsparse_status scale()
    {
        switch(MODE)
        {
        case memory_mode::host:
        case memory_mode::managed:
        {
            const I nsequences = (DIRECTION == rocsparse_direction_row) ? this->m : this->n;
            for(I i = 0; i < nsequences; ++i)
            {
                const I            end = this->ptr[i + 1] - this->base;
                floating_data_t<T> mx  = static_cast<floating_data_t<T>>(0);
                for(I k = this->ptr[i] - this->base; k < end; ++k)
                {
                    mx = std::max(mx, std::abs(this->val[k]));
                }

                if(mx == floating_data_t<T>(0))
                {
                    mx = floating_data_t<T>(1);
                }

                mx = floating_data_t<T>(1.0) / mx;
                for(I k = this->ptr[i] - this->base; k < end; ++k)
                {
                    this->val[k] *= mx;
                }
            }

            return rocsparse_status_success;
        }
        case memory_mode::device:
        {
            return rocsparse_status_not_implemented;
        }
        }
        return rocsparse_status_invalid_value;
    }

    void define(J m_, J n_, I nnz_, rocsparse_index_base base_)
    {
        if(m_ != this->m)
        {
            this->m = m_;
            if(DIRECTION == rocsparse_direction_row)
            {
                this->ptr.resize(this->m + 1);
            }
        }

        if(n_ != this->n)
        {
            this->n = n_;
            if(DIRECTION == rocsparse_direction_column)
            {
                this->ptr.resize(this->n + 1);
            }
        }

        if(nnz_ != this->nnz)
        {
            this->nnz = nnz_;
            this->ind.resize(this->nnz);
            this->val.resize(this->nnz);
        }

        if(base_ != this->base)
        {
            this->base = base_;
        }
    }

    void info() const
    {
        std::cout << "INFO CSX " << std::endl;
        std::cout << " dir  : " << DIRECTION << std::endl;
        std::cout << " m    : " << this->m << std::endl;
        std::cout << " n    : " << this->n << std::endl;
        std::cout << " nnz  : " << this->nnz << std::endl;
        std::cout << " base : " << this->base << std::endl;
    }

    void print() const
    {
        switch(MODE)
        {
        case memory_mode::host:
        case memory_mode::managed:
        {
            std::cout << "CSX MATRIX" << std::endl;
            std::cout << "DIR:" << DIRECTION << std::endl;
            std::cout << "M:" << this->m << std::endl;
            std::cout << "N:" << this->n << std::endl;
            std::cout << "NNZ:" << this->nnz << std::endl;
            std::cout << "BASE:" << this->base << std::endl;
            const I* p  = (const I*)this->ptr;
            const J* pj = (const J*)this->ind;
            const T* v  = (const T*)val;

            switch(DIRECTION)
            {
            case rocsparse_direction_row:
            {
                for(J i = 0; i < this->m; ++i)
                {
                    std::cout << "ROW " << i << std::endl;
                    for(I k = p[i] - this->base; k < p[i + 1] - this->base; ++k)
                    {
                        J j = pj[k] - this->base;
                        std::cout << "   (" << j << "," << v[k] << ")" << std::endl;
                    }
                    std::cout << std::endl;
                }
                break;
            }

            case rocsparse_direction_column:
            {
                for(J j = 0; j < this->n; ++j)
                {
                    std::cout << "COLUMN " << j << std::endl;
                    for(I k = p[j] - this->base; k < p[j + 1] - this->base; ++k)
                    {
                        J i = pj[k] - this->base;
                        std::cout << "   (" << i << "," << v[k] << ")" << std::endl;
                    }
                    std::cout << std::endl;
                }
                break;
            }
            }
            break;
        }
        case memory_mode::device:
        {
            csx_matrix<memory_mode::host, DIRECTION, T, I, J> on_host(*this);
            on_host.print();
            break;
        }
        }
    }

    bool is_invalid() const
    {
        if(this->m < 0)
            return true;
        if(this->n < 0)
            return true;
        if(this->nnz < 0)
            return true;
        if(this->nnz > this->m * this->n)
            return true;

        if(DIRECTION == rocsparse_direction_row && this->ptr.size() != this->m + 1)
            return true;
        else if(DIRECTION == rocsparse_direction_column && this->ptr.size() != this->n + 1)
            return true;

        if(this->ind.size() != this->nnz)
            return true;
        if(this->val.size() != this->nnz)
            return true;
        return false;
    };

    template <memory_mode::value_t THAT_MODE>
    void transfer_from(const csx_matrix<THAT_MODE, DIRECTION, T, I, J>& that)
    {
        CHECK_HIP_THROW_ERROR((this->m == that.m && this->n == that.n && this->nnz == that.nnz
                               && this->dir == that.dir && this->base == that.base)
                                  ? hipSuccess
                                  : hipErrorInvalidValue);
        switch(DIRECTION)
        {
        case rocsparse_direction_row:
        {
            if(this->m > 0)
            {
                this->ptr.transfer_from(that.ptr);
            }
            break;
        }
        case rocsparse_direction_column:
        {
            if(this->n > 0)
            {
                this->ptr.transfer_from(that.ptr);
            }
            break;
        }
        }
        this->ind.transfer_from(that.ind);
        this->val.transfer_from(that.val);
    };

    template <memory_mode::value_t THAT_MODE>
    void unit_check(const csx_matrix<THAT_MODE, DIRECTION, T, I, J>& that_) const
    {
        switch(MODE)
        {
        case memory_mode::device:
        {
            csx_matrix<memory_mode::host, DIRECTION, T, I, J> on_host(*this);
            on_host.unit_check(that_);
            break;
        }

        case memory_mode::managed:
        case memory_mode::host:
        {
            switch(THAT_MODE)
            {
            case memory_mode::managed:
            case memory_mode::host:
            {
                unit_check_scalar(this->m, that_.m);
                unit_check_scalar(this->n, that_.n);
                unit_check_scalar(this->nnz, that_.nnz);
                unit_check_enum(this->base, that_.base);

                switch(DIRECTION)
                {
                case rocsparse_direction_row:
                {
                    if(this->m > 0)
                    {
                        this->ptr.unit_check(that_.ptr);
                    }
                    break;
                }
                case rocsparse_direction_column:
                {
                    if(this->n > 0)
                    {
                        this->ptr.unit_check(that_.ptr);
                    }
                    break;
                }
                }

                if(this->nnz > 0)
                {
                    this->ind.unit_check(that_.ind);
                    this->val.unit_check(that_.val);
                }

                break;
            }
            case memory_mode::device:
            {
                csx_matrix<memory_mode::host, DIRECTION, T, I, J> that(that_);
                this->unit_check(that);
                break;
            }
            }
            break;
        }
        }
    }

    template <memory_mode::value_t THAT_MODE>
    void near_check(const csx_matrix<THAT_MODE, DIRECTION, T, I, J>& that_,
                    floating_data_t<T> tol = default_tolerance<T>::value) const
    {
        switch(MODE)
        {
        case memory_mode::device:
        {
            csx_matrix<memory_mode::host, DIRECTION, T, I, J> on_host(*this);
            on_host.near_check(that_, tol);
            break;
        }

        case memory_mode::managed:
        case memory_mode::host:
        {
            switch(THAT_MODE)
            {
            case memory_mode::managed:
            case memory_mode::host:
            {
                unit_check_scalar(this->m, that_.m);
                unit_check_scalar(this->n, that_.n);
                unit_check_scalar(this->nnz, that_.nnz);
                unit_check_enum(this->base, that_.base);

                switch(DIRECTION)
                {
                case rocsparse_direction_row:
                {
                    if(this->m > 0)
                    {
                        this->ptr.unit_check(that_.ptr);
                    }
                    break;
                }
                case rocsparse_direction_column:
                {
                    if(this->n > 0)
                    {
                        this->ptr.unit_check(that_.ptr);
                    }
                    break;
                }
                }
                if(this->nnz > 0)
                {
                    this->ind.unit_check(that_.ind);
                    this->val.near_check(that_.val, tol);
                }

                break;
            }
            case memory_mode::device:
            {
                csx_matrix<memory_mode::host, DIRECTION, T, I, J> that(that_);
                this->near_check(that, tol);
                break;
            }
            }
            break;
        }
        }
    }
};

template <rocsparse_direction DIRECTION,
          typename T,
          typename I = rocsparse_int,
          typename J = rocsparse_int>
using host_csx_matrix = csx_matrix<memory_mode::host, DIRECTION, T, I, J>;

template <rocsparse_direction DIRECTION,
          typename T,
          typename I = rocsparse_int,
          typename J = rocsparse_int>
using device_csx_matrix = csx_matrix<memory_mode::device, DIRECTION, T, I, J>;

template <rocsparse_direction DIRECTION,
          typename T,
          typename I = rocsparse_int,
          typename J = rocsparse_int>
using managed_csx_matrix = csx_matrix<memory_mode::managed, DIRECTION, T, I, J>;

template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
using host_csr_matrix = host_csx_matrix<rocsparse_direction_row, T, I, J>;
template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
using host_csc_matrix = host_csx_matrix<rocsparse_direction_column, T, I, J>;

template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
using device_csr_matrix = device_csx_matrix<rocsparse_direction_row, T, I, J>;
template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
using device_csc_matrix = device_csx_matrix<rocsparse_direction_column, T, I, J>;

template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
using managed_csr_matrix = managed_csx_matrix<rocsparse_direction_row, T, I, J>;
template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
using managed_csc_matrix = managed_csx_matrix<rocsparse_direction_column, T, I, J>;

#endif // ROCSPARSE_MATRIX_CSX_HPP
