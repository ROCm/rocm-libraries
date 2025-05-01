/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef ROCSPARSE_MATRIX_COO_HPP
#define ROCSPARSE_MATRIX_COO_HPP

#include "rocsparse_vector.hpp"

template <memory_mode::value_t MODE, typename T, typename I = rocsparse_int>
struct coo_matrix
{
    template <typename S>
    using array_t = typename memory_traits<MODE>::template array_t<S>;

    I                      m{};
    I                      n{};
    int64_t                nnz{};
    rocsparse_index_base   base{};
    rocsparse_storage_mode storage_mode{rocsparse_storage_mode_sorted};
    array_t<I>             row_ind{};
    array_t<I>             col_ind{};
    array_t<T>             val{};

    coo_matrix(){};
    ~coo_matrix(){};

    coo_matrix(I m_, I n_, int64_t nnz_, rocsparse_index_base base_)
        : m(m_)
        , n(n_)
        , nnz(nnz_)
        , base(base_)
        , row_ind(nnz_)
        , col_ind(nnz_)
        , val(nnz_){};

    explicit coo_matrix(const coo_matrix<MODE, T, I>& that_, bool transfer = true)
        : coo_matrix<MODE, T, I>(that_.m, that_.n, that_.nnz, that_.base)
    {
        if(transfer)
        {
            this->transfer_from(that_);
        }
    }

    template <memory_mode::value_t THAT_MODE>
    explicit coo_matrix(const coo_matrix<THAT_MODE, T, I>& that_, bool transfer = true)
        : coo_matrix<MODE, T, I>(that_.m, that_.n, that_.nnz, that_.base)
    {
        if(transfer)
        {
            this->transfer_from(that_);
        }
    }

    template <memory_mode::value_t THAT_MODE>
    void transfer_from(const coo_matrix<THAT_MODE, T, I>& that)
    {
        CHECK_HIP_THROW_ERROR((this->m == that.m && this->n == that.n && this->nnz == that.nnz
                               && this->base == that.base)
                                  ? hipSuccess
                                  : hipErrorInvalidValue);

        this->row_ind.transfer_from(that.row_ind);
        this->col_ind.transfer_from(that.col_ind);
        this->val.transfer_from(that.val);
    };

    void define(I m_, I n_, int64_t nnz_, rocsparse_index_base base_)
    {
        if(m_ != this->m)
        {
            this->m = m_;
        }

        if(n_ != this->n)
        {
            this->n = n_;
        }

        if(nnz_ != this->nnz)
        {
            this->nnz = nnz_;
            this->row_ind.resize(this->nnz);
            this->col_ind.resize(this->nnz);
            this->val.resize(this->nnz);
        }

        if(base_ != this->base)
        {
            this->base = base_;
        }
    }

    template <memory_mode::value_t THAT_MODE>
    void unit_check(const coo_matrix<THAT_MODE, T, I>& that_) const
    {
        switch(MODE)
        {
        case memory_mode::device:
        {
            coo_matrix<memory_mode::host, T, I> on_host(*this);
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

                this->row_ind.unit_check(that_.row_ind);
                this->col_ind.unit_check(that_.col_ind);
                this->val.unit_check(that_.val);

                break;
            }
            case memory_mode::device:
            {
                coo_matrix<memory_mode::host, T, I> that(that_);
                this->unit_check(that);
                break;
            }
            }
            break;
        }
        }
    }

    template <memory_mode::value_t THAT_MODE>
    void near_check(const coo_matrix<THAT_MODE, T, I>& that_,
                    floating_data_t<T>                 tol = default_tolerance<T>::value) const
    {
        switch(MODE)
        {
        case memory_mode::device:
        {
            coo_matrix<memory_mode::host, T, I> on_host(*this);
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

                this->row_ind.unit_check(that_.row_ind);
                this->col_ind.unit_check(that_.col_ind);
                this->val.near_check(that_.val, tol);
                break;
            }
            case memory_mode::device:
            {
                coo_matrix<memory_mode::host, T, I> that(that_);
                this->near_check(that, tol);
                break;
            }
            }
            break;
        }
        }
    }

    void info() const
    {
        std::cout << "INFO COO " << std::endl;
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
            const I* pi = (const I*)this->row_ind;
            const I* pj = (const I*)this->col_ind;
            const T* v  = (const T*)val;

            std::cout << "COO MATRIX" << std::endl;
            std::cout << "M:" << this->m << std::endl;
            std::cout << "N:" << this->n << std::endl;
            std::cout << "NNZ:" << this->nnz << std::endl;
            std::cout << "BASE:" << this->base << std::endl;
            for(int64_t k = 0; k < this->nnz; ++k)
            {
                I i = pi[k] - this->base;
                I j = pj[k] - this->base;
                std::cout << "( " << i << ", " << j << ", " << v[k] << " )" << std::endl;
            }
            break;
        }
        case memory_mode::device:
        {
            coo_matrix<memory_mode::host, T, I> on_host(*this);
            on_host.print();
            break;
        }
        }
    }
};

template <typename T, typename I = rocsparse_int>
using host_coo_matrix = coo_matrix<memory_mode::host, T, I>;
template <typename T, typename I = rocsparse_int>
using device_coo_matrix = coo_matrix<memory_mode::device, T, I>;
template <typename T, typename I = rocsparse_int>
using managed_coo_matrix = coo_matrix<memory_mode::managed, T, I>;

#endif // ROCSPARSE_MATRIX_COO_HPP
