/*! \file */
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
#ifndef ROCSPARSE_MATRIX_SELL_HPP
#define ROCSPARSE_MATRIX_SELL_HPP

#include "rocsparse_vector.hpp"

#include "rocsparse_clients_routine_trace.hpp"

template <memory_mode::value_t MODE,
          typename T,
          typename I = rocsparse_int,
          typename J = rocsparse_int>
struct sell_matrix
{
    template <typename S>
    using array_t = typename memory_traits<MODE>::template array_t<S>;

    J                      m{};
    J                      n{};
    I                      nnz{};
    J                      sell_slice_size{};
    I                      sell_colval_size{};
    rocsparse_index_base   base{};
    rocsparse_storage_mode storage_mode{rocsparse_storage_mode_sorted};
    array_t<I>             ptr{};
    array_t<J>             ind{};
    array_t<T>             val{};

    sell_matrix(){};
    ~sell_matrix(){};

    sell_matrix(
        J m_, J n_, I nnz_, J sell_slice_size_, I sell_colval_size_, rocsparse_index_base base_)
        : m(m_)
        , n(n_)
        , nnz(nnz_)
        , sell_slice_size(sell_slice_size_)
        , sell_colval_size(sell_colval_size_)
        , base(base_)
        , ptr(((m - 1) / sell_slice_size + 1) + 1)
        , ind(sell_colval_size)
        , val(sell_colval_size){};

    explicit sell_matrix(const sell_matrix<MODE, T, I, J>& that_, bool transfer = true)
        : sell_matrix<MODE, T, I, J>(
            that_.m, that_.n, that_.nnz, that_.sell_slice_size, that_.sell_colval_size, that_.base)
    {
        ROCSPARSE_CLIENTS_ROUTINE_TRACE;

        if(transfer)
        {
            this->transfer_from(that_);
        }
    }

    template <memory_mode::value_t THAT_MODE>
    explicit sell_matrix(const sell_matrix<THAT_MODE, T, I, J>& that_, bool transfer = true)
        : sell_matrix<MODE, T, I, J>(
            that_.m, that_.n, that_.nnz, that_.sell_slice_size, that_.sell_colval_size, that_.base)
    {
        ROCSPARSE_CLIENTS_ROUTINE_TRACE;

        if(transfer)
        {
            this->transfer_from(that_);
        }
    }

    template <memory_mode::value_t THAT_MODE>
    sell_matrix& operator()(const sell_matrix<THAT_MODE, T, I, J>& that_, bool transfer = true)
    {
        ROCSPARSE_CLIENTS_ROUTINE_TRACE;
        this->define(
            that_.m, that_.n, that_.nnz, that_.sell_slice_size, that_.sell_colval_size, that_.base);
        if(transfer)
        {
            this->transfer_from(that_);
        }
        return *this;
    }

    template <memory_mode::value_t THAT_MODE>
    void transfer_from(const sell_matrix<THAT_MODE, T, I, J>& that)
    {
        ROCSPARSE_CLIENTS_ROUTINE_TRACE;

        CHECK_HIP_THROW_ERROR((this->m == that.m && this->n == that.n && this->nnz == that.nnz
                               && this->sell_slice_size == that.sell_slice_size
                               && this->sell_colval_size == that.sell_colval_size
                               && this->base == that.base)
                                  ? hipSuccess
                                  : hipErrorInvalidValue);

        this->ptr.transfer_from(that.ptr);
        this->ind.transfer_from(that.ind);
        this->val.transfer_from(that.val);
    };

    void define(
        J m_, J n_, I nnz_, J sell_slice_size_, I sell_colval_size_, rocsparse_index_base base_)
    {
        ROCSPARSE_CLIENTS_ROUTINE_TRACE;

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
        }

        if(sell_slice_size_ != this->sell_slice_size)
        {
            this->sell_slice_size = sell_slice_size_;
        }

        if(sell_colval_size_ != this->sell_colval_size)
        {
            this->sell_colval_size = sell_colval_size_;
        }

        if(base_ != this->base)
        {
            this->base = base_;
        }

        J nslices = ((m - 1) / sell_slice_size + 1);

        this->ptr.resize(nslices + 1);
        this->ind.resize(this->sell_colval_size);
        this->val.resize(this->sell_colval_size);
    }

    template <memory_mode::value_t THAT_MODE>
    void near_check(const sell_matrix<THAT_MODE, T, I, J>& that_,
                    floating_data_t<T>                     tol = default_tolerance<T>::value) const
    {
        ROCSPARSE_CLIENTS_ROUTINE_TRACE;

        switch(MODE)
        {
        case memory_mode::device:
        {
            sell_matrix<memory_mode::host, T, I, J> on_host(*this);
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
                unit_check_scalar(this->sell_slice_size, that_.sell_slice_size);
                unit_check_scalar(this->sell_colval_size, that_.sell_colval_size);
                unit_check_enum(this->base, that_.base);

                this->ptr.unit_check(that_.ptr);
                this->ind.unit_check(that_.ind);
                this->val.near_check(that_.val, tol);

                break;
            }
            case memory_mode::device:
            {
                sell_matrix<memory_mode::host, T, I, J> that(that_);
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
        ROCSPARSE_CLIENTS_ROUTINE_TRACE;

        std::cout << "INFO SELL" << std::endl;
        std::cout << " m     : " << this->m << std::endl;
        std::cout << " n     : " << this->n << std::endl;
        std::cout << " nnz   : " << this->nnz << std::endl;
        std::cout << " sell_slice_size : " << this->sell_slice_size << std::endl;
        std::cout << " sell_colval_size : " << this->sell_colval_size << std::endl;
        std::cout << " base  : " << this->base << std::endl;
    }
};

template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
using host_sell_matrix = sell_matrix<memory_mode::host, T, I, J>;
template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
using device_sell_matrix = sell_matrix<memory_mode::device, T, I, J>;
template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
using managed_sell_matrix = sell_matrix<memory_mode::managed, T, I, J>;

#endif // ROCSPARSE_MATRIX_SELL_HPP
