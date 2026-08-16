/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef ROCSPARSE_MATRIX_BELL_HPP
#define ROCSPARSE_MATRIX_BELL_HPP

#include "rocsparse_vector.hpp"

#include "rocsparse_clients_routine_trace.hpp"

// Blocked-ELL matrix in the units expected by the rocsparse_create_bell_descr API:
//   - m, n     : scalar number of rows and columns
//   - ell_cols : scalar number of padded ELL columns (a multiple of bdim)
//   - bdim     : block dimension (ell_block_dim)
// The column-index array holds one block-column index per ELL block slot, of length
// block_rows * (ell_cols / bdim) with block_rows = (m + bdim - 1) / bdim. The value array holds
// one scalar entry per (scalar row, padded ELL column), of length m * ell_cols. These match the
// array lengths validated in rocsparse_create_bell_descr.
template <memory_mode::value_t MODE, typename T, typename I = rocsparse_int>
struct bell_matrix
{
    template <typename S>
    using array_t = typename memory_traits<MODE>::template array_t<S>;

    I                      m{};
    I                      n{};
    I                      ell_cols{};
    I                      bdim{1};
    rocsparse_index_base   base{};
    rocsparse_storage_mode storage_mode{rocsparse_storage_mode_sorted};
    array_t<I>             ind{};
    array_t<T>             val{};

    // Number of block rows.
    static I block_rows(I m_, I bdim_)
    {
        return (bdim_ > 0) ? ((m_ + bdim_ - 1) / bdim_) : 0;
    }

    // Length of the column-index array (one block-column index per ELL block slot).
    static int64_t ind_size(I m_, I ell_cols_, I bdim_)
    {
        return (bdim_ > 0) ? (int64_t(block_rows(m_, bdim_)) * (ell_cols_ / bdim_)) : 0;
    }

    // Length of the value array (one scalar entry per scalar row and padded ELL column).
    static int64_t val_size(I m_, I ell_cols_)
    {
        return int64_t(m_) * ell_cols_;
    }

    bell_matrix(){};
    ~bell_matrix(){};

    bell_matrix(I m_, I n_, I ell_cols_, I bdim_, rocsparse_index_base base_)
        : m(m_)
        , n(n_)
        , ell_cols(ell_cols_)
        , bdim(bdim_)
        , base(base_)
        , ind(ind_size(m_, ell_cols_, bdim_))
        , val(val_size(m_, ell_cols_)){};

    explicit bell_matrix(const bell_matrix<MODE, T, I>& that_, bool transfer = true)
        : bell_matrix<MODE, T, I>(that_.m, that_.n, that_.ell_cols, that_.bdim, that_.base)
    {
        ROCSPARSE_CLIENTS_ROUTINE_TRACE;

        if(transfer)
        {
            this->transfer_from(that_);
        }
    }

    template <memory_mode::value_t THAT_MODE>
    explicit bell_matrix(const bell_matrix<THAT_MODE, T, I>& that_, bool transfer = true)
        : bell_matrix<MODE, T, I>(that_.m, that_.n, that_.ell_cols, that_.bdim, that_.base)
    {
        ROCSPARSE_CLIENTS_ROUTINE_TRACE;

        if(transfer)
        {
            this->transfer_from(that_);
        }
    }

    template <memory_mode::value_t THAT_MODE>
    bell_matrix& operator()(const bell_matrix<THAT_MODE, T, I>& that_, bool transfer = true)
    {
        ROCSPARSE_CLIENTS_ROUTINE_TRACE;
        this->define(that_.m, that_.n, that_.ell_cols, that_.bdim, that_.base);
        if(transfer)
        {
            this->transfer_from(that_);
        }
        return *this;
    }

    template <memory_mode::value_t THAT_MODE>
    void transfer_from(const bell_matrix<THAT_MODE, T, I>& that)
    {
        ROCSPARSE_CLIENTS_ROUTINE_TRACE;

        CHECK_HIP_THROW_ERROR((this->m == that.m && this->n == that.n && this->bdim == that.bdim
                               && this->ell_cols == that.ell_cols && this->base == that.base)
                                  ? hipSuccess
                                  : hipErrorInvalidValue);

        this->ind.transfer_from(that.ind);
        this->val.transfer_from(that.val);
    };

    void define(I m_, I n_, I ell_cols_, I bdim_, rocsparse_index_base base_)
    {
        ROCSPARSE_CLIENTS_ROUTINE_TRACE;
        if((this->m != m_) || (this->ell_cols != ell_cols_) || (this->bdim != bdim_))
        {
            this->ind.resize(ind_size(m_, ell_cols_, bdim_));
            this->val.resize(val_size(m_, ell_cols_));
        }

        this->m        = m_;
        this->n        = n_;
        this->ell_cols = ell_cols_;
        this->bdim     = bdim_;
        this->base     = base_;
    }

    template <memory_mode::value_t THAT_MODE>
    void near_check(const bell_matrix<THAT_MODE, T, I>& that_,
                    floating_data_t<T>                  tol = default_tolerance<T>::value) const
    {
        ROCSPARSE_CLIENTS_ROUTINE_TRACE;

        switch(MODE)
        {
        case memory_mode::device:
        {
            bell_matrix<memory_mode::host, T, I> on_host(*this);
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
                unit_check_scalar(this->ell_cols, that_.ell_cols);
                unit_check_enum(this->base, that_.base);
                unit_check_scalar(this->bdim, that_.bdim);

                this->ind.unit_check(that_.ind);
                this->val.near_check(that_.val, tol);

                break;
            }
            case memory_mode::device:
            {
                bell_matrix<memory_mode::host, T, I> that(that_);
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

        std::cout << "INFO BELL" << std::endl;
        std::cout << " m        : " << this->m << std::endl;
        std::cout << " n        : " << this->n << std::endl;
        std::cout << " ell_cols : " << this->ell_cols << std::endl;
        std::cout << " bdim     : " << this->bdim << std::endl;
        std::cout << " base     : " << this->base << std::endl;
    }
};

template <typename T, typename I = rocsparse_int>
using host_bell_matrix = bell_matrix<memory_mode::host, T, I>;
template <typename T, typename I = rocsparse_int>
using device_bell_matrix = bell_matrix<memory_mode::device, T, I>;
template <typename T, typename I = rocsparse_int>
using managed_bell_matrix = bell_matrix<memory_mode::managed, T, I>;

#endif // ROCSPARSE_MATRIX_BELL_HPP
