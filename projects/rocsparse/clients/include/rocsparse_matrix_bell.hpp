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

template <memory_mode::value_t MODE, typename T, typename I = rocsparse_int>
struct bell_matrix
{
    template <typename S>
    using array_t = typename memory_traits<MODE>::template array_t<S>;

    I                      m{};
    I                      n{};
    I                      width{};
    rocsparse_direction    bdir{};
    I                      bdim{1};
    rocsparse_index_base   base{};
    rocsparse_storage_mode storage_mode{rocsparse_storage_mode_sorted};
    array_t<I>             ind{};
    array_t<T>             val{};

    bell_matrix(){};
    ~bell_matrix(){};

    bell_matrix(
        I m_, I n_, I width_, rocsparse_direction bdir_, I bdim_, rocsparse_index_base base_)
        : m(m_)
        , n(n_)
        , width(width_)
        , bdir(bdir_)
        , bdim(bdim_)
        , base(base_)
        , ind(width_ * m_)
        , val(width_ * m_ * bdim_ * bdim_){};

    explicit bell_matrix(const bell_matrix<MODE, T, I>& that_, bool transfer = true)
        : bell_matrix<MODE, T, I>(that_.m, that_.n, that_.width, that_.bdir, that_.bdim, that_.base)
    {
        ROCSPARSE_CLIENTS_ROUTINE_TRACE;

        if(transfer)
        {
            this->transfer_from(that_);
        }
    }

    template <memory_mode::value_t THAT_MODE>
    explicit bell_matrix(const bell_matrix<THAT_MODE, T, I>& that_, bool transfer = true)
        : bell_matrix<MODE, T, I>(that_.m, that_.n, that_.width, that_.bdir, that_.bdim, that_.base)
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
        this->define(that_.m, that_.n, that_.width, that_.bdir, that_.bdim, that_.base);
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
                               && this->width == that.width && this->base == that.base)
                                  ? hipSuccess
                                  : hipErrorInvalidValue);

        this->ind.transfer_from(that.ind);
        this->val.transfer_from(that.val);
    };

    void
        define(I m_, I n_, I width_, rocsparse_direction bdir_, I bdim_, rocsparse_index_base base_)
    {
        ROCSPARSE_CLIENTS_ROUTINE_TRACE;
        if((this->m != m_) || (this->width != width_) || (this->bdim != bdim_))
        {
            this->ind.resize(int64_t(m_) * width_);
            this->val.resize(int64_t(m_) * width_ * bdim_ * bdim_);
        }

        this->m     = m_;
        this->n     = n_;
        this->width = width_;
        this->bdim  = bdim_;
        this->bdir  = bdir_;
        this->base  = base_;
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
                unit_check_scalar(this->width, that_.width);
                unit_check_enum(this->base, that_.base);
                unit_check_enum(this->bdir, that_.bdir);
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
        std::cout << " m     : " << this->m << std::endl;
        std::cout << " n     : " << this->n << std::endl;
        std::cout << " width : " << this->width << std::endl;
        std::cout << " bdir  : " << this->bdir << std::endl;
        std::cout << " bdim  : " << this->bdim << std::endl;
        std::cout << " base  : " << this->base << std::endl;
    }
};

template <typename T, typename I = rocsparse_int>
using host_bell_matrix = bell_matrix<memory_mode::host, T, I>;
template <typename T, typename I = rocsparse_int>
using device_bell_matrix = bell_matrix<memory_mode::device, T, I>;
template <typename T, typename I = rocsparse_int>
using managed_bell_matrix = bell_matrix<memory_mode::managed, T, I>;

#endif // ROCSPARSE_MATRIX_BELL_HPP
