/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2023 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef ROCSPARSE_MATRIX_GEBSX_HPP
#define ROCSPARSE_MATRIX_GEBSX_HPP

#include "rocsparse_vector.hpp"

template <memory_mode::value_t MODE,
          rocsparse_direction  direction_,
          typename T,
          typename I,
          typename J>
struct gebsx_matrix
{
    template <typename S>
    using array_t = typename memory_traits<MODE>::template array_t<S>;

    static constexpr rocsparse_direction dir = direction_;

    J                      mb{};
    J                      nb{};
    I                      nnzb{};
    rocsparse_direction    block_direction{};
    J                      row_block_dim{1};
    J                      col_block_dim{1};
    rocsparse_index_base   base{};
    rocsparse_storage_mode storage_mode{rocsparse_storage_mode_sorted};
    array_t<I>             ptr{};
    array_t<J>             ind{};
    array_t<T>             val{};

    gebsx_matrix(){};
    ~gebsx_matrix(){};
    gebsx_matrix(rocsparse_direction  block_dir_,
                 J                    mb_,
                 J                    nb_,
                 I                    nnzb_,
                 J                    row_block_dim_,
                 J                    col_block_dim_,
                 rocsparse_index_base base_)
        : mb(mb_)
        , nb(nb_)
        , nnzb(nnzb_)
        , block_direction(block_dir_)
        , row_block_dim(row_block_dim_)
        , col_block_dim(col_block_dim_)
        , base(base_)
        , ptr((rocsparse_direction_row == direction_) ? ((mb > 0) ? (mb + 1) : 0)
                                                      : ((nb > 0) ? (nb + 1) : 0))
        , ind(nnzb)
        , val(size_t(nnzb) * row_block_dim * col_block_dim){};

    template <memory_mode::value_t THAT_MODE>
    explicit gebsx_matrix(const gebsx_matrix<THAT_MODE, direction_, T, I, J>& that_,
                          bool                                                transfer = true)
        : gebsx_matrix<MODE, direction_, T, I, J>(that_.block_direction,
                                                  that_.mb,
                                                  that_.nb,
                                                  that_.nnzb,
                                                  that_.row_block_dim,
                                                  that_.col_block_dim,
                                                  that_.base)
    {
        if(transfer)
        {
            this->transfer_from(that_);
        }
    }

    explicit gebsx_matrix(const gebsx_matrix<MODE, direction_, T, I, J>& that_,
                          bool                                           transfer = true)
        : gebsx_matrix<MODE, direction_, T, I, J>(that_.block_direction,
                                                  that_.mb,
                                                  that_.nb,
                                                  that_.nnzb,
                                                  that_.row_block_dim,
                                                  that_.col_block_dim,
                                                  that_.base)
    {
        if(transfer)
        {
            this->transfer_from(that_);
        }
    }

    void unit_check(gebsx_matrix<memory_mode::host, direction_, T, I, J>& that,
                    bool                                                  check_values = true)
    {

        unit_check_enum(this->block_direction, that.block_direction);
        unit_check_scalar<J>(this->mb, that.mb);
        unit_check_scalar<J>(this->nb, that.nb);
        unit_check_scalar<I>(this->nnzb, that.nnzb);
        unit_check_scalar<J>(this->row_block_dim, that.row_block_dim);
        unit_check_scalar<J>(this->col_block_dim, that.col_block_dim);
        switch(direction_)
        {
        case rocsparse_direction_row:
        {
            if(this->mb > 0)
            {
                this->ptr.unit_check(that.ptr);
            }
            break;
        }
        case rocsparse_direction_column:
        {
            if(this->nb > 0)
            {
                this->ptr.unit_check(that.ptr);
            }
            break;
        }
        }

        if(this->nnzb > 0)
        {
            this->ind.unit_check(that.ind);
        }

        if(check_values)
        {
            if(this->nnzb > 0)
            {
                this->val.unit_check(that.val);
            }
        }
    }

    template <memory_mode::value_t THAT_MODE>
    void near_check(const gebsx_matrix<THAT_MODE, direction_, T, I, J>& that_,
                    floating_data_t<T> tol = default_tolerance<T>::value) const
    {
        switch(MODE)
        {
        case memory_mode::device:
        {
            gebsx_matrix<memory_mode::host, direction_, T, I, J> on_host(*this);
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
                unit_check_enum(this->block_direction, that_.block_direction);
                unit_check_scalar<J>(this->mb, that_.mb);
                unit_check_scalar<J>(this->nb, that_.nb);
                unit_check_scalar<I>(this->nnzb, that_.nnzb);
                unit_check_scalar<J>(this->row_block_dim, that_.row_block_dim);
                unit_check_scalar<J>(this->col_block_dim, that_.col_block_dim);

                switch(direction_)
                {
                case rocsparse_direction_row:
                {
                    if(this->mb > 0)
                    {
                        this->ptr.unit_check(that_.ptr);
                    }
                    break;
                }
                case rocsparse_direction_column:
                {
                    if(this->nb > 0)
                    {
                        this->ptr.unit_check(that_.ptr);
                    }
                    break;
                }
                }
                if(this->nnzb > 0)
                {
                    this->ind.unit_check(that_.ind);
                    this->val.near_check(that_.val, tol);
                }

                break;
            }
            case memory_mode::device:
            {
                gebsx_matrix<memory_mode::host, direction_, T, I, J> that(that_);
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
        std::cout << "INFO GEBSX " << std::endl;
        std::cout << " dir            : " << direction_ << std::endl;
        std::cout << " mb             : " << this->mb << std::endl;
        std::cout << " nb             : " << this->nb << std::endl;
        std::cout << " nnzb           : " << this->nnzb << std::endl;
        std::cout << " dirb           : " << this->block_direction << std::endl;
        std::cout << " row_block_dim  : " << this->row_block_dim << std::endl;
        std::cout << " col_block_dim  : " << this->col_block_dim << std::endl;
        std::cout << " base           : " << this->base << std::endl;
    }

    bool is_invalid() const
    {
        if(this->mb < 0)
            return true;
        if(this->nb < 0)
            return true;
        if(this->nnzb < 0)
            return true;
        if(this->nnzb > this->mb * this->nb)
            return true;
        if(this->row_block_dim <= 0)
            return true;
        if(this->col_block_dim <= 0)
            return true;
        switch(direction_)
        {
        case rocsparse_direction_row:
        {
            if(this->mb > 0)
            {
                if(this->ptr.size() != (this->mb + 1))
                {
                    return true;
                }
            }
            break;
        }
        case rocsparse_direction_column:
        {
            if(this->nb > 0)
            {
                if(this->ptr.size() != (this->nb + 1))
                {
                    return true;
                }
            }
            break;
        }
        }

        return false;
    };

    void define(rocsparse_direction  block_dir_,
                J                    mb_,
                J                    nb_,
                I                    nnzb_,
                J                    row_block_dim_,
                J                    col_block_dim_,
                rocsparse_index_base base_)
    {
        if(block_dir_ != this->block_direction)
        {
            this->block_direction = block_dir_;
        }

        if(row_block_dim_ != this->row_block_dim)
        {
            this->row_block_dim = row_block_dim_;
        }

        if(col_block_dim_ != this->col_block_dim)
        {
            this->col_block_dim = col_block_dim_;
        }

        if(mb_ != this->mb)
        {
            this->mb = mb_;
            if(direction_ == rocsparse_direction_row)
            {
                this->ptr.resize((this->mb > 0) ? (this->mb + 1) : 0);
            }
        }

        if(nb_ != this->nb)
        {
            this->nb = nb_;
            if(direction_ == rocsparse_direction_column)
            {
                this->ptr.resize((this->nb > 0) ? (this->nb + 1) : 0);
            }
        }

        if(nnzb_ != this->nnzb)
        {
            this->nnzb = nnzb_;
            this->ind.resize(this->nnzb);
            this->val.resize(size_t(this->nnzb) * this->col_block_dim * this->row_block_dim);
        }

        if(base_ != this->base)
        {
            this->base = base_;
        }
    };

    template <memory_mode::value_t THAT_MODE>
    void transfer_from(const gebsx_matrix<THAT_MODE, direction_, T, I, J>& that)
    {
        CHECK_HIP_THROW_ERROR((this->mb == that.mb && this->nb == that.nb && this->nnzb == that.nnzb
                               && this->block_direction == that.block_direction
                               && this->row_block_dim == that.row_block_dim
                               && this->col_block_dim == that.col_block_dim
                               && this->base == that.base)
                                  ? hipSuccess
                                  : hipErrorInvalidValue);

        switch(direction_)
        {
        case rocsparse_direction_row:
        {
            if(this->mb > 0)
            {
                this->ptr.transfer_from(that.ptr);
            }
            break;
        }
        case rocsparse_direction_column:
        {
            if(this->nb > 0)
            {
                this->ptr.transfer_from(that.ptr);
            }
            break;
        }
        }

        if(this->nnzb > 0)
        {
            this->ind.transfer_from(that.ind);
            this->val.transfer_from(that.val);
        }
    };

    template <memory_mode::value_t THAT_MODE>
    gebsx_matrix<MODE, direction_, T, I, J>&
        operator()(const gebsx_matrix<THAT_MODE, direction_, T, I, J>& that, bool transfer = true)
    {
        this->define(that.block_direction,
                     that.mb,
                     that.nb,
                     that.nnzb,
                     that.row_block_dim,
                     that.col_block_dim,
                     that.base);
        if(transfer)
        {
            this->transfer_from(that);
        }
        return *this;
    };
};

template <rocsparse_direction direction_, typename T, typename I, typename J>
using host_gebsx_matrix = gebsx_matrix<memory_mode::host, direction_, T, I, J>;
template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
using host_gebsr_matrix = host_gebsx_matrix<rocsparse_direction_row, T, I, J>;
template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
using host_gebsc_matrix = host_gebsx_matrix<rocsparse_direction_column, T, I, J>;

template <rocsparse_direction direction_, typename T, typename I, typename J>
using device_gebsx_matrix = gebsx_matrix<memory_mode::device, direction_, T, I, J>;
template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
using device_gebsr_matrix = device_gebsx_matrix<rocsparse_direction_row, T, I, J>;
template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
using device_gebsc_matrix = device_gebsx_matrix<rocsparse_direction_column, T, I, J>;

template <rocsparse_direction direction_, typename T, typename I, typename J>
using managed_gebsx_matrix = gebsx_matrix<memory_mode::managed, direction_, T, I, J>;
template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
using managed_gebsr_matrix = managed_gebsx_matrix<rocsparse_direction_row, T, I, J>;
template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
using managed_gebsc_matrix = managed_gebsx_matrix<rocsparse_direction_column, T, I, J>;

#endif // ROCSPARSE_MATRIX_GEBSX_HPP
