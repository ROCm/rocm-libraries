/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2021 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef ROCSPARSE_MATRIX_HPP
#define ROCSPARSE_MATRIX_HPP

#include "rocsparse_check.hpp"
#include "rocsparse_matrix_dense.hpp"

template <typename T>
struct device_scalar;
template <typename T>
struct host_scalar;
template <typename T>
struct managed_scalar;

template <typename T>
struct host_scalar : public host_dense_matrix<T>
{
    host_scalar()
        : host_dense_matrix<T>(1, 1){};
    explicit host_scalar(const T& value)
        : host_dense_matrix<T>(1, 1)
    {
        T* p = *this;
        p[0] = value;
    };

    explicit host_scalar(const device_scalar<T>& that)
        : host_dense_matrix<T>(1, 1)
    {
        this->transfer_from(that);
    }

    explicit host_scalar(const managed_scalar<T>& that)
        : host_dense_matrix<T>(1, 1)
    {
        this->transfer_from(that);
    }

    inline host_scalar<T>& operator=(const T& that)
    {
        T* p = *this;
        p[0] = that;
        return *this;
    };
};

template <typename T>
struct device_scalar : public device_dense_matrix<T>
{
    device_scalar()
        : device_dense_matrix<T>(1, 1){};
    explicit device_scalar(const host_scalar<T>& that)
        : device_dense_matrix<T>(1, 1)
    {
        this->transfer_from(that);
    }
    explicit device_scalar(const managed_scalar<T>& that)
        : device_dense_matrix<T>(1, 1)
    {
        this->transfer_from(that);
    }
};

template <typename T>
struct managed_scalar : public managed_dense_matrix<T>
{
    managed_scalar()
        : managed_dense_matrix<T>(1, 1){};

    explicit managed_scalar(const host_scalar<T>& that)
        : managed_dense_matrix<T>(1, 1)
    {
        this->transfer_from(that);
    }

    explicit managed_scalar(const device_scalar<T>& that)
        : managed_dense_matrix<T>(1, 1)
    {
        this->transfer_from(that);
    }
};

#include "rocsparse_matrix_coo.hpp"
#include "rocsparse_matrix_coo_aos.hpp"
#include "rocsparse_matrix_csx.hpp"
#include "rocsparse_matrix_ell.hpp"
#include "rocsparse_matrix_gebsx.hpp"

#endif // ROCSPARSE_MATRIX_HPP.
