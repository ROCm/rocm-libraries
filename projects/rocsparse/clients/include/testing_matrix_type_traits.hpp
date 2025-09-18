/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the Software), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

#include "rocsparse_matrix.hpp"

template <rocsparse_format FORMAT, typename I, typename J, typename T>
struct testing_matrix_type_traits;

//
// TRAITS FOR CSR FORMAT.
//
template <typename I, typename J, typename T>
struct testing_matrix_type_traits<rocsparse_format_csr, I, J, T>
{
    template <typename U>
    using host_sparse_matrix = host_csr_matrix<U, I, J>;
    template <typename U>
    using device_sparse_matrix = device_csr_matrix<U, I, J>;
};

//
// TRAITS FOR CSC FORMAT.
//
template <typename I, typename J, typename T>
struct testing_matrix_type_traits<rocsparse_format_csc, I, J, T>
{
    template <typename U>
    using host_sparse_matrix = host_csc_matrix<U, I, J>;
    template <typename U>
    using device_sparse_matrix = device_csc_matrix<U, I, J>;
};

//
// TRAITS FOR BSR FORMAT.
//
template <typename I, typename J, typename T>
struct testing_matrix_type_traits<rocsparse_format_bsr, I, J, T>
{
    template <typename U>
    using host_sparse_matrix = host_gebsr_matrix<U, I, J>;
    template <typename U>
    using device_sparse_matrix = device_gebsr_matrix<U, I, J>;
};

//
// TRAITS FOR COO FORMAT.
//
template <typename I, typename T>
struct testing_matrix_type_traits<rocsparse_format_coo, I, I, T>
{
    template <typename U>
    using host_sparse_matrix = host_coo_matrix<U, I>;
    template <typename U>
    using device_sparse_matrix = device_coo_matrix<U, I>;
};

//
// TRAITS FOR COO AOS FORMAT.
//
template <typename I, typename T>
struct testing_matrix_type_traits<rocsparse_format_coo_aos, I, I, T>
{
    template <typename U>
    using host_sparse_matrix = host_coo_aos_matrix<U, I>;
    template <typename U>
    using device_sparse_matrix = device_coo_aos_matrix<U, I>;
};

//
// TRAITS FOR ELL FORMAT.
//
template <typename I, typename T>
struct testing_matrix_type_traits<rocsparse_format_ell, I, I, T>
{
    template <typename U>
    using host_sparse_matrix = host_ell_matrix<U, I>;
    template <typename U>
    using device_sparse_matrix = device_ell_matrix<U, I>;
};
