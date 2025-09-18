/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_exporter_rocsparseio.hpp"

template <typename X, typename Y>
rocsparse_status rocsparse_type_conversion(const X& x, Y& y);

template <typename X, typename Y>
inline rocsparse_status rocsparse2rocsparseio_convert(const X& x, Y& y);

template <>
inline rocsparse_status rocsparse2rocsparseio_convert(const rocsparse_order& x,
                                                      rocsparseio_order&     y)
{
    switch(x)
    {
    case rocsparse_order_row:
    {
        y = rocsparseio_order_row;
        return rocsparse_status_success;
    }
    case rocsparse_order_column:
    {
        y = rocsparseio_order_column;
        return rocsparse_status_success;
    }
    }
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status rocsparse2rocsparseio_convert(const rocsparse_direction& x,
                                                      rocsparseio_direction&     y)
{
    switch(x)
    {
    case rocsparse_direction_row:
    {
        y = rocsparseio_direction_row;
        return rocsparse_status_success;
    }
    case rocsparse_direction_column:
    {
        y = rocsparseio_direction_column;
        return rocsparse_status_success;
    }
    }
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status rocsparse2rocsparseio_convert(const rocsparse_index_base& x,
                                                      rocsparseio_index_base&     y)
{
    switch(x)
    {
    case rocsparse_index_base_zero:
    {
        y = rocsparseio_index_base_zero;
        return rocsparse_status_success;
    }
    case rocsparse_index_base_one:
    {
        y = rocsparseio_index_base_one;
        return rocsparse_status_success;
    }
    }
    return rocsparse_status_invalid_value;
}

template <typename T>
inline rocsparseio_type rocsparseio_type_convert();

template <>
inline rocsparseio_type rocsparseio_type_convert<int32_t>()
{
    return rocsparseio_type_int32;
};
template <>
inline rocsparseio_type rocsparseio_type_convert<int64_t>()
{
    return rocsparseio_type_int64;
};
template <>
inline rocsparseio_type rocsparseio_type_convert<float>()
{
    return rocsparseio_type_float32;
};
template <>
inline rocsparseio_type rocsparseio_type_convert<double>()
{
    return rocsparseio_type_float64;
};
template <>
inline rocsparseio_type rocsparseio_type_convert<rocsparse_float_complex>()
{
    return rocsparseio_type_complex32;
};
template <>
inline rocsparseio_type rocsparseio_type_convert<rocsparse_double_complex>()
{
    return rocsparseio_type_complex64;
};

rocsparse_exporter_rocsparseio::~rocsparse_exporter_rocsparseio()
{
    const char* env = getenv("GTEST_LISTENER");
    if(!env || strcmp(env, "NO_PASS_LINE_IN_LOG"))
    {
        std::cout << "Export done." << std::endl;
    }

    auto istatus = rocsparseio_close(this->m_handle);
    if(istatus != rocsparseio_status_success)
    {
    }
}

rocsparse_exporter_rocsparseio::rocsparse_exporter_rocsparseio(const std::string& filename_)
    : m_filename(filename_)
{

    const char* env = getenv("GTEST_LISTENER");
    if(!env || strcmp(env, "NO_PASS_LINE_IN_LOG"))
    {
        std::cout << "Opening file '" << this->m_filename << "' ... " << std::endl;
    }

    rocsparseio_status istatus;
    istatus = rocsparseio_open(&this->m_handle, rocsparseio_rwmode_write, this->m_filename.c_str());
    if(istatus != rocsparseio_status_success)
    {
        std::cerr << "Problem with rocsparseio_open" << std::endl;
        throw rocsparse_status_internal_error;
    }
}

template <typename T, typename I, typename J>
rocsparse_status rocsparse_exporter_rocsparseio::write_sparse_csx(rocsparse_direction dir_,
                                                                  J                   m_,
                                                                  J                   n_,
                                                                  I                   nnz_,
                                                                  const I* __restrict__ ptr_,
                                                                  const J* __restrict__ ind_,
                                                                  const T* __restrict__ val_,
                                                                  rocsparse_index_base base_)
{
    const rocsparseio_type ptr_type = rocsparseio_type_convert<I>();
    const rocsparseio_type ind_type = rocsparseio_type_convert<J>();
    const rocsparseio_type val_type = rocsparseio_type_convert<T>();

    rocsparseio_direction  dir;
    uint64_t               m;
    uint64_t               n;
    uint64_t               nnz;
    rocsparseio_index_base base;

    rocsparse_status status;

    status = rocsparse2rocsparseio_convert(dir_, dir);
    if(status != rocsparse_status_success)
    {
        return status;
    }

    status = rocsparse2rocsparseio_convert(base_, base);
    if(status != rocsparse_status_success)
    {
        return status;
    }

    status = rocsparse_type_conversion(m_, m);
    if(status != rocsparse_status_success)
    {
        return status;
    }

    status = rocsparse_type_conversion(n_, n);

    if(status != rocsparse_status_success)
    {
        return status;
    }

    status = rocsparse_type_conversion(nnz_, nnz);

    if(status != rocsparse_status_success)
    {
        return status;
    }

    const char*        name      = nullptr;
    rocsparseio_status io_status = rocsparseio_write_sparse_csx(
        this->m_handle, dir, m, n, nnz, ptr_type, ptr_, ind_type, ind_, val_type, val_, base, name);
    if(io_status != rocsparseio_status_success)
    {
        return rocsparse_status_internal_error;
    }
    return rocsparse_status_success;
}

template <typename T, typename I, typename J>
rocsparse_status rocsparse_exporter_rocsparseio::write_sparse_gebsx(rocsparse_direction dir_,
                                                                    rocsparse_direction dirb_,
                                                                    J                   mb_,
                                                                    J                   nb_,
                                                                    I                   nnzb_,
                                                                    J block_dim_row_,
                                                                    J block_dim_column_,
                                                                    const I* __restrict__ ptr_,
                                                                    const J* __restrict__ ind_,
                                                                    const T* __restrict__ val_,
                                                                    rocsparse_index_base base_)
{

    const rocsparseio_type ptr_type = rocsparseio_type_convert<I>();
    const rocsparseio_type ind_type = rocsparseio_type_convert<J>();
    const rocsparseio_type val_type = rocsparseio_type_convert<T>();

    rocsparseio_direction  dir;
    rocsparseio_direction  dirb;
    uint64_t               mb;
    uint64_t               nb;
    uint64_t               nnzb;
    uint64_t               block_dim_row;
    uint64_t               block_dim_column;
    rocsparseio_index_base base;

    rocsparse_status status;

    status = rocsparse2rocsparseio_convert(dir_, dir);
    if(status != rocsparse_status_success)
    {
        return status;
    }

    status = rocsparse2rocsparseio_convert(dirb_, dirb);
    if(status != rocsparse_status_success)
    {
        return status;
    }

    status = rocsparse2rocsparseio_convert(base_, base);
    if(status != rocsparse_status_success)
    {
        return status;
    }

    status = rocsparse_type_conversion(mb_, mb);
    if(status != rocsparse_status_success)
    {
        return status;
    }

    status = rocsparse_type_conversion(nb_, nb);

    if(status != rocsparse_status_success)
    {
        return status;
    }

    status = rocsparse_type_conversion(nnzb_, nnzb);

    if(status != rocsparse_status_success)
    {
        return status;
    }

    status = rocsparse_type_conversion(block_dim_row_, block_dim_row);

    if(status != rocsparse_status_success)
    {
        return status;
    }

    status = rocsparse_type_conversion(block_dim_column_, block_dim_column);

    if(status != rocsparse_status_success)
    {
        return status;
    }

    const char*        name      = nullptr;
    rocsparseio_status io_status = rocsparseio_write_sparse_gebsx(this->m_handle,
                                                                  dir,
                                                                  dirb,
                                                                  mb,
                                                                  nb,
                                                                  nnzb,
                                                                  block_dim_row,
                                                                  block_dim_column,
                                                                  ptr_type,
                                                                  ptr_,
                                                                  ind_type,
                                                                  ind_,
                                                                  val_type,
                                                                  val_,
                                                                  base,
                                                                  name);
    if(io_status != rocsparseio_status_success)
    {
        return rocsparse_status_internal_error;
    }
    return rocsparse_status_success;
}

template <typename T, typename I>
rocsparse_status
    rocsparse_exporter_rocsparseio::write_dense_vector(I nmemb_, const T* __restrict__ x_, I incx_)
{

    const rocsparseio_type val_type = rocsparseio_type_convert<T>();
    uint64_t               nmemb, incx;
    rocsparse_status       status;
    status = rocsparse_type_conversion(nmemb_, nmemb);
    if(status != rocsparse_status_success)
    {
        return status;
    }

    status = rocsparse_type_conversion(incx_, incx);
    if(status != rocsparse_status_success)
    {
        return status;
    }

    const char*        name = nullptr;
    rocsparseio_status io_status
        = rocsparseio_write_dense_vector(this->m_handle, val_type, nmemb, x_, incx, name);
    if(io_status != rocsparseio_status_success)
    {
        return rocsparse_status_internal_error;
    }
    return rocsparse_status_success;
}

template <typename T, typename I>
rocsparse_status rocsparse_exporter_rocsparseio::write_dense_matrix(
    rocsparse_order order_, I m_, I n_, const T* __restrict__ x_, I ld_)
{

    rocsparseio_order      order;
    uint64_t               m, n, ld;
    rocsparse_status       status;
    const rocsparseio_type val_type = rocsparseio_type_convert<T>();

    status = rocsparse2rocsparseio_convert(order_, order);
    if(status != rocsparse_status_success)
    {
        return status;
    }

    status = rocsparse_type_conversion(m_, m);
    if(status != rocsparse_status_success)
    {
        return status;
    }

    status = rocsparse_type_conversion(n_, n);

    if(status != rocsparse_status_success)
    {
        return status;
    }

    status = rocsparse_type_conversion(ld_, ld);

    if(status != rocsparse_status_success)
    {
        return status;
    }

    const char*        name = nullptr;
    rocsparseio_status io_status
        = rocsparseio_write_dense_matrix(this->m_handle, order, m, n, val_type, x_, ld, name);
    if(io_status != rocsparseio_status_success)
    {
        return rocsparse_status_internal_error;
    }
    return rocsparse_status_success;
}

template <typename T, typename I>
rocsparse_status rocsparse_exporter_rocsparseio::write_sparse_coo(I m_,
                                                                  I n_,
                                                                  I nnz_,
                                                                  const I* __restrict__ row_ind_,
                                                                  const I* __restrict__ col_ind_,
                                                                  const T* __restrict__ val_,
                                                                  rocsparse_index_base base_)
{

    const rocsparseio_type ind_type = rocsparseio_type_convert<I>();
    const rocsparseio_type val_type = rocsparseio_type_convert<T>();

    uint64_t               m, n, nnz;
    rocsparseio_index_base base;

    rocsparse_status status;
    status = rocsparse2rocsparseio_convert(base_, base);
    if(status != rocsparse_status_success)
    {
        return status;
    }

    status = rocsparse_type_conversion(m_, m);
    if(status != rocsparse_status_success)
    {
        return status;
    }

    status = rocsparse_type_conversion(n_, n);

    if(status != rocsparse_status_success)
    {
        return status;
    }
    status = rocsparse_type_conversion(nnz_, nnz);

    if(status != rocsparse_status_success)
    {
        return status;
    }

    const char*        name      = nullptr;
    rocsparseio_status io_status = rocsparseio_write_sparse_coo(this->m_handle,
                                                                m,
                                                                n,
                                                                nnz,
                                                                ind_type,
                                                                row_ind_,
                                                                ind_type,
                                                                col_ind_,
                                                                val_type,
                                                                val_,
                                                                base,
                                                                name);
    if(io_status != rocsparseio_status_success)
    {
        return rocsparse_status_internal_error;
    }
    return rocsparse_status_success;
}

#define INSTANTIATE_TIJ(T, I, J)                                                  \
    template rocsparse_status rocsparse_exporter_rocsparseio::write_sparse_csx(   \
        rocsparse_direction,                                                      \
        J,                                                                        \
        J,                                                                        \
        I,                                                                        \
        const I* __restrict__,                                                    \
        const J* __restrict__,                                                    \
        const T* __restrict__,                                                    \
        rocsparse_index_base);                                                    \
    template rocsparse_status rocsparse_exporter_rocsparseio::write_sparse_gebsx( \
        rocsparse_direction,                                                      \
        rocsparse_direction,                                                      \
        J,                                                                        \
        J,                                                                        \
        I,                                                                        \
        J,                                                                        \
        J,                                                                        \
        const I* __restrict__,                                                    \
        const J* __restrict__,                                                    \
        const T* __restrict__,                                                    \
        rocsparse_index_base)

#define INSTANTIATE_TI(T, I)                                                      \
    template rocsparse_status rocsparse_exporter_rocsparseio::write_dense_vector( \
        I, const T* __restrict__, I);                                             \
    template rocsparse_status rocsparse_exporter_rocsparseio::write_dense_matrix( \
        rocsparse_order, I, I, const T* __restrict__, I);                         \
    template rocsparse_status rocsparse_exporter_rocsparseio::write_sparse_coo(   \
        I,                                                                        \
        I,                                                                        \
        I,                                                                        \
        const I* __restrict__,                                                    \
        const I* __restrict__,                                                    \
        const T* __restrict__,                                                    \
        rocsparse_index_base)

INSTANTIATE_TIJ(float, int32_t, int32_t);
INSTANTIATE_TIJ(float, int64_t, int32_t);
INSTANTIATE_TIJ(float, int64_t, int64_t);

INSTANTIATE_TIJ(double, int32_t, int32_t);
INSTANTIATE_TIJ(double, int64_t, int32_t);
INSTANTIATE_TIJ(double, int64_t, int64_t);

INSTANTIATE_TIJ(rocsparse_float_complex, int32_t, int32_t);
INSTANTIATE_TIJ(rocsparse_float_complex, int64_t, int32_t);
INSTANTIATE_TIJ(rocsparse_float_complex, int64_t, int64_t);

INSTANTIATE_TIJ(rocsparse_double_complex, int32_t, int32_t);
INSTANTIATE_TIJ(rocsparse_double_complex, int64_t, int32_t);
INSTANTIATE_TIJ(rocsparse_double_complex, int64_t, int64_t);

INSTANTIATE_TI(float, int32_t);
INSTANTIATE_TI(float, int64_t);

INSTANTIATE_TI(double, int32_t);
INSTANTIATE_TI(double, int64_t);

INSTANTIATE_TI(rocsparse_float_complex, int32_t);
INSTANTIATE_TI(rocsparse_float_complex, int64_t);

INSTANTIATE_TI(rocsparse_double_complex, int32_t);
INSTANTIATE_TI(rocsparse_double_complex, int64_t);
