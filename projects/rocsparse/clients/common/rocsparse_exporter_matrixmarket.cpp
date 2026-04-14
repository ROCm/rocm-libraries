/*! \file */
/* ************************************************************************
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_exporter_matrixmarket.hpp"
template <typename X, typename Y>
rocsparse_status rocsparse_type_conversion(const X& x, Y& y);

rocsparse_exporter_matrixmarket::~rocsparse_exporter_matrixmarket()
{
    const char* env = getenv("GTEST_LISTENER");
    if(!env || strcmp(env, "NO_PASS_LINE_IN_LOG"))
    {
        std::cout << "Export done." << std::endl;
    }
}

rocsparse_exporter_matrixmarket::rocsparse_exporter_matrixmarket(const std::string& filename_)
    : m_filename(filename_)
{
    const char* env = getenv("GTEST_LISTENER");
    if(!env || strcmp(env, "NO_PASS_LINE_IN_LOG"))
    {
        std::cout << "Opening file '" << this->m_filename << "' ... " << std::endl;
    }
}

template <typename T, typename I, typename J>
rocsparse_status rocsparse_exporter_matrixmarket::write_sparse_csx(rocsparse_direction dir_,
                                                                   J                   m_,
                                                                   J                   n_,
                                                                   I                   nnz_,
                                                                   const I* __restrict__ ptr_,
                                                                   const J* __restrict__ ind_,
                                                                   const T* __restrict__ val_,
                                                                   rocsparse_index_base base_)
{
    std::ofstream out(this->m_filename);
    if(!out.is_open())
    {
        return rocsparse_status_internal_error;
    }
    if(std::is_same<T, double>() || std::is_same<T, rocsparse_double_complex>())
    {
        out.precision(15);
        out.setf(std::ios::scientific);
    }
    else
    {
        out.precision(7);
        out.setf(std::ios::scientific);
    }

    out << "%%MatrixMarket matrix coordinate ";
    if(std::is_same<T, rocsparse_float_complex>() || std::is_same<T, rocsparse_double_complex>())
        out << "complex";
    else
        out << "real";
    out << " general" << std::endl;
    out << m_ << " " << n_ << " " << nnz_ << std::endl;
    switch(dir_)
    {
    case rocsparse_direction_row:
    {
        for(J i = 0; i < m_; ++i)
        {
            for(I at = ptr_[i] - base_; at < ptr_[i + 1] - base_; ++at)
            {
                J j = ind_[at] - base_;
                T x = val_[at];
                out << (i + 1) << " " << (j + 1);
                if(std::is_same<T, rocsparse_float_complex>()
                   || std::is_same<T, rocsparse_double_complex>())
                {
                    out << " " << std::real(x) << " " << std::imag(x);
                }
                else
                {
                    out << " " << x;
                }
                out << std::endl;
            }
        }
        out.close();
        return rocsparse_status_success;
    }
    case rocsparse_direction_column:
    {
        for(J j = 0; j < n_; ++j)
        {
            for(I at = ptr_[j] - base_; at < ptr_[j + 1] - base_; ++at)
            {
                J i = ind_[at] - base_;
                T x = val_[at];
                out << (i + 1) << " " << (j + 1);
                if(std::is_same<T, rocsparse_float_complex>()
                   || std::is_same<T, rocsparse_double_complex>())
                {
                    out << " " << std::real(x) << " " << std::imag(x);
                }
                else
                {
                    out << " " << x;
                }
                out << std::endl;
            }
        }
        out.close();
        return rocsparse_status_success;
    }
    }
    return rocsparse_status_invalid_value;
}

template <typename T, typename I, typename J>
rocsparse_status rocsparse_exporter_matrixmarket::write_sparse_gebsx(rocsparse_direction dir_,
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
    std::ofstream out(this->m_filename);
    if(!out.is_open())
    {
        return rocsparse_status_internal_error;
    }
    if(std::is_same<T, double>() || std::is_same<T, rocsparse_double_complex>())
    {
        out.precision(15);
        out.setf(std::ios::scientific);
    }
    else
    {
        out.precision(7);
        out.setf(std::ios::scientific);
    }

    out << "%%MatrixMarket matrix coordinate ";
    if(std::is_same<T, rocsparse_float_complex>() || std::is_same<T, rocsparse_double_complex>())
        out << "complex";
    else
        out << "real";
    out << " general" << std::endl;
    out << mb_ * block_dim_row_ << " " << nb_ * block_dim_column_ << " "
        << nnzb_ * block_dim_row_ * block_dim_column_ << std::endl;
    switch(dir_)
    {
    case rocsparse_direction_row:
    {
        for(J ib = 0; ib < mb_; ++ib)
        {
            I i = ib * block_dim_row_;
            for(I at = ptr_[ib] - base_; at < ptr_[ib + 1] - base_; ++at)
            {
                J j = (ind_[at] - base_) * block_dim_column_;
                switch(dirb_)
                {
                case rocsparse_direction_row:
                {
                    for(J k = 0; k < block_dim_row_; ++k)
                    {
                        for(J l = 0; l < block_dim_column_; ++l)
                        {
                            auto v = val_[at * block_dim_row_ * block_dim_column_
                                          + block_dim_column_ * k + l];
                            out << (i + k) << " " << (j + l);
                            if(std::is_same<T, rocsparse_float_complex>()
                               || std::is_same<T, rocsparse_double_complex>())
                            {
                                out << " " << std::real(v) << " " << std::imag(v);
                            }
                            else
                            {
                                out << " " << v;
                            }
                            out << std::endl;
                        }
                    }
                    break;
                }
                case rocsparse_direction_column:
                {
                    for(J k = 0; k < block_dim_row_; ++k)
                    {
                        for(J l = 0; l < block_dim_column_; ++l)
                        {
                            auto v = val_[at * block_dim_row_ * block_dim_column_
                                          + block_dim_row_ * l + k];
                            out << (i + k) << " " << (j + l);
                            if(std::is_same<T, rocsparse_float_complex>()
                               || std::is_same<T, rocsparse_double_complex>())
                            {
                                out << " " << std::real(v) << " " << std::imag(v);
                            }
                            else
                            {
                                out << " " << v;
                            }
                            out << std::endl;
                        }
                    }
                    break;
                }
                }
            }
        }
        out.close();
        return rocsparse_status_success;
    }

    case rocsparse_direction_column:
    {
        for(J jb = 0; jb < nb_; ++jb)
        {
            I j = jb * block_dim_column_;
            for(I at = ptr_[jb] - base_; at < ptr_[jb + 1] - base_; ++at)
            {
                J i = (ind_[at] - base_) * block_dim_row_;
                switch(dirb_)
                {

                case rocsparse_direction_row:
                {
                    for(J k = 0; k < block_dim_row_; ++k)
                    {
                        for(J l = 0; l < block_dim_column_; ++l)
                        {
                            auto v = val_[at * block_dim_row_ * block_dim_column_
                                          + block_dim_column_ * k + l];
                            out << (i + k) << " " << (j + l);
                            if(std::is_same<T, rocsparse_float_complex>()
                               || std::is_same<T, rocsparse_double_complex>())
                            {
                                out << " " << std::real(v) << " " << std::imag(v);
                            }
                            else
                            {
                                out << " " << v;
                            }
                            out << std::endl;
                        }
                    }
                    break;
                }

                case rocsparse_direction_column:
                {
                    for(J k = 0; k < block_dim_row_; ++k)
                    {
                        for(J l = 0; l < block_dim_column_; ++l)
                        {
                            auto v = val_[at * block_dim_row_ * block_dim_column_
                                          + block_dim_row_ * l + k];
                            out << (i + k) << " " << (j + l);
                            if(std::is_same<T, rocsparse_float_complex>()
                               || std::is_same<T, rocsparse_double_complex>())
                            {
                                out << " " << std::real(v) << " " << std::imag(v);
                            }
                            else
                            {
                                out << " " << v;
                            }
                            out << std::endl;
                        }
                    }
                    break;
                }
                }
            }
        }
        out.close();
        return rocsparse_status_success;
    }
    }
    return rocsparse_status_invalid_value;

    std::cerr << "rocsparse_exporter_matrixmarket, gebsx not supported." << std::endl;
    return rocsparse_status_not_implemented;
}

template <typename T, typename I>
rocsparse_status
    rocsparse_exporter_matrixmarket::write_dense_vector(I nmemb_, const T* __restrict__ x_, I incx_)
{
    std::ofstream out(this->m_filename);
    if(!out.is_open())
    {
        return rocsparse_status_internal_error;
    }
    if(std::is_same<T, double>() || std::is_same<T, rocsparse_double_complex>())
    {
        out.precision(15);
        out.setf(std::ios::scientific);
    }
    else
    {
        out.precision(7);
        out.setf(std::ios::scientific);
    }

    out << "%%MatrixMarket matrix array ";
    if(std::is_same<T, rocsparse_float_complex>() || std::is_same<T, rocsparse_double_complex>())
        out << "complex";
    else
        out << "real";
    out << " general" << std::endl;
    out << nmemb_ << " 1" << std::endl;
    for(I i = 0; i < nmemb_; ++i)
    {
        if(std::is_same<T, rocsparse_float_complex>()
           || std::is_same<T, rocsparse_double_complex>())
        {
            out << std::real(x_[i * incx_]) << " " << std::imag(x_[i * incx_]) << std::endl;
        }
        else
        {
            out << x_[i * incx_] << std::endl;
        }
    }
    out.close();
    return rocsparse_status_success;
}

template <typename T, typename I>
rocsparse_status rocsparse_exporter_matrixmarket::write_dense_matrix(
    rocsparse_order order_, I m_, I n_, const T* __restrict__ x_, I ld_)
{
    std::ofstream out(this->m_filename);
    if(!out.is_open())
    {
        return rocsparse_status_internal_error;
    }
    if(std::is_same<T, double>() || std::is_same<T, rocsparse_double_complex>())
    {
        out.precision(15);
        out.setf(std::ios::scientific);
    }
    else
    {
        out.precision(7);
        out.setf(std::ios::scientific);
    }

    out << "%%MatrixMarket matrix array ";
    if(std::is_same<T, rocsparse_float_complex>() || std::is_same<T, rocsparse_double_complex>())
        out << "complex";
    else
        out << "real";
    out << " general" << std::endl;
    out << m_ << " " << n_ << std::endl;
    switch(order_)
    {
    case rocsparse_order_row:
    {
        for(I i = 0; i < m_; ++i)
        {
            for(I j = 0; j < n_; ++j)
            {
                if(std::is_same<T, rocsparse_float_complex>()
                   || std::is_same<T, rocsparse_double_complex>())
                    out << " " << std::real(x_[i * ld_ + j]) << " " << std::imag(x_[i * ld_ + j]);
                else
                    out << " " << x_[i * ld_ + j];
            }
            out << std::endl;
        }
        out.close();
        return rocsparse_status_success;
    }
    case rocsparse_order_column:
    {
        for(I i = 0; i < m_; ++i)
        {
            for(I j = 0; j < n_; ++j)
            {
                if(std::is_same<T, rocsparse_float_complex>()
                   || std::is_same<T, rocsparse_double_complex>())
                    out << " " << std::real(x_[j * ld_ + i]) << " " << std::imag(x_[j * ld_ + i]);
                else
                    out << " " << x_[j * ld_ + i];
            }
            out << std::endl;
        }
        out.close();
        return rocsparse_status_success;
    }
    }
    return rocsparse_status_invalid_value;
}

template <typename T, typename I>
rocsparse_status rocsparse_exporter_matrixmarket::write_sparse_coo(I m_,
                                                                   I n_,
                                                                   I nnz_,
                                                                   const I* __restrict__ row_ind_,
                                                                   const I* __restrict__ col_ind_,
                                                                   const T* __restrict__ val_,
                                                                   rocsparse_index_base base_)
{
    std::ofstream out(this->m_filename);
    if(!out.is_open())
    {
        return rocsparse_status_internal_error;
    }

    if(std::is_same<T, double>() || std::is_same<T, rocsparse_double_complex>())
    {
        out.precision(15);
        out.setf(std::ios::scientific);
    }
    else
    {
        out.precision(7);
        out.setf(std::ios::scientific);
    }

    out << "%%MatrixMarket matrix coordinate ";
    if(std::is_same<T, rocsparse_float_complex>() || std::is_same<T, rocsparse_double_complex>())
        out << "complex";
    else
        out << "real";

    out << " general" << std::endl;
    out << m_ << " " << n_ << " " << nnz_ << std::endl;

    for(I i = 0; i < nnz_; ++i)
    {
        out << ((row_ind_[i] - base_) + 1) << " " << ((col_ind_[i] - base_) + 1);
        if(std::is_same<T, rocsparse_float_complex>()
           || std::is_same<T, rocsparse_double_complex>())
        {
            out << " " << std::real(val_[i]) << " " << std::imag(val_[i]) << std::endl;
        }
        else
        {
            out << " " << val_[i] << std::endl;
        }
    }

    out.close();
    return rocsparse_status_success;
}

#define INSTANTIATE_TIJ(T, I, J)                                                   \
    template rocsparse_status rocsparse_exporter_matrixmarket::write_sparse_csx(   \
        rocsparse_direction,                                                       \
        J,                                                                         \
        J,                                                                         \
        I,                                                                         \
        const I* __restrict__,                                                     \
        const J* __restrict__,                                                     \
        const T* __restrict__,                                                     \
        rocsparse_index_base);                                                     \
    template rocsparse_status rocsparse_exporter_matrixmarket::write_sparse_gebsx( \
        rocsparse_direction,                                                       \
        rocsparse_direction,                                                       \
        J,                                                                         \
        J,                                                                         \
        I,                                                                         \
        J,                                                                         \
        J,                                                                         \
        const I* __restrict__,                                                     \
        const J* __restrict__,                                                     \
        const T* __restrict__,                                                     \
        rocsparse_index_base)

#define INSTANTIATE_TI(T, I)                                                       \
    template rocsparse_status rocsparse_exporter_matrixmarket::write_dense_vector( \
        I, const T* __restrict__, I);                                              \
    template rocsparse_status rocsparse_exporter_matrixmarket::write_dense_matrix( \
        rocsparse_order, I, I, const T* __restrict__, I);                          \
    template rocsparse_status rocsparse_exporter_matrixmarket::write_sparse_coo(   \
        I,                                                                         \
        I,                                                                         \
        I,                                                                         \
        const I* __restrict__,                                                     \
        const I* __restrict__,                                                     \
        const T* __restrict__,                                                     \
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
