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

#include "rocsparse_exporter_ascii.hpp"
template <typename X, typename Y>
rocsparse_status rocsparse_type_conversion(const X& x, Y& y);

rocsparse_exporter_ascii::~rocsparse_exporter_ascii()
{
    const char* env = getenv("GTEST_LISTENER");
    if(!env || strcmp(env, "NO_PASS_LINE_IN_LOG"))
    {
        std::cout << "Export done." << std::endl;
    }
}

rocsparse_exporter_ascii::rocsparse_exporter_ascii(const std::string& filename_)
    : m_filename(filename_)
{

    const char* env = getenv("GTEST_LISTENER");
    if(!env || strcmp(env, "NO_PASS_LINE_IN_LOG"))
    {
        std::cout << "Opening file '" << this->m_filename << "' ... " << std::endl;
    }
}

template <typename T>
void convert_array(int nnz, const void* data, void* mem);
template <typename T, typename I, typename J>
rocsparse_status rocsparse_exporter_ascii::write_sparse_csx(rocsparse_direction dir_,
                                                            J                   m_,
                                                            J                   n_,
                                                            I                   nnz_,
                                                            const I* __restrict__ ptr_,
                                                            const J* __restrict__ ind_,
                                                            const T* __restrict__ val_,
                                                            rocsparse_index_base base_)
{
    std::ofstream out(this->m_filename);
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
    if(dir_ == rocsparse_direction_row)
    {
        out << "matrix: sparse_csr" << std::endl;
    }
    else
    {
        out << "matrix: sparse_csc" << std::endl;
    }

    out << "dir: " << dir_ << std::endl;
    out << "m: " << m_ << std::endl;
    out << "n: " << n_ << std::endl;
    out << "nnz: " << nnz_ << std::endl;
    out << "base: " << base_ << std::endl;
    const char* dir  = (dir_ == rocsparse_direction_row) ? "row" : "col";
    const char* odir = (dir_ == rocsparse_direction_row) ? "col" : "row";
    J           L    = (dir_ == rocsparse_direction_row) ? m_ : n_;
    for(J i = 0; i < L; ++i)
    {
        out << dir << ": " << i << std::endl;
        for(int k = ptr_[i]; k < ptr_[i + 1]; ++k)
        {
            out << " " << odir << " = " << (ind_[k - base_] - base_) << ", val =  " << val_[k]
                << std::endl;
        }
    }
    out.close();
    return rocsparse_status_success;
}

template <typename T, typename I, typename J>
rocsparse_status rocsparse_exporter_ascii::write_sparse_gebsx(rocsparse_direction dir_,
                                                              rocsparse_direction dirb_,
                                                              J                   mb_,
                                                              J                   nb_,
                                                              I                   nnzb_,
                                                              J                   bm_,
                                                              J                   bn_,
                                                              const I* __restrict__ ptr_,
                                                              const J* __restrict__ ind_,
                                                              const T* __restrict__ val_,
                                                              rocsparse_index_base base_)
{
    std::ofstream out(this->m_filename);
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
    if(dir_ == rocsparse_direction_row)
    {
        out << "matrix: sparse_gebsr" << std::endl;
    }
    else
    {
        out << "matrix: sparse_gebsc" << std::endl;
    }

    out << "dir: " << dir_ << std::endl;
    out << "dirb: " << dirb_ << std::endl;
    out << "mb: " << mb_ << std::endl;
    out << "nb: " << nb_ << std::endl;
    out << "nnzb: " << nnzb_ << std::endl;
    out << "bm: " << bm_ << std::endl;
    out << "bn: " << bn_ << std::endl;
    out << "base: " << base_ << std::endl;
    out.close();
    return rocsparse_status_success;
}

template <typename T, typename I>
rocsparse_status
    rocsparse_exporter_ascii::write_dense_vector(I nmemb_, const T* __restrict__ x_, I incx_)
{
    std::ofstream out(this->m_filename);
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

    out << "matrix: dense_vector" << std::endl;
    out << "m: " << nmemb_ << std::endl;
    out << "data: " << std::endl;
    for(I i = 0; i < nmemb_; ++i)
    {
        out << x_[incx_ * i] << std::endl;
    }
    out.close();
    return rocsparse_status_success;
}

template <typename T, typename I>
rocsparse_status rocsparse_exporter_ascii::write_dense_matrix(
    rocsparse_order order_, I m_, I n_, const T* __restrict__ x_, I ld_)
{
    std::ofstream out(this->m_filename);
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

    out << "matrix: dense_matrix" << std::endl;
    out << "order: " << order_ << std::endl;
    out << "m: " << m_ << std::endl;
    out << "n: " << n_ << std::endl;
    out << "data: " << std::endl;
    for(I i = 0; i < m_; ++i)
    {
        for(I j = 0; j < n_; ++j)
        {
            if(order_ == rocsparse_order_row)
            {
                out << " " << x_[ld_ * j + i];
            }
            else
            {
                out << " " << x_[ld_ * i + j];
            }
        }
        out << std::endl;
    }
    out.close();
    return rocsparse_status_success;
}

template <typename T, typename I>
rocsparse_status rocsparse_exporter_ascii::write_sparse_coo(I m_,
                                                            I n_,
                                                            I nnz_,
                                                            const I* __restrict__ row_ind_,
                                                            const I* __restrict__ col_ind_,
                                                            const T* __restrict__ val_,
                                                            rocsparse_index_base base_)
{
    return rocsparse_status_not_implemented;
}

#define INSTANTIATE_TIJ(T, I, J)                                                                  \
    template rocsparse_status rocsparse_exporter_ascii::write_sparse_csx(rocsparse_direction,     \
                                                                         J,                       \
                                                                         J,                       \
                                                                         I,                       \
                                                                         const I* __restrict__,   \
                                                                         const J* __restrict__,   \
                                                                         const T* __restrict__,   \
                                                                         rocsparse_index_base);   \
    template rocsparse_status rocsparse_exporter_ascii::write_sparse_gebsx(rocsparse_direction,   \
                                                                           rocsparse_direction,   \
                                                                           J,                     \
                                                                           J,                     \
                                                                           I,                     \
                                                                           J,                     \
                                                                           J,                     \
                                                                           const I* __restrict__, \
                                                                           const J* __restrict__, \
                                                                           const T* __restrict__, \
                                                                           rocsparse_index_base)

#define INSTANTIATE_TI(T, I)                                                                    \
    template rocsparse_status rocsparse_exporter_ascii::write_dense_vector(                     \
        I, const T* __restrict__, I);                                                           \
    template rocsparse_status rocsparse_exporter_ascii::write_dense_matrix(                     \
        rocsparse_order, I, I, const T* __restrict__, I);                                       \
    template rocsparse_status rocsparse_exporter_ascii::write_sparse_coo(I,                     \
                                                                         I,                     \
                                                                         I,                     \
                                                                         const I* __restrict__, \
                                                                         const I* __restrict__, \
                                                                         const T* __restrict__, \
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
