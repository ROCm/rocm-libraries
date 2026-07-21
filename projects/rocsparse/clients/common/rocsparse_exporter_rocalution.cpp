/*! \file */
/* ************************************************************************
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_exporter_rocalution.hpp"
template <typename X, typename Y>
rocsparse_status rocsparse_type_conversion(const X& x, Y& y);

rocsparse_exporter_rocalution::~rocsparse_exporter_rocalution()
{
    const char* env = getenv("GTEST_LISTENER");
    if(!env || strcmp(env, "NO_PASS_LINE_IN_LOG"))
    {
        std::cout << "Export done." << std::endl;
    }
}

rocsparse_exporter_rocalution::rocsparse_exporter_rocalution(const std::string& filename_)
    : m_filename(filename_)
{

    const char* env = getenv("GTEST_LISTENER");
    if(!env || strcmp(env, "NO_PASS_LINE_IN_LOG"))
    {
        std::cout << "Opening file '" << this->m_filename << "' ... " << std::endl;
    }
}

template <typename T>
rocsparse_status rocalution_write_sparse_csx(
    const char* filename, int m, int n, int nnz, const int* ptr, const int* col, const T* val)
{
    std::ofstream out(filename, std::ios::out | std::ios::binary);

    if(!out.is_open())
    {
        return rocsparse_status_internal_error;
    }

    // Header
    out << "#rocALUTION binary csr file" << std::endl;

    // rocALUTION version
    int version = 10602;
    out.write((char*)&version, sizeof(int));

    // Data
    out.write((char*)&m, sizeof(int));
    out.write((char*)&n, sizeof(int));
    out.write((char*)&nnz, sizeof(int));
    out.write((char*)ptr, (m + 1) * sizeof(int));
    out.write((char*)col, nnz * sizeof(int));
    out.write((char*)val, nnz * sizeof(T));
    out.close();

    return rocsparse_status_success;
}

template <typename T>
void convert_array(int nnz, const void* data, void* mem)
{
    memcpy(mem, data, sizeof(T) * nnz);
}

template <>
void convert_array<rocsparse_float_complex>(int nnz, const void* data, void* mem)
{
    rocsparse_double_complex*      pmem  = (rocsparse_double_complex*)mem;
    const rocsparse_float_complex* pdata = (const rocsparse_float_complex*)data;
    for(int i = 0; i < nnz; ++i)
    {
        pmem[i] = rocsparse_double_complex(std::real(pdata[i]), std::imag(pdata[i]));
    }
}

template <>
void convert_array<float>(int nnz, const void* data, void* mem)
{
    double*      pmem  = (double*)mem;
    const float* pdata = (const float*)data;
    for(int i = 0; i < nnz; ++i)
    {
        pmem[i] = pdata[i];
    }
}

template <typename T, typename I, typename J>
rocsparse_status rocsparse_exporter_rocalution::write_sparse_csx(rocsparse_direction dir_,
                                                                 J                   m_,
                                                                 J                   n_,
                                                                 I                   nnz_,
                                                                 const I* __restrict__ ptr_,
                                                                 const J* __restrict__ ind_,
                                                                 const T* __restrict__ val_,
                                                                 rocsparse_index_base base_)
{

    if(dir_ != rocsparse_direction_row)
    {
        return rocsparse_status_not_implemented;
    }
    int              m;
    int              n;
    int              nnz;
    rocsparse_status status;

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

    const int*    ptr = nullptr;
    const int*    ind = nullptr;
    const double* val = nullptr;

    int*                  ptr_mem  = nullptr;
    int*                  ind_mem  = nullptr;
    double*               val_mem  = nullptr;
    static constexpr bool ptr_same = std::is_same<I, int>();
    static constexpr bool ind_same = std::is_same<J, int>();
    static constexpr bool val_same
        = std::is_same<T, double>() || std::is_same<T, rocsparse_double_complex>();

    bool is_T_complex = (std::is_same<T, rocsparse_double_complex>()
                         || std::is_same<T, rocsparse_float_complex>());

    ptr_mem = nullptr;
    if(!ptr_same && (base_ != rocsparse_index_base_zero))
    {
        std::ignore = rocsparse_hipHostMalloc(&ptr_mem, sizeof(int) * (m + 1));
    }

    ind_mem = nullptr;
    if(!ind_same && (base_ != rocsparse_index_base_zero))
    {
        std::ignore = rocsparse_hipHostMalloc(&ind_mem, sizeof(int) * nnz);
    }

    val_mem = nullptr;
    if(!val_same)
    {
        std::ignore
            = rocsparse_hipHostMalloc(&val_mem, sizeof(double) * (is_T_complex ? (2 * nnz) : nnz));
    }

    ptr = (ptr_same || (base_ == rocsparse_index_base_zero)) ? ((const int*)ptr_) : ptr_mem;
    ind = (ind_same || (base_ == rocsparse_index_base_zero)) ? ((const int*)ind_) : ind_mem;
    val = (val_same) ? ((const double*)val_) : val_mem;

    if(ptr_mem != nullptr)
    {
        for(int i = 0; i < m + 1; ++i)
        {
            status = rocsparse_type_conversion(ptr_[i], ptr_mem[i]);
            if(status != rocsparse_status_success)
            {
                break;
            }
        }

        if(base_ == rocsparse_index_base_one)
        {
            for(int i = 0; i < m + 1; ++i)
            {
                ptr_mem[i] = ptr_mem[i] - 1;
            }
        }

        if(status != rocsparse_status_success)
        {
            return status;
        }
    }

    if(ind_mem != nullptr)
    {
        for(int i = 0; i < nnz; ++i)
        {
            status = rocsparse_type_conversion(ind_[i], ind_mem[i]);
            if(status != rocsparse_status_success)
            {
                break;
            }
        }
        if(status != rocsparse_status_success)
        {
            return status;
        }
        if(base_ == rocsparse_index_base_one)
        {
            for(int i = 0; i < nnz; ++i)
            {
                ind_mem[i] = ind_mem[i] - 1;
            }
        }
    }

    if(val_mem != nullptr)
    {
        convert_array<T>(nnz, (const void*)val_, (void*)val_mem);
    }

    if(status != rocsparse_status_success)
    {
        return status;
    }

    status = rocalution_write_sparse_csx(this->m_filename.c_str(), m, n, nnz, ptr, ind, val);
    if(val_mem != nullptr)
    {
        std::ignore = rocsparse_hipFree(val_mem);
        val_mem     = nullptr;
    }
    if(ind_mem != nullptr)
    {
        std::ignore = rocsparse_hipFree(ind_mem);
        ind_mem     = nullptr;
    }
    if(ptr_mem != nullptr)
    {
        std::ignore = rocsparse_hipFree(ptr_mem);
        ptr_mem     = nullptr;
    }

    return status;
}

template <typename T, typename I, typename J>
rocsparse_status rocsparse_exporter_rocalution::write_sparse_gebsx(rocsparse_direction dir_,
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
    return rocsparse_status_not_implemented;
}

template <typename T, typename I>
rocsparse_status
    rocsparse_exporter_rocalution::write_dense_vector(I nmemb_, const T* __restrict__ x_, I incx_)
{
    return rocsparse_status_not_implemented;
}

template <typename T, typename I>
rocsparse_status rocsparse_exporter_rocalution::write_dense_matrix(
    rocsparse_order order_, I m_, I n_, const T* __restrict__ x_, I ld_)
{
    return rocsparse_status_not_implemented;
}

template <typename T, typename I>
rocsparse_status rocsparse_exporter_rocalution::write_sparse_coo(I m_,
                                                                 I n_,
                                                                 I nnz_,
                                                                 const I* __restrict__ row_ind_,
                                                                 const I* __restrict__ col_ind_,
                                                                 const T* __restrict__ val_,
                                                                 rocsparse_index_base base_)
{
    return rocsparse_status_not_implemented;
}

#define INSTANTIATE_TIJ(T, I, J)                                                 \
    template rocsparse_status rocsparse_exporter_rocalution::write_sparse_csx(   \
        rocsparse_direction,                                                     \
        J,                                                                       \
        J,                                                                       \
        I,                                                                       \
        const I* __restrict__,                                                   \
        const J* __restrict__,                                                   \
        const T* __restrict__,                                                   \
        rocsparse_index_base);                                                   \
    template rocsparse_status rocsparse_exporter_rocalution::write_sparse_gebsx( \
        rocsparse_direction,                                                     \
        rocsparse_direction,                                                     \
        J,                                                                       \
        J,                                                                       \
        I,                                                                       \
        J,                                                                       \
        J,                                                                       \
        const I* __restrict__,                                                   \
        const J* __restrict__,                                                   \
        const T* __restrict__,                                                   \
        rocsparse_index_base)

#define INSTANTIATE_TI(T, I)                                                     \
    template rocsparse_status rocsparse_exporter_rocalution::write_dense_vector( \
        I, const T* __restrict__, I);                                            \
    template rocsparse_status rocsparse_exporter_rocalution::write_dense_matrix( \
        rocsparse_order, I, I, const T* __restrict__, I);                        \
    template rocsparse_status rocsparse_exporter_rocalution::write_sparse_coo(   \
        I,                                                                       \
        I,                                                                       \
        I,                                                                       \
        const I* __restrict__,                                                   \
        const I* __restrict__,                                                   \
        const T* __restrict__,                                                   \
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
