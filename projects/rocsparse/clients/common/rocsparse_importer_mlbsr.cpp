/*! \file */
/* ************************************************************************
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "rocsparse_importer_mlbsr.hpp"
#include <stdio.h>
rocsparse_importer_mlbsr::~rocsparse_importer_mlbsr()
{
    if(this->m_f != nullptr)
    {
        fclose(this->m_f);
    }
}

rocsparse_importer_mlbsr::rocsparse_importer_mlbsr(const std::string& filename_)
    : m_filename(filename_)
{
}

template <typename I, typename J>
rocsparse_status

    rocsparse_importer_mlbsr::import_sparse_csx(
        rocsparse_direction* dir, J* m, J* n, I* nnz, rocsparse_index_base* base)
{
    return rocsparse_status_not_implemented;
}

template <typename T, typename I, typename J>
rocsparse_status rocsparse_importer_mlbsr::import_sparse_csx(I* ptr, J* ind, T* val)
{
    return rocsparse_status_not_implemented;
}

template <typename I, typename J>
rocsparse_status rocsparse_importer_mlbsr::import_sparse_gebsx(rocsparse_direction* dir,
                                                               rocsparse_direction* dirb,
                                                               J*                   mb,
                                                               J*                   nb,
                                                               I*                   nnzb,
                                                               J*                   block_dim_row,
                                                               J* block_dim_column,
                                                               rocsparse_index_base* base)
{
    char line[1024];
    this->m_f = fopen(this->m_filename.c_str(), "r");
    if(!this->m_f)
    {
        missing_file_error_message(this->m_filename.c_str());
        return rocsparse_status_internal_error;
    }

    //
    // Skip header.
    //
    while(0 != fgets(line, 1024, this->m_f))
    {
        const char* l = &line[0];
        while(l[0] != '\0' && (l[0] == ' ' || l[0] == '\t'))
        {
            ++l;
        }
        if(l[0] != '\0' && l[0] != '%')
        {
            break;
        }
    }

    //
    // Read dimension.
    //
    size_t inrow;
    size_t incol;
    size_t innz;
    if(EOF == sscanf(line, "%zd %zd %zd", &inrow, &incol, &innz))
    {
        return rocsparse_status_internal_error;
    }

    if(0 == fgets(line, 1024, this->m_f))
    {
        return rocsparse_status_internal_error;
    }

    size_t ibdim_row;
    size_t ibdim_col;
    if(EOF == sscanf(line, "%zd %zd", &ibdim_row, &ibdim_col))
    {
        return rocsparse_status_internal_error;
    }

    //
    // Convert.
    //
    rocsparse_status status;
    const size_t     imb = inrow / ibdim_row;
    status               = rocsparse_type_conversion(imb, mb[0]);
    if(status != rocsparse_status_success)
        return status;

    const size_t inb = incol / ibdim_col;
    status           = rocsparse_type_conversion(inb, nb[0]);
    if(status != rocsparse_status_success)
        return status;

    status = rocsparse_type_conversion(ibdim_row, block_dim_row[0]);
    if(status != rocsparse_status_success)
        return status;

    status = rocsparse_type_conversion(ibdim_col, block_dim_column[0]);
    if(status != rocsparse_status_success)
        return status;

    const size_t innzb = innz / (ibdim_col * ibdim_row);
    status             = rocsparse_type_conversion(innzb, nnzb[0]);
    if(status != rocsparse_status_success)
        return status;

    dir[0]  = rocsparse_direction_row;
    dirb[0] = rocsparse_direction_column;
    base[0] = rocsparse_index_base_zero;

    this->m_mb   = imb;
    this->m_nnzb = innzb;

    return rocsparse_status_success;
}

template <typename T, typename I, typename J>
rocsparse_status rocsparse_importer_mlbsr::import_sparse_gebsx(I* ptr, J* ind, T* val)
{
    size_t           k;
    rocsparse_status status;
    for(size_t i = 0; i <= this->m_mb; ++i)
    {
        if(EOF == fscanf(this->m_f, "%zd", &k))
        {
            return rocsparse_status_internal_error;
        }
        status = rocsparse_type_conversion(k, ptr[i]);
        if(status != rocsparse_status_success)
        {
            return status;
        }
    }
    for(size_t i = 0; i < this->m_nnzb; ++i)
    {
        if(EOF == fscanf(this->m_f, "%zd", &k))
        {
            return rocsparse_status_internal_error;
        }
        status = rocsparse_type_conversion(k, ind[i]);
        if(status != rocsparse_status_success)
        {
            return status;
        }
    }
    return rocsparse_status_success;
}

template <typename I>
rocsparse_status rocsparse_importer_mlbsr::import_sparse_coo(I*                    m,
                                                             I*                    n,
                                                             int64_t*              nnz,
                                                             rocsparse_index_base* base)
{
    return rocsparse_status_not_implemented;
}

template <typename T, typename I>
rocsparse_status rocsparse_importer_mlbsr::import_sparse_coo(I* row_ind, I* col_ind, T* val)
{
    return rocsparse_status_not_implemented;
}

#define INSTANTIATE_TIJ(T, I, J)                                                       \
    template rocsparse_status rocsparse_importer_mlbsr::import_sparse_csx(I*, J*, T*); \
    template rocsparse_status rocsparse_importer_mlbsr::import_sparse_gebsx(I*, J*, T*)

#define INSTANTIATE_TI(T, I)                                               \
    template rocsparse_status rocsparse_importer_mlbsr::import_sparse_coo( \
        I* row_ind, I* col_ind, T* val)

#define INSTANTIATE_I(I)                                                   \
    template rocsparse_status rocsparse_importer_mlbsr::import_sparse_coo( \
        I* m, I* n, int64_t* nnz, rocsparse_index_base* base)

#define INSTANTIATE_IJ(I, J)                                                 \
    template rocsparse_status rocsparse_importer_mlbsr::import_sparse_csx(   \
        rocsparse_direction*, J*, J*, I*, rocsparse_index_base*);            \
    template rocsparse_status rocsparse_importer_mlbsr::import_sparse_gebsx( \
        rocsparse_direction*, rocsparse_direction*, J*, J*, I*, J*, J*, rocsparse_index_base*)

INSTANTIATE_I(int32_t);
INSTANTIATE_I(int64_t);

INSTANTIATE_IJ(int32_t, int32_t);
INSTANTIATE_IJ(int64_t, int32_t);
INSTANTIATE_IJ(int64_t, int64_t);

INSTANTIATE_TIJ(int8_t, int32_t, int32_t);
INSTANTIATE_TIJ(int8_t, int64_t, int32_t);
INSTANTIATE_TIJ(int8_t, int64_t, int64_t);

INSTANTIATE_TIJ(_Float16, int32_t, int32_t);
INSTANTIATE_TIJ(_Float16, int64_t, int32_t);
INSTANTIATE_TIJ(_Float16, int64_t, int64_t);

INSTANTIATE_TIJ(rocsparse_bfloat16, int32_t, int32_t);
INSTANTIATE_TIJ(rocsparse_bfloat16, int64_t, int32_t);
INSTANTIATE_TIJ(rocsparse_bfloat16, int64_t, int64_t);

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

INSTANTIATE_TI(int8_t, int32_t);
INSTANTIATE_TI(int8_t, int64_t);

INSTANTIATE_TI(_Float16, int32_t);
INSTANTIATE_TI(_Float16, int64_t);

INSTANTIATE_TI(rocsparse_bfloat16, int32_t);
INSTANTIATE_TI(rocsparse_bfloat16, int64_t);

INSTANTIATE_TI(float, int32_t);
INSTANTIATE_TI(float, int64_t);

INSTANTIATE_TI(double, int32_t);
INSTANTIATE_TI(double, int64_t);

INSTANTIATE_TI(rocsparse_float_complex, int32_t);
INSTANTIATE_TI(rocsparse_float_complex, int64_t);

INSTANTIATE_TI(rocsparse_double_complex, int32_t);
INSTANTIATE_TI(rocsparse_double_complex, int64_t);
