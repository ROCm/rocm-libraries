/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_importer_rocsparseio.hpp"

#define ROCSPARSE_CHECK_ROCSPARSEIO(iostatus_)  \
    if(iostatus_ != rocsparseio_status_success) \
    {                                           \
        return rocsparse_status_internal_error; \
    }

template <typename X, typename Y>
inline rocsparseio_status rocsparseio2rocsparse_convert(const X& x, Y& y);

template <>
inline rocsparseio_status rocsparseio2rocsparse_convert(const rocsparseio_direction& x,
                                                        rocsparse_direction&         y)
{
    switch(x)
    {
    case rocsparseio_direction_row:
    {
        y = rocsparse_direction_row;
        return rocsparseio_status_success;
    }
    case rocsparseio_direction_column:
    {
        y = rocsparse_direction_column;
        return rocsparseio_status_success;
    }
    }
    return rocsparseio_status_invalid_value;
}

template <>
inline rocsparseio_status rocsparseio2rocsparse_convert(const rocsparseio_order& x,
                                                        rocsparse_order&         y)
{
    switch(x)
    {
    case rocsparseio_order_row:
    {
        y = rocsparse_order_row;
        return rocsparseio_status_success;
    }
    case rocsparseio_order_column:
    {
        y = rocsparse_order_column;
        return rocsparseio_status_success;
    }
    }
    return rocsparseio_status_invalid_value;
}

template <>
inline rocsparseio_status rocsparseio2rocsparse_convert(const rocsparseio_index_base& x,
                                                        rocsparse_index_base&         y)
{
    switch(x)
    {
    case rocsparseio_index_base_zero:
    {
        y = rocsparse_index_base_zero;
        return rocsparseio_status_success;
    }
    case rocsparseio_index_base_one:
    {
        y = rocsparse_index_base_one;
        return rocsparseio_status_success;
    }
    }
    return rocsparseio_status_invalid_value;
}

template <typename T>
inline rocsparseio_type type_tconvert();

template <>
inline rocsparseio_type type_tconvert<int8_t>()
{
    return rocsparseio_type_int8;
};

template <>
inline rocsparseio_type type_tconvert<int32_t>()
{
    return rocsparseio_type_int32;
};

template <>
inline rocsparseio_type type_tconvert<int64_t>()
{
    return rocsparseio_type_int64;
};
template <>
inline rocsparseio_type type_tconvert<_Float16>()
{
    return rocsparseio_type_float16;
};
template <>
inline rocsparseio_type type_tconvert<rocsparse_bfloat16>()
{
    return rocsparseio_type_bfloat16;
};
template <>
inline rocsparseio_type type_tconvert<float>()
{
    return rocsparseio_type_float32;
};
template <>
inline rocsparseio_type type_tconvert<double>()
{
    return rocsparseio_type_float64;
};
template <>
inline rocsparseio_type type_tconvert<rocsparse_float_complex>()
{
    return rocsparseio_type_complex32;
};
template <>
inline rocsparseio_type type_tconvert<rocsparse_double_complex>()
{
    return rocsparseio_type_complex64;
};

rocsparse_importer_rocsparseio::~rocsparse_importer_rocsparseio()
{

    const char* env = getenv("GTEST_LISTENER");
    if(!env || strcmp(env, "NO_PASS_LINE_IN_LOG"))
    {
        std::cout << "Import done." << std::endl;
    }

    auto istatus = rocsparseio_close(this->m_handle);
    if(istatus != rocsparseio_status_success)
    {
    }
}

rocsparse_importer_rocsparseio::rocsparse_importer_rocsparseio(const std::string& filename_)
    : m_filename(filename_)
{

    const char* env = getenv("GTEST_LISTENER");
    if(!env || strcmp(env, "NO_PASS_LINE_IN_LOG"))
    {
        std::cout << "Opening file '" << this->m_filename << "' ... " << std::endl;
    }

    rocsparseio_status istatus;
    istatus = rocsparseio_open(&this->m_handle, rocsparseio_rwmode_read, this->m_filename.c_str());
    if(istatus != rocsparseio_status_success)
    {
        missing_file_error_message(this->m_filename.c_str());
        throw rocsparse_status_internal_error;
    }
}

template <typename I>
rocsparse_status rocsparse_importer_rocsparseio::import_sparse_coo(I*                    m,
                                                                   I*                    n,
                                                                   int64_t*              nnz,
                                                                   rocsparse_index_base* base)
{

    uint64_t               iM;
    uint64_t               iN;
    uint64_t               innz;
    rocsparseio_index_base ibase;
    rocsparseio_status     istatus;
    istatus = rocsparseiox_read_metadata_sparse_coo(this->m_handle,
                                                    &iM,
                                                    &iN,
                                                    &innz,
                                                    &this->m_row_ind_type,
                                                    &this->m_col_ind_type,
                                                    &this->m_val_type,
                                                    &ibase);
    ROCSPARSE_CHECK_ROCSPARSEIO(istatus);
    rocsparse_status status;

    status = rocsparse_type_conversion(iM, m[0]);
    if(status != rocsparse_status_success)
        return status;

    status = rocsparse_type_conversion(iN, n[0]);
    if(status != rocsparse_status_success)
        return status;

    status = rocsparse_type_conversion(innz, nnz[0]);
    if(status != rocsparse_status_success)
        return status;

    this->m_nnz = innz;
    ROCSPARSE_CHECK_ROCSPARSEIO(rocsparseio2rocsparse_convert(ibase, *base));
    return rocsparse_status_success;
}

template <typename T, typename I>
rocsparse_status rocsparse_importer_rocsparseio::import_sparse_coo(I* row_ind, I* col_ind, T* val)
{
    rocsparseio_status     istatus;
    const rocsparseio_type csr_ind_type = type_tconvert<I>(), csr_val_type = type_tconvert<T>();

    const uint64_t NNZ = this->m_nnz;

    const bool same_ind_type = (this->m_ind_type == csr_ind_type);
    const bool same_val_type = (this->m_val_type == csr_val_type);
    const bool is_consistent = same_ind_type && same_val_type;

    if(is_consistent)
    {

        //
        // Import data.
        //
        istatus = rocsparseiox_read_sparse_coo(this->m_handle, row_ind, col_ind, val);
        ROCSPARSE_CHECK_ROCSPARSEIO(istatus);
    }
    else
    {

        void *tmp_row_ind = (void*)row_ind, *tmp_col_ind = (void*)col_ind, *tmp_val = (void*)val;

        host_dense_vector<char> tmp_row_indv, tmp_col_indv, tmp_valv;

        uint64_t sizeof_ind_type, sizeof_val_type;

        istatus = rocsparseio_type_get_size(this->m_ind_type, &sizeof_ind_type);
        ROCSPARSE_CHECK_ROCSPARSEIO(istatus);
        istatus = rocsparseio_type_get_size(this->m_val_type, &sizeof_val_type);
        ROCSPARSE_CHECK_ROCSPARSEIO(istatus);

        if(!same_ind_type)
        {
            tmp_row_indv.resize(NNZ * sizeof_ind_type);
            tmp_row_ind = tmp_row_indv;

            tmp_col_indv.resize(NNZ * sizeof_ind_type);
            tmp_col_ind = tmp_col_indv;
        }

        if(!same_val_type)
        {
            tmp_valv.resize(NNZ * sizeof_val_type);
            tmp_val = tmp_valv;
        }

        istatus = rocsparseiox_read_sparse_coo(this->m_handle, tmp_row_ind, tmp_col_ind, tmp_val);
        ROCSPARSE_CHECK_ROCSPARSEIO(istatus);

        if(!same_ind_type)
        {
            switch(this->m_ind_type)
            {
            case rocsparseio_type_int32:
            {
                //
                // copy tmp_ind to ind.
                //
                rocsparse_importer_copy_mixed_arrays(NNZ, row_ind, (const int32_t*)tmp_row_ind);
                rocsparse_importer_copy_mixed_arrays(NNZ, col_ind, (const int32_t*)tmp_col_ind);
                break;
            }

            case rocsparseio_type_int64:
            {
                rocsparse_importer_copy_mixed_arrays(NNZ, row_ind, (const int64_t*)tmp_row_ind);
                rocsparse_importer_copy_mixed_arrays(NNZ, col_ind, (const int64_t*)tmp_col_ind);
                break;
            }

            case rocsparseio_type_int8:
            case rocsparseio_type_float16:
            case rocsparseio_type_bfloat16:
            case rocsparseio_type_float32:
            case rocsparseio_type_float64:
            case rocsparseio_type_complex32:
            case rocsparseio_type_complex64:
            {
                break;
            }
            }
        }

        if(!same_val_type)
        {
            switch(this->m_val_type)
            {
            case rocsparseio_type_int32:
            case rocsparseio_type_int64:
            {
                break;
            }

            case rocsparseio_type_int8:
            {
                rocsparse_importer_copy_mixed_arrays(NNZ, val, (const int8_t*)tmp_val);
                break;
            }

            case rocsparseio_type_float16:
            {
                rocsparse_importer_copy_mixed_arrays(NNZ, val, (const _Float16*)tmp_val);
                break;
            }

            case rocsparseio_type_bfloat16:
            {
                rocsparse_importer_copy_mixed_arrays(NNZ, val, (const rocsparse_bfloat16*)tmp_val);
                break;
            }

            case rocsparseio_type_float32:
            {
                //
                // copy val2 to val.
                //
                rocsparse_importer_copy_mixed_arrays(NNZ, val, (const float*)tmp_val);
                break;
            }

            case rocsparseio_type_float64:
            {
                rocsparse_importer_copy_mixed_arrays(NNZ, val, (const double*)tmp_val);
                break;
            }

            case rocsparseio_type_complex32:
            {
                rocsparse_importer_copy_mixed_arrays(
                    NNZ, val, (const rocsparse_float_complex*)tmp_val);
                break;
            }
            case rocsparseio_type_complex64:
            {
                rocsparse_importer_copy_mixed_arrays(
                    NNZ, val, (const rocsparse_double_complex*)tmp_val);
                break;
            }
            }
        }
    }
    return rocsparse_status_success;
}

template <typename I, typename J>
rocsparse_status rocsparse_importer_rocsparseio::import_sparse_gebsx(rocsparse_direction* dir,
                                                                     rocsparse_direction* dirb,
                                                                     J*                   mb,
                                                                     J*                   nb,
                                                                     I*                   nnzb,
                                                                     J* block_dim_row,
                                                                     J* block_dim_column,
                                                                     rocsparse_index_base* base)
{
    rocsparseio_status     istatus;
    rocsparseio_direction  idir, idirb;
    rocsparseio_index_base ibase;
    istatus = rocsparseio_open(&this->m_handle, rocsparseio_rwmode_read, this->m_filename.c_str());
    ROCSPARSE_CHECK_ROCSPARSEIO(istatus);

    uint64_t iMb;
    uint64_t iNb;
    uint64_t innzb;
    uint64_t irow_block_dim, icol_block_dim;
    istatus = rocsparseiox_read_metadata_sparse_gebsx(this->m_handle,
                                                      &idir,
                                                      &idirb,
                                                      &iMb,
                                                      &iNb,
                                                      &innzb,
                                                      &irow_block_dim,
                                                      &icol_block_dim,
                                                      &this->m_ptr_type,
                                                      &this->m_ind_type,
                                                      &this->m_val_type,
                                                      &ibase);

    ROCSPARSE_CHECK_ROCSPARSEIO(istatus);

    ROCSPARSE_CHECK_ROCSPARSEIO(rocsparseio2rocsparse_convert(ibase, *base));
    ROCSPARSE_CHECK_ROCSPARSEIO(rocsparseio2rocsparse_convert(idir, *dir));
    ROCSPARSE_CHECK_ROCSPARSEIO(rocsparseio2rocsparse_convert(idirb, *dirb));

    rocsparse_status status;
    status = rocsparse_type_conversion(iMb, mb[0]);
    if(status != rocsparse_status_success)
        return status;
    status = rocsparse_type_conversion(iNb, nb[0]);
    if(status != rocsparse_status_success)
        return status;
    status = rocsparse_type_conversion(innzb, nnzb[0]);
    if(status != rocsparse_status_success)
        return status;
    status = rocsparse_type_conversion(irow_block_dim, block_dim_row[0]);
    if(status != rocsparse_status_success)
        return status;
    status = rocsparse_type_conversion(icol_block_dim, block_dim_column[0]);
    if(status != rocsparse_status_success)
        return status;

    this->m_mb            = iMb;
    this->m_nnzb          = innzb;
    this->m_row_block_dim = irow_block_dim;
    this->m_col_block_dim = icol_block_dim;
    return rocsparse_status_success;
}

template <typename T, typename I, typename J>
rocsparse_status rocsparse_importer_rocsparseio::import_sparse_gebsx(I* ptr, J* ind, T* val)
{

    rocsparseio_status     istatus;
    const rocsparseio_type csr_ptr_type = type_tconvert<I>(), csr_ind_type = type_tconvert<J>(),
                           csr_val_type = type_tconvert<T>();

    const uint64_t MB            = this->m_mb;
    const uint64_t NNZB          = this->m_nnzb;
    const uint64_t row_block_dim = this->m_row_block_dim;
    const uint64_t col_block_dim = this->m_col_block_dim;

    const bool same_ptr_type = (this->m_ptr_type == csr_ptr_type);
    const bool same_ind_type = (this->m_ind_type == csr_ind_type);
    const bool same_val_type = (this->m_val_type == csr_val_type);
    const bool is_consistent = same_ptr_type && same_ind_type && same_val_type;

    if(is_consistent)
    {

        //
        // Import data.
        //
        istatus = rocsparseiox_read_sparse_gebsx(this->m_handle, ptr, ind, val);
        ROCSPARSE_CHECK_ROCSPARSEIO(istatus);
    }
    else
    {

        void *tmp_ptr = (void*)ptr, *tmp_ind = (void*)ind, *tmp_val = (void*)val;

        host_dense_vector<char> tmp_ptrv, tmp_indv, tmp_valv;

        uint64_t sizeof_ptr_type, sizeof_ind_type, sizeof_val_type;

        istatus = rocsparseio_type_get_size(this->m_ptr_type, &sizeof_ptr_type);
        ROCSPARSE_CHECK_ROCSPARSEIO(istatus);
        istatus = rocsparseio_type_get_size(this->m_ind_type, &sizeof_ind_type);
        ROCSPARSE_CHECK_ROCSPARSEIO(istatus);
        istatus = rocsparseio_type_get_size(this->m_val_type, &sizeof_val_type);

        ROCSPARSE_CHECK_ROCSPARSEIO(istatus);
        if(!same_ptr_type)
        {
            tmp_ptrv.resize((MB + 1) * sizeof_ptr_type);
            tmp_ptr = tmp_ptrv;
        }

        if(!same_ind_type)
        {
            tmp_indv.resize(NNZB * sizeof_ind_type);
            tmp_ind = tmp_indv;
        }

        if(!same_val_type)
        {
            tmp_valv.resize(NNZB * row_block_dim * col_block_dim * sizeof_val_type);
            tmp_val = tmp_valv;
        }

        istatus = rocsparseiox_read_sparse_csx(this->m_handle, tmp_ptr, tmp_ind, tmp_val);
        ROCSPARSE_CHECK_ROCSPARSEIO(istatus);
        if(!same_ptr_type)
        {
            switch(this->m_ptr_type)
            {
            case rocsparseio_type_int32:
            {
                rocsparse_importer_copy_mixed_arrays(MB + 1, ptr, (const int32_t*)tmp_ptr);
                break;
            }

            case rocsparseio_type_int64:
            {
                rocsparse_importer_copy_mixed_arrays(MB + 1, ptr, (const int64_t*)tmp_ptr);
                break;
            }

            case rocsparseio_type_int8:
            case rocsparseio_type_float16:
            case rocsparseio_type_bfloat16:
            case rocsparseio_type_float32:
            case rocsparseio_type_float64:
            case rocsparseio_type_complex32:
            case rocsparseio_type_complex64:
            {
                break;
            }
            }
        }

        if(!same_ind_type)
        {
            switch(this->m_ind_type)
            {
            case rocsparseio_type_int32:
            {
                rocsparse_importer_copy_mixed_arrays(NNZB, ind, (const int32_t*)tmp_ind);
                break;
            }
            case rocsparseio_type_int64:
            {
                rocsparse_importer_copy_mixed_arrays(NNZB, ind, (const int64_t*)tmp_ind);
                break;
            }
            case rocsparseio_type_int8:
            case rocsparseio_type_float16:
            case rocsparseio_type_bfloat16:
            case rocsparseio_type_float32:
            case rocsparseio_type_float64:
            case rocsparseio_type_complex32:
            case rocsparseio_type_complex64:
            {
                break;
            }
            }
        }

        if(!same_val_type)
        {
            switch(this->m_val_type)
            {
            case rocsparseio_type_int32:
            case rocsparseio_type_int64:
            {
                break;
            }

            case rocsparseio_type_int8:
            {
                rocsparse_importer_copy_mixed_arrays(
                    NNZB * row_block_dim * col_block_dim, val, (const int8_t*)tmp_val);
                break;
            }

            case rocsparseio_type_float16:
            {
                rocsparse_importer_copy_mixed_arrays(
                    NNZB * row_block_dim * col_block_dim, val, (const _Float16*)tmp_val);
                break;
            }

            case rocsparseio_type_bfloat16:
            {
                rocsparse_importer_copy_mixed_arrays(
                    NNZB * row_block_dim * col_block_dim, val, (const rocsparse_bfloat16*)tmp_val);
                break;
            }

            case rocsparseio_type_float32:
            {
                rocsparse_importer_copy_mixed_arrays(
                    NNZB * row_block_dim * col_block_dim, val, (const float*)tmp_val);
                break;
            }

            case rocsparseio_type_float64:
            {
                rocsparse_importer_copy_mixed_arrays(
                    NNZB * row_block_dim * col_block_dim, val, (const double*)tmp_val);
                break;
            }

            case rocsparseio_type_complex32:
            {
                rocsparse_importer_copy_mixed_arrays(NNZB * row_block_dim * col_block_dim,
                                                     val,
                                                     (const rocsparse_float_complex*)tmp_val);
                break;
            }

            case rocsparseio_type_complex64:
            {
                rocsparse_importer_copy_mixed_arrays(NNZB * row_block_dim * col_block_dim,
                                                     val,
                                                     (const rocsparse_double_complex*)tmp_val);
                break;
            }
            }
        }
    }

    return rocsparse_status_success;
}

template <typename I, typename J>
rocsparse_status

    rocsparse_importer_rocsparseio::import_sparse_csx(
        rocsparse_direction* dir, J* m, J* n, I* nnz, rocsparse_index_base* base)
{
    rocsparseio_status     istatus;
    rocsparseio_direction  io_dir;
    rocsparseio_index_base ibase;
    istatus = rocsparseio_open(&this->m_handle, rocsparseio_rwmode_read, this->m_filename.c_str());
    ROCSPARSE_CHECK_ROCSPARSEIO(istatus);
    uint64_t iM;
    uint64_t iN;
    uint64_t innz;
    istatus = rocsparseiox_read_metadata_sparse_csx(this->m_handle,
                                                    &io_dir,
                                                    &iM,
                                                    &iN,
                                                    &innz,
                                                    &this->m_ptr_type,
                                                    &this->m_ind_type,
                                                    &this->m_val_type,
                                                    &ibase);
    ROCSPARSE_CHECK_ROCSPARSEIO(istatus);

    ROCSPARSE_CHECK_ROCSPARSEIO(rocsparseio2rocsparse_convert(ibase, *base));
    ROCSPARSE_CHECK_ROCSPARSEIO(rocsparseio2rocsparse_convert(io_dir, *dir));

    rocsparse_status status;
    status = rocsparse_type_conversion(iM, m[0]);
    if(status != rocsparse_status_success)
        return status;
    status = rocsparse_type_conversion(iN, n[0]);
    if(status != rocsparse_status_success)
        return status;
    status = rocsparse_type_conversion(innz, nnz[0]);
    if(status != rocsparse_status_success)
        return status;
    this->m_m   = iM;
    this->m_nnz = innz;
    return rocsparse_status_success;
}

template <typename T, typename I, typename J>
rocsparse_status rocsparse_importer_rocsparseio::import_sparse_csx(I* ptr, J* ind, T* val)
{

    rocsparseio_status     istatus;
    const rocsparseio_type csr_ptr_type = type_tconvert<I>(), csr_ind_type = type_tconvert<J>(),
                           csr_val_type = type_tconvert<T>();

    const uint64_t M   = this->m_m;
    const uint64_t NNZ = this->m_nnz;

    const bool same_ptr_type = (this->m_ptr_type == csr_ptr_type);
    const bool same_ind_type = (this->m_ind_type == csr_ind_type);
    const bool same_val_type = (this->m_val_type == csr_val_type);
    const bool is_consistent = same_ptr_type && same_ind_type && same_val_type;

    if(is_consistent)
    {

        //
        // Import data.
        //
        istatus = rocsparseiox_read_sparse_csx(this->m_handle, ptr, ind, val);
        ROCSPARSE_CHECK_ROCSPARSEIO(istatus);
    }
    else
    {

        void *tmp_ptr = (void*)ptr, *tmp_ind = (void*)ind, *tmp_val = (void*)val;

        host_dense_vector<char> tmp_ptrv, tmp_indv, tmp_valv;

        uint64_t sizeof_ptr_type, sizeof_ind_type, sizeof_val_type;

        istatus = rocsparseio_type_get_size(this->m_ptr_type, &sizeof_ptr_type);
        ROCSPARSE_CHECK_ROCSPARSEIO(istatus);
        istatus = rocsparseio_type_get_size(this->m_ind_type, &sizeof_ind_type);
        ROCSPARSE_CHECK_ROCSPARSEIO(istatus);
        istatus = rocsparseio_type_get_size(this->m_val_type, &sizeof_val_type);
        ROCSPARSE_CHECK_ROCSPARSEIO(istatus);

        if(!same_ptr_type)
        {
            tmp_ptrv.resize((M + 1) * sizeof_ptr_type);
            tmp_ptr = tmp_ptrv;
        }

        if(!same_ind_type)
        {
            tmp_indv.resize(NNZ * sizeof_ind_type);
            tmp_ind = tmp_indv;
        }

        if(!same_val_type)
        {
            tmp_valv.resize(NNZ * sizeof_val_type);
            tmp_val = tmp_valv;
        }

        istatus = rocsparseiox_read_sparse_csx(this->m_handle, tmp_ptr, tmp_ind, tmp_val);
        ROCSPARSE_CHECK_ROCSPARSEIO(istatus);
        if(!same_ptr_type)
        {
            switch(this->m_ptr_type)
            {
            case rocsparseio_type_int32:
            {
                rocsparse_importer_copy_mixed_arrays(M + 1, ptr, (const int32_t*)tmp_ptr);
                break;
            }

            case rocsparseio_type_int64:
            {
                rocsparse_importer_copy_mixed_arrays(M + 1, ptr, (const int64_t*)tmp_ptr);
                break;
            }

            case rocsparseio_type_int8:
            case rocsparseio_type_float16:
            case rocsparseio_type_bfloat16:
            case rocsparseio_type_float32:
            case rocsparseio_type_float64:
            case rocsparseio_type_complex32:
            case rocsparseio_type_complex64:
            {
                break;
            }
            }
        }

        if(!same_ind_type)
        {
            switch(this->m_ind_type)
            {
            case rocsparseio_type_int32:
            {
                rocsparse_importer_copy_mixed_arrays(NNZ, ind, (const int32_t*)tmp_ind);
                break;
            }
            case rocsparseio_type_int64:
            {
                rocsparse_importer_copy_mixed_arrays(NNZ, ind, (const int64_t*)tmp_ind);
                break;
            }
            case rocsparseio_type_int8:
            case rocsparseio_type_float16:
            case rocsparseio_type_bfloat16:
            case rocsparseio_type_float32:
            case rocsparseio_type_float64:
            case rocsparseio_type_complex32:
            case rocsparseio_type_complex64:
            {
                break;
            }
            }
        }

        if(!same_val_type)
        {
            switch(this->m_val_type)
            {
            case rocsparseio_type_int32:
            case rocsparseio_type_int64:
            {
                break;
            }

            case rocsparseio_type_int8:
            {
                rocsparse_importer_copy_mixed_arrays(NNZ, val, (const int8_t*)tmp_val);
                break;
            }

            case rocsparseio_type_float16:
            {
                rocsparse_importer_copy_mixed_arrays(NNZ, val, (const _Float16*)tmp_val);
                break;
            }

            case rocsparseio_type_bfloat16:
            {
                rocsparse_importer_copy_mixed_arrays(NNZ, val, (const rocsparse_bfloat16*)tmp_val);
                break;
            }

            case rocsparseio_type_float32:
            {
                rocsparse_importer_copy_mixed_arrays(NNZ, val, (const float*)tmp_val);
                break;
            }

            case rocsparseio_type_float64:
            {
                rocsparse_importer_copy_mixed_arrays(NNZ, val, (const double*)tmp_val);
                break;
            }

            case rocsparseio_type_complex32:
            {
                rocsparse_importer_copy_mixed_arrays(
                    NNZ, val, (const rocsparse_float_complex*)tmp_val);
                break;
            }

            case rocsparseio_type_complex64:
            {
                rocsparse_importer_copy_mixed_arrays(
                    NNZ, val, (const rocsparse_double_complex*)tmp_val);
                break;
            }
            }
        }
    }

    return rocsparse_status_success;
}

#define INSTANTIATE_TIJ(T, I, J)                                                             \
    template rocsparse_status rocsparse_importer_rocsparseio::import_sparse_csx(I*, J*, T*); \
    template rocsparse_status rocsparse_importer_rocsparseio::import_sparse_gebsx(I*, J*, T*)

#define INSTANTIATE_TI(T, I)                                                     \
    template rocsparse_status rocsparse_importer_rocsparseio::import_sparse_coo( \
        I* row_ind, I* col_ind, T* val)

#define INSTANTIATE_I(I)                                                         \
    template rocsparse_status rocsparse_importer_rocsparseio::import_sparse_coo( \
        I* m, I* n, int64_t* nnz, rocsparse_index_base* base)

#define INSTANTIATE_IJ(I, J)                                                       \
    template rocsparse_status rocsparse_importer_rocsparseio::import_sparse_csx(   \
        rocsparse_direction*, J*, J*, I*, rocsparse_index_base*);                  \
    template rocsparse_status rocsparse_importer_rocsparseio::import_sparse_gebsx( \
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
