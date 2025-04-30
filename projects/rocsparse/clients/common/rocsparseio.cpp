/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc.
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
#include "rocsparseio.hpp"
#include <iostream>

extern "C" rocsparseio_status rocsparseio_open(rocsparseio_handle* p_handle_,
                                               rocsparseio_rwmode  mode_,
                                               const char*         filename_,
                                               ...)
{
    ROCSPARSEIO_C_CHECK_ARG(!p_handle_, rocsparseio_status_invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::rwmode_t(mode_).is_invalid(),
                            rocsparseio_status_invalid_value);
    ROCSPARSEIO_C_CHECK_ARG(filename_ == nullptr, rocsparseio_status_invalid_pointer);

    {
        va_list args;
        va_start(args, filename_);
        ROCSPARSEIO_C_CHECK(rocsparseio::open(p_handle_, mode_, filename_, args));
        va_end(args);
    }

    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseio_close(rocsparseio_handle handle_)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle_, rocsparseio_status_invalid_handle);
    ROCSPARSEIO_C_CHECK(rocsparseio::close(handle_));
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseio_read_format(rocsparseio_handle  handle_,
                                                      rocsparseio_format* format_)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle_, rocsparseio_status_invalid_handle);
    rocsparseio::format_t f;
    ROCSPARSEIO_C_CHECK(rocsparseio::read_format(handle_, &f));
    format_[0] = (rocsparseio_format)f;
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseio_read_name(rocsparseio_handle handle_,
                                                    rocsparseio_string name_)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle_, rocsparseio_status_invalid_handle);
    ROCSPARSEIO_C_CHECK(rocsparseio::read_name(handle_, name_));
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseio_type_get_size(rocsparseio_type type_, uint64_t* p_size_)
{
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::type_t(type_).is_invalid(),
                            rocsparseio_status_invalid_value);
    ROCSPARSEIO_C_CHECK_ARG(!p_size_, rocsparseio_status_invalid_pointer);
    rocsparseio::type_t type = type_;
    p_size_[0]               = type.size();
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseio_write_dense_vector(rocsparseio_handle handle_,
                                                             rocsparseio_type   data_type_,
                                                             uint64_t           data_nmemb_,
                                                             const void*        data_,
                                                             uint64_t           data_ld_,
                                                             const char*        name_,
                                                             ...)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle_, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::type_t(data_type_).is_invalid(),
                            rocsparseio::status_t::invalid_value);

    ROCSPARSEIO_C_CHECK_ARG((0 != data_nmemb_) && (!data_), rocsparseio::status_t::invalid_pointer);

    {
        va_list args;
        va_start(args, name_);
        ROCSPARSEIO_C_CHECK(rocsparseio::write_dense_vector(
            handle_, data_type_, data_nmemb_, data_, data_ld_, name_, args));
        va_end(args);
    }

    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseiox_read_metadata_dense_vector(rocsparseio_handle handle_,
                                                                      rocsparseio_type*  data_type_,
                                                                      uint64_t* data_nmemb_)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle_, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(!data_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!data_nmemb_, rocsparseio::status_t::invalid_pointer);
    rocsparseio::type_t type;
    ROCSPARSEIO_C_CHECK(rocsparseio::read_metadata_dense_vector(handle_, &type, data_nmemb_));
    data_type_[0] = (rocsparseio_type)type;
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status
    rocsparseiox_read_dense_vector(rocsparseio_handle handle_, void* data_, uint64_t data_inc_)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle_, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK(rocsparseio::read_dense_vector(handle_, data_, data_inc_));
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseio_read_dense_vector(rocsparseio_handle handle_,
                                                            rocsparseio_type*  data_type_,
                                                            uint64_t*          data_nmemb_,
                                                            void**             data_)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle_, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(!data_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!data_nmemb_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!data_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK(rocsparseiox_read_metadata_dense_vector(handle_, data_type_, data_nmemb_));
    data_[0] = nullptr;
    if(data_nmemb_[0] > 0)
    {
        data_[0] = malloc(rocsparseio::type_t(data_type_[0]).size() * data_nmemb_[0]);
        if(!data_[0])
        {
            return rocsparseio_status_invalid_memory;
        }
        ROCSPARSEIO_C_CHECK(rocsparseiox_read_dense_vector(handle_, data_[0], 1));
    }
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseio_read_dense_matrix(rocsparseio_handle handle_,
                                                            rocsparseio_order* data_order_,
                                                            uint64_t*          m_,
                                                            uint64_t*          n_,
                                                            rocsparseio_type*  data_type_,
                                                            void**             data_)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle_, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(!data_order_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!m_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!n_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!data_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!data_, rocsparseio::status_t::invalid_pointer);

    rocsparseio::order_t order;
    rocsparseio::type_t  type;
    ROCSPARSEIO_C_CHECK(rocsparseio::read_metadata_dense_matrix(handle_, &order, m_, n_, &type));

    data_order_[0] = (rocsparseio_order)order;
    data_type_[0]  = (rocsparseio_type)type;
    data_[0]       = malloc(type.size() * m_[0] * n_[0]);
    if(!data_[0])
    {
        return rocsparseio_status_invalid_memory;
    }

    ROCSPARSEIO_C_CHECK(rocsparseiox_read_dense_matrix(
        handle_, data_[0], (order == rocsparseio::order_t::row) ? n_[0] : m_[0]));
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status
    rocsparseiox_read_metadata_dense_matrix(rocsparseio_handle handle_,
                                            rocsparseio_order* data_order_,
                                            uint64_t*          m_,
                                            uint64_t*          n_,
                                            rocsparseio_type*  data_type_)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle_, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(!data_order_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!m_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!n_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!data_type_, rocsparseio::status_t::invalid_pointer);
    rocsparseio::order_t order;
    rocsparseio::type_t  type;
    ROCSPARSEIO_C_CHECK(rocsparseio::read_metadata_dense_matrix(handle_, &order, m_, n_, &type));
    data_order_[0] = (rocsparseio_order)order;
    data_type_[0]  = (rocsparseio_type)type;
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status
    rocsparseiox_read_dense_matrix(rocsparseio_handle handle_, void* data_, uint64_t data_ld_)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle_, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK(rocsparseio::read_dense_matrix(handle_, data_, data_ld_));
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseio_write_dense_matrix(rocsparseio_handle handle_,
                                                             rocsparseio_order  order_,
                                                             uint64_t           m_,
                                                             uint64_t           n_,
                                                             rocsparseio_type   data_type_,
                                                             const void*        data_,
                                                             uint64_t           data_ld_,
                                                             const char*        name_,
                                                             ...)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle_, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::order_t(order_).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::type_t(data_type_).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG((0 != m_) && (0 != n_) && (!data_),
                            rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG((order_ == rocsparseio_order_row) && (data_ld_ < n_),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG((order_ == rocsparseio_order_column) && (data_ld_ < m_),
                            rocsparseio::status_t::invalid_value);

    {
        va_list args;
        va_start(args, name_);
        ROCSPARSEIO_C_CHECK(rocsparseio::write_dense_matrix(
            handle_, order_, m_, n_, data_type_, data_, data_ld_, name_, args));
        va_end(args);
    }

    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseio_write_sparse_coo(rocsparseio_handle handle_,
                                                           uint64_t           m_,
                                                           uint64_t           n_,
                                                           uint64_t           nnz_,

                                                           rocsparseio_type row_ind_type_,
                                                           const void*      row_ind_,

                                                           rocsparseio_type col_ind_type_,
                                                           const void*      col_ind_,

                                                           rocsparseio_type       val_type_,
                                                           const void*            val_,
                                                           rocsparseio_index_base base_,
                                                           const char*            name_,
                                                           ...)
{

    ROCSPARSEIO_C_CHECK_ARG(!handle_, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::type_t(row_ind_type_).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::type_t(col_ind_type_).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::type_t(val_type_).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::index_base_t(base_).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG((nnz_ > 0 && !row_ind_), rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG((nnz_ > 0 && !col_ind_), rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG((nnz_ > 0 && !val_), rocsparseio::status_t::invalid_pointer);

    {
        va_list args;
        va_start(args, name_);
        ROCSPARSEIO_C_CHECK(rocsparseio::write_sparse_coo(handle_,
                                                          m_,
                                                          n_,
                                                          nnz_,
                                                          row_ind_type_,
                                                          row_ind_,
                                                          col_ind_type_,
                                                          col_ind_,
                                                          val_type_,
                                                          val_,
                                                          base_,
                                                          name_,
                                                          args));
        va_end(args);
    }

    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseiox_read_metadata_sparse_coo(rocsparseio_handle handle_,
                                                                    uint64_t*          m_,
                                                                    uint64_t*          n_,
                                                                    uint64_t*          nnz_,

                                                                    rocsparseio_type* row_ind_type_,
                                                                    rocsparseio_type* col_ind_type_,
                                                                    rocsparseio_type* val_type_,
                                                                    rocsparseio_index_base* base_)
{

    ROCSPARSEIO_C_CHECK_ARG(!handle_, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(!m_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!n_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!nnz_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!row_ind_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!col_ind_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!val_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!base_, rocsparseio::status_t::invalid_pointer);
    rocsparseio::type_t       row_ind_type;
    rocsparseio::type_t       col_ind_type;
    rocsparseio::type_t       val_type;
    rocsparseio::index_base_t index_base;
    ROCSPARSEIO_C_CHECK(rocsparseio::read_metadata_sparse_coo(
        handle_, m_, n_, nnz_, &row_ind_type, &col_ind_type, &val_type, &index_base));
    row_ind_type_[0] = (rocsparseio_type)row_ind_type;
    col_ind_type_[0] = (rocsparseio_type)col_ind_type;
    val_type_[0]     = (rocsparseio_type)val_type;
    base_[0]         = (rocsparseio_index_base)index_base;
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseiox_read_sparse_coo(rocsparseio_handle handle_,
                                                           void*              row_ind_,
                                                           void*              col_ind_,
                                                           void*              val_)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle_, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK(rocsparseio::read_sparse_coo(handle_, row_ind_, col_ind_, val_));
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseio_read_sparse_coo(rocsparseio_handle handle_,
                                                          uint64_t*          m_,
                                                          uint64_t*          n_,
                                                          uint64_t*          nnz_,

                                                          rocsparseio_type* row_ind_type_,
                                                          void**            row_ind_,

                                                          rocsparseio_type* col_ind_type_,
                                                          void**            col_ind_,

                                                          rocsparseio_type*       val_type_,
                                                          void**                  val_,
                                                          rocsparseio_index_base* base_)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle_, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(!m_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!n_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!nnz_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!row_ind_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!col_ind_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!val_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!row_ind_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!col_ind_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!val_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!base_, rocsparseio::status_t::invalid_pointer);

    rocsparseio::type_t       row_ind_type;
    rocsparseio::type_t       col_ind_type;
    rocsparseio::type_t       val_type;
    rocsparseio::index_base_t index_base;
    ROCSPARSEIO_C_CHECK(rocsparseio::read_metadata_sparse_coo(
        handle_, m_, n_, nnz_, &row_ind_type, &col_ind_type, &val_type, &index_base));
    row_ind_type_[0] = (rocsparseio_type)row_ind_type;
    col_ind_type_[0] = (rocsparseio_type)col_ind_type;
    val_type_[0]     = (rocsparseio_type)val_type;
    base_[0]         = (rocsparseio_index_base)index_base;

    row_ind_[0] = malloc(row_ind_type.size() * nnz_[0]);

    if(!row_ind_[0])
    {
        return rocsparseio_status_invalid_memory;
    }

    col_ind_[0] = malloc(col_ind_type.size() * (nnz_[0]));

    if(!col_ind_[0])
    {
        return rocsparseio_status_invalid_memory;
    }

    val_[0] = malloc(val_type.size() * (nnz_[0]));

    if(!val_[0])
    {
        return rocsparseio_status_invalid_memory;
    }

    ROCSPARSEIO_C_CHECK(rocsparseiox_read_sparse_coo(handle_, row_ind_[0], col_ind_[0], val_[0]));
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseio_write_sparse_csx(rocsparseio_handle    handle_,
                                                           rocsparseio_direction dir_,
                                                           uint64_t              m_,
                                                           uint64_t              n_,
                                                           uint64_t              nnz_,

                                                           rocsparseio_type ptr_type_,
                                                           const void*      ptr_,

                                                           rocsparseio_type ind_type_,
                                                           const void*      ind_,

                                                           rocsparseio_type       val_type_,
                                                           const void*            val_,
                                                           rocsparseio_index_base base_,
                                                           const char*            name_,
                                                           ...)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle_, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::direction_t(dir_).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::type_t(ptr_type_).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::type_t(ind_type_).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::type_t(val_type_).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::index_base_t(base_).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG((m_ > 0 && !ptr_), rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG((nnz_ > 0 && !ind_), rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG((nnz_ > 0 && !val_), rocsparseio::status_t::invalid_pointer);

    {
        va_list args;
        va_start(args, name_);
        ROCSPARSEIO_C_CHECK(rocsparseio::write_sparse_csx(handle_,
                                                          dir_,
                                                          m_,
                                                          n_,
                                                          nnz_,
                                                          ptr_type_,
                                                          ptr_,
                                                          ind_type_,
                                                          ind_,
                                                          val_type_,
                                                          val_,
                                                          base_,
                                                          name_,
                                                          args));
        va_end(args);
    }

    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseiox_read_metadata_sparse_csx(rocsparseio_handle     handle_,
                                                                    rocsparseio_direction* dir_,
                                                                    uint64_t*              m_,
                                                                    uint64_t*              n_,
                                                                    uint64_t*              nnz_,

                                                                    rocsparseio_type* ptr_type_,
                                                                    rocsparseio_type* ind_type_,
                                                                    rocsparseio_type* val_type_,
                                                                    rocsparseio_index_base* base_)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle_, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(!dir_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!m_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!n_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!nnz_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!ptr_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!ind_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!val_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!base_, rocsparseio::status_t::invalid_pointer);
    rocsparseio::direction_t  dir;
    rocsparseio::type_t       ptr_type;
    rocsparseio::type_t       ind_type;
    rocsparseio::type_t       val_type;
    rocsparseio::index_base_t index_base;

    ROCSPARSEIO_C_CHECK(rocsparseio::read_metadata_sparse_csx(
        handle_, &dir, m_, n_, nnz_, &ptr_type, &ind_type, &val_type, &index_base));
    dir_[0]      = (rocsparseio_direction)dir;
    ptr_type_[0] = (rocsparseio_type)ptr_type;
    ind_type_[0] = (rocsparseio_type)ind_type;
    val_type_[0] = (rocsparseio_type)val_type;
    base_[0]     = (rocsparseio_index_base)index_base;
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status
    rocsparseiox_read_sparse_csx(rocsparseio_handle handle_, void* ptr_, void* ind_, void* val_)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle_, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK(rocsparseio::read_sparse_csx(handle_, ptr_, ind_, val_));
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseio_read_sparse_csx(rocsparseio_handle     handle_,
                                                          rocsparseio_direction* dir_,
                                                          uint64_t*              m_,
                                                          uint64_t*              n_,
                                                          uint64_t*              nnz_,

                                                          rocsparseio_type* ptr_type_,
                                                          void**            ptr_,

                                                          rocsparseio_type* ind_type_,
                                                          void**            ind_,

                                                          rocsparseio_type*       val_type_,
                                                          void**                  val_,
                                                          rocsparseio_index_base* base_)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle_, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(!dir_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!m_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!n_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!nnz_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!ptr_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!ind_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!val_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!ptr_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!ind_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!val_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!base_, rocsparseio::status_t::invalid_pointer);

    ROCSPARSEIO_C_CHECK(rocsparseiox_read_metadata_sparse_csx(
        handle_, dir_, m_, n_, nnz_, ptr_type_, ind_type_, val_type_, base_));
    uint64_t                 ptr_size = 0;
    rocsparseio::type_t      ind_type = ind_type_[0];
    rocsparseio::type_t      val_type = val_type_[0];
    rocsparseio::type_t      ptr_type = ptr_type_[0];
    rocsparseio::direction_t dir      = dir_[0];
    switch(dir)
    {
    case rocsparseio::direction_t::row:
    {
        ptr_size = m_[0] + 1;
        break;
    }
    case rocsparseio::direction_t::column:
    {
        ptr_size = n_[0] + 1;
        break;
    }
    }

    ptr_[0] = malloc(ptr_type.size() * ptr_size);

    if(!ptr_[0])
    {
        return rocsparseio_status_invalid_memory;
    }

    ind_[0] = malloc(ind_type.size() * (nnz_[0]));

    if(!ind_[0])
    {
        return rocsparseio_status_invalid_memory;
    }

    val_[0] = malloc(val_type.size() * (nnz_[0]));

    if(!val_[0])
    {
        return rocsparseio_status_invalid_memory;
    }

    ROCSPARSEIO_C_CHECK(rocsparseiox_read_sparse_csx(handle_, ptr_[0], ind_[0], val_[0]));
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseio_write_sparse_gebsx(rocsparseio_handle handle_,

                                                             rocsparseio_direction dir_,
                                                             rocsparseio_direction dirb_,

                                                             uint64_t mb_,
                                                             uint64_t nb_,
                                                             uint64_t nnzb_,

                                                             uint64_t row_block_dim_,
                                                             uint64_t col_block_dim_,

                                                             rocsparseio_type ptr_type_,
                                                             const void*      ptr_,

                                                             rocsparseio_type ind_type_,
                                                             const void*      ind_,

                                                             rocsparseio_type       val_type_,
                                                             const void*            val_,
                                                             rocsparseio_index_base base_,
                                                             const char*            name_,
                                                             ...)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle_, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::direction_t(dir_).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::direction_t(dirb_).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::type_t(ptr_type_).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::type_t(ind_type_).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::type_t(val_type_).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::index_base_t(base_).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG((mb_ > 0 && !ptr_), rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG((nnzb_ > 0 && !ind_), rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG((nnzb_ > 0 && !val_), rocsparseio::status_t::invalid_pointer);

    ROCSPARSEIO_C_CHECK_ARG(0 == handle_, rocsparseio::status_t::invalid_handle);

    {
        va_list args;
        va_start(args, name_);
        ROCSPARSEIO_C_CHECK(rocsparseio::write_sparse_gebsx(handle_,
                                                            dir_,
                                                            dirb_,
                                                            mb_,
                                                            nb_,
                                                            nnzb_,
                                                            row_block_dim_,
                                                            col_block_dim_,
                                                            ptr_type_,
                                                            ptr_,
                                                            ind_type_,
                                                            ind_,
                                                            val_type_,
                                                            val_,
                                                            base_,
                                                            name_,
                                                            args));
        va_end(args);
    }

    return rocsparseio_status_success;
}
extern "C" rocsparseio_status rocsparseiox_read_metadata_sparse_gebsx(rocsparseio_handle handle_,
                                                                      rocsparseio_direction* dir_,
                                                                      rocsparseio_direction* dirb_,
                                                                      uint64_t*              mb_,
                                                                      uint64_t*              nb_,
                                                                      uint64_t*              nnzb_,
                                                                      uint64_t* row_block_dim_,
                                                                      uint64_t* col_block_dim_,
                                                                      rocsparseio_type* ptr_type_,
                                                                      rocsparseio_type* ind_type_,
                                                                      rocsparseio_type* val_type_,
                                                                      rocsparseio_index_base* base_)
{

    ROCSPARSEIO_C_CHECK_ARG(!handle_, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(!dir_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!dirb_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!mb_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!nb_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!nnzb_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!row_block_dim_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!col_block_dim_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!ptr_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!ind_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!val_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!base_, rocsparseio::status_t::invalid_pointer);
    rocsparseio::direction_t  dir;
    rocsparseio::direction_t  dirb;
    rocsparseio::type_t       ptr_type;
    rocsparseio::type_t       ind_type;
    rocsparseio::type_t       val_type;
    rocsparseio::index_base_t index_base;
    ROCSPARSEIO_C_CHECK(rocsparseio::read_metadata_sparse_gebsx(handle_,
                                                                &dir,
                                                                &dirb,
                                                                mb_,
                                                                nb_,
                                                                nnzb_,
                                                                row_block_dim_,
                                                                col_block_dim_,
                                                                &ptr_type,
                                                                &ind_type,
                                                                &val_type,
                                                                &index_base));
    dir_[0]      = (rocsparseio_direction)dir;
    dirb_[0]     = (rocsparseio_direction)dirb;
    ptr_type_[0] = (rocsparseio_type)ptr_type;
    ind_type_[0] = (rocsparseio_type)ind_type;
    val_type_[0] = (rocsparseio_type)val_type;
    base_[0]     = (rocsparseio_index_base)index_base;
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status
    rocsparseiox_read_sparse_gebsx(rocsparseio_handle handle_, void* ptr_, void* ind_, void* val_)
{
    ROCSPARSEIO_C_CHECK_ARG(0 == handle_, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK(rocsparseio::read_sparse_gebsx(handle_, ptr_, ind_, val_));
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseio_read_sparse_gebsx(rocsparseio_handle handle_,

                                                            rocsparseio_direction* dir_,
                                                            rocsparseio_direction* dirb_,

                                                            uint64_t* mb_,
                                                            uint64_t* nb_,
                                                            uint64_t* nnzb_,

                                                            uint64_t* row_block_dim_,
                                                            uint64_t* col_block_dim_,

                                                            rocsparseio_type* ptr_type_,
                                                            void**            ptr_,

                                                            rocsparseio_type* ind_type_,
                                                            void**            ind_,

                                                            rocsparseio_type*       val_type_,
                                                            void**                  val_,
                                                            rocsparseio_index_base* base_)
{
    ROCSPARSEIO_C_CHECK_ARG(handle_ == 0, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(!dir_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!dirb_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!mb_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!nb_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!nnzb_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!row_block_dim_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!col_block_dim_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!ptr_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!ind_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!val_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!ptr_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!ind_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!val_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!base_, rocsparseio::status_t::invalid_pointer);

    ROCSPARSEIO_C_CHECK(rocsparseiox_read_metadata_sparse_gebsx(handle_,
                                                                dir_,
                                                                dirb_,
                                                                mb_,
                                                                nb_,
                                                                nnzb_,
                                                                row_block_dim_,
                                                                col_block_dim_,
                                                                ptr_type_,
                                                                ind_type_,
                                                                val_type_,
                                                                base_));

    rocsparseio::direction_t direction = dir_[0];

    uint64_t ptr_size = 0;
    switch(direction)
    {
    case rocsparseio::direction_t::row:
    {
        ptr_size = mb_[0] + 1;
        break;
    }
    case rocsparseio::direction_t::column:
    {
        ptr_size = nb_[0] + 1;
        break;
    }
    }

    rocsparseio::type_t ptr_type = ptr_type_[0];
    ptr_[0]                      = malloc(ptr_type.size() * ptr_size);

    if(!ptr_[0])
    {
        return rocsparseio_status_invalid_memory;
    }

    rocsparseio::type_t ind_type = ind_type_[0];
    ind_[0]                      = malloc(ind_type.size() * (nnzb_[0]));

    if(!ind_[0])
    {
        return rocsparseio_status_invalid_memory;
    }

    rocsparseio::type_t val_type = val_type_[0];
    val_[0] = malloc(val_type.size() * (nnzb_[0] * row_block_dim_[0] * col_block_dim_[0]));

    if(!val_[0])
    {
        return rocsparseio_status_invalid_memory;
    }

    return rocsparseiox_read_sparse_gebsx(handle_, ptr_[0], ind_[0], val_[0]);
}

extern "C" rocsparseio_status rocsparseio_write_sparse_ell(rocsparseio_handle     handle_,
                                                           uint64_t               m_,
                                                           uint64_t               n_,
                                                           uint64_t               width_,
                                                           rocsparseio_type       ind_type_,
                                                           const void*            ind_,
                                                           rocsparseio_type       val_type_,
                                                           const void*            val_,
                                                           rocsparseio_index_base base_,
                                                           const char*            name_,
                                                           ...)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle_, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::type_t(ind_type_).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::type_t(val_type_).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::index_base_t(base_).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG((m_ * width_ > 0 && !ind_), rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG((m_ * width_ > 0 && !val_), rocsparseio::status_t::invalid_pointer);

    {
        va_list args;
        va_start(args, name_);
        ROCSPARSEIO_C_CHECK(rocsparseio::write_sparse_ell(
            handle_, m_, n_, width_, ind_type_, ind_, val_type_, val_, base_, name_, args));
        va_end(args);
    }

    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseio_read_sparse_ell(rocsparseio_handle      handle_,
                                                          uint64_t*               m_,
                                                          uint64_t*               n_,
                                                          uint64_t*               width_,
                                                          rocsparseio_type*       ind_type_,
                                                          void**                  ind_,
                                                          rocsparseio_type*       val_type_,
                                                          void**                  val_,
                                                          rocsparseio_index_base* base_)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle_, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(!m_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!n_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!width_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!ind_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!val_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!ind_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!val_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!base_, rocsparseio::status_t::invalid_pointer);

    rocsparseio::type_t       ind_type;
    rocsparseio::type_t       val_type;
    rocsparseio::index_base_t index_base;
    ROCSPARSEIO_C_CHECK(rocsparseio::read_metadata_sparse_ell(
        handle_, m_, n_, width_, &ind_type, &val_type, &index_base));
    ind_type_[0] = (rocsparseio_type)ind_type;
    val_type_[0] = (rocsparseio_type)val_type;
    base_[0]     = (rocsparseio_index_base)index_base;

    const uint64_t nnz = m_[0] * width_[0];
    ind_[0]            = malloc(ind_type.size() * nnz);
    if(!ind_[0])
    {
        return rocsparseio_status_invalid_memory;
    }

    val_[0] = malloc(val_type.size() * nnz);
    if(!val_[0])
    {
        return rocsparseio_status_invalid_memory;
    }

    ROCSPARSEIO_C_CHECK(rocsparseiox_read_sparse_ell(handle_, ind_[0], val_[0]));
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseiox_read_metadata_sparse_ell(rocsparseio_handle handle_,
                                                                    uint64_t*          m_,
                                                                    uint64_t*          n_,
                                                                    uint64_t*          width_,
                                                                    rocsparseio_type*  ind_type_,
                                                                    rocsparseio_type*  val_type_,
                                                                    rocsparseio_index_base* base_)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle_, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(!m_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!n_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!width_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!ind_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!val_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!base_, rocsparseio::status_t::invalid_pointer);
    rocsparseio::type_t       ind_type;
    rocsparseio::type_t       val_type;
    rocsparseio::index_base_t index_base;
    ROCSPARSEIO_C_CHECK(rocsparseio::read_metadata_sparse_ell(
        handle_, m_, n_, width_, &ind_type, &val_type, &index_base));
    ind_type_[0] = (rocsparseio_type)ind_type;
    val_type_[0] = (rocsparseio_type)val_type;
    base_[0]     = (rocsparseio_index_base)index_base;
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status
    rocsparseiox_read_sparse_ell(rocsparseio_handle handle_, void* ind_, void* val_)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle_, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK(rocsparseio::read_sparse_ell(handle_, ind_, val_));
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseio_write_sparse_dia(rocsparseio_handle     handle_,
                                                           uint64_t               m_,
                                                           uint64_t               n_,
                                                           uint64_t               ndiag_,
                                                           rocsparseio_type       ind_type_,
                                                           const void*            ind_,
                                                           rocsparseio_type       val_type_,
                                                           const void*            val_,
                                                           rocsparseio_index_base base_,
                                                           const char*            name_,
                                                           ...)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle_, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::type_t(ind_type_).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::type_t(val_type_).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::index_base_t(base_).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG((ndiag_ > 0 && !ind_), rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG((ndiag_ > 0 && (std::min(m_, n_) > 0) && !val_),
                            rocsparseio::status_t::invalid_pointer);

    {
        va_list args;
        va_start(args, name_);
        ROCSPARSEIO_C_CHECK(rocsparseio::write_sparse_dia(
            handle_, m_, n_, ndiag_, ind_type_, ind_, val_type_, val_, base_, name_, args));
        va_end(args);
    }

    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseio_read_sparse_dia(rocsparseio_handle      handle_,
                                                          uint64_t*               m_,
                                                          uint64_t*               n_,
                                                          uint64_t*               ndiag_,
                                                          rocsparseio_type*       ind_type_,
                                                          void**                  ind_,
                                                          rocsparseio_type*       val_type_,
                                                          void**                  val_,
                                                          rocsparseio_index_base* base_)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle_, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(!m_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!n_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!ndiag_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!ind_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!val_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!ind_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!val_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!base_, rocsparseio::status_t::invalid_pointer);

    rocsparseio::type_t       ind_type;
    rocsparseio::type_t       val_type;
    rocsparseio::index_base_t index_base;
    ROCSPARSEIO_C_CHECK(rocsparseio::read_metadata_sparse_dia(
        handle_, m_, n_, ndiag_, &ind_type, &val_type, &index_base));
    ind_type_[0] = (rocsparseio_type)ind_type;
    val_type_[0] = (rocsparseio_type)val_type;
    base_[0]     = (rocsparseio_index_base)index_base;

    ind_[0] = malloc(ind_type.size() * ndiag_[0]);
    if(!ind_[0])
    {
        return rocsparseio_status_invalid_memory;
    }

    const uint64_t nnz = std::min(m_[0], n_[0]) * ndiag_[0];
    val_[0]            = malloc(val_type.size() * nnz);
    if(!val_[0])
    {
        return rocsparseio_status_invalid_memory;
    }

    ROCSPARSEIO_C_CHECK(rocsparseiox_read_sparse_dia(handle_, ind_[0], val_[0]));
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseiox_read_metadata_sparse_dia(rocsparseio_handle handle_,
                                                                    uint64_t*          m_,
                                                                    uint64_t*          n_,
                                                                    uint64_t*          ndiag_,
                                                                    rocsparseio_type*  ind_type_,
                                                                    rocsparseio_type*  val_type_,
                                                                    rocsparseio_index_base* base_)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle_, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(!m_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!n_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!ndiag_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!ind_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!val_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!base_, rocsparseio::status_t::invalid_pointer);
    rocsparseio::type_t       ind_type;
    rocsparseio::type_t       val_type;
    rocsparseio::index_base_t index_base;
    ROCSPARSEIO_C_CHECK(rocsparseio::read_metadata_sparse_dia(
        handle_, m_, n_, ndiag_, &ind_type, &val_type, &index_base));
    ind_type_[0] = (rocsparseio_type)ind_type;
    val_type_[0] = (rocsparseio_type)val_type;
    base_[0]     = (rocsparseio_index_base)index_base;
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status
    rocsparseiox_read_sparse_dia(rocsparseio_handle handle_, void* ind_, void* val_)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle_, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK(rocsparseio::read_sparse_dia(handle_, ind_, val_));
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseio_write_sparse_hyb(rocsparseio_handle     handle_,
                                                           uint64_t               m_,
                                                           uint64_t               n_,
                                                           uint64_t               coo_nnz_,
                                                           rocsparseio_type       coo_row_ind_type_,
                                                           const void*            coo_row_ind_,
                                                           rocsparseio_type       coo_col_ind_type_,
                                                           const void*            coo_col_ind_,
                                                           rocsparseio_type       coo_val_type_,
                                                           const void*            coo_val_,
                                                           rocsparseio_index_base coo_base_,

                                                           uint64_t               ell_width_,
                                                           rocsparseio_type       ell_ind_type_,
                                                           const void*            ell_ind_,
                                                           rocsparseio_type       ell_val_type_,
                                                           const void*            ell_val_,
                                                           rocsparseio_index_base ell_base_,
                                                           const char*            name_,
                                                           ...)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle_, rocsparseio::status_t::invalid_handle);

    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::type_t(coo_row_ind_type_).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::type_t(coo_col_ind_type_).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::type_t(coo_val_type_).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::index_base_t(coo_base_).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG((coo_nnz_ > 0 && !coo_row_ind_),
                            rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG((coo_nnz_ > 0 && !coo_col_ind_),
                            rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG((coo_nnz_ > 0 && !coo_val_), rocsparseio::status_t::invalid_pointer);

    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::type_t(ell_ind_type_).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::type_t(ell_val_type_).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::index_base_t(ell_base_).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG((m_ * ell_width_ > 0 && !ell_ind_),
                            rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG((m_ * ell_width_ > 0 && !ell_val_),
                            rocsparseio::status_t::invalid_pointer);

    {
        va_list args;
        va_start(args, name_);
        ROCSPARSEIO_C_CHECK(rocsparseio::write_sparse_hyb(handle_,
                                                          m_,
                                                          n_,
                                                          coo_nnz_,
                                                          coo_row_ind_type_,
                                                          coo_row_ind_,
                                                          coo_col_ind_type_,
                                                          coo_col_ind_,
                                                          coo_val_type_,
                                                          coo_val_,
                                                          coo_base_,

                                                          ell_width_,
                                                          ell_ind_type_,
                                                          ell_ind_,
                                                          ell_val_type_,
                                                          ell_val_,
                                                          ell_base_,
                                                          name_,
                                                          args));

        va_end(args);
    }

    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseio_read_sparse_hyb(rocsparseio_handle      handle_,
                                                          uint64_t*               m_,
                                                          uint64_t*               n_,
                                                          uint64_t*               coo_nnz_,
                                                          rocsparseio_type*       coo_row_ind_type_,
                                                          void**                  coo_row_ind_,
                                                          rocsparseio_type*       coo_col_ind_type_,
                                                          void**                  coo_col_ind_,
                                                          rocsparseio_type*       coo_val_type_,
                                                          void**                  coo_val_,
                                                          rocsparseio_index_base* coo_base_,
                                                          uint64_t*               ell_width_,
                                                          rocsparseio_type*       ell_ind_type_,
                                                          void**                  ell_ind_,
                                                          rocsparseio_type*       ell_val_type_,
                                                          void**                  ell_val_,
                                                          rocsparseio_index_base* ell_base_)
{

    ROCSPARSEIO_C_CHECK_ARG(!handle_, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(!m_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!n_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!coo_nnz_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!coo_row_ind_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!coo_col_ind_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!coo_val_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!coo_row_ind_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!coo_col_ind_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!coo_val_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!coo_base_, rocsparseio::status_t::invalid_pointer);

    ROCSPARSEIO_C_CHECK_ARG(!ell_width_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!ell_ind_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!ell_val_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!ell_ind_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!ell_val_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!ell_base_, rocsparseio::status_t::invalid_pointer);

    rocsparseio::type_t       coo_row_ind_type;
    rocsparseio::type_t       coo_col_ind_type;
    rocsparseio::type_t       coo_val_type;
    rocsparseio::index_base_t coo_index_base;
    rocsparseio::type_t       ell_ind_type;
    rocsparseio::type_t       ell_val_type;
    rocsparseio::index_base_t ell_index_base;

    ROCSPARSEIO_C_CHECK(rocsparseio::read_metadata_sparse_hyb(handle_,
                                                              m_,
                                                              n_,
                                                              coo_nnz_,
                                                              &coo_row_ind_type,
                                                              &coo_col_ind_type,
                                                              &coo_val_type,
                                                              &coo_index_base,
                                                              ell_width_,
                                                              &ell_ind_type,
                                                              &ell_val_type,
                                                              &ell_index_base));
    coo_row_ind_type_[0] = (rocsparseio_type)coo_row_ind_type;
    coo_col_ind_type_[0] = (rocsparseio_type)coo_col_ind_type;
    coo_val_type_[0]     = (rocsparseio_type)coo_val_type;
    coo_base_[0]         = (rocsparseio_index_base)coo_index_base;

    ell_ind_type_[0] = (rocsparseio_type)ell_ind_type;
    ell_val_type_[0] = (rocsparseio_type)ell_val_type;
    ell_base_[0]     = (rocsparseio_index_base)ell_index_base;

    coo_row_ind_[0] = malloc(coo_row_ind_type.size() * coo_nnz_[0]);
    if(!coo_row_ind_[0])
    {
        return rocsparseio_status_invalid_memory;
    }

    coo_col_ind_[0] = malloc(coo_col_ind_type.size() * (coo_nnz_[0]));
    if(!coo_col_ind_[0])
    {
        return rocsparseio_status_invalid_memory;
    }

    coo_val_[0] = malloc(coo_val_type.size() * (coo_nnz_[0]));
    if(!coo_val_[0])
    {
        return rocsparseio_status_invalid_memory;
    }

    const uint64_t ell_nnz = m_[0] * ell_width_[0];
    ell_ind_[0]            = malloc(ell_ind_type.size() * ell_nnz);
    if(!ell_ind_[0])
    {
        return rocsparseio_status_invalid_memory;
    }

    ell_val_[0] = malloc(ell_val_type.size() * ell_nnz);
    if(!ell_val_[0])
    {
        return rocsparseio_status_invalid_memory;
    }

    ROCSPARSEIO_C_CHECK(rocsparseiox_read_sparse_hyb(
        handle_, coo_row_ind_[0], coo_col_ind_[0], coo_val_[0], ell_ind_[0], ell_val_[0]));
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status
    rocsparseiox_read_metadata_sparse_hyb(rocsparseio_handle      handle_,
                                          uint64_t*               m_,
                                          uint64_t*               n_,
                                          uint64_t*               coo_nnz_,
                                          rocsparseio_type*       coo_row_ind_type_,
                                          rocsparseio_type*       coo_col_ind_type_,
                                          rocsparseio_type*       coo_val_type_,
                                          rocsparseio_index_base* coo_base_,
                                          uint64_t*               ell_width_,
                                          rocsparseio_type*       ell_ind_type_,
                                          rocsparseio_type*       ell_val_type_,
                                          rocsparseio_index_base* ell_base_)
{

    ROCSPARSEIO_C_CHECK_ARG(!handle_, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(!m_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!n_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!coo_nnz_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!coo_row_ind_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!coo_col_ind_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!coo_val_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!coo_base_, rocsparseio::status_t::invalid_pointer);

    ROCSPARSEIO_C_CHECK_ARG(!ell_width_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!ell_ind_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!ell_val_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!ell_base_, rocsparseio::status_t::invalid_pointer);

    rocsparseio::type_t       coo_row_ind_type;
    rocsparseio::type_t       coo_col_ind_type;
    rocsparseio::type_t       coo_val_type;
    rocsparseio::index_base_t coo_index_base;

    rocsparseio::type_t       ell_ind_type;
    rocsparseio::type_t       ell_val_type;
    rocsparseio::index_base_t ell_index_base;

    ROCSPARSEIO_C_CHECK(rocsparseio::read_metadata_sparse_hyb(handle_,
                                                              m_,
                                                              n_,
                                                              coo_nnz_,
                                                              &coo_row_ind_type,
                                                              &coo_col_ind_type,
                                                              &coo_val_type,
                                                              &coo_index_base,
                                                              ell_width_,
                                                              &ell_ind_type,
                                                              &ell_val_type,
                                                              &ell_index_base));

    coo_row_ind_type_[0] = (rocsparseio_type)coo_row_ind_type;
    coo_col_ind_type_[0] = (rocsparseio_type)coo_col_ind_type;
    coo_val_type_[0]     = (rocsparseio_type)coo_val_type;
    coo_base_[0]         = (rocsparseio_index_base)coo_index_base;

    ell_ind_type_[0] = (rocsparseio_type)ell_ind_type;
    ell_val_type_[0] = (rocsparseio_type)ell_val_type;
    ell_base_[0]     = (rocsparseio_index_base)ell_index_base;

    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseiox_read_sparse_hyb(rocsparseio_handle handle_,
                                                           void*              coo_row_ind_,
                                                           void*              coo_col_ind_,
                                                           void*              coo_val_,
                                                           void*              ell_ind_,
                                                           void*              ell_val_)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle_, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK(rocsparseio::read_sparse_hyb(
        handle_, coo_row_ind_, coo_col_ind_, coo_val_, ell_ind_, ell_val_));
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseio_write_sparse_mcsx(rocsparseio_handle    handle_,
                                                            rocsparseio_direction dir_,
                                                            uint64_t              m_,
                                                            uint64_t              n_,
                                                            uint64_t              nnz_,

                                                            rocsparseio_type ptr_type_,
                                                            const void*      ptr_,

                                                            rocsparseio_type ind_type_,
                                                            const void*      ind_,

                                                            rocsparseio_type       val_type_,
                                                            const void*            val_,
                                                            rocsparseio_index_base base_,
                                                            const char*            name_,
                                                            ...)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle_, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::direction_t(dir_).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::type_t(ptr_type_).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::type_t(ind_type_).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::type_t(val_type_).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::index_base_t(base_).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG((m_ > 0 && !ptr_), rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG((nnz_ > 0 && !ind_), rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG((nnz_ > 0 && !val_), rocsparseio::status_t::invalid_pointer);

    {
        va_list args;
        va_start(args, name_);
        ROCSPARSEIO_C_CHECK(rocsparseio::write_sparse_mcsx(handle_,
                                                           dir_,
                                                           m_,
                                                           n_,
                                                           nnz_,
                                                           ptr_type_,
                                                           ptr_,
                                                           ind_type_,
                                                           ind_,
                                                           val_type_,
                                                           val_,
                                                           base_,
                                                           name_,
                                                           args));
        va_end(args);
    }

    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseiox_read_metadata_sparse_mcsx(rocsparseio_handle     handle_,
                                                                     rocsparseio_direction* dir_,
                                                                     uint64_t*              m_,
                                                                     uint64_t*              n_,
                                                                     uint64_t*              nnz_,

                                                                     rocsparseio_type* ptr_type_,
                                                                     rocsparseio_type* ind_type_,
                                                                     rocsparseio_type* val_type_,
                                                                     rocsparseio_index_base* base_)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle_, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(!dir_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!m_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!n_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!nnz_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!ptr_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!ind_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!val_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!base_, rocsparseio::status_t::invalid_pointer);
    rocsparseio::direction_t  dir;
    rocsparseio::type_t       ptr_type;
    rocsparseio::type_t       ind_type;
    rocsparseio::type_t       val_type;
    rocsparseio::index_base_t index_base;

    ROCSPARSEIO_C_CHECK(rocsparseio::read_metadata_sparse_mcsx(
        handle_, &dir, m_, n_, nnz_, &ptr_type, &ind_type, &val_type, &index_base));
    dir_[0]      = (rocsparseio_direction)dir;
    ptr_type_[0] = (rocsparseio_type)ptr_type;
    ind_type_[0] = (rocsparseio_type)ind_type;
    val_type_[0] = (rocsparseio_type)val_type;
    base_[0]     = (rocsparseio_index_base)index_base;
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status
    rocsparseiox_read_sparse_mcsx(rocsparseio_handle handle_, void* ptr_, void* ind_, void* val_)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle_, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK(rocsparseio::read_sparse_mcsx(handle_, ptr_, ind_, val_));
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseio_read_sparse_mcsx(rocsparseio_handle     handle_,
                                                           rocsparseio_direction* dir_,
                                                           uint64_t*              m_,
                                                           uint64_t*              n_,
                                                           uint64_t*              nnz_,

                                                           rocsparseio_type* ptr_type_,
                                                           void**            ptr_,

                                                           rocsparseio_type* ind_type_,
                                                           void**            ind_,

                                                           rocsparseio_type*       val_type_,
                                                           void**                  val_,
                                                           rocsparseio_index_base* base_)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle_, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(!dir_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!m_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!n_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!nnz_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!ptr_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!ind_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!val_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!ptr_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!ind_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!val_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(!base_, rocsparseio::status_t::invalid_pointer);

    ROCSPARSEIO_C_CHECK(rocsparseiox_read_metadata_sparse_mcsx(
        handle_, dir_, m_, n_, nnz_, ptr_type_, ind_type_, val_type_, base_));
    uint64_t                 ptr_size = 0;
    rocsparseio::type_t      ind_type = ind_type_[0];
    rocsparseio::type_t      val_type = val_type_[0];
    rocsparseio::type_t      ptr_type = ptr_type_[0];
    rocsparseio::direction_t dir      = dir_[0];
    switch(dir)
    {
    case rocsparseio::direction_t::row:
    {
        ptr_size = m_[0] + 1;
        break;
    }
    case rocsparseio::direction_t::column:
    {
        ptr_size = n_[0] + 1;
        break;
    }
    }

    ptr_[0] = malloc(ptr_type.size() * ptr_size);

    if(!ptr_[0])
    {
        return rocsparseio_status_invalid_memory;
    }

    ind_[0] = malloc(ind_type.size() * (nnz_[0]));

    if(!ind_[0])
    {
        return rocsparseio_status_invalid_memory;
    }

    val_[0] = malloc(val_type.size() * (nnz_[0]));

    if(!val_[0])
    {
        return rocsparseio_status_invalid_memory;
    }

    ROCSPARSEIO_C_CHECK(rocsparseiox_read_sparse_mcsx(handle_, ptr_[0], ind_[0], val_[0]));
    return rocsparseio_status_success;
}
