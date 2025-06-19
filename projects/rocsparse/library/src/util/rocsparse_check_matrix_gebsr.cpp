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
#include "internal/util/rocsparse_check_matrix_gebsr.h"
#include "rocsparse_check_matrix_gebsr.hpp"
#include "rocsparse_enum_utils.hpp"
#include "rocsparse_utility.hpp"

#include "rocsparse_primitives.hpp"

#include "check_matrix_gebsr_device.h"

template <typename T, typename I, typename J>
rocsparse_status rocsparse::check_matrix_gebsr_core(rocsparse_handle       handle,
                                                    rocsparse_direction    dir,
                                                    J                      mb,
                                                    J                      nb,
                                                    I                      nnzb,
                                                    J                      row_block_dim,
                                                    J                      col_block_dim,
                                                    const T*               bsr_val,
                                                    const I*               bsr_row_ptr,
                                                    const J*               bsr_col_ind,
                                                    rocsparse_index_base   idx_base,
                                                    rocsparse_matrix_type  matrix_type,
                                                    rocsparse_fill_mode    uplo,
                                                    rocsparse_storage_mode storage,
                                                    rocsparse_data_status* data_status,
                                                    void*                  temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    *data_status = rocsparse_data_status_success;

    I start = 0;
    I end   = 0;
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(&end, &bsr_row_ptr[mb], sizeof(I), hipMemcpyDeviceToHost, handle->stream));
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(&start, &bsr_row_ptr[0], sizeof(I), hipMemcpyDeviceToHost, handle->stream));
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

    if(nnzb != (end - start))
    {
        // LCOV_EXCL_START
        rocsparse::log_debug(handle, "GEBSR row pointer array does not match nnzb.");
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }

    // Temporary buffer entry points
    char* ptr = reinterpret_cast<char*>(temp_buffer);

    rocsparse_data_status* d_data_status = reinterpret_cast<rocsparse_data_status*>(ptr);
    ptr += ((sizeof(rocsparse_data_status) - 1) / 256 + 1) * 256;

    RETURN_IF_HIP_ERROR(hipMemsetAsync(d_data_status, 0, sizeof(rocsparse_data_status)));
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::check_row_ptr_array<256>),
                                       dim3((mb - 1) / 256 + 1),
                                       dim3(256),
                                       0,
                                       handle->stream,
                                       mb,
                                       bsr_row_ptr,
                                       d_data_status);

    RETURN_IF_HIP_ERROR(hipMemcpyAsync(data_status,
                                       d_data_status,
                                       sizeof(rocsparse_data_status),
                                       hipMemcpyDeviceToHost,
                                       handle->stream));
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

    if(*data_status != rocsparse_data_status_success)
    {
        rocsparse::log_debug(handle, rocsparse::enum_utils::to_string(*data_status));

        return rocsparse_status_success;
    }

    I* tmp_offsets = nullptr;
    J* tmp_cols1   = nullptr;
    J* tmp_cols2   = nullptr;

    // If columns are unsorted, then sort them in temp buffer
    if(storage == rocsparse_storage_mode_unsorted)
    {
        // offsets buffer
        tmp_offsets = reinterpret_cast<I*>(ptr);
        ptr += ((sizeof(I) * mb) / 256 + 1) * 256;

        // columns 1 buffer
        tmp_cols1 = reinterpret_cast<J*>(ptr);
        ptr += ((sizeof(J) * nnzb - 1) / 256 + 1) * 256;

        // columns 2 buffer
        tmp_cols2 = reinterpret_cast<J*>(ptr);
        ptr += ((sizeof(J) * nnzb - 1) / 256 + 1) * 256;

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::shift_offsets_kernel<512>),
                                           dim3(mb / 512 + 1),
                                           dim3(512),
                                           0,
                                           handle->stream,
                                           mb + 1,
                                           bsr_row_ptr,
                                           tmp_offsets);

        // rocprim buffer
        void* tmp_rocprim = reinterpret_cast<void*>(ptr);

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::sort_csr_column_indices(
            handle, mb, nb, nnzb, tmp_offsets, bsr_col_ind, tmp_cols1, tmp_cols2, tmp_rocprim));
    }

    const J* bsr_col_ind_sorted
        = (storage == rocsparse_storage_mode_unsorted) ? tmp_cols2 : bsr_col_ind;

    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::check_matrix_gebsr_device<256>),
                                       dim3((mb - 1) / 256 + 1),
                                       dim3(256),
                                       0,
                                       handle->stream,
                                       dir,
                                       mb,
                                       nb,
                                       nnzb,
                                       row_block_dim,
                                       col_block_dim,
                                       bsr_val,
                                       bsr_row_ptr,
                                       bsr_col_ind,
                                       bsr_col_ind_sorted,
                                       idx_base,
                                       matrix_type,
                                       uplo,
                                       storage,
                                       d_data_status);

    RETURN_IF_HIP_ERROR(hipMemcpyAsync(data_status,
                                       d_data_status,
                                       sizeof(rocsparse_data_status),
                                       hipMemcpyDeviceToHost,
                                       handle->stream));
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

    if(*data_status != rocsparse_data_status_success)
    {
        rocsparse::log_debug(handle, rocsparse::enum_utils::to_string(*data_status));
    }

    return rocsparse_status_success;
}

namespace rocsparse
{
    template <typename T, typename I, typename J>
    static rocsparse_status check_matrix_gebsr_quickreturn(rocsparse_handle       handle,
                                                           rocsparse_direction    dir,
                                                           J                      mb,
                                                           J                      nb,
                                                           I                      nnzb,
                                                           J                      row_block_dim,
                                                           J                      col_block_dim,
                                                           const T*               bsr_val,
                                                           const I*               bsr_row_ptr,
                                                           const J*               bsr_col_ind,
                                                           rocsparse_index_base   idx_base,
                                                           rocsparse_matrix_type  matrix_type,
                                                           rocsparse_fill_mode    uplo,
                                                           rocsparse_storage_mode storage,
                                                           rocsparse_data_status* data_status,
                                                           void*                  temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        return rocsparse_status_continue;
    }
}

template <typename T, typename I, typename J>
rocsparse_status rocsparse::check_matrix_gebsr_checkarg(rocsparse_handle       handle, //0
                                                        rocsparse_direction    dir, //1
                                                        J                      mb, //2
                                                        J                      nb, //3
                                                        I                      nnzb, //4
                                                        J                      row_block_dim, //5
                                                        J                      col_block_dim, //6
                                                        const T*               bsr_val, //7
                                                        const I*               bsr_row_ptr, //8
                                                        const J*               bsr_col_ind, //9
                                                        rocsparse_index_base   idx_base, //10
                                                        rocsparse_matrix_type  matrix_type, //11
                                                        rocsparse_fill_mode    uplo, //12
                                                        rocsparse_storage_mode storage, //13
                                                        rocsparse_data_status* data_status, //14
                                                        void*                  temp_buffer) //15
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_ENUM(1, dir);
    ROCSPARSE_CHECKARG_SIZE(2, mb);
    ROCSPARSE_CHECKARG_SIZE(3, nb);
    ROCSPARSE_CHECKARG_SIZE(4, nnzb);

    ROCSPARSE_CHECKARG_SIZE(5, row_block_dim);
    ROCSPARSE_CHECKARG(5, row_block_dim, (row_block_dim == 0), rocsparse_status_invalid_size);

    ROCSPARSE_CHECKARG_SIZE(6, col_block_dim);
    ROCSPARSE_CHECKARG(6, col_block_dim, (col_block_dim == 0), rocsparse_status_invalid_size);

    ROCSPARSE_CHECKARG_ARRAY(7, nnzb, bsr_val);
    ROCSPARSE_CHECKARG_ARRAY(8, mb, bsr_row_ptr);
    ROCSPARSE_CHECKARG_ARRAY(9, nnzb, bsr_col_ind);
    ROCSPARSE_CHECKARG_ENUM(10, idx_base);
    ROCSPARSE_CHECKARG_ENUM(11, matrix_type);
    ROCSPARSE_CHECKARG_ENUM(12, uplo);
    ROCSPARSE_CHECKARG_ENUM(13, storage);
    ROCSPARSE_CHECKARG_POINTER(14, data_status);
    ROCSPARSE_CHECKARG_POINTER(15, temp_buffer);

    if(matrix_type != rocsparse_matrix_type_general)
    {
        if(row_block_dim != col_block_dim || mb != nb)
        {
            rocsparse::log_debug(handle,
                                 ("Matrix was specified to be "
                                  + std::string(rocsparse::enum_utils::to_string(matrix_type))
                                  + " but (row_block_dim != col_block_dim || mb != nb)"));
        }
    }

    ROCSPARSE_CHECKARG(11,
                       matrix_type,
                       ((matrix_type != rocsparse_matrix_type_general)
                        && (row_block_dim != col_block_dim || mb != nb)),
                       rocsparse_status_invalid_size);

    const rocsparse_status status = rocsparse::check_matrix_gebsr_quickreturn(handle,
                                                                              dir,
                                                                              mb,
                                                                              nb,
                                                                              nnzb,
                                                                              row_block_dim,
                                                                              col_block_dim,
                                                                              bsr_val,
                                                                              bsr_row_ptr,
                                                                              bsr_col_ind,
                                                                              idx_base,
                                                                              matrix_type,
                                                                              uplo,
                                                                              storage,
                                                                              data_status,
                                                                              temp_buffer);
    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    return rocsparse_status_continue;
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                                   \
    template rocsparse_status rocsparse::check_matrix_gebsr_core<TTYPE, ITYPE, JTYPE>(     \
        rocsparse_handle       handle,                                                     \
        rocsparse_direction    dir,                                                        \
        JTYPE                  mb,                                                         \
        JTYPE                  nb,                                                         \
        ITYPE                  nnzb,                                                       \
        JTYPE                  row_block_dim,                                              \
        JTYPE                  col_block_dim,                                              \
        const TTYPE*           bsr_val,                                                    \
        const ITYPE*           bsr_row_ptr,                                                \
        const JTYPE*           bsr_col_ind,                                                \
        rocsparse_index_base   idx_base,                                                   \
        rocsparse_matrix_type  matrix_type,                                                \
        rocsparse_fill_mode    uplo,                                                       \
        rocsparse_storage_mode storage,                                                    \
        rocsparse_data_status* data_status,                                                \
        void*                  temp_buffer);                                                                \
    template rocsparse_status rocsparse::check_matrix_gebsr_checkarg<TTYPE, ITYPE, JTYPE>( \
        rocsparse_handle       handle,                                                     \
        rocsparse_direction    dir,                                                        \
        JTYPE                  mb,                                                         \
        JTYPE                  nb,                                                         \
        ITYPE                  nnzb,                                                       \
        JTYPE                  row_block_dim,                                              \
        JTYPE                  col_block_dim,                                              \
        const TTYPE*           bsr_val,                                                    \
        const ITYPE*           bsr_row_ptr,                                                \
        const JTYPE*           bsr_col_ind,                                                \
        rocsparse_index_base   idx_base,                                                   \
        rocsparse_matrix_type  matrix_type,                                                \
        rocsparse_fill_mode    uplo,                                                       \
        rocsparse_storage_mode storage,                                                    \
        rocsparse_data_status* data_status,                                                \
        void*                  temp_buffer);

INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);
#undef INSTANTIATE

#define C_IMPL(NAME, T)                                                                          \
    extern "C" rocsparse_status NAME(rocsparse_handle       handle,                              \
                                     rocsparse_direction    dir,                                 \
                                     rocsparse_int          mb,                                  \
                                     rocsparse_int          nb,                                  \
                                     rocsparse_int          nnzb,                                \
                                     rocsparse_int          row_block_dim,                       \
                                     rocsparse_int          col_block_dim,                       \
                                     const T*               bsr_val,                             \
                                     const rocsparse_int*   bsr_row_ptr,                         \
                                     const rocsparse_int*   bsr_col_ind,                         \
                                     rocsparse_index_base   idx_base,                            \
                                     rocsparse_matrix_type  matrix_type,                         \
                                     rocsparse_fill_mode    uplo,                                \
                                     rocsparse_storage_mode storage,                             \
                                     rocsparse_data_status* data_status,                         \
                                     void*                  temp_buffer)                         \
    try                                                                                          \
    {                                                                                            \
        ROCSPARSE_ROUTINE_TRACE;                                                                 \
        RETURN_IF_ROCSPARSE_ERROR(                                                               \
            (rocsparse::check_matrix_gebsr_impl<T, rocsparse_int, rocsparse_int>(handle,         \
                                                                                 dir,            \
                                                                                 mb,             \
                                                                                 nb,             \
                                                                                 nnzb,           \
                                                                                 row_block_dim,  \
                                                                                 col_block_dim,  \
                                                                                 bsr_val,        \
                                                                                 bsr_row_ptr,    \
                                                                                 bsr_col_ind,    \
                                                                                 idx_base,       \
                                                                                 matrix_type,    \
                                                                                 uplo,           \
                                                                                 storage,        \
                                                                                 data_status,    \
                                                                                 temp_buffer))); \
        return rocsparse_status_success;                                                         \
    }                                                                                            \
    catch(...)                                                                                   \
    {                                                                                            \
        RETURN_ROCSPARSE_EXCEPTION();                                                            \
    }

C_IMPL(rocsparse_scheck_matrix_gebsr, float);
C_IMPL(rocsparse_dcheck_matrix_gebsr, double);
C_IMPL(rocsparse_ccheck_matrix_gebsr, rocsparse_float_complex);
C_IMPL(rocsparse_zcheck_matrix_gebsr, rocsparse_double_complex);
#undef C_IMPL
