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
#include "internal/util/rocsparse_check_matrix_csr.h"
#include "rocsparse_check_matrix_csr.hpp"
#include "rocsparse_enum_utils.hpp"
#include "rocsparse_utility.hpp"

#include "check_matrix_csr_device.h"

#include "rocsparse_primitives.hpp"

#define LAUNCH_CHECK_MATRIX_CSR(block_size, wf_size)                                              \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::check_matrix_csr_device<block_size, wf_size>), \
                                       dim3((wf_size * m - 1) / block_size + 1),                  \
                                       dim3(block_size),                                          \
                                       0,                                                         \
                                       handle->stream,                                            \
                                       m,                                                         \
                                       n,                                                         \
                                       nnz,                                                       \
                                       csr_val,                                                   \
                                       csr_row_ptr,                                               \
                                       csr_col_ind,                                               \
                                       csr_col_ind_sorted,                                        \
                                       idx_base,                                                  \
                                       matrix_type,                                               \
                                       uplo,                                                      \
                                       storage,                                                   \
                                       d_data_status);

template <typename T, typename I, typename J>
rocsparse_status rocsparse::check_matrix_csr_core(rocsparse_handle       handle,
                                                  J                      m,
                                                  J                      n,
                                                  I                      nnz,
                                                  const T*               csr_val,
                                                  const I*               csr_row_ptr,
                                                  const J*               csr_col_ind,
                                                  rocsparse_index_base   idx_base,
                                                  rocsparse_matrix_type  matrix_type,
                                                  rocsparse_fill_mode    uplo,
                                                  rocsparse_storage_mode storage,
                                                  rocsparse_data_status* data_status,
                                                  void*                  temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Check that nnz matches row pointer array
    I start = 0;
    I end   = 0;

    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(&end, &csr_row_ptr[m], sizeof(I), hipMemcpyDeviceToHost, handle->stream));
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(&start, &csr_row_ptr[0], sizeof(I), hipMemcpyDeviceToHost, handle->stream));
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

    if(nnz != (end - start))
    {
        // LCOV_EXCL_START
        rocsparse::log_debug(handle, "CSR row pointer array does not match nnz.");
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }

    // clear output status to success
    *data_status = rocsparse_data_status_success;

    // Temporary buffer entry points
    char* ptr = reinterpret_cast<char*>(temp_buffer);

    rocsparse_data_status* d_data_status = reinterpret_cast<rocsparse_data_status*>(ptr);
    ptr += ((sizeof(rocsparse_data_status) - 1) / 256 + 1) * 256;

    RETURN_IF_HIP_ERROR(hipMemsetAsync(d_data_status, 0, sizeof(rocsparse_data_status)));
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::check_row_ptr_array<256>),
                                       dim3((m - 1) / 256 + 1),
                                       dim3(256),
                                       0,
                                       handle->stream,
                                       m,
                                       csr_row_ptr,
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

    J avg_row_nnz = nnz / m;

    I* tmp_offsets = nullptr;
    J* tmp_cols1   = nullptr;
    J* tmp_cols2   = nullptr;

    // If columns are unsorted, then sort them in temp buffer
    if(storage == rocsparse_storage_mode_unsorted)
    {
        // offsets buffer
        tmp_offsets = reinterpret_cast<I*>(ptr);
        ptr += ((sizeof(I) * m) / 256 + 1) * 256;

        // columns 1 buffer
        tmp_cols1 = reinterpret_cast<J*>(ptr);
        ptr += ((sizeof(J) * nnz - 1) / 256 + 1) * 256;

        // columns 2 buffer
        tmp_cols2 = reinterpret_cast<J*>(ptr);
        ptr += ((sizeof(J) * nnz - 1) / 256 + 1) * 256;

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::shift_offsets_kernel<512>),
                                           dim3(m / 512 + 1),
                                           dim3(512),
                                           0,
                                           handle->stream,
                                           m + 1,
                                           csr_row_ptr,
                                           tmp_offsets);

        // rocprim buffer
        void* tmp_rocprim = reinterpret_cast<void*>(ptr);

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::sort_csr_column_indices(
            handle, m, n, nnz, tmp_offsets, csr_col_ind, tmp_cols1, tmp_cols2, tmp_rocprim));
    }

    const J* csr_col_ind_sorted
        = (storage == rocsparse_storage_mode_unsorted) ? tmp_cols2 : csr_col_ind;

    if(avg_row_nnz <= 4)
    {
        LAUNCH_CHECK_MATRIX_CSR(256, 4);
    }
    else if(avg_row_nnz <= 8)
    {
        LAUNCH_CHECK_MATRIX_CSR(256, 8);
    }
    else if(avg_row_nnz <= 16)
    {
        LAUNCH_CHECK_MATRIX_CSR(256, 16);
    }
    else if(avg_row_nnz <= 32)
    {
        LAUNCH_CHECK_MATRIX_CSR(256, 32);
    }
    else if(avg_row_nnz <= 64)
    {
        LAUNCH_CHECK_MATRIX_CSR(256, 64);
    }
    else if(avg_row_nnz <= 128)
    {
        LAUNCH_CHECK_MATRIX_CSR(256, 128);
    }
    else
    {
        LAUNCH_CHECK_MATRIX_CSR(256, 256);
    }

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
    static rocsparse_status check_matrix_csr_quickreturn(rocsparse_handle       handle,
                                                         J                      m,
                                                         J                      n,
                                                         I                      nnz,
                                                         const T*               csr_val,
                                                         const I*               csr_row_ptr,
                                                         const J*               csr_col_ind,
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
rocsparse_status rocsparse::check_matrix_csr_checkarg(rocsparse_handle       handle, //0
                                                      J                      m, //1
                                                      J                      n, //2
                                                      I                      nnz, //3
                                                      const T*               csr_val, //4
                                                      const I*               csr_row_ptr, //5
                                                      const J*               csr_col_ind, //6
                                                      rocsparse_index_base   idx_base, //7
                                                      rocsparse_matrix_type  matrix_type, //8
                                                      rocsparse_fill_mode    uplo, //9
                                                      rocsparse_storage_mode storage, //10
                                                      rocsparse_data_status* data_status, //11
                                                      void*                  temp_buffer) //12
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG_SIZE(2, n);
    ROCSPARSE_CHECKARG_SIZE(3, nnz);
    ROCSPARSE_CHECKARG_ARRAY(4, nnz, csr_val);
    ROCSPARSE_CHECKARG_ARRAY(5, m, csr_row_ptr);
    ROCSPARSE_CHECKARG_ARRAY(6, nnz, csr_col_ind);
    ROCSPARSE_CHECKARG_ENUM(7, idx_base);
    ROCSPARSE_CHECKARG_ENUM(8, matrix_type);
    ROCSPARSE_CHECKARG_ENUM(9, uplo);
    ROCSPARSE_CHECKARG_ENUM(10, storage);
    ROCSPARSE_CHECKARG_POINTER(11, data_status);
    ROCSPARSE_CHECKARG_POINTER(12, temp_buffer);

    if(matrix_type != rocsparse_matrix_type_general)
    {
        if(m != n)
        {
            rocsparse::log_debug(handle,
                                 ("Matrix was specified to be "
                                  + std::string(rocsparse::enum_utils::to_string(matrix_type))
                                  + " but m != n"));
        }
    }
    ROCSPARSE_CHECKARG(2,
                       n,
                       ((matrix_type != rocsparse_matrix_type_general) && (n != m)),
                       rocsparse_status_invalid_size);

    const rocsparse_status status = rocsparse::check_matrix_csr_quickreturn(handle,
                                                                            m,
                                                                            n,
                                                                            nnz,
                                                                            csr_val,
                                                                            csr_row_ptr,
                                                                            csr_col_ind,
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

template <typename T, typename I, typename J, typename... P>
rocsparse_status rocsparse_check_matrix_csr_template(P&&... p)
{
    ROCSPARSE_ROUTINE_TRACE;

    const rocsparse_status status = rocsparse::check_matrix_csr_quickreturn<T, I, J>(p...);
    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR((rocsparse::check_matrix_csr_core<T, I, J>(p...)));
    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    return rocsparse_status_success;
}

#define INSTANTIATE(I, J, T)                                                 \
    template rocsparse_status rocsparse::check_matrix_csr_core<T, I, J>(     \
        rocsparse_handle       handle,                                       \
        J                      m,                                            \
        J                      n,                                            \
        I                      nnz,                                          \
        const T*               csr_val,                                      \
        const I*               csr_row_ptr,                                  \
        const J*               csr_col_ind,                                  \
        rocsparse_index_base   idx_base,                                     \
        rocsparse_matrix_type  matrix_type,                                  \
        rocsparse_fill_mode    uplo,                                         \
        rocsparse_storage_mode storage,                                      \
        rocsparse_data_status* data_status,                                  \
        void*                  temp_buffer);                                                  \
    template rocsparse_status rocsparse::check_matrix_csr_checkarg<T, I, J>( \
        rocsparse_handle       handle,                                       \
        J                      m,                                            \
        J                      n,                                            \
        I                      nnz,                                          \
        const T*               csr_val,                                      \
        const I*               csr_row_ptr,                                  \
        const J*               csr_col_ind,                                  \
        rocsparse_index_base   idx_base,                                     \
        rocsparse_matrix_type  matrix_type,                                  \
        rocsparse_fill_mode    uplo,                                         \
        rocsparse_storage_mode storage,                                      \
        rocsparse_data_status* data_status,                                  \
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

#define C_IMPL(NAME, T)                                                                        \
    extern "C" rocsparse_status NAME(rocsparse_handle       handle,                            \
                                     rocsparse_int          m,                                 \
                                     rocsparse_int          n,                                 \
                                     rocsparse_int          nnz,                               \
                                     const T*               csr_val,                           \
                                     const rocsparse_int*   csr_row_ptr,                       \
                                     const rocsparse_int*   csr_col_ind,                       \
                                     rocsparse_index_base   idx_base,                          \
                                     rocsparse_matrix_type  matrix_type,                       \
                                     rocsparse_fill_mode    uplo,                              \
                                     rocsparse_storage_mode storage,                           \
                                     rocsparse_data_status* data_status,                       \
                                     void*                  temp_buffer)                       \
    try                                                                                        \
    {                                                                                          \
        ROCSPARSE_ROUTINE_TRACE;                                                               \
        RETURN_IF_ROCSPARSE_ERROR(                                                             \
            (rocsparse::check_matrix_csr_impl<T, rocsparse_int, rocsparse_int>(handle,         \
                                                                               m,              \
                                                                               n,              \
                                                                               nnz,            \
                                                                               csr_val,        \
                                                                               csr_row_ptr,    \
                                                                               csr_col_ind,    \
                                                                               idx_base,       \
                                                                               matrix_type,    \
                                                                               uplo,           \
                                                                               storage,        \
                                                                               data_status,    \
                                                                               temp_buffer))); \
        return rocsparse_status_success;                                                       \
    }                                                                                          \
    catch(...)                                                                                 \
    {                                                                                          \
        RETURN_ROCSPARSE_EXCEPTION();                                                          \
    }

C_IMPL(rocsparse_scheck_matrix_csr, float);
C_IMPL(rocsparse_dcheck_matrix_csr, double);
C_IMPL(rocsparse_ccheck_matrix_csr, rocsparse_float_complex);
C_IMPL(rocsparse_zcheck_matrix_csr, rocsparse_double_complex);
#undef C_IMPL
