/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/conversion/rocsparse_ell2csr.h"
#include "rocsparse_control.hpp"
#include "rocsparse_ell2csr.hpp"
#include "rocsparse_utility.hpp"

#include "ell2csr_device.h"
#include "rocsparse_primitives.hpp"

rocsparse_status rocsparse::ell2csr_quickreturn(rocsparse_handle          handle,
                                                int64_t                   m,
                                                int64_t                   n,
                                                const rocsparse_mat_descr ell_descr,
                                                int64_t                   ell_width,
                                                const void*               ell_val,
                                                const void*               ell_col_ind,
                                                const rocsparse_mat_descr csr_descr,
                                                void*                     csr_val,
                                                const void*               csr_row_ptr,
                                                void*                     csr_col_ind)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(m == 0 || n == 0 || ell_width == 0)
    {
        return rocsparse_status_success;
    }
    return rocsparse_status_continue;
}

rocsparse_status rocsparse::ell2csr_checkarg(rocsparse_handle          handle, //0
                                             int64_t                   m, //1
                                             int64_t                   n, //2
                                             const rocsparse_mat_descr ell_descr, //3
                                             int64_t                   ell_width, //4
                                             const void*               ell_val, //5
                                             const void*               ell_col_ind, //6
                                             const rocsparse_mat_descr csr_descr, //7
                                             void*                     csr_val, //8
                                             const void*               csr_row_ptr, //9
                                             void*                     csr_col_ind) //10
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG_SIZE(2, n);
    ROCSPARSE_CHECKARG_POINTER(3, ell_descr);
    ROCSPARSE_CHECKARG(3,
                       ell_descr,
                       (ell_descr->type != rocsparse_matrix_type_general),
                       rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(3,
                       ell_descr,
                       (ell_descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);
    ROCSPARSE_CHECKARG_SIZE(4, ell_width);
    ROCSPARSE_CHECKARG_POINTER(7, csr_descr);
    ROCSPARSE_CHECKARG(7,
                       csr_descr,
                       (csr_descr->type != rocsparse_matrix_type_general),
                       rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(7,
                       csr_descr,
                       (csr_descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    const rocsparse_status status = rocsparse::ell2csr_quickreturn(handle,
                                                                   m,
                                                                   n,
                                                                   ell_descr,
                                                                   ell_width,
                                                                   ell_val,
                                                                   ell_col_ind,
                                                                   csr_descr,
                                                                   csr_val,
                                                                   csr_row_ptr,
                                                                   csr_col_ind);

    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    ROCSPARSE_CHECKARG_POINTER(5, ell_val);

    ROCSPARSE_CHECKARG_POINTER(6, ell_col_ind);

    ROCSPARSE_CHECKARG_POINTER(8, csr_val);

    ROCSPARSE_CHECKARG_ARRAY(9, m, csr_row_ptr);

    ROCSPARSE_CHECKARG_POINTER(10, csr_col_ind);
    return rocsparse_status_continue;
}

template <typename T, typename I, typename J>
rocsparse_status rocsparse::ell2csr_core(rocsparse_handle          handle,
                                         J                         m,
                                         J                         n,
                                         const rocsparse_mat_descr ell_descr,
                                         J                         ell_width,
                                         const T*                  ell_val,
                                         const J*                  ell_col_ind,
                                         const rocsparse_mat_descr csr_descr,
                                         T*                        csr_val,
                                         const I*                  csr_row_ptr,
                                         J*                        csr_col_ind)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Stream
    hipStream_t stream = handle->stream;
#define ELL2CSR_DIM 256
    dim3 ell2csr_blocks((m - 1) / ELL2CSR_DIM + 1);
    dim3 ell2csr_threads(ELL2CSR_DIM);

    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::ell2csr_fill<ELL2CSR_DIM>),
                                       ell2csr_blocks,
                                       ell2csr_threads,
                                       0,
                                       stream,
                                       m,
                                       n,
                                       ell_width,
                                       ell_col_ind,
                                       ell_val,
                                       ell_descr->base,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       csr_val,
                                       csr_descr->base);
#undef ELL2CSR_DIM
    return rocsparse_status_success;
}

#define INSTANTIATE(T, I, J)                                    \
    template rocsparse_status rocsparse::ell2csr_core<T, I, J>( \
        rocsparse_handle          handle,                       \
        J                         m,                            \
        J                         n,                            \
        const rocsparse_mat_descr ell_descr,                    \
        J                         ell_width,                    \
        const T*                  ell_val,                      \
        const J*                  ell_col_ind,                  \
        const rocsparse_mat_descr csr_descr,                    \
        T*                        csr_val,                      \
        const I*                  csr_row_ptr,                  \
        J*                        csr_col_ind)

INSTANTIATE(float, int32_t, int32_t);
INSTANTIATE(rocsparse_float_complex, int32_t, int32_t);
INSTANTIATE(double, int32_t, int32_t);
INSTANTIATE(rocsparse_double_complex, int32_t, int32_t);

INSTANTIATE(float, int64_t, int32_t);
INSTANTIATE(rocsparse_float_complex, int64_t, int32_t);
INSTANTIATE(double, int64_t, int32_t);
INSTANTIATE(rocsparse_double_complex, int64_t, int32_t);

INSTANTIATE(float, int32_t, int64_t);
INSTANTIATE(rocsparse_float_complex, int32_t, int64_t);
INSTANTIATE(double, int32_t, int64_t);
INSTANTIATE(rocsparse_double_complex, int32_t, int64_t);

INSTANTIATE(float, int64_t, int64_t);
INSTANTIATE(rocsparse_float_complex, int64_t, int64_t);
INSTANTIATE(double, int64_t, int64_t);
INSTANTIATE(rocsparse_double_complex, int64_t, int64_t);

#undef INSTANTIATE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

rocsparse_status rocsparse::ell2csr_nnz_quickreturn(rocsparse_handle          handle,
                                                    int64_t                   m,
                                                    int64_t                   n,
                                                    const rocsparse_mat_descr ell_descr,
                                                    int64_t                   ell_width,
                                                    const void*               ell_col_ind,
                                                    const rocsparse_mat_descr csr_descr,
                                                    void*                     csr_row_ptr,
                                                    void*                     csr_nnz)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(m == 0 || n == 0 || ell_width == 0)
    {
        return rocsparse_status_success;
    }
    return rocsparse_status_continue;
}

rocsparse_status rocsparse::ell2csr_nnz_checkarg(rocsparse_handle          handle,
                                                 int64_t                   m,
                                                 int64_t                   n,
                                                 const rocsparse_mat_descr ell_descr,
                                                 int64_t                   ell_width,
                                                 const void*               ell_col_ind,
                                                 const rocsparse_mat_descr csr_descr,
                                                 void*                     csr_row_ptr,
                                                 void*                     csr_nnz)

{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG_SIZE(2, n);
    ROCSPARSE_CHECKARG_POINTER(3, ell_descr);
    ROCSPARSE_CHECKARG(3,
                       ell_descr,
                       (ell_descr->type != rocsparse_matrix_type_general),
                       rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(3,
                       ell_descr,
                       (ell_descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);
    ROCSPARSE_CHECKARG_SIZE(4, ell_width);
    ROCSPARSE_CHECKARG_POINTER(6, csr_descr);
    ROCSPARSE_CHECKARG(6,
                       csr_descr,
                       (csr_descr->type != rocsparse_matrix_type_general),
                       rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(6,
                       csr_descr,
                       (csr_descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    ROCSPARSE_CHECKARG_ARRAY(5, ell_width * m, ell_col_ind);
    ROCSPARSE_CHECKARG_ARRAY(7, m, csr_row_ptr);

    const rocsparse_status status = rocsparse::ell2csr_nnz_quickreturn(
        handle, m, n, ell_descr, ell_width, ell_col_ind, csr_descr, csr_row_ptr, csr_nnz);

    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }
    ROCSPARSE_CHECKARG_POINTER(8, csr_nnz);
    return rocsparse_status_continue;
}

template <typename I, typename J>
rocsparse_status rocsparse::ell2csr_nnz_core(rocsparse_handle          handle,
                                             J                         m,
                                             J                         n,
                                             const rocsparse_mat_descr ell_descr,
                                             J                         ell_width,
                                             const J*                  ell_col_ind,
                                             const rocsparse_mat_descr csr_descr,
                                             I*                        csr_row_ptr,
                                             I*                        csr_nnz)
{
    ROCSPARSE_ROUTINE_TRACE;

    hipStream_t stream = handle->stream;
    // Count nnz per row
#define ELL2CSR_DIM 256
    dim3 ell2csr_blocks((m + 1) / ELL2CSR_DIM + 1);
    dim3 ell2csr_threads(ELL2CSR_DIM);
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::ell2csr_nnz_per_row<ELL2CSR_DIM>),
                                       ell2csr_blocks,
                                       ell2csr_threads,
                                       0,
                                       stream,
                                       m,
                                       n,
                                       ell_width,
                                       ell_col_ind,
                                       ell_descr->base,
                                       csr_row_ptr,
                                       csr_descr->base);
#undef ELL2CSR_DIM
    // Exclusive sum to obtain csr_row_ptr array and number of non-zero elements
    size_t temp_storage_bytes = 0;

    // Obtain rocprim buffer size
    RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::inclusive_scan_buffer_size<I, I>(
        handle, m + 1, &temp_storage_bytes)));

    // Get rocprim buffer
    bool  d_temp_alloc;
    void* d_temp_storage;
    // Device buffer should be sufficient for rocprim in most cases
    if(handle->buffer_size >= temp_storage_bytes)
    {
        d_temp_storage = handle->buffer;
        d_temp_alloc   = false;
    }
    else
    {
        RETURN_IF_HIP_ERROR(
            rocsparse_hipMallocAsync(&d_temp_storage, temp_storage_bytes, handle->stream));
        d_temp_alloc = true;
    }
    // Perform actual inclusive sum
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::inclusive_scan(
        handle, csr_row_ptr, csr_row_ptr, m + 1, temp_storage_bytes, d_temp_storage));

    // Extract and adjust nnz
    if(csr_descr->base == rocsparse_index_base_one)
    {
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                csr_nnz, csr_row_ptr + m, sizeof(I), hipMemcpyDeviceToDevice, stream));

            // Adjust nnz according to index base
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (ell2csr_index_base<1>), dim3(1), dim3(1), 0, stream, csr_nnz);
        }
        else
        {
            RETURN_IF_HIP_ERROR(
                hipMemcpyAsync(csr_nnz, csr_row_ptr + m, sizeof(I), hipMemcpyDeviceToHost, stream));
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

            // Adjust nnz according to index base
            *csr_nnz -= csr_descr->base;
        }
    }
    else
    {
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                csr_nnz, csr_row_ptr + m, sizeof(I), hipMemcpyDeviceToDevice, stream));
        }
        else
        {
            RETURN_IF_HIP_ERROR(
                hipMemcpyAsync(csr_nnz, csr_row_ptr + m, sizeof(I), hipMemcpyDeviceToHost, stream));
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));
        }
    }
    // Free rocprim buffer, if allocated
    if(d_temp_alloc == true)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(d_temp_storage, handle->stream));
    }
    return rocsparse_status_success;
}

#define INSTANTIATE(I, J)                                        \
    template rocsparse_status rocsparse::ell2csr_nnz_core<I, J>( \
        rocsparse_handle          handle,                        \
        J                         m,                             \
        J                         n,                             \
        const rocsparse_mat_descr ell_descr,                     \
        J                         ell_width,                     \
        const J*                  ell_col_ind,                   \
        const rocsparse_mat_descr csr_descr,                     \
        I*                        csr_row_ptr,                   \
        I*                        csr_nnz)

INSTANTIATE(int32_t, int32_t);
INSTANTIATE(int64_t, int32_t);
INSTANTIATE(int32_t, int64_t);
INSTANTIATE(int64_t, int64_t);

#undef INSTANTIATE

extern "C" rocsparse_status rocsparse_ell2csr_nnz(rocsparse_handle          handle,
                                                  rocsparse_int             m,
                                                  rocsparse_int             n,
                                                  const rocsparse_mat_descr ell_descr,
                                                  rocsparse_int             ell_width,
                                                  const rocsparse_int*      ell_col_ind,
                                                  const rocsparse_mat_descr csr_descr,
                                                  rocsparse_int*            csr_row_ptr,
                                                  rocsparse_int*            csr_nnz)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::ell2csr_nnz_impl(
        handle, m, n, ell_descr, ell_width, ell_col_ind, csr_descr, csr_row_ptr, csr_nnz));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_sell2csr(rocsparse_handle          handle,
                                               rocsparse_int             m,
                                               rocsparse_int             n,
                                               const rocsparse_mat_descr ell_descr,
                                               rocsparse_int             ell_width,
                                               const float*              ell_val,
                                               const rocsparse_int*      ell_col_ind,
                                               const rocsparse_mat_descr csr_descr,
                                               float*                    csr_val,
                                               const rocsparse_int*      csr_row_ptr,
                                               rocsparse_int*            csr_col_ind)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::ell2csr_impl(handle,
                                                      m,
                                                      n,
                                                      ell_descr,
                                                      ell_width,
                                                      ell_val,
                                                      ell_col_ind,
                                                      csr_descr,
                                                      csr_val,
                                                      csr_row_ptr,
                                                      csr_col_ind));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_dell2csr(rocsparse_handle          handle,
                                               rocsparse_int             m,
                                               rocsparse_int             n,
                                               const rocsparse_mat_descr ell_descr,
                                               rocsparse_int             ell_width,
                                               const double*             ell_val,
                                               const rocsparse_int*      ell_col_ind,
                                               const rocsparse_mat_descr csr_descr,
                                               double*                   csr_val,
                                               const rocsparse_int*      csr_row_ptr,
                                               rocsparse_int*            csr_col_ind)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::ell2csr_impl(handle,
                                                      m,
                                                      n,
                                                      ell_descr,
                                                      ell_width,
                                                      ell_val,
                                                      ell_col_ind,
                                                      csr_descr,
                                                      csr_val,
                                                      csr_row_ptr,
                                                      csr_col_ind));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_cell2csr(rocsparse_handle               handle,
                                               rocsparse_int                  m,
                                               rocsparse_int                  n,
                                               const rocsparse_mat_descr      ell_descr,
                                               rocsparse_int                  ell_width,
                                               const rocsparse_float_complex* ell_val,
                                               const rocsparse_int*           ell_col_ind,
                                               const rocsparse_mat_descr      csr_descr,
                                               rocsparse_float_complex*       csr_val,
                                               const rocsparse_int*           csr_row_ptr,
                                               rocsparse_int*                 csr_col_ind)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::ell2csr_impl(handle,
                                                      m,
                                                      n,
                                                      ell_descr,
                                                      ell_width,
                                                      ell_val,
                                                      ell_col_ind,
                                                      csr_descr,
                                                      csr_val,
                                                      csr_row_ptr,
                                                      csr_col_ind));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_zell2csr(rocsparse_handle                handle,
                                               rocsparse_int                   m,
                                               rocsparse_int                   n,
                                               const rocsparse_mat_descr       ell_descr,
                                               rocsparse_int                   ell_width,
                                               const rocsparse_double_complex* ell_val,
                                               const rocsparse_int*            ell_col_ind,
                                               const rocsparse_mat_descr       csr_descr,
                                               rocsparse_double_complex*       csr_val,
                                               const rocsparse_int*            csr_row_ptr,
                                               rocsparse_int*                  csr_col_ind)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::ell2csr_impl(handle,
                                                      m,
                                                      n,
                                                      ell_descr,
                                                      ell_width,
                                                      ell_val,
                                                      ell_col_ind,
                                                      csr_descr,
                                                      csr_val,
                                                      csr_row_ptr,
                                                      csr_col_ind));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
