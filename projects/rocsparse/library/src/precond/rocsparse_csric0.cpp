/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/precond/rocsparse_csric0.h"
#include "rocsparse_csric0.hpp"

#include "internal/level2/rocsparse_csrsv.h"

#include "../level2/rocsparse_csrsv.hpp"
#include "csric0_device.h"
#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

template <typename T>
rocsparse_status rocsparse::csric0_analysis_template(rocsparse_handle          handle, //0
                                                     rocsparse_int             m, //1
                                                     rocsparse_int             nnz, //2
                                                     const rocsparse_mat_descr descr, //3
                                                     const T*                  csr_val, //4
                                                     const rocsparse_int*      csr_row_ptr, //5
                                                     const rocsparse_int*      csr_col_ind, //6
                                                     rocsparse_mat_info        info, //7
                                                     rocsparse_analysis_policy analysis, //8
                                                     rocsparse_solve_policy    solve, //9
                                                     void*                     temp_buffer) //10
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xcsric0_analysis"),
                         m,
                         nnz,
                         (const void*&)descr,
                         (const void*&)csr_val,
                         (const void*&)csr_row_ptr,
                         (const void*&)csr_col_ind,
                         (const void*&)info,
                         solve,
                         analysis);

    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG_SIZE(2, nnz);

    ROCSPARSE_CHECKARG_POINTER(3, descr);
    ROCSPARSE_CHECKARG(
        3, descr, (descr->type != rocsparse_matrix_type_general), rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(3,
                       descr,
                       (descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    ROCSPARSE_CHECKARG_ARRAY(4, nnz, csr_val);
    ROCSPARSE_CHECKARG_ARRAY(5, m, csr_row_ptr);
    ROCSPARSE_CHECKARG_ARRAY(6, nnz, csr_col_ind);
    ROCSPARSE_CHECKARG_POINTER(7, info);
    ROCSPARSE_CHECKARG_ENUM(8, analysis);

    ROCSPARSE_CHECKARG_ENUM(9, solve);
    ROCSPARSE_CHECKARG(
        9, solve, (solve != rocsparse_solve_policy_auto), rocsparse_status_invalid_value);

    ROCSPARSE_CHECKARG_ARRAY(10, m, temp_buffer);

    // Quick return if possible
    if(m == 0)
    {
        return rocsparse_status_success;
    }

    if(analysis == rocsparse_analysis_policy_reuse)
    {
        auto trm = info->get_csric0_info(rocsparse_operation_none, rocsparse_fill_mode_lower);
        if(trm == nullptr)
        {
            trm = (trm != nullptr)
                      ? trm
                      : info->get_csrilu0_info(rocsparse_operation_none, rocsparse_fill_mode_lower);

            trm = (trm != nullptr)
                      ? trm
                      : info->get_csrsv_info(rocsparse_operation_none, rocsparse_fill_mode_lower);
            trm = (trm != nullptr)
                      ? trm
                      : info->get_csrsm_info(rocsparse_operation_none, rocsparse_fill_mode_lower);
            trm = (trm != nullptr) ? trm
                                   : info->get_csrsv_info(rocsparse_operation_transpose,
                                                          rocsparse_fill_mode_upper);
            trm = (trm != nullptr) ? trm
                                   : info->get_csrsm_info(rocsparse_operation_transpose,
                                                          rocsparse_fill_mode_upper);

            if(trm != nullptr)
            {
                info->set_csric0_info(rocsparse_operation_none, rocsparse_fill_mode_lower, trm);
            }
        }

        if(trm != nullptr)
        {
            return rocsparse_status_success;
        }
    }

    auto csric0_info = info->get_csric0_info();
    // Perform analysis
    RETURN_IF_ROCSPARSE_ERROR(csric0_info->recreate(rocsparse_operation_none,
                                                    rocsparse_fill_mode_lower,
                                                    handle,
                                                    rocsparse_operation_none,
                                                    m,
                                                    nnz,
                                                    descr,
                                                    csr_val,
                                                    csr_row_ptr,
                                                    csr_col_ind,
                                                    temp_buffer));

    csric0_info->create_singular_pivot_async(rocsparse::get_indextype<rocsparse_int>(),
                                             handle->stream);

    RETURN_IF_HIP_ERROR(hipMemcpyAsync(csric0_info->get_singular_pivot(),
                                       csric0_info->get_zero_pivot(),
                                       sizeof(rocsparse_int),
                                       hipMemcpyDeviceToDevice,
                                       handle->stream));
    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse::csric0_template(rocsparse_handle          handle, //0
                                            rocsparse_int             m, //1
                                            rocsparse_int             nnz, //2
                                            const rocsparse_mat_descr descr, //3
                                            T*                        csr_val, //4
                                            const rocsparse_int*      csr_row_ptr, //5
                                            const rocsparse_int*      csr_col_ind, //6
                                            rocsparse_mat_info        info, //7
                                            rocsparse_solve_policy    policy, //8
                                            void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    // Logging
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xcsric0"),
                         m,
                         nnz,
                         (const void*&)descr,
                         (const void*&)csr_val,
                         (const void*&)csr_row_ptr,
                         (const void*&)csr_col_ind,
                         (const void*&)info,
                         policy,
                         (const void*&)temp_buffer);

    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG_SIZE(2, nnz);

    ROCSPARSE_CHECKARG_POINTER(3, descr);
    ROCSPARSE_CHECKARG(
        3, descr, (descr->type != rocsparse_matrix_type_general), rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(3,
                       descr,
                       (descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    ROCSPARSE_CHECKARG_ARRAY(4, nnz, csr_val);
    ROCSPARSE_CHECKARG_ARRAY(5, m, csr_row_ptr);
    ROCSPARSE_CHECKARG_ARRAY(6, nnz, csr_col_ind);
    ROCSPARSE_CHECKARG_POINTER(7, info);
    ROCSPARSE_CHECKARG_ENUM(8, policy);
    ROCSPARSE_CHECKARG_ARRAY(9, m, temp_buffer);

    auto                   csric0_info = info->get_csric0_info();
    rocsparse::trm_info_t* trm_info
        = info->get_csric0_info(rocsparse_operation_none, rocsparse_fill_mode_lower);

    ROCSPARSE_CHECKARG(
        7, info, ((m > 0) && (trm_info == nullptr)), rocsparse_status_invalid_pointer);

    if(m == 0)
    {
        return rocsparse_status_success;
    }

    // Stream
    hipStream_t stream = handle->stream;

    // Buffer
    char* ptr = reinterpret_cast<char*>(temp_buffer);
    ptr += 256;

    // done array
    int* d_done_array = reinterpret_cast<int*>(ptr);

    // Initialize buffers
    RETURN_IF_HIP_ERROR(hipMemsetAsync(d_done_array, 0, sizeof(int) * m, stream));

    // Max nnz per row
    const rocsparse_int max_nnz = trm_info->get_max_nnz();

    // Determine gcnArch and ASIC revision
    const std::string gcn_arch_name = rocsparse::handle_get_arch_name(handle);

#define CSRIC0_DIM 256
    dim3 csric0_blocks((m * handle->wavefront_size - 1) / CSRIC0_DIM + 1);
    dim3 csric0_threads(CSRIC0_DIM);

    if(gcn_arch_name == rocpsarse_arch_names::gfx908 && handle->asic_rev < 2)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csric0_binsearch_kernel<CSRIC0_DIM, 64, true>),
            csric0_blocks,
            csric0_threads,
            0,
            stream,
            m,
            csr_row_ptr,
            csr_col_ind,
            csr_val,
            (const rocsparse_int*)trm_info->get_diag_ind(),
            d_done_array,
            (const rocsparse_int*)trm_info->get_row_map(),
            (rocsparse_int*)csric0_info->get_zero_pivot(),
            (rocsparse_int*)csric0_info->get_singular_pivot(),
            csric0_info->get_singular_tol(),
            descr->base);
    }
    else
    {
        if(handle->wavefront_size == 32)
        {
            if(max_nnz <= 32)
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::csric0_hash_kernel<CSRIC0_DIM, 32, 1>),
                    csric0_blocks,
                    csric0_threads,
                    0,
                    stream,
                    m,
                    csr_row_ptr,
                    csr_col_ind,
                    csr_val,
                    (const rocsparse_int*)trm_info->get_diag_ind(),
                    d_done_array,
                    (const rocsparse_int*)trm_info->get_row_map(),
                    (rocsparse_int*)csric0_info->get_zero_pivot(),
                    (rocsparse_int*)csric0_info->get_singular_pivot(),
                    csric0_info->get_singular_tol(),
                    descr->base);
            }
            else if(max_nnz <= 64)
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::csric0_hash_kernel<CSRIC0_DIM, 32, 2>),
                    csric0_blocks,
                    csric0_threads,
                    0,
                    stream,
                    m,
                    csr_row_ptr,
                    csr_col_ind,
                    csr_val,
                    (const rocsparse_int*)trm_info->get_diag_ind(),
                    d_done_array,
                    (const rocsparse_int*)trm_info->get_row_map(),
                    (rocsparse_int*)csric0_info->get_zero_pivot(),
                    (rocsparse_int*)csric0_info->get_singular_pivot(),
                    csric0_info->get_singular_tol(),
                    descr->base);
            }
            else if(max_nnz <= 128)
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::csric0_hash_kernel<CSRIC0_DIM, 32, 4>),
                    csric0_blocks,
                    csric0_threads,
                    0,
                    stream,
                    m,
                    csr_row_ptr,
                    csr_col_ind,
                    csr_val,
                    (const rocsparse_int*)trm_info->get_diag_ind(),
                    d_done_array,
                    (const rocsparse_int*)trm_info->get_row_map(),
                    (rocsparse_int*)csric0_info->get_zero_pivot(),
                    (rocsparse_int*)csric0_info->get_singular_pivot(),
                    csric0_info->get_singular_tol(),
                    descr->base);
            }
            else if(max_nnz <= 256)
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::csric0_hash_kernel<CSRIC0_DIM, 32, 8>),
                    csric0_blocks,
                    csric0_threads,
                    0,
                    stream,
                    m,
                    csr_row_ptr,
                    csr_col_ind,
                    csr_val,
                    (const rocsparse_int*)trm_info->get_diag_ind(),
                    d_done_array,
                    (const rocsparse_int*)trm_info->get_row_map(),
                    (rocsparse_int*)csric0_info->get_zero_pivot(),
                    (rocsparse_int*)csric0_info->get_singular_pivot(),
                    csric0_info->get_singular_tol(),
                    descr->base);
            }
            else if(max_nnz <= 512)
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::csric0_hash_kernel<CSRIC0_DIM, 32, 16>),
                    csric0_blocks,
                    csric0_threads,
                    0,
                    stream,
                    m,
                    csr_row_ptr,
                    csr_col_ind,
                    csr_val,
                    (const rocsparse_int*)trm_info->get_diag_ind(),
                    d_done_array,
                    (const rocsparse_int*)trm_info->get_row_map(),
                    (rocsparse_int*)csric0_info->get_zero_pivot(),
                    (rocsparse_int*)csric0_info->get_singular_pivot(),
                    csric0_info->get_singular_tol(),
                    descr->base);
            }
            else
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::csric0_binsearch_kernel<CSRIC0_DIM, 32, false>),
                    csric0_blocks,
                    csric0_threads,
                    0,
                    stream,
                    m,
                    csr_row_ptr,
                    csr_col_ind,
                    csr_val,
                    (const rocsparse_int*)trm_info->get_diag_ind(),
                    d_done_array,
                    (const rocsparse_int*)trm_info->get_row_map(),
                    (rocsparse_int*)csric0_info->get_zero_pivot(),
                    (rocsparse_int*)csric0_info->get_singular_pivot(),
                    csric0_info->get_singular_tol(),
                    descr->base);
            }
        }
        else if(handle->wavefront_size == 64)
        {
            if(max_nnz <= 64)
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::csric0_hash_kernel<CSRIC0_DIM, 64, 1>),
                    csric0_blocks,
                    csric0_threads,
                    0,
                    stream,
                    m,
                    csr_row_ptr,
                    csr_col_ind,
                    csr_val,
                    (const rocsparse_int*)trm_info->get_diag_ind(),
                    d_done_array,
                    (const rocsparse_int*)trm_info->get_row_map(),
                    (rocsparse_int*)csric0_info->get_zero_pivot(),
                    (rocsparse_int*)csric0_info->get_singular_pivot(),
                    csric0_info->get_singular_tol(),
                    descr->base);
            }
            else if(max_nnz <= 128)
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::csric0_hash_kernel<CSRIC0_DIM, 64, 2>),
                    csric0_blocks,
                    csric0_threads,
                    0,
                    stream,
                    m,
                    csr_row_ptr,
                    csr_col_ind,
                    csr_val,
                    (const rocsparse_int*)trm_info->get_diag_ind(),
                    d_done_array,
                    (const rocsparse_int*)trm_info->get_row_map(),
                    (rocsparse_int*)csric0_info->get_zero_pivot(),
                    (rocsparse_int*)csric0_info->get_singular_pivot(),
                    csric0_info->get_singular_tol(),
                    descr->base);
            }
            else if(max_nnz <= 256)
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::csric0_hash_kernel<CSRIC0_DIM, 64, 4>),
                    csric0_blocks,
                    csric0_threads,
                    0,
                    stream,
                    m,
                    csr_row_ptr,
                    csr_col_ind,
                    csr_val,
                    (const rocsparse_int*)trm_info->get_diag_ind(),
                    d_done_array,
                    (const rocsparse_int*)trm_info->get_row_map(),
                    (rocsparse_int*)csric0_info->get_zero_pivot(),
                    (rocsparse_int*)csric0_info->get_singular_pivot(),
                    csric0_info->get_singular_tol(),
                    descr->base);
            }
            else if(max_nnz <= 512)
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::csric0_hash_kernel<CSRIC0_DIM, 64, 8>),
                    csric0_blocks,
                    csric0_threads,
                    0,
                    stream,
                    m,
                    csr_row_ptr,
                    csr_col_ind,
                    csr_val,
                    (const rocsparse_int*)trm_info->get_diag_ind(),
                    d_done_array,
                    (const rocsparse_int*)trm_info->get_row_map(),
                    (rocsparse_int*)csric0_info->get_zero_pivot(),
                    (rocsparse_int*)csric0_info->get_singular_pivot(),
                    csric0_info->get_singular_tol(),
                    descr->base);
            }
            else if(max_nnz <= 1024)
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::csric0_hash_kernel<CSRIC0_DIM, 64, 16>),
                    csric0_blocks,
                    csric0_threads,
                    0,
                    stream,
                    m,
                    csr_row_ptr,
                    csr_col_ind,
                    csr_val,
                    (const rocsparse_int*)trm_info->get_diag_ind(),
                    d_done_array,
                    (const rocsparse_int*)trm_info->get_row_map(),
                    (rocsparse_int*)csric0_info->get_zero_pivot(),
                    (rocsparse_int*)csric0_info->get_singular_pivot(),
                    csric0_info->get_singular_tol(),
                    descr->base);
            }
            else
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::csric0_binsearch_kernel<CSRIC0_DIM, 64, false>),
                    csric0_blocks,
                    csric0_threads,
                    0,
                    stream,
                    m,
                    csr_row_ptr,
                    csr_col_ind,
                    csr_val,
                    (const rocsparse_int*)trm_info->get_diag_ind(),
                    d_done_array,
                    (const rocsparse_int*)trm_info->get_row_map(),
                    (rocsparse_int*)csric0_info->get_zero_pivot(),
                    (rocsparse_int*)csric0_info->get_singular_pivot(),
                    csric0_info->get_singular_tol(),
                    descr->base);
            }
        }
        else
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_arch_mismatch);
        }
    }
#undef CSRIC0_DIM

    return rocsparse_status_success;
}

namespace rocsparse
{
    template <typename T>
    rocsparse_status csric0_buffer_size_template(rocsparse_handle          handle,
                                                 rocsparse_int             m,
                                                 rocsparse_int             nnz,
                                                 const rocsparse_mat_descr descr,
                                                 const T*                  csr_val,
                                                 const rocsparse_int*      csr_row_ptr,
                                                 const rocsparse_int*      csr_col_ind,
                                                 rocsparse_mat_info        info,
                                                 size_t*                   buffer_size)
    {
        ROCSPARSE_ROUTINE_TRACE;

        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::csrsv_buffer_size_template<rocsparse_int, rocsparse_int, T>(
                handle,
                rocsparse_operation_none,
                m,
                nnz,
                descr,
                csr_val,
                csr_row_ptr,
                csr_col_ind,
                info,
                buffer_size)));
        return rocsparse_status_success;
    }

    template <typename T>
    rocsparse_status csric0_buffer_size_impl(rocsparse_handle          handle,
                                             rocsparse_int             m,
                                             rocsparse_int             nnz,
                                             const rocsparse_mat_descr descr,
                                             const T*                  csr_val,
                                             const rocsparse_int*      csr_row_ptr,
                                             const rocsparse_int*      csr_col_ind,
                                             rocsparse_mat_info        info,
                                             size_t*                   buffer_size)
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);

        rocsparse::log_trace(handle,
                             rocsparse::replaceX<T>("rocsparse_Xcsric0_buffer_size"),
                             m,
                             nnz,
                             (const void*&)descr,
                             (const void*&)csr_val,
                             (const void*&)csr_row_ptr,
                             (const void*&)csr_col_ind,
                             (const void*&)info,
                             (const void*&)buffer_size);

        ROCSPARSE_CHECKARG_SIZE(1, m);
        ROCSPARSE_CHECKARG_SIZE(2, nnz);
        ROCSPARSE_CHECKARG_POINTER(3, descr);
        ROCSPARSE_CHECKARG(3,
                           descr,
                           (descr->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(3,
                           descr,
                           (descr->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);
        ROCSPARSE_CHECKARG_ARRAY(4, nnz, csr_val);
        ROCSPARSE_CHECKARG_ARRAY(5, m, csr_row_ptr);
        ROCSPARSE_CHECKARG_ARRAY(6, nnz, csr_col_ind);
        ROCSPARSE_CHECKARG_POINTER(7, info);
        ROCSPARSE_CHECKARG_POINTER(8, buffer_size);

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csric0_buffer_size_template(
            handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size));

        return rocsparse_status_success;
    }
}

#define CIMPL(NAME, T)                                                                     \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,                     \
                                     rocsparse_int             m,                          \
                                     rocsparse_int             nnz,                        \
                                     const rocsparse_mat_descr descr,                      \
                                     const T*                  csr_val,                    \
                                     const rocsparse_int*      csr_row_ptr,                \
                                     const rocsparse_int*      csr_col_ind,                \
                                     rocsparse_mat_info        info,                       \
                                     size_t*                   buffer_size)                \
    try                                                                                    \
    {                                                                                      \
        ROCSPARSE_ROUTINE_TRACE;                                                           \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csric0_buffer_size_impl(                      \
            handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size)); \
        return rocsparse_status_success;                                                   \
    }                                                                                      \
    catch(...)                                                                             \
    {                                                                                      \
        RETURN_ROCSPARSE_EXCEPTION();                                                      \
    }

CIMPL(rocsparse_scsric0_buffer_size, float);
CIMPL(rocsparse_dcsric0_buffer_size, double);
CIMPL(rocsparse_ccsric0_buffer_size, rocsparse_float_complex);
CIMPL(rocsparse_zcsric0_buffer_size, rocsparse_double_complex);
#undef CIMPL

extern "C" rocsparse_status rocsparse_scsric0_analysis(rocsparse_handle          handle,
                                                       rocsparse_int             m,
                                                       rocsparse_int             nnz,
                                                       const rocsparse_mat_descr descr,
                                                       const float*              csr_val,
                                                       const rocsparse_int*      csr_row_ptr,
                                                       const rocsparse_int*      csr_col_ind,
                                                       rocsparse_mat_info        info,
                                                       rocsparse_analysis_policy analysis,
                                                       rocsparse_solve_policy    solve,
                                                       void*                     temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csric0_analysis_template(handle,
                                                                  m,
                                                                  nnz,
                                                                  descr,
                                                                  csr_val,
                                                                  csr_row_ptr,
                                                                  csr_col_ind,
                                                                  info,
                                                                  analysis,
                                                                  solve,
                                                                  temp_buffer));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_dcsric0_analysis(rocsparse_handle          handle,
                                                       rocsparse_int             m,
                                                       rocsparse_int             nnz,
                                                       const rocsparse_mat_descr descr,
                                                       const double*             csr_val,
                                                       const rocsparse_int*      csr_row_ptr,
                                                       const rocsparse_int*      csr_col_ind,
                                                       rocsparse_mat_info        info,
                                                       rocsparse_analysis_policy analysis,
                                                       rocsparse_solve_policy    solve,
                                                       void*                     temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csric0_analysis_template(handle,
                                                                  m,
                                                                  nnz,
                                                                  descr,
                                                                  csr_val,
                                                                  csr_row_ptr,
                                                                  csr_col_ind,
                                                                  info,
                                                                  analysis,
                                                                  solve,
                                                                  temp_buffer));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_ccsric0_analysis(rocsparse_handle               handle,
                                                       rocsparse_int                  m,
                                                       rocsparse_int                  nnz,
                                                       const rocsparse_mat_descr      descr,
                                                       const rocsparse_float_complex* csr_val,
                                                       const rocsparse_int*           csr_row_ptr,
                                                       const rocsparse_int*           csr_col_ind,
                                                       rocsparse_mat_info             info,
                                                       rocsparse_analysis_policy      analysis,
                                                       rocsparse_solve_policy         solve,
                                                       void*                          temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csric0_analysis_template(handle,
                                                                  m,
                                                                  nnz,
                                                                  descr,
                                                                  csr_val,
                                                                  csr_row_ptr,
                                                                  csr_col_ind,
                                                                  info,
                                                                  analysis,
                                                                  solve,
                                                                  temp_buffer));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_zcsric0_analysis(rocsparse_handle                handle,
                                                       rocsparse_int                   m,
                                                       rocsparse_int                   nnz,
                                                       const rocsparse_mat_descr       descr,
                                                       const rocsparse_double_complex* csr_val,
                                                       const rocsparse_int*            csr_row_ptr,
                                                       const rocsparse_int*            csr_col_ind,
                                                       rocsparse_mat_info              info,
                                                       rocsparse_analysis_policy       analysis,
                                                       rocsparse_solve_policy          solve,
                                                       void*                           temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csric0_analysis_template(handle,
                                                                  m,
                                                                  nnz,
                                                                  descr,
                                                                  csr_val,
                                                                  csr_row_ptr,
                                                                  csr_col_ind,
                                                                  info,
                                                                  analysis,
                                                                  solve,
                                                                  temp_buffer));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_csric0_clear(rocsparse_handle handle, rocsparse_mat_info info)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    rocsparse::log_trace(handle, "rocsparse_csric0_clear", (const void*&)info);

    ROCSPARSE_CHECKARG_POINTER(1, info);

    info->clear_csric0_info();

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_scsric0(rocsparse_handle          handle,
                                              rocsparse_int             m,
                                              rocsparse_int             nnz,
                                              const rocsparse_mat_descr descr,
                                              float*                    csr_val,
                                              const rocsparse_int*      csr_row_ptr,
                                              const rocsparse_int*      csr_col_ind,
                                              rocsparse_mat_info        info,
                                              rocsparse_solve_policy    policy,
                                              void*                     temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csric0_template(
        handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_dcsric0(rocsparse_handle          handle,
                                              rocsparse_int             m,
                                              rocsparse_int             nnz,
                                              const rocsparse_mat_descr descr,
                                              double*                   csr_val,
                                              const rocsparse_int*      csr_row_ptr,
                                              const rocsparse_int*      csr_col_ind,
                                              rocsparse_mat_info        info,
                                              rocsparse_solve_policy    policy,
                                              void*                     temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csric0_template(
        handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_ccsric0(rocsparse_handle          handle,
                                              rocsparse_int             m,
                                              rocsparse_int             nnz,
                                              const rocsparse_mat_descr descr,
                                              rocsparse_float_complex*  csr_val,
                                              const rocsparse_int*      csr_row_ptr,
                                              const rocsparse_int*      csr_col_ind,
                                              rocsparse_mat_info        info,
                                              rocsparse_solve_policy    policy,
                                              void*                     temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csric0_template(
        handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_zcsric0(rocsparse_handle          handle,
                                              rocsparse_int             m,
                                              rocsparse_int             nnz,
                                              const rocsparse_mat_descr descr,
                                              rocsparse_double_complex* csr_val,
                                              const rocsparse_int*      csr_row_ptr,
                                              const rocsparse_int*      csr_col_ind,
                                              rocsparse_mat_info        info,
                                              rocsparse_solve_policy    policy,
                                              void*                     temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csric0_template(
        handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_csric0_zero_pivot(rocsparse_handle   handle,
                                                        rocsparse_mat_info info,
                                                        rocsparse_int*     position)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    // Logging
    rocsparse::log_trace(
        handle, "rocsparse_csric0_zero_pivot", (const void*&)info, (const void*&)position);

    ROCSPARSE_CHECKARG_POINTER(1, info);
    ROCSPARSE_CHECKARG_POINTER(2, position);

    auto csric0_info = info->get_csric0_info();
    {
        auto status = csric0_info->copy_zero_pivot_async(handle->pointer_mode,
                                                         rocsparse::get_indextype<rocsparse_int>(),
                                                         position,
                                                         handle->stream);
        if(status == rocsparse_status_zero_pivot)
        {
            return status;
        }
        RETURN_IF_ROCSPARSE_ERROR(status);
    }

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_csric0_singular_pivot(rocsparse_handle   handle,
                                                            rocsparse_mat_info info,
                                                            rocsparse_int*     position)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    // Logging
    rocsparse::log_trace(
        handle, "rocsparse_csric0_singular_pivot", (const void*&)info, (const void*&)position);

    ROCSPARSE_CHECKARG_POINTER(1, info);
    ROCSPARSE_CHECKARG_POINTER(2, position);

    // Stream
    hipStream_t stream = handle->stream;

    auto    csric0_info = info->get_csric0_info();
    int64_t zero_pivot_value;
    auto    status = csric0_info->copy_zero_pivot_async(rocsparse_pointer_mode_host,
                                                     rocsparse::get_indextype<int64_t>(),
                                                     &zero_pivot_value,
                                                     handle->stream);
    if(status != rocsparse_status_zero_pivot)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
    }
    int64_t singular_pivot_value;
    status = csric0_info->copy_singular_pivot_async(rocsparse_pointer_mode_host,
                                                    rocsparse::get_indextype<int64_t>(),
                                                    &singular_pivot_value,
                                                    handle->stream);

    if(status != rocsparse_status_zero_pivot)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
    }
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

    int64_t singular_pivot;
    if((singular_pivot_value == -1) || (zero_pivot_value == -1))
    {
        singular_pivot = ((singular_pivot_value == -1) ? zero_pivot_value : singular_pivot_value);
    }
    else
    {
        singular_pivot = std::min(zero_pivot_value, singular_pivot_value);
    }

    // Differentiate between pointer modes
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        // rocsparse_pointer_mode_device
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            position, &singular_pivot, sizeof(rocsparse_int), hipMemcpyHostToDevice, stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));
    }
    else
    {
        // rocsparse_pointer_mode_host
        *position = singular_pivot;
    }

    return (rocsparse_status_success);
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status
    rocsparse_csric0_set_tolerance(rocsparse_handle handle, rocsparse_mat_info info, double tol)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    if(info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(tol < 0)
    {
        // LCOV_EXCL_START
        return rocsparse_status_invalid_value;
        // LCOV_EXCL_STOP
    }

    // Logging
    rocsparse::log_trace(handle, "rocsparse_csric0_set_tolerance", (const void*&)info, tol);
    auto csric0_info = info->get_csric0_info();
    csric0_info->set_singular_tol(tol);

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status
    rocsparse_csric0_get_tolerance(rocsparse_handle handle, rocsparse_mat_info info, double* tol)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    if(info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(tol == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    rocsparse::log_trace(handle, "rocsparse_csric0_get_tolerance", (const void*&)info, tol);
    auto csric0_info = info->get_csric0_info();
    *tol             = csric0_info->get_singular_tol();

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
