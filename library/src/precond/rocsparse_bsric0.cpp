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

#include "internal/precond/rocsparse_bsric0.h"
#include "internal/level2/rocsparse_csrsv.h"

#include "rocsparse_bsric0.hpp"

#include "../level2/rocsparse_csrsv.hpp"
#include "bsric0_device.h"
#include "control.h"
#include "utility.h"

#define LAUNCH_BSRIC_2_8_UNROLLED(T, block_size, maz_nnzb, bsr_block_dim)             \
    THROW_IF_HIPLAUNCHKERNELGGL_ERROR(                                                \
        (rocsparse::bsric0_2_8_unrolled_kernel<block_size, maz_nnzb, bsr_block_dim>), \
        dim3(mb),                                                                     \
        dim3(bsr_block_dim, bsr_block_dim),                                           \
        0,                                                                            \
        handle->stream,                                                               \
        dir,                                                                          \
        mb,                                                                           \
        block_dim,                                                                    \
        bsr_row_ptr,                                                                  \
        bsr_col_ind,                                                                  \
        bsr_val,                                                                      \
        (rocsparse_int*)info->bsric0_info->trm_diag_ind,                              \
        done_array,                                                                   \
        (rocsparse_int*)info->bsric0_info->row_map,                                   \
        (rocsparse_int*)info->zero_pivot,                                             \
        base);

#define LAUNCH_BSRIC_2_8(T, block_size, maz_nnzb, bsr_block_dim)             \
    THROW_IF_HIPLAUNCHKERNELGGL_ERROR(                                       \
        (rocsparse::bsric0_2_8_kernel<block_size, maz_nnzb, bsr_block_dim>), \
        dim3(mb),                                                            \
        dim3(8, 8),                                                          \
        0,                                                                   \
        handle->stream,                                                      \
        dir,                                                                 \
        mb,                                                                  \
        block_dim,                                                           \
        bsr_row_ptr,                                                         \
        bsr_col_ind,                                                         \
        bsr_val,                                                             \
        (rocsparse_int*)info->bsric0_info->trm_diag_ind,                     \
        done_array,                                                          \
        (rocsparse_int*)info->bsric0_info->row_map,                          \
        (rocsparse_int*)info->zero_pivot,                                    \
        base);

#define LAUNCH_BSRIC_9_16(T, block_size, maz_nnzb, bsr_block_dim)             \
    THROW_IF_HIPLAUNCHKERNELGGL_ERROR(                                        \
        (rocsparse::bsric0_9_16_kernel<block_size, maz_nnzb, bsr_block_dim>), \
        dim3(mb),                                                             \
        dim3(4, 16),                                                          \
        0,                                                                    \
        handle->stream,                                                       \
        dir,                                                                  \
        mb,                                                                   \
        block_dim,                                                            \
        bsr_row_ptr,                                                          \
        bsr_col_ind,                                                          \
        bsr_val,                                                              \
        (rocsparse_int*)info->bsric0_info->trm_diag_ind,                      \
        done_array,                                                           \
        (rocsparse_int*)info->bsric0_info->row_map,                           \
        (rocsparse_int*)info->zero_pivot,                                     \
        base);

#define LAUNCH_BSRIC_17_32(T, block_size, maz_nnzb, bsr_block_dim)             \
    THROW_IF_HIPLAUNCHKERNELGGL_ERROR(                                         \
        (rocsparse::bsric0_17_32_kernel<block_size, maz_nnzb, bsr_block_dim>), \
        dim3(mb),                                                              \
        dim3(2, 32),                                                           \
        0,                                                                     \
        handle->stream,                                                        \
        dir,                                                                   \
        mb,                                                                    \
        block_dim,                                                             \
        bsr_row_ptr,                                                           \
        bsr_col_ind,                                                           \
        bsr_val,                                                               \
        (rocsparse_int*)info->bsric0_info->trm_diag_ind,                       \
        done_array,                                                            \
        (rocsparse_int*)info->bsric0_info->row_map,                            \
        (rocsparse_int*)info->zero_pivot,                                      \
        base);

#define LAUNCH_BSRIC_33_inf(T, block_size, wf_size, sleep)                \
    THROW_IF_HIPLAUNCHKERNELGGL_ERROR(                                    \
        (rocsparse::bsric0_binsearch_kernel<block_size, wf_size, sleep>), \
        dim3(mb),                                                         \
        dim3(block_size),                                                 \
        0,                                                                \
        handle->stream,                                                   \
        dir,                                                              \
        mb,                                                               \
        block_dim,                                                        \
        bsr_row_ptr,                                                      \
        bsr_col_ind,                                                      \
        bsr_val,                                                          \
        (rocsparse_int*)info->bsric0_info->trm_diag_ind,                  \
        done_array,                                                       \
        (rocsparse_int*)info->bsric0_info->row_map,                       \
        (rocsparse_int*)info->zero_pivot,                                 \
        base);

template <typename T>
rocsparse_status rocsparse::bsric0_analysis_template(rocsparse_handle          handle, //0
                                                     rocsparse_direction       dir, //1
                                                     rocsparse_int             mb, //2
                                                     rocsparse_int             nnzb, //3
                                                     const rocsparse_mat_descr descr, //4
                                                     const T*                  bsr_val, //5
                                                     const rocsparse_int*      bsr_row_ptr, //6
                                                     const rocsparse_int*      bsr_col_ind, //7
                                                     rocsparse_int             block_dim, //8
                                                     rocsparse_mat_info        info, //9
                                                     rocsparse_analysis_policy analysis, //10
                                                     rocsparse_solve_policy    solve, //11
                                                     void*                     temp_buffer) //12
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xbsric0_analysis"),
                         dir,
                         mb,
                         nnzb,
                         (const void*&)descr,
                         (const void*&)bsr_val,
                         (const void*&)bsr_row_ptr,
                         (const void*&)bsr_col_ind,
                         block_dim,
                         (const void*&)info,
                         solve,
                         analysis);

    ROCSPARSE_CHECKARG_ENUM(1, dir);
    ROCSPARSE_CHECKARG_SIZE(2, mb);
    ROCSPARSE_CHECKARG_SIZE(3, nnzb);
    ROCSPARSE_CHECKARG_POINTER(4, descr);
    ROCSPARSE_CHECKARG(
        4, descr, (descr->type != rocsparse_matrix_type_general), rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(4,
                       descr,
                       (descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);
    ROCSPARSE_CHECKARG_ARRAY(5, nnzb, bsr_val);
    ROCSPARSE_CHECKARG_ARRAY(6, mb, bsr_row_ptr);
    ROCSPARSE_CHECKARG_ARRAY(7, nnzb, bsr_col_ind);

    ROCSPARSE_CHECKARG_SIZE(8, block_dim);
    ROCSPARSE_CHECKARG(8, block_dim, (block_dim == 0), rocsparse_status_invalid_size);

    ROCSPARSE_CHECKARG_POINTER(9, info);
    ROCSPARSE_CHECKARG_ENUM(10, analysis);
    ROCSPARSE_CHECKARG_ENUM(11, solve);

    ROCSPARSE_CHECKARG_ARRAY(12, mb, temp_buffer);

    // Quick return if possible
    if(mb == 0)
    {
        return rocsparse_status_success;
    }

    // Differentiate the analysis policies
    if(analysis == rocsparse_analysis_policy_reuse)
    {
        // We try to re-use already analyzed lower part, if available.
        // It is the user's responsibility that this data is still valid,
        // since he passed the 'reuse' flag.

        // If bsric0 meta data is already available, do nothing
        if(info->bsric0_info != nullptr)
        {
            return rocsparse_status_success;
        }

        // Check for other lower analysis meta data
        if(info->bsrilu0_info != nullptr)
        {
            // bsrilu0 meta data
            info->bsric0_info = info->bsrilu0_info;
            return rocsparse_status_success;
        }

        if(info->bsrsv_lower_info != nullptr)
        {
            info->bsric0_info = info->bsrsv_lower_info;
            return rocsparse_status_success;
        }
    }

    // User is explicitly asking to force a re-analysis, or no valid data has been
    // found to be re-used.

    // Clear bsric0 info
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::destroy_trm_info(info->bsric0_info));

    // Create bsric0 info
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::create_trm_info(&info->bsric0_info));

    // Perform analysis
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::trm_analysis(handle,
                                                      rocsparse_operation_none,
                                                      mb,
                                                      nnzb,
                                                      descr,
                                                      bsr_val,
                                                      bsr_row_ptr,
                                                      bsr_col_ind,
                                                      info->bsric0_info,
                                                      (rocsparse_int**)&info->zero_pivot,
                                                      temp_buffer));

    return rocsparse_status_success;
}

namespace rocsparse
{
    template <typename T>
    inline void bsric0_launcher(rocsparse_handle     handle,
                                rocsparse_direction  dir,
                                rocsparse_int        mb,
                                rocsparse_int        max_nnzb,
                                rocsparse_index_base base,
                                T*                   bsr_val,
                                const rocsparse_int* bsr_row_ptr,
                                const rocsparse_int* bsr_col_ind,
                                rocsparse_int        block_dim,
                                rocsparse_mat_info   info,
                                int*                 done_array)
    {
        ROCSPARSE_ROUTINE_TRACE;

        dim3 bsric0_blocks(mb);

        if(handle->wavefront_size == 32)
        {
            LAUNCH_BSRIC_33_inf(T, 32, 32, false);
        }
        else
        {

            const std::string gcn_arch_name = rocsparse::handle_get_arch_name(handle);
            if(gcn_arch_name == rocpsarse_arch_names::gfx908 && handle->asic_rev < 2)
            {
                LAUNCH_BSRIC_33_inf(T, 64, 64, true);
            }
            else
            {
                if(max_nnzb <= 32)
                {
                    if(block_dim == 1)
                    {
                        LAUNCH_BSRIC_2_8_UNROLLED(T, 1, 32, 1);
                    }
                    else if(block_dim == 2)
                    {
                        LAUNCH_BSRIC_2_8_UNROLLED(T, 4, 32, 2);
                    }
                    else if(block_dim == 3)
                    {
                        LAUNCH_BSRIC_2_8_UNROLLED(T, 9, 32, 3);
                    }
                    else if(block_dim == 4)
                    {
                        LAUNCH_BSRIC_2_8_UNROLLED(T, 16, 32, 4);
                    }
                    else if(block_dim == 5)
                    {
                        LAUNCH_BSRIC_2_8_UNROLLED(T, 25, 32, 5);
                    }
                    else if(block_dim == 6)
                    {
                        LAUNCH_BSRIC_2_8_UNROLLED(T, 36, 32, 6);
                    }
                    else if(block_dim == 7)
                    {
                        LAUNCH_BSRIC_2_8_UNROLLED(T, 49, 32, 7);
                    }
                    else if(block_dim == 8)
                    {
                        LAUNCH_BSRIC_2_8_UNROLLED(T, 64, 32, 8);
                    }
                    else if(block_dim <= 16)
                    {
                        LAUNCH_BSRIC_9_16(T, 64, 32, 16);
                    }
                    else if(block_dim <= 32)
                    {
                        LAUNCH_BSRIC_17_32(T, 64, 32, 32);
                    }
                    else
                    {
                        LAUNCH_BSRIC_33_inf(T, 64, 64, false);
                    }
                }
                else if(max_nnzb <= 64)
                {
                    if(block_dim <= 8)
                    {
                        LAUNCH_BSRIC_2_8(T, 64, 64, 8);
                    }
                    else if(block_dim <= 16)
                    {
                        LAUNCH_BSRIC_9_16(T, 64, 64, 16);
                    }
                    else if(block_dim <= 32)
                    {
                        LAUNCH_BSRIC_17_32(T, 64, 64, 32);
                    }
                    else
                    {
                        LAUNCH_BSRIC_33_inf(T, 64, 64, false);
                    }
                }
                else if(max_nnzb <= 128)
                {
                    if(block_dim <= 8)
                    {
                        LAUNCH_BSRIC_2_8(T, 64, 128, 8);
                    }
                    else if(block_dim <= 16)
                    {
                        LAUNCH_BSRIC_9_16(T, 64, 128, 16);
                    }
                    else if(block_dim <= 32)
                    {
                        LAUNCH_BSRIC_17_32(T, 64, 128, 32);
                    }
                    else
                    {
                        LAUNCH_BSRIC_33_inf(T, 64, 64, false);
                    }
                }
                else
                {
                    LAUNCH_BSRIC_33_inf(T, 64, 64, false);
                }
            }
        }
    }
}

template <typename T>
rocsparse_status rocsparse::bsric0_template(rocsparse_handle          handle,
                                            rocsparse_direction       dir,
                                            rocsparse_int             mb,
                                            rocsparse_int             nnzb,
                                            const rocsparse_mat_descr descr,
                                            T*                        bsr_val,
                                            const rocsparse_int*      bsr_row_ptr,
                                            const rocsparse_int*      bsr_col_ind,
                                            rocsparse_int             block_dim,
                                            rocsparse_mat_info        info,
                                            rocsparse_solve_policy    policy,
                                            void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xbsric0"),
                         mb,
                         nnzb,
                         (const void*&)descr,
                         (const void*&)bsr_val,
                         (const void*&)bsr_row_ptr,
                         (const void*&)bsr_col_ind,
                         block_dim,
                         (const void*&)info,
                         policy,
                         (const void*&)temp_buffer);

    ROCSPARSE_CHECKARG_ENUM(1, dir);
    ROCSPARSE_CHECKARG_SIZE(2, mb);
    ROCSPARSE_CHECKARG_SIZE(3, nnzb);
    ROCSPARSE_CHECKARG_POINTER(4, descr);
    ROCSPARSE_CHECKARG(
        4, descr, (descr->type != rocsparse_matrix_type_general), rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(4,
                       descr,
                       (descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);
    ROCSPARSE_CHECKARG_ARRAY(5, nnzb, bsr_val);
    ROCSPARSE_CHECKARG_ARRAY(6, mb, bsr_row_ptr);
    ROCSPARSE_CHECKARG_ARRAY(7, nnzb, bsr_col_ind);

    ROCSPARSE_CHECKARG_SIZE(8, block_dim);
    ROCSPARSE_CHECKARG(8, block_dim, (block_dim == 0), rocsparse_status_invalid_size);

    ROCSPARSE_CHECKARG_POINTER(9, info);

    ROCSPARSE_CHECKARG_ENUM(10, policy);
    ROCSPARSE_CHECKARG_ARRAY(11, mb, temp_buffer);

    ROCSPARSE_CHECKARG(
        9, info, ((mb > 0) && (info->bsric0_info == nullptr)), rocsparse_status_invalid_pointer);

    if(mb == 0)
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
    RETURN_IF_HIP_ERROR(hipMemsetAsync(d_done_array, 0, sizeof(int) * mb, stream));

    // Max nnz blocks per row
    rocsparse_int max_nnzb = info->bsric0_info->max_nnz;

    rocsparse::bsric0_launcher<T>(handle,
                                  dir,
                                  mb,
                                  max_nnzb,
                                  descr->base,
                                  bsr_val,
                                  bsr_row_ptr,
                                  bsr_col_ind,
                                  block_dim,
                                  info,
                                  d_done_array);

    return rocsparse_status_success;
}

namespace rocsparse
{
    template <typename T>
    rocsparse_status bsric0_buffer_size_template(rocsparse_handle          handle,
                                                 rocsparse_direction       dir,
                                                 rocsparse_int             mb,
                                                 rocsparse_int             nnzb,
                                                 const rocsparse_mat_descr descr,
                                                 const T*                  bsr_val,
                                                 const rocsparse_int*      bsr_row_ptr,
                                                 const rocsparse_int*      bsr_col_ind,
                                                 rocsparse_int             block_dim,
                                                 rocsparse_mat_info        info,
                                                 size_t*                   buffer_size)
    {
        ROCSPARSE_ROUTINE_TRACE;

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsv_buffer_size_template(handle,
                                                                        rocsparse_operation_none,
                                                                        mb,
                                                                        nnzb,
                                                                        descr,
                                                                        bsr_val,
                                                                        bsr_row_ptr,
                                                                        bsr_col_ind,
                                                                        info,
                                                                        buffer_size));
        return rocsparse_status_success;
    }

    template <typename T>
    rocsparse_status bsric0_buffer_size_impl(rocsparse_handle          handle, //0
                                             rocsparse_direction       dir, //1
                                             rocsparse_int             mb, //2
                                             rocsparse_int             nnzb, //3
                                             const rocsparse_mat_descr descr, //4
                                             const T*                  bsr_val, //5
                                             const rocsparse_int*      bsr_row_ptr, //6
                                             const rocsparse_int*      bsr_col_ind, //7
                                             rocsparse_int             block_dim, //8
                                             rocsparse_mat_info        info, //9
                                             size_t*                   buffer_size)
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);

        rocsparse::log_trace(handle,
                             rocsparse::replaceX<T>("rocsparse_Xbsric0_buffer_size"),
                             dir,
                             mb,
                             nnzb,
                             (const void*&)descr,
                             (const void*&)bsr_val,
                             (const void*&)bsr_row_ptr,
                             (const void*&)bsr_col_ind,
                             block_dim,
                             (const void*&)info,
                             (const void*&)buffer_size);

        ROCSPARSE_CHECKARG_ENUM(1, dir);
        ROCSPARSE_CHECKARG_SIZE(2, mb);
        ROCSPARSE_CHECKARG_SIZE(3, nnzb);
        ROCSPARSE_CHECKARG_POINTER(4, descr);
        ROCSPARSE_CHECKARG(4,
                           descr,
                           (descr->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(4,
                           descr,
                           (descr->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);
        ROCSPARSE_CHECKARG_ARRAY(5, nnzb, bsr_val);
        ROCSPARSE_CHECKARG_ARRAY(6, mb, bsr_row_ptr);
        ROCSPARSE_CHECKARG_ARRAY(7, nnzb, bsr_col_ind);
        ROCSPARSE_CHECKARG_SIZE(8, block_dim);
        ROCSPARSE_CHECKARG(8, block_dim, (block_dim == 0), rocsparse_status_invalid_size);
        ROCSPARSE_CHECKARG_POINTER(9, info);
        ROCSPARSE_CHECKARG_POINTER(10, buffer_size);

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsric0_buffer_size_template(handle,
                                                                         dir,
                                                                         mb,
                                                                         nnzb,
                                                                         descr,
                                                                         bsr_val,
                                                                         bsr_row_ptr,
                                                                         bsr_col_ind,
                                                                         block_dim,
                                                                         info,
                                                                         buffer_size));

        return rocsparse_status_success;
    }
}

#define CIMPL(NAME, T)                                                              \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,              \
                                     rocsparse_direction       dir,                 \
                                     rocsparse_int             mb,                  \
                                     rocsparse_int             nnzb,                \
                                     const rocsparse_mat_descr descr,               \
                                     const T*                  bsr_val,             \
                                     const rocsparse_int*      bsr_row_ptr,         \
                                     const rocsparse_int*      bsr_col_ind,         \
                                     rocsparse_int             block_dim,           \
                                     rocsparse_mat_info        info,                \
                                     size_t*                   buffer_size)         \
    try                                                                             \
    {                                                                               \
        ROCSPARSE_ROUTINE_TRACE;                                                    \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsric0_buffer_size_impl(handle,        \
                                                                     dir,           \
                                                                     mb,            \
                                                                     nnzb,          \
                                                                     descr,         \
                                                                     bsr_val,       \
                                                                     bsr_row_ptr,   \
                                                                     bsr_col_ind,   \
                                                                     block_dim,     \
                                                                     info,          \
                                                                     buffer_size)); \
        return rocsparse_status_success;                                            \
    }                                                                               \
    catch(...)                                                                      \
    {                                                                               \
        RETURN_ROCSPARSE_EXCEPTION();                                               \
    }

CIMPL(rocsparse_sbsric0_buffer_size, float);
CIMPL(rocsparse_dbsric0_buffer_size, double);
CIMPL(rocsparse_cbsric0_buffer_size, rocsparse_float_complex);
CIMPL(rocsparse_zbsric0_buffer_size, rocsparse_double_complex);
#undef CIMPL

#define CIMPL(NAME, T)                                                               \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,               \
                                     rocsparse_direction       dir,                  \
                                     rocsparse_int             mb,                   \
                                     rocsparse_int             nnzb,                 \
                                     const rocsparse_mat_descr descr,                \
                                     const T*                  bsr_val,              \
                                     const rocsparse_int*      bsr_row_ptr,          \
                                     const rocsparse_int*      bsr_col_ind,          \
                                     rocsparse_int             block_dim,            \
                                     rocsparse_mat_info        info,                 \
                                     rocsparse_analysis_policy analysis,             \
                                     rocsparse_solve_policy    solve,                \
                                     void*                     temp_buffer)          \
    try                                                                              \
    {                                                                                \
        ROCSPARSE_ROUTINE_TRACE;                                                     \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsric0_analysis_template(handle,        \
                                                                      dir,           \
                                                                      mb,            \
                                                                      nnzb,          \
                                                                      descr,         \
                                                                      bsr_val,       \
                                                                      bsr_row_ptr,   \
                                                                      bsr_col_ind,   \
                                                                      block_dim,     \
                                                                      info,          \
                                                                      analysis,      \
                                                                      solve,         \
                                                                      temp_buffer)); \
        return rocsparse_status_success;                                             \
    }                                                                                \
    catch(...)                                                                       \
    {                                                                                \
        RETURN_ROCSPARSE_EXCEPTION();                                                \
    }

CIMPL(rocsparse_sbsric0_analysis, float);
CIMPL(rocsparse_dbsric0_analysis, double);
CIMPL(rocsparse_cbsric0_analysis, rocsparse_float_complex);
CIMPL(rocsparse_zbsric0_analysis, rocsparse_double_complex);
#undef CIMPL

#define CIMPL(NAME, T)                                                      \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,      \
                                     rocsparse_direction       dir,         \
                                     rocsparse_int             mb,          \
                                     rocsparse_int             nnzb,        \
                                     const rocsparse_mat_descr descr,       \
                                     T*                        bsr_val,     \
                                     const rocsparse_int*      bsr_row_ptr, \
                                     const rocsparse_int*      bsr_col_ind, \
                                     rocsparse_int             block_dim,   \
                                     rocsparse_mat_info        info,        \
                                     rocsparse_solve_policy    policy,      \
                                     void*                     temp_buffer) \
    try                                                                     \
    {                                                                       \
        ROCSPARSE_ROUTINE_TRACE;                                            \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsric0_template(handle,        \
                                                             dir,           \
                                                             mb,            \
                                                             nnzb,          \
                                                             descr,         \
                                                             bsr_val,       \
                                                             bsr_row_ptr,   \
                                                             bsr_col_ind,   \
                                                             block_dim,     \
                                                             info,          \
                                                             policy,        \
                                                             temp_buffer)); \
        return rocsparse_status_success;                                    \
    }                                                                       \
    catch(...)                                                              \
    {                                                                       \
        RETURN_ROCSPARSE_EXCEPTION();                                       \
    }

CIMPL(rocsparse_sbsric0, float);
CIMPL(rocsparse_dbsric0, double);
CIMPL(rocsparse_cbsric0, rocsparse_float_complex);
CIMPL(rocsparse_zbsric0, rocsparse_double_complex);
#undef CIMPL

extern "C" rocsparse_status rocsparse_bsric0_clear(rocsparse_handle handle, rocsparse_mat_info info)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    rocsparse::log_trace(handle, "rocsparse_bsric0_clear", (const void*&)info);

    ROCSPARSE_CHECKARG_POINTER(1, info);

    if(!rocsparse::check_trm_shared(info, info->bsric0_info))
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::destroy_trm_info(info->bsric0_info));
    }

    info->bsric0_info = nullptr;

    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

extern "C" rocsparse_status rocsparse_bsric0_zero_pivot(rocsparse_handle   handle,
                                                        rocsparse_mat_info info,
                                                        rocsparse_int*     position)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    // Logging
    rocsparse::log_trace(
        handle, "rocsparse_bsric0_zero_pivot", (const void*&)info, (const void*&)position);

    ROCSPARSE_CHECKARG_POINTER(1, info);
    ROCSPARSE_CHECKARG_POINTER(2, position);

    // Stream
    hipStream_t stream = handle->stream;

    // If mb == 0 || nnzb == 0 it can happen, that info structure is not created.
    // In this case, always return -1.
    if(info->bsric0_info == nullptr)
    {
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            RETURN_IF_HIP_ERROR(hipMemsetAsync(position, 0xFF, sizeof(rocsparse_int), stream));
        }
        else
        {
            *position = -1;
        }

        return rocsparse_status_success;
    }

    // Differentiate between pointer modes
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        // rocsparse_pointer_mode_device
        rocsparse_int pivot;

        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &pivot, info->zero_pivot, sizeof(rocsparse_int), hipMemcpyDeviceToHost, stream));

        // Wait for host transfer to finish
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

        if(pivot == std::numeric_limits<rocsparse_int>::max())
        {
            RETURN_IF_HIP_ERROR(hipMemsetAsync(position, 0xFF, sizeof(rocsparse_int), stream));
        }
        else
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(position,
                                               info->zero_pivot,
                                               sizeof(rocsparse_int),
                                               hipMemcpyDeviceToDevice,
                                               stream));

            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_zero_pivot);
        }
    }
    else
    {
        // rocsparse_pointer_mode_host
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            position, info->zero_pivot, sizeof(rocsparse_int), hipMemcpyDeviceToHost, stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

        // If no zero pivot is found, set -1
        if(*position == std::numeric_limits<rocsparse_int>::max())
        {
            *position = -1;
        }
        else
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_zero_pivot);
        }
    }

    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
