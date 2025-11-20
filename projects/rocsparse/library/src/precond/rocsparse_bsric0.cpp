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
#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

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
        (const rocsparse_int*)trm_info->get_diag_ind(),                               \
        done_array,                                                                   \
        (const rocsparse_int*)trm_info->get_row_map(),                                \
        (rocsparse_int*)zero_pivot,                                                   \
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
        (const rocsparse_int*)trm_info->get_diag_ind(),                      \
        done_array,                                                          \
        (const rocsparse_int*)trm_info->get_row_map(),                       \
        (rocsparse_int*)zero_pivot,                                          \
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
        (const rocsparse_int*)trm_info->get_diag_ind(),                       \
        done_array,                                                           \
        (const rocsparse_int*)trm_info->get_row_map(),                        \
        (rocsparse_int*)zero_pivot,                                           \
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
        (const rocsparse_int*)trm_info->get_diag_ind(),                        \
        done_array,                                                            \
        (const rocsparse_int*)trm_info->get_row_map(),                         \
        (rocsparse_int*)zero_pivot,                                            \
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
        (const rocsparse_int*)trm_info->get_diag_ind(),                   \
        done_array,                                                       \
        (const rocsparse_int*)trm_info->get_row_map(),                    \
        (rocsparse_int*)zero_pivot,                                       \
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

    if(analysis == rocsparse_analysis_policy_reuse)
    {
        auto trm = info->get_bsric0_info(rocsparse_operation_none, rocsparse_fill_mode_lower);

        trm = (trm != nullptr)
                  ? trm
                  : info->get_bsrilu0_info(rocsparse_operation_none, rocsparse_fill_mode_lower);

        trm = (trm != nullptr)
                  ? trm
                  : info->get_bsrsv_info(rocsparse_operation_none, rocsparse_fill_mode_lower);

        if(trm != nullptr)
        {
            info->set_bsric0_info(rocsparse_operation_none, rocsparse_fill_mode_lower, trm);
            return rocsparse_status_success;
        }
    }

    auto bsric0_info = info->get_bsric0_info();
    // Perform analysis
    RETURN_IF_ROCSPARSE_ERROR(bsric0_info->recreate(rocsparse_operation_none,
                                                    rocsparse_fill_mode_lower,
                                                    handle,
                                                    rocsparse_operation_none,
                                                    mb,
                                                    nnzb,
                                                    descr,
                                                    bsr_val,
                                                    bsr_row_ptr,
                                                    bsr_col_ind,
                                                    temp_buffer));

    return rocsparse_status_success;
}

namespace rocsparse
{
    template <typename T>
    inline void bsric0_launcher(rocsparse_handle       handle,
                                rocsparse_direction    dir,
                                rocsparse_int          mb,
                                rocsparse_int          max_nnzb,
                                rocsparse_index_base   base,
                                T*                     bsr_val,
                                const rocsparse_int*   bsr_row_ptr,
                                const rocsparse_int*   bsr_col_ind,
                                rocsparse_int          block_dim,
                                rocsparse::trm_info_t* trm_info,
                                void*                  zero_pivot,
                                int*                   done_array)
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

    auto  bsric0_info = info->get_bsric0_info();
    auto  trm_info    = info->get_bsric0_info(rocsparse_operation_none, rocsparse_fill_mode_lower);
    void* zero_pivot  = bsric0_info->get_zero_pivot();
    ROCSPARSE_CHECKARG(
        9, info, ((mb > 0) && (trm_info == nullptr)), rocsparse_status_invalid_pointer);

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
    const rocsparse_int max_nnzb = trm_info->get_max_nnz();

    rocsparse::bsric0_launcher<T>(handle,
                                  dir,
                                  mb,
                                  max_nnzb,
                                  descr->base,
                                  bsr_val,
                                  bsr_row_ptr,
                                  bsr_col_ind,
                                  block_dim,
                                  trm_info,
                                  zero_pivot,
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

        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::csrsv_buffer_size_template<rocsparse_int, rocsparse_int, T>(
                handle,
                rocsparse_operation_none,
                mb,
                nnzb,
                descr,
                bsr_val,
                bsr_row_ptr,
                bsr_col_ind,
                info,
                buffer_size)));
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

    info->clear_bsric0_info();

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

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

    auto bsric0_info = info->get_bsric0_info();
    {
        auto status = bsric0_info->copy_zero_pivot_async(handle->pointer_mode,
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
