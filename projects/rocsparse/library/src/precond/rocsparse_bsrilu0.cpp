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

#include "internal/precond/rocsparse_bsrilu0.h"
#include "internal/level2/rocsparse_csrsv.h"

#include "rocsparse_bsrilu0.hpp"

#include "../level2/rocsparse_csrsv.hpp"
#include "bsrilu0_device.h"
#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

#define LAUNCH_BSRILU28()                                                   \
    THROW_IF_HIPLAUNCHKERNELGGL_ERROR(                                      \
        (rocsparse::bsrilu0_2_8<64, 64, 8>),                                \
        dim3(mb),                                                           \
        dim3(8, 8),                                                         \
        0,                                                                  \
        handle->stream,                                                     \
        dir,                                                                \
        mb,                                                                 \
        bsr_row_ptr,                                                        \
        bsr_col_ind,                                                        \
        bsr_val,                                                            \
        (const rocsparse_int*)trm_info->get_diag_ind(),                     \
        block_dim,                                                          \
        done_array,                                                         \
        (const rocsparse_int*)trm_info->get_row_map(),                      \
        (rocsparse_int*)zero_pivot,                                         \
        base,                                                               \
        boost_enable,                                                       \
        boost_tol_size,                                                     \
        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_tol_32), \
        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_tol_64), \
        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_val),    \
        handle->pointer_mode == rocsparse_pointer_mode_host)

#define LAUNCH_BSRILU932(dim)                                               \
    THROW_IF_HIPLAUNCHKERNELGGL_ERROR(                                      \
        (rocsparse::bsrilu0_9_32<64, 64, dim>),                             \
        dim3(mb),                                                           \
        dim3(dim, 64 / dim),                                                \
        0,                                                                  \
        handle->stream,                                                     \
        dir,                                                                \
        mb,                                                                 \
        bsr_row_ptr,                                                        \
        bsr_col_ind,                                                        \
        bsr_val,                                                            \
        (const rocsparse_int*)trm_info->get_diag_ind(),                     \
        block_dim,                                                          \
        done_array,                                                         \
        (const rocsparse_int*)trm_info->get_row_map(),                      \
        (rocsparse_int*)zero_pivot,                                         \
        base,                                                               \
        boost_enable,                                                       \
        boost_tol_size,                                                     \
        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_tol_32), \
        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_tol_64), \
        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_val),    \
        handle->pointer_mode == rocsparse_pointer_mode_host)

#define LAUNCH_BSRILU3364()                                                 \
    THROW_IF_HIPLAUNCHKERNELGGL_ERROR(                                      \
        (rocsparse::bsrilu0_33_64<64, 64, 64>),                             \
        dim3(mb),                                                           \
        dim3(64),                                                           \
        0,                                                                  \
        handle->stream,                                                     \
        dir,                                                                \
        mb,                                                                 \
        bsr_row_ptr,                                                        \
        bsr_col_ind,                                                        \
        bsr_val,                                                            \
        (const rocsparse_int*)trm_info->get_diag_ind(),                     \
        block_dim,                                                          \
        done_array,                                                         \
        (const rocsparse_int*)trm_info->get_row_map(),                      \
        (rocsparse_int*)zero_pivot,                                         \
        base,                                                               \
        boost_enable,                                                       \
        boost_tol_size,                                                     \
        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_tol_32), \
        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_tol_64), \
        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_val),    \
        handle->pointer_mode == rocsparse_pointer_mode_host)

#define LAUNCH_BSRILU65inf(sleep, wfsize)                                   \
    THROW_IF_HIPLAUNCHKERNELGGL_ERROR(                                      \
        (rocsparse::bsrilu0_general<128, wfsize, sleep>),                   \
        dim3((wfsize * mb - 1) / 128 + 1),                                  \
        dim3(128),                                                          \
        0,                                                                  \
        handle->stream,                                                     \
        dir,                                                                \
        mb,                                                                 \
        bsr_row_ptr,                                                        \
        bsr_col_ind,                                                        \
        bsr_val,                                                            \
        (const rocsparse_int*)trm_info->get_diag_ind(),                     \
        block_dim,                                                          \
        done_array,                                                         \
        (const rocsparse_int*)trm_info->get_row_map(),                      \
        (rocsparse_int*)zero_pivot,                                         \
        base,                                                               \
        boost_enable,                                                       \
        boost_tol_size,                                                     \
        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_tol_32), \
        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_tol_64), \
        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_val),    \
        handle->pointer_mode == rocsparse_pointer_mode_host)
namespace rocsparse
{
    template <typename T>
    rocsparse_status bsrilu0_numeric_boost_template(rocsparse_handle   handle,
                                                    rocsparse_mat_info info,
                                                    int                enable_boost,
                                                    size_t             boost_tol_size,
                                                    const void*        boost_tol,
                                                    const T*           boost_val)
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);

        rocsparse::log_trace(handle,
                             rocsparse::replaceX<T>("rocsparse_Xbsrilu0_numeric_boost"),
                             (const void*&)info,
                             enable_boost,
                             (const void*&)boost_tol,
                             (const void*&)boost_val);

        ROCSPARSE_CHECKARG_POINTER(1, info);

        // Reset boost
        info->boost_enable = 0;

        if(enable_boost)
        {
            ROCSPARSE_CHECKARG_POINTER(3, boost_tol);
            ROCSPARSE_CHECKARG_POINTER(4, boost_val);

            info->boost_enable   = enable_boost;
            info->boost_tol_size = boost_tol_size;
            info->boost_tol      = boost_tol;
            info->boost_val      = reinterpret_cast<const void*>(boost_val);
        }

        return rocsparse_status_success;
    }
}

template <typename T>
rocsparse_status rocsparse::bsrilu0_analysis_template(rocsparse_handle          handle, //0
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

    // Logging
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xbsrilu0_analysis"),
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

    if(mb == 0)
    {
        return rocsparse_status_success;
    }

    if(analysis == rocsparse_analysis_policy_reuse)
    {
        auto trm = info->get_bsrilu0_info(rocsparse_operation_none, rocsparse_fill_mode_lower);

        trm = (trm != nullptr)
                  ? trm
                  : info->get_bsric0_info(rocsparse_operation_none, rocsparse_fill_mode_lower);

        trm = (trm != nullptr)
                  ? trm
                  : info->get_bsrsv_info(rocsparse_operation_none, rocsparse_fill_mode_lower);

        if(trm != nullptr)
        {
            info->set_bsrilu0_info(rocsparse_operation_none, rocsparse_fill_mode_lower, trm);
            return rocsparse_status_success;
        }
    }

    auto bsrilu0_info = info->get_bsrilu0_info();
    // Perform analysis
    RETURN_IF_ROCSPARSE_ERROR(bsrilu0_info->recreate(rocsparse_operation_none,
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
    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, uint32_t BSRDIM, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsrilu0_2_8(rocsparse_direction  dir,
                     rocsparse_int        mb,
                     const rocsparse_int* bsr_row_ptr,
                     const rocsparse_int* bsr_col_ind,
                     T*                   bsr_val,
                     const rocsparse_int* bsr_diag_ind,
                     rocsparse_int        bsr_dim,
                     int*                 done_array,
                     const rocsparse_int* map,
                     rocsparse_int*       zero_pivot,
                     rocsparse_index_base idx_base,
                     int                  enable_boost,
                     size_t               size_boost_tol,
                     ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(float, boost_tol_32),
                     ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(double, boost_tol_64),
                     ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, boost_val),
                     bool is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET_IF(enable_boost, boost_tol_32);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET_IF(enable_boost, boost_tol_64);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET_IF(enable_boost, boost_val);
        const double boost_tol = (size_boost_tol == sizeof(double)) ? boost_tol_64 : boost_tol_32;

        rocsparse::bsrilu0_2_8_device<BLOCKSIZE, WFSIZE, BSRDIM>(dir,
                                                                 mb,
                                                                 bsr_row_ptr,
                                                                 bsr_col_ind,
                                                                 bsr_val,
                                                                 bsr_diag_ind,
                                                                 bsr_dim,
                                                                 done_array,
                                                                 map,
                                                                 zero_pivot,
                                                                 idx_base,
                                                                 enable_boost,
                                                                 boost_tol,
                                                                 boost_val);
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, uint32_t BSRDIM, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsrilu0_9_32(rocsparse_direction  dir,
                      rocsparse_int        mb,
                      const rocsparse_int* bsr_row_ptr,
                      const rocsparse_int* bsr_col_ind,
                      T*                   bsr_val,
                      const rocsparse_int* bsr_diag_ind,
                      rocsparse_int        bsr_dim,
                      int*                 done_array,
                      const rocsparse_int* map,
                      rocsparse_int*       zero_pivot,
                      rocsparse_index_base idx_base,
                      int                  enable_boost,
                      size_t               size_boost_tol,
                      ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(float, boost_tol_32),
                      ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(double, boost_tol_64),
                      ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, boost_val),
                      bool is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET_IF(enable_boost, boost_tol_32);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET_IF(enable_boost, boost_tol_64);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET_IF(enable_boost, boost_val);
        const double boost_tol = (size_boost_tol == sizeof(double)) ? boost_tol_64 : boost_tol_32;

        rocsparse::bsrilu0_9_32_device<BLOCKSIZE, WFSIZE, BSRDIM>(dir,
                                                                  mb,
                                                                  bsr_row_ptr,
                                                                  bsr_col_ind,
                                                                  bsr_val,
                                                                  bsr_diag_ind,
                                                                  bsr_dim,
                                                                  done_array,
                                                                  map,
                                                                  zero_pivot,
                                                                  idx_base,
                                                                  enable_boost,
                                                                  boost_tol,
                                                                  boost_val);
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, uint32_t BSRDIM, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsrilu0_33_64(rocsparse_direction  dir,
                       rocsparse_int        mb,
                       const rocsparse_int* bsr_row_ptr,
                       const rocsparse_int* bsr_col_ind,
                       T*                   bsr_val,
                       const rocsparse_int* bsr_diag_ind,
                       rocsparse_int        bsr_dim,
                       int*                 done_array,
                       const rocsparse_int* map,
                       rocsparse_int*       zero_pivot,
                       rocsparse_index_base idx_base,
                       int                  enable_boost,
                       size_t               size_boost_tol,
                       ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(float, boost_tol_32),
                       ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(double, boost_tol_64),
                       ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, boost_val),
                       bool is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET_IF(enable_boost, boost_tol_32);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET_IF(enable_boost, boost_tol_64);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET_IF(enable_boost, boost_val);
        const double boost_tol = (size_boost_tol == sizeof(double)) ? boost_tol_64 : boost_tol_32;

        rocsparse::bsrilu0_33_64_device<BLOCKSIZE, WFSIZE, BSRDIM>(dir,
                                                                   mb,
                                                                   bsr_row_ptr,
                                                                   bsr_col_ind,
                                                                   bsr_val,
                                                                   bsr_diag_ind,
                                                                   bsr_dim,
                                                                   done_array,
                                                                   map,
                                                                   zero_pivot,
                                                                   idx_base,
                                                                   enable_boost,
                                                                   boost_tol,
                                                                   boost_val);
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, bool SLEEP, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsrilu0_general(rocsparse_direction  dir,
                         rocsparse_int        mb,
                         const rocsparse_int* bsr_row_ptr,
                         const rocsparse_int* bsr_col_ind,
                         T*                   bsr_val,
                         const rocsparse_int* bsr_diag_ind,
                         rocsparse_int        bsr_dim,
                         int*                 done_array,
                         const rocsparse_int* map,
                         rocsparse_int*       zero_pivot,
                         rocsparse_index_base idx_base,
                         int                  enable_boost,
                         size_t               size_boost_tol,
                         ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(float, boost_tol_32),
                         ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(double, boost_tol_64),
                         ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, boost_val),
                         bool is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET_IF(enable_boost, boost_tol_32);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET_IF(enable_boost, boost_tol_64);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET_IF(enable_boost, boost_val);
        const double boost_tol = (size_boost_tol == sizeof(double)) ? boost_tol_64 : boost_tol_32;

        rocsparse::bsrilu0_general_device<BLOCKSIZE, WFSIZE, SLEEP>(dir,
                                                                    mb,
                                                                    bsr_row_ptr,
                                                                    bsr_col_ind,
                                                                    bsr_val,
                                                                    bsr_diag_ind,
                                                                    bsr_dim,
                                                                    done_array,
                                                                    map,
                                                                    zero_pivot,
                                                                    idx_base,
                                                                    enable_boost,
                                                                    boost_tol,
                                                                    boost_val);
    }

    template <
        typename T,
        typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, double>::value
                                    || std::is_same<T, rocsparse_float_complex>::value,
                                int>::type
        = 0>
    inline void bsrilu0_launcher(rocsparse_handle       handle,
                                 rocsparse_direction    dir,
                                 rocsparse_int          mb,
                                 rocsparse_index_base   base,
                                 T*                     bsr_val,
                                 const rocsparse_int*   bsr_row_ptr,
                                 const rocsparse_int*   bsr_col_ind,
                                 rocsparse_int          block_dim,
                                 rocsparse::trm_info_t* trm_info,
                                 void*                  zero_pivot,
                                 bool                   boost_enable,
                                 size_t                 boost_tol_size,
                                 int*                   done_array,
                                 const void*            boost_tol,
                                 const T*               boost_val)
    {
        ROCSPARSE_ROUTINE_TRACE;

        const float*      boost_tol_32  = (boost_enable) ? (const float*)boost_tol : nullptr;
        const double*     boost_tol_64  = (boost_enable) ? (const double*)boost_tol : nullptr;
        const std::string gcn_arch_name = rocsparse::handle_get_arch_name(handle);
        if(gcn_arch_name == rocpsarse_arch_names::gfx908 && handle->asic_rev < 2)
        {
            LAUNCH_BSRILU65inf(true, 64);
        }
        else if(handle->wavefront_size == 32)
        {
            LAUNCH_BSRILU65inf(false, 32);
        }
        else
        {
            if(block_dim <= 8)
            {
                LAUNCH_BSRILU28();
            }
            else if(block_dim <= 16)
            {
                LAUNCH_BSRILU932(16);
            }
            else if(block_dim <= 32)
            {
                LAUNCH_BSRILU932(32);
            }
            else if(block_dim <= 64)
            {
                LAUNCH_BSRILU3364();
            }
            else
            {
                LAUNCH_BSRILU65inf(false, 64);
            }
        }
    }

    template <typename T,
              typename std::enable_if<std::is_same<T, rocsparse_double_complex>::value, int>::type
              = 0>
    inline void bsrilu0_launcher(rocsparse_handle       handle,
                                 rocsparse_direction    dir,
                                 rocsparse_int          mb,
                                 rocsparse_index_base   base,
                                 T*                     bsr_val,
                                 const rocsparse_int*   bsr_row_ptr,
                                 const rocsparse_int*   bsr_col_ind,
                                 rocsparse_int          block_dim,
                                 rocsparse::trm_info_t* trm_info,
                                 void*                  zero_pivot,
                                 bool                   boost_enable,
                                 size_t                 boost_tol_size,
                                 int*                   done_array,
                                 const void*            boost_tol,
                                 const T*               boost_val)
    {
        ROCSPARSE_ROUTINE_TRACE;

        const float*      boost_tol_32  = (boost_enable) ? (const float*)boost_tol : nullptr;
        const double*     boost_tol_64  = (boost_enable) ? (const double*)boost_tol : nullptr;
        const std::string gcn_arch_name = rocsparse::handle_get_arch_name(handle);
        if(gcn_arch_name == rocpsarse_arch_names::gfx908 && handle->asic_rev < 2)
        {
            LAUNCH_BSRILU65inf(true, 64);
        }
        else if(handle->wavefront_size == 32)
        {
            LAUNCH_BSRILU65inf(false, 32);
        }
        else
        {
            if(block_dim <= 8)
            {
                LAUNCH_BSRILU28();
            }
            else if(block_dim <= 16)
            {
                LAUNCH_BSRILU932(16);
            }
            else if(block_dim <= 32)
            {
                LAUNCH_BSRILU932(32);
            }
            else
            {
                LAUNCH_BSRILU65inf(false, 64);
            }
        }
    }
}

template <typename T>
rocsparse_status rocsparse::bsrilu0_template(rocsparse_handle          handle,
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
                         rocsparse::replaceX<T>("rocsparse_Xbsrilu0"),
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

    rocsparse::trm_info_t* trm_info
        = info->get_bsrilu0_info(rocsparse_operation_none, rocsparse_fill_mode_lower);
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
    int* done_array = reinterpret_cast<int*>(ptr);

    // Initialize buffers
    RETURN_IF_HIP_ERROR(hipMemsetAsync(done_array, 0, sizeof(int) * mb, stream));

    auto  bsrilu0_info = info->get_bsrilu0_info();
    void* zero_pivot   = bsrilu0_info->get_zero_pivot();
    rocsparse::bsrilu0_launcher(handle,
                                dir,
                                mb,
                                descr->base,
                                bsr_val,
                                bsr_row_ptr,
                                bsr_col_ind,
                                block_dim,
                                trm_info,
                                zero_pivot,
                                info->boost_enable,
                                info->boost_tol_size,
                                done_array,
                                info->boost_tol,
                                reinterpret_cast<const T*>(info->boost_val));

    return rocsparse_status_success;
}

extern "C" rocsparse_status rocsparse_bsrilu0_clear(rocsparse_handle   handle,
                                                    rocsparse_mat_info info)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    // Logging
    rocsparse::log_trace(handle, "rocsparse_bsrilu0_clear", (const void*&)info);

    ROCSPARSE_CHECKARG_POINTER(1, info);

    info->clear_bsrilu0_info();

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

namespace rocsparse
{
    template <typename T>
    rocsparse_status bsrilu0_buffer_size_template(rocsparse_handle          handle,
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
    rocsparse_status bsrilu0_buffer_size_impl(rocsparse_handle          handle,
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

        ROCSPARSE_CHECKARG_HANDLE(0, handle);

        rocsparse::log_trace(handle,
                             rocsparse::replaceX<T>("rocsparse_Xbsrilu0_buffer_size"),
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
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrilu0_buffer_size_template(handle,
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
                                     size_t*                   buffer_size)          \
    try                                                                              \
    {                                                                                \
        ROCSPARSE_ROUTINE_TRACE;                                                     \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrilu0_buffer_size_impl(handle,        \
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
        return rocsparse_status_success;                                             \
    }                                                                                \
    catch(...)                                                                       \
    {                                                                                \
        RETURN_ROCSPARSE_EXCEPTION();                                                \
    }

CIMPL(rocsparse_sbsrilu0_buffer_size, float);
CIMPL(rocsparse_dbsrilu0_buffer_size, double);
CIMPL(rocsparse_cbsrilu0_buffer_size, rocsparse_float_complex);
CIMPL(rocsparse_zbsrilu0_buffer_size, rocsparse_double_complex);
#undef CIMPL

#define CIMPL(NAME, U, V)                                                    \
    extern "C" rocsparse_status NAME(rocsparse_handle   handle,              \
                                     rocsparse_mat_info info,                \
                                     int                enable_boost,        \
                                     const U*           boost_tol,           \
                                     const V*           boost_val)           \
    try                                                                      \
    {                                                                        \
        ROCSPARSE_ROUTINE_TRACE;                                             \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrilu0_numeric_boost_template( \
            handle, info, enable_boost, sizeof(U), boost_tol, boost_val));   \
        return rocsparse_status_success;                                     \
    }                                                                        \
    catch(...)                                                               \
    {                                                                        \
        RETURN_ROCSPARSE_EXCEPTION();                                        \
    }

CIMPL(rocsparse_sbsrilu0_numeric_boost, float, float);
CIMPL(rocsparse_dbsrilu0_numeric_boost, double, double);
CIMPL(rocsparse_cbsrilu0_numeric_boost, float, rocsparse_float_complex);
CIMPL(rocsparse_zbsrilu0_numeric_boost, double, rocsparse_double_complex);
CIMPL(rocsparse_dsbsrilu0_numeric_boost, double, float);
CIMPL(rocsparse_dcbsrilu0_numeric_boost, double, rocsparse_float_complex);
#undef CIMPL

#define CIMPL(NAME, T)                                                                \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,                \
                                     rocsparse_direction       dir,                   \
                                     rocsparse_int             mb,                    \
                                     rocsparse_int             nnzb,                  \
                                     const rocsparse_mat_descr descr,                 \
                                     const T*                  bsr_val,               \
                                     const rocsparse_int*      bsr_row_ptr,           \
                                     const rocsparse_int*      bsr_col_ind,           \
                                     rocsparse_int             block_dim,             \
                                     rocsparse_mat_info        info,                  \
                                     rocsparse_analysis_policy analysis,              \
                                     rocsparse_solve_policy    solve,                 \
                                     void*                     temp_buffer)           \
    try                                                                               \
    {                                                                                 \
        ROCSPARSE_ROUTINE_TRACE;                                                      \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrilu0_analysis_template(handle,        \
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
        return rocsparse_status_success;                                              \
    }                                                                                 \
    catch(...)                                                                        \
    {                                                                                 \
        RETURN_ROCSPARSE_EXCEPTION();                                                 \
    }

CIMPL(rocsparse_sbsrilu0_analysis, float);
CIMPL(rocsparse_dbsrilu0_analysis, double);
CIMPL(rocsparse_cbsrilu0_analysis, rocsparse_float_complex);
CIMPL(rocsparse_zbsrilu0_analysis, rocsparse_double_complex);
#undef CIMPL

extern "C" rocsparse_status rocsparse_sbsrilu0(rocsparse_handle          handle,
                                               rocsparse_direction       dir,
                                               rocsparse_int             mb,
                                               rocsparse_int             nnzb,
                                               const rocsparse_mat_descr descr,
                                               float*                    bsr_val,
                                               const rocsparse_int*      bsr_row_ptr,
                                               const rocsparse_int*      bsr_col_ind,
                                               rocsparse_int             block_dim,
                                               rocsparse_mat_info        info,
                                               rocsparse_solve_policy    policy,
                                               void*                     temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    RETURN_IF_ROCSPARSE_ERROR((rocsparse::bsrilu0_template(handle,
                                                           dir,
                                                           mb,
                                                           nnzb,
                                                           descr,
                                                           bsr_val,
                                                           bsr_row_ptr,
                                                           bsr_col_ind,
                                                           block_dim,
                                                           info,
                                                           policy,
                                                           temp_buffer)));

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_dbsrilu0(rocsparse_handle          handle,
                                               rocsparse_direction       dir,
                                               rocsparse_int             mb,
                                               rocsparse_int             nnzb,
                                               const rocsparse_mat_descr descr,
                                               double*                   bsr_val,
                                               const rocsparse_int*      bsr_row_ptr,
                                               const rocsparse_int*      bsr_col_ind,
                                               rocsparse_int             block_dim,
                                               rocsparse_mat_info        info,
                                               rocsparse_solve_policy    policy,
                                               void*                     temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    RETURN_IF_ROCSPARSE_ERROR((rocsparse::bsrilu0_template(handle,
                                                           dir,
                                                           mb,
                                                           nnzb,
                                                           descr,
                                                           bsr_val,
                                                           bsr_row_ptr,
                                                           bsr_col_ind,
                                                           block_dim,
                                                           info,
                                                           policy,
                                                           temp_buffer)));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_cbsrilu0(rocsparse_handle          handle,
                                               rocsparse_direction       dir,
                                               rocsparse_int             mb,
                                               rocsparse_int             nnzb,
                                               const rocsparse_mat_descr descr,
                                               rocsparse_float_complex*  bsr_val,
                                               const rocsparse_int*      bsr_row_ptr,
                                               const rocsparse_int*      bsr_col_ind,
                                               rocsparse_int             block_dim,
                                               rocsparse_mat_info        info,
                                               rocsparse_solve_policy    policy,
                                               void*                     temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    RETURN_IF_ROCSPARSE_ERROR((rocsparse::bsrilu0_template(handle,
                                                           dir,
                                                           mb,
                                                           nnzb,
                                                           descr,
                                                           bsr_val,
                                                           bsr_row_ptr,
                                                           bsr_col_ind,
                                                           block_dim,
                                                           info,
                                                           policy,
                                                           temp_buffer)));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_zbsrilu0(rocsparse_handle          handle,
                                               rocsparse_direction       dir,
                                               rocsparse_int             mb,
                                               rocsparse_int             nnzb,
                                               const rocsparse_mat_descr descr,
                                               rocsparse_double_complex* bsr_val,
                                               const rocsparse_int*      bsr_row_ptr,
                                               const rocsparse_int*      bsr_col_ind,
                                               rocsparse_int             block_dim,
                                               rocsparse_mat_info        info,
                                               rocsparse_solve_policy    policy,
                                               void*                     temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    RETURN_IF_ROCSPARSE_ERROR((rocsparse::bsrilu0_template(handle,
                                                           dir,
                                                           mb,
                                                           nnzb,
                                                           descr,
                                                           bsr_val,
                                                           bsr_row_ptr,
                                                           bsr_col_ind,
                                                           block_dim,
                                                           info,
                                                           policy,
                                                           temp_buffer)));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_bsrilu0_zero_pivot(rocsparse_handle   handle,
                                                         rocsparse_mat_info info,
                                                         rocsparse_int*     position)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    // Logging
    rocsparse::log_trace(
        handle, "rocsparse_bsrilu0_zero_pivot", (const void*&)info, (const void*&)position);

    ROCSPARSE_CHECKARG_POINTER(1, info);
    ROCSPARSE_CHECKARG_POINTER(2, position);

    auto bsrilu0_info = info->get_bsrilu0_info();
    {
        auto status = bsrilu0_info->copy_zero_pivot_async(handle->pointer_mode,
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
