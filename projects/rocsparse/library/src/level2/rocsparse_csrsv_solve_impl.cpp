/*! \file */
/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/level2/rocsparse_csrsv.h"
#include "rocsparse_csrsv.hpp"

#include "../level1/rocsparse_gthr.hpp"
#include "csrsv_device.h"
#include "rocsparse_assign_async.hpp"
#include "rocsparse_common.h"
#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, bool SLEEP, typename I, typename J, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrsv_kernel(J m,
                      ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                      const I* __restrict__ csr_row_ptr,
                      const J* __restrict__ csr_col_ind,
                      const T* __restrict__ csr_val,
                      const T* __restrict__ x,
                      int64_t x_inc,
                      T* __restrict__ y,
                      int* __restrict__ done_array,
                      const J* __restrict__ map,
                      int offset,
                      J* __restrict__ zero_pivot,
                      rocsparse_index_base idx_base,
                      rocsparse_fill_mode  fill_mode,
                      rocsparse_diag_type  diag_type,
                      bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        rocsparse::csrsv_device<BLOCKSIZE, WF_SIZE, SLEEP>(m,
                                                           alpha,
                                                           csr_row_ptr,
                                                           csr_col_ind,
                                                           csr_val,
                                                           x,
                                                           x_inc,
                                                           y,
                                                           done_array,
                                                           map,
                                                           offset,
                                                           zero_pivot,
                                                           idx_base,
                                                           fill_mode,
                                                           diag_type);
    }

    template <typename I, typename J, typename T>
    rocsparse_status csrsv_solve_dispatch(rocsparse_handle          handle,
                                          rocsparse_operation       trans,
                                          J                         m,
                                          I                         nnz,
                                          const T*                  alpha_device_host,
                                          const rocsparse_mat_descr descr,
                                          const T*                  csr_val,
                                          const I*                  csr_row_ptr,
                                          const J*                  csr_col_ind,
                                          rocsparse_mat_info        info,
                                          const T*                  x,
                                          int64_t                   x_inc,
                                          T*                        y,
                                          rocsparse_solve_policy    policy,
                                          rocsparse_csrsv_info      csrsv_info,
                                          void*                     temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Stream
        hipStream_t stream = handle->stream;

        // Buffer
        char* ptr = reinterpret_cast<char*>(temp_buffer);

        ptr += 256;

        // done array
        int* done_array = reinterpret_cast<int*>(ptr);
        ptr += ((sizeof(int) * m - 1) / 256 + 1) * 256;

        // Initialize buffers
        RETURN_IF_HIP_ERROR(hipMemsetAsync(done_array, 0, sizeof(int) * m, stream));

        const rocsparse::trm_info_t* csrsv = csrsv_info->get(trans, descr->fill_mode);
        if(csrsv == nullptr)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_pointer);
        }

        // If diag type is unit, re-initialize zero pivot to remove structural zeros
        if(descr->diag_type == rocsparse_diag_type_unit)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::assign_async(reinterpret_cast<J*>(csrsv_info->get_zero_pivot()),
                                        std::numeric_limits<J>::max(),
                                        stream));
        }

        // Pointers to differentiate between transpose mode
        const I* local_csr_row_ptr = csr_row_ptr;
        const J* local_csr_col_ind = csr_col_ind;
        const T* local_csr_val     = csr_val;

        rocsparse_fill_mode fill_mode = descr->fill_mode;

        // When computing transposed triangular solve, we first need to update the
        // transposed matrix values
        if(trans == rocsparse_operation_transpose
           || trans == rocsparse_operation_conjugate_transpose)
        {
            T* csrt_val = reinterpret_cast<T*>(ptr);

            // Gather values
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::gthr_template<I, T>(handle,
                                                nnz,
                                                csr_val,
                                                csrt_val,
                                                (const I*)csrsv->get_transposed_perm(),
                                                rocsparse_index_base_zero)));

            if(trans == rocsparse_operation_conjugate_transpose)
            {
                // conjugate csrt_val
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::conjugate(handle, nnz, csrt_val));
            }

            local_csr_row_ptr = (const I*)csrsv->get_transposed_row_ptr();
            local_csr_col_ind = (const J*)csrsv->get_transposed_col_ind();
            local_csr_val     = (const T*)csrt_val;

            fill_mode = (fill_mode == rocsparse_fill_mode_lower) ? rocsparse_fill_mode_upper
                                                                 : rocsparse_fill_mode_lower;
        }

        // Determine gcn_arch
        const std::string gcn_arch_name = rocsparse::handle_get_arch_name(handle);
        const int         asicRev       = handle->asic_rev;

#define CSRSV_DIM 1024
        dim3 csrsv_blocks(((int64_t)handle->wavefront_size * m - 1) / CSRSV_DIM + 1);
        dim3 csrsv_threads(CSRSV_DIM);

        // gfx908
        if(gcn_arch_name == rocpsarse_arch_names::gfx908 && asicRev < 2)
        {
            // LCOV_EXCL_START
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::csrsv_kernel<CSRSV_DIM, 64, true>),
                csrsv_blocks,
                csrsv_threads,
                0,
                stream,
                m,
                ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
                local_csr_row_ptr,
                local_csr_col_ind,
                local_csr_val,
                x,
                x_inc,
                y,
                done_array,
                (const J*)csrsv->get_row_map(),
                0,
                (J*)csrsv_info->get_zero_pivot(),
                descr->base,
                fill_mode,
                descr->diag_type,
                handle->pointer_mode == rocsparse_pointer_mode_host);
            // LCOV_EXCL_STOP
        }
        else
        {
            // rocsparse_pointer_mode_device
            if(handle->wavefront_size == 32)
            {
                // LCOV_EXCL_START
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::csrsv_kernel<CSRSV_DIM, 32, false>),
                    csrsv_blocks,
                    csrsv_threads,
                    0,
                    stream,
                    m,
                    ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
                    local_csr_row_ptr,
                    local_csr_col_ind,
                    local_csr_val,
                    x,
                    x_inc,
                    y,
                    done_array,
                    (const J*)csrsv->get_row_map(),
                    0,
                    (J*)csrsv_info->get_zero_pivot(),
                    descr->base,
                    fill_mode,
                    descr->diag_type,
                    handle->pointer_mode == rocsparse_pointer_mode_host);
                // LCOV_EXCL_STOP
            }
            else
            {
                rocsparse_host_assert(handle->wavefront_size == 64,
                                      "Wrong wavefront size dispatch.");
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::csrsv_kernel<CSRSV_DIM, 64, false>),
                    csrsv_blocks,
                    csrsv_threads,
                    0,
                    stream,
                    m,
                    ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
                    local_csr_row_ptr,
                    local_csr_col_ind,
                    local_csr_val,
                    x,
                    x_inc,
                    y,
                    done_array,
                    (const J*)csrsv->get_row_map(),
                    0,
                    (J*)csrsv_info->get_zero_pivot(),
                    descr->base,
                    fill_mode,
                    descr->diag_type,
                    handle->pointer_mode == rocsparse_pointer_mode_host);
            }
        }
#undef CSRSV_DIM

        return rocsparse_status_success;
    }
}

template <typename I, typename J, typename T>
rocsparse_status rocsparse::csrsv_solve_template(rocsparse_handle          handle, //0
                                                 rocsparse_operation       trans, //1
                                                 int64_t                   m_, //2
                                                 int64_t                   nnz_, //3
                                                 const void*               alpha_device_host_, //4
                                                 const rocsparse_mat_descr descr, //5
                                                 const void*               csr_val_, //6
                                                 const void*               csr_row_ptr_, //7
                                                 const void*               csr_col_ind_, //8
                                                 rocsparse_mat_info        info, //9
                                                 const void*               x_, //10
                                                 int64_t                   x_inc, // non-classified
                                                 void*                     y_, //11
                                                 rocsparse_solve_policy    policy, //12
                                                 rocsparse_csrsv_info      csrsv_info, // ignore
                                                 void*                     temp_buffer) //13
{
    ROCSPARSE_ROUTINE_TRACE;

    const J  m                 = static_cast<J>(m_);
    const I  nnz               = static_cast<I>(nnz_);
    const T* alpha_device_host = reinterpret_cast<const T*>(alpha_device_host_);
    const T* csr_val           = reinterpret_cast<const T*>(csr_val_);
    const I* csr_row_ptr       = reinterpret_cast<const I*>(csr_row_ptr_);
    const J* csr_col_ind       = reinterpret_cast<const J*>(csr_col_ind_);
    const T* x                 = reinterpret_cast<const T*>(x_);
    T*       y                 = reinterpret_cast<T*>(y_);

    // Check for valid handle and matrix descriptor
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(5, descr);
    ROCSPARSE_CHECKARG_POINTER(9, info);

    // Logging
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xcsrsv"),
                         trans,
                         m,
                         nnz,
                         LOG_TRACE_SCALAR_VALUE(handle, alpha_device_host),
                         descr,
                         csr_val,
                         csr_row_ptr,
                         csr_col_ind,
                         info,
                         x,
                         y,
                         policy,
                         csrsv_info,
                         temp_buffer);

    ROCSPARSE_CHECKARG_ENUM(1, trans);
    ROCSPARSE_CHECKARG_ENUM(12, policy);

    // Check matrix type
    ROCSPARSE_CHECKARG(5,
                       descr,
                       (descr->type != rocsparse_matrix_type_general
                        && descr->type != rocsparse_matrix_type_triangular),
                       rocsparse_status_not_implemented);

    // Check matrix sorting mode

    ROCSPARSE_CHECKARG(5,
                       descr,
                       (descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    // Check sizes
    ROCSPARSE_CHECKARG_SIZE(2, m);
    ROCSPARSE_CHECKARG_SIZE(3, nnz);

    // Quick return if possible
    if(m == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    ROCSPARSE_CHECKARG_ARRAY(10, m, x);
    ROCSPARSE_CHECKARG_ARRAY(11, m, y);

    ROCSPARSE_CHECKARG_ARRAY(6, nnz, csr_val);
    ROCSPARSE_CHECKARG_ARRAY(7, m, csr_row_ptr);
    ROCSPARSE_CHECKARG_ARRAY(8, nnz, csr_col_ind);
    ROCSPARSE_CHECKARG_POINTER(13, temp_buffer);
    ROCSPARSE_CHECKARG_POINTER(4, alpha_device_host);

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsv_solve_dispatch(handle,
                                                              trans,
                                                              m,
                                                              nnz,
                                                              alpha_device_host,
                                                              descr,
                                                              csr_val,
                                                              csr_row_ptr,
                                                              csr_col_ind,
                                                              info,
                                                              x,
                                                              x_inc,
                                                              y,
                                                              policy,
                                                              csrsv_info,
                                                              temp_buffer));
    return rocsparse_status_success;
}

#define INSTANTIATE(I, J, T)                                            \
    template rocsparse_status rocsparse::csrsv_solve_template<I, J, T>( \
        rocsparse_handle          handle,                               \
        rocsparse_operation       trans,                                \
        int64_t                   m,                                    \
        int64_t                   nnz,                                  \
        const void*               alpha_device_host,                    \
        const rocsparse_mat_descr descr,                                \
        const void*               csr_val,                              \
        const void*               csr_row_ptr,                          \
        const void*               csr_col_ind,                          \
        rocsparse_mat_info        info,                                 \
        const void*               x,                                    \
        int64_t                   x_inc,                                \
        void*                     y,                                    \
        rocsparse_solve_policy    policy,                               \
        rocsparse_csrsv_info      csrsv_info,                           \
        void*                     temp_buffer)

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
INSTANTIATE(int32_t, int64_t, float);
INSTANTIATE(int32_t, int64_t, double);
INSTANTIATE(int32_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int64_t, rocsparse_double_complex);
#undef INSTANTIATE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

#define C_IMPL(NAME, T)                                                        \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,         \
                                     rocsparse_operation       trans,          \
                                     rocsparse_int             m,              \
                                     rocsparse_int             nnz,            \
                                     const T*                  alpha,          \
                                     const rocsparse_mat_descr descr,          \
                                     const T*                  csr_val,        \
                                     const rocsparse_int*      csr_row_ptr,    \
                                     const rocsparse_int*      csr_col_ind,    \
                                     rocsparse_mat_info        info,           \
                                     const T*                  x,              \
                                     T*                        y,              \
                                     rocsparse_solve_policy    policy,         \
                                     void*                     temp_buffer)    \
    try                                                                        \
    {                                                                          \
        ROCSPARSE_ROUTINE_TRACE;                                               \
        RETURN_IF_ROCSPARSE_ERROR(                                             \
            (rocsparse::csrsv_solve_template<rocsparse_int, rocsparse_int, T>( \
                handle,                                                        \
                trans,                                                         \
                m,                                                             \
                nnz,                                                           \
                alpha,                                                         \
                descr,                                                         \
                csr_val,                                                       \
                csr_row_ptr,                                                   \
                csr_col_ind,                                                   \
                info,                                                          \
                x,                                                             \
                static_cast<int64_t>(1),                                       \
                y,                                                             \
                policy,                                                        \
                (info != nullptr) ? info->get_csrsv_info() : nullptr,          \
                temp_buffer)));                                                \
        return rocsparse_status_success;                                       \
    }                                                                          \
    catch(...)                                                                 \
    {                                                                          \
        RETURN_ROCSPARSE_EXCEPTION();                                          \
    }

C_IMPL(rocsparse_scsrsv_solve, float);
C_IMPL(rocsparse_dcsrsv_solve, double);
C_IMPL(rocsparse_ccsrsv_solve, rocsparse_float_complex);
C_IMPL(rocsparse_zcsrsv_solve, rocsparse_double_complex);

#undef C_IMPL
