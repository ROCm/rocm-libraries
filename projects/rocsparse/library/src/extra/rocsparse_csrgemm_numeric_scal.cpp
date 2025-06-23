/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "../conversion/rocsparse_identity.hpp"
#include "internal/extra/rocsparse_csrgemm.h"
#include "rocsparse_csrgemm.hpp"

#include "rocsparse_common.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_csrgemm_numeric_scal.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    // Copy an array
    template <uint32_t BLOCKSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgemm_numeric_copy(I size,
                              const J* __restrict__ in,
                              J* __restrict__ out,
                              rocsparse_index_base idx_base_in,
                              rocsparse_index_base idx_base_out)
    {
        I idx = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

        if(idx >= size)
        {
            return;
        }

        out[idx] = in[idx] - idx_base_in + idx_base_out;
    }

    // Copy and scale an array
    template <uint32_t BLOCKSIZE, typename I, typename T>
    ROCSPARSE_DEVICE_ILF void
        csrgemm_numeric_copy_scale_device(I size, T alpha, const T* in, T* out)
    {
        I idx = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

        if(idx >= size)
        {
            return;
        }

        out[idx] = alpha * in[idx];
    }

    template <uint32_t BLOCKSIZE, typename I, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgemm_numeric_copy_scale_kernel(I size,
                                           ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                           const T* __restrict__ in,
                                           T* __restrict__ out,
                                           bool is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        rocsparse::csrgemm_numeric_copy_scale_device<BLOCKSIZE>(size, alpha, in, out);
    }
}

rocsparse_status rocsparse::csrgemm_numeric_scal_quickreturn(rocsparse_handle handle,
                                                             int64_t          m,
                                                             int64_t          n,
                                                             const void*      beta_device_host,
                                                             const rocsparse_mat_descr descr_D,
                                                             int64_t                   nnz_D,
                                                             const void*               csr_val_D,
                                                             const void* csr_row_ptr_D,
                                                             const void* csr_col_ind_D,
                                                             const rocsparse_mat_descr descr_C,
                                                             int64_t                   nnz_C,
                                                             void*                     csr_val_C,
                                                             const void*              csr_row_ptr_C,
                                                             const void*              csr_col_ind_C,
                                                             const rocsparse_mat_info info_C,
                                                             void*                    temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(m == 0 || n == 0 || nnz_D == 0 || nnz_C == 0)
    {
        return rocsparse_status_success;
    }
    return rocsparse_status_continue;
}

template <typename I, typename J, typename T>
inline rocsparse_status rocsparse::csrgemm_numeric_scal_core(rocsparse_handle handle,
                                                             J                m,
                                                             J                n,
                                                             const T*         beta_device_host,
                                                             const rocsparse_mat_descr descr_D,
                                                             I                         nnz_D,
                                                             const T*                  csr_val_D,
                                                             const I* csr_row_ptr_D,
                                                             const J* csr_col_ind_D,
                                                             const rocsparse_mat_descr descr_C,
                                                             I                         nnz_C,
                                                             T*                        csr_val_C,
                                                             const I*                 csr_row_ptr_C,
                                                             const J*                 csr_col_ind_C,
                                                             const rocsparse_mat_info info_C,
                                                             void*                    temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    const bool mul = info_C->csrgemm_info->mul;
    const bool add = info_C->csrgemm_info->add;
    if(mul == false && add == true)
    {

        if(descr_C->type != rocsparse_matrix_type_general)
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                rocsparse_status_not_implemented,
                " failed on condition (descr_C->type != rocsparse_matrix_type_general)");
        }

        if(descr_D->type != rocsparse_matrix_type_general)
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                rocsparse_status_not_implemented,
                " failed on condition (descr_D->type != rocsparse_matrix_type_general)");
        }

        // Stream
#define CSRGEMM_DIM 1024
        dim3 csrgemm_numeric_blocks((nnz_D - 1) / CSRGEMM_DIM + 1);
        dim3 csrgemm_numeric_threads(CSRGEMM_DIM);
        // Scale the matrix
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrgemm_numeric_copy_scale_kernel<CSRGEMM_DIM>),
            csrgemm_numeric_blocks,
            csrgemm_numeric_threads,
            0,
            handle->stream,
            nnz_D,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
            csr_val_D,
            csr_val_C,
            handle->pointer_mode == rocsparse_pointer_mode_host);
#undef CSRGEMM_DIM
        return rocsparse_status_success;
    }
    else
    {
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error,
                                               "failed condition (mul == false && add == true)");
    }
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                                 \
    template rocsparse_status rocsparse::csrgemm_numeric_scal_core<ITYPE, JTYPE, TTYPE>( \
        rocsparse_handle          handle,                                                \
        JTYPE                     m,                                                     \
        JTYPE                     n,                                                     \
        const TTYPE*              beta,                                                  \
        const rocsparse_mat_descr descr_D,                                               \
        ITYPE                     nnz_D,                                                 \
        const TTYPE*              csr_val_D,                                             \
        const ITYPE*              csr_row_ptr_D,                                         \
        const JTYPE*              csr_col_ind_D,                                         \
        const rocsparse_mat_descr descr_C,                                               \
        ITYPE                     nnz_C,                                                 \
        TTYPE*                    csr_val_C,                                             \
        const ITYPE*              csr_row_ptr_C,                                         \
        const JTYPE*              csr_col_ind_C,                                         \
        const rocsparse_mat_info  info_C,                                                \
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
#undef INSTANTIATE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */
