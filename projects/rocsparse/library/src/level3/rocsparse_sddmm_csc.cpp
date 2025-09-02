/* ************************************************************************
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "../conversion/rocsparse_csx2dense_impl.hpp"
#include "rocsparse_sddmm_csx_kernel.hpp"

template <typename T, typename I, typename J, typename A, typename B, typename C>
struct rocsparse::rocsparse_sddmm_st<rocsparse_format_csc, T, I, J, A, B, C>
{
    static rocsparse_status buffer_size(rocsparse_handle     handle,
                                        rocsparse_operation  trans_A,
                                        rocsparse_operation  trans_B,
                                        rocsparse_order      order_A,
                                        rocsparse_order      order_B,
                                        J                    m,
                                        J                    n,
                                        J                    k,
                                        I                    nnz,
                                        const T*             alpha,
                                        const A*             A_val,
                                        int64_t              A_ld,
                                        const B*             B_val,
                                        int64_t              B_ld,
                                        const T*             beta,
                                        const I*             C_ptr_data,
                                        const J*             C_ind_data,
                                        C*                   C_val_data,
                                        rocsparse_index_base C_base,
                                        rocsparse_mat_descr  C_descr,
                                        rocsparse_sddmm_alg  alg,
                                        size_t*              buffer_size)
    {
        ROCSPARSE_ROUTINE_TRACE;
        switch(alg)
        {
        case rocsparse_sddmm_alg_dense:
        {
            if(nnz == 0)
            {
                *buffer_size = 0;
                return rocsparse_status_success;
            }

            *buffer_size = ((sizeof(C) * m * n - 1) / 256 + 1) * 256;
            return rocsparse_status_success;
        }
        case rocsparse_sddmm_alg_default:
        {
            *buffer_size = 0;
            return rocsparse_status_success;
        }
            // LCOV_EXCL_START
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }

    static rocsparse_status preprocess(rocsparse_handle     handle,
                                       rocsparse_operation  trans_A,
                                       rocsparse_operation  trans_B,
                                       rocsparse_order      order_A,
                                       rocsparse_order      order_B,
                                       J                    m,
                                       J                    n,
                                       J                    k,
                                       I                    nnz,
                                       const T*             alpha,
                                       const A*             A_val,
                                       int64_t              A_ld,
                                       const B*             B_val,
                                       int64_t              B_ld,
                                       const T*             beta,
                                       const I*             C_ptr_data,
                                       const J*             C_ind_data,
                                       C*                   C_val_data,
                                       rocsparse_index_base C_base,
                                       rocsparse_mat_descr  C_descr,
                                       rocsparse_sddmm_alg  alg,
                                       void*                buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;
        switch(alg)
        {
        case rocsparse_sddmm_alg_dense:
        case rocsparse_sddmm_alg_default:
        {
            return rocsparse_status_success;
        }
            // LCOV_EXCL_START
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }

    static rocsparse_status compute(rocsparse_handle     handle,
                                    rocsparse_operation  trans_A,
                                    rocsparse_operation  trans_B,
                                    rocsparse_order      order_A,
                                    rocsparse_order      order_B,
                                    J                    m,
                                    J                    n,
                                    J                    k,
                                    I                    nnz,
                                    const T*             alpha,
                                    const A*             A_val,
                                    int64_t              A_ld,
                                    const B*             B_val,
                                    int64_t              B_ld,
                                    const T*             beta,
                                    const I*             C_ptr_data,
                                    const J*             C_ind_data,
                                    C*                   C_val_data,
                                    rocsparse_index_base C_base,
                                    rocsparse_mat_descr  C_descr,
                                    rocsparse_sddmm_alg  alg,
                                    void*                buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;
        switch(alg)
        {
        case rocsparse_sddmm_alg_dense:
        {

            if(nnz == 0)
            {
                return rocsparse_status_success;
            }

            if(buffer == nullptr)
            {
                return rocsparse_status_invalid_pointer;
            }

            char* ptr   = reinterpret_cast<char*>(buffer);
            C*    dense = reinterpret_cast<C*>(ptr);

            // Convert to Dense
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::csx2dense_impl<rocsparse_direction_column, I, J, C>(
                    handle,
                    m,
                    n,
                    C_descr,
                    C_val_data,
                    C_ptr_data,
                    C_ind_data,
                    dense,
                    m,
                    rocsparse_order_column)));

            const bool A_col_major = (order_A == rocsparse_order_column);
            const bool B_col_major = (order_B == rocsparse_order_column);

            const rocsparse_operation trans_A_adjusted
                = (A_col_major != (trans_A == rocsparse_operation_none))
                      ? rocsparse_operation_transpose
                      : rocsparse_operation_none;
            const rocsparse_operation trans_B_adjusted
                = (B_col_major != (trans_B == rocsparse_operation_none))
                      ? rocsparse_operation_transpose
                      : rocsparse_operation_none;

            // Compute
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::blas_gemm_ex(handle->blas_handle,
                                                              trans_A_adjusted,
                                                              trans_B_adjusted,
                                                              m,
                                                              n,
                                                              k,
                                                              alpha,
                                                              A_val,
                                                              rocsparse::get_datatype<A>(),
                                                              A_ld,
                                                              B_val,
                                                              rocsparse::get_datatype<B>(),
                                                              B_ld,
                                                              beta,
                                                              dense,
                                                              rocsparse::get_datatype<C>(),
                                                              m,
                                                              dense,
                                                              rocsparse::get_datatype<C>(),
                                                              m,
                                                              rocsparse::get_datatype<T>(),
                                                              rocsparse::blas_gemm_alg_standard,
                                                              0,
                                                              0));

            // Sample dense C
            static constexpr int NB = 512;

#define SMPL_LAUNCH(NT_)                                                              \
    const int64_t num_blocks_x = (n - 1) / (NB / NT_) + 1;                            \
    const dim3    blocks(num_blocks_x);                                               \
    const dim3    threads(NB);                                                        \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                               \
        (rocsparse::sddmm_csx_sample_kernel<NB, NT_, rocsparse_direction_column, T>), \
        blocks,                                                                       \
        threads,                                                                      \
        0,                                                                            \
        handle->stream,                                                               \
        m,                                                                            \
        n,                                                                            \
        nnz,                                                                          \
        dense,                                                                        \
        m,                                                                            \
        C_val_data,                                                                   \
        C_ptr_data,                                                                   \
        C_ind_data,                                                                   \
        C_base)

            const I avg_nnz = rocsparse::max(static_cast<I>(1), nnz / n);

            if(avg_nnz > 32 && handle->wavefront_size == 64)
            {
                SMPL_LAUNCH(64);
            }
            else if(avg_nnz > 16)
            {
                SMPL_LAUNCH(32);
            }
            else if(avg_nnz > 8)
            {
                SMPL_LAUNCH(16);
            }
            else if(avg_nnz > 4)
            {
                SMPL_LAUNCH(8);
            }
            else if(avg_nnz > 2)
            {
                SMPL_LAUNCH(4);
            }
            else if(avg_nnz > 1)
            {
                SMPL_LAUNCH(2);
            }
            else
            {
                SMPL_LAUNCH(1);
            }
#undef SMPL_LAUNCH

            return rocsparse_status_success;
        }
        case rocsparse_sddmm_alg_default:
        {
#define LAUNCH_WAVEFRONT_PER_ROWCOL(BLOCKSIZE, WFSIZE, NTHREADS_PER_DOTPRODUCT)       \
    dim3 blocks((n - 1) / (BLOCKSIZE / WFSIZE) + 1);                                  \
    dim3 threads(BLOCKSIZE);                                                          \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                               \
        (rocsparse::sddmm_csx_kernel_wavefront_per_rowcol<BLOCKSIZE,                  \
                                                          WFSIZE,                     \
                                                          NTHREADS_PER_DOTPRODUCT,    \
                                                          rocsparse_direction_column, \
                                                          T>),                        \
        blocks,                                                                       \
        threads,                                                                      \
        0,                                                                            \
        handle->stream,                                                               \
        trans_A,                                                                      \
        trans_B,                                                                      \
        order_A,                                                                      \
        order_B,                                                                      \
        m,                                                                            \
        n,                                                                            \
        k,                                                                            \
        nnz,                                                                          \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha),                             \
        A_val,                                                                        \
        A_ld,                                                                         \
        B_val,                                                                        \
        B_ld,                                                                         \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta),                              \
        C_val_data,                                                                   \
        C_ptr_data,                                                                   \
        C_ind_data,                                                                   \
        C_base,                                                                       \
        handle->pointer_mode == rocsparse_pointer_mode_host)

            if(handle->pointer_mode == rocsparse_pointer_mode_host)
            {
                if(*alpha == static_cast<T>(0) && *beta == static_cast<T>(1))
                {
                    return rocsparse_status_success;
                }
            }
            if(k > 4)
            {
                LAUNCH_WAVEFRONT_PER_ROWCOL(256, 32, 8);
            }
            else if(k > 2)
            {
                LAUNCH_WAVEFRONT_PER_ROWCOL(256, 32, 4);
            }
            else if(k > 1)
            {
                LAUNCH_WAVEFRONT_PER_ROWCOL(256, 32, 2);
            }
            else
            {
                LAUNCH_WAVEFRONT_PER_ROWCOL(256, 32, 1);
            }

            return rocsparse_status_success;
        }
            // LCOV_EXCL_START
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }
};

#define INSTANTIATE(TTYPE, ITYPE, JTYPE, ATYPE, BTYPE, CTYPE) \
    template struct rocsparse::                               \
        rocsparse_sddmm_st<rocsparse_format_csc, TTYPE, ITYPE, JTYPE, ATYPE, BTYPE, CTYPE>

// Uniform precision
INSTANTIATE(_Float16, int32_t, int32_t, _Float16, _Float16, _Float16);
INSTANTIATE(float, int32_t, int32_t, float, float, float);
INSTANTIATE(double, int32_t, int32_t, double, double, double);
INSTANTIATE(rocsparse_float_complex,
            int32_t,
            int32_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex,
            int32_t,
            int32_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);

INSTANTIATE(_Float16, int64_t, int32_t, _Float16, _Float16, _Float16);
INSTANTIATE(float, int64_t, int32_t, float, float, float);
INSTANTIATE(double, int64_t, int32_t, double, double, double);
INSTANTIATE(rocsparse_float_complex,
            int64_t,
            int32_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int32_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);

INSTANTIATE(_Float16, int64_t, int64_t, _Float16, _Float16, _Float16);
INSTANTIATE(float, int64_t, int64_t, float, float, float);
INSTANTIATE(double, int64_t, int64_t, double, double, double);
INSTANTIATE(rocsparse_float_complex,
            int64_t,
            int64_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int64_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);

// Mixed precision
INSTANTIATE(float, int32_t, int32_t, _Float16, _Float16, float);
INSTANTIATE(float, int64_t, int32_t, _Float16, _Float16, float);
INSTANTIATE(float, int64_t, int64_t, _Float16, _Float16, float);
INSTANTIATE(float, int32_t, int32_t, _Float16, _Float16, _Float16);
INSTANTIATE(float, int64_t, int32_t, _Float16, _Float16, _Float16);
INSTANTIATE(float, int64_t, int64_t, _Float16, _Float16, _Float16);

INSTANTIATE(float, int32_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE(float, int64_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE(float, int64_t, int64_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE(float, int32_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, rocsparse_bfloat16);
INSTANTIATE(float, int64_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, rocsparse_bfloat16);
INSTANTIATE(float, int64_t, int64_t, rocsparse_bfloat16, rocsparse_bfloat16, rocsparse_bfloat16);

#undef INSTANTIATE
