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

template <typename I, typename J, typename T>
struct rocsparse::rocsparse_sddmm_st<rocsparse_format_csr, rocsparse_sddmm_alg_default, I, J, T>
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
                                        const T*             A_val,
                                        int64_t              A_ld,
                                        const T*             B_val,
                                        int64_t              B_ld,
                                        const T*             beta,
                                        const I*             C_row_data,
                                        const J*             C_col_data,
                                        T*                   C_val_data,
                                        rocsparse_index_base C_base,
                                        rocsparse_mat_descr  C_descr,
                                        rocsparse_sddmm_alg  alg,
                                        size_t*              buffer_size)
    {
        ROCSPARSE_ROUTINE_TRACE;

        *buffer_size = 0;
        return rocsparse_status_success;
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
                                       const T*             A_val,
                                       int64_t              A_ld,
                                       const T*             B_val,
                                       int64_t              B_ld,
                                       const T*             beta,
                                       const I*             C_row_data,
                                       const J*             C_col_data,
                                       T*                   C_val_data,
                                       rocsparse_index_base C_base,
                                       rocsparse_mat_descr  C_descr,
                                       rocsparse_sddmm_alg  alg,
                                       void*                buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        return rocsparse_status_success;
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
                                    const T*             A_val,
                                    int64_t              A_ld,
                                    const T*             B_val,
                                    int64_t              B_ld,
                                    const T*             beta,
                                    const I*             C_row_data,
                                    const J*             C_col_data,
                                    T*                   C_val_data,
                                    rocsparse_index_base C_base,
                                    rocsparse_mat_descr  C_descr,
                                    rocsparse_sddmm_alg  alg,
                                    void*                buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        static constexpr int NB = 512;

#define LAUNCH(NT_)                                                               \
    int64_t num_blocks_x = (m - 1) / (NB / NT_) + 1;                              \
    dim3    blocks(num_blocks_x);                                                 \
    dim3    threads(NB);                                                          \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                           \
        (rocsparse::sddmm_csx_kernel<NB, NT_, rocsparse_direction_row, I, J, T>), \
        blocks,                                                                   \
        threads,                                                                  \
        0,                                                                        \
        handle->stream,                                                           \
        trans_A,                                                                  \
        trans_B,                                                                  \
        order_A,                                                                  \
        order_B,                                                                  \
        m,                                                                        \
        n,                                                                        \
        k,                                                                        \
        nnz,                                                                      \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha),                         \
        A_val,                                                                    \
        A_ld,                                                                     \
        B_val,                                                                    \
        B_ld,                                                                     \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta),                          \
        (T*)C_val_data,                                                           \
        (const I*)C_row_data,                                                     \
        (const J*)C_col_data,                                                     \
        C_base,                                                                   \
        (T*)buffer,                                                               \
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
            LAUNCH(8);
        }
        else if(k > 2)
        {
            LAUNCH(4);
        }
        else if(k > 1)
        {
            LAUNCH(2);
        }
        else
        {
            LAUNCH(1);
        }

        return rocsparse_status_success;
    }
};

template <typename I, typename J, typename T>
struct rocsparse::rocsparse_sddmm_st<rocsparse_format_csr, rocsparse_sddmm_alg_dense, I, J, T>
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
                                        const T*             A_val,
                                        int64_t              A_ld,
                                        const T*             B_val,
                                        int64_t              B_ld,
                                        const T*             beta,
                                        const I*             C_row_data,
                                        const J*             C_col_data,
                                        T*                   C_val_data,
                                        rocsparse_index_base C_base,
                                        rocsparse_mat_descr  C_descr,
                                        rocsparse_sddmm_alg  alg,
                                        size_t*              buffer_size)
    {
        ROCSPARSE_ROUTINE_TRACE;

        if(nnz == 0)
        {
            *buffer_size = 0;
            return rocsparse_status_success;
        }

        *buffer_size = ((sizeof(T) * m * n - 1) / 256 + 1) * 256;
        return rocsparse_status_success;
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
                                       const T*             A_val,
                                       int64_t              A_ld,
                                       const T*             B_val,
                                       int64_t              B_ld,
                                       const T*             beta,
                                       const I*             C_row_data,
                                       const J*             C_col_data,
                                       T*                   C_val_data,
                                       rocsparse_index_base C_base,
                                       rocsparse_mat_descr  C_descr,
                                       rocsparse_sddmm_alg  alg,
                                       void*                buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        return rocsparse_status_success;
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
                                    const T*             A_val,
                                    int64_t              A_ld,
                                    const T*             B_val,
                                    int64_t              B_ld,
                                    const T*             beta,
                                    const I*             C_row_data,
                                    const J*             C_col_data,
                                    T*                   C_val_data,
                                    rocsparse_index_base C_base,
                                    rocsparse_mat_descr  C_descr,
                                    rocsparse_sddmm_alg  alg,
                                    void*                buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        if(nnz == 0)
        {
            return rocsparse_status_success;
        }

        if(buffer == nullptr)
        {
            return rocsparse_status_invalid_pointer;
        }

        char* ptr   = reinterpret_cast<char*>(buffer);
        T*    dense = reinterpret_cast<T*>(ptr);

        // Convert to Dense

        // Compute C'
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::csx2dense_impl<rocsparse_direction_column, I, J, T>(
            handle,
            n,
            m,
            C_descr,
            C_val_data,
            C_row_data,
            C_col_data,
            dense,
            n,
            rocsparse_order_column)));

        const bool A_col_major = (order_A == rocsparse_order_column);
        const bool B_col_major = (order_B == rocsparse_order_column);

        // Compute (AB)' = B'A'
        const rocsparse_operation trans_A_adjusted
            = (A_col_major != (trans_A == rocsparse_operation_none))
                  ? rocsparse_operation_none
                  : rocsparse_operation_transpose;
        const rocsparse_operation trans_B_adjusted
            = (B_col_major != (trans_B == rocsparse_operation_none))
                  ? rocsparse_operation_none
                  : rocsparse_operation_transpose;

        // Compute C' = C' + B'A'
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::blas_gemm_ex(handle->blas_handle,
                                                          trans_B_adjusted,
                                                          trans_A_adjusted,
                                                          n,
                                                          m,
                                                          k,
                                                          alpha,
                                                          B_val,
                                                          rocsparse::get_datatype<T>(),
                                                          B_ld,
                                                          A_val,
                                                          rocsparse::get_datatype<T>(),
                                                          A_ld,
                                                          beta,
                                                          dense,
                                                          rocsparse::get_datatype<T>(),
                                                          n,
                                                          dense,
                                                          rocsparse::get_datatype<T>(),
                                                          n,
                                                          rocsparse::get_datatype<T>(),
                                                          rocsparse::blas_gemm_alg_standard,
                                                          0,
                                                          0));

        // Sample dense C
        static constexpr int NB = 512;

#define SMPL_LAUNCH(NT_)                                                                    \
    const int64_t num_blocks_x = (m - 1) / (NB / NT_) + 1;                                  \
    const dim3    blocks(num_blocks_x);                                                     \
    const dim3    threads(NB);                                                              \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                                     \
        (rocsparse::sddmm_csx_sample_kernel<NB, NT_, rocsparse_direction_column, I, J, T>), \
        blocks,                                                                             \
        threads,                                                                            \
        0,                                                                                  \
        handle->stream,                                                                     \
        n,                                                                                  \
        m,                                                                                  \
        nnz,                                                                                \
        dense,                                                                              \
        n,                                                                                  \
        C_val_data,                                                                         \
        C_row_data,                                                                         \
        C_col_data,                                                                         \
        C_base)

        const I avg_nnz = rocsparse::max(static_cast<I>(1), nnz / m);

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
};

#define INSTANTIATE(ITYPE_, JTYPE_, TTYPE_)                                    \
    template struct rocsparse::rocsparse_sddmm_st<rocsparse_format_csr,        \
                                                  rocsparse_sddmm_alg_default, \
                                                  ITYPE_,                      \
                                                  JTYPE_,                      \
                                                  TTYPE_>;                     \
    template struct rocsparse::rocsparse_sddmm_st<rocsparse_format_csr,        \
                                                  rocsparse_sddmm_alg_dense,   \
                                                  ITYPE_,                      \
                                                  JTYPE_,                      \
                                                  TTYPE_>

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
