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

#include "internal/conversion/rocsparse_csr2bsr.h"
#include "rocsparse_calculate_nnz.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_csr2bsr.hpp"
#include "rocsparse_utility.hpp"

#include "csr2bsr_device.h"

namespace rocsparse
{
#define launch_csr2bsr_wavefront_per_row_multipass_kernel(blocksize, wfsize, blockdim)        \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                                       \
        (rocsparse::csr2bsr_wavefront_per_row_multipass_kernel<blocksize, wfsize, blockdim>), \
        dim3((mb - 1) / (blocksize / wfsize) + 1),                                            \
        dim3(blocksize),                                                                      \
        0,                                                                                    \
        stream,                                                                               \
        dir,                                                                                  \
        m,                                                                                    \
        n,                                                                                    \
        mb,                                                                                   \
        nb,                                                                                   \
        block_dim,                                                                            \
        csr_descr->base,                                                                      \
        csr_val,                                                                              \
        csr_row_ptr,                                                                          \
        csr_col_ind,                                                                          \
        bsr_descr->base,                                                                      \
        bsr_val,                                                                              \
        bsr_row_ptr,                                                                          \
        bsr_col_ind);

#define launch_csr2bsr_block_per_row_multipass_kernel(blocksize, blockdim)        \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                           \
        (rocsparse::csr2bsr_block_per_row_multipass_kernel<blocksize, blockdim>), \
        dim3(mb),                                                                 \
        dim3(blocksize),                                                          \
        0,                                                                        \
        stream,                                                                   \
        dir,                                                                      \
        m,                                                                        \
        n,                                                                        \
        mb,                                                                       \
        nb,                                                                       \
        block_dim,                                                                \
        csr_descr->base,                                                          \
        csr_val,                                                                  \
        csr_row_ptr,                                                              \
        csr_col_ind,                                                              \
        bsr_descr->base,                                                          \
        bsr_val,                                                                  \
        bsr_row_ptr,                                                              \
        bsr_col_ind);

    template <typename T,
              typename I,
              typename J,
              typename std::enable_if<std::is_same<T, rocsparse_double_complex>::value, int>::type
              = 0>
    static inline rocsparse_status csr2bsr_64_launcher(rocsparse_handle          handle,
                                                       rocsparse_direction       dir,
                                                       J                         m,
                                                       J                         n,
                                                       J                         mb,
                                                       J                         nb,
                                                       const rocsparse_mat_descr csr_descr,
                                                       const T*                  csr_val,
                                                       const I*                  csr_row_ptr,
                                                       const J*                  csr_col_ind,
                                                       J                         block_dim,
                                                       const rocsparse_mat_descr bsr_descr,
                                                       T*                        bsr_val,
                                                       I*                        bsr_row_ptr,
                                                       J*                        bsr_col_ind)
    {
        ROCSPARSE_ROUTINE_TRACE;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
    }

    template <
        typename T,
        typename I,
        typename J,
        typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, int32_t>::value
                                    || std::is_same<T, double>::value
                                    || std::is_same<T, rocsparse_float_complex>::value,
                                int>::type
        = 0>
    static inline rocsparse_status csr2bsr_64_launcher(rocsparse_handle          handle,
                                                       rocsparse_direction       dir,
                                                       J                         m,
                                                       J                         n,
                                                       J                         mb,
                                                       J                         nb,
                                                       const rocsparse_mat_descr csr_descr,
                                                       const T*                  csr_val,
                                                       const I*                  csr_row_ptr,
                                                       const J*                  csr_col_ind,
                                                       J                         block_dim,
                                                       const rocsparse_mat_descr bsr_descr,
                                                       T*                        bsr_val,
                                                       I*                        bsr_row_ptr,
                                                       J*                        bsr_col_ind)
    {
        ROCSPARSE_ROUTINE_TRACE;
        hipStream_t stream = handle->stream;

        launch_csr2bsr_block_per_row_multipass_kernel(256, 64);

        return rocsparse_status_success;
    }
}

template <typename T, typename I, typename J>
rocsparse_status rocsparse::csr2bsr_core(rocsparse_handle          handle,
                                         rocsparse_direction       dir,
                                         J                         m,
                                         J                         n,
                                         const rocsparse_mat_descr csr_descr,
                                         const T*                  csr_val,
                                         const I*                  csr_row_ptr,
                                         const J*                  csr_col_ind,
                                         J                         block_dim,
                                         const rocsparse_mat_descr bsr_descr,
                                         T*                        bsr_val,
                                         I*                        bsr_row_ptr,
                                         J*                        bsr_col_ind,
                                         int64_t                   nnzb)
{
    ROCSPARSE_ROUTINE_TRACE;
    const J mb = (m + block_dim - 1) / block_dim;
    const J nb = (n + block_dim - 1) / block_dim;

    //
    // TODO: should it be user responsibility ?
    //
    RETURN_IF_HIP_ERROR(
        hipMemsetAsync(bsr_val, 0, sizeof(T) * nnzb * block_dim * block_dim, handle->stream));
    // Stream
    hipStream_t stream = handle->stream;

    if(block_dim == 1)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csr2bsr_block_dim_equals_one_kernel<256>),
                                           dim3((mb - 1) / 256 + 1),
                                           dim3(256),
                                           0,
                                           stream,
                                           m,
                                           n,
                                           mb,
                                           nb,
                                           csr_descr->base,
                                           csr_val,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           bsr_descr->base,
                                           bsr_val,
                                           bsr_row_ptr,
                                           bsr_col_ind);
        return rocsparse_status_success;
    }

    if(block_dim <= 4)
    {
        launch_csr2bsr_wavefront_per_row_multipass_kernel(256, 16, 4);
    }
    else if(block_dim <= 8)
    {
        if(handle->wavefront_size == 64)
        {
            launch_csr2bsr_wavefront_per_row_multipass_kernel(256, 64, 8);
        }
        else
        {
            launch_csr2bsr_wavefront_per_row_multipass_kernel(256, 32, 8);
        }
    }
    else if(block_dim <= 16)
    {
        if(handle->wavefront_size == 64)
        {
            launch_csr2bsr_wavefront_per_row_multipass_kernel(256, 64, 16);
        }
        else
        {
            launch_csr2bsr_wavefront_per_row_multipass_kernel(256, 32, 16);
        }
    }
    else if(block_dim <= 32)
    {
        launch_csr2bsr_block_per_row_multipass_kernel(256, 32);
    }
    else if(block_dim <= 64 && !std::is_same<T, rocsparse_double_complex>())
    {
        rocsparse::csr2bsr_64_launcher(handle,
                                       dir,
                                       m,
                                       n,
                                       mb,
                                       nb,
                                       csr_descr,
                                       csr_val,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       block_dim,
                                       bsr_descr,
                                       bsr_val,
                                       bsr_row_ptr,
                                       bsr_col_ind);
    }
    else
    {
        // Use a blocksize of 32 to handle each block row
        static constexpr J block_size       = 32;
        J                  rows_per_segment = (block_dim + block_size - 1) / block_size;

        size_t buffer_size = 0;
        buffer_size += ((sizeof(I) * mb * block_size * rows_per_segment * 2 - 1) / 256 + 1) * 256;
        buffer_size += ((sizeof(J) * mb * block_size * rows_per_segment - 1) / 256 + 1) * 256;
        buffer_size
            += sizeof(T) * ((size_t(mb) * block_size * rows_per_segment - 1) / 256 + 1) * 256;

        bool  temp_alloc       = false;
        void* temp_storage_ptr = nullptr;
        if(handle->buffer_size >= buffer_size)
        {
            temp_storage_ptr = handle->buffer;
            temp_alloc       = false;
        }
        else
        {
            RETURN_IF_HIP_ERROR(
                rocsparse_hipMallocAsync(&temp_storage_ptr, buffer_size, handle->stream));
            temp_alloc = true;
        }

        char* ptr   = reinterpret_cast<char*>(temp_storage_ptr);
        I*    temp1 = reinterpret_cast<I*>(ptr);
        ptr += ((sizeof(I) * mb * block_size * 2 * rows_per_segment - 1) / 256 + 1) * 256;
        J* temp2 = reinterpret_cast<J*>(ptr);
        ptr += ((sizeof(J) * mb * block_size * rows_per_segment - 1) / 256 + 1) * 256;
        T* temp3 = reinterpret_cast<T*>(ptr);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csr2bsr_65_inf_kernel<block_size>),
                                           dim3(mb),
                                           dim3(block_size),
                                           0,
                                           handle->stream,
                                           dir,
                                           m,
                                           n,
                                           mb,
                                           nb,
                                           block_dim,
                                           rows_per_segment,
                                           csr_descr->base,
                                           csr_val,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           bsr_descr->base,
                                           bsr_val,
                                           bsr_row_ptr,
                                           bsr_col_ind,
                                           temp1,
                                           temp2,
                                           temp3);

        if(temp_alloc)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(temp_storage_ptr, handle->stream));
        }
    }

    return rocsparse_status_success;
}

template <typename I>
rocsparse_status rocsparse::csr2bsr_quickreturn(rocsparse_handle          handle,
                                                rocsparse_direction       dir,
                                                int64_t                   m,
                                                int64_t                   n,
                                                const rocsparse_mat_descr csr_descr,
                                                const void*               csr_val,
                                                const void*               csr_row_ptr,
                                                const void*               csr_col_ind,
                                                int64_t                   block_dim,
                                                const rocsparse_mat_descr bsr_descr,
                                                void*                     bsr_val,
                                                I*                        bsr_row_ptr,
                                                void*                     bsr_col_ind,
                                                int64_t*                  nnzb)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Quick return if possible
    if(m == 0 || n == 0)
    {
        return rocsparse_status_success;
    }

    const int64_t mb    = (m + block_dim - 1) / block_dim;
    I             start = 0;
    I             end   = 0;
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(&end, &bsr_row_ptr[mb], sizeof(I), hipMemcpyDeviceToHost, handle->stream));
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(&start, &bsr_row_ptr[0], sizeof(I), hipMemcpyDeviceToHost, handle->stream));
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

    nnzb[0] = int64_t(end) - start;
    if(nnzb[0] == 0)
    {
        return rocsparse_status_success;
    }

    return rocsparse_status_continue;
}

namespace rocsparse
{
    template <typename I>
    static rocsparse_status csr2bsr_checkarg(rocsparse_handle          handle, //0
                                             rocsparse_direction       dir, //1
                                             int64_t                   m, //2
                                             int64_t                   n, //3
                                             const rocsparse_mat_descr csr_descr, //4
                                             const void*               csr_val, //5
                                             const I*                  csr_row_ptr, //6
                                             const void*               csr_col_ind, //7
                                             int64_t                   block_dim, //8
                                             const rocsparse_mat_descr bsr_descr, //9
                                             void*                     bsr_val, //10
                                             I*                        bsr_row_ptr, //11
                                             void*                     bsr_col_ind, //12
                                             int64_t*                  nnzb)
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_ENUM(1, dir);
        ROCSPARSE_CHECKARG_SIZE(2, m);
        ROCSPARSE_CHECKARG_SIZE(3, n);
        ROCSPARSE_CHECKARG_POINTER(4, csr_descr);
        ROCSPARSE_CHECKARG(4,
                           csr_descr,
                           (csr_descr->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);
        ROCSPARSE_CHECKARG_SIZE(8, block_dim);
        ROCSPARSE_CHECKARG(8, block_dim, (block_dim == 0), rocsparse_status_invalid_size);
        ROCSPARSE_CHECKARG_POINTER(9, bsr_descr);
        ROCSPARSE_CHECKARG(9,
                           bsr_descr,
                           (bsr_descr->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);

        ROCSPARSE_CHECKARG_ARRAY(11, m, bsr_row_ptr);

        const rocsparse_status status = rocsparse::csr2bsr_quickreturn(handle,
                                                                       dir,
                                                                       m,
                                                                       n,
                                                                       csr_descr,
                                                                       csr_val,
                                                                       csr_row_ptr,
                                                                       csr_col_ind,
                                                                       block_dim,
                                                                       bsr_descr,
                                                                       bsr_val,
                                                                       bsr_row_ptr,
                                                                       bsr_col_ind,
                                                                       nnzb);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        // if nnzb = 0 then it must return success from the quick return.
        // Therefore, these must be not null at this point.
        ROCSPARSE_CHECKARG_POINTER(10, bsr_val);
        ROCSPARSE_CHECKARG_POINTER(12, bsr_col_ind);

        ROCSPARSE_CHECKARG_ARRAY(6, m, csr_row_ptr);
        if(csr_val == nullptr || csr_col_ind == nullptr)
        {
            int64_t nnz;
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::calculate_nnz(
                m, rocsparse::get_indextype<I>(), csr_row_ptr, &nnz, handle->stream));

            ROCSPARSE_CHECKARG_ARRAY(5, nnz, csr_val);
            ROCSPARSE_CHECKARG_ARRAY(7, nnz, csr_col_ind);
        }

        return rocsparse_status_continue;
    }

    template <typename... P>
    static rocsparse_status csr2bsr_impl(P&&... p)
    {
        ROCSPARSE_ROUTINE_TRACE;

        rocsparse::log_trace("rocsparse_Xcsr2bsr", p...);

        int64_t                nnzb;
        const rocsparse_status status = rocsparse::csr2bsr_checkarg(p..., &nnzb);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csr2bsr_core(p..., nnzb));
        return rocsparse_status_success;
    }
}

#define INSTANTIATE(T, I, J)                                                                 \
    template rocsparse_status rocsparse::csr2bsr_core(rocsparse_handle          handle,      \
                                                      rocsparse_direction       dir,         \
                                                      J                         m,           \
                                                      J                         n,           \
                                                      const rocsparse_mat_descr csr_descr,   \
                                                      const T*                  csr_val,     \
                                                      const I*                  csr_row_ptr, \
                                                      const J*                  csr_col_ind, \
                                                      J                         block_dim,   \
                                                      const rocsparse_mat_descr bsr_descr,   \
                                                      T*                        bsr_val,     \
                                                      I*                        bsr_row_ptr, \
                                                      J*                        bsr_col_ind, \
                                                      int64_t                   nnzb)

INSTANTIATE(int32_t, int32_t, int32_t);
INSTANTIATE(int32_t, int64_t, int32_t);
INSTANTIATE(int32_t, int32_t, int64_t);
INSTANTIATE(int32_t, int64_t, int64_t);

INSTANTIATE(float, int32_t, int32_t);
INSTANTIATE(float, int64_t, int32_t);
INSTANTIATE(float, int32_t, int64_t);
INSTANTIATE(float, int64_t, int64_t);

INSTANTIATE(rocsparse_float_complex, int32_t, int32_t);
INSTANTIATE(rocsparse_float_complex, int64_t, int32_t);
INSTANTIATE(rocsparse_float_complex, int32_t, int64_t);
INSTANTIATE(rocsparse_float_complex, int64_t, int64_t);

INSTANTIATE(double, int32_t, int32_t);
INSTANTIATE(double, int64_t, int32_t);
INSTANTIATE(double, int32_t, int64_t);
INSTANTIATE(double, int64_t, int64_t);

INSTANTIATE(rocsparse_double_complex, int32_t, int32_t);
INSTANTIATE(rocsparse_double_complex, int64_t, int32_t);
INSTANTIATE(rocsparse_double_complex, int32_t, int64_t);
INSTANTIATE(rocsparse_double_complex, int64_t, int64_t);

#undef INSTANTIATE

#define INSTANTIATE(I)                                                                              \
    template rocsparse_status rocsparse::csr2bsr_quickreturn(rocsparse_handle          handle,      \
                                                             rocsparse_direction       dir,         \
                                                             int64_t                   m,           \
                                                             int64_t                   n,           \
                                                             const rocsparse_mat_descr csr_descr,   \
                                                             const void*               csr_val,     \
                                                             const void*               csr_row_ptr, \
                                                             const void*               csr_col_ind, \
                                                             int64_t                   block_dim,   \
                                                             const rocsparse_mat_descr bsr_descr,   \
                                                             void*                     bsr_val,     \
                                                             I*                        bsr_row_ptr, \
                                                             void*                     bsr_col_ind, \
                                                             int64_t*                  nnzb)
INSTANTIATE(int32_t);
INSTANTIATE(int64_t);
#undef INSTANTIATE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

#define CIMPL(NAME, T)                                                      \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,      \
                                     rocsparse_direction       dir,         \
                                     rocsparse_int             m,           \
                                     rocsparse_int             n,           \
                                     const rocsparse_mat_descr csr_descr,   \
                                     const T*                  csr_val,     \
                                     const rocsparse_int*      csr_row_ptr, \
                                     const rocsparse_int*      csr_col_ind, \
                                     rocsparse_int             block_dim,   \
                                     const rocsparse_mat_descr bsr_descr,   \
                                     T*                        bsr_val,     \
                                     rocsparse_int*            bsr_row_ptr, \
                                     rocsparse_int*            bsr_col_ind) \
    try                                                                     \
    {                                                                       \
        ROCSPARSE_ROUTINE_TRACE;                                            \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csr2bsr_impl(handle,           \
                                                          dir,              \
                                                          m,                \
                                                          n,                \
                                                          csr_descr,        \
                                                          csr_val,          \
                                                          csr_row_ptr,      \
                                                          csr_col_ind,      \
                                                          block_dim,        \
                                                          bsr_descr,        \
                                                          bsr_val,          \
                                                          bsr_row_ptr,      \
                                                          bsr_col_ind));    \
        return rocsparse_status_success;                                    \
    }                                                                       \
    catch(...)                                                              \
    {                                                                       \
        RETURN_ROCSPARSE_EXCEPTION();                                       \
    }

CIMPL(rocsparse_scsr2bsr, float);
CIMPL(rocsparse_dcsr2bsr, double);
CIMPL(rocsparse_ccsr2bsr, rocsparse_float_complex);
CIMPL(rocsparse_zcsr2bsr, rocsparse_double_complex);

#undef CIMPL
