/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_assign_async.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_csrsv.hpp"
#include "rocsparse_scalar.hpp"
#include "rocsparse_utility.hpp"

#if defined(ROCSPARSE_WITH_SPTRSV_TRANSPOSE_FREE)

#include "csrsv_transpose_free_device.h"

namespace rocsparse
{
    static constexpr uint32_t s_transpose_free_blocksize          = 256;
    static constexpr uint32_t s_transpose_free_analysis_groupsize = 16;

    static size_t transpose_free_aligned(size_t size_in_bytes)
    {
        return (size_in_bytes == 0) ? 0 : ((size_in_bytes - 1) / 256 + 1) * 256;
    }

    template <uint32_t BLOCKSIZE, uint32_t L, typename I, typename J, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrsv_transpose_free_kernel(J m,
                                     ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                     const I* __restrict__ csr_row_ptr,
                                     const J* __restrict__ csr_col_ind,
                                     const T* __restrict__ csr_val,
                                     int64_t csr_val_inc,
                                     int64_t csr_val_stride,
                                     const I* __restrict__ csr_diag_ind,
                                     const T* __restrict__ x,
                                     int64_t x_inc,
                                     int64_t x_stride,
                                     T*      y,
                                     int64_t y_inc,
                                     int64_t y_stride,
                                     J* __restrict__ rem,
                                     T* __restrict__ sum,
                                     int32_t* __restrict__ ticket,
                                     J*                   zero_pivot,
                                     int64_t              zero_pivot_stride,
                                     rocsparse_index_base idx_base,
                                     rocsparse_fill_mode  fill_mode,
                                     rocsparse_diag_type  diag_type,
                                     bool                 conj,
                                     bool                 is_host_mode)
    {
        const uint32_t batch_index = hipBlockIdx_y;
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        rocsparse::csrsv_transpose_free_device<BLOCKSIZE, L>(
            m,
            alpha,
            csr_row_ptr,
            csr_col_ind,
            csr_val + batch_index * csr_val_stride,
            csr_val_inc,
            csr_diag_ind,
            x + batch_index * x_stride,
            x_inc,
            y + batch_index * y_stride,
            y_inc,
            rem + batch_index * static_cast<int64_t>(m),
            sum + batch_index * static_cast<int64_t>(m),
            ticket + batch_index,
            zero_pivot + batch_index * zero_pivot_stride,
            idx_base,
            fill_mode,
            diag_type,
            conj);
    }

    template <typename I, typename J>
    static rocsparse_status csrsv_transpose_free_analysis_template(rocsparse_handle    handle,
                                                                   int64_t             m,
                                                                   const void*         csr_row_ptr,
                                                                   const void*         csr_col_ind,
                                                                   void*               diag_ind,
                                                                   void*               in_degree,
                                                                   void*               max_nnz,
                                                                   void*               zero_pivot,
                                                                   rocsparse_mat_descr descr)
    {
        static constexpr uint32_t BLOCKSIZE = s_transpose_free_blocksize;
        static constexpr uint32_t L         = s_transpose_free_analysis_groupsize;
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrsv_transpose_free_analysis_kernel<BLOCKSIZE, L>),
            dim3((m * L - 1) / BLOCKSIZE + 1),
            dim3(BLOCKSIZE),
            0,
            handle->stream,
            static_cast<J>(m),
            reinterpret_cast<const I*>(csr_row_ptr),
            reinterpret_cast<const J*>(csr_col_ind),
            reinterpret_cast<I*>(diag_ind),
            reinterpret_cast<J*>(in_degree),
            reinterpret_cast<I*>(max_nnz),
            reinterpret_cast<J*>(zero_pivot),
            descr->base,
            descr->diag_type,
            descr->fill_mode);
        return rocsparse_status_success;
    }

    template <uint32_t L, typename I, typename J, typename T>
    static rocsparse_status csrsv_transpose_free_launch(rocsparse_handle            handle,
                                                        int64_t                     batch_count,
                                                        const void*                 alpha,
                                                        rocsparse_const_spmat_descr A,
                                                        const void*                 val,
                                                        int64_t                     val_stride,
                                                        const void*                 diag_ind,
                                                        rocsparse_const_dnvec_descr x,
                                                        rocsparse_dnvec_descr       y,
                                                        void*                       rem,
                                                        void*                       sum,
                                                        int32_t*                    ticket,
                                                        void*                       zero_pivot,
                                                        bool                        conj)
    {
        static constexpr uint32_t BLOCKSIZE = s_transpose_free_blocksize;
        const int64_t             m         = A->rows;
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrsv_transpose_free_kernel<BLOCKSIZE, L, I, J, T>),
            dim3((m * L - 1) / BLOCKSIZE + 1, batch_count),
            dim3(BLOCKSIZE),
            0,
            handle->stream,
            static_cast<J>(m),
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, reinterpret_cast<const T*>(alpha)),
            reinterpret_cast<const I*>(A->const_row_data),
            reinterpret_cast<const J*>(A->const_col_data),
            reinterpret_cast<const T*>(val),
            static_cast<int64_t>(1),
            val_stride,
            reinterpret_cast<const I*>(diag_ind),
            reinterpret_cast<const T*>(x->const_values),
            x->inc,
            x->batch_stride,
            reinterpret_cast<T*>(y->values),
            y->inc,
            y->batch_stride,
            reinterpret_cast<J*>(rem),
            reinterpret_cast<T*>(sum),
            ticket,
            reinterpret_cast<J*>(zero_pivot),
            static_cast<int64_t>(1),
            A->descr->base,
            A->descr->fill_mode,
            A->descr->diag_type,
            conj,
            handle->pointer_mode == rocsparse_pointer_mode_host);
        return rocsparse_status_success;
    }

    template <typename I, typename J, typename T, typename... P>
    static rocsparse_status csrsv_transpose_free_dispatch_group(uint32_t group_size, P&&... p)
    {
        switch(group_size)
        {
        case 2:
            return csrsv_transpose_free_launch<2, I, J, T>(p...);
        case 4:
            return csrsv_transpose_free_launch<4, I, J, T>(p...);
        case 8:
            return csrsv_transpose_free_launch<8, I, J, T>(p...);
        case 16:
            return csrsv_transpose_free_launch<16, I, J, T>(p...);
        case 32:
            return csrsv_transpose_free_launch<32, I, J, T>(p...);
        case 64:
            return csrsv_transpose_free_launch<64, I, J, T>(p...);
        }
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
        // LCOV_EXCL_STOP
    }

    template <typename I, typename J, typename... P>
    static rocsparse_status csrsv_transpose_free_dispatch_datatype(rocsparse_datatype datatype,
                                                                   P&&... p)
    {
        switch(datatype)
        {
        case rocsparse_datatype_f32_r:
            return csrsv_transpose_free_dispatch_group<I, J, float>(p...);
        case rocsparse_datatype_f64_r:
            return csrsv_transpose_free_dispatch_group<I, J, double>(p...);
        case rocsparse_datatype_f32_c:
            return csrsv_transpose_free_dispatch_group<I, J, rocsparse_float_complex>(p...);
        case rocsparse_datatype_f64_c:
            return csrsv_transpose_free_dispatch_group<I, J, rocsparse_double_complex>(p...);
        default:
            break;
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }
}

namespace rocsparse
{
    // Calls f(I{}, J{}) for the row offset and column index types of the matrix.
    template <typename F>
    static rocsparse_status
        csrsv_transpose_free_dispatch_indextype(rocsparse_indextype i, rocsparse_indextype j, F&& f)
    {
        if(i == rocsparse_indextype_i32 && j == rocsparse_indextype_i32)
        {
            return f(int32_t{}, int32_t{});
        }
        if(i == rocsparse_indextype_i32 && j == rocsparse_indextype_i64)
        {
            return f(int32_t{}, int64_t{});
        }
        if(i == rocsparse_indextype_i64 && j == rocsparse_indextype_i32)
        {
            return f(int64_t{}, int32_t{});
        }
        if(i == rocsparse_indextype_i64 && j == rocsparse_indextype_i64)
        {
            return f(int64_t{}, int64_t{});
        }
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }
}

size_t rocsparse::csrsv_transpose_free_analysis_buffer_size()
{
    // max_nnz only
    return 256;
}

size_t rocsparse::csrsv_transpose_free_solve_buffer_size(rocsparse_const_spmat_descr A,
                                                         int64_t                     batch_count)
{
    const size_t m        = A->rows;
    const size_t sizeof_T = rocsparse::datatype_sizeof(A->data_type);
    // Same leading layout as csrsv_solve (256 bytes + done array), then ticket, rem and sum.
    return 256 + rocsparse::transpose_free_aligned(sizeof(int32_t) * m * batch_count)
           + rocsparse::transpose_free_aligned(sizeof(int32_t) * batch_count)
           + rocsparse::transpose_free_aligned(sizeof(int64_t) * m * batch_count)
           + rocsparse::transpose_free_aligned(sizeof_T * m * batch_count);
}

rocsparse_status rocsparse::gtrm_analysis_transpose_free(rocsparse_handle          handle,
                                                         int64_t                   m,
                                                         int64_t                   nnz,
                                                         const rocsparse_mat_descr descr,
                                                         rocsparse_indextype csr_row_ptr_indextype,
                                                         const void*         csr_row_ptr,
                                                         rocsparse_indextype csr_col_ind_indextype,
                                                         const void*         csr_col_ind,
                                                         rocsparse::trm_info_t*   info,
                                                         rocsparse::pivot_info_t* pivot_info,
                                                         void*                    temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;
    const size_t sizeof_I = rocsparse::indextype_sizeof(csr_row_ptr_indextype);
    const size_t sizeof_J = rocsparse::indextype_sizeof(csr_col_ind_indextype);
    hipStream_t  stream   = handle->stream;

    info->set_m(m);
    info->set_nnz(nnz);
    info->set_descr(descr);
    info->set_row_ptr(csr_row_ptr);
    info->set_col_ind(csr_col_ind);
    info->set_offset_indextype(csr_row_ptr_indextype);
    info->set_index_indextype(csr_col_ind_indextype);

    void* d_max_nnz = temp_buffer;
    RETURN_IF_HIP_ERROR(rocsparse_hipMemsetAsync(d_max_nnz, 0, 256, stream));

    RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(info->get_ref_diag_ind(), sizeof_I * m, stream));
    RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(info->get_ref_in_degree(), sizeof_J * m, stream));
    pivot_info->create_zero_pivot_async(csr_col_ind_indextype, stream);
    RETURN_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(stream));

    void* zero_pivot = pivot_info->get_position();
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::assign_max_async(
        pivot_info->get_batch_count(), csr_col_ind_indextype, zero_pivot, stream));

    // -1 marks a missing diagonal entry
    RETURN_IF_HIP_ERROR(rocsparse_hipMemsetAsync(info->get_diag_ind(), 0xFF, sizeof_I * m, stream));
    RETURN_IF_HIP_ERROR(rocsparse_hipMemsetAsync(info->get_in_degree(), 0, sizeof_J * m, stream));

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsv_transpose_free_dispatch_indextype(
        csr_row_ptr_indextype, csr_col_ind_indextype, [&](auto i, auto j) {
            using I = decltype(i);
            using J = decltype(j);
            return rocsparse::csrsv_transpose_free_analysis_template<I, J>(handle,
                                                                           m,
                                                                           csr_row_ptr,
                                                                           csr_col_ind,
                                                                           info->get_diag_ind(),
                                                                           info->get_in_degree(),
                                                                           d_max_nnz,
                                                                           zero_pivot,
                                                                           descr);
        }));

    int64_t max_nnz = 0;
    RETURN_IF_HIP_ERROR(
        rocsparse_hipMemcpyAsync(&max_nnz, d_max_nnz, sizeof_I, hipMemcpyDeviceToHost, stream));
    RETURN_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(stream));
    info->set_max_nnz(max_nnz);

    return rocsparse_status_success;
}

rocsparse_status rocsparse::csrsv_solve_transpose_free(rocsparse_handle             handle,
                                                       rocsparse_operation          trans,
                                                       const void*                  alpha,
                                                       rocsparse_const_spmat_descr  A,
                                                       rocsparse_const_dnvec_descr  x,
                                                       rocsparse_dnvec_descr        y,
                                                       const rocsparse::trm_info_t* trm_info,
                                                       void*                        zero_pivot,
                                                       void*                        temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;
    hipStream_t   stream      = handle->stream;
    const int64_t batch_count = y->batch_count;
    const size_t  m           = A->rows;
    const size_t  sizeof_T    = rocsparse::datatype_sizeof(A->data_type);

    char* ptr = reinterpret_cast<char*>(temp_buffer);
    ptr += 256 + rocsparse::transpose_free_aligned(sizeof(int32_t) * m * batch_count);

    int32_t*     ticket = reinterpret_cast<int32_t*>(ptr);
    const size_t ticket_size_bytes
        = rocsparse::transpose_free_aligned(sizeof(int32_t) * batch_count);
    ptr += ticket_size_bytes;

    void* rem = ptr;
    ptr += rocsparse::transpose_free_aligned(sizeof(int64_t) * m * batch_count);

    void*        sum            = ptr;
    const size_t sum_size_bytes = sizeof_T * m * batch_count;

    RETURN_IF_HIP_ERROR(rocsparse_hipMemsetAsync(ticket, 0, ticket_size_bytes, stream));
    RETURN_IF_HIP_ERROR(rocsparse_hipMemsetAsync(sum, 0, sum_size_bytes, stream));

    // Smallest power of two covering the longest row, capped at the wavefront size.
    uint32_t group_size = 2;
    while(static_cast<int64_t>(group_size) < trm_info->get_max_nnz()
          && group_size < static_cast<uint32_t>(handle->wavefront_size))
    {
        group_size *= 2;
    }

    const bool    conj       = (trans == rocsparse_operation_conjugate_transpose);
    const int64_t val_stride = (A->batch_count > 1) ? A->batch_stride : 0;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsv_transpose_free_dispatch_indextype(
        A->row_type, A->col_type, [&](auto i, auto j) {
            using I                             = decltype(i);
            using J                             = decltype(j);
            static constexpr uint32_t BLOCKSIZE = rocsparse::s_transpose_free_blocksize;
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::csrsv_transpose_free_init_kernel<BLOCKSIZE, J>),
                dim3((m * batch_count - 1) / BLOCKSIZE + 1),
                dim3(BLOCKSIZE),
                0,
                stream,
                static_cast<J>(m),
                batch_count,
                reinterpret_cast<const J*>(trm_info->get_in_degree()),
                reinterpret_cast<J*>(rem));

            return rocsparse::csrsv_transpose_free_dispatch_datatype<I, J>(A->data_type,
                                                                           group_size,
                                                                           handle,
                                                                           batch_count,
                                                                           alpha,
                                                                           A,
                                                                           A->const_val_data,
                                                                           val_stride,
                                                                           trm_info->get_diag_ind(),
                                                                           x,
                                                                           y,
                                                                           rem,
                                                                           sum,
                                                                           ticket,
                                                                           zero_pivot,
                                                                           conj);
        }));

    return rocsparse_status_success;
}

#endif
