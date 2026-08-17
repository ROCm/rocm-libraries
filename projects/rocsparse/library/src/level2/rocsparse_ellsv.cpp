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

#include "rocsparse_ellsv.hpp"

#include "../conversion/rocsparse_gcreate_identity_permutation.hpp"
#include "ellsv_device.h"
#include "rocsparse_common.h"
#include "rocsparse_control.hpp"
#include "rocsparse_dnvec_descr.hpp"
#include "rocsparse_ellsv_info.hpp"
#include "rocsparse_mat_info.hpp"
#include "rocsparse_primitives.hpp"
#include "rocsparse_scalar.hpp"
#include "rocsparse_spmat_descr.hpp"
#include "rocsparse_utility.hpp"

#include <vector>

namespace rocsparse
{
    // Round a byte count up to the nearest multiple of 256.
    static inline size_t ellsv_align256(size_t bytes)
    {
        return ((bytes - 1) / 256 + 1) * 256;
    }

    // Select the wavefront size / sleep mode the same way the CSR solve does.
    static void ellsv_select_launch(rocsparse_handle handle, bool* sleep, uint32_t* wfsize)
    {
        const std::string gcn_arch_name = rocsparse::handle_get_arch_name(handle);
        const int         asic_rev      = handle->asic_rev;
        *sleep  = (gcn_arch_name == rocpsarse_arch_names::gfx908 && asic_rev < 2);
        *wfsize = (*sleep) ? 64 : handle->wavefront_size;
    }

    // Solve kernel wrapper resolving the (host/device) alpha scalar.
    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, bool SLEEP, typename I, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void ellsv_solve_kernel(I       m,
                            I       n,
                            int64_t ell_width,
                            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                            const I* __restrict__ ell_col_ind,
                            const T* __restrict__ ell_val,
                            const T* __restrict__ x,
                            int64_t x_inc,
                            T*      y,
                            int64_t y_inc,
                            int* __restrict__ done_array,
                            const I* __restrict__ map,
                            rocsparse_index_base idx_base,
                            rocsparse_fill_mode  fill_mode,
                            rocsparse_diag_type  diag_type,
                            bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        rocsparse::ellsv_device<BLOCKSIZE, WF_SIZE, SLEEP, I, T>(m,
                                                                 n,
                                                                 ell_width,
                                                                 alpha,
                                                                 ell_col_ind,
                                                                 ell_val,
                                                                 x,
                                                                 x_inc,
                                                                 y,
                                                                 y_inc,
                                                                 done_array,
                                                                 map,
                                                                 idx_base,
                                                                 fill_mode,
                                                                 diag_type);
    }

    template <uint32_t WF_SIZE, bool SLEEP, typename I>
    static rocsparse_status ellsv_launch_analysis(rocsparse_handle     handle,
                                                  I                    m,
                                                  I                    n,
                                                  int64_t              ell_width,
                                                  const I*             ell_col_ind,
                                                  rocsparse_index_base base,
                                                  rocsparse_fill_mode  fill_mode,
                                                  rocsparse_indextype  index_type,
                                                  I*                   row_map,
                                                  void*                temp_buffer)
    {
        constexpr uint32_t BLOCKSIZE = 1024;

        hipStream_t  stream   = handle->stream;
        const size_t sizeof_I = sizeof(I);

        // Temporary buffer layout: done_array | workspace | workspace2 | rocprim.
        char*        ptr        = reinterpret_cast<char*>(temp_buffer);
        const size_t done_bytes = ellsv_align256(sizeof(int32_t) * static_cast<size_t>(m));
        int32_t*     done_array = reinterpret_cast<int32_t*>(ptr);
        ptr += done_bytes;

        void* workspace = ptr;
        ptr += ellsv_align256(sizeof_I * static_cast<size_t>(m));

        void* workspace2 = ptr;
        ptr += ellsv_align256(sizeof(int32_t) * static_cast<size_t>(m));

        void* rocprim_buffer = ptr;

        RETURN_IF_HIP_ERROR(rocsparse_hipMemsetAsync(done_array, 0, done_bytes, stream));

        dim3 blocks((static_cast<size_t>(m) * WF_SIZE - 1) / BLOCKSIZE + 1);
        dim3 threads(BLOCKSIZE);

        if(fill_mode == rocsparse_fill_mode_lower)
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::ellsv_analysis_lower_kernel<BLOCKSIZE, WF_SIZE, SLEEP, I>),
                blocks,
                threads,
                0,
                stream,
                m,
                n,
                ell_width,
                ell_col_ind,
                done_array,
                base);
        }
        else
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::ellsv_analysis_upper_kernel<BLOCKSIZE, WF_SIZE, SLEEP, I>),
                blocks,
                threads,
                0,
                stream,
                m,
                n,
                ell_width,
                ell_col_ind,
                done_array,
                base);
        }

        // Identity permutation of the rows.
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::gcreate_identity_permutation(handle, m, index_type, workspace));

        // Sort the rows by dependency depth to obtain the row execution order.
        const uint32_t startbit = 0;
        const uint32_t endbit   = rocsparse::clz(static_cast<int64_t>(m));

        size_t rocprim_size = 0;
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::radix_sort_pairs_buffer_size<int32_t, I>(
            handle, m, startbit, endbit, &rocprim_size)));

        rocsparse::primitives::double_buffer<int32_t> keys(done_array,
                                                           reinterpret_cast<int32_t*>(workspace2));
        rocsparse::primitives::double_buffer<I> vals(reinterpret_cast<I*>(workspace), row_map);

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::radix_sort_pairs(
            handle, keys, vals, m, startbit, endbit, rocprim_size, rocprim_buffer));
        RETURN_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(stream));

        if(vals.current() != row_map)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(row_map,
                                                         vals.current(),
                                                         sizeof_I * static_cast<size_t>(m),
                                                         hipMemcpyDeviceToDevice,
                                                         stream));
            RETURN_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(stream));
        }

        return rocsparse_status_success;
    }

    template <typename I>
    static rocsparse_status ellsv_analysis_dispatch(rocsparse_handle     handle,
                                                    bool                 sleep,
                                                    uint32_t             wfsize,
                                                    I                    m,
                                                    I                    n,
                                                    int64_t              ell_width,
                                                    const void*          ell_col_ind,
                                                    rocsparse_index_base base,
                                                    rocsparse_fill_mode  fill_mode,
                                                    rocsparse_indextype  index_type,
                                                    void*                row_map,
                                                    void*                temp_buffer)
    {
        const I* col_ind = reinterpret_cast<const I*>(ell_col_ind);
        I*       map     = reinterpret_cast<I*>(row_map);

        if(sleep)
        {
            return rocsparse::ellsv_launch_analysis<64, true, I>(
                handle, m, n, ell_width, col_ind, base, fill_mode, index_type, map, temp_buffer);
        }
        else if(wfsize == 64)
        {
            return rocsparse::ellsv_launch_analysis<64, false, I>(
                handle, m, n, ell_width, col_ind, base, fill_mode, index_type, map, temp_buffer);
        }

        return rocsparse::ellsv_launch_analysis<32, false, I>(
            handle, m, n, ell_width, col_ind, base, fill_mode, index_type, map, temp_buffer);
    }

    template <uint32_t WF_SIZE, bool SLEEP, typename I, typename T>
    static rocsparse_status ellsv_launch_solve(rocsparse_handle     handle,
                                               I                    m,
                                               I                    n,
                                               int64_t              ell_width,
                                               const void*          alpha,
                                               const void*          ell_col_ind,
                                               const void*          ell_val,
                                               const void*          x,
                                               int64_t              x_inc,
                                               void*                y,
                                               int64_t              y_inc,
                                               const void*          row_map,
                                               rocsparse_index_base base,
                                               rocsparse_fill_mode  fill_mode,
                                               rocsparse_diag_type  diag_type,
                                               void*                temp_buffer,
                                               bool                 is_host_mode)
    {
        constexpr uint32_t BLOCKSIZE = 1024;

        hipStream_t stream = handle->stream;

        char* ptr = reinterpret_cast<char*>(temp_buffer);
        ptr += 256;

        const size_t done_bytes = ellsv_align256(sizeof(int32_t) * static_cast<size_t>(m));
        int32_t*     done_array = reinterpret_cast<int32_t*>(ptr);

        RETURN_IF_HIP_ERROR(rocsparse_hipMemsetAsync(done_array, 0, done_bytes, stream));

        dim3 blocks((static_cast<size_t>(m) * WF_SIZE - 1) / BLOCKSIZE + 1);
        dim3 threads(BLOCKSIZE);

        auto alpha_ = reinterpret_cast<const T*>(alpha);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::ellsv_solve_kernel<BLOCKSIZE, WF_SIZE, SLEEP, I, T>),
            blocks,
            threads,
            0,
            stream,
            m,
            n,
            ell_width,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_),
            reinterpret_cast<const I*>(ell_col_ind),
            reinterpret_cast<const T*>(ell_val),
            reinterpret_cast<const T*>(x),
            x_inc,
            reinterpret_cast<T*>(y),
            y_inc,
            done_array,
            reinterpret_cast<const I*>(row_map),
            base,
            fill_mode,
            diag_type,
            is_host_mode);

        return rocsparse_status_success;
    }

    template <typename I, typename T>
    static rocsparse_status ellsv_solve_dispatch(rocsparse_handle     handle,
                                                 bool                 sleep,
                                                 uint32_t             wfsize,
                                                 I                    m,
                                                 I                    n,
                                                 int64_t              ell_width,
                                                 const void*          alpha,
                                                 const void*          ell_col_ind,
                                                 const void*          ell_val,
                                                 const void*          x,
                                                 int64_t              x_inc,
                                                 void*                y,
                                                 int64_t              y_inc,
                                                 const void*          row_map,
                                                 rocsparse_index_base base,
                                                 rocsparse_fill_mode  fill_mode,
                                                 rocsparse_diag_type  diag_type,
                                                 void*                temp_buffer,
                                                 bool                 is_host_mode)
    {
        if(sleep)
        {
            return rocsparse::ellsv_launch_solve<64, true, I, T>(handle,
                                                                 m,
                                                                 n,
                                                                 ell_width,
                                                                 alpha,
                                                                 ell_col_ind,
                                                                 ell_val,
                                                                 x,
                                                                 x_inc,
                                                                 y,
                                                                 y_inc,
                                                                 row_map,
                                                                 base,
                                                                 fill_mode,
                                                                 diag_type,
                                                                 temp_buffer,
                                                                 is_host_mode);
        }
        else if(wfsize == 64)
        {
            return rocsparse::ellsv_launch_solve<64, false, I, T>(handle,
                                                                  m,
                                                                  n,
                                                                  ell_width,
                                                                  alpha,
                                                                  ell_col_ind,
                                                                  ell_val,
                                                                  x,
                                                                  x_inc,
                                                                  y,
                                                                  y_inc,
                                                                  row_map,
                                                                  base,
                                                                  fill_mode,
                                                                  diag_type,
                                                                  temp_buffer,
                                                                  is_host_mode);
        }

        return rocsparse::ellsv_launch_solve<32, false, I, T>(handle,
                                                              m,
                                                              n,
                                                              ell_width,
                                                              alpha,
                                                              ell_col_ind,
                                                              ell_val,
                                                              x,
                                                              x_inc,
                                                              y,
                                                              y_inc,
                                                              row_map,
                                                              base,
                                                              fill_mode,
                                                              diag_type,
                                                              temp_buffer,
                                                              is_host_mode);
    }

    // Materialize the transpose of the ELL matrix A as another ELL matrix stored
    // in the info object. Entry (row, col) of A becomes entry (col, row) of A^T.
    template <typename I, typename T>
    static rocsparse_status ellsv_build_transpose(rocsparse_handle         handle,
                                                  I                        m,
                                                  I                        n,
                                                  int64_t                  ell_width,
                                                  const void*              ell_col_ind,
                                                  const void*              ell_val,
                                                  rocsparse_index_base     base,
                                                  bool                     conj,
                                                  rocsparse::ellsv_info_t* ei)
    {
        constexpr uint32_t BLOCKSIZE = 256;

        hipStream_t stream = handle->stream;

        // Per-column entry counts (== per-row counts of the transpose).
        unsigned long long* counts = nullptr;
        RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(
            &counts, sizeof(unsigned long long) * static_cast<size_t>(m), stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipMemsetAsync(
            counts, 0, sizeof(unsigned long long) * static_cast<size_t>(m), stream));

        dim3 blocks((static_cast<size_t>(m) - 1) / BLOCKSIZE + 1);
        dim3 threads(BLOCKSIZE);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::ellsv_transpose_count_kernel<BLOCKSIZE, I>),
                                           blocks,
                                           threads,
                                           0,
                                           stream,
                                           m,
                                           n,
                                           ell_width,
                                           reinterpret_cast<const I*>(ell_col_ind),
                                           base,
                                           counts);

        // Determine the transposed width (maximum per-column count).
        std::vector<unsigned long long> h_counts(static_cast<size_t>(m));
        RETURN_IF_HIP_ERROR(
            rocsparse_hipMemcpyAsync(h_counts.data(),
                                     counts,
                                     sizeof(unsigned long long) * static_cast<size_t>(m),
                                     hipMemcpyDeviceToHost,
                                     stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(stream));

        int64_t t_width = 0;
        for(int64_t i = 0; i < static_cast<int64_t>(m); ++i)
        {
            t_width = rocsparse::max(t_width, static_cast<int64_t>(h_counts[i]));
        }

        RETURN_IF_ROCSPARSE_ERROR(ei->allocate_transposed(t_width, stream));

        // Initialize the transposed column indices with a padding sentinel that
        // is out-of-range for the transposed matrix (which has n columns).
        const int64_t total = static_cast<int64_t>(m) * t_width;
        if(total > 0)
        {
            dim3 fill_blocks((total - 1) / BLOCKSIZE + 1);
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::ellsv_fill_col_ind_kernel<BLOCKSIZE, I>),
                                               fill_blocks,
                                               threads,
                                               0,
                                               stream,
                                               total,
                                               reinterpret_cast<I*>(ei->get_transposed_col_ind()),
                                               static_cast<I>(m + base));
        }

        // Reuse the counts array as per-column write positions.
        RETURN_IF_HIP_ERROR(rocsparse_hipMemsetAsync(
            counts, 0, sizeof(unsigned long long) * static_cast<size_t>(m), stream));

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::ellsv_transpose_scatter_kernel<BLOCKSIZE, I, T>),
            blocks,
            threads,
            0,
            stream,
            m,
            n,
            ell_width,
            reinterpret_cast<const I*>(ell_col_ind),
            reinterpret_cast<const T*>(ell_val),
            base,
            conj,
            counts,
            reinterpret_cast<I*>(ei->get_transposed_col_ind()),
            reinterpret_cast<T*>(ei->get_transposed_val()));

        RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(counts, stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(stream));

        return rocsparse_status_success;
    }

    // Flip the fill mode (used when the transpose is applied).
    static rocsparse_fill_mode ellsv_flip_fill(rocsparse_fill_mode fill_mode)
    {
        return (fill_mode == rocsparse_fill_mode_lower) ? rocsparse_fill_mode_upper
                                                        : rocsparse_fill_mode_lower;
    }

    // Analysis for a given index/value type: for a transposed operation the
    // transpose is materialized first and the level-scheduling then runs on it
    // with a flipped fill mode.
    template <typename I, typename T>
    static rocsparse_status ellsv_preprocess(rocsparse_handle            handle,
                                             rocsparse::ellsv_info_t*    ei,
                                             bool                        sleep,
                                             uint32_t                    wfsize,
                                             rocsparse_operation         trans,
                                             rocsparse_const_spmat_descr A,
                                             void*                       temp_buffer)
    {
        const I             m    = static_cast<I>(A->rows);
        const I             cols = static_cast<I>(A->cols);
        rocsparse_fill_mode fill = A->descr->fill_mode;

        const void* col_ind  = A->const_col_data;
        I           n_solver = cols;
        int64_t     width    = A->ell_width;

        if(trans != rocsparse_operation_none)
        {
            fill                 = rocsparse::ellsv_flip_fill(fill);
            const bool conjugate = (trans == rocsparse_operation_conjugate_transpose);

            RETURN_IF_ROCSPARSE_ERROR((rocsparse::ellsv_build_transpose<I, T>(handle,
                                                                              m,
                                                                              cols,
                                                                              A->ell_width,
                                                                              A->const_col_data,
                                                                              A->const_val_data,
                                                                              A->idx_base,
                                                                              conjugate,
                                                                              ei)));

            col_ind  = ei->get_transposed_col_ind();
            n_solver = m;
            width    = ei->get_transposed_width();
        }

        return rocsparse::ellsv_analysis_dispatch<I>(handle,
                                                     sleep,
                                                     wfsize,
                                                     m,
                                                     n_solver,
                                                     width,
                                                     col_ind,
                                                     A->idx_base,
                                                     fill,
                                                     A->col_type,
                                                     ei->get_row_map(),
                                                     temp_buffer);
    }

    // Solve for a given index/value type, reading either A or its materialized
    // transpose depending on the requested operation.
    template <typename I, typename T>
    static rocsparse_status ellsv_compute(rocsparse_handle            handle,
                                          rocsparse::ellsv_info_t*    ei,
                                          bool                        sleep,
                                          uint32_t                    wfsize,
                                          rocsparse_operation         trans,
                                          rocsparse_const_spmat_descr A,
                                          const void*                 alpha,
                                          rocsparse_const_dnvec_descr x,
                                          rocsparse_dnvec_descr       y,
                                          void*                       temp_buffer,
                                          bool                        is_host_mode)
    {
        const I             m    = static_cast<I>(A->rows);
        const I             cols = static_cast<I>(A->cols);
        rocsparse_fill_mode fill = A->descr->fill_mode;

        const void* col_ind  = A->const_col_data;
        const void* val      = A->const_val_data;
        I           n_solver = cols;
        int64_t     width    = A->ell_width;

        if(trans != rocsparse_operation_none)
        {
            fill     = rocsparse::ellsv_flip_fill(fill);
            col_ind  = ei->get_transposed_col_ind();
            val      = ei->get_transposed_val();
            n_solver = m;
            width    = ei->get_transposed_width();
        }

        return rocsparse::ellsv_solve_dispatch<I, T>(handle,
                                                     sleep,
                                                     wfsize,
                                                     m,
                                                     n_solver,
                                                     width,
                                                     alpha,
                                                     col_ind,
                                                     val,
                                                     x->const_values,
                                                     x->inc,
                                                     y->values,
                                                     y->inc,
                                                     ei->get_row_map(),
                                                     A->idx_base,
                                                     fill,
                                                     A->descr->diag_type,
                                                     temp_buffer,
                                                     is_host_mode);
    }

    // Common argument validation shared by the ELL solve entry points.
    static rocsparse_status ellsv_check(rocsparse_const_spmat_descr A)
    {
        rocsparse_mat_descr descr = A->descr;
        ROCSPARSE_CHECKARG(2,
                           descr,
                           (descr->type != rocsparse_matrix_type_general
                            && descr->type != rocsparse_matrix_type_triangular),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(2,
                           descr,
                           (descr->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);
        return rocsparse_status_success;
    }
}

rocsparse_status rocsparse::ellsv_analysis_buffer_size(rocsparse_handle            handle,
                                                       rocsparse_operation         trans,
                                                       rocsparse_const_spmat_descr A,
                                                       size_t* buffer_size_in_bytes)
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_ENUM(1, trans);
    ROCSPARSE_CHECKARG_POINTER(2, A);
    ROCSPARSE_CHECKARG_POINTER(3, buffer_size_in_bytes);

    // Quick return if possible
    if(A->rows == 0 || A->batch_count == 0)
    {
        *buffer_size_in_bytes = 0;
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::ellsv_check(A));

    const int64_t m        = A->rows;
    const size_t  sizeof_I = rocsparse::indextype_sizeof(A->col_type);

    // done_array | workspace | workspace2 | rocprim buffer
    size_t size = rocsparse::ellsv_align256(sizeof(int32_t) * static_cast<size_t>(m));
    size += rocsparse::ellsv_align256(sizeof_I * static_cast<size_t>(m));
    size += rocsparse::ellsv_align256(sizeof(int32_t) * static_cast<size_t>(m));

    const uint32_t startbit = 0;
    const uint32_t endbit   = rocsparse::clz(m);

    size_t rocprim_size = 0;
    auto   calculate_rocprim_size
        = rocsparse::find_radix_sort_pairs_buffer_size(rocsparse_indextype_i32, A->col_type);
    RETURN_IF_ROCSPARSE_ERROR(
        (calculate_rocprim_size(handle, m, startbit, endbit, &rocprim_size, true)));

    size += rocprim_size;

    *buffer_size_in_bytes = size;

    return rocsparse_status_success;
}

rocsparse_status rocsparse::ellsv_solve_buffer_size(rocsparse_handle            handle,
                                                    rocsparse_operation         trans,
                                                    rocsparse_const_spmat_descr A,
                                                    rocsparse_const_dnvec_descr x,
                                                    rocsparse_const_dnvec_descr y,
                                                    size_t* buffer_size_in_bytes)
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_ENUM(1, trans);
    ROCSPARSE_CHECKARG_POINTER(2, A);
    ROCSPARSE_CHECKARG_POINTER(3, buffer_size_in_bytes);

    const int64_t batch_count = (y) ? y->batch_count : A->batch_count;

    // Quick return if possible
    if(A->rows == 0 || batch_count == 0)
    {
        *buffer_size_in_bytes = 0;
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::ellsv_check(A));

    const int64_t m = A->rows;

    // 256 bytes of padding followed by the per-row done flags.
    size_t size = 256;
    size += rocsparse::ellsv_align256(sizeof(int32_t) * static_cast<size_t>(m)
                                      * static_cast<size_t>(batch_count));

    *buffer_size_in_bytes = size;

    return rocsparse_status_success;
}

rocsparse_status rocsparse::ellsv_analysis(rocsparse_handle            handle,
                                           rocsparse_operation         trans,
                                           rocsparse_const_spmat_descr A,
                                           void*                       temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_ENUM(1, trans);
    ROCSPARSE_CHECKARG_POINTER(2, A);

    // Quick return if possible
    if(A->rows == 0 || A->batch_count == 0)
    {
        return rocsparse_status_success;
    }

    ROCSPARSE_CHECKARG_ARRAY(3, (A->rows > 0 && A->batch_count > 0), temp_buffer);

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::ellsv_check(A));

    rocsparse_mat_descr descr = A->descr;
    rocsparse_mat_info  info  = A->info;

    // Reuse (or recreate) the persistent storage held in the info object.
    rocsparse::ellsv_info_t* ei = info->get_ellsv_info();
    if(ei != nullptr
       && (ei->get_num_rows() != A->rows || ei->get_index_type() != A->col_type
           || ei->get_value_type() != A->data_type))
    {
        std::ignore = ei->free_memory(handle->stream);
        delete ei;
        ei = nullptr;
        info->set_ellsv_info(nullptr);
    }

    if(ei == nullptr)
    {
        ei = new rocsparse::ellsv_info_t(A->rows, A->col_type, A->data_type, handle->stream);
        info->set_ellsv_info(ei);
    }

    bool     sleep  = false;
    uint32_t wfsize = 0;
    rocsparse::ellsv_select_launch(handle, &sleep, &wfsize);

#define ELLSV_PREPROCESS_DISPATCH(ITYPE, TTYPE) \
    rocsparse::ellsv_preprocess<ITYPE, TTYPE>(handle, ei, sleep, wfsize, trans, A, temp_buffer)

    switch(A->col_type)
    {
    case rocsparse_indextype_i32:
    {
        switch(A->data_type)
        {
        case rocsparse_datatype_f32_r:
            RETURN_IF_ROCSPARSE_ERROR(ELLSV_PREPROCESS_DISPATCH(int32_t, float));
            break;
        case rocsparse_datatype_f64_r:
            RETURN_IF_ROCSPARSE_ERROR(ELLSV_PREPROCESS_DISPATCH(int32_t, double));
            break;
        case rocsparse_datatype_f32_c:
            RETURN_IF_ROCSPARSE_ERROR(ELLSV_PREPROCESS_DISPATCH(int32_t, rocsparse_float_complex));
            break;
        case rocsparse_datatype_f64_c:
            RETURN_IF_ROCSPARSE_ERROR(ELLSV_PREPROCESS_DISPATCH(int32_t, rocsparse_double_complex));
            break;
        default:
            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            // LCOV_EXCL_STOP
        }
        break;
    }
    case rocsparse_indextype_i64:
    {
        switch(A->data_type)
        {
        case rocsparse_datatype_f32_r:
            RETURN_IF_ROCSPARSE_ERROR(ELLSV_PREPROCESS_DISPATCH(int64_t, float));
            break;
        case rocsparse_datatype_f64_r:
            RETURN_IF_ROCSPARSE_ERROR(ELLSV_PREPROCESS_DISPATCH(int64_t, double));
            break;
        case rocsparse_datatype_f32_c:
            RETURN_IF_ROCSPARSE_ERROR(ELLSV_PREPROCESS_DISPATCH(int64_t, rocsparse_float_complex));
            break;
        case rocsparse_datatype_f64_c:
            RETURN_IF_ROCSPARSE_ERROR(ELLSV_PREPROCESS_DISPATCH(int64_t, rocsparse_double_complex));
            break;
        default:
            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            // LCOV_EXCL_STOP
        }
        break;
    }
    case deprecated_rocsparse_indextype_u16:
    {
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        // LCOV_EXCL_STOP
    }
    }

#undef ELLSV_PREPROCESS_DISPATCH

    ei->set_config(trans, descr->fill_mode, descr->diag_type);

    return rocsparse_status_success;
}

rocsparse_status rocsparse::ellsv_solve(rocsparse_handle            handle,
                                        rocsparse_operation         trans,
                                        rocsparse_datatype          alpha_datatype,
                                        const void*                 alpha,
                                        int64_t                     alpha_stride,
                                        rocsparse_const_spmat_descr A,
                                        rocsparse_const_dnvec_descr x,
                                        rocsparse_dnvec_descr       y,
                                        void*                       temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_ENUM(1, trans);
    ROCSPARSE_CHECKARG_ENUM(2, alpha_datatype);
    ROCSPARSE_CHECKARG_POINTER(5, A);
    ROCSPARSE_CHECKARG_ARRAY(3, A->batch_count, alpha);
    ROCSPARSE_CHECKARG_POINTER(6, x);
    ROCSPARSE_CHECKARG_POINTER(7, y);

    if(A->rows == 0 || A->batch_count == 0)
    {
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::ellsv_check(A));

    // A single right-hand side is supported.
    ROCSPARSE_CHECKARG(
        7, y, (y->batch_count > 1 || A->batch_count > 1), rocsparse_status_not_implemented);

    rocsparse_mat_descr descr = A->descr;
    rocsparse_mat_info  info  = A->info;

    rocsparse::ellsv_info_t* ei = info->get_ellsv_info();
    if(ei == nullptr || !ei->matches(trans, descr->fill_mode, descr->diag_type))
    {
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
            rocsparse_status_internal_error,
            "ellsv row map is not available, it looks like the analysis phase of this "
            "algorithm was not previously executed.");
    }

    const bool is_host_mode = (handle->pointer_mode == rocsparse_pointer_mode_host);

    bool     sleep  = false;
    uint32_t wfsize = 0;
    rocsparse::ellsv_select_launch(handle, &sleep, &wfsize);

#define ELLSV_SOLVE_DISPATCH(ITYPE, TTYPE)  \
    rocsparse::ellsv_compute<ITYPE, TTYPE>( \
        handle, ei, sleep, wfsize, trans, A, alpha, x, y, temp_buffer, is_host_mode)

    switch(A->col_type)
    {
    case rocsparse_indextype_i32:
    {
        switch(A->data_type)
        {
        case rocsparse_datatype_f32_r:
            RETURN_IF_ROCSPARSE_ERROR(ELLSV_SOLVE_DISPATCH(int32_t, float));
            break;
        case rocsparse_datatype_f64_r:
            RETURN_IF_ROCSPARSE_ERROR(ELLSV_SOLVE_DISPATCH(int32_t, double));
            break;
        case rocsparse_datatype_f32_c:
            RETURN_IF_ROCSPARSE_ERROR(ELLSV_SOLVE_DISPATCH(int32_t, rocsparse_float_complex));
            break;
        case rocsparse_datatype_f64_c:
            RETURN_IF_ROCSPARSE_ERROR(ELLSV_SOLVE_DISPATCH(int32_t, rocsparse_double_complex));
            break;
        default:
            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            // LCOV_EXCL_STOP
        }
        break;
    }
    case rocsparse_indextype_i64:
    {
        switch(A->data_type)
        {
        case rocsparse_datatype_f32_r:
            RETURN_IF_ROCSPARSE_ERROR(ELLSV_SOLVE_DISPATCH(int64_t, float));
            break;
        case rocsparse_datatype_f64_r:
            RETURN_IF_ROCSPARSE_ERROR(ELLSV_SOLVE_DISPATCH(int64_t, double));
            break;
        case rocsparse_datatype_f32_c:
            RETURN_IF_ROCSPARSE_ERROR(ELLSV_SOLVE_DISPATCH(int64_t, rocsparse_float_complex));
            break;
        case rocsparse_datatype_f64_c:
            RETURN_IF_ROCSPARSE_ERROR(ELLSV_SOLVE_DISPATCH(int64_t, rocsparse_double_complex));
            break;
        default:
            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            // LCOV_EXCL_STOP
        }
        break;
    }
    case deprecated_rocsparse_indextype_u16:
    {
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        // LCOV_EXCL_STOP
    }
    }

#undef ELLSV_SOLVE_DISPATCH

    return rocsparse_status_success;
}
