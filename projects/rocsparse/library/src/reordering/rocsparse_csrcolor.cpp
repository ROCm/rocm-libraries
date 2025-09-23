/*! \file */
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

#include "internal/reordering/rocsparse_csrcolor.h"
#include "csrcolor_device.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_csrcolor.hpp"
#include "rocsparse_primitives.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    template <rocsparse_int n, typename I>
    ROCSPARSE_DEVICE_ILF void count_uncolored_reduce_device(rocsparse_int tx, I* sdata)
    {
        __syncthreads();
        if(tx < n / 2)
        {
            sdata[tx] += sdata[tx + n / 2];
        }
        count_uncolored_reduce_device<n / 2>(tx, sdata);
    }

    template <>
    __forceinline__ __device__ void count_uncolored_reduce_device<0, int32_t>(rocsparse_int tx,
                                                                              int32_t*      sdata)
    {
    }

    template <rocsparse_int NB_X, typename J>
    ROCSPARSE_KERNEL(NB_X)
    void count_uncolored(
        J size, J m, J n, const J* __restrict__ colors, J* __restrict__ uncolored_per_sequence)
    {
        static constexpr J s_uncolored_value = static_cast<J>(-1);

        J tx  = hipThreadIdx_x;
        J col = hipBlockIdx_x;

        J m_full = (m / NB_X) * NB_X;
        J res    = static_cast<J>(0);

        __shared__ J sdata[NB_X];

        colors += col * m + ((tx < m) ? tx : 0);
        for(J i = 0; i < m_full; i += NB_X)
        {
            res += (colors[i] == s_uncolored_value) ? 1 : 0;
        }

        if(tx + m_full < m)
        {
            res += (colors[m_full] == s_uncolored_value) ? 1 : 0;
        }

        sdata[tx] = res;
        if(NB_X > 16 && m >= NB_X)
        {
            count_uncolored_reduce_device<NB_X>(tx, sdata);
        }
        else
        {
            __syncthreads();

            if(tx == 0)
            {
                for(J i = 1; i < m && i < NB_X; i++)
                    sdata[0] += sdata[i];
            }

            __syncthreads();
        }

        if(tx == 0)
        {
            uncolored_per_sequence[col] = sdata[0];
        }
    }

    template <rocsparse_int BLOCKSIZE, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrcolor_reordering_identity(J size, J* identity)
    {
        const J gid = BLOCKSIZE * hipBlockIdx_x + hipThreadIdx_x;
        if(gid < size)
        {
            identity[gid] = gid;
        }
    }

    template <rocsparse_int NUMCOLUMNS_PER_BLOCK, rocsparse_int WF_SIZE, typename J>
    ROCSPARSE_KERNEL(WF_SIZE* NUMCOLUMNS_PER_BLOCK)
    void csrcolor_assign_uncolored_kernel(
        J size, J m, J n, J shift_color, J* __restrict__ colors, J* __restrict__ index_sequence)
    {
        static constexpr J  s_uncolored_value = static_cast<J>(-1);
        const rocsparse_int wavefront_index   = hipThreadIdx_x / WF_SIZE;
        const J             lane_index        = hipThreadIdx_x % WF_SIZE;
        const uint64_t      filter            = 0xffffffffffffffff >> (63 - lane_index);
        const J             column_index = NUMCOLUMNS_PER_BLOCK * hipBlockIdx_x + wavefront_index;

        if(column_index < n)
        {
            J shift = shift_color + index_sequence[column_index];
            //
            // The warp handles the entire column.
            //
            for(J row_index = lane_index; row_index < m; row_index += WF_SIZE)
            {
                const J gid = column_index * m + row_index;

                //
                // Get value.
                //
                J* pcolor = colors + gid;

                //
                // Predicate.
                //
                const bool predicate = (gid < size) ? (s_uncolored_value == *pcolor) : false;

                //
                // Mask of the wavefront.
                //
                const uint64_t wavefront_mask = __ballot(predicate);

                //
                // Get the number of previous non-zero in the row.
                //
                const uint64_t count_previous_uncolored = __popcll(wavefront_mask & filter);

                //
                // Synchronize for cache considerations.
                //
                __syncthreads();

                if(predicate)
                {
                    //
                    // Calculate local index.
                    //
                    const uint64_t local_index = count_previous_uncolored - 1;

                    //
                    // Populate the sparse matrix.
                    //
                    *pcolor = shift + local_index;
                }

                //
                // Broadcast the update of the shift to all 64 threads for the next set of 64 columns.
                // Choose the last lane since that it contains the size of the sparse row (even if its predicate is false).
                //
                shift += rocsparse::shfl(static_cast<J>(count_previous_uncolored), WF_SIZE - 1);
            }
        }
    }

    template <typename J>
    static rocsparse_status csrcolor_assign_uncolored(rocsparse_handle handle,
                                                      J                num_colored,
                                                      J                colors_length,
                                                      J*               colors)
    {
        ROCSPARSE_ROUTINE_TRACE;

        hipStream_t stream = handle->stream;
        J           m, n;

        J*                 seq_ptr = nullptr;
        static constexpr J NB      = 256;

        m = NB * 4;
        n = (colors_length - 1) / m + 1;

        //
        // Allocation.
        //
        RETURN_IF_HIP_ERROR(
            rocsparse_hipMallocAsync(&seq_ptr, sizeof(J) * (n + 1), handle->stream));

        //
        // Set to 0.
        //
        RETURN_IF_HIP_ERROR(hipMemsetAsync(seq_ptr, 0, sizeof(J) * (n + 1), stream));

        //
        // Count uncolored values.
        //
        {
            dim3 kernel_blocks(n);
            dim3 kernel_threads(NB);
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::count_uncolored<NB, J>),
                                               kernel_blocks,
                                               kernel_threads,
                                               0,
                                               stream,
                                               colors_length,
                                               m,
                                               n,
                                               colors,
                                               seq_ptr + 1);
        }

        //
        // Next perform an inclusive sum.
        //
        size_t temp_storage_bytes = 0;

        //
        // Obtain rocprim buffer size
        //
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::inclusive_scan_buffer_size<J, J>(
            handle, n + 1, &temp_storage_bytes)));

        //
        // Get rocprim buffer
        //
        bool  d_temp_alloc;
        void* d_temp_storage = nullptr;

        //
        // Device buffer should be sufficient for rocprim in most cases
        //
        if(handle->buffer_size >= temp_storage_bytes)
        {
            d_temp_storage = handle->buffer;
            d_temp_alloc   = false;
        }
        else
        {
            RETURN_IF_HIP_ERROR(
                rocsparse_hipMallocAsync(&d_temp_storage, temp_storage_bytes, handle->stream));
            d_temp_alloc = true;
        }

        //
        // Perform actual inclusive sum.
        //
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::inclusive_scan(
            handle, seq_ptr, seq_ptr, n + 1, temp_storage_bytes, d_temp_storage));

        //
        // Free rocprim buffer, if allocated.
        //
        if(d_temp_alloc == true)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(d_temp_storage, handle->stream));
        }

        //
        // Now we traverse again and we use num_colored_per_sequence.
        //
        static constexpr rocsparse_int data_ratio = sizeof(J) / sizeof(float);
        if(handle->wavefront_size == 32)
        {
            static constexpr rocsparse_int WF_SIZE = 32;
            static constexpr rocsparse_int NCOLUMNS_PER_BLOCK
                = 16 / (data_ratio > 0 ? data_ratio : 1);
            rocsparse_int blocks = (n - 1) / NCOLUMNS_PER_BLOCK + 1;
            dim3          k_blocks(blocks), k_threads(WF_SIZE * NCOLUMNS_PER_BLOCK);

            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::csrcolor_assign_uncolored_kernel<NCOLUMNS_PER_BLOCK, WF_SIZE>),
                k_blocks,
                k_threads,
                0,
                stream,
                colors_length,
                m,
                n,
                num_colored,
                colors,
                seq_ptr);
        }
        else
        {
            static constexpr rocsparse_int WF_SIZE = 64;
            static constexpr rocsparse_int NCOLUMNS_PER_BLOCK
                = 16 / (data_ratio > 0 ? data_ratio : 1);
            rocsparse_int blocks = (n - 1) / NCOLUMNS_PER_BLOCK + 1;

            dim3 k_blocks(blocks), k_threads(WF_SIZE * NCOLUMNS_PER_BLOCK);

            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::csrcolor_assign_uncolored_kernel<NCOLUMNS_PER_BLOCK, WF_SIZE>),
                k_blocks,
                k_threads,
                0,
                stream,
                colors_length,
                m,
                n,
                num_colored,
                colors,
                seq_ptr);
        }

        RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(seq_ptr, handle->stream));
        return rocsparse_status_success;
    }
}

template <typename T, typename I, typename J>
rocsparse_status rocsparse::csrcolor_core(rocsparse_handle          handle,
                                          J                         m,
                                          I                         nnz,
                                          const rocsparse_mat_descr descr,
                                          const T*                  csr_val,
                                          const I*                  csr_row_ptr,
                                          const J*                  csr_col_ind,
                                          const floating_data_t<T>* fraction_to_color,
                                          J*                        ncolors,
                                          J*                        colors,
                                          J*                        reordering,
                                          rocsparse_mat_info        info)
{
    ROCSPARSE_ROUTINE_TRACE;

    static constexpr rocsparse_int blocksize = 256;

    hipStream_t stream = handle->stream;
    *ncolors           = -2;

    J num_uncolored     = m;
    J max_num_uncolored = m - m * fraction_to_color[0];

    //
    // Create workspace.
    //
    J* workspace;
    RETURN_IF_HIP_ERROR(
        rocsparse_hipMallocAsync(&workspace, sizeof(J) * blocksize, handle->stream));

    //
    // Initialize colors
    //
    RETURN_IF_HIP_ERROR(hipMemsetAsync(colors, -1, sizeof(J) * m, stream));

    //
    // Iterate until the desired fraction of colored vertices is reached
    //
    while(num_uncolored > max_num_uncolored)
    {
        *ncolors += 2;

        //
        // Run Jones-Plassmann Luby algorithm
        //
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csrcolor_kernel_jpl<blocksize, I, J>),
                                           dim3((m - 1) / blocksize + 1),
                                           dim3(blocksize),
                                           0,
                                           stream,
                                           m,
                                           *ncolors,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           descr->base,
                                           colors);

        //
        // Count colored vertices
        //
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrcolor_kernel_count_uncolored<blocksize, J>),
            dim3(blocksize),
            dim3(blocksize),
            0,
            stream,
            m,
            colors,
            workspace);

        //
        // Gather results.
        //
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrcolor_kernel_count_uncolored_finalize<blocksize, J>),
            dim3(1),
            dim3(blocksize),
            0,
            stream,
            workspace);

        //
        // Copy colored max vertices for current iteration to host
        //
        RETURN_IF_HIP_ERROR(
            hipMemcpyAsync(&num_uncolored, workspace, sizeof(J), hipMemcpyDeviceToHost, stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));
    }

    //
    // Need to count the number of colors, compute the maximum value + 1.
    // This is something I'll need to figure out.
    // *ncolors += 2 is not the right number of colors, sometimes yes, sometimes no.
    //
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csrcolor_kernel_count_colors<blocksize, J>),
                                           dim3(blocksize),
                                           dim3(blocksize),
                                           0,
                                           stream,
                                           m,
                                           colors,
                                           workspace);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrcolor_kernel_count_colors_finalize<blocksize, J>),
            dim3(1),
            dim3(blocksize),
            0,
            stream,
            workspace);

        RETURN_IF_HIP_ERROR(
            hipMemcpyAsync(ncolors, workspace, sizeof(J), hipMemcpyDeviceToHost, stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));
        *ncolors += 1;
    }
    //
    // Free workspace.
    //
    RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(workspace, handle->stream));

    if(num_uncolored > 0)
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::csrcolor_assign_uncolored(handle, *ncolors, m, colors));
        *ncolors += num_uncolored;
    }

    //
    // Calculating reorering if required.
    //
    if(nullptr != reordering)
    {
        rocsparse_int* reordering_identity = nullptr;
        rocsparse_int* sorted_colors       = nullptr;
        //
        // Create identity.
        //
        RETURN_IF_HIP_ERROR(
            rocsparse_hipMallocAsync(&reordering_identity, sizeof(J) * m, handle->stream));

        //
        //
        //
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csrcolor_reordering_identity<1024, J>),
                                           dim3((m - 1) / 1024 + 1),
                                           dim3(1024),
                                           0,
                                           stream,
                                           m,
                                           reordering_identity);

        //
        // Alloc output sorted colors.
        //
        RETURN_IF_HIP_ERROR(
            rocsparse_hipMallocAsync(&sorted_colors, sizeof(J) * m, handle->stream));

        {
            size_t temporary_storage_size_bytes;
            void*  temporary_storage_ptr = nullptr;

            //
            // Get required size of the temporary storage
            //
            static constexpr bool s_using_double_buffers = false;
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::primitives::radix_sort_pairs_buffer_size<rocsparse_int, rocsparse_int>(
                    handle,
                    m,
                    0,
                    sizeof(rocsparse_int) * 8,
                    &temporary_storage_size_bytes,
                    s_using_double_buffers)));

            //
            // allocate temporary storage
            //
            RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(
                &temporary_storage_ptr, temporary_storage_size_bytes, handle->stream));

            //
            // perform sort
            //
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::primitives::radix_sort_pairs(handle,
                                                        colors,
                                                        sorted_colors,
                                                        reordering_identity,
                                                        reordering,
                                                        m,
                                                        0,
                                                        sizeof(rocsparse_int) * 8,
                                                        temporary_storage_size_bytes,
                                                        temporary_storage_ptr));

            RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(temporary_storage_ptr, handle->stream));
        }

        RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(reordering_identity, handle->stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(sorted_colors, handle->stream));
    }

    return rocsparse_status_success;
}

rocsparse_status rocsparse::csrcolor_quickreturn(rocsparse_handle          handle,
                                                 int64_t                   m,
                                                 int64_t                   nnz,
                                                 const rocsparse_mat_descr descr,
                                                 const void*               csr_val,
                                                 const void*               csr_row_ptr,
                                                 const void*               csr_col_ind,
                                                 const void*               fraction_to_color,
                                                 void*                     ncolors,
                                                 void*                     coloring,
                                                 void*                     reordering,
                                                 rocsparse_mat_info        info)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(m == 0 || nnz == 0)
    {
        return rocsparse_status_success;
    }
    return rocsparse_status_continue;
}

namespace rocsparse
{
    static rocsparse_status csrcolor_checkarg(rocsparse_handle          handle, //0
                                              int64_t                   m, //1
                                              int64_t                   nnz, //2
                                              const rocsparse_mat_descr descr, //3
                                              const void*               csr_val, //4
                                              const void*               csr_row_ptr, //5
                                              const void*               csr_col_ind, //6
                                              const void*               fraction_to_color, //7
                                              void*                     ncolors, //8
                                              void*                     coloring, //9
                                              void*                     reordering, //10
                                              rocsparse_mat_info        info) //11
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_SIZE(1, m);
        ROCSPARSE_CHECKARG_SIZE(2, nnz);

        const rocsparse_status status = rocsparse::csrcolor_quickreturn(handle,
                                                                        m,
                                                                        nnz,
                                                                        descr,
                                                                        csr_val,
                                                                        csr_row_ptr,
                                                                        csr_col_ind,
                                                                        fraction_to_color,
                                                                        ncolors,
                                                                        coloring,
                                                                        reordering,
                                                                        info);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        ROCSPARSE_CHECKARG_POINTER(3, descr);
        ROCSPARSE_CHECKARG(3,
                           descr,
                           (descr->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(3,
                           descr,
                           (descr->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);
        ROCSPARSE_CHECKARG_ARRAY(4, nnz, csr_val);
        ROCSPARSE_CHECKARG_ARRAY(5, m, csr_row_ptr);
        ROCSPARSE_CHECKARG_ARRAY(6, nnz, csr_col_ind);
        ROCSPARSE_CHECKARG_POINTER(7, fraction_to_color);
        ROCSPARSE_CHECKARG_POINTER(8, ncolors);
        ROCSPARSE_CHECKARG_ARRAY(9, m, coloring);
        ROCSPARSE_CHECKARG_POINTER(11, info);
        return rocsparse_status_continue;
    }

    template <typename T, typename I, typename J>
    static rocsparse_status csrcolor_impl(rocsparse_handle          handle,
                                          J                         m,
                                          I                         nnz,
                                          const rocsparse_mat_descr descr,
                                          const T*                  csr_val,
                                          const I*                  csr_row_ptr,
                                          const J*                  csr_col_ind,
                                          const floating_data_t<T>* fraction_to_color,
                                          J*                        ncolors,
                                          J*                        coloring,
                                          J*                        reordering,
                                          rocsparse_mat_info        info)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Logging
        rocsparse::log_trace(handle,
                             rocsparse::replaceX<T>("rocsparse_Xcsrcolor"),
                             m,
                             nnz,
                             (const void*&)descr,
                             (const void*&)csr_val,
                             (const void*&)csr_row_ptr,
                             (const void*&)csr_col_ind,
                             (const void*&)fraction_to_color,
                             (const void*&)ncolors,
                             (const void*&)coloring,
                             (const void*&)reordering,
                             (const void*&)info);

        //
        // Check arguments.
        //
        const rocsparse_status status = rocsparse::csrcolor_checkarg(handle,
                                                                     m,
                                                                     nnz,
                                                                     descr,
                                                                     csr_val,
                                                                     csr_row_ptr,
                                                                     csr_col_ind,
                                                                     fraction_to_color,
                                                                     ncolors,
                                                                     coloring,
                                                                     reordering,
                                                                     info);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrcolor_core(handle,
                                                           m,
                                                           nnz,
                                                           descr,
                                                           csr_val,
                                                           csr_row_ptr,
                                                           csr_col_ind,
                                                           fraction_to_color,
                                                           ncolors,
                                                           coloring,
                                                           reordering,
                                                           info));
        return rocsparse_status_success;
    }
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */
#define C_IMPL(NAME, T, U)                                                        \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,            \
                                     rocsparse_int             m,                 \
                                     rocsparse_int             nnz,               \
                                     const rocsparse_mat_descr descr,             \
                                     const T*                  csr_val,           \
                                     const rocsparse_int*      csr_row_ptr,       \
                                     const rocsparse_int*      csr_col_ind,       \
                                     const U*                  fraction_to_color, \
                                     rocsparse_int*            ncolors,           \
                                     rocsparse_int*            coloring,          \
                                     rocsparse_int*            reordering,        \
                                     rocsparse_mat_info        info)              \
    try                                                                           \
    {                                                                             \
        ROCSPARSE_ROUTINE_TRACE;                                                  \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrcolor_impl(handle,                \
                                                           m,                     \
                                                           nnz,                   \
                                                           descr,                 \
                                                           csr_val,               \
                                                           csr_row_ptr,           \
                                                           csr_col_ind,           \
                                                           fraction_to_color,     \
                                                           ncolors,               \
                                                           coloring,              \
                                                           reordering,            \
                                                           info));                \
        return rocsparse_status_success;                                          \
    }                                                                             \
    catch(...)                                                                    \
    {                                                                             \
        RETURN_ROCSPARSE_EXCEPTION();                                             \
    }

C_IMPL(rocsparse_scsrcolor, float, float);
C_IMPL(rocsparse_dcsrcolor, double, double);
C_IMPL(rocsparse_ccsrcolor, rocsparse_float_complex, float);
C_IMPL(rocsparse_zcsrcolor, rocsparse_double_complex, double);

#undef C_IMPL
