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

#include "rocsparse_csrilu0_kernel_binsearch.hpp"
#include "rocsparse_common.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, bool SLEEP, typename T, typename I, typename J>
    ROCSPARSE_DEVICE_ILF void csrilu0_binsearch_device(J m_,
                                                       const I* __restrict__ csr_row_ptr,
                                                       const J* __restrict__ csr_col_ind,
                                                       T* __restrict__ csr_val,
                                                       const I* __restrict__ csr_diag_ind,
                                                       int* __restrict__ done,
                                                       const J* __restrict__ map,
                                                       J* __restrict__ zero_pivot,
                                                       J* __restrict__ singular_pivot,
                                                       double               tol,
                                                       rocsparse_index_base idx_base,
                                                       int                  boost,
                                                       double               boost_tol,
                                                       T                    boost_val)
    {
        const auto lid = hipThreadIdx_x & (WFSIZE - 1);
        const auto wid = hipThreadIdx_x / WFSIZE;

        const auto idx = hipBlockIdx_x * BLOCKSIZE / WFSIZE + wid;

        // Do not run out of bounds
        if(idx >= m_)
        {
            return;
        }

        // Current row this wavefront is working on
        J row = map[idx];

        // Diagonal entry point of the current row
        I row_diag = csr_diag_ind[row];

        // Row entry point
        I row_begin = csr_row_ptr[row] - idx_base;
        I row_end   = csr_row_ptr[row + 1] - idx_base;

        // Loop over column of current row
        for(I j = row_begin; j < row_diag; ++j)
        {
            // Column index currently being processes
            J local_col = csr_col_ind[j] - idx_base;

            // Corresponding value
            T local_val = csr_val[j];

            // End of the row that corresponds to local_col
            I local_end = csr_row_ptr[local_col + 1] - idx_base;

            // Diagonal entry point of row local_col
            I local_diag = csr_diag_ind[local_col];

            // Structural zero pivot, do not process this row
            if(local_diag == -1)
            {
                local_diag = local_end - 1;
            }

            // Spin loop until dependency has been resolved
            (void)rocsparse::spin_loop<SLEEP>(&done[local_col], __HIP_MEMORY_SCOPE_AGENT);

            // Make sure updated csr_val is visible
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

            // Load diagonal entry
            T diag_val = csr_val[local_diag];

            if(diag_val == static_cast<T>(0))
            {
                // Skip this row if it has a zero pivot
                break;
            }

            csr_val[j] = local_val = local_val / diag_val;

            // Loop over the row the current column index depends on
            // Each lane processes one entry
            I l = j + 1;
            for(I k = local_diag + 1 + lid; k < local_end; k += WFSIZE)
            {
                // Perform a binary search to find matching columns
                I r     = row_end - 1;
                I m     = (r + l) >> 1;
                J col_j = csr_col_ind[m];

                J col_k = csr_col_ind[k];

                // Binary search
                while(l < r)
                {
                    if(col_j < col_k)
                    {
                        l = m + 1;
                    }
                    else
                    {
                        r = m;
                    }

                    m     = (r + l) >> 1;
                    col_j = csr_col_ind[m];
                }

                // Check if a match has been found
                if(col_j == col_k)
                {
                    // If a match has been found, do ILU computation
                    csr_val[l] = rocsparse::fma(-local_val, csr_val[k], csr_val[l]);
                }
            }
        }

        __threadfence_block();

        const bool is_diag = (row_diag >= 0);
        if(is_diag)
        {
            const auto diag_val     = csr_val[row_diag];
            const auto abs_diag_val = rocsparse::abs(diag_val);
            if(boost)
            {
                const bool is_too_small = (abs_diag_val <= boost_tol);

                if(is_too_small)
                {
                    if(lid == 0)
                    {
                        csr_val[row_diag] = boost_val;
                    };
                };
            }
            else
            {

                const bool is_singular_pivot = (abs_diag_val <= tol);
                if(is_singular_pivot)
                {
                    if(lid == 0)
                    {
                        rocsparse::atomic_min(singular_pivot, (row + idx_base));
                    }
                }

                const bool is_zero_pivot = (diag_val == static_cast<T>(0));
                if(is_zero_pivot)
                {
                    if(lid == 0)
                    {
                        rocsparse::atomic_min(zero_pivot, (row + idx_base));
                    }
                }
            }
        }

        // Make sure updated csr_val is written to global memory
        __threadfence();

        if(lid == 0)
        {
            // First lane writes "we are done" flag
            __hip_atomic_store(&done[row], 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, bool SLEEP, typename T, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrilu0_strided_batched_kernel_binsearch(J m,
                                                  const I* __restrict__ csr_row_ptr,
                                                  const J* __restrict__ csr_col_ind,
                                                  T* __restrict__ csr_val,
                                                  int64_t csr_val_stride,
                                                  const I* __restrict__ csr_diag_ind,
                                                  int32_t* __restrict__ done,
                                                  int64_t done_stride,
                                                  const J* __restrict__ map,
                                                  J* __restrict__ zero_pivot,
                                                  int64_t zero_pivot_stride,
                                                  J* __restrict__ singular_pivot,
                                                  int64_t              singular_pivot_stride,
                                                  double               tol,
                                                  rocsparse_index_base idx_base,
                                                  int                  boost_enable,
                                                  size_t               boost_tol_size,
                                                  ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(float,
                                                                                      boost_tol_32),
                                                  ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(double,
                                                                                      boost_tol_64),
                                                  ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, boost_val),
                                                  bool is_host_mode)
    {
        const auto i = hipBlockIdx_y;
        ROCSPARSE_DEVICE_HOST_SCALAR_GET_IF(boost_enable, boost_tol_32);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET_IF(boost_enable, boost_tol_64);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET_IF(boost_enable, boost_val);
        const double boost_tol = (boost_tol_size == sizeof(double)) ? boost_tol_64 : boost_tol_32;
        rocsparse::csrilu0_binsearch_device<BLOCKSIZE, WFSIZE, SLEEP, T, I, J>(
            m,
            csr_row_ptr,
            csr_col_ind,
            csr_val + i * csr_val_stride,
            csr_diag_ind,
            done + i * done_stride,
            map,
            zero_pivot + i * zero_pivot_stride,
            singular_pivot + i * singular_pivot_stride,
            tol,
            idx_base,
            boost_enable,
            boost_tol,
            boost_val);
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, bool SLEEP, typename T, typename I, typename J>
    rocsparse_status
        csrilu0_strided_batched_kernel_binsearch_launch(rocsparse_handle handle,
                                                        int64_t          batch_count,
                                                        int64_t          m,
                                                        const void* __restrict__ csr_row_ptr,
                                                        const void* __restrict__ csr_col_ind,
                                                        void* __restrict__ csr_val,
                                                        int64_t csr_val_stride,
                                                        const void* __restrict__ csr_diag_ind,
                                                        int32_t* __restrict__ done,
                                                        int64_t done_stride,
                                                        const void* __restrict__ map,
                                                        void* __restrict__ zero_pivot,
                                                        int64_t zero_pivot_stride,
                                                        void* __restrict__ singular_pivot,
                                                        int64_t              singular_pivot_stride,
                                                        double               tol,
                                                        rocsparse_index_base idx_base,
                                                        int                  boost_enable,
                                                        size_t               boost_tol_size,
                                                        const void*          boost_tol,
                                                        const void*          boost_val_)
    {
        dim3         csrilu0_blocks((m * handle->wavefront_size - 1) / BLOCKSIZE + 1, batch_count);
        dim3         csrilu0_threads(BLOCKSIZE);
        const T*     boost_val = reinterpret_cast<const T*>(boost_val_);
        const float* boost_tol_32
            = reinterpret_cast<const float*>((boost_enable) ? boost_tol : nullptr);
        const double* boost_tol_64
            = reinterpret_cast<const double*>((boost_enable) ? boost_tol : nullptr);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::
                 csrilu0_strided_batched_kernel_binsearch<BLOCKSIZE, WFSIZE, SLEEP, T, I, J>),
            csrilu0_blocks,
            csrilu0_threads,
            0,
            handle->stream,
            static_cast<J>(m),
            reinterpret_cast<const I*>(csr_row_ptr),
            reinterpret_cast<const J*>(csr_col_ind),
            reinterpret_cast<T*>(csr_val),
            csr_val_stride,
            reinterpret_cast<const I*>(csr_diag_ind),
            done,
            done_stride,
            reinterpret_cast<const J*>(map),
            reinterpret_cast<J*>(zero_pivot),
            zero_pivot_stride,
            reinterpret_cast<J*>(singular_pivot),
            singular_pivot_stride,
            tol,
            idx_base,
            boost_enable,
            boost_tol_size,
            ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_tol_32),
            ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_tol_64),
            ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_val),
            handle->pointer_mode == rocsparse_pointer_mode_host);
        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              bool     SLEEP,
              typename T,
              typename I,
              typename... P>
    static launch_csrilu0_kernel_binsearch_t find_csrilu0_kernel_binsearch_j(rocsparse_indextype j,
                                                                             P... p)
    {
        return (j == rocsparse_indextype_i32)
                   ? rocsparse::csrilu0_strided_batched_kernel_binsearch_launch<BLOCKSIZE,
                                                                                WF_SIZE,
                                                                                SLEEP,
                                                                                T,
                                                                                I,
                                                                                int32_t>
               : (j == rocsparse_indextype_i64)
                   ? rocsparse::csrilu0_strided_batched_kernel_binsearch_launch<BLOCKSIZE,
                                                                                WF_SIZE,
                                                                                SLEEP,
                                                                                T,
                                                                                I,
                                                                                int64_t>
                   : nullptr;
    }

    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, bool SLEEP, typename T, typename... P>
    static launch_csrilu0_kernel_binsearch_t find_csrilu0_kernel_binsearch_i(rocsparse_indextype i,
                                                                             P... p)
    {
        return (i == rocsparse_indextype_i32) ? rocsparse::
                       find_csrilu0_kernel_binsearch_j<BLOCKSIZE, WF_SIZE, SLEEP, T, int32_t>(p...)
               : (i == rocsparse_indextype_i64) ? rocsparse::
                       find_csrilu0_kernel_binsearch_j<BLOCKSIZE, WF_SIZE, SLEEP, T, int64_t>(p...)
                                                : nullptr;
    }

    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, bool SLEEP, typename... P>
    static launch_csrilu0_kernel_binsearch_t
        find_launch_csrilu0_kernel_binsearch_t(rocsparse_datatype i, P... p)
    {
        return (i == rocsparse_datatype_f32_r)
                   ? rocsparse::find_csrilu0_kernel_binsearch_i<BLOCKSIZE, WF_SIZE, SLEEP, float>(
                       p...)
               : (i == rocsparse_datatype_f32_c)
                   ? rocsparse::find_csrilu0_kernel_binsearch_i<BLOCKSIZE,
                                                                WF_SIZE,
                                                                SLEEP,
                                                                rocsparse_float_complex>(p...)
               : (i == rocsparse_datatype_f64_c)
                   ? rocsparse::find_csrilu0_kernel_binsearch_i<BLOCKSIZE,
                                                                WF_SIZE,
                                                                SLEEP,
                                                                rocsparse_double_complex>(p...)
               : (i == rocsparse_datatype_f64_r)
                   ? rocsparse::find_csrilu0_kernel_binsearch_i<BLOCKSIZE, WF_SIZE, SLEEP, double>(
                       p...)
                   : nullptr;
    }

}

rocsparse::launch_csrilu0_kernel_binsearch_t
    rocsparse::find_launch_csrilu0_kernel_binsearch(uint32_t            blocksize,
                                                    uint32_t            wfsize,
                                                    bool                sleep,
                                                    rocsparse_datatype  t_type,
                                                    rocsparse_indextype i_type,
                                                    rocsparse_indextype j_type)
{
    if(blocksize == 256 && (wfsize == 32) && (sleep == false))
    {
        return rocsparse::find_launch_csrilu0_kernel_binsearch_t<256, 32, false>(
            t_type, i_type, j_type);
    }

    else if(blocksize == 256 && (wfsize == 64) && (sleep == false))
    {
        return rocsparse::find_launch_csrilu0_kernel_binsearch_t<256, 64, false>(
            t_type, i_type, j_type);
    }
    else if(blocksize == 256 && (wfsize == 64) && (sleep == true))
    {
        return rocsparse::find_launch_csrilu0_kernel_binsearch_t<256, 64, true>(
            t_type, i_type, j_type);
    }
    else
    {
        return nullptr;
    }
}
