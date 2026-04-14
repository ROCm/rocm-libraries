/*! \file */
/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_csrilu0_kernel_hash.hpp"
#include "rocsparse_common.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              uint32_t HASH,
              typename T,
              typename I,
              typename J>
    ROCSPARSE_DEVICE_ILF void csrilu0_device_hash(J m,
                                                  const I* __restrict__ csr_row_ptr,
                                                  const J* __restrict__ csr_col_ind,
                                                  T* __restrict__ csr_val,
                                                  const I* __restrict__ csr_diag_ind,
                                                  int32_t* __restrict__ done,
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

        __shared__ J stable[BLOCKSIZE * HASH];
        __shared__ I sdata[BLOCKSIZE * HASH];

        // Pointer to each wavefronts shared data
        J* table = &stable[wid * WFSIZE * HASH];
        I* data  = &sdata[wid * WFSIZE * HASH];

        // Initialize hash table with -1
        for(uint32_t j = lid; j < WFSIZE * HASH; j += WFSIZE)
        {
            table[j] = -1;
        }

        __threadfence_block();

        const auto idx = hipBlockIdx_x * BLOCKSIZE / WFSIZE + wid;

        // Do not run out of bounds
        if(idx >= m)
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

        // Fill hash table
        // Loop over columns of current row and fill hash table with row dependencies
        // Each lane processes one entry
        for(I j = row_begin + lid; j < row_end; j += WFSIZE)
        {
            // Insert key into hash table
            J key = csr_col_ind[j];
            // Compute hash
            J hash = (key * 103) & (WFSIZE * HASH - 1);

            // Hash operation
#pragma unroll 4
            for(uint32_t h = 0; h < WFSIZE * HASH; ++h)
            {
                if(table[hash] == key)
                {
                    // key is already inserted, done
                    break;
                }
                else if(rocsparse::atomic_cas(&table[hash], static_cast<J>(-1), key)
                        == static_cast<J>(-1))
                {
                    // inserted key into the table, done
                    data[hash] = j;
                    break;
                }
                else
                {
                    // collision, compute new hash
                    hash = (hash + 1) & (WFSIZE * HASH - 1);
                }
            }
        }

        __threadfence_block();

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
            while(!__hip_atomic_load(&done[local_col], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT))
                ;

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
            for(I k = local_diag + 1 + lid; k < local_end; k += WFSIZE)
            {
                // Get value from hash table
                J key = csr_col_ind[k];

                // Compute hash
                J hash = (key * 103) & (WFSIZE * HASH - 1);

                // Hash operation
#pragma unroll 4
                for(uint32_t h = 0; h < WFSIZE * HASH; ++h)
                {
                    if(table[hash] == -1)
                    {
                        // No entry for the key, done
                        break;
                    }
                    else if(table[hash] == key)
                    {
                        // Entry found, do ILU computation
                        I idx_data = data[hash];
                        csr_val[idx_data]
                            = rocsparse::fma(-local_val, csr_val[k], csr_val[idx_data]);
                        break;
                    }
                    else
                    {
                        // Collision, compute new hash
                        hash = (hash + 1) & (WFSIZE * HASH - 1);
                    }
                }
            }
        }

        // Make sure updated csr_val is written to global memory
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
                        csr_val[row_diag] = rocsparse::assign_ilu0_boost_value(diag_val, boost_val);
                        __threadfence(); // make sure this is written out before ready flag is set
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

    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              uint32_t HASH,
              typename T,
              typename I,
              typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrilu0_kernel_hash(J m,
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
                             int64_t singular_pivot_stride,
                             //
                             rocsparse_datatype tolerance_datatype,
                             ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(float, tolerance_32),
                             ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(double, tolerance_64),
                             bool is_singular_tol_host_mode,
                             //
                             rocsparse_index_base idx_base,
                             //
                             int    enable_boost,
                             size_t size_boost_tol,
                             ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(float, boost_tol_32),
                             ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(double, boost_tol_64),
                             bool is_tol_host_mode,
                             ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, boost_val),
                             bool is_val_host_mode)
    {
        const auto batch_index = hipBlockIdx_y;

        ROCSPARSE_SCALAR_HOST_DEVICE_GET_IF(
            enable_boost && (size_boost_tol == sizeof(float)), is_tol_host_mode, boost_tol_32);
        ROCSPARSE_SCALAR_HOST_DEVICE_GET_IF(
            enable_boost && (size_boost_tol == sizeof(double)), is_tol_host_mode, boost_tol_64);
        ROCSPARSE_SCALAR_HOST_DEVICE_GET_IF(enable_boost, is_val_host_mode, boost_val);

        const double boost_tol = (size_boost_tol == sizeof(double)) ? boost_tol_64 : boost_tol_32;

        ROCSPARSE_SCALAR_HOST_DEVICE_GET(is_singular_tol_host_mode, tolerance_32);
        ROCSPARSE_SCALAR_HOST_DEVICE_GET(is_singular_tol_host_mode, tolerance_64);

        const double tolerance
            = (tolerance_datatype == rocsparse_datatype_f64_r) ? tolerance_64 : tolerance_32;

        rocsparse::csrilu0_device_hash<BLOCKSIZE, WFSIZE, HASH>(
            m,
            csr_row_ptr,
            csr_col_ind,
            csr_val + batch_index * csr_val_stride,
            csr_diag_ind,
            done + batch_index * done_stride,
            map,
            zero_pivot + batch_index * zero_pivot_stride,
            singular_pivot + batch_index * singular_pivot_stride,
            tolerance,
            idx_base,
            enable_boost,
            boost_tol,
            boost_val);
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              uint32_t HASH,
              typename T,
              typename I,
              typename J>
    static rocsparse_status csrilu0_kernel_hash_launch(rocsparse_handle          handle,
                                                       rocsparse_csrilu0_info    csrilu0_info,
                                                       rocsparse_spmat_descr     A,
                                                       rocsparse::numeric_boost* boost,
                                                       size_t                    buffer_size,
                                                       void*                     buffer)
    {
        auto trm_info = csrilu0_info->get(rocsparse_operation_none, rocsparse_fill_mode_lower);

        // done array
        int32_t* done_array = reinterpret_cast<int32_t*>(reinterpret_cast<char*>(buffer) + 256);
        const int64_t done_array_stride = A->rows;
        const int64_t A_batch_count     = (A->batch_stride == 0) ? 1 : A->batch_count;

        const auto boost_enable           = boost->get_enable();
        const auto boost_tol_size         = rocsparse::datatype_sizeof(boost->get_tol_datatype());
        const auto boost_tol_pointer_mode = boost->get_tol_pointer_mode();
        const auto boost_val_pointer_mode = boost->get_val_pointer_mode();

        const float*  boost_tol_32 = (boost_tol_size == sizeof(float))
                                         ? reinterpret_cast<const float*>(boost->get_tol())
                                         : nullptr;
        const double* boost_tol_64 = (boost_tol_size == sizeof(double))
                                         ? reinterpret_cast<const double*>(boost->get_tol())
                                         : nullptr;
        const T*      boost_val    = reinterpret_cast<const T*>(boost->get_val());

        int64_t stride = A->columns_values_batch_stride;
        dim3 csrilu0_blocks((A->rows * handle->wavefront_size - 1) / BLOCKSIZE + 1, A_batch_count);
        dim3 csrilu0_threads(BLOCKSIZE);

        auto                         numeric_exact = csrilu0_info->get_singularity_numeric_exact();
        auto                         numeric_near  = csrilu0_info->get_singularity_numeric_near();
        const rocsparse_pointer_mode tolerance_pointer_mode
            = numeric_near->get_tolerance_pointer_mode();
        const rocsparse_datatype tolerance_datatype = numeric_near->get_tolerance_datatype();
        const float*             tolerance_pointer_32
            = reinterpret_cast<const float*>(numeric_near->get_tolerance_pointer());
        const double* tolerance_pointer_64
            = reinterpret_cast<const double*>(numeric_near->get_tolerance_pointer());

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrilu0_kernel_hash<BLOCKSIZE, WFSIZE, HASH>),
            csrilu0_blocks,
            csrilu0_threads,
            0,
            handle->stream,
            static_cast<J>(A->rows),
            reinterpret_cast<const I*>(A->const_row_data),
            reinterpret_cast<const J*>(A->const_col_data),
            reinterpret_cast<T*>(A->val_data),
            stride,
            reinterpret_cast<const I*>(trm_info->get_diag_ind()),
            done_array,
            done_array_stride,
            reinterpret_cast<const J*>(trm_info->get_row_map()),
            reinterpret_cast<J*>(numeric_exact->get_position()),
            numeric_exact->get_stride(),
            reinterpret_cast<J*>(numeric_near->get_position()),
            numeric_near->get_stride(),
            //
            tolerance_datatype,
            ROCSPARSE_SCALAR_HOST_DEVICE_ARGUMENT(tolerance_pointer_mode, tolerance_pointer_32),
            ROCSPARSE_SCALAR_HOST_DEVICE_ARGUMENT(tolerance_pointer_mode, tolerance_pointer_64),
            (tolerance_pointer_mode == rocsparse_pointer_mode_host),
            //
            A->descr->base,
            boost_enable,
            boost_tol_size,
            ROCSPARSE_SCALAR_HOST_DEVICE_PERMISSIVE_ARGUMENT(boost_tol_pointer_mode, boost_tol_32),
            ROCSPARSE_SCALAR_HOST_DEVICE_PERMISSIVE_ARGUMENT(boost_tol_pointer_mode, boost_tol_64),
            boost_tol_pointer_mode == rocsparse_pointer_mode_host,
            ROCSPARSE_SCALAR_HOST_DEVICE_PERMISSIVE_ARGUMENT(boost_val_pointer_mode, boost_val),
            boost_val_pointer_mode == rocsparse_pointer_mode_host);

        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              uint32_t HASH,
              typename T,
              typename I,
              typename... P>
    static csrilu0_kernel_launch_t transform_j_type(const rocsparse_indextype j, P... p)
    {
        return //
            (j == rocsparse_indextype_i32) //
                ? csrilu0_kernel_hash_launch<BLOCKSIZE,
                                             WF_SIZE,
                                             HASH,
                                             T,
                                             I,
                                             int32_t>
                : (j == rocsparse_indextype_i64) //
                      ? csrilu0_kernel_hash_launch<BLOCKSIZE, WF_SIZE, HASH, T, I, int64_t>
                      : nullptr;
    }

    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, uint32_t HASH, typename T, typename... P>
    static csrilu0_kernel_launch_t transform_i_type(const rocsparse_indextype i, P... p)
    {
        return //
            (i == rocsparse_indextype_i32) //
                ? rocsparse::transform_j_type<BLOCKSIZE, WF_SIZE, HASH, T, int32_t>(p...) //
                : (i == rocsparse_indextype_i64) //
                      ? rocsparse::transform_j_type<BLOCKSIZE, WF_SIZE, HASH, T, int64_t>(p...) //
                      : nullptr;
    }

    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, uint32_t HASH, typename... P>
    static csrilu0_kernel_launch_t transform_t_type(const rocsparse_datatype i, P... p)
    {
        return //
            (i == rocsparse_datatype_f32_r) //
                ? rocsparse::transform_i_type<BLOCKSIZE, WF_SIZE, HASH, float>(p...) //
                : (i == rocsparse_datatype_f32_c) //
                      ? rocsparse::transform_i_type<BLOCKSIZE,
                                                    WF_SIZE,
                                                    HASH,
                                                    rocsparse_float_complex>(p...) //
                      : (i == rocsparse_datatype_f64_c) //
                            ? rocsparse::transform_i_type<BLOCKSIZE,
                                                          WF_SIZE,
                                                          HASH,
                                                          rocsparse_double_complex>(p...) //
                            : (i == rocsparse_datatype_f64_r) //
                                  ? rocsparse::transform_i_type<BLOCKSIZE, WF_SIZE, HASH, double>(
                                      p...) //
                                  : nullptr;
    }

    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, typename... P>
    static csrilu0_kernel_launch_t transform_mxnnz(const int32_t max_nnz, P... p)
    {
        return //
            (max_nnz < WF_SIZE) //
                ? rocsparse::transform_t_type<BLOCKSIZE, WF_SIZE, 1>(p...) //
                : (max_nnz < WF_SIZE * 2) //
                      ? rocsparse::transform_t_type<BLOCKSIZE, WF_SIZE, 2>(p...) //
                      : (max_nnz < WF_SIZE * 4) //
                            ? rocsparse::transform_t_type<BLOCKSIZE, WF_SIZE, 4>(p...) //
                            : (max_nnz < WF_SIZE * 8) //
                                  ? rocsparse::transform_t_type<BLOCKSIZE, WF_SIZE, 8>(p...) //
                                  : (max_nnz < WF_SIZE * 16) //
                                        ? rocsparse::transform_t_type<BLOCKSIZE,
                                                                      WF_SIZE,
                                                                      16>(p...) //
                                        : nullptr;
    }

    template <uint32_t BLOCKSIZE, typename... P>
    static csrilu0_kernel_launch_t transform_wf(const int32_t i, P... p)
    {
        return //
            (i == 32) //
                ? rocsparse::transform_mxnnz<BLOCKSIZE, 32>(p...) //
                : (i == 64) //
                      ? rocsparse::transform_mxnnz<BLOCKSIZE, 64>(p...) //
                      : nullptr;
    }

}

rocsparse::csrilu0_kernel_launch_t rocsparse::find_csrilu0_kernel_hash_launch(
    rocsparse_handle handle, rocsparse_csrilu0_info csrilu0_info, rocsparse_const_spmat_descr A)
{
    auto trm_info = csrilu0_info->get(rocsparse_operation_none, rocsparse_fill_mode_lower);

    return rocsparse::transform_wf<256>(
        handle->wavefront_size, trm_info->get_max_nnz(), A->data_type, A->row_type, A->col_type);
}
