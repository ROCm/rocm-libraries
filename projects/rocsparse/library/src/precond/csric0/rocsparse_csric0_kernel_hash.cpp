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

#include "rocsparse_csric0_kernel_hash.hpp"
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
    ROCSPARSE_DEVICE_ILF void csric0_device_hash(J m,
                                                 const I* __restrict__ csr_row_ptr,
                                                 const J* __restrict__ csr_col_ind,
                                                 T* __restrict__ csr_val,
                                                 const I* __restrict__ csr_diag_ind,
                                                 int32_t* __restrict__ done,
                                                 const J* __restrict__ map,
                                                 J* __restrict__ zero_pivot,
                                                 J* __restrict__ singular_pivot,
                                                 double               tol,
                                                 rocsparse_index_base idx_base)
    {
        static_assert(WFSIZE > 0 && (WFSIZE & (WFSIZE - 1)) == 0, "WFSIZE must be a power of two.");
        static_assert(BLOCKSIZE > 0, "BLOCKSIZE must be positive.");
        static_assert(BLOCKSIZE % WFSIZE == 0, "BLOCKSIZE must be a multiple of WFSIZE.");
        static_assert(HASH > 0 && (HASH & (HASH - 1)) == 0, "HASH must be a power of two.");
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

        const auto tol_sq = tol * tol;

        // Current row this wavefront is working on
        J row = map[idx];

        // Diagonal entry point of the current row
        I row_diag = csr_diag_ind[row];

        // Row entry point
        I row_begin = csr_row_ptr[row] - idx_base;
        I row_end   = csr_row_ptr[row + 1] - idx_base;

        // Row sum accumulator
        T sum = static_cast<T>(0);

        // Fill hash table
        // Loop over columns of current row and fill hash table with row dependencies
        // Each lane processes one entry
        for(I j = row_begin + lid; j < row_end; j += WFSIZE)
        {
            // Insert key into hash table
            J key = csr_col_ind[j];
            // Compute hash
            int32_t hash = (key * 103) & (WFSIZE * HASH - 1);

            // Hash operation
            while(true)
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

            // Beginning of the row that corresponds to local_col
            I local_begin = csr_row_ptr[local_col] - idx_base;

            // Diagonal entry point of row local_col
            I local_diag = csr_diag_ind[local_col];

            // Local row sum
            T local_sum = static_cast<T>(0);

            // Structural zero pivot, do not process this row
            if(local_diag == -1)
            {
                local_diag = row_diag - 1;
            }

            // Spin loop until dependency has been resolved
            while(!__hip_atomic_load(&done[local_col], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT))
                ;

            // Make sure updated csr_val is visible globally
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

            // Load diagonal entry
            T diag_val = csr_val[local_diag];
            //
            if(diag_val == static_cast<T>(0))
            {
                break;
            };

            // Compute reciprocal
            diag_val = static_cast<T>(1) / diag_val;

            // Loop over the row the current column index depends on
            // Each lane processes one entry
            for(I k = local_begin + lid; k < local_diag; k += WFSIZE)
            {
                // Get value from hash table
                J key = csr_col_ind[k];

                // Compute hash
                J hash = (key * 103) & (WFSIZE * HASH - 1);

                // Hash operation
                while(true)
                {
                    if(table[hash] == -1)
                    {
                        // No entry for the key, done
                        break;
                    }
                    else if(table[hash] == key)
                    {
                        // Entry found, do linear combination
                        I idx = data[hash];
                        local_sum
                            = rocsparse::fma(csr_val[k], rocsparse::conj(csr_val[idx]), local_sum);
                        break;
                    }
                    else
                    {
                        // Collision, compute new hash
                        hash = (hash + 1) & (WFSIZE * HASH - 1);
                    }
                }
            }

            // Accumulate row sum
            local_sum = rocsparse::wfreduce_sum<WFSIZE>(local_sum);

            // Last lane id computes the Cholesky factor and writes it to global memory
            if(lid == WFSIZE - 1)
            {
                local_val = (local_val - local_sum) * diag_val;
                sum       = rocsparse::fma(local_val, rocsparse::conj(local_val), sum);

                csr_val[j] = local_val;
            }
        }

        // Last lane processes the diagonal entry
        if(lid == WFSIZE - 1)
        {
            if((row_diag >= 0))
            {
                const T diag_val = csr_val[row_diag] - sum;

                // test for negative value and numerical small value
                if((rocsparse::real(diag_val) <= (tol_sq)) && (rocsparse::imag(diag_val) == 0))
                {
                    rocsparse::atomic_min(singular_pivot, (row + idx_base));
                }

                if((csr_val[row_diag] = rocsparse::sqrt(rocsparse::abs(diag_val)))
                   == static_cast<T>(0))
                {
                    rocsparse::atomic_min(zero_pivot, (row + idx_base));
                }
            }
        }

        if(lid == WFSIZE - 1)
        {
            // Last lane writes "we are done" flag
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
    void csric0_kernel_hash(J m,
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
                            int64_t            singular_pivot_stride,
                            rocsparse_datatype tolerance_datatype,
                            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(float, tolerance_32),
                            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(double, tolerance_64),
                            bool                 is_singular_tol_host_mode,
                            rocsparse_index_base idx_base)
    {
        ROCSPARSE_SCALAR_HOST_DEVICE_GET(is_singular_tol_host_mode, tolerance_32);
        ROCSPARSE_SCALAR_HOST_DEVICE_GET(is_singular_tol_host_mode, tolerance_64);
        const double tolerance
            = (tolerance_datatype == rocsparse_datatype_f64_r) ? tolerance_64 : tolerance_32;

        const auto batch_index = hipBlockIdx_y;
        rocsparse::csric0_device_hash<BLOCKSIZE, WFSIZE, HASH>(
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
            idx_base);
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              uint32_t HASH,
              typename T,
              typename I,
              typename J>
    static rocsparse_status csric0_kernel_hash_launch(rocsparse_handle      handle,
                                                      rocsparse_csric0_info csric0_info,
                                                      rocsparse_spmat_descr A,
                                                      size_t                buffer_size,
                                                      void*                 buffer)
    {
        auto trm_info = csric0_info->get(rocsparse_operation_none, rocsparse_fill_mode_lower);

        // done array
        int32_t* done_array = reinterpret_cast<int32_t*>(reinterpret_cast<char*>(buffer) + 256);
        const int64_t done_array_stride = A->rows;
        const dim3    csric0_blocks((A->rows * handle->wavefront_size - 1) / BLOCKSIZE + 1,
                                 A->batch_count);
        const dim3    csric0_threads(BLOCKSIZE);

        auto                         numeric_exact = csric0_info->get_singularity_numeric_exact();
        auto                         numeric_near  = csric0_info->get_singularity_numeric_near();
        const rocsparse_pointer_mode tolerance_pointer_mode
            = numeric_near->get_tolerance_pointer_mode();
        const rocsparse_datatype tolerance_datatype = numeric_near->get_tolerance_datatype();
        const float*             tolerance_pointer_32
            = reinterpret_cast<const float*>(numeric_near->get_tolerance_pointer());
        const double* tolerance_pointer_64
            = reinterpret_cast<const double*>(numeric_near->get_tolerance_pointer());

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csric0_kernel_hash<BLOCKSIZE, WFSIZE, HASH>),
            csric0_blocks,
            csric0_threads,
            0,
            handle->stream,
            static_cast<J>(A->rows),
            reinterpret_cast<const I*>(A->const_row_data),
            reinterpret_cast<const J*>(A->const_col_data),
            reinterpret_cast<T*>(A->val_data),
            A->batch_stride,
            reinterpret_cast<const I*>(trm_info->get_diag_ind()),
            done_array,
            done_array_stride,
            reinterpret_cast<const J*>(trm_info->get_row_map()),
            reinterpret_cast<J*>(numeric_exact->get_position()),
            numeric_exact->get_stride(),
            reinterpret_cast<J*>(numeric_near->get_position()),
            numeric_near->get_stride(),
            tolerance_datatype,
            ROCSPARSE_SCALAR_HOST_DEVICE_ARGUMENT(tolerance_pointer_mode, tolerance_pointer_32),
            ROCSPARSE_SCALAR_HOST_DEVICE_ARGUMENT(tolerance_pointer_mode, tolerance_pointer_64),
            (tolerance_pointer_mode == rocsparse_pointer_mode_host),
            A->descr->base);

        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, uint32_t HASH, typename T, typename I>
    static csric0_kernel_launch_t transform_j_type(const rocsparse_indextype value)
    {
        switch(value)
        {
        case rocsparse_indextype_i32:
        {
            return rocsparse::csric0_kernel_hash_launch<BLOCKSIZE, WF_SIZE, HASH, T, I, int32_t>;
        }
        case rocsparse_indextype_i64:
        {
            return rocsparse::csric0_kernel_hash_launch<BLOCKSIZE, WF_SIZE, HASH, T, I, int64_t>;
        }
        case deprecated_rocsparse_indextype_u16:
        {
            THROW_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                  "rocsparse_indextype_u16 not supported");
        }
        }

        THROW_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
    }

    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, uint32_t HASH, typename T, typename... P>
    static csric0_kernel_launch_t transform_i_type(const rocsparse_indextype value, P... p)
    {
        switch(value)
        {
        case rocsparse_indextype_i32:
        {
            return rocsparse::transform_j_type<BLOCKSIZE, WF_SIZE, HASH, T, int32_t>(
                std::forward<P>(p)...);
        }
        case rocsparse_indextype_i64:
        {
            return rocsparse::transform_j_type<BLOCKSIZE, WF_SIZE, HASH, T, int64_t>(
                std::forward<P>(p)...);
        }
        case deprecated_rocsparse_indextype_u16:
        {
            THROW_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                  "rocsparse_indextype_u16 not supported");
        }
        }
        THROW_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
    }

    template <uint32_t... V, typename... P>
    static csric0_kernel_launch_t transform_t_type(const rocsparse_datatype value, P... p)
    {
        switch(value)
        {

        case rocsparse_datatype_f32_r:
        {
            return rocsparse::transform_i_type<V..., float>(std::forward<P>(p)...);
        }

        case rocsparse_datatype_f32_c:
        {
            return rocsparse::transform_i_type<V..., rocsparse_float_complex>(
                std::forward<P>(p)...);
        }

        case rocsparse_datatype_f64_r:
        {
            return rocsparse::transform_i_type<V..., double>(std::forward<P>(p)...);
        }

        case rocsparse_datatype_f64_c:
        {
            return rocsparse::transform_i_type<V..., rocsparse_double_complex>(
                std::forward<P>(p)...);
        }

        case rocsparse_datatype_i32_r:
        case rocsparse_datatype_u32_r:
        case rocsparse_datatype_i8_r:
        case rocsparse_datatype_u8_r:
        case rocsparse_datatype_f16_r:
        case rocsparse_datatype_bf16_r:
        {
            std::stringstream sstr;
            sstr << rocsparse::enum_utils::to_string(value) << " not supported";
            THROW_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                  sstr.str().c_str());
        }
        }

        THROW_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
    }

    template <uint32_t... V, typename... P>
    static csric0_kernel_launch_t transform_mxnnz(const int32_t max_nnz, P... p)
    {
        if(max_nnz <= 32)
        {
            return rocsparse::transform_t_type<V..., 1>(std::forward<P>(p)...);
        }
        else if(max_nnz <= 64)
        {
            return rocsparse::transform_t_type<V..., 2>(std::forward<P>(p)...);
        }
        else if(max_nnz <= 128)
        {
            return rocsparse::transform_t_type<V..., 4>(std::forward<P>(p)...);
        }
        else if(max_nnz <= 256)
        {
            return rocsparse::transform_t_type<V..., 8>(std::forward<P>(p)...);
        }
        else if(max_nnz <= 512)
        {
            return rocsparse::transform_t_type<V..., 16>(std::forward<P>(p)...);
        }
        else
        {
            THROW_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                  "max_nnz > 512 is not supported");
        }
    }

}
rocsparse::csric0_kernel_launch_t rocsparse::find_csric0_kernel_hash_launch(
    rocsparse_handle handle, rocsparse_csric0_info csric0_info, rocsparse_const_spmat_descr A)
{
    auto       trm_info = csric0_info->get(rocsparse_operation_none, rocsparse_fill_mode_lower);
    const auto max_nnz  = trm_info->get_max_nnz();
    switch(handle->wavefront_size)
    {
    case 32:
    {
        return rocsparse::transform_mxnnz<256, 32>(max_nnz, A->data_type, A->row_type, A->col_type);
    }
    case 64:
    {
        return rocsparse::transform_mxnnz<256, 64>(max_nnz, A->data_type, A->row_type, A->col_type);
    }
    default:
    {
        THROW_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error,
                                              "invalid wavefront size");
    }
    }
}
