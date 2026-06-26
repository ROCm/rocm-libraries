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

#include "rocsparse_csrildlt0_kernel_hash.hpp"
#include "rocsparse-complex-types.h"
#include "rocsparse_common.hpp"
#include "rocsparse_floating_data_t.hpp"
#include "rocsparse_numeric_boost.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    // ILDLT(0) hash-based device kernel.
    //
    // For real types, computes A ≈ L D L^T (LDL-transpose).
    // For complex types, computes A ≈ L D L^H (LDL-conjugate-transpose, Hermitian).
    //
    // For row i:
    //   D_i = real(A_{ii} - sum_{k<i} |L_{ik}|^2 * D_k)   (D is always real for Hermitian A)
    //   L_{ij} = (A_{ij} - sum_{k<j} L_{ik} * D_k * conj(L_{jk})) / D_j   (off-diagonal)
    //
    // The diagonal D is stored as floating_data_t<T> (real scalar) in a separate dense array.
    // csr_val stores the strictly lower-triangular entries of L (unit diagonal implicit).
    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              uint32_t HASH,
              typename T,
              typename I,
              typename J>
    ROCSPARSE_DEVICE_ILF void csrildlt0_device_hash(J m,
                                                    const I* __restrict__ csr_row_ptr,
                                                    const J* __restrict__ csr_col_ind,
                                                    T* __restrict__ csr_val,
                                                    floating_data_t<T>* __restrict__ diag,
                                                    const I* __restrict__ csr_diag_ind,
                                                    int32_t* __restrict__ done,
                                                    const J* __restrict__ map,
                                                    J* __restrict__ zero_pivot,
                                                    J* __restrict__ singular_pivot,
                                                    double               tol,
                                                    rocsparse_index_base idx_base,
                                                    int                  boost,
                                                    double               boost_tol,
                                                    floating_data_t<T>   boost_val)
    {
        static_assert(WFSIZE > 0 && (WFSIZE & (WFSIZE - 1)) == 0, "WFSIZE must be a power of two.");
        static_assert(BLOCKSIZE > 0, "BLOCKSIZE must be positive.");
        static_assert(BLOCKSIZE % WFSIZE == 0, "BLOCKSIZE must be a multiple of WFSIZE.");
        static_assert(HASH > 0 && (HASH & (HASH - 1)) == 0, "HASH must be a power of two.");
        const auto lid = hipThreadIdx_x & (WFSIZE - 1);
        const auto wid = hipThreadIdx_x / WFSIZE;

        __shared__ J stable[BLOCKSIZE * HASH];
        __shared__ I sdata[BLOCKSIZE * HASH];

        // Pointer to each wavefront's shared data
        J* table = &stable[wid * WFSIZE * HASH];
        I* data  = &sdata[wid * WFSIZE * HASH];

        // Initialize hash table with -1
        for(uint32_t j = lid; j < WFSIZE * HASH; j += WFSIZE)
        {
            table[j] = -1;
        }

        __threadfence_block();

        const auto idx = hipBlockIdx_x * BLOCKSIZE / WFSIZE + wid;

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

        // Accumulate: sum_{k<i} |L_{ik}|^2 * D_k  (for diagonal update D_i, always real)
        floating_data_t<T> diag_sum = static_cast<floating_data_t<T>>(0);

        // Fill hash table with column indices of the current row
        for(I j = row_begin + lid; j < row_end; j += WFSIZE)
        {
            J       key  = csr_col_ind[j];
            int32_t hash = (key * 103) & (WFSIZE * HASH - 1);

            while(true)
            {
                if(table[hash] == key)
                {
                    break;
                }
                else if(rocsparse::atomic_cas(&table[hash], static_cast<J>(-1), key)
                        == static_cast<J>(-1))
                {
                    data[hash] = j;
                    break;
                }
                else
                {
                    hash = (hash + 1) & (WFSIZE * HASH - 1);
                }
            }
        }

        __threadfence_block();

        // Loop over strictly lower-triangular columns of current row (j < i)
        for(I j = row_begin; j < row_diag; ++j)
        {
            // Column index j (< row)
            J local_col = csr_col_ind[j] - idx_base;

            // Current value L_{row, local_col} (will be updated in-place)
            T local_val = csr_val[j];

            // Beginning of row local_col
            I local_begin = csr_row_ptr[local_col] - idx_base;

            // Diagonal position of row local_col
            I local_diag = csr_diag_ind[local_col];

            // Accumulate: sum_{k<local_col} L_{row,k} * D_k * L_{local_col,k}
            T local_sum = static_cast<T>(0);

            if(local_diag == -1)
            {
                local_diag = row_diag - 1;
            }

            // Spin until row local_col is done
            while(!__hip_atomic_load(&done[local_col], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT))
                ;

            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

            // Load D_{local_col} from the diagonal array (real scalar)
            floating_data_t<T> d_j = diag[local_col];

            if(d_j == static_cast<floating_data_t<T>>(0))
            {
                break;
            }

            // Reciprocal of D_{local_col} for computing L_{row, local_col}
            floating_data_t<T> inv_d_j = static_cast<floating_data_t<T>>(1) / d_j;

            // Loop over k < local_col to compute sum_{k<j} L_{row,k} * D_k * L_{j,k}
            for(I k = local_begin + lid; k < local_diag; k += WFSIZE)
            {
                J key  = csr_col_ind[k];
                J hash = (key * 103) & (WFSIZE * HASH - 1);

                while(true)
                {
                    if(table[hash] == -1)
                    {
                        break;
                    }
                    else if(table[hash] == key)
                    {
                        // L_{row,k} is at data[hash], L_{j,k} is at csr_val[k]
                        // D_k is at diag[key] (real scalar)
                        // Update: L_{row,k} * D_k * conj(L_{j,k})
                        I                  idx_row_k = data[hash];
                        floating_data_t<T> d_k       = diag[key - idx_base];
                        local_sum
                            = rocsparse::fma(csr_val[idx_row_k],
                                             static_cast<T>(d_k) * rocsparse::conj(csr_val[k]),
                                             local_sum);
                        break;
                    }
                    else
                    {
                        hash = (hash + 1) & (WFSIZE * HASH - 1);
                    }
                }
            }

            local_sum = rocsparse::wfreduce_sum<WFSIZE>(local_sum);

            if(lid == WFSIZE - 1)
            {
                // L_{row, local_col} = (A_{row, local_col} - sum) / D_{local_col}
                local_val  = (local_val - local_sum) * inv_d_j;
                csr_val[j] = local_val;
                // Accumulate |L_{row,j}|^2 * D_j = (re^2 + im^2) * D_j for the diagonal update (real)
                const floating_data_t<T> re_l = rocsparse::real(local_val);
                const floating_data_t<T> im_l = rocsparse::imag(local_val);
                diag_sum = rocsparse::fma(rocsparse::fma(re_l, re_l, im_l * im_l), d_j, diag_sum);
            }
        }

        // Last lane processes the diagonal entry
        if(lid == WFSIZE - 1)
        {
            if(row_diag >= 0)
            {
                // D_i = real(A_{ii}) - sum_{k<i} |L_{ik}|^2 * D_k
                // For Hermitian A, A_{ii} is real; take real part to handle floating-point noise.
                floating_data_t<T> d_i = rocsparse::real(csr_val[row_diag]) - diag_sum;

                // Check for singular pivot
                if(rocsparse::abs(d_i) <= tol)
                {
                    rocsparse::atomic_min(singular_pivot, (row + idx_base));
                }

                if(boost && rocsparse::abs(d_i) <= boost_tol)
                {
                    d_i = boost_val;
                }

                // Store D_i (real scalar); zero the diagonal slot in csr_val (unit diagonal)
                diag[row]         = d_i;
                csr_val[row_diag] = static_cast<T>(0);

                if(d_i == static_cast<floating_data_t<T>>(0))
                {
                    rocsparse::atomic_min(zero_pivot, (row + idx_base));
                }
            }
        }

        if(lid == WFSIZE - 1)
        {
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
    void csrildlt0_kernel_hash(J m,
                               const I* __restrict__ csr_row_ptr,
                               const J* __restrict__ csr_col_ind,
                               T* __restrict__ csr_val,
                               int64_t csr_val_stride,
                               floating_data_t<T>* __restrict__ diag,
                               int64_t diag_stride,
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
                               rocsparse_index_base idx_base,
                               int                  boost,
                               ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(float, boost_tol_32),
                               ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(double, boost_tol_64),
                               rocsparse_datatype boost_tol_datatype,
                               bool               is_boost_tol_host_mode,
                               ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(floating_data_t<T>, boost_val),
                               bool is_boost_val_host_mode)
    {
        ROCSPARSE_SCALAR_HOST_DEVICE_GET(is_singular_tol_host_mode, tolerance_32);
        ROCSPARSE_SCALAR_HOST_DEVICE_GET(is_singular_tol_host_mode, tolerance_64);
        const double tolerance
            = (tolerance_datatype == rocsparse_datatype_f64_r) ? tolerance_64 : tolerance_32;

        ROCSPARSE_SCALAR_HOST_DEVICE_GET_IF(boost
                                                && (boost_tol_datatype == rocsparse_datatype_f32_r),
                                            is_boost_tol_host_mode,
                                            boost_tol_32);
        ROCSPARSE_SCALAR_HOST_DEVICE_GET_IF(boost
                                                && (boost_tol_datatype == rocsparse_datatype_f64_r),
                                            is_boost_tol_host_mode,
                                            boost_tol_64);
        const double b_tol
            = (boost_tol_datatype == rocsparse_datatype_f64_r) ? boost_tol_64 : boost_tol_32;

        ROCSPARSE_SCALAR_HOST_DEVICE_GET_IF(boost, is_boost_val_host_mode, boost_val);

        const auto batch_index = hipBlockIdx_y;
        rocsparse::csrildlt0_device_hash<BLOCKSIZE, WFSIZE, HASH>(
            m,
            csr_row_ptr,
            csr_col_ind,
            csr_val + batch_index * csr_val_stride,
            diag + batch_index * diag_stride, // floating_data_t<T>* stride in real elements
            csr_diag_ind,
            done + batch_index * done_stride,
            map,
            zero_pivot + batch_index * zero_pivot_stride,
            singular_pivot + batch_index * singular_pivot_stride,
            tolerance,
            idx_base,
            boost,
            b_tol,
            boost_val);
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              uint32_t HASH,
              typename T,
              typename I,
              typename J>
    static rocsparse_status csrildlt0_kernel_hash_launch(rocsparse_handle         handle,
                                                         rocsparse_csrildlt0_info csrildlt0_info,
                                                         rocsparse_spmat_descr    A,
                                                         void*                    diag,
                                                         size_t                   buffer_size,
                                                         void*                    buffer)
    {
        auto trm_info = csrildlt0_info->get(rocsparse_operation_none, rocsparse_fill_mode_lower);

        int32_t* done_array = reinterpret_cast<int32_t*>(reinterpret_cast<char*>(buffer) + 256);
        const int64_t done_array_stride = A->rows;
        const dim3    csrildlt0_blocks((A->rows * handle->wavefront_size - 1) / BLOCKSIZE + 1,
                                    A->batch_count);
        const dim3    csrildlt0_threads(BLOCKSIZE);

        auto numeric_exact = csrildlt0_info->get_singularity_numeric_exact();
        auto numeric_near  = csrildlt0_info->get_singularity_numeric_near();
        const rocsparse_pointer_mode tolerance_pointer_mode
            = numeric_near->get_tolerance_pointer_mode();
        const rocsparse_datatype tolerance_datatype = numeric_near->get_tolerance_datatype();
        const float*             tolerance_pointer_32
            = reinterpret_cast<const float*>(numeric_near->get_tolerance_pointer());
        const double* tolerance_pointer_64
            = reinterpret_cast<const double*>(numeric_near->get_tolerance_pointer());

        // Boost parameters
        auto                     boost              = A->info->get_boost();
        const int                boost_enable       = boost->get_enable();
        const rocsparse_datatype boost_tol_datatype = boost->get_tol_datatype();
        const auto               boost_tol_size = rocsparse::datatype_sizeof(boost_tol_datatype);
        const rocsparse_pointer_mode boost_tol_pointer_mode = boost->get_tol_pointer_mode();
        const rocsparse_pointer_mode boost_val_pointer_mode = boost->get_val_pointer_mode();
        const float*                 boost_tol_32           = (boost_tol_size == sizeof(float))
                                                                  ? reinterpret_cast<const float*>(boost->get_tol())
                                                                  : nullptr;
        const double*                boost_tol_64           = (boost_tol_size == sizeof(double))
                                                                  ? reinterpret_cast<const double*>(boost->get_tol())
                                                                  : nullptr;
        const floating_data_t<T>*    boost_val_ptr
            = reinterpret_cast<const floating_data_t<T>*>(boost->get_val());

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrildlt0_kernel_hash<BLOCKSIZE, WFSIZE, HASH>),
            csrildlt0_blocks,
            csrildlt0_threads,
            0,
            handle->stream,
            static_cast<J>(A->rows),
            reinterpret_cast<const I*>(A->const_row_data),
            reinterpret_cast<const J*>(A->const_col_data),
            reinterpret_cast<T*>(A->val_data),
            A->batch_stride,
            reinterpret_cast<floating_data_t<T>*>(diag),
            static_cast<int64_t>(A->rows),
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
            A->descr->base,
            boost_enable,
            ROCSPARSE_SCALAR_HOST_DEVICE_PERMISSIVE_ARGUMENT(boost_tol_pointer_mode, boost_tol_32),
            ROCSPARSE_SCALAR_HOST_DEVICE_PERMISSIVE_ARGUMENT(boost_tol_pointer_mode, boost_tol_64),
            boost_tol_datatype,
            (boost_tol_pointer_mode == rocsparse_pointer_mode_host),
            ROCSPARSE_SCALAR_HOST_DEVICE_PERMISSIVE_ARGUMENT(boost_val_pointer_mode, boost_val_ptr),
            (boost_val_pointer_mode == rocsparse_pointer_mode_host));

        return rocsparse_status_success;
    }

    // Instantiate for all HASH sizes (1,2,4,8,16) and type combos.
    // HASH is selected at runtime based on max_nnz from analysis (like csric0).
#define INST_HASH_LAUNCH(WF, HASH, T, I, J)                                         \
    template rocsparse_status csrildlt0_kernel_hash_launch<256, WF, HASH, T, I, J>( \
        rocsparse_handle, rocsparse_csrildlt0_info, rocsparse_spmat_descr, void*, size_t, void*)

#define INST_ALL_HASHES(WF, T, I, J)  \
    INST_HASH_LAUNCH(WF, 1, T, I, J); \
    INST_HASH_LAUNCH(WF, 2, T, I, J); \
    INST_HASH_LAUNCH(WF, 4, T, I, J); \
    INST_HASH_LAUNCH(WF, 8, T, I, J); \
    INST_HASH_LAUNCH(WF, 16, T, I, J)

    INST_ALL_HASHES(64, float, int32_t, int32_t);
    INST_ALL_HASHES(32, float, int32_t, int32_t);
    INST_ALL_HASHES(64, double, int32_t, int32_t);
    INST_ALL_HASHES(32, double, int32_t, int32_t);
    INST_ALL_HASHES(64, rocsparse_float_complex, int32_t, int32_t);
    INST_ALL_HASHES(32, rocsparse_float_complex, int32_t, int32_t);
    INST_ALL_HASHES(64, rocsparse_double_complex, int32_t, int32_t);
    INST_ALL_HASHES(32, rocsparse_double_complex, int32_t, int32_t);
    INST_ALL_HASHES(64, float, int64_t, int32_t);
    INST_ALL_HASHES(32, float, int64_t, int32_t);
    INST_ALL_HASHES(64, double, int64_t, int32_t);
    INST_ALL_HASHES(32, double, int64_t, int32_t);
    INST_ALL_HASHES(64, rocsparse_float_complex, int64_t, int32_t);
    INST_ALL_HASHES(32, rocsparse_float_complex, int64_t, int32_t);
    INST_ALL_HASHES(64, rocsparse_double_complex, int64_t, int32_t);
    INST_ALL_HASHES(32, rocsparse_double_complex, int64_t, int32_t);
    INST_ALL_HASHES(64, float, int64_t, int64_t);
    INST_ALL_HASHES(32, float, int64_t, int64_t);
    INST_ALL_HASHES(64, double, int64_t, int64_t);
    INST_ALL_HASHES(32, double, int64_t, int64_t);
    INST_ALL_HASHES(64, rocsparse_float_complex, int64_t, int64_t);
    INST_ALL_HASHES(32, rocsparse_float_complex, int64_t, int64_t);
    INST_ALL_HASHES(64, rocsparse_double_complex, int64_t, int64_t);
    INST_ALL_HASHES(32, rocsparse_double_complex, int64_t, int64_t);

#undef INST_ALL_HASHES
#undef INST_HASH_LAUNCH

    // Transform helpers: dispatch on index types, then data type, then max_nnz → HASH.
    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, uint32_t HASH, typename T, typename I>
    static csrildlt0_kernel_launch_t transform_j_type(const rocsparse_indextype value)
    {
        switch(value)
        {
        case rocsparse_indextype_i32:
            return rocsparse::csrildlt0_kernel_hash_launch<BLOCKSIZE, WF_SIZE, HASH, T, I, int32_t>;
        case rocsparse_indextype_i64:
            return rocsparse::csrildlt0_kernel_hash_launch<BLOCKSIZE, WF_SIZE, HASH, T, I, int64_t>;
        case rocsparse_indextype_u16:
            THROW_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                  "rocsparse_indextype_u16 not supported");
        }
        THROW_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
    }

    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, uint32_t HASH, typename T, typename... P>
    static csrildlt0_kernel_launch_t transform_i_type(const rocsparse_indextype value, P... p)
    {
        switch(value)
        {
        case rocsparse_indextype_i32:
            return rocsparse::transform_j_type<BLOCKSIZE, WF_SIZE, HASH, T, int32_t>(
                std::forward<P>(p)...);
        case rocsparse_indextype_i64:
            return rocsparse::transform_j_type<BLOCKSIZE, WF_SIZE, HASH, T, int64_t>(
                std::forward<P>(p)...);
        case rocsparse_indextype_u16:
            THROW_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                  "rocsparse_indextype_u16 not supported");
        }
        THROW_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
    }

    template <uint32_t... V, typename... P>
    static csrildlt0_kernel_launch_t transform_t_type(const rocsparse_datatype value, P... p)
    {
        switch(value)
        {
        case rocsparse_datatype_f32_r:
            return rocsparse::transform_i_type<V..., float>(std::forward<P>(p)...);
        case rocsparse_datatype_f64_r:
            return rocsparse::transform_i_type<V..., double>(std::forward<P>(p)...);
        case rocsparse_datatype_f32_c:
            return rocsparse::transform_i_type<V..., rocsparse_float_complex>(
                std::forward<P>(p)...);
        case rocsparse_datatype_f64_c:
            return rocsparse::transform_i_type<V..., rocsparse_double_complex>(
                std::forward<P>(p)...);
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
    static csrildlt0_kernel_launch_t transform_mxnnz(const int32_t max_nnz, P... p)
    {
        if(max_nnz <= 32)
            return rocsparse::transform_t_type<V..., 1>(std::forward<P>(p)...);
        else if(max_nnz <= 64)
            return rocsparse::transform_t_type<V..., 2>(std::forward<P>(p)...);
        else if(max_nnz <= 128)
            return rocsparse::transform_t_type<V..., 4>(std::forward<P>(p)...);
        else if(max_nnz <= 256)
            return rocsparse::transform_t_type<V..., 8>(std::forward<P>(p)...);
        else if(max_nnz <= 512)
            return rocsparse::transform_t_type<V..., 16>(std::forward<P>(p)...);
        else
            THROW_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                  "max_nnz > 512 is not supported");
    }

    // find_csrildlt0_kernel_hash_launch: select kernel based on device and matrix properties
    csrildlt0_kernel_launch_t
        find_csrildlt0_kernel_hash_launch(rocsparse_handle            handle,
                                          rocsparse_csrildlt0_info    csrildlt0_info,
                                          rocsparse_const_spmat_descr A)
    {
        auto trm_info = csrildlt0_info->get(rocsparse_operation_none, rocsparse_fill_mode_lower);
        const auto max_nnz = trm_info->get_max_nnz();
        switch(handle->wavefront_size)
        {
        case 32:
            return rocsparse::transform_mxnnz<256, 32>(
                max_nnz, A->data_type, A->row_type, A->col_type);
        case 64:
            return rocsparse::transform_mxnnz<256, 64>(
                max_nnz, A->data_type, A->row_type, A->col_type);
        default:
            THROW_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error,
                                                  "invalid wavefront size");
        }
    }

} // namespace rocsparse
