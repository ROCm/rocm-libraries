/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2017-2025, Advanced Micro Devices, Inc.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#ifndef HIPCUB_ROCPRIM_DEVICE_DEVICE_SPMV_HPP_
#define HIPCUB_ROCPRIM_DEVICE_DEVICE_SPMV_HPP_

#include "../../../config.hpp"
#include "../../../util_deprecated.hpp"

#include "../iterator/tex_obj_input_iterator.hpp"

#include "../util_sync.hpp"

#include <chrono>

BEGIN_HIPCUB_NAMESPACE

class HIPCUB_DEPRECATED_BECAUSE("Use the hipSPARSE library instead") DeviceSpmv
{

public:
    template<typename ValueT, ///< Matrix and vector value type
             typename OffsetT> ///< Signed integer type for sequence offsets
    struct HIPCUB_DEPRECATED_BECAUSE("Use the rocSPARSE library instead") SpmvParams
    {
        ValueT*
            d_values; ///< Pointer to the array of \p num_nonzeros values of the corresponding nonzero elements of matrix <b>A</b>.
        OffsetT*
            d_row_end_offsets; ///< Pointer to the array of \p m offsets demarcating the end of every row in \p d_column_indices and \p d_values
        OffsetT*
            d_column_indices; ///< Pointer to the array of \p num_nonzeros column-indices of the corresponding nonzero elements of matrix <b>A</b>.  (Indices are zero-valued.)
        ValueT*
            d_vector_x; ///< Pointer to the array of \p num_cols values corresponding to the dense input vector <em>x</em>
        ValueT*
            d_vector_y; ///< Pointer to the array of \p num_rows values corresponding to the dense output vector <em>y</em>
        int    num_rows; ///< Number of rows of matrix <b>A</b>.
        int    num_cols; ///< Number of columns of matrix <b>A</b>.
        int    num_nonzeros; ///< Number of nonzero elements of matrix <b>A</b>.
        ValueT alpha; ///< Alpha multiplicand
        ValueT beta; ///< Beta addend-multiplicand

        ::hipcub::TexObjInputIterator<ValueT, OffsetT> t_vector_x;
    };

static constexpr uint32_t CsrMVKernel_MaxThreads = 256;

template <typename ValueT>
static __global__ void
CsrMVKernel(SpmvParams<ValueT, int> spmv_params)
{
    __shared__ ValueT partial;

    const int32_t row_id = blockIdx.x;

    if(threadIdx.x == 0)
    {
        partial = spmv_params.beta * spmv_params.d_vector_y[row_id];
    }
    __syncthreads();

    int32_t row_offset = (row_id == 0) ? (0) : (spmv_params.d_row_end_offsets[row_id - 1]);
    for(uint32_t thread_offset = 0; thread_offset < spmv_params.num_cols / blockDim.x;
        thread_offset++)
    {
        int32_t offset = row_offset + thread_offset * blockDim.x + threadIdx.x;

        if(offset < spmv_params.d_row_end_offsets[row_id])
        {
            ValueT t_value =
                spmv_params.alpha *
                spmv_params.d_values[offset] *
                spmv_params.d_vector_x[spmv_params.d_column_indices[offset]];

            atomicAdd(&partial, t_value);

            __syncthreads();

            if(threadIdx.x == 0)
            {
                spmv_params.d_vector_y[row_id] = partial;
            }
        }
    }
}

template<typename ValueT>
HIPCUB_DEPRECATED_BECAUSE("Use the rocSPARSE library instead")
HIPCUB_RUNTIME_FUNCTION static hipError_t CsrMV(void*       d_temp_storage,
                                                size_t&     temp_storage_bytes,
                                                ValueT*     d_values,
                                                int*        d_row_offsets,
                                                int*        d_column_indices,
                                                ValueT*     d_vector_x,
                                                ValueT*     d_vector_y,
                                                int         num_rows,
                                                int         num_cols,
                                                int         num_nonzeros,
                                                hipStream_t stream = 0)
{
    HIPCUB_CLANG_SUPPRESS_DEPRECATED_PUSH
    SpmvParams<ValueT, int> spmv_params;
    HIPCUB_CLANG_SUPPRESS_DEPRECATED_POP
    spmv_params.d_values          = d_values;
    spmv_params.d_row_end_offsets = d_row_offsets + 1;
    spmv_params.d_column_indices  = d_column_indices;
    spmv_params.d_vector_x        = d_vector_x;
    spmv_params.d_vector_y        = d_vector_y;
    spmv_params.num_rows          = num_rows;
    spmv_params.num_cols          = num_cols;
    spmv_params.num_nonzeros      = num_nonzeros;
    spmv_params.alpha             = 1.0;
    spmv_params.beta              = 0.0;

    if(d_temp_storage == nullptr)
    {
        // Make sure user won't try to allocate 0 bytes memory, because
        // hipMalloc will return nullptr when size is zero.
        temp_storage_bytes = 4;
        return hipError_t(0);
    } else
    {
        size_t block_size = min(num_cols, static_cast<int>(DeviceSpmv::CsrMVKernel_MaxThreads));
        size_t grid_size  = num_rows;

        std::chrono::high_resolution_clock::time_point start;
        if HIPCUB_IF_CONSTEXPR(HIPCUB_DETAIL_DEBUG_SYNC_VALUE)
        {
            start = std::chrono::high_resolution_clock::now();
        }
        HIPCUB_CLANG_SUPPRESS_DEPRECATED_PUSH
        CsrMVKernel<<<grid_size, block_size, 0, stream>>>(spmv_params);
        HIPCUB_CLANG_SUPPRESS_DEPRECATED_PUSH
        HIPCUB_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("CsrMV", block_size * grid_size, start);
    }
    return hipSuccess;
}

template<typename ValueT>
HIPCUB_DEPRECATED_BECAUSE("Use the rocSPARSE library instead")
HIPCUB_RUNTIME_FUNCTION static hipError_t CsrMV(void*       d_temp_storage,
                                                size_t&     temp_storage_bytes,
                                                ValueT*     d_values,
                                                int*        d_row_offsets,
                                                int*        d_column_indices,
                                                ValueT*     d_vector_x,
                                                ValueT*     d_vector_y,
                                                int         num_rows,
                                                int         num_cols,
                                                int         num_nonzeros,
                                                hipStream_t stream,
                                                bool /*debug_synchronous*/)
{
    return CsrMV(d_temp_storage,
                 temp_storage_bytes,
                 d_values,
                 d_row_offsets,
                 d_column_indices,
                 d_vector_x,
                 d_vector_y,
                 num_rows,
                 num_cols,
                 num_nonzeros,
                 stream);
}
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_CUB_DEVICE_DEVICE_SELECT_HPP_

