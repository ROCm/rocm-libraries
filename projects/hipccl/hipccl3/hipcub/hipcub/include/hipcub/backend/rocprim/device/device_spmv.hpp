/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2017-2021, Advanced Micro Devices, Inc.  All rights reserved.
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

#include "../iterator/tex_ref_input_iterator.hpp"

BEGIN_HIPCUB_NAMESPACE

class DeviceSpmv
{

public:

template <
    typename        ValueT,              ///< Matrix and vector value type
    typename        OffsetT>             ///< Signed integer type for sequence offsets
struct SpmvParams
{
    ValueT*         d_values;            ///< Pointer to the array of \p num_nonzeros values of the corresponding nonzero elements of matrix <b>A</b>.
    OffsetT*        d_row_end_offsets;   ///< Pointer to the array of \p m offsets demarcating the end of every row in \p d_column_indices and \p d_values
    OffsetT*        d_column_indices;    ///< Pointer to the array of \p num_nonzeros column-indices of the corresponding nonzero elements of matrix <b>A</b>.  (Indices are zero-valued.)
    ValueT*         d_vector_x;          ///< Pointer to the array of \p num_cols values corresponding to the dense input vector <em>x</em>
    ValueT*         d_vector_y;          ///< Pointer to the array of \p num_rows values corresponding to the dense output vector <em>y</em>
    int             num_rows;            ///< Number of rows of matrix <b>A</b>.
    int             num_cols;            ///< Number of columns of matrix <b>A</b>.
    int             num_nonzeros;        ///< Number of nonzero elements of matrix <b>A</b>.
    ValueT          alpha;               ///< Alpha multiplicand
    ValueT          beta;                ///< Beta addend-multiplicand

    ::hipcub::TexRefInputIterator<ValueT, 66778899, OffsetT>  t_vector_x;
};

static constexpr uint32_t CsrMVKernel_MaxThreads = 256;

template <typename ValueT>
static __global__ void
CsrMVKernel(SpmvParams<ValueT, int> spmv_params)
{
    __shared__ ValueT partial;

    const int32_t row_id = hipBlockIdx_x;

    if(hipThreadIdx_x == 0)
    {
        partial = spmv_params.beta * spmv_params.d_vector_y[row_id];
    }
    __syncthreads();

    int32_t row_offset = (row_id == 0) ? (0) : (spmv_params.d_row_end_offsets[row_id - 1]);
    for(uint32_t thread_offset = 0; thread_offset < spmv_params.num_cols / hipBlockDim_x; thread_offset++)
    {
        int32_t offset = row_offset + thread_offset * hipBlockDim_x + hipThreadIdx_x;

        if(offset < spmv_params.d_row_end_offsets[row_id])
        {
            ValueT t_value =
                spmv_params.alpha *
                spmv_params.d_values[offset] *
                spmv_params.d_vector_x[spmv_params.d_column_indices[offset]];

            atomicAdd(&partial, t_value);

            __syncthreads();

            if(hipThreadIdx_x == 0)
            {
                spmv_params.d_vector_y[row_id] = partial;
            }
        }
    }
}

template <typename ValueT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t CsrMV(
        void*               d_temp_storage,                     ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t&             temp_storage_bytes,                 ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        ValueT*             d_values,                           ///< [in] Pointer to the array of \p num_nonzeros values of the corresponding nonzero elements of matrix <b>A</b>.
        int*                d_row_offsets,                      ///< [in] Pointer to the array of \p m + 1 offsets demarcating the start of every row in \p d_column_indices and \p d_values (with the final entry being equal to \p num_nonzeros)
        int*                d_column_indices,                   ///< [in] Pointer to the array of \p num_nonzeros column-indices of the corresponding nonzero elements of matrix <b>A</b>.  (Indices are zero-valued.)
        ValueT*             d_vector_x,                         ///< [in] Pointer to the array of \p num_cols values corresponding to the dense input vector <em>x</em>
        ValueT*             d_vector_y,                         ///< [out] Pointer to the array of \p num_rows values corresponding to the dense output vector <em>y</em>
        int                 num_rows,                           ///< [in] number of rows of matrix <b>A</b>.
        int                 num_cols,                           ///< [in] number of columns of matrix <b>A</b>.
        int                 num_nonzeros,                       ///< [in] number of nonzero elements of matrix <b>A</b>.
        hipStream_t         stream                  = 0,        ///< [in] <b>[optional]</b> hip stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous       = false)    ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        SpmvParams<ValueT, int> spmv_params;
        spmv_params.d_values             = d_values;
        spmv_params.d_row_end_offsets    = d_row_offsets + 1;
        spmv_params.d_column_indices     = d_column_indices;
        spmv_params.d_vector_x           = d_vector_x;
        spmv_params.d_vector_y           = d_vector_y;
        spmv_params.num_rows             = num_rows;
        spmv_params.num_cols             = num_cols;
        spmv_params.num_nonzeros         = num_nonzeros;
        spmv_params.alpha                = 1.0;
        spmv_params.beta                 = 0.0;

        hipError_t status;
        if(d_temp_storage == nullptr)
        {
            // Make sure user won't try to allocate 0 bytes memory, because
            // hipMalloc will return nullptr when size is zero.
            temp_storage_bytes = 4;
            return hipError_t(0);
        }
        else
        {
            size_t block_size = min(num_cols, DeviceSpmv::CsrMVKernel_MaxThreads);
            size_t grid_size = num_rows;
            CsrMVKernel<<<grid_size, block_size, 0, stream>>>(spmv_params);
            status = hipGetLastError();
        }
        return status;
    }
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_CUB_DEVICE_DEVICE_SELECT_HPP_

