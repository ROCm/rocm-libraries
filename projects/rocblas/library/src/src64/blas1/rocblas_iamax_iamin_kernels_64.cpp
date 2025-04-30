/* ************************************************************************
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#include "blas1/rocblas_iamax_iamin.hpp" // rocblas_int API called

#include "rocblas_iamax_iamin_kernels_64.hpp"

/*! \brief

    \details
    rocblas_internal_iamax_iamin_launcher computes a reduction over multiple vectors x_i
              Template parameters allow threads per block, data, and specific phase kernel overrides
              At least two kernels are needed to finish the reduction
              kernel 1 write partial result per thread block in workspace, blocks partial results
              kernel 2 gathers all the partial result in workspace and finishes the final reduction.
    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.
    @param[in]
    n         rocblas_int
              number of elements in each vector x_i
    @param[in]
    x         pointer to the first vector x_i on the GPU.
    @param[in]
    shiftx    rocblas_int
              specifies a base offset increment for the start of each x_i.
    @param[in]
    incx      rocblas_int
              specifies the increment for the elements of each x_i.
    @param[in]
    stridex   rocblas_int
              specifies the pointer increment between batches for x. stridex must be >= n*incx
    @param[in]
    batch_count rocblas_int
              number of instances in the batch
    @param[out]
    workspace To*
              temporary GPU buffer for inidividual block results for each batch
              and results buffer in case result pointer is to host memory
              Size must be (blocks+1)*batch_count*sizeof(To)
    @param[out]
    result
              pointers to array of batch_count size for results. either on the host CPU or device GPU.
              return is 0.0 if n, incx<=0.
    ********************************************************************/
template <typename API_INT,
          int NB,
          typename FETCH,
          typename REDUCE,
          typename TPtrX,
          typename To,
          typename Tr>
rocblas_status rocblas_internal_iamax_iamin_launcher_64(rocblas_handle handle,
                                                        API_INT        n_64,
                                                        TPtrX          x,
                                                        rocblas_stride shiftx,
                                                        API_INT        incx_64,
                                                        rocblas_stride stridex,
                                                        API_INT        batch_count_64,
                                                        To*            workspace,
                                                        Tr*            results)
{

    for(int64_t b_base = 0; b_base < batch_count_64; b_base += c_i64_grid_YZ_chunk)
    {
        int32_t batch_count = int32_t(std::min(batch_count_64 - b_base, c_i64_grid_YZ_chunk));

        auto x_ptr = adjust_ptr_batch(x, b_base, stridex);

        Tr* output          = &results[b_base];
        To* partial_results = workspace;
        if(handle->pointer_mode == rocblas_pointer_mode_host)
        {
            output          = (Tr*)(workspace);
            partial_results = (To*)(output + batch_count); // move past output
        }

        int32_t n      = int32_t(std::min(n_64, c_i64_grid_X_chunk));
        int     blocks = rocblas_reduction_kernel_block_count(n, NB);

        ROCBLAS_LAUNCH_KERNEL((rocblas_iamax_iamin_kernel_part1_64<NB, FETCH, REDUCE>),
                              dim3(blocks, 1, batch_count),
                              dim3(NB),
                              0,
                              handle->get_stream(),
                              n_64,
                              x_ptr,
                              shiftx,
                              incx_64,
                              stridex,
                              partial_results);

        // reduce all n partial results within batch chunk
        ROCBLAS_LAUNCH_KERNEL((rocblas_iamax_iamin_kernel_part2_64<NB, REDUCE>),
                              dim3(batch_count),
                              dim3(NB),
                              0,
                              handle->get_stream(),
                              blocks,
                              partial_results,
                              output);

        if(handle->pointer_mode == rocblas_pointer_mode_host)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(&results[b_base],
                                               output,
                                               sizeof(Tr) * batch_count,
                                               hipMemcpyDeviceToHost,
                                               handle->get_stream()));
        }

    } // for chunk of batch_count

    if(handle->pointer_mode == rocblas_pointer_mode_host)
    {
        // sync here to match legacy BLAS
        hipStreamSynchronize(handle->get_stream());
    }

    return rocblas_status_success;
}

template <typename T, typename S>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_iamax_template_64(rocblas_handle               handle,
                                       int64_t                      n,
                                       const T*                     x,
                                       rocblas_stride               shiftx,
                                       int64_t                      incx,
                                       rocblas_stride               stridex,
                                       int64_t                      batch_count,
                                       int64_t*                     result,
                                       rocblas_index_64_value_t<S>* workspace)
{
    return rocblas_internal_iamax_iamin_launcher_64<int64_t,
                                                    ROCBLAS_IAMAX_NB,
                                                    rocblas_fetch_amax_amin_64<S>,
                                                    rocblas_reduce_amax_64>(
        handle, n, x, shiftx, incx, stridex, batch_count, workspace, result);
}

template <typename T, typename S>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_iamin_template_64(rocblas_handle               handle,
                                       int64_t                      n,
                                       const T*                     x,
                                       rocblas_stride               shiftx,
                                       int64_t                      incx,
                                       rocblas_stride               stridex,
                                       int64_t                      batch_count,
                                       int64_t*                     result,
                                       rocblas_index_64_value_t<S>* workspace)
{
    return rocblas_internal_iamax_iamin_launcher_64<int64_t,
                                                    ROCBLAS_IAMAX_NB,
                                                    rocblas_fetch_amax_amin_64<S>,
                                                    rocblas_reduce_amin_64>(
        handle, n, x, shiftx, incx, stridex, batch_count, workspace, result);
}

template <typename T, typename S>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_iamax_batched_template_64(rocblas_handle               handle,
                                               int64_t                      n,
                                               const T* const*              x,
                                               rocblas_stride               shiftx,
                                               int64_t                      incx,
                                               rocblas_stride               stridex,
                                               int64_t                      batch_count,
                                               int64_t*                     result,
                                               rocblas_index_64_value_t<S>* workspace)
{
    return rocblas_internal_iamax_iamin_launcher_64<int64_t,
                                                    ROCBLAS_IAMAX_NB,
                                                    rocblas_fetch_amax_amin_64<S>,
                                                    rocblas_reduce_amax_64>(
        handle, n, x, shiftx, incx, stridex, batch_count, workspace, result);
}

template <typename T, typename S>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_iamin_batched_template_64(rocblas_handle               handle,
                                               int64_t                      n,
                                               const T* const*              x,
                                               rocblas_stride               shiftx,
                                               int64_t                      incx,
                                               rocblas_stride               stridex,
                                               int64_t                      batch_count,
                                               int64_t*                     result,
                                               rocblas_index_64_value_t<S>* workspace)
{
    return rocblas_internal_iamax_iamin_launcher_64<int64_t,
                                                    ROCBLAS_IAMAX_NB,
                                                    rocblas_fetch_amax_amin_64<S>,
                                                    rocblas_reduce_amin_64>(
        handle, n, x, shiftx, incx, stridex, batch_count, workspace, result);
}

#ifdef INST_IAMAX_TEMPLATE_64
#error INST_IAMAX_TEMPLATE_64 IS ALREADY DEFINED
#endif

#define INST_IAMAX_TEMPLATE_64(T_, S_)                                                        \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                                  \
        rocblas_internal_iamax_template_64<T_, S_>(rocblas_handle                handle,      \
                                                   int64_t                       n,           \
                                                   const T_*                     x,           \
                                                   rocblas_stride                shiftx,      \
                                                   int64_t                       incx,        \
                                                   rocblas_stride                stridex,     \
                                                   int64_t                       batch_count, \
                                                   int64_t*                      workspace,   \
                                                   rocblas_index_64_value_t<S_>* result);

INST_IAMAX_TEMPLATE_64(float, float)
INST_IAMAX_TEMPLATE_64(double, double)
INST_IAMAX_TEMPLATE_64(rocblas_float_complex, float)
INST_IAMAX_TEMPLATE_64(rocblas_double_complex, double)

#undef INST_IAMAX_TEMPLATE_64

#ifdef INST_IAMAX_BATCHED_TEMPLATE_64
#error INST_IAMAX_BATCHED_TEMPLATE_64 IS ALREADY DEFINED
#endif

#define INST_IAMAX_BATCHED_TEMPLATE_64(T_, S_)                                           \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                             \
        rocblas_internal_iamax_batched_template_64<T_, S_>(rocblas_handle   handle,      \
                                                           int64_t          n,           \
                                                           const T_* const* x,           \
                                                           rocblas_stride   shiftx,      \
                                                           int64_t          incx,        \
                                                           rocblas_stride   stridex,     \
                                                           int64_t          batch_count, \
                                                           int64_t*         workspace,   \
                                                           rocblas_index_64_value_t<S_>* result);

INST_IAMAX_BATCHED_TEMPLATE_64(float, float)
INST_IAMAX_BATCHED_TEMPLATE_64(double, double)
INST_IAMAX_BATCHED_TEMPLATE_64(rocblas_float_complex, float)
INST_IAMAX_BATCHED_TEMPLATE_64(rocblas_double_complex, double)

#undef INST_IAMAX_BATCHED_TEMPLATE_64

#ifdef INST_IAMIN_TEMPLATE_64
#error INST_IAMIN_TEMPLATE_64 IS ALREADY DEFINED
#endif

#define INST_IAMIN_TEMPLATE_64(T_, S_)                                                        \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                                  \
        rocblas_internal_iamin_template_64<T_, S_>(rocblas_handle                handle,      \
                                                   int64_t                       n,           \
                                                   const T_*                     x,           \
                                                   rocblas_stride                shiftx,      \
                                                   int64_t                       incx,        \
                                                   rocblas_stride                stridex,     \
                                                   int64_t                       batch_count, \
                                                   int64_t*                      workspace,   \
                                                   rocblas_index_64_value_t<S_>* result);

INST_IAMIN_TEMPLATE_64(float, float)
INST_IAMIN_TEMPLATE_64(double, double)
INST_IAMIN_TEMPLATE_64(rocblas_float_complex, float)
INST_IAMIN_TEMPLATE_64(rocblas_double_complex, double)

#undef INST_IAMIN_TEMPLATE_64

#ifdef INST_IAMIN_BATCHED_TEMPLATE_64
#error INST_IAMIN_BATCHED_TEMPLATE_64 IS ALREADY DEFINED
#endif

#define INST_IAMIN_BATCHED_TEMPLATE_64(T_, S_)                                           \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                             \
        rocblas_internal_iamin_batched_template_64<T_, S_>(rocblas_handle   handle,      \
                                                           int64_t          n,           \
                                                           const T_* const* x,           \
                                                           rocblas_stride   shiftx,      \
                                                           int64_t          incx,        \
                                                           rocblas_stride   stridex,     \
                                                           int64_t          batch_count, \
                                                           int64_t*         workspace,   \
                                                           rocblas_index_64_value_t<S_>* result);

INST_IAMIN_BATCHED_TEMPLATE_64(float, float)
INST_IAMIN_BATCHED_TEMPLATE_64(double, double)
INST_IAMIN_BATCHED_TEMPLATE_64(rocblas_float_complex, float)
INST_IAMIN_BATCHED_TEMPLATE_64(rocblas_double_complex, double)

#undef INST_IAMIN_BATCHED_TEMPLATE_64
