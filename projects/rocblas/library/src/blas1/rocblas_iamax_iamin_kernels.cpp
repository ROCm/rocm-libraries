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

#include "rocblas_iamax_iamin_kernels.hpp"

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
          rocblas_int NB,
          typename FETCH,
          typename REDUCE,
          typename TPtrX,
          typename To,
          typename Tr>
rocblas_status rocblas_internal_iamax_iamin_launcher(rocblas_handle handle,
                                                     API_INT        n,
                                                     TPtrX          x,
                                                     rocblas_stride shiftx,
                                                     API_INT        incx,
                                                     rocblas_stride stridex,
                                                     API_INT        batch_count,
                                                     To*            workspace,
                                                     Tr*            result)
{
    rocblas_int blocks = rocblas_reduction_kernel_block_count(n, NB);

    int batches = handle->getBatchGridDim((int)batch_count);

    ROCBLAS_LAUNCH_KERNEL((rocblas_iamax_iamin_kernel_part1<NB, FETCH, REDUCE>),
                          dim3(blocks, 1, batches),
                          dim3(NB),
                          0,
                          handle->get_stream(),
                          n,
                          blocks,
                          x,
                          shiftx,
                          incx,
                          stridex,
                          batch_count,
                          workspace);

    if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
        ROCBLAS_LAUNCH_KERNEL((rocblas_iamax_iamin_kernel_part2<NB, REDUCE>),
                              dim3(batch_count),
                              dim3(NB),
                              0,
                              handle->get_stream(),
                              blocks,
                              workspace,
                              result);
    }
    else
    {
        // If in host pointer mode, workspace is converted to Tr* and the result is
        // placed there, and then copied from device to host. If To is a class type,
        // it must be a standard layout type and its first member must be of type Tr.
        static_assert(std::is_standard_layout<To>{}, "To must be a standard layout type");

        bool reduceKernel = blocks > 1 || batch_count > 1;
        // result is in the beginning of workspace[0]+offset, and can be copied directly.
        size_t offset = reduceKernel ? size_t(batch_count) * blocks : 0;

        if(reduceKernel)
        {
            ROCBLAS_LAUNCH_KERNEL((rocblas_iamax_iamin_kernel_part2<NB, REDUCE>),
                                  dim3(batch_count),
                                  dim3(NB),
                                  0,
                                  handle->get_stream(),
                                  blocks,
                                  workspace,
                                  (Tr*)(workspace + offset));
        }

        RETURN_IF_HIP_ERROR(hipMemcpyAsync(result,
                                           workspace + offset,
                                           batch_count * sizeof(Tr),
                                           hipMemcpyDeviceToHost,
                                           handle->get_stream()));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->get_stream()));
    }
    return rocblas_status_success;
}

template <typename T, typename S>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_iamax_template(rocblas_handle            handle,
                                    rocblas_int               n,
                                    const T*                  x,
                                    rocblas_stride            shiftx,
                                    rocblas_int               incx,
                                    rocblas_stride            stridex,
                                    rocblas_int               batch_count,
                                    rocblas_int*              result,
                                    rocblas_index_value_t<S>* workspace)
{
    return rocblas_internal_iamax_iamin_launcher<rocblas_int,
                                                 ROCBLAS_IAMAX_NB,
                                                 rocblas_fetch_amax_amin<S>,
                                                 rocblas_reduce_amax>(
        handle, n, x, shiftx, incx, stridex, batch_count, workspace, result);
}

template <typename T, typename S>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_iamax_batched_template(rocblas_handle            handle,
                                            rocblas_int               n,
                                            const T* const*           x,
                                            rocblas_stride            shiftx,
                                            rocblas_int               incx,
                                            rocblas_stride            stridex,
                                            rocblas_int               batch_count,
                                            rocblas_int*              result,
                                            rocblas_index_value_t<S>* workspace)
{
    return rocblas_internal_iamax_iamin_launcher<rocblas_int,
                                                 ROCBLAS_IAMAX_NB,
                                                 rocblas_fetch_amax_amin<S>,
                                                 rocblas_reduce_amax>(
        handle, n, x, shiftx, incx, stridex, batch_count, workspace, result);
}

template <typename T, typename S>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_iamin_template(rocblas_handle            handle,
                                    rocblas_int               n,
                                    const T*                  x,
                                    rocblas_stride            shiftx,
                                    rocblas_int               incx,
                                    rocblas_stride            stridex,
                                    rocblas_int               batch_count,
                                    rocblas_int*              result,
                                    rocblas_index_value_t<S>* workspace)
{
    return rocblas_internal_iamax_iamin_launcher<rocblas_int,
                                                 ROCBLAS_IAMAX_NB,
                                                 rocblas_fetch_amax_amin<S>,
                                                 rocblas_reduce_amin>(
        handle, n, x, shiftx, incx, stridex, batch_count, workspace, result);
}

template <typename T, typename S>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_iamin_batched_template(rocblas_handle            handle,
                                            rocblas_int               n,
                                            const T* const*           x,
                                            rocblas_stride            shiftx,
                                            rocblas_int               incx,
                                            rocblas_stride            stridex,
                                            rocblas_int               batch_count,
                                            rocblas_int*              result,
                                            rocblas_index_value_t<S>* workspace)
{
    return rocblas_internal_iamax_iamin_launcher<rocblas_int,
                                                 ROCBLAS_IAMAX_NB,
                                                 rocblas_fetch_amax_amin<S>,
                                                 rocblas_reduce_amin>(
        handle, n, x, shiftx, incx, stridex, batch_count, workspace, result);
}

// clang-format off
#ifdef INSTANTIATE_IAMAX_TEMPLATE
#error INSTANTIATE_IAMAX_TEMPLATE IS ALREADY DEFINED
#endif

#define INSTANTIATE_IAMAX_TEMPLATE(T_, S_)                                            \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE                                         \
    rocblas_status rocblas_internal_iamax_template<T_, S_>(rocblas_handle    handle,  \
                                                          rocblas_int    n,           \
                                                          const T_*      x,           \
                                                          rocblas_stride shiftx,      \
                                                          rocblas_int    incx,        \
                                                          rocblas_stride stridex,     \
                                                          rocblas_int    batch_count, \
                                                          rocblas_int*   workspace,   \
                                                          rocblas_index_value_t<S_>* result);

INSTANTIATE_IAMAX_TEMPLATE(float, float)
INSTANTIATE_IAMAX_TEMPLATE(double, double)
INSTANTIATE_IAMAX_TEMPLATE(rocblas_float_complex, float)
INSTANTIATE_IAMAX_TEMPLATE(rocblas_double_complex, double)

#undef INSTANTIATE_IAMAX_TEMPLATE

#ifdef INSTANTIATE_IAMAX_BATCHED_TEMPLATE
#error INSTANTIATE_IAMAX_BATCHED_TEMPLATE IS ALREADY DEFINED
#endif

#define INSTANTIATE_IAMAX_BATCHED_TEMPLATE(T_, S_)                                           \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE                                                \
    rocblas_status rocblas_internal_iamax_batched_template<T_, S_>(rocblas_handle   handle,  \
                                                               rocblas_int      n,           \
                                                               const T_* const* x,           \
                                                               rocblas_stride   shiftx,      \
                                                               rocblas_int      incx,        \
                                                               rocblas_stride   stridex,     \
                                                               rocblas_int      batch_count, \
                                                               rocblas_int*     workspace,   \
                                                               rocblas_index_value_t<S_>* result);

INSTANTIATE_IAMAX_BATCHED_TEMPLATE(float, float)
INSTANTIATE_IAMAX_BATCHED_TEMPLATE(double, double)
INSTANTIATE_IAMAX_BATCHED_TEMPLATE(rocblas_float_complex, float)
INSTANTIATE_IAMAX_BATCHED_TEMPLATE(rocblas_double_complex, double)

#undef INSTANTIATE_IAMAX_BATCHED_TEMPLATE

#ifdef INSTANTIATE_IAMIN_TEMPLATE
#error INSTANTIATE_IAMIN_TEMPLATE IS ALREADY DEFINED
#endif

#define INSTANTIATE_IAMIN_TEMPLATE(T_, S_)                                            \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE                                         \
    rocblas_status rocblas_internal_iamin_template<T_, S_>(rocblas_handle    handle,  \
                                                          rocblas_int    n,           \
                                                          const T_*      x,           \
                                                          rocblas_stride shiftx,      \
                                                          rocblas_int    incx,        \
                                                          rocblas_stride stridex,     \
                                                          rocblas_int    batch_count, \
                                                          rocblas_int*   workspace,   \
                                                          rocblas_index_value_t<S_>* result);

INSTANTIATE_IAMIN_TEMPLATE(float, float)
INSTANTIATE_IAMIN_TEMPLATE(double, double)
INSTANTIATE_IAMIN_TEMPLATE(rocblas_float_complex, float)
INSTANTIATE_IAMIN_TEMPLATE(rocblas_double_complex, double)

#undef INSTANTIATE_IAMIN_TEMPLATE

#ifdef INSTANTIATE_IAMIN_BATCHED_TEMPLATE
#error INSTANTIATE_IAMIN_BATCHED_TEMPLATE IS ALREADY DEFINED
#endif

#define INSTANTIATE_IAMIN_BATCHED_TEMPLATE(T_, S_)                                           \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE                                                \
    rocblas_status rocblas_internal_iamin_batched_template<T_, S_>(rocblas_handle   handle,  \
                                                               rocblas_int      n,           \
                                                               const T_* const* x,           \
                                                               rocblas_stride   shiftx,      \
                                                               rocblas_int      incx,        \
                                                               rocblas_stride   stridex,     \
                                                               rocblas_int      batch_count, \
                                                               rocblas_int*     workspace,   \
                                                               rocblas_index_value_t<S_>* result);

INSTANTIATE_IAMIN_BATCHED_TEMPLATE(float, float)
INSTANTIATE_IAMIN_BATCHED_TEMPLATE(double, double)
INSTANTIATE_IAMIN_BATCHED_TEMPLATE(rocblas_float_complex, float)
INSTANTIATE_IAMIN_BATCHED_TEMPLATE(rocblas_double_complex, double)

#undef INSTANTIATE_IAMIN_BATCHED_TEMPLATE

// clang-format on
