/* ************************************************************************
 * Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../blas1/rocblas_copy.hpp"
#include "check_numerics_vector.hpp"
#include "device_macros.hpp"
#include "rocblas_tpmv.hpp"
#include "utility.hpp"

#ifdef tmpv_calc_upperat
#error tmpv_calc_upperat is already defined
#endif
#define tmpv_calc_upperat(_i, _j) ((size_t(_j) * ((_j) + 1)) / 2 + (_i))

#ifdef tmpv_calc_lowerat
#error tmpv_calc_lowerat is already defined
#endif
#define tmpv_calc_lowerat(_i, _j) (size_t(_j) * n + ((_i) - (_j)) - ((size_t(_j) - 1) * (_j)) / 2)

template <rocblas_int NB, typename T>
ROCBLAS_KERNEL_ILF void rocblas_tpmvn_kernel_calc(bool        is_upper,
                                                  bool        is_unit_diag,
                                                  rocblas_int n,
                                                  const T*    AP,
                                                  const T*    x,
                                                  int64_t     incx,
                                                  T*          workspace)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < n)
    {
        T res = x[tid * incx];
        if(is_upper)
        {
            if(!is_unit_diag)
            {
                res *= AP[tmpv_calc_upperat(tid, tid)];
            }
            for(rocblas_int col = tid + 1; col < n; ++col)
            {
                res += AP[tmpv_calc_upperat(tid, col)] * x[col * incx];
            }
        }
        else
        {
            if(!is_unit_diag)
            {
                //cppcheck-suppress duplicateExpression
                res *= AP[tmpv_calc_lowerat(tid, tid)];
            }
            for(rocblas_int col = 0; col < tid; ++col)
            {
                res += AP[tmpv_calc_lowerat(tid, col)] * x[col * incx];
            }
        }

        workspace[tid] = res;
    }
}

template <rocblas_int NB, typename T>
ROCBLAS_KERNEL_ILF void rocblas_tpmvc_kernel_calc(bool        is_upper,
                                                  bool        is_unit_diag,
                                                  rocblas_int n,
                                                  const T*    AP,
                                                  const T*    x,
                                                  int64_t     incx,
                                                  T*          workspace)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < n)
    {
        T res = x[tid * incx];
        if(is_upper)
        {
            if(!is_unit_diag)
            {
                res *= conj(AP[tmpv_calc_upperat(tid, tid)]);
            }
            for(rocblas_int row = 0; row < tid; ++row)
            {
                res += conj(AP[tmpv_calc_upperat(row, tid)]) * x[row * incx];
            }
        }
        else
        {
            if(!is_unit_diag)
            {
                //cppcheck-suppress duplicateExpression
                res *= conj(AP[tmpv_calc_lowerat(tid, tid)]);
            }
            for(rocblas_int row = tid + 1; row < n; ++row)
            {
                res += conj(AP[tmpv_calc_lowerat(row, tid)]) * x[row * incx];
            }
        }
        workspace[tid] = res;
    }
}

template <rocblas_int NB, typename T>
ROCBLAS_KERNEL_ILF void rocblas_tpmvt_kernel_calc(bool        is_upper,
                                                  bool        is_unit_diag,
                                                  rocblas_int n,
                                                  const T*    AP,
                                                  const T*    x,
                                                  int64_t     incx,
                                                  T*          workspace)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    T           res;
    rocblas_int row;

    if(tid < n)
    {
        res = x[tid * incx];
        if(is_upper)
        {
            if(!is_unit_diag)
            {
                res *= AP[tmpv_calc_upperat(tid, tid)];
            }

            for(row = 0; row < tid; ++row)
            {
                res += AP[tmpv_calc_upperat(row, tid)] * x[row * incx];
            }
        }
        else
        {
            if(!is_unit_diag)
            {
                //cppcheck-suppress duplicateExpression
                res *= AP[tmpv_calc_lowerat(tid, tid)];
            }

            for(row = tid + 1; row < n; ++row)
            {
                res += AP[tmpv_calc_lowerat(row, tid)] * x[row * incx];
            }
        }
        workspace[tid] = res;
    }
}

template <rocblas_int NB, typename TConstPtr, typename TPtr, typename TWork>
ROCBLAS_KERNEL(NB)
rocblas_tpmvn_kernel(bool           is_upper,
                     bool           is_unit_diag,
                     rocblas_int    n,
                     TConstPtr      AP,
                     rocblas_stride shift_AP,
                     rocblas_stride stride_AP,
                     TPtr           x,
                     rocblas_stride shift_x,
                     int64_t        incx,
                     rocblas_stride stride_x,
                     TWork          workspace,
                     rocblas_stride stride_w,
                     rocblas_int    batch_count)
{
    static constexpr ptrdiff_t shift_w = 0;

    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif
        rocblas_tpmvn_kernel_calc<NB>(is_upper,
                                      is_unit_diag,
                                      n,
                                      load_ptr_batch(AP, batch, shift_AP, stride_AP),
                                      load_ptr_batch(x, batch, shift_x, stride_x),
                                      incx,
                                      load_ptr_batch(workspace, batch, shift_w, stride_w));

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

template <rocblas_int NB, typename TConstPtr, typename TPtr, typename TWork>
ROCBLAS_KERNEL(NB)
rocblas_tpmvt_kernel(bool           is_upper,
                     bool           is_unit_diag,
                     rocblas_int    n,
                     TConstPtr      AP,
                     rocblas_stride shift_AP,
                     rocblas_stride stride_AP,
                     TPtr           x,
                     rocblas_stride shift_x,
                     int64_t        incx,
                     rocblas_stride stride_x,
                     TWork          workspace,
                     rocblas_stride stride_w,
                     rocblas_int    batch_count)
{
    static constexpr ptrdiff_t shift_w = 0;

    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        rocblas_tpmvt_kernel_calc<NB>(is_upper,
                                      is_unit_diag,
                                      n,
                                      load_ptr_batch(AP, batch, shift_AP, stride_AP),
                                      load_ptr_batch(x, batch, shift_x, stride_x),
                                      incx,
                                      load_ptr_batch(workspace, batch, shift_w, stride_w));
#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

template <rocblas_int NB, typename TConstPtr, typename TPtr, typename TWork>
ROCBLAS_KERNEL(NB)
rocblas_tpmvc_kernel(bool           is_upper,
                     bool           is_unit_diag,
                     rocblas_int    n,
                     TConstPtr      AP,
                     rocblas_stride shift_AP,
                     rocblas_stride stride_AP,
                     TPtr           x,
                     rocblas_stride shift_x,
                     int64_t        incx,
                     rocblas_stride stride_x,
                     TWork          workspace,
                     rocblas_stride stride_w,
                     rocblas_int    batch_count)
{
    static constexpr ptrdiff_t shift_w = 0;

    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif
        rocblas_tpmvc_kernel_calc<NB>(is_upper,
                                      is_unit_diag,
                                      n,
                                      load_ptr_batch(AP, batch, shift_AP, stride_AP),
                                      load_ptr_batch(x, batch, shift_x, stride_x),
                                      incx,
                                      load_ptr_batch(workspace, batch, shift_w, stride_w));
#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

#undef tmpv_calc_upperat
#undef tmpv_calc_lowerat

template <typename TConstPtr, typename TPtr, typename TWork>
rocblas_status rocblas_internal_tpmv_launcher(rocblas_handle    handle,
                                              rocblas_fill      uplo,
                                              rocblas_operation transa,
                                              rocblas_diagonal  diag,
                                              rocblas_int       n,
                                              TConstPtr         AP,
                                              rocblas_stride    offset_AP,
                                              rocblas_stride    stride_AP,
                                              TPtr              x,
                                              rocblas_stride    offset_x,
                                              int64_t           incx,
                                              rocblas_stride    stride_x,
                                              TWork             workspace,
                                              rocblas_stride    stride_w,
                                              rocblas_int       batch_count)
{
    //
    // quick return
    //
    if(!n || !batch_count)
    {
        return rocblas_status_success;
    }

    static constexpr rocblas_int NB = ROCBLAS_TPMV_NB;

    hipStream_t rocblas_stream = handle->get_stream();

    int64_t shift_x = incx < 0 ? offset_x + incx * (1 - n) : offset_x;

    int batches = handle->getBatchGridDim((int)batch_count);

    dim3 tpmv_grid((n - 1) / NB + 1, 1, batches);
    dim3 tpmv_threads(NB);

    switch(transa)
    {
    case rocblas_operation_none:
    {
        ROCBLAS_LAUNCH_KERNEL(rocblas_tpmvn_kernel<NB>,
                              tpmv_grid,
                              tpmv_threads,
                              0,
                              rocblas_stream,
                              uplo == rocblas_fill_upper,
                              diag == rocblas_diagonal_unit,
                              n,
                              AP,
                              offset_AP,
                              stride_AP,
                              x,
                              shift_x,
                              incx,
                              stride_x,
                              workspace,
                              stride_w,
                              batch_count);
        break;
    }

    case rocblas_operation_transpose:
    {
        ROCBLAS_LAUNCH_KERNEL(rocblas_tpmvt_kernel<NB>,
                              tpmv_grid,
                              tpmv_threads,
                              0,
                              rocblas_stream,
                              uplo == rocblas_fill_upper,
                              diag == rocblas_diagonal_unit,
                              n,
                              AP,
                              offset_AP,
                              stride_AP,
                              x,
                              shift_x,
                              incx,
                              stride_x,
                              workspace,
                              stride_w,
                              batch_count);
        break;
    }

    case rocblas_operation_conjugate_transpose:
    {
        ROCBLAS_LAUNCH_KERNEL(rocblas_tpmvc_kernel<NB>,
                              tpmv_grid,
                              tpmv_threads,
                              0,
                              rocblas_stream,
                              uplo == rocblas_fill_upper,
                              diag == rocblas_diagonal_unit,
                              n,
                              AP,
                              offset_AP,
                              stride_AP,
                              x,
                              shift_x,
                              incx,
                              stride_x,
                              workspace,
                              stride_w,
                              batch_count);

        break;
    }
    }

    //
    // Copy workspace to x.
    //
    {
        static constexpr rocblas_int offset_w = 0;
        static constexpr rocblas_int incw     = 1;
        return rocblas_internal_copy_launcher<int64_t, ROCBLAS_COPY_NB>(handle,
                                                                        n,
                                                                        workspace,
                                                                        offset_w,
                                                                        incw,
                                                                        stride_w,
                                                                        x,
                                                                        offset_x,
                                                                        incx,
                                                                        stride_x,
                                                                        batch_count);
    }
}

//TODO :-Add rocblas_check_numerics_tp_matrix_template for checking Matrix `AP` which is a Triangular Packed Matrix
template <typename T, typename U>
rocblas_status rocblas_tpmv_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           int64_t        n,
                                           T              AP,
                                           rocblas_stride offset_a,
                                           rocblas_stride stride_a,
                                           U              x,
                                           rocblas_stride offset_x,
                                           int64_t        incx,
                                           rocblas_stride stride_x,
                                           int64_t        batch_count,
                                           const int      check_numerics,
                                           bool           is_input)
{
    rocblas_status check_numerics_status
        = rocblas_internal_check_numerics_vector_template(function_name,
                                                          handle,
                                                          n,
                                                          x,
                                                          offset_x,
                                                          incx,
                                                          stride_x,
                                                          batch_count,
                                                          check_numerics,
                                                          is_input);

    return check_numerics_status;
}

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files *tpmv*.cpp

#ifdef INSTANTIATE_TPMV_NUMERICS
#error INSTANTIATE_TPMV_NUMERICS  already defined
#endif

#define INSTANTIATE_TPMV_NUMERICS(T_, U_)                                                      \
    template rocblas_status rocblas_tpmv_check_numerics<T_, U_>(const char*    function_name,  \
                                                                rocblas_handle handle,         \
                                                                int64_t        n,              \
                                                                T_             AP,             \
                                                                rocblas_stride offset_a,       \
                                                                rocblas_stride stride_a,       \
                                                                U_             x,              \
                                                                rocblas_stride offset_x,       \
                                                                int64_t        incx,           \
                                                                rocblas_stride stride_x,       \
                                                                int64_t        batch_count,    \
                                                                const int      check_numerics, \
                                                                bool           is_input);

INSTANTIATE_TPMV_NUMERICS(float const*, float*)
INSTANTIATE_TPMV_NUMERICS(double const*, double*)
INSTANTIATE_TPMV_NUMERICS(rocblas_float_complex const*, rocblas_float_complex*)
INSTANTIATE_TPMV_NUMERICS(rocblas_double_complex const*, rocblas_double_complex*)
INSTANTIATE_TPMV_NUMERICS(float const* const*, float* const*)
INSTANTIATE_TPMV_NUMERICS(double const* const*, double* const*)
INSTANTIATE_TPMV_NUMERICS(rocblas_float_complex const* const*, rocblas_float_complex* const*)
INSTANTIATE_TPMV_NUMERICS(rocblas_double_complex const* const*, rocblas_double_complex* const*)

#undef INSTANTIATE_TPMV_NUMERICS

#ifdef INSTANTIATE_TPMV_LAUNCHER
#error INSTANTIATE_TPMV_LAUNCHER  already defined
#endif
#define INSTANTIATE_TPMV_LAUNCHER(TConstPtr_, TPtr_, TWork_)                           \
    template rocblas_status rocblas_internal_tpmv_launcher<TConstPtr_, TPtr_, TWork_>( \
        rocblas_handle    handle,                                                      \
        rocblas_fill      uplo,                                                        \
        rocblas_operation transa,                                                      \
        rocblas_diagonal  diag,                                                        \
        rocblas_int       n,                                                           \
        TConstPtr_        AP,                                                          \
        rocblas_stride    offset_AP,                                                   \
        rocblas_stride    stride_AP,                                                   \
        TPtr_             x,                                                           \
        rocblas_stride    offset_x,                                                    \
        int64_t           incx,                                                        \
        rocblas_stride    stride_x,                                                    \
        TWork_            workspace,                                                   \
        rocblas_stride    stride_w,                                                    \
        rocblas_int       batch_count);

INSTANTIATE_TPMV_LAUNCHER(float const*, float*, float*)
INSTANTIATE_TPMV_LAUNCHER(double const*, double*, double*)
INSTANTIATE_TPMV_LAUNCHER(rocblas_float_complex const*,
                          rocblas_float_complex*,
                          rocblas_float_complex*)
INSTANTIATE_TPMV_LAUNCHER(rocblas_double_complex const*,
                          rocblas_double_complex*,
                          rocblas_double_complex*)
INSTANTIATE_TPMV_LAUNCHER(float const* const*, float* const*, float*)
INSTANTIATE_TPMV_LAUNCHER(double const* const*, double* const*, double*)
INSTANTIATE_TPMV_LAUNCHER(rocblas_float_complex const* const*,
                          rocblas_float_complex* const*,
                          rocblas_float_complex*)
INSTANTIATE_TPMV_LAUNCHER(rocblas_double_complex const* const*,
                          rocblas_double_complex* const*,
                          rocblas_double_complex*)

#undef INSTANTIATE_TPMV_LAUNCHER
