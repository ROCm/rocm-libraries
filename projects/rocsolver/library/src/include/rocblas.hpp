/* **************************************************************************
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include <rocblas/rocblas.h>

#include "common_host_helpers.hpp"
#include "init_scalars.hpp"
#include "lib_device_helpers.hpp"
#include "lib_host_helpers.hpp"
#include "rocblas/internal/rocblas-exported-proto.hpp"
#include "rocblas/internal/rocblas_device_malloc.hpp"
#include "rocsolver_logger.hpp"

#ifndef HAVE_ROCBLAS_64
#if ROCBLAS_VERSION_MAJOR > 4 || (ROCBLAS_VERSION_MAJOR == 4 && ROCBLAS_VERSION_MINOR >= 3)
#define HAVE_ROCBLAS_64 1
#endif
#endif

// These function templates help to provide compatibility with older versions
// of rocblas. We declare these function templates only if rocBLAS does not,
// and we delete them so that it is a compile-time error if they are used.
// The calls to these functions are guarded by if constexpr, but they still
// need to be declared even if they're only used in the branch not taken.
#if ROCBLAS_VERSION_MAJOR < 4 || (ROCBLAS_VERSION_MAJOR == 4 && ROCBLAS_VERSION_MINOR < 2)
// scal
template <typename T, typename Ta>
rocblas_status rocblas_internal_scal_template_64(rocblas_handle,
                                                 int64_t,
                                                 const Ta*,
                                                 rocblas_stride,
                                                 T*,
                                                 rocblas_stride,
                                                 int64_t,
                                                 rocblas_stride,
                                                 int64_t)
    = delete;
template <typename T, typename Ta>
rocblas_status rocblas_internal_scal_batched_template_64(rocblas_handle,
                                                         int64_t,
                                                         const Ta*,
                                                         rocblas_stride,
                                                         T* const*,
                                                         rocblas_stride,
                                                         int64_t,
                                                         rocblas_stride,
                                                         int64_t)
    = delete;

// ger / gerc
template <typename T>
rocblas_status rocblas_internal_ger_template_64(rocblas_handle,
                                                int64_t,
                                                int64_t,
                                                const T*,
                                                rocblas_stride,
                                                const T*,
                                                rocblas_stride,
                                                int64_t,
                                                rocblas_stride,
                                                const T*,
                                                rocblas_stride,
                                                int64_t,
                                                rocblas_stride,
                                                T*,
                                                rocblas_stride,
                                                int64_t,
                                                rocblas_stride,
                                                int64_t)
    = delete;
template <typename T>
rocblas_status rocblas_internal_gerc_template_64(rocblas_handle,
                                                 int64_t,
                                                 int64_t,
                                                 const T*,
                                                 rocblas_stride,
                                                 const T*,
                                                 rocblas_stride,
                                                 int64_t,
                                                 rocblas_stride,
                                                 const T*,
                                                 rocblas_stride,
                                                 int64_t,
                                                 rocblas_stride,
                                                 T*,
                                                 rocblas_stride,
                                                 int64_t,
                                                 rocblas_stride,
                                                 int64_t)
    = delete;
template <typename T>
rocblas_status rocblas_internal_ger_batched_template_64(rocblas_handle,
                                                        int64_t,
                                                        int64_t,
                                                        const T*,
                                                        rocblas_stride,
                                                        const T* const*,
                                                        rocblas_stride,
                                                        int64_t,
                                                        rocblas_stride,
                                                        const T* const*,
                                                        rocblas_stride,
                                                        int64_t,
                                                        rocblas_stride,
                                                        T* const*,
                                                        rocblas_stride,
                                                        int64_t,
                                                        rocblas_stride,
                                                        int64_t)
    = delete;
template <typename T>
rocblas_status rocblas_internal_gerc_batched_template_64(rocblas_handle,
                                                         int64_t,
                                                         int64_t,
                                                         const T*,
                                                         rocblas_stride,
                                                         const T* const*,
                                                         rocblas_stride,
                                                         int64_t,
                                                         rocblas_stride,
                                                         const T* const*,
                                                         rocblas_stride,
                                                         int64_t,
                                                         rocblas_stride,
                                                         T* const*,
                                                         rocblas_stride,
                                                         int64_t,
                                                         rocblas_stride,
                                                         int64_t)
    = delete;

// trsm
template <typename T>
rocblas_status rocblas_internal_trsm_workspace_size_64(rocblas_side,
                                                       rocblas_operation,
                                                       int64_t,
                                                       int64_t,
                                                       int64_t,
                                                       int64_t,
                                                       int64_t,
                                                       int64_t,
                                                       size_t*,
                                                       size_t*,
                                                       size_t*,
                                                       size_t*,
                                                       size_t*)
    = delete;
template <typename T>
rocblas_status rocblas_internal_trsm_batched_workspace_size_64(rocblas_side,
                                                               rocblas_operation,
                                                               int64_t,
                                                               int64_t,
                                                               int64_t,
                                                               int64_t,
                                                               int64_t,
                                                               int64_t,
                                                               size_t*,
                                                               size_t*,
                                                               size_t*,
                                                               size_t*,
                                                               size_t*)
    = delete;
template <typename T>
rocblas_status rocblas_internal_trsm_template_64(rocblas_handle,
                                                 rocblas_side,
                                                 rocblas_fill,
                                                 rocblas_operation,
                                                 rocblas_diagonal,
                                                 int64_t,
                                                 int64_t,
                                                 const T*,
                                                 const T*,
                                                 rocblas_stride,
                                                 int64_t,
                                                 rocblas_stride,
                                                 T*,
                                                 rocblas_stride,
                                                 int64_t,
                                                 rocblas_stride,
                                                 int64_t,
                                                 bool,
                                                 void*,
                                                 void*,
                                                 void* invA = nullptr,
                                                 void* invAarr = nullptr,
                                                 const T* supplied_invA = nullptr,
                                                 int64_t supplied_invA_size_64 = 0,
                                                 rocblas_stride offset_invA = 0,
                                                 rocblas_stride stride_invA = 0)
    = delete;
template <typename T>
rocblas_status rocblas_internal_trsm_batched_template_64(rocblas_handle,
                                                         rocblas_side,
                                                         rocblas_fill,
                                                         rocblas_operation,
                                                         rocblas_diagonal,
                                                         int64_t,
                                                         int64_t,
                                                         const T*,
                                                         const T* const*,
                                                         rocblas_stride,
                                                         int64_t,
                                                         rocblas_stride,
                                                         T* const*,
                                                         rocblas_stride,
                                                         int64_t,
                                                         rocblas_stride,
                                                         int64_t,
                                                         bool,
                                                         void*,
                                                         void*,
                                                         void* invA = nullptr,
                                                         void* invAarr = nullptr,
                                                         const T* const* supplied_invA = nullptr,
                                                         int64_t supplied_invA_size_64 = 0,
                                                         rocblas_stride offset_invA = 0,
                                                         rocblas_stride stride_invA = 0)
    = delete;
#endif
#if ROCBLAS_VERSION_MAJOR < 4 || (ROCBLAS_VERSION_MAJOR == 4 && ROCBLAS_VERSION_MINOR < 3)
// gemm
template <typename T>
rocblas_status rocblas_internal_gemm_template_64(rocblas_handle,
                                                 rocblas_operation,
                                                 rocblas_operation,
                                                 int64_t,
                                                 int64_t,
                                                 int64_t,
                                                 const T*,
                                                 const T*,
                                                 rocblas_stride,
                                                 int64_t,
                                                 rocblas_stride,
                                                 const T*,
                                                 rocblas_stride,
                                                 int64_t,
                                                 rocblas_stride,
                                                 const T*,
                                                 T*,
                                                 rocblas_stride,
                                                 int64_t,
                                                 rocblas_stride,
                                                 int64_t)
    = delete;
template <typename T>
rocblas_status rocblas_internal_gemm_batched_template_64(rocblas_handle,
                                                         rocblas_operation,
                                                         rocblas_operation,
                                                         int64_t,
                                                         int64_t,
                                                         int64_t,
                                                         const T*,
                                                         const T* const*,
                                                         rocblas_stride,
                                                         int64_t,
                                                         rocblas_stride,
                                                         const T* const*,
                                                         rocblas_stride,
                                                         int64_t,
                                                         rocblas_stride,
                                                         const T*,
                                                         T* const*,
                                                         rocblas_stride,
                                                         int64_t,
                                                         rocblas_stride,
                                                         int64_t)
    = delete;
#endif

ROCSOLVER_BEGIN_NAMESPACE

constexpr auto rocblas2string_status(rocblas_status status)
{
    switch(status)
    {
    case rocblas_status_success: return "rocblas_status_success";
    case rocblas_status_invalid_handle: return "rocblas_status_invalid_handle";
    case rocblas_status_not_implemented: return "rocblas_status_not_implemented";
    case rocblas_status_invalid_pointer: return "rocblas_status_invalid_pointer";
    case rocblas_status_invalid_size: return "rocblas_status_invalid_size";
    case rocblas_status_memory_error: return "rocblas_status_memory_error";
    case rocblas_status_internal_error: return "rocblas_status_internal_error";
    case rocblas_status_perf_degraded: return "rocblas_status_perf_degraded";
    case rocblas_status_size_query_mismatch: return "rocblas_status_size_query_mismatch";
    case rocblas_status_size_increased: return "rocblas_status_size_increased";
    case rocblas_status_size_unchanged: return "rocblas_status_size_unchanged";
    case rocblas_status_invalid_value: return "rocblas_status_invalid_value";
    case rocblas_status_continue: return "rocblas_status_continue";
    case rocblas_status_check_numerics_fail: return "rocblas_status_check_numerics_fail";
    default: return "unknown";
    }
}

#define HIP_CHECK(...)                                         \
    {                                                          \
        hipError_t _status = (__VA_ARGS__);                    \
        if(_status != hipSuccess)                              \
            return get_rocblas_status_for_hip_status(_status); \
    }

#define ROCBLAS_CHECK(...)                      \
    {                                           \
        rocblas_status _status = (__VA_ARGS__); \
        if(_status != rocblas_status_success)   \
            return _status;                     \
    }
#define THROW_IF_ROCBLAS_ERROR(...)             \
    {                                           \
        rocblas_status _status = (__VA_ARGS__); \
        if(_status != rocblas_status_success)   \
            throw _status;                      \
    }

template <typename T>
struct rocblas_index_value_t;

// axpy
template <typename T>
rocblas_status rocblasCall_axpy(rocblas_handle handle,
                                rocblas_int n,
                                const T* alpha,
                                rocblas_stride stride_alpha,
                                const T* x,
                                rocblas_stride shiftx,
                                rocblas_int incx,
                                rocblas_stride stridex,
                                T* y,
                                rocblas_stride shifty,
                                rocblas_int incy,
                                rocblas_stride stridey,
                                rocblas_int batch_count)
{
    // TODO: How to get alpha for trace logging
    ROCBLAS_ENTER("axpy", "n:", n, "shiftX:", shiftx, "incx:", incx, "shiftY:", shifty,
                  "incy:", incy, "bc:", batch_count);

    return rocblas_internal_axpy_template(handle, n, alpha, stride_alpha, x, shiftx, incx, stridex,
                                          y, shifty, incy, stridey, batch_count);
}

// batched axpy
template <typename T>
rocblas_status rocblasCall_axpy(rocblas_handle handle,
                                rocblas_int n,
                                const T* alpha,
                                rocblas_stride stride_alpha,
                                const T* const* x,
                                rocblas_stride shiftx,
                                rocblas_int incx,
                                rocblas_stride stridex,
                                T* const* y,
                                rocblas_stride shifty,
                                rocblas_int incy,
                                rocblas_stride stridey,
                                rocblas_int batch_count)
{
    // TODO: How to get alpha for trace logging
    ROCBLAS_ENTER("axpy", "n:", n, "shiftX:", shiftx, "incx:", incx, "shiftY:", shifty,
                  "incy:", incy, "bc:", batch_count);

    return rocblas_internal_axpy_batched_template(handle, n, alpha, stride_alpha, x, shiftx, incx,
                                                  stridex, y, shifty, incy, stridey, batch_count);
}

// iamax
template <typename T, typename S>
rocblas_status rocblasCall_iamax(rocblas_handle handle,
                                 rocblas_int n,
                                 const T* x,
                                 rocblas_stride shiftx,
                                 rocblas_int incx,
                                 rocblas_stride stridex,
                                 rocblas_int batch_count,
                                 rocblas_int* result,
                                 rocblas_index_value_t<S>* workspace)
{
    ROCBLAS_ENTER("iamax", "n:", n, "shiftX:", shiftx, "incx:", incx, "bc:", batch_count);

    return rocblas_internal_iamax_template(handle, n, x, shiftx, incx, stridex, batch_count, result,
                                           workspace);
}

// batched iamax
template <typename T, typename S>
rocblas_status rocblasCall_iamax(rocblas_handle handle,
                                 rocblas_int n,
                                 const T* const* x,
                                 rocblas_stride shiftx,
                                 rocblas_int incx,
                                 rocblas_stride stridex,
                                 rocblas_int batch_count,
                                 rocblas_int* result,
                                 rocblas_index_value_t<S>* workspace)
{
    ROCBLAS_ENTER("iamax", "n:", n, "shiftX:", shiftx, "incx:", incx, "bc:", batch_count);

    return rocblas_internal_iamax_batched_template(handle, n, x, shiftx, incx, stridex, batch_count,
                                                   result, workspace);
}

// scal
template <typename T, typename I, typename S>
rocblas_status rocblasCall_scal(rocblas_handle handle,
                                I n,
                                const S* alpha,
                                rocblas_stride stridea,
                                T* x,
                                rocblas_stride offsetx,
                                I incx,
                                rocblas_stride stridex,
                                I batch_count)
{
    // TODO: How to get alpha for trace logging
    ROCBLAS_ENTER("scal", "n:", n, "shiftX:", offsetx, "incx:", incx, "bc:", batch_count);

    if constexpr(std::is_same<I, int64_t>::value)
        return rocblas_internal_scal_template_64(handle, n, alpha, stridea, x, offsetx, incx,
                                                 stridex, batch_count);
    else
        return rocblas_internal_scal_template(handle, n, alpha, stridea, x, offsetx, incx, stridex,
                                              batch_count);
}

// batched scal
template <typename T, typename I, typename S>
rocblas_status rocblasCall_scal(rocblas_handle handle,
                                I n,
                                const S* alpha,
                                rocblas_stride stridea,
                                T* const* x,
                                rocblas_stride offsetx,
                                I incx,
                                rocblas_stride stridex,
                                I batch_count)
{
    // TODO: How to get alpha for trace logging
    ROCBLAS_ENTER("scal", "n:", n, "shiftX:", offsetx, "incx:", incx, "bc:", batch_count);

    if constexpr(std::is_same<I, int64_t>::value)
        return rocblas_internal_scal_batched_template_64(handle, n, alpha, stridea, x, offsetx,
                                                         incx, stridex, batch_count);
    else
        return rocblas_internal_scal_batched_template(handle, n, alpha, stridea, x, offsetx, incx,
                                                      stridex, batch_count);
}

// dot
template <bool CONJ, typename T, typename Tex>
rocblas_status rocblasCall_dot(rocblas_handle handle,
                               rocblas_int n,
                               const T* x,
                               rocblas_stride offsetx,
                               rocblas_int incx,
                               rocblas_stride stridex,
                               const T* y,
                               rocblas_stride offsety,
                               rocblas_int incy,
                               rocblas_stride stridey,
                               rocblas_int batch_count,
                               T* results,
                               Tex* workspace,
                               T** work = nullptr)
{
    ROCBLAS_ENTER("dot", "n:", n, "shiftX:", offsetx, "incx:", incx, "shiftY:", offsety,
                  "incy:", incy, "bc:", batch_count);

    if constexpr(CONJ)
        return rocblas_internal_dotc_template(handle, n, x, offsetx, incx, stridex, y, offsety,
                                              incy, stridey, batch_count, results, workspace);
    else
        return rocblas_internal_dot_template(handle, n, x, offsetx, incx, stridex, y, offsety, incy,
                                             stridey, batch_count, results, workspace);
}

// batched dot
template <bool CONJ, typename T, typename Tex>
rocblas_status rocblasCall_dot(rocblas_handle handle,
                               rocblas_int n,
                               const T* const* x,
                               rocblas_stride offsetx,
                               rocblas_int incx,
                               rocblas_stride stridex,
                               const T* const* y,
                               rocblas_stride offsety,
                               rocblas_int incy,
                               rocblas_stride stridey,
                               rocblas_int batch_count,
                               T* results,
                               Tex* workspace,
                               T** work = nullptr)
{
    ROCBLAS_ENTER("dot", "n:", n, "shiftX:", offsetx, "incx:", incx, "shiftY:", offsety,
                  "incy:", incy, "bc:", batch_count);

    if constexpr(CONJ)
        return rocblas_internal_dotc_batched_template(handle, n, x, offsetx, incx, stridex, y,
                                                      offsety, incy, stridey, batch_count, results,
                                                      workspace);
    else
        return rocblas_internal_dot_batched_template(handle, n, x, offsetx, incx, stridex, y, offsety,
                                                     incy, stridey, batch_count, results, workspace);
}

// dot overload
template <bool CONJ, typename T>
rocblas_status rocblasCall_dot(rocblas_handle handle,
                               rocblas_int n,
                               T* x,
                               rocblas_stride offsetx,
                               rocblas_int incx,
                               rocblas_stride stridex,
                               T* const y[],
                               rocblas_stride offsety,
                               rocblas_int incy,
                               rocblas_stride stridey,
                               rocblas_int batch_count,
                               T* results,
                               T* workspace,
                               T** work)
{
    ROCBLAS_ENTER("dot", "n:", n, "shiftX:", offsetx, "incx:", incx, "shiftY:", offsety,
                  "incy:", incy, "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, work, x, stridex,
                            batch_count);

    if constexpr(CONJ)
        return rocblas_internal_dotc_batched_template(handle, n, cast2constType<T>(work), offsetx,
                                                      incx, stridex, y, offsety, incy, stridey,
                                                      batch_count, results, workspace);
    else
        return rocblas_internal_dot_batched_template(handle, n, cast2constType<T>(work), offsetx,
                                                     incx, stridex, y, offsety, incy, stridey,
                                                     batch_count, results, workspace);
}

// ger - non batched
template <bool CONJ, typename T, typename I>
rocblas_status rocblasCall_ger(rocblas_handle handle,
                               I m,
                               I n,
                               const T* alpha,
                               rocblas_stride stridea,
                               const T* x,
                               rocblas_stride offsetx,
                               I incx,
                               rocblas_stride stridex,
                               const T* y,
                               rocblas_stride offsety,
                               I incy,
                               rocblas_stride stridey,
                               T* A,
                               rocblas_stride offsetA,
                               I lda,
                               rocblas_stride strideA,
                               I batch_count,
                               T** work)
{
    // TODO: How to get alpha for trace logging
    ROCBLAS_ENTER("ger", "m:", m, "n:", n, "shiftX:", offsetx, "incx:", incx, "shiftY:", offsety,
                  "incy:", incy, "shiftA:", offsetA, "lda:", lda, "bc:", batch_count);

    if constexpr(std::is_same<I, int64_t>::value)
    {
        if constexpr(CONJ)
            return rocblas_internal_gerc_template_64(handle, m, n, alpha, stridea, x, offsetx, incx,
                                                     stridex, y, offsety, incy, stridey, A, offsetA,
                                                     lda, strideA, batch_count);
        else
            return rocblas_internal_ger_template_64(handle, m, n, alpha, stridea, x, offsetx, incx,
                                                    stridex, y, offsety, incy, stridey, A, offsetA,
                                                    lda, strideA, batch_count);
    }
    else
    {
        if constexpr(CONJ)
            return rocblas_internal_gerc_template(handle, m, n, alpha, stridea, x, offsetx, incx,
                                                  stridex, y, offsety, incy, stridey, A, offsetA,
                                                  lda, strideA, batch_count);
        else
            return rocblas_internal_ger_template(handle, m, n, alpha, stridea, x, offsetx, incx,
                                                 stridex, y, offsety, incy, stridey, A, offsetA,
                                                 lda, strideA, batch_count);
    }
}

// ger batched
template <bool CONJ, typename T, typename I>
rocblas_status rocblasCall_ger(rocblas_handle handle,
                               I m,
                               I n,
                               const T* alpha,
                               rocblas_stride stridea,
                               const T* const* x,
                               rocblas_stride offsetx,
                               I incx,
                               rocblas_stride stridex,
                               const T* const* y,
                               rocblas_stride offsety,
                               I incy,
                               rocblas_stride stridey,
                               T* const* A,
                               rocblas_stride offsetA,
                               I lda,
                               rocblas_stride strideA,
                               I batch_count,
                               T** work)
{
    // TODO: How to get alpha for trace logging
    ROCBLAS_ENTER("ger", "m:", m, "n:", n, "shiftX:", offsetx, "incx:", incx, "shiftY:", offsety,
                  "incy:", incy, "shiftA:", offsetA, "lda:", lda, "bc:", batch_count);

    if constexpr(std::is_same<I, int64_t>::value)
    {
        if constexpr(CONJ)
            return rocblas_internal_gerc_batched_template_64(
                handle, m, n, alpha, stridea, x, offsetx, incx, stridex, y, offsety, incy, stridey,
                A, offsetA, lda, strideA, batch_count);
        else
            return rocblas_internal_ger_batched_template_64(handle, m, n, alpha, stridea, x, offsetx,
                                                            incx, stridex, y, offsety, incy, stridey,
                                                            A, offsetA, lda, strideA, batch_count);
    }
    else
    {
        if constexpr(CONJ)
            return rocblas_internal_gerc_batched_template(handle, m, n, alpha, stridea, x, offsetx,
                                                          incx, stridex, y, offsety, incy, stridey,
                                                          A, offsetA, lda, strideA, batch_count);
        else
            return rocblas_internal_ger_batched_template(handle, m, n, alpha, stridea, x, offsetx,
                                                         incx, stridex, y, offsety, incy, stridey,
                                                         A, offsetA, lda, strideA, batch_count);
    }
}

// ger overload - batched with strided y
template <bool CONJ, typename T, typename I>
rocblas_status rocblasCall_ger(rocblas_handle handle,
                               I m,
                               I n,
                               const T* alpha,
                               rocblas_stride stridea,
                               const T* const* x,
                               rocblas_stride offsetx,
                               I incx,
                               rocblas_stride stridex,
                               T* y,
                               rocblas_stride offsety,
                               I incy,
                               rocblas_stride stridey,
                               T* const* A,
                               rocblas_stride offsetA,
                               I lda,
                               rocblas_stride strideA,
                               I batch_count,
                               T** work)
{
    // TODO: How to get alpha for trace logging
    ROCBLAS_ENTER("ger", "m:", m, "n:", n, "shiftX:", offsetx, "incx:", incx, "shiftY:", offsety,
                  "incy:", incy, "shiftA:", offsetA, "lda:", lda, "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    I blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, work, y, stridey,
                            batch_count);

    if constexpr(std::is_same<I, int64_t>::value)
    {
        if constexpr(CONJ)
            return rocblas_internal_gerc_batched_template_64(
                handle, m, n, alpha, stridea, x, offsetx, incx, stridex, cast2constType<T>(work),
                offsety, incy, stridey, A, offsetA, lda, strideA, batch_count);
        else
            return rocblas_internal_ger_batched_template_64(
                handle, m, n, alpha, stridea, x, offsetx, incx, stridex, cast2constType<T>(work),
                offsety, incy, stridey, A, offsetA, lda, strideA, batch_count);
    }
    else
    {
        if constexpr(CONJ)
            return rocblas_internal_gerc_batched_template(
                handle, m, n, alpha, stridea, x, offsetx, incx, stridex, cast2constType<T>(work),
                offsety, incy, stridey, A, offsetA, lda, strideA, batch_count);
        else
            return rocblas_internal_ger_batched_template(
                handle, m, n, alpha, stridea, x, offsetx, incx, stridex, cast2constType<T>(work),
                offsety, incy, stridey, A, offsetA, lda, strideA, batch_count);
    }
}

// ger overload - batched with strided x
template <bool CONJ, typename T, typename I>
rocblas_status rocblasCall_ger(rocblas_handle handle,
                               I m,
                               I n,
                               const T* alpha,
                               rocblas_stride stridea,
                               T* x,
                               rocblas_stride offsetx,
                               I incx,
                               rocblas_stride stridex,
                               const T* const* y,
                               rocblas_stride offsety,
                               I incy,
                               rocblas_stride stridey,
                               T* const* A,
                               rocblas_stride offsetA,
                               I lda,
                               rocblas_stride strideA,
                               I batch_count,
                               T** work)
{
    // TODO: How to get alpha for trace logging
    ROCBLAS_ENTER("ger", "m:", m, "n:", n, "shiftX:", offsetx, "incx:", incx, "shiftY:", offsety,
                  "incy:", incy, "shiftA:", offsetA, "lda:", lda, "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    I blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, work, x, stridex,
                            batch_count);

    if constexpr(std::is_same<I, int64_t>::value)
    {
        if constexpr(CONJ)
            return rocblas_internal_gerc_batched_template_64(
                handle, m, n, alpha, stridea, cast2constType<T>(work), offsetx, incx, stridex, y,
                offsety, incy, stridey, A, offsetA, lda, strideA, batch_count);
        else
            return rocblas_internal_ger_batched_template_64(
                handle, m, n, alpha, stridea, cast2constType<T>(work), offsetx, incx, stridex, y,
                offsety, incy, stridey, A, offsetA, lda, strideA, batch_count);
    }
    else
    {
        if constexpr(CONJ)
            return rocblas_internal_gerc_batched_template(
                handle, m, n, alpha, stridea, cast2constType<T>(work), offsetx, incx, stridex, y,
                offsety, incy, stridey, A, offsetA, lda, strideA, batch_count);
        else
            return rocblas_internal_ger_batched_template(
                handle, m, n, alpha, stridea, cast2constType<T>(work), offsetx, incx, stridex, y,
                offsety, incy, stridey, A, offsetA, lda, strideA, batch_count);
    }
}

// gemv - non batched
template <typename T>
rocblas_status rocblasCall_gemv(rocblas_handle handle,
                                rocblas_operation transA,
                                rocblas_int m,
                                rocblas_int n,
                                const T* alpha,
                                rocblas_stride stride_alpha,
                                const T* A,
                                rocblas_stride offseta,
                                rocblas_int lda,
                                rocblas_stride strideA,
                                const T* x,
                                rocblas_stride offsetx,
                                rocblas_int incx,
                                rocblas_stride stridex,
                                const T* beta,
                                rocblas_stride stride_beta,
                                T* y,
                                rocblas_stride offsety,
                                rocblas_int incy,
                                rocblas_stride stridey,
                                rocblas_int batch_count,
                                T** work)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("gemv", "trans:", transA, "m:", m, "n:", n, "shiftA:", offseta, "lda:", lda,
                  "shiftX:", offsetx, "incx:", incx, "shiftY:", offsety, "incy:", incy,
                  "bc:", batch_count);

    return rocblas_internal_gemv_template(handle, transA, m, n, alpha, stride_alpha, A, offseta,
                                          lda, strideA, x, offsetx, incx, stridex, beta,
                                          stride_beta, y, offsety, incy, stridey, batch_count);
}

// gemv - batched
template <typename T>
rocblas_status rocblasCall_gemv(rocblas_handle handle,
                                rocblas_operation transA,
                                rocblas_int m,
                                rocblas_int n,
                                const T* alpha,
                                rocblas_stride stride_alpha,
                                const T* const* A,
                                rocblas_stride offseta,
                                rocblas_int lda,
                                rocblas_stride strideA,
                                const T* const* x,
                                rocblas_stride offsetx,
                                rocblas_int incx,
                                rocblas_stride stridex,
                                const T* beta,
                                rocblas_stride stride_beta,
                                T* const* y,
                                rocblas_stride offsety,
                                rocblas_int incy,
                                rocblas_stride stridey,
                                rocblas_int batch_count,
                                T** work)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("gemv", "trans:", transA, "m:", m, "n:", n, "shiftA:", offseta, "lda:", lda,
                  "shiftX:", offsetx, "incx:", incx, "shiftY:", offsety, "incy:", incy,
                  "bc:", batch_count);

    return rocblas_internal_gemv_batched_template(
        handle, transA, m, n, alpha, stride_alpha, A, offseta, lda, strideA, x, offsetx, incx,
        stridex, beta, stride_beta, y, offsety, incy, stridey, batch_count);
}

// gemv overload - batched with strided A
template <typename T>
rocblas_status rocblasCall_gemv(rocblas_handle handle,
                                rocblas_operation transA,
                                rocblas_int m,
                                rocblas_int n,
                                const T* alpha,
                                rocblas_stride stride_alpha,
                                T* A,
                                rocblas_stride offseta,
                                rocblas_int lda,
                                rocblas_stride strideA,
                                const T* const* x,
                                rocblas_stride offsetx,
                                rocblas_int incx,
                                rocblas_stride stridex,
                                const T* beta,
                                rocblas_stride stride_beta,
                                T* const* y,
                                rocblas_stride offsety,
                                rocblas_int incy,
                                rocblas_stride stridey,
                                rocblas_int batch_count,
                                T** work)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("gemv", "trans:", transA, "m:", m, "n:", n, "shiftA:", offseta, "lda:", lda,
                  "shiftX:", offsetx, "incx:", incx, "shiftY:", offsety, "incy:", incy,
                  "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, work, A, strideA,
                            batch_count);

    return rocblas_internal_gemv_batched_template(
        handle, transA, m, n, alpha, stride_alpha, cast2constType<T>(work), offseta, lda, strideA,
        x, offsetx, incx, stridex, beta, stride_beta, y, offsety, incy, stridey, batch_count);
}

// gemv overload - batched with strided x
template <typename T>
rocblas_status rocblasCall_gemv(rocblas_handle handle,
                                rocblas_operation transA,
                                rocblas_int m,
                                rocblas_int n,
                                const T* alpha,
                                rocblas_stride stride_alpha,
                                const T* const* A,
                                rocblas_stride offseta,
                                rocblas_int lda,
                                rocblas_stride strideA,
                                T* x,
                                rocblas_stride offsetx,
                                rocblas_int incx,
                                rocblas_stride stridex,
                                const T* beta,
                                rocblas_stride stride_beta,
                                T* const* y,
                                rocblas_stride offsety,
                                rocblas_int incy,
                                rocblas_stride stridey,
                                rocblas_int batch_count,
                                T** work)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("gemv", "trans:", transA, "m:", m, "n:", n, "shiftA:", offseta, "lda:", lda,
                  "shiftX:", offsetx, "incx:", incx, "shiftY:", offsety, "incy:", incy,
                  "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, work, x, stridex,
                            batch_count);

    return rocblas_internal_gemv_batched_template(handle, transA, m, n, alpha, stride_alpha, A,
                                                  offseta, lda, strideA, cast2constType<T>(work),
                                                  offsetx, incx, stridex, beta, stride_beta, y,
                                                  offsety, incy, stridey, batch_count);
}

// gemv overload - batched with strided y
template <typename T>
rocblas_status rocblasCall_gemv(rocblas_handle handle,
                                rocblas_operation transA,
                                rocblas_int m,
                                rocblas_int n,
                                const T* alpha,
                                rocblas_stride stride_alpha,
                                const T* const* A,
                                rocblas_stride offseta,
                                rocblas_int lda,
                                rocblas_stride strideA,
                                const T* const* x,
                                rocblas_stride offsetx,
                                rocblas_int incx,
                                rocblas_stride stridex,
                                const T* beta,
                                rocblas_stride stride_beta,
                                T* y,
                                rocblas_stride offsety,
                                rocblas_int incy,
                                rocblas_stride stridey,
                                rocblas_int batch_count,
                                T** work)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("gemv", "trans:", transA, "m:", m, "n:", n, "shiftA:", offseta, "lda:", lda,
                  "shiftX:", offsetx, "incx:", incx, "shiftY:", offsety, "incy:", incy,
                  "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, work, y, stridey,
                            batch_count);

    return rocblas_internal_gemv_batched_template(handle, transA, m, n, alpha, stride_alpha, A,
                                                  offseta, lda, strideA, x, offsetx, incx, stridex,
                                                  beta, stride_beta, cast2constPointer<T>(work),
                                                  offsety, incy, stridey, batch_count);
}

// gemv overload - batched with strided x and y
template <typename T>
rocblas_status rocblasCall_gemv(rocblas_handle handle,
                                rocblas_operation transA,
                                rocblas_int m,
                                rocblas_int n,
                                const T* alpha,
                                rocblas_stride stride_alpha,
                                const T* const* A,
                                rocblas_stride offseta,
                                rocblas_int lda,
                                rocblas_stride strideA,
                                T* x,
                                rocblas_stride offsetx,
                                rocblas_int incx,
                                rocblas_stride stridex,
                                const T* beta,
                                rocblas_stride stride_beta,
                                T* y,
                                rocblas_stride offsety,
                                rocblas_int incy,
                                rocblas_stride stridey,
                                rocblas_int batch_count,
                                T** work)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("gemv", "trans:", transA, "m:", m, "n:", n, "shiftA:", offseta, "lda:", lda,
                  "shiftX:", offsetx, "incx:", incx, "shiftY:", offsety, "incy:", incy,
                  "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, work, x, stridex,
                            batch_count);
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, (work + batch_count), y,
                            stridey, batch_count);

    return rocblas_internal_gemv_batched_template(
        handle, transA, m, n, alpha, stride_alpha, A, offseta, lda, strideA,
        cast2constType<T>(work), offsetx, incx, stridex, beta, stride_beta,
        cast2constPointer<T>(work + batch_count), offsety, incy, stridey, batch_count);
}

// gemv overload - batched with strided A and y
template <typename T>
rocblas_status rocblasCall_gemv(rocblas_handle handle,
                                rocblas_operation transA,
                                rocblas_int m,
                                rocblas_int n,
                                const T* alpha,
                                rocblas_stride stride_alpha,
                                T* A,
                                rocblas_stride offseta,
                                rocblas_int lda,
                                rocblas_stride strideA,
                                const T* const* x,
                                rocblas_stride offsetx,
                                rocblas_int incx,
                                rocblas_stride stridex,
                                const T* beta,
                                rocblas_stride stride_beta,
                                T* y,
                                rocblas_stride offsety,
                                rocblas_int incy,
                                rocblas_stride stridey,
                                rocblas_int batch_count,
                                T** work)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("gemv", "trans:", transA, "m:", m, "n:", n, "shiftA:", offseta, "lda:", lda,
                  "shiftX:", offsetx, "incx:", incx, "shiftY:", offsety, "incy:", incy,
                  "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, work, A, strideA,
                            batch_count);
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, (work + batch_count), y,
                            stridey, batch_count);

    return rocblas_internal_gemv_batched_template(
        handle, transA, m, n, alpha, stride_alpha, cast2constType<T>(work), offseta, lda, strideA,
        x, offsetx, incx, stridex, beta, stride_beta, cast2constPointer<T>(work + batch_count),
        offsety, incy, stridey, batch_count);
}

// trmv
template <typename T>
rocblas_status rocblasCall_trmv(rocblas_handle handle,
                                rocblas_fill uplo,
                                rocblas_operation transa,
                                rocblas_diagonal diag,
                                rocblas_int m,
                                const T* a,
                                rocblas_stride offseta,
                                rocblas_int lda,
                                rocblas_stride stridea,
                                T* x,
                                rocblas_stride offsetx,
                                rocblas_int incx,
                                rocblas_stride stridex,
                                T* w,
                                rocblas_stride stridew,
                                rocblas_int batch_count)
{
    ROCBLAS_ENTER("trmv", "trans:", transa, "diag:", diag, "m:", m, "shiftA:", offseta, "lda:", lda,
                  "shiftX:", offsetx, "incx:", incx, "bc:", batch_count);

    return rocblas_internal_trmv_template(handle, uplo, transa, diag, m, a, offseta, lda, stridea,
                                          x, offsetx, incx, stridex, w, stridew, batch_count);
}

template <typename T>
rocblas_status rocblasCall_trmv(rocblas_handle handle,
                                rocblas_fill uplo,
                                rocblas_operation transa,
                                rocblas_diagonal diag,
                                rocblas_int m,
                                const T* const* a,
                                rocblas_stride offseta,
                                rocblas_int lda,
                                rocblas_stride stridea,
                                T* const* x,
                                rocblas_stride offsetx,
                                rocblas_int incx,
                                rocblas_stride stridex,
                                T* w,
                                rocblas_stride stridew,
                                rocblas_int batch_count)
{
    ROCBLAS_ENTER("trmv", "trans:", transa, "diag:", diag, "m:", m, "shiftA:", offseta, "lda:", lda,
                  "shiftX:", offsetx, "incx:", incx, "bc:", batch_count);

    return rocblas_internal_trmv_batched_template(handle, uplo, transa, diag, m, a, offseta, lda,
                                                  stridea, x, offsetx, incx, stridex, w, stridew,
                                                  batch_count);
}

// gemm - non batched
template <typename T, typename I>
rocblas_status rocblasCall_gemm(rocblas_handle handle,
                                rocblas_operation trans_a,
                                rocblas_operation trans_b,
                                I m,
                                I n,
                                I k,
                                const T* alpha,
                                const T* A,
                                rocblas_stride offset_a,
                                I ld_a,
                                rocblas_stride stride_a,
                                const T* B,
                                rocblas_stride offset_b,
                                I ld_b,
                                rocblas_stride stride_b,
                                const T* beta,
                                T* C,
                                rocblas_stride offset_c,
                                I ld_c,
                                rocblas_stride stride_c,
                                I batch_count,
                                T** work)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("gemm", "transA:", trans_a, "transB:", trans_b, "m:", m, "n:", n, "k:", k,
                  "shiftA:", offset_a, "lda:", ld_a, "shiftB:", offset_b, "ldb:", ld_b,
                  "shiftC:", offset_c, "ldc:", ld_c, "bc:", batch_count);

    if constexpr(std::is_same<I, int64_t>::value)
        return rocblas_internal_gemm_template_64(
            handle, trans_a, trans_b, m, n, k, alpha, A, offset_a, ld_a, stride_a, B, offset_b,
            ld_b, stride_b, beta, C, offset_c, ld_c, stride_c, batch_count);
    else
        return rocblas_internal_gemm_template(handle, trans_a, trans_b, m, n, k, alpha, A, offset_a,
                                              ld_a, stride_a, B, offset_b, ld_b, stride_b, beta, C,
                                              offset_c, ld_c, stride_c, batch_count);
}

// gemm - batched
template <typename T, typename I>
rocblas_status rocblasCall_gemm(rocblas_handle handle,
                                rocblas_operation trans_a,
                                rocblas_operation trans_b,
                                I m,
                                I n,
                                I k,
                                const T* alpha,
                                const T* const* A,
                                rocblas_stride offset_a,
                                I ld_a,
                                rocblas_stride stride_a,
                                const T* const* B,
                                rocblas_stride offset_b,
                                I ld_b,
                                rocblas_stride stride_b,
                                const T* beta,
                                T* const* C,
                                rocblas_stride offset_c,
                                I ld_c,
                                rocblas_stride stride_c,
                                I batch_count,
                                T** work)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("gemm", "transA:", trans_a, "transB:", trans_b, "m:", m, "n:", n, "k:", k,
                  "shiftA:", offset_a, "lda:", ld_a, "shiftB:", offset_b, "ldb:", ld_b,
                  "shiftC:", offset_c, "ldc:", ld_c, "bc:", batch_count);

    if constexpr(std::is_same<I, int64_t>::value)
        return rocblas_internal_gemm_batched_template_64(
            handle, trans_a, trans_b, m, n, k, alpha, A, offset_a, ld_a, stride_a, B, offset_b,
            ld_b, stride_b, beta, C, offset_c, ld_c, stride_c, batch_count);
    else
        return rocblas_internal_gemm_batched_template(
            handle, trans_a, trans_b, m, n, k, alpha, A, offset_a, ld_a, stride_a, B, offset_b,
            ld_b, stride_b, beta, C, offset_c, ld_c, stride_c, batch_count);
}

// gemm overload - batched with strided A
template <typename T, typename I>
rocblas_status rocblasCall_gemm(rocblas_handle handle,
                                rocblas_operation trans_a,
                                rocblas_operation trans_b,
                                I m,
                                I n,
                                I k,
                                const T* alpha,
                                T* A,
                                rocblas_stride offset_a,
                                I ld_a,
                                rocblas_stride stride_a,
                                const T* const* B,
                                rocblas_stride offset_b,
                                I ld_b,
                                rocblas_stride stride_b,
                                const T* beta,
                                T* const* C,
                                rocblas_stride offset_c,
                                I ld_c,
                                rocblas_stride stride_c,
                                I batch_count,
                                T** work)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("gemm", "transA:", trans_a, "transB:", trans_b, "m:", m, "n:", n, "k:", k,
                  "shiftA:", offset_a, "lda:", ld_a, "shiftB:", offset_b, "ldb:", ld_b,
                  "shiftC:", offset_c, "ldc:", ld_c, "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    I blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, work, A, stride_a,
                            batch_count);

    if constexpr(std::is_same<I, int64_t>::value)
        return rocblas_internal_gemm_batched_template_64(
            handle, trans_a, trans_b, m, n, k, alpha, cast2constType<T>(work), offset_a, ld_a,
            stride_a, B, offset_b, ld_b, stride_b, beta, C, offset_c, ld_c, stride_c, batch_count);
    else
        return rocblas_internal_gemm_batched_template(
            handle, trans_a, trans_b, m, n, k, alpha, cast2constType<T>(work), offset_a, ld_a,
            stride_a, B, offset_b, ld_b, stride_b, beta, C, offset_c, ld_c, stride_c, batch_count);
}

// gemm overload - batched with strided B
template <typename T, typename I>
rocblas_status rocblasCall_gemm(rocblas_handle handle,
                                rocblas_operation trans_a,
                                rocblas_operation trans_b,
                                I m,
                                I n,
                                I k,
                                const T* alpha,
                                const T* const* A,
                                rocblas_stride offset_a,
                                I ld_a,
                                rocblas_stride stride_a,
                                T* B,
                                rocblas_stride offset_b,
                                I ld_b,
                                rocblas_stride stride_b,
                                const T* beta,
                                T* const* C,
                                rocblas_stride offset_c,
                                I ld_c,
                                rocblas_stride stride_c,
                                I batch_count,
                                T** work)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("gemm", "transA:", trans_a, "transB:", trans_b, "m:", m, "n:", n, "k:", k,
                  "shiftA:", offset_a, "lda:", ld_a, "shiftB:", offset_b, "ldb:", ld_b,
                  "shiftC:", offset_c, "ldc:", ld_c, "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    I blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, work, B, stride_b,
                            batch_count);

    if constexpr(std::is_same<I, int64_t>::value)
        return rocblas_internal_gemm_batched_template_64(
            handle, trans_a, trans_b, m, n, k, alpha, A, offset_a, ld_a, stride_a,
            cast2constType<T>(work), offset_b, ld_b, stride_b, beta, C, offset_c, ld_c, stride_c,
            batch_count);
    else
        return rocblas_internal_gemm_batched_template(
            handle, trans_a, trans_b, m, n, k, alpha, A, offset_a, ld_a, stride_a,
            cast2constType<T>(work), offset_b, ld_b, stride_b, beta, C, offset_c, ld_c, stride_c,
            batch_count);
}

// gemm overload - batched with strided C
template <typename T, typename I>
rocblas_status rocblasCall_gemm(rocblas_handle handle,
                                rocblas_operation trans_a,
                                rocblas_operation trans_b,
                                I m,
                                I n,
                                I k,
                                const T* alpha,
                                const T* const* A,
                                rocblas_stride offset_a,
                                I ld_a,
                                rocblas_stride stride_a,
                                const T* const* B,
                                rocblas_stride offset_b,
                                I ld_b,
                                rocblas_stride stride_b,
                                const T* beta,
                                T* C,
                                rocblas_stride offset_c,
                                I ld_c,
                                rocblas_stride stride_c,
                                I batch_count,
                                T** work)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("gemm", "transA:", trans_a, "transB:", trans_b, "m:", m, "n:", n, "k:", k,
                  "shiftA:", offset_a, "lda:", ld_a, "shiftB:", offset_b, "ldb:", ld_b,
                  "shiftC:", offset_c, "ldc:", ld_c, "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    I blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, work, C, stride_c,
                            batch_count);

    if constexpr(std::is_same<I, int64_t>::value)
        return rocblas_internal_gemm_batched_template_64(
            handle, trans_a, trans_b, m, n, k, alpha, A, offset_a, ld_a, stride_a, B, offset_b,
            ld_b, stride_b, beta, cast2constPointer(work), offset_c, ld_c, stride_c, batch_count);
    else
        return rocblas_internal_gemm_batched_template(
            handle, trans_a, trans_b, m, n, k, alpha, A, offset_a, ld_a, stride_a, B, offset_b,
            ld_b, stride_b, beta, cast2constPointer(work), offset_c, ld_c, stride_c, batch_count);
}

// gemm overload - batched with strided B and C
template <typename T, typename I>
rocblas_status rocblasCall_gemm(rocblas_handle handle,
                                rocblas_operation trans_a,
                                rocblas_operation trans_b,
                                I m,
                                I n,
                                I k,
                                const T* alpha,
                                const T* const* A,
                                rocblas_stride offset_a,
                                I ld_a,
                                rocblas_stride stride_a,
                                T* B,
                                rocblas_stride offset_b,
                                I ld_b,
                                rocblas_stride stride_b,
                                const T* beta,
                                T* C,
                                rocblas_stride offset_c,
                                I ld_c,
                                rocblas_stride stride_c,
                                I batch_count,
                                T** work)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("gemm", "transA:", trans_a, "transB:", trans_b, "m:", m, "n:", n, "k:", k,
                  "shiftA:", offset_a, "lda:", ld_a, "shiftB:", offset_b, "ldb:", ld_b,
                  "shiftC:", offset_c, "ldc:", ld_c, "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    I blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, work, B, stride_b,
                            batch_count);
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, work + batch_count, C,
                            stride_c, batch_count);

    if constexpr(std::is_same<I, int64_t>::value)
        return rocblas_internal_gemm_batched_template_64(
            handle, trans_a, trans_b, m, n, k, alpha, A, offset_a, ld_a, stride_a,
            cast2constType<T>(work), offset_b, ld_b, stride_b, beta,
            cast2constPointer(work + batch_count), offset_c, ld_c, stride_c, batch_count);
    else
        return rocblas_internal_gemm_batched_template(
            handle, trans_a, trans_b, m, n, k, alpha, A, offset_a, ld_a, stride_a,
            cast2constType<T>(work), offset_b, ld_b, stride_b, beta,
            cast2constPointer(work + batch_count), offset_c, ld_c, stride_c, batch_count);
}

// gemm overload - batched with strided A and C
template <typename T, typename I>
rocblas_status rocblasCall_gemm(rocblas_handle handle,
                                rocblas_operation trans_a,
                                rocblas_operation trans_b,
                                I m,
                                I n,
                                I k,
                                const T* alpha,
                                T* A,
                                rocblas_stride offset_a,
                                I ld_a,
                                rocblas_stride stride_a,
                                const T* const* B,
                                rocblas_stride offset_b,
                                I ld_b,
                                rocblas_stride stride_b,
                                const T* beta,
                                T* C,
                                rocblas_stride offset_c,
                                I ld_c,
                                rocblas_stride stride_c,
                                I batch_count,
                                T** work)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("gemm", "transA:", trans_a, "transB:", trans_b, "m:", m, "n:", n, "k:", k,
                  "shiftA:", offset_a, "lda:", ld_a, "shiftB:", offset_b, "ldb:", ld_b,
                  "shiftC:", offset_c, "ldc:", ld_c, "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    I blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, work, A, stride_a,
                            batch_count);
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, work + batch_count, C,
                            stride_c, batch_count);

    if constexpr(std::is_same<I, int64_t>::value)
        return rocblas_internal_gemm_batched_template_64(
            handle, trans_a, trans_b, m, n, k, alpha, cast2constType<T>(work), offset_a, ld_a,
            stride_a, B, offset_b, ld_b, stride_b, beta, cast2constPointer(work + batch_count),
            offset_c, ld_c, stride_c, batch_count);
    else
        return rocblas_internal_gemm_batched_template(
            handle, trans_a, trans_b, m, n, k, alpha, cast2constType<T>(work), offset_a, ld_a,
            stride_a, B, offset_b, ld_b, stride_b, beta, cast2constPointer(work + batch_count),
            offset_c, ld_c, stride_c, batch_count);
}

// gemm overload - batched with strided A and B
template <typename T, typename I>
rocblas_status rocblasCall_gemm(rocblas_handle handle,
                                rocblas_operation trans_a,
                                rocblas_operation trans_b,
                                I m,
                                I n,
                                I k,
                                const T* alpha,
                                T* A,
                                rocblas_stride offset_a,
                                I ld_a,
                                rocblas_stride stride_a,
                                T* B,
                                rocblas_stride offset_b,
                                I ld_b,
                                rocblas_stride stride_b,
                                const T* beta,
                                T* const* C,
                                rocblas_stride offset_c,
                                I ld_c,
                                rocblas_stride stride_c,
                                I batch_count,
                                T** work)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("gemm", "transA:", trans_a, "transB:", trans_b, "m:", m, "n:", n, "k:", k,
                  "shiftA:", offset_a, "lda:", ld_a, "shiftB:", offset_b, "ldb:", ld_b,
                  "shiftC:", offset_c, "ldc:", ld_c, "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    I blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, work, A, stride_a,
                            batch_count);
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, work + batch_count, B,
                            stride_b, batch_count);

    if constexpr(std::is_same<I, int64_t>::value)
        return rocblas_internal_gemm_batched_template_64(
            handle, trans_a, trans_b, m, n, k, alpha, cast2constType<T>(work), offset_a, ld_a,
            stride_a, cast2constType<T>(work + batch_count), offset_b, ld_b, stride_b, beta, C,
            offset_c, ld_c, stride_c, batch_count);
    else
        return rocblas_internal_gemm_batched_template(
            handle, trans_a, trans_b, m, n, k, alpha, cast2constType<T>(work), offset_a, ld_a,
            stride_a, cast2constType<T>(work + batch_count), offset_b, ld_b, stride_b, beta, C,
            offset_c, ld_c, stride_c, batch_count);
}

// trmm
template <typename T>
rocblas_status rocblasCall_trmm(rocblas_handle handle,
                                rocblas_side side,
                                rocblas_fill uplo,
                                rocblas_operation transA,
                                rocblas_diagonal diag,
                                rocblas_int m,
                                rocblas_int n,
                                const T* alpha,
                                rocblas_stride stride_alpha,
                                const T* A,
                                rocblas_stride offsetA,
                                rocblas_int lda,
                                rocblas_stride strideA,
                                T* B,
                                rocblas_stride offsetB,
                                rocblas_int ldb,
                                rocblas_stride strideB,
                                rocblas_int batch_count,
                                T** workArr = nullptr)
{
    // TODO: How to get alpha for trace logging
    ROCBLAS_ENTER("trmm", "side:", side, "uplo:", uplo, "trans:", transA, "diag:", diag, "m:", m,
                  "n:", n, "shiftA:", offsetA, "lda:", lda, "shiftB:", offsetB, "ldb:", ldb,
                  "bc:", batch_count);

    return rocblas_internal_trmm_template(handle, side, uplo, transA, diag, m, n, alpha, stride_alpha,
                                          A, offsetA, lda, strideA, cast2constType<T>(B), offsetB,
                                          ldb, strideB, B, offsetB, ldb, strideB, batch_count);
}

template <typename T>
rocblas_status rocblasCall_trmm(rocblas_handle handle,
                                rocblas_side side,
                                rocblas_fill uplo,
                                rocblas_operation transA,
                                rocblas_diagonal diag,
                                rocblas_int m,
                                rocblas_int n,
                                const T* alpha,
                                rocblas_stride stride_alpha,
                                const T* const* A,
                                rocblas_stride offsetA,
                                rocblas_int lda,
                                rocblas_stride strideA,
                                T* const* B,
                                rocblas_stride offsetB,
                                rocblas_int ldb,
                                rocblas_stride strideB,
                                rocblas_int batch_count,
                                T** workArr = nullptr)
{
    // TODO: How to get alpha for trace logging
    ROCBLAS_ENTER("trmm", "side:", side, "uplo:", uplo, "trans:", transA, "diag:", diag, "m:", m,
                  "n:", n, "shiftA:", offsetA, "lda:", lda, "shiftB:", offsetB, "ldb:", ldb,
                  "bc:", batch_count);

    return rocblas_internal_trmm_batched_template(
        handle, side, uplo, transA, diag, m, n, alpha, stride_alpha, A, offsetA, lda, strideA,
        cast2constType<T>(B), offsetB, ldb, strideB, B, offsetB, ldb, strideB, batch_count);
}

// trmm overload
template <typename T>
rocblas_status rocblasCall_trmm(rocblas_handle handle,
                                rocblas_side side,
                                rocblas_fill uplo,
                                rocblas_operation transA,
                                rocblas_diagonal diag,
                                rocblas_int m,
                                rocblas_int n,
                                const T* alpha,
                                rocblas_stride stride_alpha,
                                const T* const* A,
                                rocblas_stride offsetA,
                                rocblas_int lda,
                                rocblas_stride strideA,
                                T* B,
                                rocblas_stride offsetB,
                                rocblas_int ldb,
                                rocblas_stride strideB,
                                rocblas_int batch_count,
                                T** workArr)
{
    // TODO: How to get alpha for trace logging
    ROCBLAS_ENTER("trmm", "side:", side, "uplo:", uplo, "trans:", transA, "diag:", diag, "m:", m,
                  "n:", n, "shiftA:", offsetA, "lda:", lda, "shiftB:", offsetB, "ldb:", ldb,
                  "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    rocblas_int blocks = (batch_count - 1) / 256 + 1;

    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, workArr, B, strideB,
                            batch_count);

    return rocblas_internal_trmm_batched_template(
        handle, side, uplo, transA, diag, m, n, alpha, stride_alpha, A, offsetA, lda, strideA,
        cast2constType<T>(workArr), offsetB, ldb, strideB, cast2constPointer<T>(workArr), offsetB,
        ldb, strideB, batch_count);
}

// syr2/her2
template <typename T>
rocblas_status rocblasCall_syr2_her2(rocblas_handle handle,
                                     rocblas_fill uplo,
                                     rocblas_int n,
                                     const T* alpha,
                                     const T* x,
                                     rocblas_stride offsetx,
                                     rocblas_int incx,
                                     rocblas_stride stridex,
                                     const T* y,
                                     rocblas_stride offsety,
                                     rocblas_int incy,
                                     rocblas_stride stridey,
                                     T* A,
                                     rocblas_stride offsetA,
                                     rocblas_int lda,
                                     rocblas_stride strideA,
                                     rocblas_int batch_count,
                                     T** work)
{
    constexpr auto name = rocblas_is_complex<T> ? "her2" : "syr2";
    // TODO: How to get alpha for trace logging
    ROCBLAS_ENTER(name, "uplo:", uplo, "n:", n, "shiftX:", offsetx, "incx:", incx, "shiftY:", offsety,
                  "incy:", incy, "shiftA:", offsetA, "lda:", lda, "bc:", batch_count);

    if constexpr(!rocblas_is_complex<T>)
        return rocblas_internal_syr2_template(handle, uplo, n, alpha, x, offsetx, incx, stridex, y,
                                              offsety, incy, stridey, A, lda, offsetA, strideA,
                                              batch_count);
    else
        return rocblas_internal_her2_template(handle, uplo, n, alpha, x, offsetx, incx, stridex, y,
                                              offsety, incy, stridey, A, lda, offsetA, strideA,
                                              batch_count);
}

// syr2/her2 batched
template <typename T>
rocblas_status rocblasCall_syr2_her2(rocblas_handle handle,
                                     rocblas_fill uplo,
                                     rocblas_int n,
                                     const T* alpha,
                                     const T* const* x,
                                     rocblas_stride offsetx,
                                     rocblas_int incx,
                                     rocblas_stride stridex,
                                     const T* const* y,
                                     rocblas_stride offsety,
                                     rocblas_int incy,
                                     rocblas_stride stridey,
                                     T* const* A,
                                     rocblas_stride offsetA,
                                     rocblas_int lda,
                                     rocblas_stride strideA,
                                     rocblas_int batch_count,
                                     T** work)
{
    constexpr auto name = rocblas_is_complex<T> ? "her2" : "syr2";
    // TODO: How to get alpha for trace logging
    ROCBLAS_ENTER(name, "uplo:", uplo, "n:", n, "shiftX:", offsetx, "incx:", incx, "shiftY:", offsety,
                  "incy:", incy, "shiftA:", offsetA, "lda:", lda, "bc:", batch_count);

    if constexpr(!rocblas_is_complex<T>)
        return rocblas_internal_syr2_batched_template(handle, uplo, n, alpha, x, offsetx, incx,
                                                      stridex, y, offsety, incy, stridey, A, lda,
                                                      offsetA, strideA, batch_count);
    else
        return rocblas_internal_her2_batched_template(handle, uplo, n, alpha, x, offsetx, incx,
                                                      stridex, y, offsety, incy, stridey, A, lda,
                                                      offsetA, strideA, batch_count);
}

// syr2/her2 overload - complex with strided y
template <typename T>
rocblas_status rocblasCall_syr2_her2(rocblas_handle handle,
                                     rocblas_fill uplo,
                                     rocblas_int n,
                                     const T* alpha,
                                     const T* const* x,
                                     rocblas_stride offsetx,
                                     rocblas_int incx,
                                     rocblas_stride stridex,
                                     T* y,
                                     rocblas_stride offsety,
                                     rocblas_int incy,
                                     rocblas_stride stridey,
                                     T* const* A,
                                     rocblas_stride offsetA,
                                     rocblas_int lda,
                                     rocblas_stride strideA,
                                     rocblas_int batch_count,
                                     T** work)
{
    constexpr auto name = rocblas_is_complex<T> ? "her2" : "syr2";
    // TODO: How to get alpha for trace logging
    ROCBLAS_ENTER(name, "uplo:", uplo, "n:", n, "shiftX:", offsetx, "incx:", incx, "shiftY:", offsety,
                  "incy:", incy, "shiftA:", offsetA, "lda:", lda, "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, work, y, stridey,
                            batch_count);

    if constexpr(!rocblas_is_complex<T>)
        return rocblas_internal_syr2_batched_template(handle, uplo, n, alpha, x, offsetx, incx,
                                                      stridex, work, offsety, incy, stridey, A, lda,
                                                      offsetA, strideA, batch_count);
    else
        return rocblas_internal_her2_batched_template(handle, uplo, n, alpha, x, offsetx, incx,
                                                      stridex, work, offsety, incy, stridey, A, lda,
                                                      offsetA, strideA, batch_count);
}

// syrk
template <bool BATCHED, typename T, typename U, typename V, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
rocblas_status rocblasCall_syrk_herk(rocblas_handle handle,
                                     rocblas_fill uplo,
                                     rocblas_operation transA,
                                     rocblas_int n,
                                     rocblas_int k,
                                     U alpha,
                                     V A,
                                     rocblas_stride offsetA,
                                     rocblas_int lda,
                                     rocblas_stride strideA,
                                     U beta,
                                     V C,
                                     rocblas_stride offsetC,
                                     rocblas_int ldc,
                                     rocblas_stride strideC,
                                     rocblas_int batch_count)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("syrk", "uplo:", uplo, "trans:", transA, "n:", n, "k:", k, "shiftA:", offsetA,
                  "lda:", lda, "shiftC:", offsetC, "ldc:", ldc, "bc:", batch_count);

    using S = decltype(std::real(T{}));

    if constexpr(BATCHED)
        return rocblas_internal_syrk_batched_template(
            handle, uplo, transA, n, k, cast2constType<S>(alpha), cast2constType<T>(A), offsetA,
            lda, strideA, cast2constType<S>(beta), C, offsetC, ldc, strideC, batch_count);
    else
        return rocblas_internal_syrk_template(
            handle, uplo, transA, n, k, cast2constType<S>(alpha), cast2constType<T>(A), offsetA,
            lda, strideA, cast2constType<S>(beta), C, offsetC, ldc, strideC, batch_count);
}

// herk
template <bool BATCHED, typename T, typename U, typename V, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
rocblas_status rocblasCall_syrk_herk(rocblas_handle handle,
                                     rocblas_fill uplo,
                                     rocblas_operation transA,
                                     rocblas_int n,
                                     rocblas_int k,
                                     U alpha,
                                     V A,
                                     rocblas_stride offsetA,
                                     rocblas_int lda,
                                     rocblas_stride strideA,
                                     U beta,
                                     V C,
                                     rocblas_stride offsetC,
                                     rocblas_int ldc,
                                     rocblas_stride strideC,
                                     rocblas_int batch_count)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("herk", "uplo:", uplo, "trans:", transA, "n:", n, "k:", k, "shiftA:", offsetA,
                  "lda:", lda, "shiftC:", offsetC, "ldc:", ldc, "bc:", batch_count);

    using S = decltype(std::real(T{}));

    if constexpr(BATCHED)
        return rocblas_internal_herk_batched_template(
            handle, uplo, transA, n, k, cast2constType<S>(alpha), cast2constType<T>(A), offsetA,
            lda, strideA, cast2constType<S>(beta), C, offsetC, ldc, strideC, batch_count);
    else
        return rocblas_internal_herk_template(
            handle, uplo, transA, n, k, cast2constType<S>(alpha), cast2constType<T>(A), offsetA,
            lda, strideA, cast2constType<S>(beta), C, offsetC, ldc, strideC, batch_count);
}

// syr2k
template <bool BATCHED,
          typename T,
          typename Ua,
          typename Ub,
          typename V,
          std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
rocblas_status rocblasCall_syr2k_her2k(rocblas_handle handle,
                                       rocblas_fill uplo,
                                       rocblas_operation trans,
                                       rocblas_int n,
                                       rocblas_int k,
                                       Ua alpha,
                                       V A,
                                       rocblas_stride offsetA,
                                       rocblas_int lda,
                                       rocblas_stride strideA,
                                       V B,
                                       rocblas_stride offsetB,
                                       rocblas_int ldb,
                                       rocblas_stride strideB,
                                       Ub beta,
                                       V C,
                                       rocblas_stride offsetC,
                                       rocblas_int ldc,
                                       rocblas_stride strideC,
                                       rocblas_int batch_count,
                                       T** work = nullptr)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("syr2k", "uplo:", uplo, "trans:", trans, "n:", n, "k:", k, "shiftA:", offsetA,
                  "lda:", lda, "shiftB:", offsetB, "ldb:", ldb, "shiftC:", offsetC, "ldc:", ldc,
                  "bc:", batch_count);

    if constexpr(BATCHED)
        return rocblas_internal_syr2k_batched_template(
            handle, uplo, trans, n, k, cast2constType<T>(alpha), cast2constType<T>(A), offsetA, lda,
            strideA, cast2constType<T>(B), offsetB, ldb, strideB, cast2constType<T>(beta), C,
            offsetC, ldc, strideC, batch_count);
    else
        return rocblas_internal_syr2k_template(
            handle, uplo, trans, n, k, cast2constType<T>(alpha), cast2constType<T>(A), offsetA, lda,
            strideA, cast2constType<T>(B), offsetB, ldb, strideB, cast2constType<T>(beta), C,
            offsetC, ldc, strideC, batch_count);
}

// syr2k overload
template <bool BATCHED, typename T, typename Ua, typename Ub, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
rocblas_status rocblasCall_syr2k_her2k(rocblas_handle handle,
                                       rocblas_fill uplo,
                                       rocblas_operation trans,
                                       rocblas_int n,
                                       rocblas_int k,
                                       Ua alpha,
                                       T* const A[],
                                       rocblas_stride offsetA,
                                       rocblas_int lda,
                                       rocblas_stride strideA,
                                       T* B,
                                       rocblas_stride offsetB,
                                       rocblas_int ldb,
                                       rocblas_stride strideB,
                                       Ub beta,
                                       T* const C[],
                                       rocblas_stride offsetC,
                                       rocblas_int ldc,
                                       rocblas_stride strideC,
                                       rocblas_int batch_count,
                                       T** work = nullptr)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("syr2k", "uplo:", uplo, "trans:", trans, "n:", n, "k:", k, "shiftA:", offsetA,
                  "lda:", lda, "shiftB:", offsetB, "ldb:", ldb, "shiftC:", offsetC, "ldc:", ldc,
                  "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, work, B, strideB,
                            batch_count);

    if constexpr(BATCHED)
        return rocblas_internal_syr2k_batched_template(
            handle, uplo, trans, n, k, cast2constType<T>(alpha), cast2constType<T>(A), offsetA, lda,
            strideA, cast2constType<T>(work), offsetB, ldb, strideB, cast2constType<T>(beta), C,
            offsetC, ldc, strideC, batch_count);
    else
        return rocblas_internal_syr2k_template(
            handle, uplo, trans, n, k, cast2constType<T>(alpha), cast2constType<T>(A), offsetA, lda,
            strideA, cast2constType<T>(work), offsetB, ldb, strideB, cast2constType<T>(beta), C,
            offsetC, ldc, strideC, batch_count);
}

// her2k
template <bool BATCHED,
          typename T,
          typename Ua,
          typename Ub,
          typename V,
          std::enable_if_t<rocblas_is_complex<T>, int> = 0>
rocblas_status rocblasCall_syr2k_her2k(rocblas_handle handle,
                                       rocblas_fill uplo,
                                       rocblas_operation trans,
                                       rocblas_int n,
                                       rocblas_int k,
                                       Ua alpha,
                                       V A,
                                       rocblas_stride offsetA,
                                       rocblas_int lda,
                                       rocblas_stride strideA,
                                       V B,
                                       rocblas_stride offsetB,
                                       rocblas_int ldb,
                                       rocblas_stride strideB,
                                       Ub beta,
                                       V C,
                                       rocblas_stride offsetC,
                                       rocblas_int ldc,
                                       rocblas_stride strideC,
                                       rocblas_int batch_count,
                                       T** work = nullptr)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("her2k", "uplo:", uplo, "trans:", trans, "n:", n, "k:", k, "shiftA:", offsetA,
                  "lda:", lda, "shiftB:", offsetB, "ldb:", ldb, "shiftC:", offsetC, "ldc:", ldc,
                  "bc:", batch_count);

    using S = decltype(std::real(T{}));

    if constexpr(BATCHED)
        return rocblas_internal_her2k_batched_template(
            handle, uplo, trans, n, k, cast2constType<T>(alpha), cast2constType<T>(A), offsetA, lda,
            strideA, cast2constType<T>(B), offsetB, ldb, strideB, cast2constType<S>(beta), C,
            offsetC, ldc, strideC, batch_count);
    else
        return rocblas_internal_her2k_template(
            handle, uplo, trans, n, k, cast2constType<T>(alpha), cast2constType<T>(A), offsetA, lda,
            strideA, cast2constType<T>(B), offsetB, ldb, strideB, cast2constType<S>(beta), C,
            offsetC, ldc, strideC, batch_count);
}

// her2k overload
template <bool BATCHED, typename T, typename Ua, typename Ub, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
rocblas_status rocblasCall_syr2k_her2k(rocblas_handle handle,
                                       rocblas_fill uplo,
                                       rocblas_operation trans,
                                       rocblas_int n,
                                       rocblas_int k,
                                       Ua alpha,
                                       T* const A[],
                                       rocblas_stride offsetA,
                                       rocblas_int lda,
                                       rocblas_stride strideA,
                                       T* B,
                                       rocblas_stride offsetB,
                                       rocblas_int ldb,
                                       rocblas_stride strideB,
                                       Ub beta,
                                       T* const C[],
                                       rocblas_stride offsetC,
                                       rocblas_int ldc,
                                       rocblas_stride strideC,
                                       rocblas_int batch_count,
                                       T** work = nullptr)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("her2k", "uplo:", uplo, "trans:", trans, "n:", n, "k:", k, "shiftA:", offsetA,
                  "lda:", lda, "shiftB:", offsetB, "ldb:", ldb, "shiftC:", offsetC, "ldc:", ldc,
                  "bc:", batch_count);

    using S = decltype(std::real(T{}));

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, work, B, strideB,
                            batch_count);

    if constexpr(BATCHED)
        return rocblas_internal_her2k_batched_template(
            handle, uplo, trans, n, k, cast2constType<T>(alpha), cast2constType<T>(A), offsetA, lda,
            strideA, cast2constType<T>(work), offsetB, ldb, strideB, cast2constType<S>(beta), C,
            offsetC, ldc, strideC, batch_count);
    else
        return rocblas_internal_her2k_template(
            handle, uplo, trans, n, k, cast2constType<T>(alpha), cast2constType<T>(A), offsetA, lda,
            strideA, cast2constType<T>(work), offsetB, ldb, strideB, cast2constType<S>(beta), C,
            offsetC, ldc, strideC, batch_count);
}

// symv/hemv memory sizes
template <bool BATCHED, typename T>
void rocblasCall_symv_hemv_mem(rocblas_int n, rocblas_int batch_count, size_t* w_temp)
{
    *w_temp = rocblas_internal_hemv_symv_kernel_workspace_size<T>(n, batch_count);
}

// symv/hemv
template <typename T>
rocblas_status rocblasCall_symv_hemv(rocblas_handle handle,
                                     rocblas_fill uplo,
                                     rocblas_int n,
                                     const T* alpha,
                                     rocblas_stride stridea,
                                     const T* A,
                                     rocblas_stride offsetA,
                                     rocblas_int lda,
                                     rocblas_stride strideA,
                                     const T* x,
                                     rocblas_stride offsetx,
                                     rocblas_int incx,
                                     rocblas_stride stridex,
                                     const T* beta,
                                     rocblas_stride strideb,
                                     T* y,
                                     rocblas_stride offsety,
                                     rocblas_int incy,
                                     rocblas_stride stridey,
                                     rocblas_int batch_count,
                                     T* work,
                                     T** workArr)
{
    constexpr auto name = rocblas_is_complex<T> ? "hemv" : "symv";
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER(name, "uplo:", uplo, "n:", n, "shiftA:", offsetA, "lda:", lda, "shiftX:", offsetx,
                  "incx:", incx, "shiftY:", offsety, "incy:", incy, "bc:", batch_count);

    if constexpr(!rocblas_is_complex<T>)
        return rocblas_internal_symv_template(handle, uplo, n, alpha, stridea, A, offsetA, lda,
                                              strideA, x, offsetx, incx, stridex, beta, strideb, y,
                                              offsety, incy, stridey, batch_count, work);
    else
        return rocblas_internal_hemv_template(handle, uplo, n, alpha, stridea, A, offsetA, lda,
                                              strideA, x, offsetx, incx, stridex, beta, strideb, y,
                                              offsety, incy, stridey, batch_count, work);
}

// symv/hemv batched
template <typename T>
rocblas_status rocblasCall_symv_hemv(rocblas_handle handle,
                                     rocblas_fill uplo,
                                     rocblas_int n,
                                     const T* alpha,
                                     rocblas_stride stridea,
                                     const T* const* A,
                                     rocblas_stride offsetA,
                                     rocblas_int lda,
                                     rocblas_stride strideA,
                                     const T* const* x,
                                     rocblas_stride offsetx,
                                     rocblas_int incx,
                                     rocblas_stride stridex,
                                     const T* beta,
                                     rocblas_stride strideb,
                                     T* const* y,
                                     rocblas_stride offsety,
                                     rocblas_int incy,
                                     rocblas_stride stridey,
                                     rocblas_int batch_count,
                                     T* work,
                                     T** workArr)
{
    constexpr auto name = rocblas_is_complex<T> ? "hemv" : "symv";
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER(name, "uplo:", uplo, "n:", n, "shiftA:", offsetA, "lda:", lda, "shiftX:", offsetx,
                  "incx:", incx, "shiftY:", offsety, "incy:", incy, "bc:", batch_count);

    if constexpr(!rocblas_is_complex<T>)
        return rocblas_internal_symv_batched_template(
            handle, uplo, n, alpha, stridea, A, offsetA, lda, strideA, x, offsetx, incx, stridex,
            beta, strideb, y, offsety, incy, stridey, batch_count, work);
    else
        return rocblas_internal_hemv_batched_template(
            handle, uplo, n, alpha, stridea, A, offsetA, lda, strideA, x, offsetx, incx, stridex,
            beta, strideb, y, offsety, incy, stridey, batch_count, work);
}

// symv/hemv overload - batched with strided y
template <typename T>
rocblas_status rocblasCall_symv_hemv(rocblas_handle handle,
                                     rocblas_fill uplo,
                                     rocblas_int n,
                                     const T* alpha,
                                     rocblas_stride stridea,
                                     const T* const* A,
                                     rocblas_stride offsetA,
                                     rocblas_int lda,
                                     rocblas_stride strideA,
                                     const T* const* x,
                                     rocblas_stride offsetx,
                                     rocblas_int incx,
                                     rocblas_stride stridex,
                                     const T* beta,
                                     rocblas_stride strideb,
                                     T* y,
                                     rocblas_stride offsety,
                                     rocblas_int incy,
                                     rocblas_stride stridey,
                                     rocblas_int batch_count,
                                     T* work,
                                     T** workArr)
{
    constexpr auto name = rocblas_is_complex<T> ? "hemv" : "symv";
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER(name, "uplo:", uplo, "n:", n, "shiftA:", offsetA, "lda:", lda, "shiftX:", offsetx,
                  "incx:", incx, "shiftY:", offsety, "incy:", incy, "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, workArr, y, stridey,
                            batch_count);

    if constexpr(!rocblas_is_complex<T>)
        return rocblas_internal_symv_batched_template(handle, uplo, n, alpha, stridea, A, offsetA,
                                                      lda, strideA, x, offsetx, incx, stridex, beta,
                                                      strideb, cast2constPointer<T>(workArr),
                                                      offsety, incy, stridey, batch_count, work);
    else
        return rocblas_internal_hemv_batched_template(handle, uplo, n, alpha, stridea, A, offsetA,
                                                      lda, strideA, x, offsetx, incx, stridex, beta,
                                                      strideb, cast2constPointer<T>(workArr),
                                                      offsety, incy, stridey, batch_count, work);
}

// symm/hemm
template <typename T>
rocblas_status rocblasCall_symm_hemm(rocblas_handle handle,
                                     rocblas_side side,
                                     rocblas_fill uplo,
                                     rocblas_int m,
                                     rocblas_int n,
                                     const T* alpha,
                                     const T* A,
                                     rocblas_stride offsetA,
                                     rocblas_int lda,
                                     rocblas_stride strideA,
                                     const T* B,
                                     rocblas_stride offsetB,
                                     rocblas_int ldb,
                                     rocblas_stride strideB,
                                     const T* beta,
                                     T* C,
                                     rocblas_stride offsetC,
                                     rocblas_int ldc,
                                     rocblas_stride strideC,
                                     rocblas_int batch_count)
{
    constexpr auto name = rocblas_is_complex<T> ? "hemm" : "symm";
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER(name, "side:", side, "uplo:", uplo, "m:", m, "n:", n, "shiftA:", offsetA,
                  "lda:", lda, "shiftB:", offsetB, "ldb:", ldb, "shiftC:", offsetC, "ldc:", ldc,
                  "bc:", batch_count);

    if constexpr(!rocblas_is_complex<T>)
        return rocblas_internal_symm_template(handle, side, uplo, m, n, alpha, A, offsetA, lda,
                                              strideA, B, offsetB, ldb, strideB, beta, C, offsetC,
                                              ldc, strideC, batch_count);
    else
        return rocblas_internal_hemm_template(handle, side, uplo, m, n, alpha, A, offsetA, lda,
                                              strideA, B, offsetB, ldb, strideB, beta, C, offsetC,
                                              ldc, strideC, batch_count);
}

// symm/hemm batched
template <typename T>
rocblas_status rocblasCall_symm_hemm(rocblas_handle handle,
                                     rocblas_side side,
                                     rocblas_fill uplo,
                                     rocblas_int m,
                                     rocblas_int n,
                                     const T* alpha,
                                     const T* const* A,
                                     rocblas_stride offsetA,
                                     rocblas_int lda,
                                     rocblas_stride strideA,
                                     const T* const* B,
                                     rocblas_stride offsetB,
                                     rocblas_int ldb,
                                     rocblas_stride strideB,
                                     const T* beta,
                                     T* const* C,
                                     rocblas_stride offsetC,
                                     rocblas_int ldc,
                                     rocblas_stride strideC,
                                     rocblas_int batch_count)
{
    constexpr auto name = rocblas_is_complex<T> ? "hemm" : "symm";
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER(name, "side:", side, "uplo:", uplo, "m:", m, "n:", n, "shiftA:", offsetA,
                  "lda:", lda, "shiftB:", offsetB, "ldb:", ldb, "shiftC:", offsetC, "ldc:", ldc,
                  "bc:", batch_count);

    if constexpr(!rocblas_is_complex<T>)
        return rocblas_internal_symm_batched_template(handle, side, uplo, m, n, alpha, A, offsetA,
                                                      lda, strideA, B, offsetB, ldb, strideB, beta,
                                                      C, offsetC, ldc, strideC, batch_count);
    else
        return rocblas_internal_hemm_batched_template(handle, side, uplo, m, n, alpha, A, offsetA,
                                                      lda, strideA, B, offsetB, ldb, strideB, beta,
                                                      C, offsetC, ldc, strideC, batch_count);
}

// trsv
template <typename T>
rocblas_status rocblasCall_trsv(rocblas_handle handle,
                                rocblas_fill uplo,
                                rocblas_operation transA,
                                rocblas_diagonal diag,
                                rocblas_int m,
                                const T* A,
                                rocblas_stride offset_A,
                                rocblas_int lda,
                                rocblas_stride stride_A,
                                T* x,
                                rocblas_stride offset_x,
                                rocblas_int incx,
                                rocblas_stride stride_x,
                                rocblas_int batch_count,
                                rocblas_int* w_completed_sec,
                                T** workArr = nullptr)
{
    ROCBLAS_ENTER("trsv", "uplo:", uplo, "trans:", transA, "diag:", diag, "m:", m,
                  "shiftA:", offset_A, "lda:", lda, "shiftx:", offset_x, "incx:", incx,
                  "bc:", batch_count);

    return rocblas_internal_trsv_template(handle, uplo, transA, diag, m, A, offset_A, lda, stride_A,
                                          x, offset_x, incx, stride_x, batch_count, w_completed_sec);
}

// batched trsv
template <typename T>
rocblas_status rocblasCall_trsv(rocblas_handle handle,
                                rocblas_fill uplo,
                                rocblas_operation transA,
                                rocblas_diagonal diag,
                                rocblas_int m,
                                const T* const* A,
                                rocblas_stride offset_A,
                                rocblas_int lda,
                                rocblas_stride stride_A,
                                T* const* x,
                                rocblas_stride offset_x,
                                rocblas_int incx,
                                rocblas_stride stride_x,
                                rocblas_int batch_count,
                                rocblas_int* w_completed_sec,
                                T** workArr = nullptr)
{
    ROCBLAS_ENTER("trsv", "uplo:", uplo, "trans:", transA, "diag:", diag, "m:", m,
                  "shiftA:", offset_A, "lda:", lda, "shiftx:", offset_x, "incx:", incx,
                  "bc:", batch_count);

    return rocblas_internal_trsv_batched_template(handle, uplo, transA, diag, m, A, offset_A, lda,
                                                  stride_A, x, offset_x, incx, stride_x,
                                                  batch_count, w_completed_sec);
}

// trsm memory sizes
template <bool BATCHED, typename T, typename I>
rocblas_status rocblasCall_trsm_mem(rocblas_side side,
                                    rocblas_operation transA,
                                    I m,
                                    I n,
                                    I lda,
                                    I ldb,
                                    I batch_count,
                                    size_t* x_temp,
                                    size_t* x_temp_arr,
                                    size_t* invA,
                                    size_t* invA_arr)
{
    size_t no_opt_size = 0;
    /** TODO: For now, we always request the size for optimal performance.
        no_opt_size could be used in the future if we generalize the use of
        rocblas_workmode parameter **/

    // can't infer batched based on input params
    if constexpr(std::is_same<I, int64_t>::value)
    {
        if constexpr(BATCHED)
            return rocblas_internal_trsm_batched_workspace_size_64<T>(
                side, transA, m, n, lda, ldb, batch_count, 0, x_temp, x_temp_arr, invA, invA_arr,
                &no_opt_size);
        else
            return rocblas_internal_trsm_workspace_size_64<T>(side, transA, m, n, lda, ldb,
                                                              batch_count, 0, x_temp, x_temp_arr,
                                                              invA, invA_arr, &no_opt_size);
    }
    else
    {
        if constexpr(BATCHED)
            return rocblas_internal_trsm_batched_workspace_size<T>(side, transA, m, n, batch_count,
                                                                   0, x_temp, x_temp_arr, invA,
                                                                   invA_arr, &no_opt_size);
        else
            return rocblas_internal_trsm_workspace_size<T>(side, transA, m, n, batch_count, 0, x_temp,
                                                           x_temp_arr, invA, invA_arr, &no_opt_size);
    }
}

// trsm
template <typename T, typename I>
rocblas_status rocblasCall_trsm(rocblas_handle handle,
                                rocblas_side side,
                                rocblas_fill uplo,
                                rocblas_operation transA,
                                rocblas_diagonal diag,
                                I m,
                                I n,
                                const T* alpha,
                                const T* A,
                                rocblas_stride offset_A,
                                I lda,
                                rocblas_stride stride_A,
                                T* B,
                                rocblas_stride offset_B,
                                I ldb,
                                rocblas_stride stride_B,
                                I batch_count,
                                bool optimal_mem,
                                void* x_temp,
                                void* x_temp_arr,
                                void* invA,
                                void* invA_arr,
                                T** workArr = nullptr)
{
    // TODO: How to get alpha for trace logging
    ROCBLAS_ENTER("trsm", "side:", side, "uplo:", uplo, "trans:", transA, "diag:", diag, "m:", m,
                  "n:", n, "shiftA:", offset_A, "lda:", lda, "shiftB:", offset_B, "ldb:", ldb,
                  "bc:", batch_count);

    const T* supplied_invA = nullptr;
    if constexpr(std::is_same<I, int64_t>::value)
        return rocblas_internal_trsm_template_64(handle, side, uplo, transA, diag, m, n, alpha, A,
                                                 offset_A, lda, stride_A, B, offset_B, ldb,
                                                 stride_B, batch_count, optimal_mem, x_temp,
                                                 x_temp_arr, invA, invA_arr, supplied_invA, 0);
    else
        return rocblas_internal_trsm_template(handle, side, uplo, transA, diag, m, n, alpha, A,
                                              offset_A, lda, stride_A, B, offset_B, ldb, stride_B,
                                              batch_count, optimal_mem, x_temp, x_temp_arr, invA,
                                              invA_arr, supplied_invA, 0);
}

// batched trsm
template <typename T, typename I>
rocblas_status rocblasCall_trsm(rocblas_handle handle,
                                rocblas_side side,
                                rocblas_fill uplo,
                                rocblas_operation transA,
                                rocblas_diagonal diag,
                                I m,
                                I n,
                                const T* alpha,
                                const T* const* A,
                                rocblas_stride offset_A,
                                I lda,
                                rocblas_stride stride_A,
                                T* const* B,
                                rocblas_stride offset_B,
                                I ldb,
                                rocblas_stride stride_B,
                                I batch_count,
                                bool optimal_mem,
                                void* x_temp,
                                void* x_temp_arr,
                                void* invA,
                                void* invA_arr,
                                T** workArr = nullptr)
{
    // TODO: How to get alpha for trace logging
    ROCBLAS_ENTER("trsm", "side:", side, "uplo:", uplo, "trans:", transA, "diag:", diag, "m:", m,
                  "n:", n, "shiftA:", offset_A, "lda:", lda, "shiftB:", offset_B, "ldb:", ldb,
                  "bc:", batch_count);

    const T* const* supplied_invA = nullptr;
    if constexpr(std::is_same<I, int64_t>::value)
        return rocblas_internal_trsm_batched_template_64(
            handle, side, uplo, transA, diag, m, n, alpha, A, offset_A, lda, stride_A, B, offset_B,
            ldb, stride_B, batch_count, optimal_mem, x_temp, x_temp_arr, invA, invA_arr,
            supplied_invA, 0);
    else
        return rocblas_internal_trsm_batched_template(handle, side, uplo, transA, diag, m, n, alpha,
                                                      A, offset_A, lda, stride_A, B, offset_B, ldb,
                                                      stride_B, batch_count, optimal_mem, x_temp,
                                                      x_temp_arr, invA, invA_arr, supplied_invA, 0);
}

// trsm overload
template <typename T, typename I>
rocblas_status rocblasCall_trsm(rocblas_handle handle,
                                rocblas_side side,
                                rocblas_fill uplo,
                                rocblas_operation transA,
                                rocblas_diagonal diag,
                                I m,
                                I n,
                                const T* alpha,
                                T* A,
                                rocblas_stride offset_A,
                                I lda,
                                rocblas_stride stride_A,
                                T* const B[],
                                rocblas_stride offset_B,
                                I ldb,
                                rocblas_stride stride_B,
                                I batch_count,
                                bool optimal_mem,
                                void* x_temp,
                                void* x_temp_arr,
                                void* invA,
                                void* invA_arr,
                                T** workArr)
{
    // TODO: How to get alpha for trace logging
    ROCBLAS_ENTER("trsm", "side:", side, "uplo:", uplo, "trans:", transA, "diag:", diag, "m:", m,
                  "n:", n, "shiftA:", offset_A, "lda:", lda, "shiftB:", offset_B, "ldb:", ldb,
                  "bc:", batch_count);

    using U = const T* const*;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    I blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, workArr, A, stride_A,
                            batch_count);

    U supplied_invA = nullptr;
    if constexpr(std::is_same<I, int64_t>::value)
        return rocblas_internal_trsm_batched_template_64(
            handle, side, uplo, transA, diag, m, n, alpha, cast2constType((U)workArr), offset_A,
            lda, stride_A, B, offset_B, ldb, stride_B, batch_count, optimal_mem, x_temp, x_temp_arr,
            invA, invA_arr, supplied_invA, 0);
    else
        return rocblas_internal_trsm_batched_template(
            handle, side, uplo, transA, diag, m, n, alpha, cast2constType((U)workArr), offset_A,
            lda, stride_A, B, offset_B, ldb, stride_B, batch_count, optimal_mem, x_temp, x_temp_arr,
            invA, invA_arr, supplied_invA, 0);
}

// trtri memory sizes
template <bool BATCHED, typename T>
void rocblasCall_trtri_mem(rocblas_int n, rocblas_int batch_count, size_t* c_temp, size_t* c_temp_arr)
{
    size_t c_temp_els = rocblas_internal_trtri_temp_elements(n, batch_count);
    *c_temp = c_temp_els * sizeof(T);

    *c_temp_arr = BATCHED ? sizeof(T*) * batch_count : 0;
}

// trtri
template <typename T>
rocblas_status rocblasCall_trtri(rocblas_handle handle,
                                 rocblas_fill uplo,
                                 rocblas_diagonal diag,
                                 rocblas_int n,
                                 const T* A,
                                 rocblas_stride offset_A,
                                 rocblas_int lda,
                                 rocblas_stride stride_A,
                                 T* invA,
                                 rocblas_stride offset_invA,
                                 rocblas_int ldinvA,
                                 rocblas_stride stride_invA,
                                 rocblas_int batch_count,
                                 T* c_temp,
                                 T** c_temp_arr,
                                 T** workArr)
{
    ROCBLAS_ENTER("trtri", "uplo:", uplo, "diag:", diag, "n:", n, "shiftA:", offset_A, "lda:", lda,
                  "shiftC:", offset_invA, "ldc:", ldinvA, "bc:", batch_count);

    return rocblas_internal_trtri_template(handle, uplo, diag, n, A, offset_A, lda, stride_A, 0,
                                           invA, offset_invA, ldinvA, stride_invA, 0, batch_count,
                                           1, c_temp);
}

// batched trtri
template <typename T>
rocblas_status rocblasCall_trtri(rocblas_handle handle,
                                 rocblas_fill uplo,
                                 rocblas_diagonal diag,
                                 rocblas_int n,
                                 const T* const* A,
                                 rocblas_stride offset_A,
                                 rocblas_int lda,
                                 rocblas_stride stride_A,
                                 T* const* invA,
                                 rocblas_stride offset_invA,
                                 rocblas_int ldinvA,
                                 rocblas_stride stride_invA,
                                 rocblas_int batch_count,
                                 T* const* c_temp,
                                 T** c_temp_arr,
                                 T** workArr)
{
    ROCBLAS_ENTER("trtri", "uplo:", uplo, "diag:", diag, "n:", n, "shiftA:", offset_A, "lda:", lda,
                  "shiftC:", offset_invA, "ldc:", ldinvA, "bc:", batch_count);

    return rocblas_internal_trtri_batched_template(handle, uplo, diag, n, A, offset_A, lda,
                                                   stride_A, 0, invA, offset_invA, ldinvA,
                                                   stride_invA, 0, batch_count, 1, c_temp);
}

// trtri overload
template <typename T>
rocblas_status rocblasCall_trtri(rocblas_handle handle,
                                 rocblas_fill uplo,
                                 rocblas_diagonal diag,
                                 rocblas_int n,
                                 const T* const A[],
                                 rocblas_stride offset_A,
                                 rocblas_int lda,
                                 rocblas_stride stride_A,
                                 T* const invA[],
                                 rocblas_stride offset_invA,
                                 rocblas_int ldinvA,
                                 rocblas_stride stride_invA,
                                 rocblas_int batch_count,
                                 T* c_temp,
                                 T** c_temp_arr,
                                 T** workArr)
{
    ROCBLAS_ENTER("trtri", "uplo:", uplo, "diag:", diag, "n:", n, "shiftA:", offset_A, "lda:", lda,
                  "shiftC:", offset_invA, "ldc:", ldinvA, "bc:", batch_count);

    size_t c_temp_els = rocblas_internal_trtri_temp_elements(n, 1);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, c_temp_arr, c_temp,
                            c_temp_els, batch_count);

    return rocblas_internal_trtri_template(handle, uplo, diag, n, A, offset_A, lda, stride_A, 0,
                                           invA, offset_invA, ldinvA, stride_invA, 0, batch_count,
                                           1, cast2constPointer(c_temp_arr));
}

// trtri overload
template <typename T>
rocblas_status rocblasCall_trtri(rocblas_handle handle,
                                 rocblas_fill uplo,
                                 rocblas_diagonal diag,
                                 rocblas_int n,
                                 const T* const A[],
                                 rocblas_stride offset_A,
                                 rocblas_int lda,
                                 rocblas_stride stride_A,
                                 T* invA,
                                 rocblas_stride offset_invA,
                                 rocblas_int ldinvA,
                                 rocblas_stride stride_invA,
                                 rocblas_int batch_count,
                                 T* c_temp,
                                 T** c_temp_arr,
                                 T** workArr)
{
    ROCBLAS_ENTER("trtri", "uplo:", uplo, "diag:", diag, "n:", n, "shiftA:", offset_A, "lda:", lda,
                  "shiftC:", offset_invA, "ldc:", ldinvA, "bc:", batch_count);

    size_t c_temp_els = rocblas_internal_trtri_temp_elements(n, 1);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, workArr, invA,
                            stride_invA, batch_count);
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, c_temp_arr, c_temp,
                            c_temp_els, batch_count);

    return rocblas_internal_trtri_batched_template(
        handle, uplo, diag, n, A, offset_A, lda, stride_A, 0, cast2constPointer(workArr),
        offset_invA, ldinvA, stride_invA, 0, batch_count, 1, cast2constPointer(c_temp_arr));
}

ROCSOLVER_END_NAMESPACE
