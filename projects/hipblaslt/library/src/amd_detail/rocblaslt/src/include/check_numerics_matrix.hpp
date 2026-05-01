/* ************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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

/*! \file
 * \brief Post-GEMM NaN scanner for hipblasLtMatmul output (D matrix).
 *        Enabled by HIPBLASLT_CHECK_NUMERICS env var (read once in handle ctor).
 */

#pragma once
#ifndef HIPBLASLT_CHECK_NUMERICS_MATRIX_HPP
#define HIPBLASLT_CHECK_NUMERICS_MATRIX_HPP

#include "auxiliary.hpp"
#include "handle.h"
#include "rocblaslt-types.h"
#include "utility.hpp"

#include <algorithm>
#include <hip/hip_runtime.h>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

#ifndef LEGACY_HIPBLAS_DIRECT
#include <hipblas-common/hipblas-common.h>
#else
#include <hipblas/hipblas.h>
#endif

// One thread per element of D. Sets *flag = 1 (via atomicOr) when a NaN is found.
// batch_base lets the host chunk grid-z to stay under HIP's 65535 limit.
//
// Strided-batched form: D is a single base pointer; batch i lives at
// D + i*stride_d. Used by hipblasLtMatrixLayout's strided batch mode.
template <int DIM_X, int DIM_Y, typename T>
__global__ void hipblaslt_check_nan_kernel(int64_t               m,
                                           int64_t               n,
                                           const T* __restrict__ D,
                                           int64_t               ldd,
                                           int64_t               stride_d,
                                           int                   row_major,
                                           int32_t               batch_base,
                                           uint32_t* __restrict__ flag)
{
    int64_t tx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t ty = blockIdx.y * (int64_t)blockDim.y + threadIdx.y;

    if(tx < m && ty < n)
    {
        const T*      batch_D = D + (int64_t)(batch_base + (int32_t)blockIdx.z) * stride_d;
        const int64_t offset  = row_major ? (tx * ldd + ty) : (tx + ldd * ty);
        if(hipblaslt_isnan(batch_D[offset]))
            atomicOr(flag, 1u);
    }
}

// Pointer-array batched form: batch_D_ptrs[i] is the device-resident base
// pointer for batch i. Used by Tensile's batchD field when the matmul caller
// supplied a pointer array instead of a strided base+stride pair. Null entries
// in the array are skipped (treated as "no batch present").
template <int DIM_X, int DIM_Y, typename T>
__global__ void hipblaslt_check_nan_kernel_ptr_array(
    int64_t                            m,
    int64_t                            n,
    T const* const* __restrict__       batch_D_ptrs,
    int64_t                            ldd,
    int                                row_major,
    int32_t                            batch_base,
    uint32_t* __restrict__             flag)
{
    int64_t tx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t ty = blockIdx.y * (int64_t)blockDim.y + threadIdx.y;

    if(tx < m && ty < n)
    {
        const T* batch_D = batch_D_ptrs[batch_base + (int32_t)blockIdx.z];
        if(batch_D == nullptr)
            return;
        const int64_t offset = row_major ? (tx * ldd + ty) : (tx + ldd * ty);
        if(hipblaslt_isnan(batch_D[offset]))
            atomicOr(flag, 1u);
    }
}

template <typename T>
inline rocblaslt_status hipblaslt_launch_nan_kernel(int64_t     m,
                                                    int64_t     n,
                                                    int32_t     batch,
                                                    const void* D,
                                                    int64_t     ldd,
                                                    int64_t     stride_d,
                                                    bool        row_major,
                                                    uint32_t*   d_flag,
                                                    hipStream_t stream)
{
    constexpr int     DIM_X       = 16;
    constexpr int32_t MAX_GRID_Z  = 65535;

    dim3 threads(DIM_X, DIM_X);
    const unsigned grid_x = (unsigned)((m + DIM_X - 1) / DIM_X);
    const unsigned grid_y = (unsigned)((n + DIM_X - 1) / DIM_X);

    // Chunk batch over grid-z to avoid the 65535 hardware cap.
    for(int32_t base = 0; base < batch; base += MAX_GRID_Z)
    {
        const int32_t  this_batch = std::min<int32_t>(MAX_GRID_Z, batch - base);
        dim3           blocks(grid_x, grid_y, (unsigned)this_batch);

        hipLaunchKernelGGL((hipblaslt_check_nan_kernel<DIM_X, DIM_X, T>),
                           blocks, threads, 0, stream,
                           m, n,
                           reinterpret_cast<const T*>(D),
                           ldd, stride_d,
                           row_major ? 1 : 0,
                           base,
                           d_flag);

        if(hipGetLastError() != hipSuccess)
            return rocblaslt_status_internal_error;
    }
    return rocblaslt_status_success;
}

// Counterpart launcher for the pointer-array kernel. batch_D_ptrs must be a
// device-accessible array of length >= batch holding the per-batch device
// base pointers. Same grid-z chunking as the strided launcher.
template <typename T>
inline rocblaslt_status hipblaslt_launch_nan_kernel_ptr_array(int64_t            m,
                                                              int64_t            n,
                                                              int32_t            batch,
                                                              void const* const* batch_D_ptrs,
                                                              int64_t            ldd,
                                                              bool               row_major,
                                                              uint32_t*          d_flag,
                                                              hipStream_t        stream)
{
    constexpr int     DIM_X      = 16;
    constexpr int32_t MAX_GRID_Z = 65535;

    dim3           threads(DIM_X, DIM_X);
    const unsigned grid_x = (unsigned)((m + DIM_X - 1) / DIM_X);
    const unsigned grid_y = (unsigned)((n + DIM_X - 1) / DIM_X);

    for(int32_t base = 0; base < batch; base += MAX_GRID_Z)
    {
        const int32_t this_batch = std::min<int32_t>(MAX_GRID_Z, batch - base);
        dim3          blocks(grid_x, grid_y, (unsigned)this_batch);

        hipLaunchKernelGGL((hipblaslt_check_nan_kernel_ptr_array<DIM_X, DIM_X, T>),
                           blocks,
                           threads,
                           0,
                           stream,
                           m,
                           n,
                           reinterpret_cast<T const* const*>(batch_D_ptrs),
                           ldd,
                           row_major ? 1 : 0,
                           base,
                           d_flag);

        if(hipGetLastError() != hipSuccess)
            return rocblaslt_status_internal_error;
    }
    return rocblaslt_status_success;
}

// Public entry point. Called from rocblaslt_matmul_impl() after a successful matmul.
//
// Two batching layouts are supported:
//   - Strided-batched: pass D (single base ptr) + stride_d. batch_D_ptrs is nullptr.
//   - Pointer-array:   pass batch_D_ptrs (device-resident array of per-batch base
//                      ptrs). D and stride_d are ignored on this path.
// batch_D_ptrs defaults to nullptr so existing strided-only callers compile
// unchanged.
inline rocblaslt_status hipblaslt_check_numerics_output_D(const char*                   fn,
                                                          hipStream_t                   stream,
                                                          int64_t                       m,
                                                          int64_t                       n,
                                                          int64_t                       k,
                                                          int32_t                       batch,
                                                          hipDataType                   type_d,
                                                          const void*                   D,
                                                          int64_t                       ldd,
                                                          int64_t                       stride_d,
                                                          bool                          row_major,
                                                          hipblasOperation_t            opA,
                                                          hipblasOperation_t            opB,
                                                          rocblaslt_epilogue            epilogue,
                                                          int64_t                       algo_index,
                                                          const std::string&            solution_name,
                                                          const std::string&            kernel_name,
                                                          hipblaslt_check_numerics_mode mode,
                                                          void const* const*            batch_D_ptrs = nullptr)
{
    const bool ptr_array = (batch_D_ptrs != nullptr);
    // Early exit: nothing to scan if the caller-supplied buffer/array is null
    // for the chosen layout, or if any dimension is zero.
    if((!ptr_array && !D) || (ptr_array && !batch_D_ptrs)
       || m == 0 || n == 0 || batch == 0)
        return rocblaslt_status_success;

    // hipStreamSynchronize is illegal during HIP graph capture -- skip the
    // scan silently so enabling HIPBLASLT_CHECK_NUMERICS doesn't break
    // customer code that builds graphs around hipblasLtMatmul.
    hipStreamCaptureStatus cap = hipStreamCaptureStatusNone;
    if(hipStreamIsCapturing(stream, &cap) == hipSuccess
       && cap != hipStreamCaptureStatusNone)
        return rocblaslt_status_success;

    // Best-effort: if the scanner's own bookkeeping alloc fails, do not
    // surface that as a matmul failure -- the matmul itself already succeeded.
    uint32_t* d_flag = nullptr;
    if(hipMalloc(&d_flag, sizeof(uint32_t)) != hipSuccess)
        return rocblaslt_status_success;
    if(hipMemsetAsync(d_flag, 0, sizeof(uint32_t), stream) != hipSuccess)
    {
        static_cast<void>(hipFree(d_flag));
        return rocblaslt_status_success;
    }

    rocblaslt_status launch_st = rocblaslt_status_success;
    // Per-dtype dispatch. Each case picks the strided launcher (single base D
    // + stride_d) or the pointer-array launcher (device array of per-batch
    // base ptrs) based on which layout the caller passed.
    switch(type_d)
    {
    case HIP_R_32F:
        launch_st = ptr_array
            ? hipblaslt_launch_nan_kernel_ptr_array<float>(
                  m, n, batch, batch_D_ptrs, ldd, row_major, d_flag, stream)
            : hipblaslt_launch_nan_kernel<float>(
                  m, n, batch, D, ldd, stride_d, row_major, d_flag, stream);
        break;
    case HIP_R_64F:
        launch_st = ptr_array
            ? hipblaslt_launch_nan_kernel_ptr_array<double>(
                  m, n, batch, batch_D_ptrs, ldd, row_major, d_flag, stream)
            : hipblaslt_launch_nan_kernel<double>(
                  m, n, batch, D, ldd, stride_d, row_major, d_flag, stream);
        break;
    case HIP_R_16F:
        launch_st = ptr_array
            ? hipblaslt_launch_nan_kernel_ptr_array<hipblasLtHalf>(
                  m, n, batch, batch_D_ptrs, ldd, row_major, d_flag, stream)
            : hipblaslt_launch_nan_kernel<hipblasLtHalf>(
                  m, n, batch, D, ldd, stride_d, row_major, d_flag, stream);
        break;
    case HIP_R_16BF:
        launch_st = ptr_array
            ? hipblaslt_launch_nan_kernel_ptr_array<hip_bfloat16>(
                  m, n, batch, batch_D_ptrs, ldd, row_major, d_flag, stream)
            : hipblaslt_launch_nan_kernel<hip_bfloat16>(
                  m, n, batch, D, ldd, stride_d, row_major, d_flag, stream);
        break;
    case HIP_R_8F_E4M3_FNUZ:
        launch_st = ptr_array
            ? hipblaslt_launch_nan_kernel_ptr_array<hipblaslt_f8_fnuz>(
                  m, n, batch, batch_D_ptrs, ldd, row_major, d_flag, stream)
            : hipblaslt_launch_nan_kernel<hipblaslt_f8_fnuz>(
                  m, n, batch, D, ldd, stride_d, row_major, d_flag, stream);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        launch_st = ptr_array
            ? hipblaslt_launch_nan_kernel_ptr_array<hipblaslt_bf8_fnuz>(
                  m, n, batch, batch_D_ptrs, ldd, row_major, d_flag, stream)
            : hipblaslt_launch_nan_kernel<hipblaslt_bf8_fnuz>(
                  m, n, batch, D, ldd, stride_d, row_major, d_flag, stream);
        break;
    case HIP_R_8F_E4M3:
        launch_st = ptr_array
            ? hipblaslt_launch_nan_kernel_ptr_array<hipblaslt_f8>(
                  m, n, batch, batch_D_ptrs, ldd, row_major, d_flag, stream)
            : hipblaslt_launch_nan_kernel<hipblaslt_f8>(
                  m, n, batch, D, ldd, stride_d, row_major, d_flag, stream);
        break;
    case HIP_R_8F_E5M2:
        launch_st = ptr_array
            ? hipblaslt_launch_nan_kernel_ptr_array<hipblaslt_bf8>(
                  m, n, batch, batch_D_ptrs, ldd, row_major, d_flag, stream)
            : hipblaslt_launch_nan_kernel<hipblaslt_bf8>(
                  m, n, batch, D, ldd, stride_d, row_major, d_flag, stream);
        break;
    default:
        // Skipped silently:
        //   - Integer types (HIP_R_8I, HIP_R_32I): cannot represent NaN by
        //     construction, so a NaN scan is a no-op rather than a missing
        //     check. Customers running int GEMMs are not "missing" anything.
        //   - Sub-byte packed extension types (HIP_R_6F_*, HIP_R_4F_*,
        //     HIP_R_8F_UE8M0, HIP_R_8F_E5M3_EXT): no scalar hipblaslt_isnan
        //     overload exists and the kernel templates over a scalar T, so
        //     these need a separate packed-aware kernel before they can be
        //     supported.
        static_cast<void>(hipFree(d_flag));
        return rocblaslt_status_success;
    }

    if(launch_st != rocblaslt_status_success)
    {
        static_cast<void>(hipFree(d_flag));
        return launch_st;
    }

    uint32_t h_flag = 0;
    if(hipMemcpyAsync(&h_flag, d_flag, sizeof(uint32_t), hipMemcpyDeviceToHost, stream)
           != hipSuccess
       || hipStreamSynchronize(stream) != hipSuccess)
    {
        static_cast<void>(hipFree(d_flag));
        return rocblaslt_status_internal_error;
    }
    static_cast<void>(hipFree(d_flag));

    const bool has_nan  = (h_flag != 0);
    const bool do_print = (mode & hipblaslt_check_numerics_mode_info)
                          || ((mode & hipblaslt_check_numerics_mode_warn) && has_nan);

    if(do_print)
    {
        std::ostringstream os;
        os << "[hipBLASLt CHECK_NUMERICS] " << fn << " :- Output D :"
           << " has_NaN=" << has_nan
           << " | shape m=" << m << " n=" << n << " k=" << k << " batch=" << batch
           << " dtype=" << hipDataType_to_string(type_d)
           << " order=" << (row_major ? "row" : "col")
           << " ldd=" << ldd
           << (ptr_array ? " D=ptr_array@" : " D=")
           << (ptr_array ? static_cast<const void*>(batch_D_ptrs) : D)
           << " | opA=" << hipblasOperation_to_string(opA)
           << " opB=" << hipblasOperation_to_string(opB)
           << " epilogue=" << rocblaslt_epilogue_to_string(epilogue)
           << " | algo_idx=" << algo_index
           << " solution=" << (solution_name.empty() ? "<unknown>" : solution_name)
           << " kernel=" << (kernel_name.empty() ? "<unknown>" : kernel_name);

        std::lock_guard<std::mutex> lk(log_mutex);
        std::ostream*               sink = get_logger_os();
        if(!sink)
            sink = &std::cerr;
        *sink << os.str() << std::endl;
    }

    if(has_nan && (mode & hipblaslt_check_numerics_mode_fail))
        return rocblaslt_status_check_numerics_fail;

    return rocblaslt_status_success;
}

#endif // HIPBLASLT_CHECK_NUMERICS_MATRIX_HPP
