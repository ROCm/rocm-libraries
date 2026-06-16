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
#include <chrono>
#include <cstdio>
#include <ctime>
#include <hip/hip_runtime.h>
#include <iostream>
#include <mutex>

#ifndef LEGACY_HIPBLAS_DIRECT
#include <hipblas-common/hipblas-common.h>
#else
#include <hipblas/hipblas.h>
#endif

// Wall-clock timestamp prefix on every CHECK_NUMERICS log line so users can
// correlate with their wall-clock-stamped training logs.
inline std::string hipblaslt_check_numerics_ts()
{
    using namespace std::chrono;
    const auto now    = system_clock::now();
    const auto t      = system_clock::to_time_t(now);
    const auto ms_part
        = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[40];
    std::snprintf(buf, sizeof(buf),
                  "[%04d-%02d-%02d %02d:%02d:%02d.%03lld]",
                  tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                  tm.tm_hour, tm.tm_min, tm.tm_sec,
                  static_cast<long long>(ms_part.count()));
    return std::string(buf);
}

// One thread per element of D. The first NaN seen claims the device flag with
// this call's id via atomicCAS(0, call_id); subsequent NaNs no-op so the slot
// retains the FIRST call_id that observed a NaN.
template <int DIM_X, int DIM_Y, typename T>
__global__ void hipblaslt_check_nan_kernel(int64_t               m,
                                           int64_t               n,
                                           const T* __restrict__ D,
                                           int64_t               ldd,
                                           int64_t               stride_d,
                                           int                   row_major,
                                           int32_t               batch_base,
                                           uint32_t* __restrict__ flag,
                                           uint32_t              call_id)
{
    int64_t tx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t ty = blockIdx.y * (int64_t)blockDim.y + threadIdx.y;

    if(tx < m && ty < n)
    {
        const T*      batch_D = D + (int64_t)(batch_base + (int32_t)blockIdx.z) * stride_d;
        const int64_t offset  = row_major ? (tx * ldd + ty) : (tx + ldd * ty);
        if(hipblaslt_isnan(batch_D[offset]))
            atomicCAS(flag, 0u, call_id);
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
                                                    uint32_t    call_id,
                                                    hipStream_t stream)
{
    constexpr int     DIM_X      = 16;
    constexpr int32_t MAX_GRID_Z = 65535;

    dim3           threads(DIM_X, DIM_X);
    const unsigned grid_x = (unsigned)((m + DIM_X - 1) / DIM_X);
    const unsigned grid_y = (unsigned)((n + DIM_X - 1) / DIM_X);

    // Chunk batch over grid-z to avoid the 65535 hardware cap.
    for(int32_t base = 0; base < batch; base += MAX_GRID_Z)
    {
        const int32_t this_batch = std::min<int32_t>(MAX_GRID_Z, batch - base);
        dim3          blocks(grid_x, grid_y, (unsigned)this_batch);

        hipLaunchKernelGGL((hipblaslt_check_nan_kernel<DIM_X, DIM_X, T>),
                           blocks, threads, 0, stream,
                           m, n,
                           reinterpret_cast<const T*>(D),
                           ldd, stride_d,
                           row_major ? 1 : 0,
                           base,
                           d_flag,
                           call_id);

        if(hipGetLastError() != hipSuccess)
            return rocblaslt_status_internal_error;
    }
    return rocblaslt_status_success;
}

// d_flag: caller-owned 4-byte device slot. Holds the call_id of the FIRST
// scanned matmul with a NaN in D (0 = none). nullptr disables scanning.
inline rocblaslt_status hipblaslt_check_numerics_output_D(hipStream_t stream,
                                                          int64_t     m,
                                                          int64_t     n,
                                                          int32_t     batch,
                                                          hipDataType type_d,
                                                          const void* D,
                                                          int64_t     ldd,
                                                          int64_t     stride_d,
                                                          bool        row_major,
                                                          uint32_t*   d_flag,
                                                          uint32_t    call_id,
                                                          uint32_t    scan_every,
                                                          uint32_t    scan_from,
                                                          uint32_t    scan_until)
{
    if(!d_flag || !D || m == 0 || n == 0 || batch == 0)
        return rocblaslt_status_success;
    if(call_id < scan_from || call_id > scan_until)
        return rocblaslt_status_success;
    const uint32_t every = scan_every ? scan_every : 1u;
    if((call_id % every) != 0)
        return rocblaslt_status_success;

    // Skip during HIP graph capture: the deferred drain happens at handle
    // teardown and cannot be sequenced into a captured graph.
    hipStreamCaptureStatus cap = hipStreamCaptureStatusNone;
    if(hipStreamIsCapturing(stream, &cap) == hipSuccess
       && cap != hipStreamCaptureStatusNone)
        return rocblaslt_status_success;

    // Default arm silently skips integers and sub-byte packed types (no
    // scalar isnan overload).
    switch(type_d)
    {
    case HIP_R_32F:
        return hipblaslt_launch_nan_kernel<float>(
            m, n, batch, D, ldd, stride_d, row_major, d_flag, call_id, stream);
    case HIP_R_64F:
        return hipblaslt_launch_nan_kernel<double>(
            m, n, batch, D, ldd, stride_d, row_major, d_flag, call_id, stream);
    case HIP_R_16F:
        return hipblaslt_launch_nan_kernel<hipblasLtHalf>(
            m, n, batch, D, ldd, stride_d, row_major, d_flag, call_id, stream);
    case HIP_R_16BF:
        return hipblaslt_launch_nan_kernel<hip_bfloat16>(
            m, n, batch, D, ldd, stride_d, row_major, d_flag, call_id, stream);
    case HIP_R_8F_E4M3_FNUZ:
        return hipblaslt_launch_nan_kernel<hipblaslt_f8_fnuz>(
            m, n, batch, D, ldd, stride_d, row_major, d_flag, call_id, stream);
    case HIP_R_8F_E5M2_FNUZ:
        return hipblaslt_launch_nan_kernel<hipblaslt_bf8_fnuz>(
            m, n, batch, D, ldd, stride_d, row_major, d_flag, call_id, stream);
    case HIP_R_8F_E4M3:
        return hipblaslt_launch_nan_kernel<hipblaslt_f8>(
            m, n, batch, D, ldd, stride_d, row_major, d_flag, call_id, stream);
    case HIP_R_8F_E5M2:
        return hipblaslt_launch_nan_kernel<hipblaslt_bf8>(
            m, n, batch, D, ldd, stride_d, row_major, d_flag, call_id, stream);
    default:
        return rocblaslt_status_success;
    }
}

// Pure logging helper -- emits the standard CHECK_NUMERICS log line for a
// given h_flag value (0 = no NaN observed). Shared by both the drain helper
// (which obtains h_flag via sync+memcpy from the device flag) and scan_D's
// auto-drain path on STOP_ON_FIRST host-peek (which already has h_flag from
// the mapped flag and must NOT issue a sync on the matmul hot path).
inline void hipblaslt_log_check_numerics_window(uint32_t                       h_flag,
                                                hipblaslt_check_numerics_mode  mode,
                                                uint32_t                       window_lo,
                                                uint32_t                       window_hi,
                                                uint32_t                       scan_every,
                                                uint32_t                       scan_from,
                                                uint32_t                       scan_until,
                                                const char*                    label,
                                                bool                           stop_on_first)
{
    const bool log_anything = (mode
                               & (hipblaslt_check_numerics_mode_info
                                  | hipblaslt_check_numerics_mode_warn))
                              != 0;
    if(!log_anything)
        return;

    const uint32_t every_eff = scan_every ? scan_every : 1u;
    // Effective window: clamp the requested window to what the SCAN_FROM/UNTIL
    // gates could actually have inspected, so the printed bounds never lie
    // outside the slice the scanner saw.
    const uint32_t eff_lo   = std::max(window_lo, scan_from);
    const uint32_t eff_hi   = std::min(window_hi, scan_until);
    const bool     sampling = (every_eff > 1u);

    std::lock_guard<std::mutex> lk(log_mutex);
    std::ostream*               sink = get_logger_os();
    if(!sink)
        sink = &std::cerr;
    if(h_flag != 0)
    {
        // With sampling, the true first NaN sits in (prev_sampled, h_flag];
        // emit a bisect hint clamped to scan_from.
        uint32_t bisect_lo = 0, bisect_hi = 0;
        bool     have_bisect = false;
        if(sampling)
        {
            const uint32_t prev_sampled = ((h_flag - 1u) / every_eff) * every_eff;
            bisect_lo   = std::max<uint32_t>(prev_sampled + 1u, scan_from);
            bisect_hi   = h_flag;
            have_bisect = (bisect_lo < bisect_hi);
        }

        *sink << hipblaslt_check_numerics_ts()
              << "[hipBLASLt CHECK_NUMERICS] " << label
              << ": first NaN observed at sampled matmul call #" << h_flag;
        if(have_bisect)
            *sink << " (true first NaN somewhere in (" << (bisect_lo - 1u)
                  << ".." << bisect_hi << "] due to scan_every=" << every_eff << ")";
        *sink << ", effective window [" << eff_lo << ".." << eff_hi << "]"
              << ", mode=" << static_cast<int>(mode)
              << ", scan_every=" << every_eff << ".";
        if(have_bisect)
            *sink << " To bisect, re-run with HIPBLASLT_CHECK_NUMERICS_SCAN_FROM="
                  << bisect_lo << " HIPBLASLT_CHECK_NUMERICS_SCAN_UNTIL="
                  << bisect_hi << " HIPBLASLT_CHECK_NUMERICS_SCAN_EVERY=1.";
        if(stop_on_first)
            *sink << " (STOP_ON_FIRST: further scans suppressed after this call.)";
        *sink << std::endl;
    }
    else if(mode & hipblaslt_check_numerics_mode_info)
    {
        *sink << hipblaslt_check_numerics_ts()
              << "[hipBLASLt CHECK_NUMERICS] " << label
              << ": no NaN observed in effective window ["
              << eff_lo << ".." << eff_hi << "]"
              << " (mode=" << static_cast<int>(mode)
              << ", scan_every=" << every_eff;
        if(sampling && eff_lo <= eff_hi)
            *sink << " -- sampled 1 in " << every_eff
                  << " calls; NaNs in unsampled calls would be missed."
                  << " Re-run with HIPBLASLT_CHECK_NUMERICS_SCAN_EVERY=1 to confirm";
        *sink << ")." << std::endl;
    }
}

// Drain helper: device-wide sync, read the flag, reset it (unless STOP_ON_FIRST
// has fired -- the slot is sticky so re-drains keep returning the first id),
// and log the per-window result. Returns the first NaN call_id (0 = none).
// Errors swallowed: a benign drain failure must not take down the matmul stream.
inline uint32_t hipblaslt_drain_check_numerics_window(uint32_t* d_flag,
                                                     hipblaslt_check_numerics_mode mode,
                                                     uint32_t    window_lo,
                                                     uint32_t    window_hi,
                                                     uint32_t    scan_every,
                                                     uint32_t    scan_from,
                                                     uint32_t    scan_until,
                                                     const char* label,
                                                     bool                   stop_on_first,
                                                     std::atomic<bool>*     short_circuit_out)
{
    if(!d_flag)
        return 0u;

    uint32_t h_flag = 0;
    static_cast<void>(hipDeviceSynchronize());
    static_cast<void>(hipMemcpy(&h_flag, d_flag, sizeof(uint32_t), hipMemcpyDeviceToHost));

    // Sticky on STOP_ON_FIRST after the first NaN: don't reset, and trip the
    // cross-call short-circuit so subsequent scan_D calls bail immediately.
    const bool sticky = stop_on_first && h_flag != 0u;
    if(!sticky)
        static_cast<void>(hipMemset(d_flag, 0, sizeof(uint32_t)));

    // CAS-elect logger to avoid double-emission when the scan_D host-peek
    // auto-drain has already logged this same first-NaN event. Only the
    // path that flips short_circuit false->true is the canonical logger;
    // any other path observing short_circuit already true skips the log
    // line. Restricted to the sticky case: in non-STOP_ON_FIRST mode the
    // device flag is reset on every drain and per-window logging remains
    // the documented behavior.
    bool should_log = true;
    if(sticky && short_circuit_out)
    {
        bool expected = false;
        should_log    = short_circuit_out->compare_exchange_strong(
            expected, true, std::memory_order_acq_rel);
    }

    if(should_log)
        hipblaslt_log_check_numerics_window(h_flag, mode, window_lo, window_hi,
                                            scan_every, scan_from, scan_until,
                                            label, stop_on_first);
    return h_flag;
}

// Increment + return the per-handle call counter. Returns 0 when scanning is
// disabled, which doubles as the "skip the whole hook" signal for callers.
inline uint32_t hipblaslt_check_numerics_begin_call(rocblaslt_handle handle)
{
    if(!handle || !handle->check_numerics)
        return 0u;
    return handle->check_numerics_call_id.fetch_add(1, std::memory_order_relaxed) + 1u;
}

// Run the scanner on one D buffer. For grouped GEMM, call once per sub-problem
// with a shared call_id.
inline rocblaslt_status hipblaslt_check_numerics_scan_D(rocblaslt_handle handle,
                                                        hipStream_t      stream,
                                                        uint32_t         call_id,
                                                        int64_t          m,
                                                        int64_t          n,
                                                        int32_t          batch,
                                                        hipDataType      type_d,
                                                        const void*      D,
                                                        int64_t          ldd,
                                                        int64_t          stride_d,
                                                        bool             row_major)
{
    if(!handle || !handle->check_numerics)
        return rocblaslt_status_success;

    // STOP_ON_FIRST short-circuit. Sticky bypass first.
    if(handle->check_numerics_short_circuit.load(std::memory_order_acquire))
        return rocblaslt_status_success;

    // Best-effort host peek of the mapped flag. RELAXED is sufficient: the
    // load does NOT establish host<->device happens-before -- on PCIe parts
    // the device-side atomicCAS may not become host-visible until the next
    // stream/device sync. A miss here just means the next explicit drain
    // (or handle teardown) will observe the flag instead.
    if(handle->check_numerics_stop_on_first && handle->check_numerics_flag_host)
    {
        const uint32_t h_flag
            = __atomic_load_n(handle->check_numerics_flag_host, __ATOMIC_RELAXED);
        if(h_flag != 0u)
        {
            // CAS-elect a single logger thread. The winner emits the
            // auto-drain log line directly from the mapped flag (no
            // hipDeviceSynchronize on the matmul hot path); losers fall
            // through to the shared `return` below, then bail on the
            // sticky short_circuit check on their next call. The shared
            // return is intentional -- both winner and loser exit through
            // the same line so neither path leaks into scan_D's normal
            // scanning code path after the short-circuit fires.
            bool expected = false;
            if(handle->check_numerics_short_circuit.compare_exchange_strong(
                   expected, true, std::memory_order_acq_rel))
            {
                const uint32_t window_hi
                    = handle->check_numerics_call_id.load(std::memory_order_relaxed);
                hipblaslt_log_check_numerics_window(
                    h_flag,
                    handle->check_numerics,
                    1u,
                    window_hi,
                    handle->check_numerics_scan_every,
                    handle->check_numerics_scan_from,
                    handle->check_numerics_scan_until,
                    "auto-drain on host peek",
                    true);
            }
            return rocblaslt_status_success;
        }
    }

    return hipblaslt_check_numerics_output_D(stream,
                                             m, n, batch, type_d,
                                             D, ldd, stride_d, row_major,
                                             handle->check_numerics_flag,
                                             call_id,
                                             handle->check_numerics_scan_every,
                                             handle->check_numerics_scan_from,
                                             handle->check_numerics_scan_until);
}

// Public on-demand drain (hipblasLtCheckNumericsDrain). Forces sync+read+reset
// at a caller-chosen point. Returns the first NaN call_id since the last drain
// (0 = none).
inline uint32_t hipblaslt_check_numerics_drain_handle(rocblaslt_handle handle)
{
    if(!handle || !handle->check_numerics_flag)
        return 0u;
    const uint32_t window_hi
        = handle->check_numerics_call_id.load(std::memory_order_relaxed);
    int prev_device = -1;
    static_cast<void>(hipGetDevice(&prev_device));
    if(prev_device != handle->device)
        static_cast<void>(hipSetDevice(handle->device));
    const uint32_t result = hipblaslt_drain_check_numerics_window(
        handle->check_numerics_flag, handle->check_numerics,
        1u, window_hi,
        handle->check_numerics_scan_every,
        handle->check_numerics_scan_from,
        handle->check_numerics_scan_until,
        "on-demand drain",
        handle->check_numerics_stop_on_first,
        &handle->check_numerics_short_circuit);
    if(prev_device != -1 && prev_device != handle->device)
        static_cast<void>(hipSetDevice(prev_device));
    return result;
}

#endif // HIPBLASLT_CHECK_NUMERICS_MATRIX_HPP
