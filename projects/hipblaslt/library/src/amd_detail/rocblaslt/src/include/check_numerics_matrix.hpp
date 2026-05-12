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
#include <sstream>
#include <string>

#ifndef LEGACY_HIPBLAS_DIRECT
#include <hipblas-common/hipblas-common.h>
#else
#include <hipblas/hipblas.h>
#endif

// Wall-clock timestamp prefix used by every CHECK_NUMERICS log line. We pick
// wall clock (not monotonic) on purpose so users can correlate these messages
// with their training-loop logs, which are typically wall-clock stamped.
// localtime_r is POSIX; hipBLASLt is Linux/ROCm-only so this is safe.
inline std::string hipblaslt_check_numerics_ts()
{
    using namespace std::chrono;
    const auto now    = system_clock::now();
    const auto t      = system_clock::to_time_t(now);
    const auto ms_part
        = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    std::tm tm{};
    localtime_r(&t, &tm);
    char buf[40];
    std::snprintf(buf, sizeof(buf),
                  "[%04d-%02d-%02d %02d:%02d:%02d.%03lld]",
                  tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                  tm.tm_hour, tm.tm_min, tm.tm_sec,
                  static_cast<long long>(ms_part.count()));
    return std::string(buf);
}

// One thread per element of D. On first NaN found in the current scan window,
// claims the device flag with this call's id via atomicCAS(flag, 0, call_id).
// Subsequent threads (this kernel or later kernels in the window) see the
// slot already non-zero and the CAS is a no-op, so the slot retains the
// FIRST call_id that observed a NaN.
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
                           d_flag,
                           call_id);

        // v0 deferred sync: with no per-call hipStreamSynchronize, this only
        // catches launch-time argument errors (bad grid/block, bad pointer
        // type). Kernel runtime errors (e.g. invalid memory access during the
        // scan) are not detected here -- they surface at the next sync, which
        // for v0 is the handle dtor's hipDeviceSynchronize.
        if(hipGetLastError() != hipSuccess)
            return rocblaslt_status_internal_error;
    }
    return rocblaslt_status_success;
}

// Internal scanner entry. Called from the wiring helpers below.
//
// d_flag contract (v0 deferred-sync, atomicCAS first-id):
//   - Caller-owned single 4-byte device slot. In production this is the
//     persistent flag allocated once in the rocblaslt handle ctor and freed
//     in its dtor.
//   - Holds the call_id of the FIRST scanned matmul whose D had a NaN within
//     the current window (0 = none observed). The kernel atomicCAS(0,
//     call_id)'s into the slot; later scanners see non-zero and become no-ops.
//   - Must outlive every in-flight stream that received a launch from this
//     function. The handle owns the lifetime; do not free until after a
//     hipDeviceSynchronize.
//   - This function only writes to *d_flag (kernel side). It never reads it
//     back. The drain helper is responsible for D2H read, reset, and reporting.
//   - Passing nullptr short-circuits the function -- this is the runtime
//     "scanning disabled" gate (the handle leaves d_flag null when the env
//     var is unset or hipMalloc failed).
//
// Sampling: scanner is launched only when call_id % scan_every == 0. Pass
// scan_every=1 (or 0, treated as 1 for safety) to scan every call.
//
// Bisect window: only scan when scan_from <= call_id <= scan_until. The window
// gate runs BEFORE the sampling gate so excluded calls cost nothing past the
// gate compare. Defaults (1, ~0u) = unbounded.
inline rocblaslt_status hipblaslt_check_numerics_output_D(hipStream_t                   stream,
                                                          int64_t                       m,
                                                          int64_t                       n,
                                                          int32_t                       batch,
                                                          hipDataType                   type_d,
                                                          const void*                   D,
                                                          int64_t                       ldd,
                                                          int64_t                       stride_d,
                                                          bool                          row_major,
                                                          uint32_t*                     d_flag,
                                                          uint32_t                      call_id,
                                                          uint32_t                      scan_every,
                                                          uint32_t                      scan_from,
                                                          uint32_t                      scan_until)
{
    // Early exit: scanning disabled (no flag from handle), null buffer, or
    // any zero dimension.
    if(!d_flag || !D || m == 0 || n == 0 || batch == 0)
        return rocblaslt_status_success;

    // Window gate: bisect re-runs only need to scan a known interval. Excluded
    // calls cost just this compare; cheaper than letting them through and
    // having the kernel/sync no-op the way they'd have to without a window.
    if(call_id < scan_from || call_id > scan_until)
        return rocblaslt_status_success;

    // Sampling gate: skip if this call isn't the Nth one. scan_every==0 is
    // defensively treated as 1 (always scan) so a misconfigured handle never
    // silently drops every call.
    const uint32_t every = scan_every ? scan_every : 1u;
    if((call_id % every) != 0)
        return rocblaslt_status_success;

    // hipStreamSynchronize is illegal during HIP graph capture, but more
    // importantly the v0 deferred-flag read happens at handle destruction
    // time and cannot be sequenced into the captured graph -- so any NaN
    // observed during a captured replay would be invisible to our drain.
    // Skipping silently keeps customer graph-capture code working; the
    // captured graph itself is not replayed against this scanner anyway,
    // so no false negatives result.
    hipStreamCaptureStatus cap = hipStreamCaptureStatusNone;
    if(hipStreamIsCapturing(stream, &cap) == hipSuccess
       && cap != hipStreamCaptureStatusNone)
        return rocblaslt_status_success;

    // Per-dtype dispatch. The default arm silently skips integers (cannot
    // represent NaN) and sub-byte packed extension types (no scalar isnan
    // overload). Note: no info-mode trace is emitted on the default arm --
    // mode_info customers will see no log line for these dtypes, which is
    // intentional (there is nothing to report).
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

// Drain helper. Called from the handle dtor for the final drain, and from
// the public hipblasLtCheckNumericsDrain entry point on demand. Does
// device-wide sync (the simplest correct scope -- scanner kernels may have
// been dispatched on whatever stream the matmul caller passed, and the
// handle does not track those streams), reads the flag, resets it to 0,
// and logs the per-window result.
//
// "Window" = the inclusive range of call_ids (window_lo..window_hi) covered
// by this drain. The reporter prints the first NaN's call_id (or "no NaN
// observed") with that range, so customers can immediately tell which slice
// of their workload the report applies to.
//
// Returns the first NaN's call_id observed in this window (0 if none) so
// callers (handle dtor's sampling note, the public API) can react without
// re-reading the flag. Errors (sync/copy/memset) are swallowed: a benign
// drain failure must not take down the matmul stream or the handle.
inline uint32_t hipblaslt_drain_check_numerics_window(uint32_t* d_flag,
                                                     hipblaslt_check_numerics_mode mode,
                                                     uint32_t    window_lo,
                                                     uint32_t    window_hi,
                                                     uint32_t    scan_every,
                                                     uint32_t    scan_from,
                                                     uint32_t    scan_until,
                                                     const char* label /* "window" or "teardown" */,
                                                     bool                   stop_on_first,
                                                     std::atomic<bool>*     short_circuit_out,
                                                     std::atomic<uint32_t>* first_nan_out)
{
    if(!d_flag)
        return 0u;

    uint32_t h_flag = 0;
    static_cast<void>(hipDeviceSynchronize());
    static_cast<void>(hipMemcpy(&h_flag,
                                d_flag,
                                sizeof(uint32_t),
                                hipMemcpyDeviceToHost));
    // Under STOP_ON_FIRST after the first NaN, the slot is sticky -- do NOT
    // reset (scan_D's host poll reads it; clearing would let post-NaN calls
    // re-launch the kernel and re-drains would lose the first-NaN id). All
    // other cases reset so the next window starts fresh.
    // Same case also trips the cross-call short-circuit (idempotent; cache id
    // with release BEFORE the gate so acquire readers see a non-zero id).
    const bool sticky = stop_on_first && h_flag != 0u;
    if(!sticky)
        static_cast<void>(hipMemset(d_flag, 0, sizeof(uint32_t)));
    if(sticky)
    {
        if(first_nan_out)
        {
            uint32_t expected = 0u;
            first_nan_out->compare_exchange_strong(
                expected, h_flag, std::memory_order_release, std::memory_order_relaxed);
        }
        if(short_circuit_out)
            short_circuit_out->store(true, std::memory_order_release);
    }

    const bool log_anything = (mode
                               & (hipblaslt_check_numerics_mode_info
                                  | hipblaslt_check_numerics_mode_warn))
                              != 0;
    if(!log_anything)
        return h_flag;

    // Match the scanner's coercion (0 -> 1) so the value reported here equals
    // the value actually applied per-call by the sampler.
    const uint32_t every_eff = scan_every ? scan_every : 1u;

    // Effective window: the slice of [window_lo..window_hi] the scanner could
    // actually have inspected, given SCAN_FROM/SCAN_UNTIL. Reporting the
    // *requested* window when SCAN_FROM=100000 was set would let users assume
    // calls 1..99999 were inspected and clean -- they were never scanned at
    // all. We clamp to the requested window so the printed bounds never lie
    // outside what the caller asked about.
    const uint32_t eff_lo = std::max(window_lo, scan_from);
    const uint32_t eff_hi = std::min(window_hi, scan_until);
    const bool     sampling = (every_eff > 1u);

    std::lock_guard<std::mutex> lk(log_mutex);
    std::ostream*               sink = get_logger_os();
    if(!sink)
        sink = &std::cerr;
    if(h_flag != 0)
    {
        // Sampling bound: with scan_every=N, the *true* first NaN sits in
        // the interval (prev_sampled, h_flag], because the previous scan
        // fired at the largest multiple of N strictly less than h_flag.
        // Clamp the lower bound to scan_from so we don't suggest bisecting
        // calls the user explicitly excluded.
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
    return h_flag;
}

// --- Wiring helpers (R3): single source of truth for the post-GEMM hook ---
//
// All three matmul paths (rocblaslt_mat.cpp public, tensile_host.cpp ext,
// tensile_host.cpp grouped) share the same shape:
//
//   id = begin_call(handle);
//   if(id) {
//       (per (sub-)problem) scan_D(handle, ..., id, ...);
//   }
//
// begin_call returns 0 when scanning is disabled so callers can branch on a
// single value. Grouped-GEMM callers loop scan_D over each sub-problem with
// the SAME id (one user matmul = one call_id).

// Increment + return the per-handle call counter for this matmul. Returns 0
// when scanning is disabled, which doubles as the "skip the whole hook"
// signal for callers.
inline uint32_t hipblaslt_check_numerics_begin_call(rocblaslt_handle handle)
{
    if(!handle || !handle->check_numerics)
        return 0u;
    return handle->check_numerics_call_id.fetch_add(1, std::memory_order_relaxed) + 1u;
}

// Run the scanner on one D buffer. For grouped GEMM, call this once per
// sub-problem with a shared call_id.
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

    // STOP_ON_FIRST short-circuit. Sticky bypass first; otherwise (when the
    // mapped flag is available) poll it with ACQUIRE -- a plain *flag_host
    // would race and the compiler could hoist/elide it, and the bypass would
    // silently never trip under -O. ACQUIRE pairs with the kernel's atomicCAS.
    //
    // Visibility caveat: without a stream sync between the producing matmul
    // and this poll, the short-circuit can fire one call late (the late call
    // still launches the scanner; the call after that takes the sticky path).
    // We accept that to preserve the v0 zero-per-call-sync property.
    if(handle->check_numerics_short_circuit.load(std::memory_order_acquire))
        return rocblaslt_status_success;

    if(handle->check_numerics_stop_on_first && handle->check_numerics_flag_host)
    {
        const uint32_t observed
            = __atomic_load_n(handle->check_numerics_flag_host, __ATOMIC_ACQUIRE);
        if(observed != 0u)
        {
            // Cache the id with release BEFORE flipping the gate with release,
            // so any acquire-reader that sees short_circuit=true also sees the id.
            uint32_t expected = 0u;
            handle->check_numerics_first_nan_call.compare_exchange_strong(
                expected, observed, std::memory_order_release, std::memory_order_relaxed);
            handle->check_numerics_short_circuit.store(true, std::memory_order_release);
            return rocblaslt_status_success;
        }
    }

    const rocblaslt_status st
        = hipblaslt_check_numerics_output_D(stream,
                                            m, n, batch, type_d,
                                            D, ldd, stride_d, row_major,
                                            handle->check_numerics_flag,
                                            call_id,
                                            handle->check_numerics_scan_every,
                                            handle->check_numerics_scan_from,
                                            handle->check_numerics_scan_until);

    // One-shot launch-failure log. If the scanner kernel can't launch
    // (registration failure, OOM, illegal arg, etc.) we'd otherwise see
    // a "clean" teardown drain even though nothing was actually inspected.
    // Set the flag unconditionally on every failure so the dtor summary
    // fires even when info/warn is off; only the winner of the
    // exchange(true) race emits the per-call log line, and only when
    // info/warn is enabled.
    if(st != rocblaslt_status_success)
    {
        const bool first_failure
            = !handle->check_numerics_launch_failed.exchange(true, std::memory_order_relaxed);
        if(first_failure
           && (handle->check_numerics & (hipblaslt_check_numerics_mode_info
                                         | hipblaslt_check_numerics_mode_warn)))
        {
            std::ostream* sink = get_logger_os();
            if(!sink)
                sink = &std::cerr;
            *sink << hipblaslt_check_numerics_ts()
                  << "[hipBLASLt CHECK_NUMERICS] scanner kernel launch failed at call_id="
                  << call_id << " (rocblaslt_status=" << static_cast<int>(st)
                  << "); scanner is non-functional for this handle, results may be"
                  << " incomplete. Further per-call launch errors will be suppressed."
                  << std::endl;
        }
    }
    return st;
}

// On-demand drain entry point used by the public hipblasLtCheckNumericsDrain
// API. Forces the same device-sync + flag-read + reset that the handle dtor
// does, but at a caller-chosen point. Returns the first NaN call_id seen
// since the previous drain or handle creation (0 = none). Frameworks call
// this from a per-step hook so they don't have to wait for handle teardown
// -- which may not happen if the process is killed.
inline uint32_t hipblaslt_check_numerics_drain_handle(rocblaslt_handle handle)
{
    if(!handle || !handle->check_numerics_flag)
        return 0u;
    const uint32_t window_hi
        = handle->check_numerics_call_id.load(std::memory_order_relaxed);
    return hipblaslt_drain_check_numerics_window(
        handle->check_numerics_flag, handle->check_numerics,
        1u, window_hi,
        handle->check_numerics_scan_every,
        handle->check_numerics_scan_from,
        handle->check_numerics_scan_until,
        "on-demand drain",
        handle->check_numerics_stop_on_first,
        &handle->check_numerics_short_circuit,
        &handle->check_numerics_first_nan_call);
}

#endif // HIPBLASLT_CHECK_NUMERICS_MATRIX_HPP
