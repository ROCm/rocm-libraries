/* ************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
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

#include "handle.h"
#include "check_numerics_matrix.hpp"
#include "definitions.h"
#include "logging.h"
#include "rocroller_host.hpp"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <string>

namespace
{
    // Single source of truth for the CHECK_NUMERICS env-var names. Used by
    // the parser in the ctor and by the "how to bisect on a re-run" hint
    // emitted by the dtor; previously the dtor copy-pasted the strings,
    // which drifted once and had to be re-aligned by hand.
    constexpr const char* kEnvCheckNumerics = "HIPBLASLT_CHECK_NUMERICS";
    constexpr const char* kEnvScanEvery     = "HIPBLASLT_CHECK_NUMERICS_SCAN_EVERY";
    constexpr const char* kEnvScanFrom      = "HIPBLASLT_CHECK_NUMERICS_SCAN_FROM";
    constexpr const char* kEnvScanUntil     = "HIPBLASLT_CHECK_NUMERICS_SCAN_UNTIL";
} // namespace

/*******************************************************************************
 * constructor
 ******************************************************************************/
_rocblaslt_handle::_rocblaslt_handle()
{
    // Default device is active device
    THROW_IF_HIP_ERROR(hipGetDevice(&device));
    THROW_IF_HIP_ERROR(hipGetDeviceProperties(&properties, device));

    // Device wavefront size
    wavefront_size = properties.warpSize;

#if HIP_VERSION >= 307
    // ASIC revision
    asic_rev = properties.asicRevision;
#else
    asic_rev = 0;
#endif

#ifdef HIPBLASLT_USE_ROCROLLER
    rocroller_create_handle(&rocroller_handle);
    const char* rocRollerEnvVal = std::getenv("HIPBLASLT_USE_ROCROLLER");
    if(rocRollerEnvVal)
    {
        if(strncmp(rocRollerEnvVal, "1", 1) == 0)
        {
            useRocRoller = 1;
        }
        else
        {
            useRocRoller = 0;
        }
    }
    else
    {
        useRocRoller = -1;
    }
#endif

    // HIPBLASLT_CHECK_NUMERICS: 1/info, 2/warn, 0/none/off. Accepts only the
    // documented string/numeric forms; anything else collapses to no_check so
    // a typo (or the removed =4 fail bit, or a stray =8) gives the user
    // nothing rather than a half-broken mode they have to debug.
    if(const char* cn = std::getenv(kEnvCheckNumerics))
    {
        // Lowercase for word compare. Skip leading whitespace so "  warn"
        // works the way a user would expect when copy-pasting.
        std::string s(cn);
        size_t      first = s.find_first_not_of(" \t");
        if(first == std::string::npos)
            first = s.size();
        s.erase(0, first);
        size_t last = s.find_last_not_of(" \t");
        if(last != std::string::npos)
            s.erase(last + 1);
        for(auto& c : s)
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

        if(s == "none" || s == "off" || s == "0" || s.empty())
            check_numerics = hipblaslt_check_numerics_mode_no_check;
        else if(s == "info" || s == "1")
            check_numerics = hipblaslt_check_numerics_mode_info;
        else if(s == "warn" || s == "2")
            check_numerics = hipblaslt_check_numerics_mode_warn;
        else
            check_numerics = hipblaslt_check_numerics_mode_no_check;
    }

    // Sampling knob. Defaults to scan every call when unset. Mid-stream
    // drains are explicit (hipblasLtCheckNumericsDrain); the dtor handles
    // the final drain.
    //
    //   HIPBLASLT_CHECK_NUMERICS_SCAN_EVERY=N   scanner runs only on every Nth
    //                                           matmul (default 1)
    //
    // Negative or non-numeric input collapses to the safe default (1) via
    // std::atoi returning 0; we coerce 0 to 1 since "scan every 0th call"
    // is undefined.
    if(const char* se = std::getenv(kEnvScanEvery))
    {
        const int v = std::atoi(se);
        check_numerics_scan_every = (v > 0) ? static_cast<uint32_t>(v) : 1u;
    }

    // Bisect window. Both env vars are optional; unset = unbounded (defaults
    // already set in the handle struct). Inverted/zero values collapse to
    // defaults rather than silently disabling all scans.
    //
    //   HIPBLASLT_CHECK_NUMERICS_SCAN_FROM=A    only scan calls with id >= A
    //   HIPBLASLT_CHECK_NUMERICS_SCAN_UNTIL=B   only scan calls with id <= B
    if(const char* sf = std::getenv(kEnvScanFrom))
    {
        const int v = std::atoi(sf);
        check_numerics_scan_from = (v > 0) ? static_cast<uint32_t>(v) : 1u;
    }
    if(const char* su = std::getenv(kEnvScanUntil))
    {
        const int v = std::atoi(su);
        check_numerics_scan_until = (v > 0) ? static_cast<uint32_t>(v) : ~uint32_t(0);
    }
    // If user inverted the window, warn loudly and restore defaults rather
    // than silently scanning nothing forever. Without the warning a typo
    // (FROM=600 UNTIL=500 instead of FROM=500 UNTIL=600) produces an empty
    // report that looks indistinguishable from "no NaN seen".
    if(check_numerics_scan_from > check_numerics_scan_until)
    {
        // Route through the same sink as every other CHECK_NUMERICS line so
        // users who redirected the logger to a file still capture this. Going
        // straight to std::cerr here would lose the warning for those users.
        std::lock_guard<std::mutex> lk(log_mutex);
        std::ostream*               sink = get_logger_os();
        if(!sink)
            sink = &std::cerr;
        *sink << hipblaslt_check_numerics_ts()
              << "[hipBLASLt CHECK_NUMERICS] " << kEnvScanFrom << "="
              << check_numerics_scan_from << " > " << kEnvScanUntil << "="
              << check_numerics_scan_until
              << " is inverted; resetting to defaults (full range scanned)."
              << std::endl;
        check_numerics_scan_from  = 1u;
        check_numerics_scan_until = ~uint32_t(0);
    }

    // v0 deferred-sync: allocate the persistent NaN flag once per handle.
    // The scanner kernel atomicCAS(0, call_id)'s into this slot for every
    // scanned matmul. Failure to allocate is best-effort: we disable
    // scanning rather than failing handle creation, since the matmul itself
    // is unaffected.
    if(check_numerics != hipblaslt_check_numerics_mode_no_check)
    {
        if(hipMalloc(&check_numerics_flag, sizeof(uint32_t)) != hipSuccess
           || hipMemset(check_numerics_flag, 0, sizeof(uint32_t)) != hipSuccess)
        {
            // Best-effort: leave flag null and disable scanning. The scanner
            // entrypoint short-circuits on a null flag pointer.
            if(check_numerics_flag)
            {
                static_cast<void>(hipFree(check_numerics_flag));
                check_numerics_flag = nullptr;
            }
            check_numerics = hipblaslt_check_numerics_mode_no_check;
        }
    }
}

_rocblaslt_handle::~_rocblaslt_handle()
{
    if(!check_numerics_flag)
        return;

    // We deliberately do NOT relaunch a scanner kernel from the destructor
    // for any unscanned tail. The cached last.D is a raw user-owned device
    // pointer with no ownership/refcount on our side; by the time the dtor
    // runs the user may have hipFree()'d it, making any kernel launch a
    // use-after-free hazard (SEGV, sticky HIP errors leaking into unrelated
    // user calls, or false-positive NaN flags on garbage). Production-safe
    // policy: accept the false negative for the unscanned tail and tell the
    // user (below) exactly what range was missed and how to catch it on a
    // re-run.
    const uint32_t every = check_numerics_scan_every ? check_numerics_scan_every : 1u;

    // Final drain: device-wide sync, read flag, reset, log. Window is the
    // entire handle lifetime [1..call_id]. Snapshot the atomic counter once
    // -- by dtor time no other thread should still be using this handle, but
    // .load() pins the value for both the drain and the tail computation.
    const uint32_t call_id_snapshot = check_numerics_call_id.load(std::memory_order_relaxed);
    const uint32_t window_lo = 1u;
    const uint32_t window_hi = call_id_snapshot;
    static_cast<void>(hipblaslt_drain_check_numerics_window(check_numerics_flag,
                                                            check_numerics,
                                                            window_lo,
                                                            window_hi,
                                                            every,
                                                            check_numerics_scan_from,
                                                            check_numerics_scan_until,
                                                            "handle teardown"));

    // Unscanned-tail warning. Compute the exact [tail_lo..tail_hi] range of
    // call_ids that completed but were never inspected, and tell the user
    // how to catch them on a re-run. Independent of first_nan: even if we
    // already caught a NaN earlier, the user should know the picture is
    // incomplete past the last sampled boundary.
    //
    // Definitions:
    //   observed_hi  = last call_id we could have scanned this run
    //                  (capped by SCAN_UNTIL, so we don't pretend we missed
    //                  calls the user explicitly opted out of)
    //   last_sampled = largest multiple of `every` <= observed_hi that is
    //                  also >= SCAN_FROM (i.e. the last call_id whose scan
    //                  kernel actually fired)
    //   tail         = (last_sampled, observed_hi]  if any sample fired
    //                  [SCAN_FROM, observed_hi]     if none did
    if(call_id_snapshot > 0
       && (check_numerics & (hipblaslt_check_numerics_mode_info
                             | hipblaslt_check_numerics_mode_warn)))
    {
        const uint32_t observed_hi
            = std::min(call_id_snapshot, check_numerics_scan_until);
        const uint32_t last_sampled_candidate = (observed_hi / every) * every;
        const bool     had_any_sample
            = (last_sampled_candidate >= check_numerics_scan_from)
              && (last_sampled_candidate > 0)
              && (last_sampled_candidate <= observed_hi);
        const uint32_t tail_lo = had_any_sample ? (last_sampled_candidate + 1)
                                                : check_numerics_scan_from;
        const uint32_t tail_hi = observed_hi;

        if(tail_hi >= tail_lo)
        {
            std::ostream* sink = get_logger_os();
            if(!sink)
                sink = &std::cerr;
            *sink << hipblaslt_check_numerics_ts()
                  << "[hipBLASLt CHECK_NUMERICS] handle teardown: matmul calls ["
                  << tail_lo << ".." << tail_hi
                  << "] were NOT scanned (scan_every=" << every
                  << ", scan_from=" << check_numerics_scan_from
                  << ", scan_until=" << check_numerics_scan_until
                  << "). To inspect this range on a re-run, set:\n"
                  << "    " << kEnvCheckNumerics << "=1\n"
                  << "    " << kEnvScanEvery     << "=1\n"
                  << "    " << kEnvScanFrom      << "=" << tail_lo << "\n"
                  << "    " << kEnvScanUntil     << "=" << tail_hi
                  << std::endl;
        }
    }
    // Persistent launch-failure flag: at least one scanner kernel launch on
    // this handle returned non-success. The first failure was already logged
    // (or suppressed by mode bits) at the call site, but we always emit a
    // single teardown line so users running with mode_check alone -- or who
    // missed the first log line -- get a clear "scanner was broken" signal
    // before the handle disappears.
    if(check_numerics_launch_failed.load(std::memory_order_relaxed))
    {
        std::ostream* sink = get_logger_os();
        if(!sink)
            sink = &std::cerr;
        *sink << hipblaslt_check_numerics_ts()
              << "[hipBLASLt CHECK_NUMERICS] handle teardown: one or more scanner"
              << " kernel launches FAILED on this handle; NaN/Inf reports above"
              << " may be incomplete or absent. Check earlier log lines for the"
              << " first failure's rocblaslt_status code." << std::endl;
    }

    static_cast<void>(hipFree(check_numerics_flag));
    check_numerics_flag = nullptr;
}

_rocblaslt_attribute::~_rocblaslt_attribute()
{
    clear();
}

void _rocblaslt_attribute::clear()
{
    set(nullptr, 0);
}

const void* _rocblaslt_attribute::data()
{
    return _data;
}
size_t _rocblaslt_attribute::length()
{
    return _data_size;
}

size_t _rocblaslt_attribute::get(void* out, size_t size)
{
    if(out != nullptr && _data != nullptr && _data_size >= size)
    {
        memcpy(out, _data, size);
        return size;
    }
    return 0;
}

void _rocblaslt_attribute::set(const void* in, size_t size)
{
    if(in == nullptr || (_data != nullptr && _data_size != size))
    {
        free(_data);
        _data      = nullptr;
        _data_size = 0;
    }
    if(in != nullptr)
    {
        if(_data == nullptr)
            _data = malloc(size);
        memcpy(_data, in, size);
        _data_size = size;
    }
}
