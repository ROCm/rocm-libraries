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
    // Env-var names: shared between ctor parser and dtor's bisect hint.
    constexpr const char* kEnvCheckNumerics = "HIPBLASLT_CHECK_NUMERICS";
    constexpr const char* kEnvScanEvery     = "HIPBLASLT_CHECK_NUMERICS_SCAN_EVERY";
    constexpr const char* kEnvScanFrom      = "HIPBLASLT_CHECK_NUMERICS_SCAN_FROM";
    constexpr const char* kEnvScanUntil     = "HIPBLASLT_CHECK_NUMERICS_SCAN_UNTIL";
    constexpr const char* kEnvStopOnFirst   = "HIPBLASLT_CHECK_NUMERICS_STOP_ON_FIRST";
    constexpr const char* kEnvCheckSynchronizer = "HIPBLASLT_CHECK_SYNCHRONIZER";

    // Trim leading/trailing whitespace + lowercase. Used for the two
    // text-token env vars; numeric-only vars use std::atoi.
    std::string normalize_env(const char* raw)
    {
        std::string s(raw);
        size_t      first = s.find_first_not_of(" \t");
        if(first == std::string::npos)
            return {};
        s.erase(0, first);
        if(auto p = s.find_last_not_of(" \t"); p != std::string::npos)
            s.erase(p + 1);
        for(auto& c : s)
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return s;
    }
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

    // ASIC revision
    asic_rev = properties.asicRevision;

#ifdef HIPBLASLT_USE_ROCROLLER
    rocroller_create_handle(&rocroller_handle);
#endif
    const char* rocRollerEnvVal = std::getenv("HIPBLASLT_USE_ROCROLLER");
    if((std::string(properties.gcnArchName).find("gfx1250") != std::string::npos))
    {
        useRocRoller = 0;
    }
    else if(rocRollerEnvVal)
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

    // HIPBLASLT_CHECK_NUMERICS: 1/info, 2/warn, 0/none/off. Anything else
    // (including the removed =4 fail bit) collapses to no_check.
    if(const char* cn = std::getenv(kEnvCheckNumerics))
    {
        const std::string s = normalize_env(cn);
        if(s == "info" || s == "1")
            check_numerics = hipblaslt_check_numerics_mode_info;
        else if(s == "warn" || s == "2")
            check_numerics = hipblaslt_check_numerics_mode_warn;
    }

    // SCAN_EVERY=N: scan only every Nth matmul. 0/negative/garbage -> 1.
    if(const char* se = std::getenv(kEnvScanEvery))
    {
        const int v = std::atoi(se);
        check_numerics_scan_every = (v > 0) ? static_cast<uint32_t>(v) : 1u;
    }
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
    // Inverted window: warn (so a typo doesn't silently mask every NaN) and
    // restore defaults.
    if(check_numerics_scan_from > check_numerics_scan_until)
    {
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

    if(const char* sof = std::getenv(kEnvStopOnFirst))
    {
        const std::string s = normalize_env(sof);
        check_numerics_stop_on_first = (s == "1" || s == "on" || s == "true");
    }

    // Allocate the device flag the scanner claims with atomicCAS. The atomic
    // stays in device memory: targeting hipHostMalloc(MAPPED) makes it a PCIe
    // AtomicOp, and a root port without an AtomicOp completer (typical of a
    // PCH-attached GPU) drops it. The slot then stays 0 through a later device
    // sync, so both the host peek and hipblasLtCheckNumericsDrain miss the NaN
    // (ROCM-30523). STOP_ON_FIRST publishes the winning call_id with a plain
    // store to a coherent host mirror so scan_D can still peek without syncing.
    // If that mirror cannot be allocated, drains and teardown still report the
    // device flag. If hipMalloc itself fails we disable scanning -- the matmul
    // is unaffected.
    if(check_numerics != hipblaslt_check_numerics_mode_no_check)
    {
        const hipError_t alloc_err = hipMalloc(&check_numerics_flag, sizeof(uint32_t));
        const hipError_t memset_err
            = (alloc_err == hipSuccess)
                  ? hipMemset(check_numerics_flag, 0, sizeof(uint32_t))
                  : hipSuccess;
        if(alloc_err != hipSuccess || memset_err != hipSuccess)
        {
            if(check_numerics_flag)
            {
                static_cast<void>(hipFree(check_numerics_flag));
                check_numerics_flag = nullptr;
            }
            check_numerics = hipblaslt_check_numerics_mode_no_check;
            std::lock_guard<std::mutex> lk(log_mutex);
            std::ostream*               sink = get_logger_os();
            if(!sink)
                sink = &std::cerr;
            *sink << hipblaslt_check_numerics_ts()
                  << "[hipBLASLt CHECK_NUMERICS] scanner disabled: hipMalloc(4B) failed (hipError="
                  << static_cast<int>(alloc_err != hipSuccess ? alloc_err : memset_err) << "). "
                  << kEnvCheckNumerics
                  << " was set but no scans will run for this handle." << std::endl;
        }
        else if(check_numerics_stop_on_first)
        {
            hipError_t mapped_err
                = hipHostMalloc(reinterpret_cast<void**>(&check_numerics_flag_host),
                                sizeof(uint32_t),
                                hipHostMallocMapped | hipHostMallocCoherent);
            if(mapped_err == hipSuccess && check_numerics_flag_host)
            {
                __atomic_store_n(check_numerics_flag_host, 0u, __ATOMIC_RELAXED);
                mapped_err = hipHostGetDevicePointer(
                    reinterpret_cast<void**>(&check_numerics_flag_host_dev),
                    check_numerics_flag_host,
                    0);
            }
            if(mapped_err != hipSuccess || !check_numerics_flag_host
               || !check_numerics_flag_host_dev)
            {
                if(check_numerics_flag_host)
                {
                    static_cast<void>(hipHostFree(check_numerics_flag_host));
                    check_numerics_flag_host = nullptr;
                }
                check_numerics_flag_host_dev = nullptr;
                std::lock_guard<std::mutex> lk(log_mutex);
                std::ostream*               sink = get_logger_os();
                if(!sink)
                    sink = &std::cerr;
                *sink << hipblaslt_check_numerics_ts()
                      << "[hipBLASLt CHECK_NUMERICS] " << kEnvStopOnFirst
                      << ": host-mapped mirror unavailable (hipError="
                      << static_cast<int>(mapped_err)
                      << "); host-peek auto-drain disabled. Explicit drains and"
                      << " handle teardown still report the first NaN." << std::endl;
            }
        }
    }

    // HIPBLASLT_CHECK_SYNCHRONIZER: opt-in debug check that a Stream-K or GSU
    // MBSK kernel leaves the handle's flag buffers all-zero on exit.
    if(const char* cs = std::getenv(kEnvCheckSynchronizer))
    {
        const std::string s = normalize_env(cs);
        check_synchronizer  = (s == "1" || s == "on" || s == "true");
    }
}

_rocblaslt_handle::~_rocblaslt_handle()
{
    if(!check_numerics_flag)
        return;

    // The dtor does NOT relaunch a scanner for the unscanned tail: the user's
    // last.D pointer may already have been hipFree'd, making any kernel launch
    // a use-after-free.
    const uint32_t every = check_numerics_scan_every ? check_numerics_scan_every : 1u;
    const uint32_t call_id_snapshot
        = check_numerics_call_id.load(std::memory_order_relaxed);

    // Pin the thread to this handle's device: drain runs hipDeviceSynchronize
    // + hipMemcpy on the current device, and the dtor may run on any thread.
    int prev_device = -1;
    static_cast<void>(hipGetDevice(&prev_device));
    if(prev_device != device)
        static_cast<void>(hipSetDevice(device));
    const uint32_t first_nan
        = hipblaslt_drain_check_numerics_window(check_numerics_flag,
                                                check_numerics,
                                                1u,
                                                call_id_snapshot,
                                                every,
                                                check_numerics_scan_from,
                                                check_numerics_scan_until,
                                                "handle teardown",
                                                check_numerics_stop_on_first,
                                                &check_numerics_short_circuit);
    // Free while this handle's device is current. The flag is device memory.
    if(check_numerics_flag_host)
        static_cast<void>(hipHostFree(check_numerics_flag_host));
    static_cast<void>(hipFree(check_numerics_flag));
    check_numerics_flag_host     = nullptr;
    check_numerics_flag_host_dev = nullptr;
    check_numerics_flag          = nullptr;
    if(prev_device != -1 && prev_device != device)
        static_cast<void>(hipSetDevice(prev_device));

    // Tail report: the slice of [scan_from..min(call_id, scan_until)] that
    // never had a scanner fire. Under STOP_ON_FIRST we report the deliberately
    // skipped post-NaN range instead so the user isn't told to chase calls
    // they explicitly opted out of.
    if(call_id_snapshot > 0
       && (check_numerics & (hipblaslt_check_numerics_mode_info
                             | hipblaslt_check_numerics_mode_warn)))
    {
        const bool short_circuited
            = check_numerics_short_circuit.load(std::memory_order_acquire);
        std::lock_guard<std::mutex> lk(log_mutex);
        std::ostream*               sink = get_logger_os();
        if(!sink)
            sink = &std::cerr;

        if(short_circuited && first_nan > 0 && call_id_snapshot > first_nan)
        {
            *sink << hipblaslt_check_numerics_ts()
                  << "[hipBLASLt CHECK_NUMERICS] handle teardown: matmul calls ("
                  << first_nan << ".." << call_id_snapshot
                  << "] were intentionally skipped due to "
                  << kEnvStopOnFirst << "=1 (first NaN at call #"
                  << first_nan << ")." << std::endl;
        }
        else if(!short_circuited)
        {
            const uint32_t observed_hi
                = std::min(call_id_snapshot, check_numerics_scan_until);
            const uint32_t last_sampled = (observed_hi / every) * every;
            const bool     had_sample
                = last_sampled >= check_numerics_scan_from && last_sampled > 0;
            const uint32_t tail_lo
                = had_sample ? last_sampled + 1 : check_numerics_scan_from;
            const uint32_t tail_hi = observed_hi;
            if(tail_hi >= tail_lo)
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
