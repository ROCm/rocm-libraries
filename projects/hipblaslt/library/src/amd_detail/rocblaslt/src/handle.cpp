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
#include <cstring>
#include <iostream>
#include <sstream>
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

const rocblaslt::adaptive_sm::Config& rocblaslt_adaptive_sm_config()
{
    static const rocblaslt::adaptive_sm::Config cfg = rocblaslt::adaptive_sm::Config::fromEnv();
    return cfg;
}

uint32_t _rocblaslt_handle::adaptiveSmCountForLaunch(hipStream_t stream,
                                                     bool        tiny,
                                                     void**      probeAddr,
                                                     uint32_t*   probeEpoch,
                                                     bool*       probePad,
                                                     bool*       probeLaunch)
{
    using namespace rocblaslt::adaptive_sm;

    *probeAddr   = nullptr;
    *probeEpoch  = 0;
    *probePad    = false;
    *probeLaunch = false;
    if(!adaptive_streams)
        return 0;

    // A captured launch replays with frozen kernargs: no probe, no hint.
    hipStreamCaptureStatus cap = hipStreamCaptureStatusNone;
    if(hipStreamIsCapturing(stream, &cap) == hipSuccess && cap != hipStreamCaptureStatusNone)
        return 0;

    const int slot = streamSlot(stream);
    if(slot < 0)
        return 0;

    const Config&     cfg = rocblaslt_adaptive_sm_config();
    AdaptiveSmStream& st  = adaptive_streams[slot];

    // A tiny launch takes the hint but no epoch: it never probes. A long run
    // of them drops the hint, which nothing re-measures meanwhile.
    if(tiny)
    {
        if(cfg.force)
            return cfg.force;
        const uint32_t run  = st.tinyRun.fetch_add(1, std::memory_order_relaxed) + 1;
        const uint32_t hint = st.cus.load(std::memory_order_relaxed);
        if(!tinyRunDecays(run, hint, cfg) || st.busy.test_and_set(std::memory_order_acquire))
            return hint;
        st.estimator.reset();
        st.probeAt.store(0, std::memory_order_relaxed);
        st.cus.store(0, std::memory_order_relaxed);
        adaptive_query_cus.store(0, std::memory_order_relaxed);
        if(cfg.log)
        {
            std::ostringstream msg;
            msg << "hipBLASLt-ADAPTIVE-SM hint stream=" << static_cast<const void*>(stream)
                << " slot=" << slot << " tiny=" << run << " hint=" << hint << "->0\n";
            std::cerr << msg.str();
        }
        st.busy.clear(std::memory_order_release);
        return 0;
    }
    st.tinyRun.store(0, std::memory_order_relaxed);

    // 0 marks a slot no kernel has written.
    uint32_t epoch = st.epoch.fetch_add(1, std::memory_order_relaxed) + 1;
    if(epoch == 0)
        epoch = st.epoch.fetch_add(1, std::memory_order_relaxed) + 1;

    uint32_t* slots = adaptive_probe_host + size_t(slot) * c_probeSlots;
    *probeAddr      = static_cast<char*>(adaptive_probe_dev) + size_t(slot) * c_probeSlots * 4;
    *probeEpoch     = epoch;

    auto publish = [&](uint32_t hint) {
        const uint32_t old = st.cus.exchange(hint, std::memory_order_relaxed);
        if(old == hint)
            return;
        adaptive_query_cus.store(hint, std::memory_order_relaxed);
        if(cfg.log)
        {
            std::ostringstream msg;
            msg << "hipBLASLt-ADAPTIVE-SM hint stream=" << static_cast<const void*>(stream)
                << " slot=" << slot << " epoch=" << epoch << " count=" << st.estimator.lastCount()
                << " known=" << st.estimator.knownSlots() << " hint=" << old << "->" << hint
                << "\n";
            std::cerr << msg.str();
        }
    };

    // The first launch at or past the scheduled probe epoch claims it.
    uint32_t   at      = st.probeAt.load(std::memory_order_relaxed);
    const bool claimed = probeDueAt(epoch, at)
                         && st.probeAt.compare_exchange_strong(at, 0, std::memory_order_relaxed);
    const LaunchPlan plan = planLaunch(epoch, claimed, cfg);
    *probeLaunch          = plan.probeLaunch;
    *probePad             = plan.pad;

    if(cfg.force)
    {
        publish(cfg.force);
        return cfg.force;
    }

    if(!plan.read || st.busy.test_and_set(std::memory_order_acquire))
        return st.cus.load(std::memory_order_relaxed);

    // A slot is never released, so a new stream can inherit a destroyed one's
    // pointer and with it a different CU mask. Its launches use the old
    // stream's hint until this check runs on its first read.
    unsigned long long id = 0;
    if(hipStreamGetId(stream, &id) == hipSuccess && st.estimator.bindStream(id))
        st.probeAt.store(0, std::memory_order_relaxed);
    st.estimator.read(slots, epoch, uint32_t(properties.multiProcessorCount), cfg);
    const uint32_t next = st.estimator.probeDue(epoch, cfg);
    const uint32_t hint = st.estimator.hint();
    publish(hint);
    if(next)
    {
        st.probeAt.store(next, std::memory_order_relaxed);
        if(cfg.log)
        {
            std::ostringstream msg;
            msg << "hipBLASLt-ADAPTIVE-SM probe stream=" << static_cast<const void*>(stream)
                << " slot=" << slot << " epoch=" << next << " anchor=" << st.estimator.anchor()
                << " sampled=" << st.estimator.lastSampleEpoch() << " hint=" << hint << "\n";
            std::cerr << msg.str();
        }
    }
    st.busy.clear(std::memory_order_release);
    return hint;
}

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
#endif

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

    // Adaptive sm_count_target: one probe slot array per stream slot, written
    // by the GPU with plain stores and read by the host without a sync.
    if(rocblaslt_adaptive_sm_config().enabled)
    {
        constexpr size_t bytes
            = c_syncSkStreamSlots * rocblaslt::adaptive_sm::c_probeSlots * sizeof(uint32_t);
        void*      host = nullptr;
        hipError_t err  = hipHostMalloc(&host, bytes, hipHostMallocCoherent | hipHostMallocMapped);
        if(err == hipSuccess)
        {
            std::memset(host, 0, bytes);
            err = hipHostGetDevicePointer(&adaptive_probe_dev, host, 0);
        }
        if(err == hipSuccess)
        {
            adaptive_probe_host = static_cast<uint32_t*>(host);
            adaptive_streams    = std::make_unique<AdaptiveSmStream[]>(c_syncSkStreamSlots);
        }
        else
        {
            if(host)
                static_cast<void>(hipHostFree(host));
            adaptive_probe_dev = nullptr;
            std::cerr << "hipBLASLt-ADAPTIVE-SM disabled: probe allocation failed (hipError="
                      << static_cast<int>(err) << ")\n";
        }
    }

    // Allocate the device flag. STOP_ON_FIRST uses hipHostMalloc(MAPPED) so
    // scan_D can poll without sync; on any failure we fall back to hipMalloc
    // (the cross-call short-circuit then only trips on drains). If hipMalloc
    // also fails we disable scanning -- the matmul itself is unaffected.
    if(check_numerics != hipblaslt_check_numerics_mode_no_check)
    {
        bool       ok         = false;
        hipError_t mapped_err = hipSuccess;
        if(check_numerics_stop_on_first)
        {
            mapped_err = hipHostMalloc(reinterpret_cast<void**>(&check_numerics_flag_host),
                                       sizeof(uint32_t),
                                       hipHostMallocMapped);
            if(mapped_err == hipSuccess && check_numerics_flag_host)
            {
                *check_numerics_flag_host = 0u;
                mapped_err                = hipHostGetDevicePointer(
                    reinterpret_cast<void**>(&check_numerics_flag),
                    check_numerics_flag_host,
                    0);
                ok = (mapped_err == hipSuccess);
            }
            if(!ok)
            {
                if(check_numerics_flag_host)
                {
                    static_cast<void>(hipHostFree(check_numerics_flag_host));
                    check_numerics_flag_host = nullptr;
                }
                check_numerics_flag = nullptr;
                std::lock_guard<std::mutex> lk(log_mutex);
                std::ostream*               sink = get_logger_os();
                if(!sink)
                    sink = &std::cerr;
                *sink << hipblaslt_check_numerics_ts()
                      << "[hipBLASLt CHECK_NUMERICS] " << kEnvStopOnFirst
                      << ": hipHostMalloc(MAPPED) failed (hipError=" << static_cast<int>(mapped_err)
                      << "); falling back to device-only flag -- cross-call short-circuit"
                      << " will only trip on explicit drains." << std::endl;
            }
        }
        if(!ok)
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
    if(adaptive_probe_host)
        static_cast<void>(hipHostFree(adaptive_probe_host));
    adaptive_probe_host = nullptr;
    adaptive_probe_dev  = nullptr;

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

    // Mapped path: host pointer owns; device pointer is an alias.
    if(check_numerics_flag_host)
        static_cast<void>(hipHostFree(check_numerics_flag_host));
    else
        static_cast<void>(hipFree(check_numerics_flag));
    check_numerics_flag_host = nullptr;
    check_numerics_flag      = nullptr;
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
