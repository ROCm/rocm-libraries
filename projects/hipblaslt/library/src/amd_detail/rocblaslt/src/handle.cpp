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
#include "rocblaslt_secure_env.hpp"
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
} // namespace

// Gated by HIPBLASLT_ADAPTIVE_SM_COUNT_LOG rather than the log mask, written
// where the library log goes.
void rocblaslt_adaptive_sm_log(const std::string& line)
{
    std::lock_guard<std::mutex> lk(log_mutex);
    std::ostream*               sink = get_logger_os();
    if(!sink)
        sink = &std::cerr;
    *sink << "[hipBLASLt ADAPTIVE_SM] " << line << std::endl;
}

const rocblaslt::adaptive_sm::Config& rocblaslt_adaptive_sm_config()
{
    static const rocblaslt::adaptive_sm::Config cfg = [] {
        std::vector<std::string> warnings;
        const auto               c = rocblaslt::adaptive_sm::Config::fromEnv(&warnings);
        for(auto const& w : warnings)
            rocblaslt_adaptive_sm_log(w);
        if(c.log)
            rocblaslt_adaptive_sm_log(c.describe());
        return c;
    }();
    return cfg;
}

namespace
{
    bool streamCapturing(hipStream_t stream)
    {
        hipStreamCaptureStatus cap = hipStreamCaptureStatusNone;
        return hipStreamIsCapturing(stream, &cap) == hipSuccess
               && cap != hipStreamCaptureStatusNone;
    }

    // Creates `ev` on `device` unless it exists. Returns whether it exists.
    bool ensureProbeEvent(hipEvent_t& ev, int device)
    {
        if(ev)
            return true;
        int prev = -1;
        static_cast<void>(hipGetDevice(&prev));
        const bool switchDevice = prev != device;
        if(switchDevice)
            static_cast<void>(hipSetDevice(device));
        if(hipEventCreateWithFlags(&ev, hipEventDisableTiming) != hipSuccess)
        {
            ev = nullptr;
            static_cast<void>(hipGetLastError());
        }
        if(switchDevice && prev >= 0)
            static_cast<void>(hipSetDevice(prev));
        return ev != nullptr;
    }

    // Records `ev` on `stream`, creating it first if needed. A failed record
    // destroys it, to be created again on next use.
    bool recordProbeEvent(hipEvent_t& ev, hipStream_t stream, int device)
    {
        if(!ensureProbeEvent(ev, device))
            return false;
        if(hipEventRecord(ev, stream) == hipSuccess)
            return true;
        static_cast<void>(hipEventDestroy(ev));
        ev = nullptr;
        static_cast<void>(hipGetLastError());
        return false;
    }

    void logAdaptiveHint(hipStream_t                              stream,
                         int                                      slot,
                         uint32_t                                 old,
                         const rocblaslt::adaptive_sm::Estimator& est)
    {
        std::ostringstream msg;
        msg << "hint stream=" << static_cast<const void*>(stream) << " slot=" << slot
            << " hint=" << old << "->" << est.hint();
        if(est.readingCount() == 0)
            msg << " missed=" << est.missed();
        else
            msg << " readings=";
        for(uint32_t i = 0; i < est.readingCount(); ++i)
            msg << (i ? "," : "") << est.readings()[i];
        rocblaslt_adaptive_sm_log(msg.str());
    }
} // namespace

void _rocblaslt_handle::publishAdaptiveSmCount(hipStream_t stream, int slot, uint32_t old)
{
    AdaptiveSmStream& st   = adaptive_streams[slot];
    const uint32_t    hint = st.estimator.hint();
    st.cus.store(hint, std::memory_order_relaxed);
    adaptive_query_cus.store(hint, std::memory_order_relaxed);
    if(rocblaslt_adaptive_sm_config().log && old != hint)
        logAdaptiveHint(stream, slot, old, st.estimator);
}

void _rocblaslt_handle::adaptiveSmCountForLaunch(hipStream_t       stream,
                                                 bool              tiny,
                                                 AdaptiveSmLaunch* launch)
{
    using namespace rocblaslt::adaptive_sm;

    *launch = {};
    if(!adaptive_streams)
        return;
    const Config& cfg  = rocblaslt_adaptive_sm_config();
    const int     slot = streamSlot(stream);
    if(slot < 0)
    {
        // A stream without a slot has no published hint; a forced one still holds.
        launch->hint = launchHint(cfg, /*capturing=*/false, /*published=*/0);
        return;
    }

    AdaptiveSmStream& st        = adaptive_streams[slot];
    const bool        capturing = streamCapturing(stream);
    launch->hint = launchHint(cfg, capturing, st.cus.load(std::memory_order_relaxed));
    if(!tracksLaunch(cfg, capturing))
        return;
    // Another thread on this stream holds the state: launch without it.
    std::unique_lock<std::mutex> guard(st.lock, std::try_to_lock);
    if(!guard.owns_lock())
        return;

    Estimator&      est   = st.estimator;
    const uint32_t  nCu   = uint32_t(properties.multiProcessorCount);
    const uint32_t* slots = adaptive_probe_host + size_t(slot) * c_probeSlots;
    launch->slot          = slot;
    launch->plan          = est.startLaunch(
        cfg,
        !tiny,
        [&](unsigned long long* id) { return hipStreamGetId(stream, id) == hipSuccess; },
        slots,
        nCu,
        [&](uint32_t i) {
            // hipErrorNotReady does not set the last error; other failures do.
            const hipError_t q = hipEventQuery(st.events[i]);
            if(q == hipSuccess)
                return ProbeState::Done;
            if(q == hipErrorNotReady)
                return ProbeState::Pending;
            static_cast<void>(hipGetLastError());
            return ProbeState::Failed;
        });

    if(cfg.log >= 2 && launch->plan.read)
    {
        std::ostringstream msg;
        msg << "reading stream=" << static_cast<const void*>(stream) << " slot=" << slot
            << " count=" << est.lastCount() << (cfg.force ? " (discarded)" : "");
        rocblaslt_adaptive_sm_log(msg.str());
    }
    if(!cfg.force)
    {
        if(est.hint() != launch->hint)
            publishAdaptiveSmCount(stream, slot, launch->plan.oldHint);
        launch->hint = est.hint();
    }

    if(launch->plan.probe)
    {
        // Zeroed only with no probe unread. A late mark from a dropped,
        // rebound or captured probe carries an older epoch and reads as older.
        if(st.epoch.advance(est.outstanding()))
            clearSlots(adaptive_probe_host + size_t(slot) * c_probeSlots);
        launch->probeAddr = static_cast<char*>(adaptive_probe_dev)
                            + size_t(slot) * c_probeSlots * sizeof(uint32_t);
        launch->probeEpoch = st.epoch.epoch;
        launch->probeLock  = std::move(guard);
    }
}

void _rocblaslt_handle::adaptiveSmCountAfterLaunch(hipStream_t stream, AdaptiveSmLaunch& launch)
{
    using namespace rocblaslt::adaptive_sm;

    std::unique_lock<std::mutex> guard = std::move(launch.probeLock);
    if(launch.slot < 0 || (!launch.plan.cadence && !launch.probed))
        return;
    AdaptiveSmStream& st = adaptive_streams[launch.slot];
    if(!guard.owns_lock())
        guard = std::unique_lock<std::mutex>(st.lock);
    Estimator& est = st.estimator;

    // A capture begun since the first phase would take the record into its
    // graph: the probe then stays due.
    const bool     recordable = launch.probed && !streamCapturing(stream);
    const uint32_t old        = est.hint();
    const bool     changed
        = est.finishLaunch(launch.plan, recordable, launch.probeEpoch, [&](uint32_t i) {
              return recordProbeEvent(st.events[i], stream, device);
          });
    if(changed && !rocblaslt_adaptive_sm_config().force)
        publishAdaptiveSmCount(stream, launch.slot, old);
}

bool _rocblaslt_handle::adaptiveSmCountStateForTest(hipStream_t stream,
                                                    rocblaslt::adaptive_sm::StreamStateForTest* out)
{
    if(!adaptive_streams)
        return false;
    const int slot = streamSlot(stream);
    if(slot < 0)
        return false;
    AdaptiveSmStream&           st = adaptive_streams[slot];
    std::lock_guard<std::mutex> guard(st.lock);
    const auto&                 est   = st.estimator;
    const uint32_t              force = rocblaslt_adaptive_sm_config().force;

    out->hint          = force ? force : st.cus.load(std::memory_order_relaxed);
    out->epoch         = st.epoch.epoch;
    out->outstanding   = est.outstanding();
    out->missed        = est.missed();
    out->sinceCadence  = est.sinceCadence();
    out->due           = est.due();
    out->readingsTaken = est.readingsTaken();
    out->probesRead    = est.probesRead();
    out->lastCount     = est.lastCount();
    return true;
}

bool _rocblaslt_handle::adaptiveSmCountLaunchLockForTest(
    hipStream_t stream, bool tiny, bool* probe, bool* heldBetween, bool* heldAfter)
{
    if(!adaptive_streams)
        return false;
    const int slot = streamSlot(stream);
    if(slot < 0)
        return false;
    AdaptiveSmLaunch launch;
    adaptiveSmCountForLaunch(stream, tiny, &launch);
    *probe       = launch.plan.probe;
    *heldBetween = launch.probeLock.owns_lock();
    adaptiveSmCountAfterLaunch(stream, launch);
    // This thread no longer owns the lock, so try_lock is well defined.
    std::mutex& lock = adaptive_streams[slot].lock;
    *heldAfter       = !lock.try_lock();
    if(!*heldAfter)
        lock.unlock();
    return true;
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
        const std::string s = rocblaslt::normalizeEnv(cn);
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
        const std::string s          = rocblaslt::normalizeEnv(sof);
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
            rocblaslt_adaptive_sm_log("disabled: probe allocation failed (hipError="
                                      + std::to_string(static_cast<int>(err)) + ")");
        }
        // Warns once per process, for the first device it exceeds.
        const uint32_t           force = rocblaslt_adaptive_sm_config().force;
        static std::atomic<bool> warnedForce{false};
        if(force > uint32_t(properties.multiProcessorCount) && !warnedForce.exchange(true))
            rocblaslt_adaptive_sm_log(std::string(rocblaslt::adaptive_sm::kEnvForce) + "="
                                      + std::to_string(force) + " exceeds the device's "
                                      + std::to_string(properties.multiProcessorCount) + " CUs");
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
        const std::string s = rocblaslt::normalizeEnv(cs);
        check_synchronizer  = (s == "1" || s == "on" || s == "true");
    }
}

_rocblaslt_handle::~_rocblaslt_handle()
{
    if(adaptive_streams)
        for(size_t i = 0; i < c_syncSkStreamSlots; ++i)
            for(hipEvent_t ev : adaptive_streams[i].events)
                if(ev)
                    static_cast<void>(hipEventDestroy(ev));
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
