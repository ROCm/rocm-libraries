// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "PerfMeasurement.hpp"

#include <algorithm>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <iomanip>
#include <sstream>
#include <string>

namespace ck_dsl_provider {

namespace {

[[noreturn]] void throwHipError(hipError_t err, std::string_view context) {
    const char* name = hipGetErrorName(err);
    const char* msg = hipGetErrorString(err);
    std::ostringstream oss;
    oss << context << ": " << (name != nullptr ? name : "hipError(unknown)") << ": "
        << (msg != nullptr ? msg : "no error string available")
        << " (code=" << static_cast<int>(err) << ")";
    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, oss.str());
}

inline void checkHip(hipError_t err, std::string_view context) {
    if (err != hipSuccess) {
        throwHipError(err, context);
    }
}

/// RAII wrapper around ``hipEvent_t`` so an exception part-way through
/// event setup doesn't leak handles. ``release()`` transfers
/// ownership for the rare case the caller wants to keep the event
/// past scope (not used in this file -- every event lives only as
/// long as the measure() body).
class ScopedEvent {
   public:
    ScopedEvent() {
        // Default flags are fine: timing-enabled events on the
        // current device. hipEventDisableTiming would speed up
        // record/destroy but disables hipEventElapsedTime which is
        // exactly what we need.
        checkHip(hipEventCreate(&_event), "PerfMeasurement::ScopedEvent ctor hipEventCreate");
    }
    ~ScopedEvent() noexcept {
        if (_event != nullptr) {
            hipError_t err = hipEventDestroy(_event);
            if (err != hipSuccess) {
                // Destructor is noexcept; log + swallow so we don't
                // mask the in-flight measurement error.
                try {
                    HIPDNN_PLUGIN_LOG_INFO(
                        "PerfMeasurement::~ScopedEvent hipEventDestroy failed: code="
                        << static_cast<int>(err));
                } catch (...) {  // NOLINT(bugprone-empty-catch)
                }
            }
        }
    }

    ScopedEvent(const ScopedEvent&) = delete;
    ScopedEvent& operator=(const ScopedEvent&) = delete;
    ScopedEvent(ScopedEvent&& other) noexcept : _event(other._event) {
        other._event = nullptr;
    }
    ScopedEvent& operator=(ScopedEvent&&) = delete;

    hipEvent_t get() const noexcept {
        return _event;
    }

   private:
    hipEvent_t _event{nullptr};
};

}  // namespace

PerfStats computePerfStats(std::vector<double> samplesUs) {
    if (samplesUs.empty()) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "computePerfStats: samples vector is empty (caller must run >=1 timed iter)");
    }

    // Capture the min value BEFORE the nth_element passes below
    // permute the vector -- the min iterator would otherwise dangle
    // (still valid as an iterator into the same buffer, but the
    // element it points to is whatever swapped into that slot).
    const double minUs = *std::min_element(samplesUs.begin(), samplesUs.end());

    const auto n = samplesUs.size();
    double median = 0.0;
    if ((n & 1u) == 1u) {
        // Odd N: middle element.
        const auto mid = samplesUs.begin() + static_cast<std::ptrdiff_t>(n / 2);
        std::nth_element(samplesUs.begin(), mid, samplesUs.end());
        median = *mid;
    } else {
        // Even N: arithmetic mean of the two central elements. Two
        // nth_element passes: one for the upper-middle, then a
        // partial sort of the lower half for the lower-middle. The
        // second call is cheap because the upper half is already
        // partitioned out.
        const auto upperMid = samplesUs.begin() + static_cast<std::ptrdiff_t>(n / 2);
        std::nth_element(samplesUs.begin(), upperMid, samplesUs.end());
        const auto lowerMid = samplesUs.begin() + static_cast<std::ptrdiff_t>(n / 2 - 1);
        std::nth_element(samplesUs.begin(), lowerMid, upperMid);
        median = (*lowerMid + *upperMid) / 2.0;
    }

    return PerfStats{minUs, median};
}

PerfMeasurement::PerfMeasurement(std::uint32_t warmupIters, std::uint32_t timedIters)
    : _warmupIters(warmupIters), _timedIters(timedIters) {
    if (_timedIters == 0) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "PerfMeasurement: timedIters must be >= 1 (got 0)");
    }
}

PerfResult PerfMeasurement::measureImpl(const std::function<void()>& launchFn, double flops,
                                        hipStream_t stream) {
    if (!launchFn) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "PerfMeasurement::measure: launchFn callable is empty");
    }

    // Warmup: untimed launches to populate caches, JIT-warm
    // anything the kernel calls into, and let the GPU clocks settle.
    for (std::uint32_t i = 0; i < _warmupIters; ++i) {
        launchFn();
    }

    // Per-iter event pairs. Vector storage so the events outlive the
    // launch loop without per-iter allocation.
    std::vector<ScopedEvent> starts;
    std::vector<ScopedEvent> stops;
    starts.reserve(_timedIters);
    stops.reserve(_timedIters);
    for (std::uint32_t i = 0; i < _timedIters; ++i) {
        starts.emplace_back();
        stops.emplace_back();
    }

    for (std::uint32_t i = 0; i < _timedIters; ++i) {
        checkHip(hipEventRecord(starts[i].get(), stream),
                 "PerfMeasurement::measure hipEventRecord(start)");
        launchFn();
        checkHip(hipEventRecord(stops[i].get(), stream),
                 "PerfMeasurement::measure hipEventRecord(stop)");
    }

    // Sync only on the final stop event: ensures all prior events
    // have completed too (HIP serialises events on a stream).
    checkHip(hipEventSynchronize(stops.back().get()),
             "PerfMeasurement::measure hipEventSynchronize");

    std::vector<double> samplesUs;
    samplesUs.reserve(_timedIters);
    for (std::uint32_t i = 0; i < _timedIters; ++i) {
        float ms = 0.0f;
        checkHip(hipEventElapsedTime(&ms, starts[i].get(), stops[i].get()),
                 "PerfMeasurement::measure hipEventElapsedTime");
        samplesUs.push_back(static_cast<double>(ms) * 1000.0);  // ms -> us
    }

    PerfStats stats = computePerfStats(std::move(samplesUs));

    PerfResult result;
    result.warmupIters = _warmupIters;
    result.timedIters = _timedIters;
    result.minUs = stats.minUs;
    result.medianUs = stats.medianUs;
    if (flops > 0.0 && stats.medianUs > 0.0) {
        // TFLOPS = flops / median_seconds / 1e12.
        const double medianSeconds = stats.medianUs / 1.0e6;
        result.tflops = flops / medianSeconds / 1.0e12;
    }
    return result;
}

void PerfMeasurement::log(std::string_view tag, const PerfResult& r) const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << "[CkDslPerf] tag=" << tag
        << " warmup=" << r.warmupIters << " iters=" << r.timedIters << " min_us=" << r.minUs
        << " median_us=" << r.medianUs << " tflops=" << r.tflops;
    HIPDNN_PLUGIN_LOG_INFO(oss.str());
}

}  // namespace ck_dsl_provider
