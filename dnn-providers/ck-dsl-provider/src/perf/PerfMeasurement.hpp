// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ck_dsl_provider {

/// Summary statistics for one measurement run.
///
/// Both ``minUs`` and ``medianUs`` are reported: min is the best-case,
/// median is the robust typical, and the plan
/// uses median for the TFLOPS calculation so it isn't swayed by a
/// single fast outlier.
///
/// ``tflops`` is computed from ``medianUs`` and the ``flops`` value
/// the caller passed to ``measure()``. If ``flops <= 0`` (the no-FLOPS
/// caller use case, e.g. the smoke kernel that has no defined
/// arithmetic intensity), ``tflops`` is zero.
struct PerfResult {
    std::uint32_t warmupIters{0};
    std::uint32_t timedIters{0};
    double minUs{0.0};
    double medianUs{0.0};
    double tflops{0.0};
};

/// Compute (min, median) microseconds over a vector of per-iter
/// samples. Exposed at namespace scope so the host-only unit suite
/// can exercise the stats path without driving real hipEvents.
///
/// Median convention: for odd N the middle element; for even N the
/// arithmetic mean of the two central elements. Uses ``nth_element``
/// twice (no full sort) for the even-N case.
struct PerfStats {
    double minUs{0.0};
    double medianUs{0.0};
};
PerfStats computePerfStats(std::vector<double> samplesUs);

/// hipEvent-based warmup-and-iterate timing helper.
///
/// Defaults (5 warmup, 50 timed) keep the integration test under
/// roughly a second of kernel time while keeping the median stable.
///
/// **What this measures.** GPU stream time only. Each timed iter
/// brackets the launch with hipEventRecord(start) /
/// hipEventRecord(stop) on the same stream, then
/// hipEventElapsedTime reports the GPU-side elapsed time between the
/// two events. Host-side work performed BETWEEN ``execute()`` entry
/// and the stream submission (argument packing, validation, logging
/// formatting) is NOT included in the reported numbers. If you need
/// to measure host overhead, wrap a separate std::chrono pair around
/// the launch callable; the reference launcher.cpp does this for
/// total wall time.
///
/// **Sync protocol:** no ``hipDeviceSynchronize`` between launches.
/// Each timed iter records a start event, runs the launch callable,
/// then records a stop event -- all on the same stream so the GPU
/// sees the launches in order. After the warmup loop the helper
/// drains the stream so the first timed iter starts from a clean
/// queue; after the timed loop the helper synchronises on the final
/// stop event and walks the event pairs to compute per-iter elapsed
/// times.
///
/// **No assertions** -- this is a logging-only helper. The integration
/// test prints the result; future work adds perf-target checks.
class PerfMeasurement {
   public:
    static constexpr std::uint32_t kDefaultWarmupIters = 5;
    static constexpr std::uint32_t kDefaultTimedIters = 50;

    explicit PerfMeasurement(std::uint32_t warmupIters = kDefaultWarmupIters,
                             std::uint32_t timedIters = kDefaultTimedIters);

    /// Run ``launchFn`` ``warmupIters`` times (untimed) followed by
    /// ``timedIters`` times with per-iter ``hipEvent`` timing.
    ///
    /// ``flops`` is the kernel's arithmetic intensity (the
    /// implicit-GEMM conv formula ``2 * N * Ho * Wo * K * C * R * S``
    /// per plan §4). Pass 0 to skip TFLOPS computation -- useful for
    /// the smoke kernel whose arithmetic intensity is not well-defined.
    ///
    /// All HIP calls run on ``stream``. The caller is responsible for
    /// the kernel-side preconditions (device set, buffers allocated,
    /// stream valid).
    ///
    /// Throws ``hipdnn_plugin_sdk::HipdnnPluginException`` on any
    /// underlying HIP failure (event create / record / synchronize /
    /// elapsed-time / destroy). The HIP error name surfaces in the
    /// message via the same pattern HipModule uses.
    template <typename LaunchFn>
    PerfResult measure(LaunchFn&& launchFn, double flops, hipStream_t stream);

    /// Emit a single-line ``[CkDslPerf]`` log via
    /// HIPDNN_PLUGIN_LOG_INFO. Format:
    ///
    ///   [CkDslPerf] tag=<tag> warmup=N iters=M min_us=X.X median_us=Y.Y tflops=Z.Z
    ///
    /// The log goes through hipDNN's plugin logging so the
    /// integration-test harness's log recorder captures it.
    void log(std::string_view tag, const PerfResult& r) const;

    std::uint32_t warmupIters() const noexcept {
        return _warmupIters;
    }
    std::uint32_t timedIters() const noexcept {
        return _timedIters;
    }

   private:
    /// Non-template body of measure. The template instantiation
    /// captures launchFn and dispatches into this once per call.
    /// Splitting the body out keeps the HIP machinery in the .cpp
    /// so the header doesn't drag <hip/hip_runtime.h> internals
    /// into every TU that includes it.
    PerfResult measureImpl(const std::function<void()>& launchFn, double flops, hipStream_t stream);

    std::uint32_t _warmupIters;
    std::uint32_t _timedIters;
};

// --- template impl --------------------------------------------------------

template <typename LaunchFn>
inline PerfResult PerfMeasurement::measure(LaunchFn&& launchFn, double flops, hipStream_t stream) {
    // Wrap once -- the std::function captures the callable by
    // forwarding, then the impl drives it. The function-object
    // indirection costs a few ns per call which is negligible next
    // to a kernel launch.
    std::function<void()> wrapped(std::forward<LaunchFn>(launchFn));
    return measureImpl(wrapped, flops, stream);
}

}  // namespace ck_dsl_provider
