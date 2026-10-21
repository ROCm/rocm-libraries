/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *
 * ************************************************************************/

// Single-process, all-GPU benchmarking for hipblaslt-bench (`--device all`).
//
// Mirrors the tensilelite-client AllDevicesClient: the SAME GEMM/workload runs on
// every HIP-visible device at once, the timed region on each device is released
// simultaneously (a device-side spin gate or a host thread barrier), and the
// per-GPU time/TFLOPS/GB/s are aggregated (median and mean across GPUs) and the
// total-node throughput derived.
//
// hipblaslt-bench runs one worker host thread per device; each thread calls the
// normal single-device benchmark path (testing_matmul -> run_measurement). The
// coordinator here is the shared rendezvous between those threads:
//   * run_measurement calls all_devices_sync_point(stream) once, right before the
//     timed region, so every device's timed launches start together.
//   * testing_matmul calls all_devices_record_result(...) with its winning
//     solution's per-call timing so the orchestrator can report per-GPU and
//     aggregate numbers after every worker joins.
// When `--device all` is not set every entry point below is a cheap no-op.

#pragma once

#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <string>
#include <vector>

namespace hipblaslt_bench
{
    // Per-device winning-solution measurement recorded by testing_matmul.
    struct AllDevicesResult
    {
        bool        valid        = false;
        int         device_id    = -1;
        double      time_us      = 0.0; // per-call time (us), winning solution
        double      gflop_count  = 0.0; // GEMM GFLOP *count* per call (not per second)
        double      gbytes       = 0.0; // bytes moved per call, GiB (rotating-buffer footprint)
        int64_t     batch_count  = 1;
        std::string solution_name;
    };

    // How the timed region is released across devices (matches the tensilelite-client).
    enum class LaunchSyncMode
    {
        Gate, // device-side spin-gate kernel + single host release flag (ns-level alignment)
        Barrier // host thread barrier right before the timed enqueue (simpler; host skew remains)
    };

    // Headline aggregation across GPUs (both are always reported).
    enum class TimeAggregation
    {
        Median,
        Mean
    };

    // Enable all-GPU mode for `count` worker threads. `syncMode`/`aggregation` come from
    // the CLI. Must be called once (from the orchestrator) before any worker thread runs.
    // In Gate mode this allocates the shared host-coherent release flag.
    void all_devices_enable(int                   count,
                            const LaunchSyncMode   syncMode,
                            const TimeAggregation  aggregation);

    // True once all_devices_enable() has put the client in all-GPU mode.
    bool all_devices_enabled();

    // Bind the calling worker thread to its device slot [0, count). Call once per worker
    // thread (after hipSetDevice) before entering the benchmark path.
    void all_devices_bind_thread(int slot, int device_id);

    // Rendezvous called from run_measurement exactly once, immediately before the timed
    // region, so every device releases its timed launches together. No-op unless all-GPU
    // mode is active. Safe to call from the single-device path (does nothing there).
    void all_devices_sync_point(hipStream_t stream);

    // Record the calling worker's winning-solution result (called by testing_matmul).
    void all_devices_record_result(const AllDevicesResult& result);

    // Signal that the calling worker is bailing out (exception / fatal error) so the other
    // workers waiting at a barrier are released instead of deadlocking.
    void all_devices_abort();

    // After every worker thread has joined: print the per-GPU and aggregate
    // (median/mean + total-node) report and release the coordinator's resources.
    // Returns a process exit code (0 == every device produced a valid result).
    int all_devices_report_and_reset();
} // namespace hipblaslt_bench
