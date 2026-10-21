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

#include "all_devices_bench.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <thread>
#include <vector>

namespace hipblaslt_bench
{
    namespace
    {
        // Device-side release gate. Each device's timed stream enqueues one instance of
        // this kernel BEFORE its timing-start event, so the stream blocks here until the
        // host flips *flag. Because the flag is flipped once for all devices, every
        // device's first timed launch begins within a few hundred nanoseconds of the
        // others regardless of host enqueue skew. maxSpins bounds the wait so a lost
        // release cannot hang the GPU indefinitely.
        __global__ void all_devices_release_gate(uint32_t* flag, unsigned long long maxSpins)
        {
            unsigned long long i = 0;
            while(__hip_atomic_load(flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) == 0u)
            {
                if(++i >= maxSpins)
                    break;
            }
        }

        // ~tens of billions of spins is multiple seconds of timeout; the host flips the
        // flag within a few milliseconds in the happy path.
        constexpr unsigned long long kGateMaxSpins = 50000000000ull;

        // Reusable, abortable host thread barrier. All N worker threads block in wait()
        // until the Nth arrives; the Nth runs `onLast` (while still holding the lock, so
        // its side effects are visible before any thread proceeds) and then releases all.
        // abort() unblocks every current and future waiter so a failing worker cannot
        // deadlock the rest.
        class Barrier
        {
        public:
            explicit Barrier(int count)
                : m_threshold(count)
                , m_count(count)
            {
            }

            // Returns true if this call was the releasing (last) arrival; false if it
            // returned because the barrier was aborted.
            bool wait(const std::function<void()>& onLast)
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                if(m_aborted)
                    return false;
                const unsigned gen = m_generation;
                if(--m_count == 0)
                {
                    if(onLast)
                        onLast();
                    m_generation++;
                    m_count = m_threshold;
                    m_cv.notify_all();
                    return true;
                }
                m_cv.wait(lock, [this, gen] { return gen != m_generation || m_aborted; });
                return false;
            }

            void abort()
            {
                {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    m_aborted = true;
                }
                m_cv.notify_all();
            }

        private:
            std::mutex              m_mutex;
            std::condition_variable m_cv;
            int                     m_threshold;
            int                     m_count;
            unsigned                m_generation = 0;
            bool                    m_aborted    = false;
        };

        // Shared coordinator for all worker threads. Lives from all_devices_enable() to
        // all_devices_report_and_reset().
        struct Coordinator
        {
            bool            enabled = false;
            int             W       = 0;
            LaunchSyncMode  mode    = LaunchSyncMode::Gate;
            TimeAggregation agg     = TimeAggregation::Median;

            // Gate-mode host-coherent release flag (mapped to a device pointer usable by
            // every device's gate kernel via system-scope atomics).
            uint32_t* flagHost = nullptr;
            uint32_t* flagDev  = nullptr;

            std::unique_ptr<Barrier> entryBarrier; // round start: reset flag, all GPUs done
            std::unique_ptr<Barrier> releaseBarrier; // all gates enqueued: schedule release

            std::mutex                    resultsMutex;
            std::vector<AllDevicesResult> results;

            // Return to the disabled state, releasing owned resources. (Coordinator holds
            // unique_ptr/mutex members and so is not assignable from a fresh instance.)
            void reset()
            {
                enabled = false;
                W       = 0;
                mode    = LaunchSyncMode::Gate;
                agg     = TimeAggregation::Median;
                flagHost = nullptr;
                flagDev  = nullptr;
                entryBarrier.reset();
                releaseBarrier.reset();
                results.clear();
            }
        };

        Coordinator g_coord;

        // Per-worker-thread binding set by all_devices_bind_thread().
        thread_local int t_slot      = -1;
        thread_local int t_device_id = -1;

        double median_of(std::vector<double> v)
        {
            if(v.empty())
                return std::numeric_limits<double>::quiet_NaN();
            std::sort(v.begin(), v.end());
            const size_t n = v.size();
            return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
        }

        double mean_of(const std::vector<double>& v)
        {
            if(v.empty())
                return std::numeric_limits<double>::quiet_NaN();
            return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
        }

        // Per-call GFLOP/s from a per-call time and GEMM GFLOP count (batch folded in).
        double gflops_per_sec(double gflopCount, double batch, double timeUs)
        {
            return timeUs > 0.0 ? gflopCount * batch / timeUs * 1e6
                                : std::numeric_limits<double>::quiet_NaN();
        }

        double gbytes_per_sec(double gbytes, double timeUs)
        {
            return timeUs > 0.0 ? gbytes / timeUs * 1e6
                                : std::numeric_limits<double>::quiet_NaN();
        }
    } // namespace

    void all_devices_enable(const int             count,
                            const LaunchSyncMode  syncMode,
                            const TimeAggregation aggregation)
    {
        g_coord.reset();
        g_coord.enabled  = true;
        g_coord.W        = std::max(1, count);
        g_coord.mode     = syncMode;
        g_coord.agg      = aggregation;
        g_coord.results.assign(g_coord.W, AllDevicesResult{});
        g_coord.entryBarrier   = std::make_unique<Barrier>(g_coord.W);
        g_coord.releaseBarrier = std::make_unique<Barrier>(g_coord.W);

        if(g_coord.mode == LaunchSyncMode::Gate)
        {
            // Allocate the shared release flag on device 0; the mapped device pointer is
            // valid for every device's gate kernel through system-scope atomics.
            int prevDevice = 0;
            (void)hipGetDevice(&prevDevice);
            (void)hipSetDevice(0);
            if(hipHostMalloc((void**)&g_coord.flagHost,
                             sizeof(uint32_t),
                             hipHostMallocMapped | hipHostMallocCoherent)
               != hipSuccess)
            {
                std::cerr << "[all-devices] failed to allocate the release flag; falling back "
                             "to barrier launch-sync mode"
                          << std::endl;
                g_coord.flagHost = nullptr;
                g_coord.mode     = LaunchSyncMode::Barrier;
            }
            else
            {
                *g_coord.flagHost = 0u;
                (void)hipHostGetDevicePointer((void**)&g_coord.flagDev, g_coord.flagHost, 0);
            }
            (void)hipSetDevice(prevDevice);
        }
    }

    bool all_devices_enabled()
    {
        return g_coord.enabled;
    }

    void all_devices_bind_thread(const int slot, const int device_id)
    {
        t_slot      = slot;
        t_device_id = device_id;
    }

    void all_devices_sync_point(hipStream_t stream)
    {
        if(!g_coord.enabled)
            return;

        if(g_coord.mode == LaunchSyncMode::Barrier)
        {
            // Align the host-side start of the timed region across all worker threads.
            g_coord.releaseBarrier->wait(nullptr);
            return;
        }

        // Gate mode. Two rendezvous frame every round:
        //  (A) entry: every worker is here => the previous round's GPU work is complete
        //      (each blocked on it before looping), so the release flag can be reset to 0
        //      before any gate kernel reads it this round.
        //  (B) release: every worker has enqueued its gate kernel, so a detached releaser
        //      can flip the flag after a short delay; by then each worker has also
        //      enqueued its timed region (while its GPU spins at the gate), so all devices
        //      leave the gate together.
        uint32_t* flagHost = g_coord.flagHost;
        g_coord.entryBarrier->wait([flagHost] {
            if(flagHost)
                __atomic_store_n(flagHost, 0u, __ATOMIC_RELEASE);
        });

        hipLaunchKernelGGL(all_devices_release_gate,
                           dim3(1),
                           dim3(1),
                           0,
                           stream,
                           g_coord.flagDev,
                           kGateMaxSpins);

        g_coord.releaseBarrier->wait([flagHost] {
            if(!flagHost)
                return;
            std::thread([flagHost] {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                __atomic_store_n(flagHost, 1u, __ATOMIC_RELEASE);
            }).detach();
        });
    }

    void all_devices_record_result(const AllDevicesResult& result)
    {
        if(!g_coord.enabled)
            return;
        std::lock_guard<std::mutex> lock(g_coord.resultsMutex);
        if(t_slot >= 0 && t_slot < static_cast<int>(g_coord.results.size()))
        {
            g_coord.results[t_slot]           = result;
            g_coord.results[t_slot].valid     = true;
            g_coord.results[t_slot].device_id = t_device_id;
        }
    }

    void all_devices_abort()
    {
        if(!g_coord.enabled)
            return;
        if(g_coord.entryBarrier)
            g_coord.entryBarrier->abort();
        if(g_coord.releaseBarrier)
            g_coord.releaseBarrier->abort();
    }

    int all_devices_report_and_reset()
    {
        if(!g_coord.enabled)
            return 0;

        const int W = g_coord.W;

        std::vector<double> us, tflops, gbps;
        bool                anyBytes = false;
        int                 missing  = 0;

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "\n[all-devices] ===== per-GPU results ("
                  << (g_coord.mode == LaunchSyncMode::Gate ? "gate" : "barrier")
                  << " launch-sync) =====" << std::endl;

        for(int i = 0; i < W; i++)
        {
            const AllDevicesResult& r = g_coord.results[i];
            if(!r.valid || !(r.time_us > 0.0))
            {
                std::cout << "[all-devices]   slot " << i << " (dev " << r.device_id
                          << "): no valid result" << std::endl;
                missing++;
                continue;
            }
            const double t  = r.time_us;
            const double tf = gflops_per_sec(r.gflop_count, (double)r.batch_count, t) / 1000.0;
            const double gb = gbytes_per_sec(r.gbytes, t);
            us.push_back(t);
            tflops.push_back(tf);
            if(r.gbytes > 0.0)
            {
                gbps.push_back(gb);
                anyBytes = true;
            }
            std::cout << "[all-devices]   dev " << r.device_id << ": " << t << " us, " << tf
                      << " TFLOP/s";
            if(r.gbytes > 0.0)
                std::cout << ", " << gb << " GB/s";
            if(!r.solution_name.empty())
                std::cout << "  [" << r.solution_name << "]";
            std::cout << std::endl;
        }

        if(!us.empty())
        {
            const char* aggName
                = (g_coord.agg == TimeAggregation::Median) ? "median" : "mean";
            const double usMed  = median_of(us);
            const double usMean = mean_of(us);
            const double tfMed  = median_of(tflops);
            const double tfMean = mean_of(tflops);
            const double tfTotal
                = std::accumulate(tflops.begin(), tflops.end(), 0.0);

            std::cout << "[all-devices] ----- across " << us.size() << " GPU(s) (headline="
                      << aggName << ") -----" << std::endl;
            std::cout << "[all-devices]   time_us:    median=" << usMed << " mean=" << usMean
                      << std::endl;
            std::cout << "[all-devices]   per-GPU TFLOP/s: median=" << tfMed << " mean=" << tfMean
                      << std::endl;
            std::cout << "[all-devices]   total-node TFLOP/s: " << tfTotal << std::endl;
            if(anyBytes && !gbps.empty())
            {
                const double gbMed   = median_of(gbps);
                const double gbMean  = mean_of(gbps);
                const double gbTotal = std::accumulate(gbps.begin(), gbps.end(), 0.0);
                std::cout << "[all-devices]   per-GPU GB/s: median=" << gbMed << " mean=" << gbMean
                          << std::endl;
                std::cout << "[all-devices]   total-node GB/s: " << gbTotal << std::endl;
            }
        }
        else
        {
            std::cout << "[all-devices]   no GPU produced a valid timing" << std::endl;
        }

        // Release resources and leave all-devices mode.
        if(g_coord.flagHost)
            (void)hipHostFree(g_coord.flagHost);
        const int rc = missing;
        g_coord.reset();
        return rc == 0 ? 0 : 1;
    }
} // namespace hipblaslt_bench
