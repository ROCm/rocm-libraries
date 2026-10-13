// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "benchmark_timing.hpp" // run_measurement, the HIPBLASLT_BENCHMARK_* hooks

#include <cstddef>
#include <cstdint>
#include <functional>
#include <hip/hip_runtime.h>
#include <limits>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

// ============================================================================
// Picking the fastest of several candidate solutions for one problem.
//
// Each candidate is prepared, given its cold (untimed) launches, and then timed by
// run_measurement, with the instruction cache flushed after every timed launch when
// asked. With a skip-slow ratio set, the cold launches are timed too, and a candidate
// whose cold launches are slower than the best seen so far by more than the ratio is
// dropped before its timed phase. The winner is the candidate with the lowest median;
// on a tie the earlier candidate stays.
// ============================================================================

namespace hipblaslt_bench
{
    // Invalidates the instruction cache, so the next kernel is not timed with its own
    // code resident. A template so that every translation unit can instantiate it.
    template <typename = void>
    __global__ void flush_icache()
    {
        asm __volatile__("s_icache_inv \n\t"
                         "s_nop 0 \n\t"
                         "s_nop 0 \n\t"
                         "s_nop 0 \n\t"
                         "s_nop 0 \n\t"
                         "s_nop 0 \n\t"
                         "s_nop 0 \n\t"
                         "s_nop 0 \n\t"
                         "s_nop 0 \n\t"
                         "s_nop 0 \n\t"
                         "s_nop 0 \n\t"
                         "s_nop 0 \n\t"
                         "s_nop 0 \n\t"
                         "s_nop 0 \n\t"
                         "s_nop 0 \n\t"
                         "s_nop 0 \n\t"
                         "s_nop 0 \n\t" ::
                             :);
    }

    // Workgroups in one flush launch, each of 64 lanes.
    inline int32_t flush_icache_grid(const hipDeviceProp_t& props)
    {
        return props.multiProcessorCount * 60;
    }

    inline void launch_flush_icache(int32_t grid, hipStream_t stream)
    {
        hipLaunchKernelGGL(flush_icache<>, dim3(grid), dim3(64), 0, stream);
    }

    // Opens a timed span: records event_start on the GPU timer, otherwise reads the CPU
    // clock into time_used.
    inline void pre_gpu_time(bool         use_gpu_timer,
                             hipEvent_t&  event_gpu_time_start,
                             double&      gpu_time_used,
                             hipStream_t& stream)
    {
        if(use_gpu_timer)
            HIPBLASLT_BENCHMARK_CHECK_HIP(hipEventRecord(event_gpu_time_start, stream));
        else
            gpu_time_used = HIPBLASLT_BENCHMARK_TIME_US_SYNC(stream);
    }

    // Closes the span pre_gpu_time opened, leaving its length in microseconds in
    // gpu_time_used.
    inline void post_gpu_time(bool         use_gpu_timer,
                              hipEvent_t&  event_gpu_time_start,
                              hipEvent_t&  event_gpu_time_end,
                              double&      gpu_time_used,
                              hipStream_t& stream)
    {
        if(use_gpu_timer)
        {
            HIPBLASLT_BENCHMARK_CHECK_HIP(hipEventRecord(event_gpu_time_end, stream));
            HIPBLASLT_BENCHMARK_CHECK_HIP(hipEventSynchronize(event_gpu_time_end));
            float gpu_time_ms;
            HIPBLASLT_BENCHMARK_CHECK_HIP(
                hipEventElapsedTime(&gpu_time_ms, event_gpu_time_start, event_gpu_time_end));
            gpu_time_used = gpu_time_ms * 1000; // ms to us
        }
        else
        {
            gpu_time_used = HIPBLASLT_BENCHMARK_TIME_US_SYNC(stream) - gpu_time_used;
        }
    }

    // Microseconds one flush launch costs on the device `props` describes, for
    // TimingConfig::flush_us. Measured with the given timer the first time a device is
    // asked about, then reused for the life of the process.
    inline double flush_cost_us(bool                   use_gpu_timer,
                                const hipDeviceProp_t& props,
                                hipEvent_t&            event_gpu_time_start,
                                hipEvent_t&            event_gpu_time_end,
                                hipStream_t&           stream)
    {
        const int32_t gpu_block3      = flush_icache_grid(props);
        int           flush_iter      = 100000;
        double        flush_time_used = 0;

        static std::unordered_map<std::string, double> flush_times_cache;
        static std::mutex                              mtx;
        std::lock_guard<std::mutex>                    lock(mtx);
        std::string                                    device_uuid(props.uuid.bytes);
        if(!flush_times_cache.count(device_uuid))
        {
            for(int i = 0; i < flush_iter; i++)
                launch_flush_icache(gpu_block3, stream);
            pre_gpu_time(use_gpu_timer, event_gpu_time_start, flush_time_used, stream);
            for(int i = 0; i < flush_iter; i++)
                launch_flush_icache(gpu_block3, stream);
            post_gpu_time(
                use_gpu_timer, event_gpu_time_start, event_gpu_time_end, flush_time_used, stream);
            flush_time_used /= flush_iter;
            flush_times_cache[device_uuid] = flush_time_used;
        }
        else
        {
            flush_time_used = flush_times_cache[device_uuid];
        }
        return flush_time_used;
    }

    struct SearchConfig
    {
        int32_t cold_calls      = 0; // untimed launches before each candidate is timed
        float   skip_slow_ratio = 0.0f; // 0 keeps every candidate
        bool    flush           = false; // flush the instruction cache after each timed launch
        int32_t flush_grid      = 0; // workgroups per flush, from flush_icache_grid
    };

    struct SearchResult
    {
        size_t       best = std::numeric_limits<size_t>::max(); // none measured
        TimingResult best_timing;

        // Preparing a candidate failed fatally under gtest, which ends the search.
        bool aborted = false;

        // should_stop ended the search before every candidate had been tried.
        bool stopped = false;
    };

    namespace detail
    {
        // A hook that returns nothing counts as having returned true.
        template <typename Hook, typename... Args>
        bool call_hook(Hook& hook, Args&&... args)
        {
            if constexpr(std::is_void_v<std::invoke_result_t<Hook&, Args...>>)
            {
                hook(std::forward<Args>(args)...);
                return true;
            }
            else
            {
                return static_cast<bool>(hook(std::forward<Args>(args)...));
            }
        }
    } // namespace detail

    // Searches `candidates` candidates, numbered from 0, and returns the winner.
    //   prepare(sol)                       before anything else runs for sol; returning
    //                                      false skips sol
    //   cold(sol, i)                       one cold launch, i counting from 0
    //   launch(sol, i)                     one timed launch, i counting from 0
    //   skipped(sol, warm_us, best_warm_us) sol was dropped by the skip-slow screen
    //   measuring(sol)                     just before sol's timed phase
    //   measured(sol, timing)              just after it; returning false keeps sol from
    //                                      winning
    // should_abort is polled as run_measurement polls it. should_stop, when given, is
    // asked before each candidate and ends the search by returning true.
    template <typename Prepare,
              typename Cold,
              typename Launch,
              typename Skipped,
              typename Measuring,
              typename Measured>
    SearchResult search_candidates(size_t                       candidates,
                                   const SearchConfig&          search,
                                   const TimingConfig&          timing,
                                   hipEvent_t&                  event_gpu_time_start,
                                   hipEvent_t&                  event_gpu_time_end,
                                   hipStream_t&                 stream,
                                   Prepare&&                    prepare,
                                   Cold&&                       cold,
                                   Launch&&                     launch,
                                   Skipped&&                    skipped,
                                   Measuring&&                  measuring,
                                   Measured&&                   measured,
                                   const std::function<bool()>& should_abort = {},
                                   const std::function<bool()>& should_stop  = {})
    {
        SearchResult result;
        double       best_gpu_time  = std::numeric_limits<double>::max();
        double       best_warm_time = std::numeric_limits<double>::max();

        for(size_t sol = 0; sol < candidates; sol++)
        {
            if(should_stop && should_stop())
            {
                result.stopped = true;
                return result;
            }

            const bool failed_before = detail::aborted(should_abort);
            const bool prepared      = detail::call_hook(prepare, sol);
            if(!failed_before && detail::aborted(should_abort))
            {
                result.aborted = true;
                return result;
            }
            if(!prepared)
                continue;

            double warm_time = 0;
            if(search.skip_slow_ratio)
                pre_gpu_time(timing.use_gpu_timer, event_gpu_time_start, warm_time, stream);
            for(int32_t i = 0; i < search.cold_calls; i++)
                cold(sol, i);
            if(search.skip_slow_ratio)
            {
                post_gpu_time(timing.use_gpu_timer,
                              event_gpu_time_start,
                              event_gpu_time_end,
                              warm_time,
                              stream);
                best_warm_time = best_warm_time < warm_time ? best_warm_time : warm_time;
                if((warm_time * search.skip_slow_ratio) > best_warm_time)
                {
                    skipped(sol, warm_time, best_warm_time);
                    continue;
                }
            }

            TimingResult measurement;
            measuring(sol);
            run_measurement(
                [&](int64_t i) {
                    launch(sol, i);
                    if(search.flush)
                        launch_flush_icache(search.flush_grid, stream);
                },
                timing,
                event_gpu_time_start,
                event_gpu_time_end,
                stream,
                measurement,
                should_abort);
            const bool eligible = detail::call_hook(measured, sol, measurement);

            if(eligible && best_gpu_time > measurement.median_us)
            {
                best_gpu_time      = measurement.median_us;
                result.best        = sol;
                result.best_timing = measurement;
            }
        }
        return result;
    }
} // namespace hipblaslt_bench
