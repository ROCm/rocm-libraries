// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "benchmark_timing.hpp" // run_measurement, DefaultHooks

#include <cstddef>
#include <cstdint>
#include <functional>
#include <hip/hip_runtime.h>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
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
// whose cold launches take longer than the best cold time so far divided by the ratio
// is dropped before its timed phase. The useful range is (0, 1): a ratio above 1 drops
// every candidate, the first included. The winner is the candidate with the lowest
// median; on a tie the earlier candidate stays.
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

    // Sets cost_us to the microseconds one flush launch costs on the device `props`
    // describes, for TimingConfig::flush_us, and returns false if the hooks report a
    // failure. Measured with the given timer the first time a device is asked about with
    // that timer, then reused for the life of the process. Devices calibrate in parallel,
    // but one device never runs two calibrations at once, since each would time the
    // other's launches.
    template <typename Hooks = DefaultHooks>
    [[nodiscard]] bool flush_cost_us(bool                   use_gpu_timer,
                                     const hipDeviceProp_t& props,
                                     hipEvent_t             event_start,
                                     hipEvent_t             event_stop,
                                     hipStream_t            stream,
                                     double&                cost_us)
    {
        struct Device
        {
            std::mutex            mutex;
            std::optional<double> cost_us[2]; // by use_gpu_timer
        };
        static std::mutex                                               devices_mutex;
        static std::unordered_map<std::string, std::unique_ptr<Device>> devices;

        Device* device = nullptr;
        {
            std::lock_guard<std::mutex> lock(devices_mutex);
            auto& slot = devices[std::string(props.uuid.bytes, sizeof(props.uuid.bytes))];
            if(!slot)
                slot = std::make_unique<Device>();
            device = slot.get();
        }

        std::lock_guard<std::mutex> lock(device->mutex);
        std::optional<double>&      cached = device->cost_us[use_gpu_timer ? 1 : 0];
        if(!cached)
        {
            const int32_t flush_iter = 100000;
            const int32_t grid       = flush_icache_grid(props);
            auto          flush      = [&](int64_t) { launch_flush_icache(grid, stream); };
            for(int32_t i = 0; i < flush_iter; i++)
                flush(i);
            int64_t index      = 0;
            double  elapsed_us = 0.0;
            if(!detail::time_batch<Hooks>(flush,
                                          flush_iter,
                                          index,
                                          use_gpu_timer,
                                          {event_start, event_stop, stream},
                                          elapsed_us))
                return false;
            cached = elapsed_us / flush_iter;
        }
        cost_us = *cached;
        return true;
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

        // A fatal failure under gtest, reported through should_abort, ended the search.
        bool aborted = false;

        // should_stop ended the search before every candidate had been tried.
        bool stopped = false;
    };

    namespace detail
    {
        // What search_callbacks() sets every callback but should_stop to: nothing.
        struct no_callback
        {
            template <typename... Args>
            void operator()(Args&&...) const
            {
            }
        };

        // What search_callbacks() sets should_stop to.
        struct never_stop
        {
            bool operator()() const
            {
                return false;
            }
        };

        // A callback that returns nothing counts as having returned true.
        template <typename Callback, typename... Args>
        bool call_hook(Callback& callback, Args&&... args)
        {
            if constexpr(std::is_void_v<std::invoke_result_t<Callback&, Args...>>)
            {
                callback(std::forward<Args>(args)...);
                return true;
            }
            else
            {
                return static_cast<bool>(callback(std::forward<Args>(args)...));
            }
        }
    } // namespace detail

    // The callbacks search_candidates calls, each set by name, so that two taking the
    // same arguments cannot take each other's place:
    //   search_callbacks().prepare(p).cold(c).launch(l).skipped(s).measuring(m).measured(d)
    //       .should_stop(t)
    // For each candidate sol, in this order:
    //   should_stop()                        returning true ends the search
    //   prepare(sol)                         before anything else runs for sol; returning
    //                                        false skips sol
    //   cold(sol, i)                         one cold launch, i counting from 0
    //   skipped(sol, warm_us, best_warm_us)  sol was dropped by the skip-slow screen
    //   measuring(sol)                       just before sol's timed phase
    //   launch(sol, i)                       one timed launch, i counting from 0
    //   measured(sol, timing)                just after a timed phase that succeeded;
    //                                        returning false keeps sol from winning
    // A callback left unset does nothing, and an unset should_stop never stops.
    template <typename Prepare    = detail::no_callback,
              typename Cold       = detail::no_callback,
              typename Launch     = detail::no_callback,
              typename Skipped    = detail::no_callback,
              typename Measuring  = detail::no_callback,
              typename Measured   = detail::no_callback,
              typename ShouldStop = detail::never_stop>
    struct SearchCallbacks
    {
        Prepare    on_prepare;
        Cold       on_cold;
        Launch     on_launch;
        Skipped    on_skipped;
        Measuring  on_measuring;
        Measured   on_measured;
        ShouldStop on_should_stop;

        template <typename F>
        SearchCallbacks<F, Cold, Launch, Skipped, Measuring, Measured, ShouldStop>
            prepare(F f) const
        {
            return {f, on_cold, on_launch, on_skipped, on_measuring, on_measured, on_should_stop};
        }

        template <typename F>
        SearchCallbacks<Prepare, F, Launch, Skipped, Measuring, Measured, ShouldStop>
            cold(F f) const
        {
            return {
                on_prepare, f, on_launch, on_skipped, on_measuring, on_measured, on_should_stop};
        }

        template <typename F>
        SearchCallbacks<Prepare, Cold, F, Skipped, Measuring, Measured, ShouldStop>
            launch(F f) const
        {
            return {on_prepare, on_cold, f, on_skipped, on_measuring, on_measured, on_should_stop};
        }

        template <typename F>
        SearchCallbacks<Prepare, Cold, Launch, F, Measuring, Measured, ShouldStop>
            skipped(F f) const
        {
            return {on_prepare, on_cold, on_launch, f, on_measuring, on_measured, on_should_stop};
        }

        template <typename F>
        SearchCallbacks<Prepare, Cold, Launch, Skipped, F, Measured, ShouldStop>
            measuring(F f) const
        {
            return {on_prepare, on_cold, on_launch, on_skipped, f, on_measured, on_should_stop};
        }

        template <typename F>
        SearchCallbacks<Prepare, Cold, Launch, Skipped, Measuring, F, ShouldStop>
            measured(F f) const
        {
            return {on_prepare, on_cold, on_launch, on_skipped, on_measuring, f, on_should_stop};
        }

        template <typename F>
        SearchCallbacks<Prepare, Cold, Launch, Skipped, Measuring, Measured, F>
            should_stop(F f) const
        {
            return {on_prepare, on_cold, on_launch, on_skipped, on_measuring, on_measured, f};
        }
    };

    inline SearchCallbacks<> search_callbacks()
    {
        return {};
    }

    // Searches `candidates` candidates, numbered from 0, calling `callbacks` from
    // search_callbacks(), and returns the winner. A candidate whose cold or timed span
    // fails, as the hooks report, is passed over: neither skipped nor measured, it cannot
    // win. An invalid adaptive TimingConfig searches nothing; validate_adaptive_config gives
    // the reason. should_abort is asked after each prepare and cold launch, after the cold
    // span is timed and after the timed phase, and a failure it reports that was not there
    // when the candidate began ends the search.
    template <typename Hooks = DefaultHooks, typename Callbacks>
    SearchResult search_candidates(size_t                       candidates,
                                   const SearchConfig&          search,
                                   const TimingConfig&          timing,
                                   hipEvent_t                   event_start,
                                   hipEvent_t                   event_stop,
                                   hipStream_t                  stream,
                                   Callbacks&&                  callbacks,
                                   const std::function<bool()>& should_abort = {})
    {
        SearchResult result;
        if(timing.adaptive && !validate_adaptive_config(timing).empty())
            return result;

        const detail::event_context events{event_start, event_stop, stream};
        double                      best_gpu_time  = std::numeric_limits<double>::max();
        double                      best_warm_time = std::numeric_limits<double>::max();

        for(size_t sol = 0; sol < candidates; sol++)
        {
            if(callbacks.on_should_stop())
            {
                result.stopped = true;
                return result;
            }

            const bool failed_before = detail::aborted(should_abort);
            auto newly_failed = [&] { return !failed_before && detail::aborted(should_abort); };

            const bool prepared = detail::call_hook(callbacks.on_prepare, sol);
            if(newly_failed())
            {
                result.aborted = true;
                return result;
            }
            if(!prepared)
                continue;

            bool cold_failed = false;
            auto cold        = [&](int64_t i) {
                if(cold_failed)
                    return;
                callbacks.on_cold(sol, static_cast<int32_t>(i));
                cold_failed = newly_failed();
            };
            double warm_time = 0.0;
            bool   warm_ok   = true;
            if(search.skip_slow_ratio)
            {
                int64_t index = 0;
                warm_ok       = detail::time_batch<Hooks>(
                    cold, search.cold_calls, index, timing.use_gpu_timer, events, warm_time);
            }
            else
            {
                for(int32_t i = 0; i < search.cold_calls; i++)
                    cold(i);
            }
            if(newly_failed())
            {
                result.aborted = true;
                return result;
            }
            if(!warm_ok)
                continue;

            if(search.skip_slow_ratio)
            {
                best_warm_time = best_warm_time < warm_time ? best_warm_time : warm_time;
                if((warm_time * search.skip_slow_ratio) > best_warm_time)
                {
                    callbacks.on_skipped(sol, warm_time, best_warm_time);
                    continue;
                }
            }

            TimingResult measurement;
            callbacks.on_measuring(sol);
            const bool measured = run_measurement<Hooks>(
                [&](int64_t i) {
                    callbacks.on_launch(sol, i);
                    if(search.flush)
                        launch_flush_icache(search.flush_grid, stream);
                },
                timing,
                event_start,
                event_stop,
                stream,
                measurement,
                should_abort);
            if(newly_failed())
            {
                result.aborted = true;
                return result;
            }
            if(!measured)
                continue;
            const bool eligible = detail::call_hook(callbacks.on_measured, sol, measurement);
            if(newly_failed())
            {
                result.aborted = true;
                return result;
            }

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
