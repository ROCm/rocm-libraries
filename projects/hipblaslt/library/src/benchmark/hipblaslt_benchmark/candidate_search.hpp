// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "benchmark_timing.hpp" // run_measurement, DefaultHooks

#include <algorithm>
#include <cmath>
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
#include <vector>

// ============================================================================
// Picking the fastest of several candidate solutions for one problem.
//
// Each candidate is prepared, given its cold (untimed) launches, and then timed by
// run_measurement, with the instruction cache flushed after every timed launch when
// asked. With a skip-slow ratio set, the cold launches are timed too, and a candidate
// whose cold launches take longer than the best cold time so far divided by the ratio
// is dropped before its timed phase. The useful range is (0, 1): a ratio above 1 drops
// every candidate, the first included. With the screen prepass, every candidate's cold
// launches are timed before any candidate is timed, and each is compared with the best
// cold time of them all, so the screen does not depend on the order of the candidates.
// The winner is the candidate with the lowest median; on a tie the earlier candidate
// stays.
//
// With more than one pass, every candidate is timed once per pass, the passes taking
// the candidates in the reversed and the listed order by turns and the last in the
// listed order, and its time is the median of its passes. A device whose clock is
// still moving when a candidate is timed reads it faster or slower by where it sits in
// the list; two passes give every candidate the same mean place. With more than one
// pass the skip screen, when set, always runs as the prepass, once, before the passes.
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
        float   preheat_ms      = 0.0f; // untimed launches of the first candidate, 0 for none
        int32_t passes          = 1; // timed passes over the candidates (search_candidates)
        // With skip_slow_ratio set, screen every candidate before timing any: prepare it
        // and time its cold launches, after one untimed launch that keeps lazy code object
        // loading out of the time; then prepare and time only the candidates within the
        // ratio of the best cold time of all of them.
        bool screen_prepass = false;
    };

    struct SearchResult
    {
        size_t       best = std::numeric_limits<size_t>::max(); // none measured
        TimingResult best_timing;

        // A fatal failure under gtest, reported through should_abort, ended the search.
        bool aborted = false;

        // should_stop ended the search before every candidate had been tried.
        bool stopped = false;

        // Launches the preheat ran, 0 without one.
        int64_t preheat_launches = 0;
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

        // Calls launch(i) back to back, i counting from 0, until preheat_us have passed on
        // the hooks' clock, and adds the launches to `launches`. The first launch is not
        // counted in the time, since it can include loading the kernel's code. The host
        // synchronizes only after each batch; the first batch is one launch, and each later
        // one is sized from the time per launch so far to end the preheat with the fewest
        // launches past it. Stops early when the clock fails or `stop` returns true, and
        // after 2^24 launches.
        template <typename Hooks, typename Launch, typename Stop>
        void preheat(Launch&     launch,
                     double      preheat_us,
                     hipStream_t stream,
                     Stop&&      stop,
                     int64_t&    launches)
        {
            constexpr int64_t max_batch    = int64_t(1) << 16;
            constexpr int64_t max_launches = int64_t(1) << 24;
            int64_t           index        = 0;
            launch(index++);
            double start = 0.0;
            if(!Hooks::time_us_sync(stream, start) || stop())
            {
                launches += index;
                return;
            }
            double  now   = start;
            int64_t batch = 1;
            while(now - start < preheat_us && index < max_launches)
            {
                for(int64_t k = 0; k < batch; k++)
                    launch(index++);
                const double before = now;
                if(!Hooks::time_us_sync(stream, now) || stop())
                    break;
                const double remaining  = preheat_us - (now - start);
                const double per_launch = (now - before) / static_cast<double>(batch);
                if(remaining <= 0.0)
                    break;
                batch = per_launch > 0.0
                            ? static_cast<int64_t>(std::min(
                                std::ceil(remaining / per_launch), static_cast<double>(max_batch)))
                            : std::min(max_batch, batch * 2);
                batch = std::max<int64_t>(1, std::min(batch, max_launches - index));
            }
            launches += index;
        }

        // The timing of a candidate timed in several passes, from the timings of the
        // passes before the last (`earlier`) and of the last: the median of the passes'
        // medians (the mean of the two middle ones of an even count), the lowest minimum,
        // the mean over every timed launch, the samples and launches of all of them, the
        // largest dispersion, converged and stable only when every pass was, and the rest
        // from the last pass.
        inline TimingResult combine_passes(const std::vector<TimingResult>& earlier,
                                           const TimingResult&              last)
        {
            std::vector<TimingResult> all(earlier);
            all.push_back(last);
            TimingResult        out = last;
            std::vector<double> medians;
            double              weighted = 0.0;
            double              means    = 0.0;
            out.samples                  = 0;
            out.hot_iters                = 0;
            for(const TimingResult& t : all)
            {
                medians.push_back(t.median_us);
                out.min_us    = std::min(out.min_us, t.min_us);
                out.cv        = std::max(out.cv, t.cv);
                out.rel_iqr   = std::max(out.rel_iqr, t.rel_iqr);
                out.samples   = out.samples + t.samples;
                out.hot_iters = out.hot_iters + t.hot_iters;
                out.converged = out.converged && t.converged;
                out.stable    = out.stable && t.stable;
                weighted += t.mean_us * static_cast<double>(t.hot_iters);
                means += t.mean_us;
            }
            std::sort(medians.begin(), medians.end());
            out.median_us = quantile(medians, 0.5);
            out.mean_us   = out.hot_iters > 0 ? weighted / static_cast<double>(out.hot_iters)
                                              : means / static_cast<double>(all.size());
            return out;
        }

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
    // For each candidate sol, in this order (with SearchConfig::screen_prepass, a first pass
    // calls should_stop(), prepare(sol) and the cold launches of every candidate, its first
    // cold launch untimed, and the screen compares each candidate with the best cold time of
    // them all; a second pass then calls should_stop() and, for a candidate the first pass
    // timed, skipped() if the screen drops it, else prepare(sol), its cold launches untimed
    // and the steps below; a candidate the first pass could not prepare or time is passed
    // over. With SearchConfig::passes above 1, the steps from should_stop() to the timed
    // launches run once per pass, the passes alternating between the reversed and the
    // listed order and ending in the listed order, but skipped(), measuring() and
    // measured() only in the last, measured() with the candidate's timing over all its
    // passes; a candidate a pass could not prepare or time is passed over in the passes
    // after it):
    //   should_stop()                        returning true ends the search
    //   prepare(sol)                         before anything else runs for sol; returning
    //                                        false skips sol
    //   launch(sol, i)                       the preheat, untimed, for the first candidate
    //                                        that prepares, when SearchConfig::preheat_ms is set
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
        bool                        preheat_due    = search.preheat_ms > 0.0f;

        const int32_t passes = std::max<int32_t>(1, search.passes);

        // The screen prepass: each candidate's cold time, or a negative value for one
        // that was not prepared or whose cold launches could not be timed.
        const bool prepass = search.skip_slow_ratio && (search.screen_prepass || passes > 1);
        std::vector<double> screened;
        if(prepass)
        {
            screened.assign(candidates, -1.0);
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
                int64_t index = 0;
                if(search.cold_calls > 0)
                    cold(index++);
                double     warm_time = 0.0;
                const bool warm_ok   = detail::time_batch<Hooks>(
                    cold, search.cold_calls, index, timing.use_gpu_timer, events, warm_time);
                if(newly_failed())
                {
                    result.aborted = true;
                    return result;
                }
                if(!warm_ok)
                    continue;
                screened[sol]  = warm_time;
                best_warm_time = best_warm_time < warm_time ? best_warm_time : warm_time;
            }
        }

        // With more than one pass, the timings the passes before the last took of each
        // candidate, and whether one of them passed it over.
        std::vector<std::vector<TimingResult>> earlier(passes > 1 ? candidates : 0);
        std::vector<char>                      passed_over(passes > 1 ? candidates : 0, 0);

        for(int32_t pass = 0; pass < passes; pass++)
        {
            const bool last     = pass + 1 == passes;
            const bool reversed = (passes - 1 - pass) % 2 != 0;
            for(size_t step = 0; step < candidates; step++)
            {
                const size_t sol = reversed ? candidates - 1 - step : step;
                if(callbacks.on_should_stop())
                {
                    result.stopped = true;
                    return result;
                }

                if(passes > 1 && passed_over[sol])
                    continue;
                if(prepass)
                {
                    if(screened[sol] < 0.0)
                        continue;
                    if((screened[sol] * search.skip_slow_ratio) > best_warm_time)
                    {
                        if(last)
                            callbacks.on_skipped(sol, screened[sol], best_warm_time);
                        continue;
                    }
                }

                const bool failed_before = detail::aborted(should_abort);
                auto       newly_failed
                    = [&] { return !failed_before && detail::aborted(should_abort); };
                // A pass before the last that cannot prepare or time sol passes it over
                // in the passes after it.
                auto pass_over = [&] {
                    if(!last)
                        passed_over[sol] = 1;
                };

                const bool prepared = detail::call_hook(callbacks.on_prepare, sol);
                if(newly_failed())
                {
                    result.aborted = true;
                    return result;
                }
                if(!prepared)
                {
                    pass_over();
                    continue;
                }

                if(preheat_due)
                {
                    preheat_due = false;
                    auto launch = [&](int64_t i) {
                        callbacks.on_launch(sol, i);
                        if(search.flush)
                            launch_flush_icache(search.flush_grid, stream);
                    };
                    detail::preheat<Hooks>(launch,
                                           static_cast<double>(search.preheat_ms) * 1000.0,
                                           stream,
                                           newly_failed,
                                           result.preheat_launches);
                    if(newly_failed())
                    {
                        result.aborted = true;
                        return result;
                    }
                }

                bool cold_failed = false;
                auto cold        = [&](int64_t i) {
                    if(cold_failed)
                        return;
                    callbacks.on_cold(sol, static_cast<int32_t>(i));
                    cold_failed = newly_failed();
                };
                double warm_time = 0.0;
                bool   warm_ok   = true;
                if(search.skip_slow_ratio && !prepass)
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
                {
                    pass_over();
                    continue;
                }

                if(search.skip_slow_ratio && !prepass)
                {
                    best_warm_time = best_warm_time < warm_time ? best_warm_time : warm_time;
                    if((warm_time * search.skip_slow_ratio) > best_warm_time)
                    {
                        callbacks.on_skipped(sol, warm_time, best_warm_time);
                        continue;
                    }
                }

                TimingResult measurement;
                if(last)
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
                {
                    pass_over();
                    continue;
                }
                if(!last)
                {
                    earlier[sol].push_back(measurement);
                    continue;
                }
                if(passes > 1)
                    measurement = detail::combine_passes(earlier[sol], measurement);
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
        }
        return result;
    }
} // namespace hipblaslt_bench
