// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "benchmark_stats.hpp" // TimingConfig, TimingResult
#include "hipblaslt_test.hpp" // CHECK_HIP_ERROR
#include "utility.hpp" // get_time_us_sync
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <hip/hip_runtime.h>
#include <limits>
#include <numeric>
#include <vector>

// ============================================================================
// Adaptive, distribution-based timing for the hipBLASLt benchmark clients.
//
// A *sample* is one event span over `batch` back-to-back enqueues, yielding one
// per-iteration throughput number. The batch is sized from a warmup so each sample
// spans ~sample_time. Samples are collected for at least the floor (min_iters and
// measure_time); past the floor the run stops when either (a) the mean's relative
// standard error drops below noise_threshold (converged), or (b) the robust spread
// (rel_iqr) has plateaued, so more samples will not change the picture (stable) --
// a fallback for heavy-tailed/unlocked-clock kernels that never hit the precision
// target. Otherwise it runs to the max_measure_time / max_iters ceiling (noisy).
// noise_threshold <= 0 disables both checks, so the run goes to the ceiling. The
// median is the headline statistic; mean, min, cv and rel_iqr are also returned.
// With no adaptive knob set, a single fixed-count sample of `iters` is taken.
// ============================================================================

namespace hipblaslt_bench
{
    namespace detail
    {
        inline double mean_of(const std::vector<double>& v)
        {
            return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        }

        inline double sum_sq_dev(const std::vector<double>& v, double mean)
        {
            return std::accumulate(v.begin(), v.end(), 0.0, [mean](double acc, double x) {
                return acc + (x - mean) * (x - mean);
            });
        }

        // Incremental mean/variance via Welford's recurrence: O(1) per sample.
        struct running_stats
        {
            int64_t count = 0;
            double  mean  = 0.0;
            double  m2    = 0.0; // sum of squared deviations from the running mean

            void add(double x)
            {
                ++count;
                const double delta = x - mean;
                mean += delta / static_cast<double>(count);
                m2 += delta * (x - mean); // second delta uses the updated mean
            }

            // Relative standard error of the mean: (sample stddev / mean) / sqrt(n).
            double relative_std_error() const
            {
                if(count < 2 || mean <= 0.0)
                    return std::numeric_limits<double>::infinity(); // too few / degenerate
                const double var = m2 / static_cast<double>(count - 1); // unbiased variance
                return (std::sqrt(var) / mean) / std::sqrt(static_cast<double>(count));
            }
        };

        // Minimum samples before the convergence test is trusted; a stddev from fewer
        // is unreliable.
        inline constexpr int min_samples_for_convergence = 10;

        // Linear-interpolated quantile of a sorted, non-empty vector (q in [0, 1]).
        inline double quantile(const std::vector<double>& sorted, double q)
        {
            const double pos = q * static_cast<double>(sorted.size() - 1);
            const size_t lo  = static_cast<size_t>(std::floor(pos));
            const size_t hi  = static_cast<size_t>(std::ceil(pos));
            return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - static_cast<double>(lo));
        }

        // Robust relative dispersion (IQR / median) over all samples so far -- quartiles are
        // order statistics, so unlike cv this is insensitive to the outlier tail. Sorts
        // `samples` in place: callers must not rely on its order (nothing does -- mean/variance
        // come from running_stats and fill_distribution sorts its own view). Caller ensures
        // samples is non-empty.
        inline double cumulative_rel_iqr(std::vector<double>& samples)
        {
            std::sort(samples.begin(), samples.end());
            const double median = quantile(samples, 0.5);
            const double iqr    = quantile(samples, 0.75) - quantile(samples, 0.25);
            return median > 0.0 ? iqr / median : 0.0;
        }

        // Whether the rel_iqr trajectory has flattened: the last `window` readings have a
        // relative standard deviation below `threshold`. A window < 2 has no spread to test.
        inline bool
            noise_plateaued(const std::vector<double>& iqr_history, int window, double threshold)
        {
            if(window < 2 || static_cast<int>(iqr_history.size()) < window)
                return false;
            const std::vector<double> recent(iqr_history.end() - window, iqr_history.end());
            const double              mean = mean_of(recent);
            if(mean <= 0.0)
                return false;
            const double rel_spread
                = std::sqrt(sum_sq_dev(recent, mean) / (recent.size() - 1)) / mean;
            return rel_spread < threshold;
        }

        inline bool aborted(const std::function<bool()>& should_abort)
        {
            return should_abort && should_abort();
        }

        // The HIP events and stream used to time a batch.
        struct event_context
        {
            hipEvent_t  start  = nullptr;
            hipEvent_t  stop   = nullptr;
            hipStream_t stream = nullptr;
        };

        // Run `batch` back-to-back enqueues as one timed sample; returns elapsed
        // microseconds via `elapsed_us`. `global_index` advances per enqueue so rotating
        // buffers keep cycling across samples. Void (not value-returning) so CHECK_HIP_ERROR,
        // which can expand to `return;` under gtest, stays valid here.
        template <typename Launch>
        inline void time_batch(Launch&              launch,
                               int32_t              batch,
                               int64_t&             global_index,
                               bool                 use_gpu_timer,
                               const event_context& events,
                               double&              elapsed_us)
        {
            double cpu_start = 0.0;
            if(use_gpu_timer)
                CHECK_HIP_ERROR(hipEventRecord(events.start, events.stream));
            else
                cpu_start = get_time_us_sync(events.stream);

            for(int32_t k = 0; k < batch; ++k)
                launch(global_index++);

            if(use_gpu_timer)
            {
                CHECK_HIP_ERROR(hipEventRecord(events.stop, events.stream));
                CHECK_HIP_ERROR(hipEventSynchronize(events.stop));
                float ms = 0.0f;
                CHECK_HIP_ERROR(hipEventElapsedTime(&ms, events.start, events.stop));
                elapsed_us = static_cast<double>(ms) * 1000.0;
            }
            else
            {
                elapsed_us = get_time_us_sync(events.stream) - cpu_start;
            }
        }

        // Largest per-sample batch allowed: keep room for at least two samples within
        // max_iters (a variance estimate needs >= 2); unbounded when max_iters is unset.
        inline int64_t max_batch_size(const TimingConfig& cfg)
        {
            return cfg.max_iters > 0 ? std::max<int64_t>(1, cfg.max_iters / 2)
                                     : std::numeric_limits<int32_t>::max();
        }

        // Size of one timed sample: how many back-to-back enqueues run for about
        // `sample_time` ms, given the measured per-iteration time. At least 1 (one enqueue
        // per sample when sample_time is unset), at most `cap`.
        inline int32_t batch_size(const TimingConfig& cfg, double per_iter_us, int64_t cap)
        {
            if(cfg.sample_time <= 0.0f || per_iter_us <= 0.0)
                return 1;
            const double want
                = std::ceil(static_cast<double>(cfg.sample_time) * 1000.0 / per_iter_us);
            return static_cast<int32_t>(std::clamp(
                static_cast<int64_t>(want), static_cast<int64_t>(1), cap)); // cap <= INT32_MAX
        }

        // Fill the distribution fields of `out` (mean/median/min/cv/rel_iqr/samples) from
        // the collected per-iteration samples.
        inline void fill_distribution(const std::vector<double>& samples, TimingResult& out)
        {
            std::vector<double> sorted(samples);
            std::sort(sorted.begin(), sorted.end());
            const size_t n      = sorted.size();
            const double mean   = mean_of(samples);
            const double var    = n > 1 ? sum_sq_dev(samples, mean) / (n - 1) : 0.0;
            const double median = quantile(sorted, 0.5);
            const double iqr    = quantile(sorted, 0.75) - quantile(sorted, 0.25);

            out.mean_us   = mean;
            out.median_us = median;
            out.min_us    = sorted.front();
            out.cv        = mean > 0.0 ? std::sqrt(var) / mean : 0.0;
            out.rel_iqr   = median > 0.0 ? iqr / median : 0.0;
            out.samples   = static_cast<int32_t>(n);
        }

        // Floor and ceiling for the adaptive measure loop.
        struct measure_bounds
        {
            int64_t min_iters; // floor: minimum total timed iterations
            double  min_us; // floor: minimum total measured time
            double  max_us; // ceiling: maximum total measured time (0 = none)
        };

        // Whether the measure loop should stop after the latest sample, updating `converged`
        // and `stable`. Past the floor, the run ends on convergence (mean precise enough) or
        // on the robust-dispersion plateau (the fallback), else at the ceiling. noise_threshold
        // <= 0 disables both checks (so the run goes to the ceiling), and both need enough
        // samples for their estimates to be trustworthy. `iqr_history` accumulates the throttled
        // rel_iqr trajectory the plateau check reads.
        inline bool
            reached_target(const TimingConfig&   cfg,
                           std::vector<double>&  samples, // reordered in place by the rel_iqr check
                           const running_stats&  stats,
                           std::vector<double>&  iqr_history,
                           int64_t               total_iters,
                           double                total_us,
                           const measure_bounds& bounds,
                           bool&                 converged,
                           bool&                 stable)
        {
            const bool floor_met = total_iters >= bounds.min_iters && total_us >= bounds.min_us;
            const int  n         = static_cast<int>(samples.size());
            if(floor_met && cfg.noise_threshold > 0.0f && n >= min_samples_for_convergence)
            {
                converged              = stats.relative_std_error() < cfg.noise_threshold;
                const bool fallback_on = cfg.stability_threshold > 0.0f
                                         && cfg.stability_interval >= 1
                                         && cfg.stability_window >= 2;
                if(!converged && fallback_on && n % cfg.stability_interval == 0)
                {
                    iqr_history.push_back(cumulative_rel_iqr(samples));
                    stable = noise_plateaued(
                        iqr_history, cfg.stability_window, cfg.stability_threshold);
                }
            }
            const bool ceiling = (bounds.max_us > 0.0 && total_us >= bounds.max_us)
                                 || (cfg.max_iters > 0 && total_iters >= cfg.max_iters);
            // Floors take precedence: the ceiling can only fire after the floor is satisfied.
            return floor_met && (converged || stable || ceiling);
        }

    } // namespace detail

    // Run `launch(i)` and fill `out` with timing statistics. `launch` performs exactly one
    // enqueue for a monotonically increasing global index `i` (the callee handles any
    // `i % block_count` rotation and per-iteration icache flush). Fixed-count mode times one
    // batch of cfg.iters; adaptive mode self-sizes the batch and collects samples until the
    // mean converges, the robust spread plateaus, or a ceiling is hit.
    template <typename Launch>
    inline void run_measurement(Launch&&                     launch,
                                const TimingConfig&          cfg,
                                hipEvent_t                   event_start,
                                hipEvent_t                   event_stop,
                                hipStream_t                  stream,
                                TimingResult&                out,
                                const std::function<bool()>& should_abort = {})
    {
        const detail::event_context events{event_start, event_stop, stream};
        int64_t                     global_index = 0;

        // ---- Fixed-count fast path: a single sample of `iters` enqueues ----
        // cfg.iters is used here and ONLY here; the adaptive path below is self-sizing.
        if(!cfg.adaptive)
        {
            const int32_t iters    = cfg.iters > 0 ? cfg.iters : 1;
            double        batch_us = 0.0;
            detail::time_batch(launch, iters, global_index, cfg.use_gpu_timer, events, batch_us);
            const double per_iter = batch_us / iters;
            out.mean_us = out.median_us = out.min_us = per_iter;
            out.cv = out.rel_iqr = 0.0;
            out.batch            = iters;
            out.samples          = 1;
            out.hot_iters        = iters;
            out.adaptive         = false;
            out.noise_active     = false;
            out.converged        = false;
            return;
        }

        // Reject invalid configs before touching the GPU.
        if(const auto err = validate_adaptive_config(cfg); !err.empty())
        {
            hipblaslt_cerr << "hipblaslt adaptive timing: invalid config: " << err << "\n";
            return;
        }

        // ---- Warmup: one cold probe seeds the batch size, then warm up in self-sized
        //      chunks until warmup_time, refining the per-iteration estimate. ----
        const int64_t cap          = detail::max_batch_size(cfg);
        double        per_iter_est = 0.0;
        {
            double probe_us = 0.0;
            detail::time_batch(launch,
                               1,
                               global_index,
                               cfg.use_gpu_timer,
                               events,
                               probe_us); // cold probe, discarded
            per_iter_est             = probe_us;
            const double warm_min_us = static_cast<double>(cfg.warmup_time) * 1000.0;
            double       warm_us     = 0.0; // exclude the cold probe from the warmup budget
            int32_t      chunk       = detail::batch_size(cfg, per_iter_est, cap);
            while(warm_us < warm_min_us && !detail::aborted(should_abort))
            {
                double chunk_us = 0.0;
                detail::time_batch(
                    launch, chunk, global_index, cfg.use_gpu_timer, events, chunk_us);
                warm_us += chunk_us;
                // Refine the estimate from each chunk. A chunk below timer resolution
                // (chunk_us == 0) can't advance warm_us, so grow the batch instead; give up if
                // even the max batch still reads as zero.
                if(chunk_us > 0.0)
                {
                    per_iter_est = chunk_us / chunk;
                    chunk        = detail::batch_size(cfg, per_iter_est, cap);
                }
                else if(chunk < cap)
                    chunk = static_cast<int32_t>(std::min<int64_t>(cap, int64_t{chunk} * 2));
                else
                    break;
            }
        }
        if(detail::aborted(should_abort))
            return; // out left default-zeroed; caller (e.g. gtest) already failed

        const int32_t batch = detail::batch_size(cfg, per_iter_est, cap);

        // ---- Measure: run at least the floor (min_iters AND measure_time), then until the
        //      mean converges or the ceiling (max_measure_time / max_iters) is hit. ----
        const detail::measure_bounds bounds{std::max<int64_t>(1, cfg.min_iters),
                                            static_cast<double>(cfg.measure_time) * 1000.0,
                                            static_cast<double>(cfg.max_measure_time) * 1000.0};

        std::vector<double>   samples;
        std::vector<double>   iqr_history; // throttled rel_iqr trajectory for the plateau check
        detail::running_stats stats; // incremental mean/variance for the convergence test
        double                total_us    = 0.0;
        int64_t               total_iters = 0;
        bool                  converged   = false;
        bool                  stable      = false;

        global_index = 0; // restart rotation for the timed phase
        while(!detail::aborted(should_abort))
        {
            double batch_us = 0.0;
            detail::time_batch(launch, batch, global_index, cfg.use_gpu_timer, events, batch_us);
            const double per_iter = batch_us / batch;
            samples.push_back(per_iter);
            stats.add(per_iter);
            total_us += batch_us;
            total_iters += batch;
            if(detail::reached_target(cfg,
                                      samples,
                                      stats,
                                      iqr_history,
                                      total_iters,
                                      total_us,
                                      bounds,
                                      converged,
                                      stable))
                break;
        }

        if(samples.empty())
            return; // aborted before any sample completed; out left default-zeroed

        detail::fill_distribution(samples, out);
        out.batch        = batch;
        out.hot_iters    = total_iters;
        out.adaptive     = true;
        out.noise_active = cfg.noise_threshold > 0.0f;
        out.converged    = converged;
        out.stable       = stable;
    }
} // namespace hipblaslt_bench
