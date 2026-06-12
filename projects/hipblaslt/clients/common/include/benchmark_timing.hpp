/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

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
// measure_time); past the floor the run continues until the mean's relative standard
// error drops below noise_threshold (converged) or the max_measure_time / max_iters
// ceiling is reached. noise_threshold <= 0 disables convergence, so the run goes to
// the ceiling. The mean is the headline statistic; median, min and cv are also
// returned. With no adaptive knob set, a single fixed-count sample of `iters` is taken.
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

        // Relative standard error of the mean: (sample stddev / mean) / sqrt(n).
        inline double relative_std_error(const std::vector<double>& v)
        {
            const size_t n = v.size();
            if(n < 2)
                return std::numeric_limits<double>::infinity();
            const double mean = mean_of(v);
            if(mean <= 0.0)
                return 0.0;
            const double var = sum_sq_dev(v, mean) / (n - 1); // unbiased sample variance
            return (std::sqrt(var) / mean) / std::sqrt(static_cast<double>(n));
        }

        // Minimum samples before the convergence test is trusted; a stddev from fewer
        // is unreliable. Matches nvbench's min-samples default.
        inline constexpr int min_samples_for_convergence = 10;

        // Linear-interpolated quantile of a sorted, non-empty vector (q in [0, 1]).
        inline double quantile(const std::vector<double>& sorted, double q)
        {
            const double pos = q * static_cast<double>(sorted.size() - 1);
            const size_t lo  = static_cast<size_t>(std::floor(pos));
            const size_t hi  = static_cast<size_t>(std::ceil(pos));
            return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - static_cast<double>(lo));
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

    } // namespace detail

    // Run `launch(i)` and fill `out` with timing statistics. `launch` performs exactly one
    // enqueue for a monotonically increasing global index `i` (the callee handles any
    // `i % block_count` rotation and per-iteration icache flush). Fixed-count mode times one
    // batch of cfg.iters; adaptive mode self-sizes the batch and collects samples until the
    // mean converges or a ceiling is hit. Precondition (adaptive): a ceiling is set
    // (max_measure_time or max_iters > 0) -- the caller validates this.
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

        // ---- Warmup: one cold probe seeds the batch size, then warm up in self-sized
        //      chunks until warmup_time, refining the per-iteration estimate. ----
        const int64_t cap          = detail::max_batch_size(cfg);
        double        per_iter_est = 0.0;
        {
            double probe_us = 0.0;
            detail::time_batch(launch, 1, global_index, cfg.use_gpu_timer, events, probe_us); // cold probe, discarded
            per_iter_est             = probe_us;
            const double warm_min_us = static_cast<double>(cfg.warmup_time) * 1000.0;
            double       warm_us     = 0.0; // exclude the cold probe from the warmup budget
            while(warm_us < warm_min_us && !detail::aborted(should_abort))
            {
                const int32_t chunk    = detail::batch_size(cfg, per_iter_est, cap);
                double        chunk_us = 0.0;
                detail::time_batch(launch, chunk, global_index, cfg.use_gpu_timer, events, chunk_us);
                warm_us += chunk_us;
                per_iter_est = chunk_us / chunk;
            }
        }
        if(detail::aborted(should_abort))
            return; // out left default-zeroed; caller (e.g. gtest) already failed

        const int32_t batch = detail::batch_size(cfg, per_iter_est, cap);

        // ---- Measure: run at least the floor (min_iters AND measure_time), then until the
        //      mean converges or the ceiling (max_measure_time / max_iters) is hit. ----
        const double  measure_min_us = static_cast<double>(cfg.measure_time) * 1000.0;
        const double  measure_max_us = static_cast<double>(cfg.max_measure_time) * 1000.0;
        const int64_t min_iters      = std::max(1, cfg.min_iters);

        std::vector<double> samples;
        double              total_us    = 0.0;
        int64_t             total_iters = 0;
        bool                converged   = false;

        global_index = 0; // restart rotation for the timed phase
        while(!detail::aborted(should_abort))
        {
            double batch_us = 0.0;
            detail::time_batch(launch, batch, global_index, cfg.use_gpu_timer, events, batch_us);
            samples.push_back(batch_us / batch);
            total_us += batch_us;
            total_iters += batch;

            const bool floor_met = total_iters >= min_iters && total_us >= measure_min_us;

            // Need enough samples for the stddev (hence the convergence test) to be
            // trustworthy; noise_threshold <= 0 disables convergence (runs to the ceiling).
            if(floor_met && cfg.noise_threshold > 0.0f
               && static_cast<int>(samples.size()) >= detail::min_samples_for_convergence)
                converged = detail::relative_std_error(samples) < cfg.noise_threshold;

            const bool ceiling = (measure_max_us > 0.0 && total_us >= measure_max_us)
                                 || (cfg.max_iters > 0 && total_iters >= cfg.max_iters);

            if((floor_met && converged) || ceiling)
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
    }
} // namespace hipblaslt_bench
