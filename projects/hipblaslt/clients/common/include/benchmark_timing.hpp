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
            return (std::sqrt(var) / mean) / std::sqrt(double(n));
        }

        inline bool any_adaptive(const TimingConfig& cfg)
        {
            return cfg.warmup_time > 0.0f || cfg.sample_time > 0.0f || cfg.measure_time > 0.0f
                   || cfg.max_measure_time > 0.0f || cfg.min_iters > 1 || cfg.max_iters > 0
                   || cfg.noise_threshold > 0.0f;
        }
    } // namespace detail

    // Run `launch(i)` adaptively and fill `out` with the resulting statistics.
    //
    // `launch` performs exactly one enqueue for a monotonically increasing global
    // index `i` (the callee is responsible for any `i % block_count` rotation and
    // for any per-iteration icache flush). The index restarts at 0 for the timed
    // phase so rotating buffers cycle identically regardless of sample count.
    template <typename Launch>
    inline void run_measurement(Launch&&            launch,
                                const TimingConfig&          cfg,
                                hipEvent_t                   event_start,
                                hipEvent_t                   event_stop,
                                hipStream_t                  stream,
                                TimingResult&                out,
                                const std::function<bool()>& should_abort = {})
    {
        const bool    use_gpu_timer = cfg.use_gpu_timer;
        const int32_t floor_iters   = cfg.iters > 0 ? cfg.iters : 1;
        // Caller signal to abort the sample loop (e.g. a gtest fatal failure in `launch`).
        auto aborted = [&]() { return should_abort && should_abort(); };

        int64_t global_index = 0;

        // Time one batch of `batch` enqueues; returns elapsed microseconds via `elapsed_us`.
        // A void lambda, so CHECK_HIP_ERROR (which may expand to `return;`) is valid here.
        auto time_batch = [&](int32_t batch, double& elapsed_us) {
            double cpu_start = 0.0;
            if(use_gpu_timer)
                CHECK_HIP_ERROR(hipEventRecord(event_start, stream));
            else
                cpu_start = get_time_us_sync(stream);

            for(int32_t k = 0; k < batch; ++k)
                launch(global_index++);

            if(use_gpu_timer)
            {
                CHECK_HIP_ERROR(hipEventRecord(event_stop, stream));
                CHECK_HIP_ERROR(hipEventSynchronize(event_stop));
                float ms = 0.0f;
                CHECK_HIP_ERROR(hipEventElapsedTime(&ms, event_start, event_stop));
                elapsed_us = double(ms) * 1000.0;
            }
            else
            {
                elapsed_us = get_time_us_sync(stream) - cpu_start;
            }
        };

        // ---- Fixed-count fast path: a single sample of `iters` enqueues ----
        if(!detail::any_adaptive(cfg))
        {
            double batch_us = 0.0;
            time_batch(floor_iters, batch_us);
            const double per_iter = batch_us / floor_iters;
            out.median_us    = out.min_us = out.mean_us = per_iter;
            out.cv           = 0.0;
            out.batch        = floor_iters;
            out.samples      = 1;
            out.hot_iters    = floor_iters;
            out.adaptive     = false;
            out.noise_active = false;
            out.converged    = false;
            return;
        }

        // Batch size for a target per-iteration time: enough enqueues to span
        // ~sample_time, clamped to [1, cap]. The adaptive path is self-sizing and
        // does not use cfg.iters.
        int64_t cap = std::numeric_limits<int32_t>::max();
        if(cfg.max_iters > 0)
            cap = std::max<int64_t>(1, cfg.max_iters / 2); // leave room for >=2 samples
        auto size_batch = [&](double per_iter_us) -> int32_t {
            if(cfg.sample_time <= 0.0f || per_iter_us <= 0.0)
                return 1;
            const double want = std::ceil(double(cfg.sample_time) * 1000.0 / per_iter_us);
            // cap <= INT32_MAX, so the clamped value always fits in int32_t.
            return int32_t(std::clamp(int64_t(want), int64_t(1), cap));
        };

        // ---- Warmup: a 1-enqueue probe seeds the chunk size, then warm up in
        // self-sized chunks until warmup_time, refining the per-iteration estimate. ----
        double per_iter_est = 0.0;
        {
            double e = 0.0;
            time_batch(1, e); // cold probe: seeds the first chunk size only
            per_iter_est             = e;
            const double warm_min_us = double(cfg.warmup_time) * 1000.0;
            double       warm_us     = 0.0; // don't count the cold probe toward warmup
            while(warm_us < warm_min_us && !aborted())
            {
                const int32_t chunk = size_batch(per_iter_est);
                double        ce    = 0.0;
                time_batch(chunk, ce);
                warm_us += ce;
                per_iter_est = ce / chunk;
            }
        }
        if(aborted())
            return; // out left default-zeroed; caller (e.g. gtest) already failed

        const int32_t batch = size_batch(per_iter_est);

        // ---- Measure: run at least the floor, then until convergence or the ceiling ----
        // Precondition: a ceiling must be set (max_measure_time or max_iters > 0); the
        // caller validates this. Without it, a run that never converges has no bound.
        const double  measure_min_us = double(cfg.measure_time) * 1000.0;
        const double  measure_max_us = double(cfg.max_measure_time) * 1000.0;
        const int64_t min_iters      = std::max(1, cfg.min_iters);

        std::vector<double> per_iter_samples;
        double              total_us    = 0.0;
        int64_t             total_iters = 0;
        bool                converged   = false;

        global_index = 0; // restart rotation for the timed phase
        while(true)
        {
            if(aborted())
                break;
            double batch_us = 0.0;
            time_batch(batch, batch_us);
            per_iter_samples.push_back(batch_us / batch);
            total_us += batch_us;
            total_iters += batch;

            const bool floor_met = total_iters >= min_iters && total_us >= measure_min_us;

            // Past the floor, stop once the mean has converged; otherwise keep going to
            // the ceiling. With noise_threshold <= 0, convergence never triggers, so the
            // run goes to the ceiling.
            if(floor_met && cfg.noise_threshold > 0.0f)
                converged = detail::relative_std_error(per_iter_samples) < cfg.noise_threshold;

            const bool ceiling = (measure_max_us > 0.0 && total_us >= measure_max_us)
                                 || (cfg.max_iters > 0 && total_iters >= cfg.max_iters);

            if(floor_met && converged)
                break;
            if(ceiling)
                break;
        }

        // ---- Statistics ----
        if(per_iter_samples.empty())
            return; // aborted before any sample completed; out left default-zeroed

        std::vector<double> sorted(per_iter_samples);
        std::sort(sorted.begin(), sorted.end());
        const size_t n    = sorted.size();
        const double mean = detail::mean_of(per_iter_samples);
        const double var  = detail::sum_sq_dev(per_iter_samples, mean) / n;

        out.min_us    = sorted.front();
        out.median_us = (n % 2) ? sorted[n / 2] : 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
        out.mean_us   = mean;
        out.cv        = mean > 0.0 ? std::sqrt(var) / mean : 0.0;
        out.batch        = batch;
        out.samples      = int32_t(n);
        out.hot_iters    = total_iters;
        out.adaptive     = true;
        out.noise_active = cfg.noise_threshold > 0.0f;
        out.converged    = converged;
    }
} // namespace hipblaslt_bench
