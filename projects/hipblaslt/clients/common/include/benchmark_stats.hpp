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

#include <cstdint>

// Dependency-free config/result types for the adaptive timing routine in
// benchmark_timing.hpp, separated so argument_model.hpp can use TimingResult
// without including the timing implementation's HIP/test-harness headers.

namespace hipblaslt_bench
{
    // All time budgets are in milliseconds.
    struct TimingConfig
    {
        int32_t iters = 10; // enqueues in the fixed-count sample; unused by the adaptive path
        float   warmup_time = 0.0f; // warm up until this wall-time is reached
        float   sample_time = 0.0f; // target span per sample; 0 => one enqueue per sample
        float   measure_time = 0.0f; // minimum total measure time (floor)
        float   max_measure_time = 0.0f; // measure ceiling; 0 => unbounded
        int32_t min_iters = 0; // floor on total timed iterations (0 in the fixed-count fast path)
        int32_t max_iters = 0; // ceiling on total timed iterations; 0 => unbounded
        float   noise_threshold = 0.0f; // rel. std error convergence target; 0 => disabled
        bool    use_gpu_timer = false; // hipEvent timing vs CPU wall clock
    };

    // Default values for the --adaptive preset, used as the CLI/YAML defaults so they
    // appear directly in --help. Warmup/measure budgets are in the range of the
    // benchmarking tool's cold/hot targets; the ceiling gives convergence headroom.
    // Keep hipblaslt_common.yaml's Defaults block in sync with these.
    namespace adaptive_defaults
    {
        constexpr float   warmup_time      = 50.0f; // ms warmup budget
        constexpr float   sample_time      = 1.0f; // ms span per timed sample
        constexpr float   measure_time     = 200.0f; // ms measurement floor
        constexpr float   max_measure_time = 1000.0f; // ms ceiling for convergence
        constexpr int32_t min_iters        = 10; // floor on total timed iterations
        constexpr int32_t max_iters        = 0; // unbounded
        constexpr float   noise_threshold  = 0.01f; // 1% relative standard error
    }

    // Per-iteration statistics, in microseconds.
    struct TimingResult
    {
        double  median_us = 0.0;
        double  min_us    = 0.0;
        double  mean_us   = 0.0;
        double  cv        = 0.0; // coefficient of variation across samples (stddev/mean, n-1)
        double  rel_iqr   = 0.0; // robust dispersion: interquartile range / median
        int32_t batch        = 0; // enqueues per sample (B)
        int32_t samples      = 0; // number of samples collected (K)
        int64_t hot_iters    = 0; // total timed enqueues (B*K)
        bool    adaptive     = false; // the adaptive path ran (vs the fixed-count fast path)
        bool    noise_active = false; // convergence checking was enabled (noise_threshold > 0)
        bool    converged    = false; // noise target met (meaningful only when noise_active)
    };
} // namespace hipblaslt_bench
