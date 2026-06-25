// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>

// Dependency-free config/result types (and their validation) for the adaptive
// timing routine in benchmark_timing.hpp, separated so argument_model.hpp can
// use TimingResult without including the timing implementation's HIP headers.

namespace hipblaslt_bench
{
    // All time budgets are in milliseconds.
    struct TimingConfig
    {
        bool    adaptive = false; // false => fixed-count fast path (uses iters); true => adaptive
        int32_t iters    = 10; // enqueues in the fixed-count sample; unused by the adaptive path
        float   warmup_time      = 0.0f; // warm up until this wall-time is reached
        float   sample_time      = 0.0f; // target span per sample; 0 => one enqueue per sample
        float   measure_time     = 0.0f; // minimum total measure time (floor)
        float   max_measure_time = 0.0f; // measure ceiling; 0 => unbounded
        int32_t min_iters = 0; // floor on total timed iterations (0 in the fixed-count fast path)
        int32_t max_iters = 0; // ceiling on total timed iterations; 0 => unbounded
        float   noise_threshold = 0.0f; // rel. std error convergence target; 0 => disabled
        // Noise-plateau fallback: past the floor, end the run if convergence cannot be
        // reached but the robust spread (rel_iqr) has settled. 0 threshold disables it.
        float   stability_threshold = 0.0f; // max rel. spread of recent rel_iqr readings to stop
        int32_t stability_window    = 0; // rel_iqr readings tested for the plateau (>= 2)
        int32_t stability_interval  = 0; // record a rel_iqr reading every N samples (>= 1)
        bool    use_gpu_timer       = false; // hipEvent timing vs CPU wall clock
    };

    // Validate the semantic constraints of an adaptive TimingConfig. Returns an empty
    // string on success, or a human-readable error description on failure. All entry
    // points (CLI, YAML, run_measurement) call this as their single source of truth.
    inline std::string validate_adaptive_config(const TimingConfig& cfg)
    {
        // Reject negative inputs up front: a negative time floor (measure_time) would be
        // instantly satisfied and a negative warmup_time would skip warmup, both yielding
        // subtly wrong measurements; negative thresholds would silently read as "disabled".
        if(cfg.warmup_time < 0.0f)
            return "warmup_time (" + std::to_string(cfg.warmup_time) + " ms) must be >= 0";
        if(cfg.sample_time < 0.0f)
            return "sample_time (" + std::to_string(cfg.sample_time) + " ms) must be >= 0";
        if(cfg.measure_time < 0.0f)
            return "measure_time (" + std::to_string(cfg.measure_time) + " ms) must be >= 0";
        if(cfg.max_measure_time < 0.0f)
            return "max_measure_time (" + std::to_string(cfg.max_measure_time)
                   + " ms) must be >= 0";
        if(cfg.min_iters < 0)
            return "min_iters (" + std::to_string(cfg.min_iters) + ") must be >= 0";
        if(cfg.max_iters < 0)
            return "max_iters (" + std::to_string(cfg.max_iters) + ") must be >= 0";
        if(cfg.noise_threshold < 0.0f)
            return "noise_threshold (" + std::to_string(cfg.noise_threshold) + ") must be >= 0";
        if(cfg.stability_threshold < 0.0f)
            return "stability_threshold (" + std::to_string(cfg.stability_threshold)
                   + ") must be >= 0";

        if(cfg.max_iters > 0 && cfg.min_iters > cfg.max_iters)
            return "min_iters (" + std::to_string(cfg.min_iters) + ") > max_iters ("
                   + std::to_string(cfg.max_iters) + ")";
        if(cfg.max_measure_time > 0.0f && cfg.measure_time > cfg.max_measure_time)
            return "measure_time (" + std::to_string(cfg.measure_time)
                   + " ms) > max_measure_time (" + std::to_string(cfg.max_measure_time) + " ms)";
        if(cfg.max_measure_time <= 0.0f && cfg.max_iters <= 0)
            return "adaptive timing requires a ceiling: set max_measure_time or max_iters > 0";
        if(cfg.stability_threshold > 0.0f
           && (cfg.stability_window < 2 || cfg.stability_interval < 1))
            return "stability_threshold > 0 requires stability_window >= 2 and "
                   "stability_interval >= 1";
        return {};
    }

    // Default values for the --adaptive preset, used as the CLI/YAML defaults so they
    // appear directly in --help. Keep hipblaslt_common.yaml's Defaults block in sync.
    namespace adaptive_defaults
    {
        constexpr float   warmup_time      = 50.0f; // ms warmup budget
        constexpr float   sample_time      = 1.0f; // ms span per timed sample
        constexpr float   measure_time     = 500.0f; // ms measurement floor
        constexpr float   max_measure_time = 2000.0f; // ms ceiling for convergence
        constexpr int32_t min_iters        = 10; // floor on total timed iterations
        constexpr int32_t max_iters        = 0; // unbounded
        constexpr float   noise_threshold  = 0.01f; // 1% relative standard error
        // Noise-plateau fallback (see TimingConfig). window * interval = 512-sample look-back,
        // so the fallback can only fire ~512 samples past the measure_time floor: kernels that
        // converge first report "converged", only those that cannot report "stable".
        constexpr float   stability_threshold = 0.05f; // 5% rel. spread over the window
        constexpr int32_t stability_window    = 32; // readings tested (512-sample look-back)
        constexpr int32_t stability_interval  = 16; // a reading every 16 samples
    }

    // Per-iteration statistics, in microseconds.
    struct TimingResult
    {
        double  median_us    = 0.0;
        double  min_us       = 0.0;
        double  mean_us      = 0.0;
        double  cv           = 0.0; // coefficient of variation across samples (stddev/mean, n-1)
        double  rel_iqr      = 0.0; // robust dispersion: interquartile range / median
        int32_t batch        = 0; // enqueues per sample (B)
        int32_t samples      = 0; // number of samples collected (K)
        int64_t hot_iters    = 0; // total timed enqueues (B*K)
        bool    adaptive     = false; // the adaptive path ran (vs the fixed-count fast path)
        bool    noise_active = false; // convergence checking was enabled (noise_threshold > 0)
        bool    converged    = false; // precision target met (meaningful only when noise_active)
        bool    stable       = false; // robust-dispersion (rel_iqr) plateau reached: the
        // distribution is characterized though the precision target was not met (fallback exit)
    };
} // namespace hipblaslt_bench
