/*******************************************************************************
 *
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 *******************************************************************************/

// Host-only unit tests for benchmark_stats.hpp: no GPU/HIP calls, pure logic.

#include "benchmark_stats.hpp"

#include <gtest/gtest.h>

#include <limits>

using hipblaslt_bench::compute_rotating_buffer_plan;
using hipblaslt_bench::rate_per_second;
using hipblaslt_bench::TimingConfig;
using hipblaslt_bench::validate_adaptive_config;

namespace
{
    TimingConfig valid_adaptive_config()
    {
        TimingConfig cfg;
        cfg.adaptive            = true;
        cfg.warmup_time         = hipblaslt_bench::adaptive_defaults::warmup_time;
        cfg.sample_time         = hipblaslt_bench::adaptive_defaults::sample_time;
        cfg.measure_time        = hipblaslt_bench::adaptive_defaults::measure_time;
        cfg.max_measure_time    = hipblaslt_bench::adaptive_defaults::max_measure_time;
        cfg.min_iters           = hipblaslt_bench::adaptive_defaults::min_iters;
        cfg.max_iters           = hipblaslt_bench::adaptive_defaults::max_iters;
        cfg.noise_threshold     = hipblaslt_bench::adaptive_defaults::noise_threshold;
        cfg.stability_threshold = hipblaslt_bench::adaptive_defaults::stability_threshold;
        cfg.stability_window    = hipblaslt_bench::adaptive_defaults::stability_window;
        cfg.stability_interval  = hipblaslt_bench::adaptive_defaults::stability_interval;
        return cfg;
    }
} // namespace

// ---------------------------------------------------------------------------
// validate_adaptive_config
// ---------------------------------------------------------------------------

TEST(ValidateAdaptiveConfig, DefaultsAreValid)
{
    EXPECT_TRUE(validate_adaptive_config(valid_adaptive_config()).empty());
}

TEST(ValidateAdaptiveConfig, RejectsNegativeFields)
{
    auto cfg        = valid_adaptive_config();
    cfg.warmup_time = -1.0f;
    EXPECT_FALSE(validate_adaptive_config(cfg).empty());

    cfg           = valid_adaptive_config();
    cfg.min_iters = -1;
    EXPECT_FALSE(validate_adaptive_config(cfg).empty());
}

TEST(ValidateAdaptiveConfig, RejectsMinItersGreaterThanMaxIters)
{
    auto cfg      = valid_adaptive_config();
    cfg.min_iters = 20;
    cfg.max_iters = 10;
    EXPECT_FALSE(validate_adaptive_config(cfg).empty());
}

TEST(ValidateAdaptiveConfig, AllowsMinGreaterThanMaxWhenMaxIsUnbounded)
{
    auto cfg      = valid_adaptive_config();
    cfg.min_iters = 20;
    cfg.max_iters = 0; // unbounded: min_iters > max_iters check does not apply
    EXPECT_TRUE(validate_adaptive_config(cfg).empty());
}

TEST(ValidateAdaptiveConfig, RejectsMeasureTimeGreaterThanMaxMeasureTime)
{
    auto cfg             = valid_adaptive_config();
    cfg.measure_time     = 3000.0f;
    cfg.max_measure_time = 2000.0f;
    EXPECT_FALSE(validate_adaptive_config(cfg).empty());
}

TEST(ValidateAdaptiveConfig, RequiresAtLeastOneCeiling)
{
    auto cfg             = valid_adaptive_config();
    cfg.max_measure_time = 0.0f;
    cfg.max_iters        = 0;
    EXPECT_FALSE(validate_adaptive_config(cfg).empty());
}

TEST(ValidateAdaptiveConfig, UnboundedMeasureTimeAloneIsValid)
{
    auto cfg             = valid_adaptive_config();
    cfg.max_measure_time = 0.0f;
    cfg.max_iters        = 100;
    EXPECT_TRUE(validate_adaptive_config(cfg).empty());
}

TEST(ValidateAdaptiveConfig, StabilityThresholdRequiresWindowAndInterval)
{
    auto cfg                = valid_adaptive_config();
    cfg.stability_threshold = 0.05f;
    cfg.stability_window    = 1; // < 2
    EXPECT_FALSE(validate_adaptive_config(cfg).empty());

    cfg                     = valid_adaptive_config();
    cfg.stability_threshold = 0.05f;
    cfg.stability_interval  = 0; // < 1
    EXPECT_FALSE(validate_adaptive_config(cfg).empty());
}

TEST(ValidateAdaptiveConfig, DisabledStabilityFallbackSkipsWindowInterval)
{
    auto cfg                = valid_adaptive_config();
    cfg.stability_threshold = 0.0f;
    cfg.stability_window    = 0;
    cfg.stability_interval  = 0;
    EXPECT_TRUE(validate_adaptive_config(cfg).empty());
}

// ---------------------------------------------------------------------------
// compute_rotating_buffer_plan
// ---------------------------------------------------------------------------

TEST(RotatingBufferPlan, NonAdaptiveCapsAtIterCount)
{
    // 512 MiB budget / 11 MiB per block ~ 44 blocks worth of memory, but only
    // max(cold_iters=2, iters=10) = 10 iterations will ever run.
    auto plan = compute_rotating_buffer_plan(
        /*adaptive=*/false,
        /*adaptive_max_iters=*/0,
        /*cold_iters=*/2,
        /*iters=*/10,
        /*rotating_bytes=*/512LL * 1024 * 1024,
        /*total_size_needed_bytes=*/11LL * 1024 * 1024);
    EXPECT_EQ(plan.block_count, 10);
    EXPECT_TRUE(plan.capped);
    EXPECT_EQ(plan.iter_cap, 10);
}

TEST(RotatingBufferPlan, NonAdaptiveNotCappedWhenMemoryIsTighter)
{
    auto plan = compute_rotating_buffer_plan(
        /*adaptive=*/false,
        /*adaptive_max_iters=*/0,
        /*cold_iters=*/1000,
        /*iters=*/1000,
        /*rotating_bytes=*/512LL * 1024 * 1024,
        /*total_size_needed_bytes=*/11LL * 1024 * 1024);
    EXPECT_EQ(plan.block_count, 47);
    EXPECT_FALSE(plan.capped);
}

TEST(RotatingBufferPlan, AdaptiveIgnoresColdItersAndIters)
{
    // cold_iters/iters are stuck at CLI defaults (2/10) under --adaptive; the plan must
    // not cap block_count at 10 the way the non-adaptive path would.
    auto plan = compute_rotating_buffer_plan(
        /*adaptive=*/true,
        /*adaptive_max_iters=*/0,
        /*cold_iters=*/2,
        /*iters=*/10,
        /*rotating_bytes=*/512LL * 1024 * 1024,
        /*total_size_needed_bytes=*/11LL * 1024 * 1024);
    EXPECT_EQ(plan.block_count, 47);
    EXPECT_FALSE(plan.capped);
}

TEST(RotatingBufferPlan, AdaptiveBoundedAndBinding)
{
    auto plan = compute_rotating_buffer_plan(
        /*adaptive=*/true,
        /*adaptive_max_iters=*/20,
        /*cold_iters=*/2,
        /*iters=*/10,
        /*rotating_bytes=*/512LL * 1024 * 1024,
        /*total_size_needed_bytes=*/11LL * 1024 * 1024);
    EXPECT_EQ(plan.block_count, 20);
    EXPECT_TRUE(plan.capped);
    EXPECT_EQ(plan.iter_cap, 20);
}

TEST(RotatingBufferPlan, AdaptiveBoundedButNotBinding)
{
    auto plan = compute_rotating_buffer_plan(
        /*adaptive=*/true,
        /*adaptive_max_iters=*/1000,
        /*cold_iters=*/2,
        /*iters=*/10,
        /*rotating_bytes=*/512LL * 1024 * 1024,
        /*total_size_needed_bytes=*/11LL * 1024 * 1024);
    EXPECT_EQ(plan.block_count, 47);
    EXPECT_FALSE(plan.capped);
}

TEST(RotatingBufferPlan, BlockCountNeverLessThanOne)
{
    // Memory budget smaller than a single block's footprint still yields at least 1 block.
    auto plan = compute_rotating_buffer_plan(
        /*adaptive=*/true,
        /*adaptive_max_iters=*/0,
        /*cold_iters=*/2,
        /*iters=*/10,
        /*rotating_bytes=*/1,
        /*total_size_needed_bytes=*/11LL * 1024 * 1024);
    EXPECT_EQ(plan.block_count, 1);
}

TEST(RotatingBufferPlan, ZeroSizeNeededYieldsSingleBlock)
{
    // A degenerate 0-byte-per-iteration problem has nothing to rotate: a single block is
    // correct regardless of whether an iteration cap applies.
    auto capped_plan = compute_rotating_buffer_plan(
        /*adaptive=*/false,
        /*adaptive_max_iters=*/0,
        /*cold_iters=*/2,
        /*iters=*/10,
        /*rotating_bytes=*/512LL * 1024 * 1024,
        /*total_size_needed_bytes=*/0);
    EXPECT_EQ(capped_plan.block_count, 1);
    EXPECT_FALSE(capped_plan.capped);

    auto unbounded_plan = compute_rotating_buffer_plan(
        /*adaptive=*/true,
        /*adaptive_max_iters=*/0,
        /*cold_iters=*/2,
        /*iters=*/10,
        /*rotating_bytes=*/512LL * 1024 * 1024,
        /*total_size_needed_bytes=*/0);
    EXPECT_EQ(unbounded_plan.block_count, 1);
    EXPECT_FALSE(unbounded_plan.capped);
}

TEST(RotatingBufferPlan, OverflowingMemoryBudgetFallsBackToIterCap)
{
    // rotating_bytes far larger than total_size_needed_bytes would overflow int32_t if the
    // quotient were cast directly; with an iteration cap in play, that cap alone decides
    // (it's already a valid int32_t, so this can never overflow).
    auto plan = compute_rotating_buffer_plan(
        /*adaptive=*/false,
        /*adaptive_max_iters=*/0,
        /*cold_iters=*/2,
        /*iters=*/10,
        /*rotating_bytes=*/static_cast<int64_t>(std::numeric_limits<int32_t>::max()) * 4,
        /*total_size_needed_bytes=*/1);
    EXPECT_EQ(plan.block_count, 10);
    EXPECT_TRUE(plan.capped);
}

TEST(RotatingBufferPlan, OverflowingMemoryBudgetUnboundedFallsBackToOneBlock)
{
    // Same overflow, but with no iteration cap either: no usable value on either side, so
    // fall back to a single block instead of casting an out-of-range double (UB).
    auto plan = compute_rotating_buffer_plan(
        /*adaptive=*/true,
        /*adaptive_max_iters=*/0,
        /*cold_iters=*/2,
        /*iters=*/10,
        /*rotating_bytes=*/static_cast<int64_t>(std::numeric_limits<int32_t>::max()) * 4,
        /*total_size_needed_bytes=*/1);
    EXPECT_EQ(plan.block_count, 1);
    EXPECT_FALSE(plan.capped);
}

// ---------------------------------------------------------------------------
// rate_per_second
// ---------------------------------------------------------------------------

TEST(RatePerSecond, ComputesRateForPositiveTime)
{
    // 2 units over 500 us == 4000 units/sec.
    EXPECT_DOUBLE_EQ(rate_per_second(2.0, 500.0, -1.0), 4000.0);
}

TEST(RatePerSecond, ReturnsInvalidValueForZeroTime)
{
    EXPECT_EQ(rate_per_second(2.0, 0.0, -1.0), -1.0);
}

TEST(RatePerSecond, ReturnsInvalidValueForNegativeTime)
{
    // A flush-overhead-corrected time_us can land below zero for a near-instant kernel;
    // this must not divide into -inf.
    EXPECT_EQ(rate_per_second(2.0, -0.5, -1.0), -1.0);
}
