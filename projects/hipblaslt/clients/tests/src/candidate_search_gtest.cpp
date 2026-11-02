// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipblaslt_benchmark/candidate_search.hpp>

#include <algorithm>
#include <functional>
#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace
{
    // A hooks policy whose clock moves only when a launch below moves it, so every time
    // is exact and no GPU is involved. A launch can make the next read of it fail.
    struct FakeClock
    {
        static inline double now_us         = 0.0;
        static inline bool   fail_next_read = false;
        static inline int    failed_reads   = 0;

        static bool check(hipError_t status, const char*, const char*, int)
        {
            return status == hipSuccess;
        }

        static bool time_us_sync(hipStream_t, double& us)
        {
            if(fail_next_read)
            {
                fail_next_read = false;
                failed_reads++;
                return false;
            }
            us = now_us;
            return true;
        }
    };

    std::string entry(const char* what, size_t sol)
    {
        return std::string(what) + " " + std::to_string(sol);
    }

    std::string entry(const char* what, size_t sol, int64_t i)
    {
        return entry(what, sol) + " " + std::to_string(i);
    }

    // A search over host-only candidates, each launch moving the clock on by its
    // candidate's time, that records every callback.
    struct HostSearch
    {
        std::vector<double>                  cold_us;
        std::vector<double>                  launch_us;
        std::vector<std::string>             log;
        std::function<bool()>                should_abort;
        std::function<void(size_t)>          on_prepare;
        std::function<void(size_t, int32_t)> on_cold;
        std::vector<bool>                    runnable; // prepare's answers, empty for all yes
        std::vector<bool>                    eligible; // measured's answers, empty for all yes
        std::function<bool()>                should_stop;
        std::function<void(size_t, int64_t)> on_launch;

        // The candidate whose cold or timed launches the clock cannot time: the first
        // read after them fails.
        std::optional<size_t> untimed_cold;
        std::optional<size_t> untimed_launch;

        HostSearch()
        {
            FakeClock::now_us         = 0.0;
            FakeClock::fail_next_read = false;
            FakeClock::failed_reads   = 0;
        }

        hipblaslt_bench::SearchResult run(const hipblaslt_bench::SearchConfig& search,
                                          const hipblaslt_bench::TimingConfig& timing)
        {
            return hipblaslt_bench::search_candidates<FakeClock>(
                launch_us.size(),
                search,
                timing,
                nullptr,
                nullptr,
                nullptr,
                hipblaslt_bench::search_callbacks()
                    .prepare([&](size_t sol) {
                        log.push_back(entry("prepare", sol));
                        if(on_prepare)
                            on_prepare(sol);
                        return runnable.empty() || runnable[sol];
                    })
                    .cold([&](size_t sol, int32_t i) {
                        log.push_back(entry("cold", sol, i));
                        FakeClock::now_us += cold_us.empty() ? 0.0 : cold_us[sol];
                        FakeClock::fail_next_read = untimed_cold == sol;
                        if(on_cold)
                            on_cold(sol, i);
                    })
                    .launch([&](size_t sol, int64_t i) {
                        log.push_back(entry("launch", sol, i));
                        FakeClock::now_us += launch_us[sol];
                        FakeClock::fail_next_read = untimed_launch == sol;
                        if(on_launch)
                            on_launch(sol, i);
                    })
                    .skipped(
                        [&](size_t sol, double, double) { log.push_back(entry("skipped", sol)); })
                    .measuring([&](size_t sol) { log.push_back(entry("measuring", sol)); })
                    .measured([&](size_t sol, hipblaslt_bench::TimingResult&) {
                        log.push_back(entry("measured", sol));
                        return eligible.empty() || eligible[sol];
                    })
                    .should_stop([&] { return should_stop && should_stop(); }),
                should_abort);
        }

        long count(const std::string& line) const
        {
            return std::count(log.begin(), log.end(), line);
        }
    };

    hipblaslt_bench::TimingConfig cpu_timing(int32_t iters)
    {
        hipblaslt_bench::TimingConfig timing;
        timing.iters         = iters;
        timing.use_gpu_timer = false;
        return timing;
    }

    // Adaptive timing that takes exactly three one-launch samples, after one probe launch.
    hipblaslt_bench::TimingConfig adaptive_timing()
    {
        hipblaslt_bench::TimingConfig timing = cpu_timing(1);
        timing.adaptive                      = true;
        timing.min_iters                     = 3;
        timing.max_iters                     = 3;
        return timing;
    }
} // namespace

// Every candidate is prepared, given its cold launches and then timed, in order, and
// the one with the lowest median wins.
TEST(CandidateSearch_pre_checkin, RunsEachCandidateInOrderAndPicksTheFastest)
{
    HostSearch host;
    host.launch_us = {300, 100, 200};

    hipblaslt_bench::SearchConfig search;
    search.cold_calls = 2;

    const auto result = host.run(search, cpu_timing(3));
    EXPECT_FALSE(result.aborted);
    EXPECT_EQ(result.best, 1u);
    EXPECT_EQ(result.best_timing.median_us, 100.0);

    std::vector<std::string> expected;
    for(size_t sol = 0; sol < host.launch_us.size(); sol++)
    {
        expected.push_back(entry("prepare", sol));
        expected.push_back(entry("cold", sol, 0));
        expected.push_back(entry("cold", sol, 1));
        expected.push_back(entry("measuring", sol));
        for(int64_t i = 0; i < 3; i++)
            expected.push_back(entry("launch", sol, i));
        expected.push_back(entry("measured", sol));
    }
    EXPECT_EQ(host.log, expected);
}

// With a skip ratio, a candidate whose cold launches take more than the best cold time
// divided by the ratio is dropped before it is timed, and cannot win.
TEST(CandidateSearch_pre_checkin, SkipSlowDropsOnlyACandidateWithSlowColdLaunches)
{
    HostSearch host;
    host.cold_us   = {200, 800, 240};
    host.launch_us = {300, 50, 200};

    hipblaslt_bench::SearchConfig search;
    search.cold_calls      = 2;
    search.skip_slow_ratio = 0.5f;

    const auto result = host.run(search, cpu_timing(2));
    EXPECT_EQ(result.best, 2u) << "the fastest launch belongs to the skipped candidate";
    EXPECT_EQ(host.count(entry("skipped", 0)), 0);
    EXPECT_EQ(host.count(entry("skipped", 1)), 1);
    EXPECT_EQ(host.count(entry("skipped", 2)), 0);
    EXPECT_EQ(host.count(entry("measuring", 1)), 0);
    EXPECT_EQ(host.count(entry("measured", 0)), 1);
    EXPECT_EQ(host.count(entry("measured", 2)), 1);
}

// A fatal failure while a candidate is prepared ends the search there.
TEST(CandidateSearch_pre_checkin, AFailureWhilePreparingEndsTheSearch)
{
    HostSearch host;
    host.launch_us = {50, 50, 50};

    bool failed       = false;
    host.should_abort = [&] { return failed; };
    host.on_prepare   = [&](size_t sol) { failed = failed || sol == 1; };

    hipblaslt_bench::SearchConfig search;
    const auto                    result = host.run(search, cpu_timing(1));
    EXPECT_TRUE(result.aborted);
    ASSERT_FALSE(host.log.empty());
    EXPECT_EQ(host.log.back(), entry("prepare", 1));
}

// A fatal failure in a cold launch ends the search at that launch, whether or not the
// cold launches are being timed.
TEST(CandidateSearch_pre_checkin, AFailureDuringAColdLaunchEndsTheSearch)
{
    for(float ratio : {0.0f, 0.5f})
    {
        SCOPED_TRACE(ratio);
        HostSearch host;
        host.launch_us = {50, 50};

        bool failed       = false;
        host.should_abort = [&] { return failed; };
        host.on_cold      = [&](size_t, int32_t) { failed = true; };

        hipblaslt_bench::SearchConfig search;
        search.cold_calls      = 3;
        search.skip_slow_ratio = ratio;
        const auto result      = host.run(search, cpu_timing(2));
        EXPECT_TRUE(result.aborted);
        EXPECT_EQ(result.best, std::numeric_limits<size_t>::max());
        EXPECT_EQ(host.log, (std::vector<std::string>{entry("prepare", 0), entry("cold", 0, 0)}));
    }
}

// A candidate whose timed launches could not be timed is passed over, rather than
// winning with the zero it was never measured at.
TEST(CandidateSearch_pre_checkin, ACandidateThatCouldNotBeTimedCannotWin)
{
    HostSearch host;
    host.launch_us      = {10, 300, 200};
    host.untimed_launch = 0;

    const auto result = host.run(hipblaslt_bench::SearchConfig{}, cpu_timing(2));
    EXPECT_FALSE(result.aborted);
    EXPECT_EQ(result.best, 2u);
    EXPECT_EQ(result.best_timing.median_us, 200.0);
    EXPECT_EQ(host.count(entry("measured", 0)), 0);
}

// A candidate whose cold launches could not be timed is passed over without setting
// the best cold time, which would otherwise get every later candidate skipped.
TEST(CandidateSearch_pre_checkin, AFailedColdTimingSkipsNoOtherCandidate)
{
    HostSearch host;
    host.cold_us         = {100, 100, 100};
    host.launch_us       = {300, 200, 100};
    host.untimed_cold    = 0;

    hipblaslt_bench::SearchConfig search;
    search.cold_calls      = 1;
    search.skip_slow_ratio = 0.5f;

    const auto result = host.run(search, cpu_timing(1));
    EXPECT_EQ(result.best, 2u);
    EXPECT_EQ(host.count(entry("measuring", 0)), 0);
    EXPECT_EQ(host.count(entry("skipped", 1)), 0);
    EXPECT_EQ(host.count(entry("skipped", 2)), 0);
    EXPECT_EQ(host.count(entry("measured", 1)), 1);
    EXPECT_EQ(host.count(entry("measured", 2)), 1);
}

// A timing failure the hooks report as fatal, as ClientBenchmarkHooks does under gtest,
// ends the search when it happens while the cold launches are timed.
TEST(CandidateSearch_pre_checkin, AFatalFailureTimingTheColdLaunchesEndsTheSearch)
{
    HostSearch host;
    host.cold_us      = {100, 100};
    host.launch_us    = {50, 50};
    host.untimed_cold = 0;
    host.should_abort = [] { return FakeClock::failed_reads > 0; };

    hipblaslt_bench::SearchConfig search;
    search.cold_calls      = 1;
    search.skip_slow_ratio = 0.5f;

    const auto result = host.run(search, cpu_timing(1));
    EXPECT_TRUE(result.aborted);
    EXPECT_EQ(host.count(entry("measuring", 0)), 0);
    EXPECT_EQ(host.count(entry("prepare", 1)), 0);
}

// An adaptive configuration that run_measurement would reject searches nothing.
TEST(CandidateSearch_pre_checkin, AnInvalidAdaptiveConfigurationSearchesNothing)
{
    HostSearch host;
    host.launch_us = {100, 200};

    hipblaslt_bench::TimingConfig timing = cpu_timing(1);
    timing.adaptive                      = true;
    timing.min_iters                     = -1;
    ASSERT_FALSE(hipblaslt_bench::validate_adaptive_config(timing).empty());

    const auto result = host.run(hipblaslt_bench::SearchConfig{}, timing);
    EXPECT_FALSE(result.aborted);
    EXPECT_EQ(result.best, std::numeric_limits<size_t>::max());
    EXPECT_TRUE(host.log.empty());
}

// On a tie the earlier candidate stays.
TEST(CandidateSearch_pre_checkin, ATieKeepsTheEarlierCandidate)
{
    HostSearch host;
    host.launch_us = {200, 100, 100};

    const auto result = host.run(hipblaslt_bench::SearchConfig{}, cpu_timing(2));
    EXPECT_EQ(result.best, 1u);
}

// A fatal failure during a candidate's timed launches ends the search before that
// candidate is reported as measured, whether its timing is fixed or adaptive.
TEST(CandidateSearch_pre_checkin, AFailureDuringTheTimedLaunchesEndsTheSearch)
{
    for(const auto& timing : {cpu_timing(2), adaptive_timing()})
    {
        SCOPED_TRACE(timing.adaptive ? "adaptive" : "fixed");
        HostSearch host;
        host.launch_us = {50, 40, 30};

        bool failed       = false;
        host.should_abort = [&] { return failed; };
        host.on_launch    = [&](size_t sol, int64_t) { failed = failed || sol == 1; };

        const auto result = host.run(hipblaslt_bench::SearchConfig{}, timing);
        EXPECT_TRUE(result.aborted);
        EXPECT_EQ(host.count(entry("measured", 0)), 1);
        EXPECT_EQ(host.count(entry("measured", 1)), 0);
        EXPECT_EQ(host.count(entry("prepare", 2)), 0);
    }
}

// A valid adaptive configuration times every candidate and keeps the fastest.
TEST(CandidateSearch_pre_checkin, AnAdaptiveConfigurationPicksTheFastest)
{
    HostSearch host;
    host.launch_us = {300, 100, 200};

    const auto result = host.run(hipblaslt_bench::SearchConfig{}, adaptive_timing());
    EXPECT_FALSE(result.aborted);
    EXPECT_EQ(result.best, 1u);
    EXPECT_TRUE(result.best_timing.adaptive);
    EXPECT_EQ(result.best_timing.samples, 3);
    EXPECT_EQ(result.best_timing.median_us, 100.0);
}

// The library's default policy fails on any HIP error and reads a clock that does not
// run backwards.
TEST(CandidateSearch_pre_checkin, DefaultHooksFailOnAHipError)
{
    EXPECT_TRUE(hipblaslt_bench::DefaultHooks::check(hipSuccess, "", "", 0));
    EXPECT_FALSE(hipblaslt_bench::DefaultHooks::check(hipErrorInvalidValue, "", "", 0));

    int devices = 0;
    if(hipGetDeviceCount(&devices) != hipSuccess || devices == 0)
        GTEST_SKIP() << "reading the clock synchronizes a stream, which needs a GPU";
    double first = 0.0, second = 0.0;
    ASSERT_TRUE(hipblaslt_bench::DefaultHooks::time_us_sync(nullptr, first));
    ASSERT_TRUE(hipblaslt_bench::DefaultHooks::time_us_sync(nullptr, second));
    EXPECT_GE(second, first);
}

// A candidate whose preparation says it cannot run is neither launched nor timed.
TEST(CandidateSearch_pre_checkin, PrepareCanSkipACandidate)
{
    HostSearch host;
    host.launch_us = {300, 50, 200};
    host.runnable  = {true, false, true};

    const auto result = host.run(hipblaslt_bench::SearchConfig{}, cpu_timing(2));
    EXPECT_EQ(result.best, 2u);
    EXPECT_EQ(host.count(entry("measuring", 1)), 0);
}

// A candidate measured but ruled out afterwards cannot win, however fast it was.
TEST(CandidateSearch_pre_checkin, MeasuredCanRuleOutACandidate)
{
    HostSearch host;
    host.launch_us = {300, 50, 200};
    host.eligible  = {true, false, true};

    const auto result = host.run(hipblaslt_bench::SearchConfig{}, cpu_timing(2));
    EXPECT_EQ(result.best, 2u);
    EXPECT_EQ(host.count(entry("measuring", 1)), 1);
}

// Validation after measurement can fail the test. Stop before selecting that
// candidate or running another one, even if the callback also returns false.
TEST(CandidateSearch_pre_checkin, AFailureAfterMeasurementEndsTheSearch)
{
    for(bool eligible : {false, true})
    {
        SCOPED_TRACE(eligible);
        bool   failed             = false;
        size_t prepared           = 0;
        size_t measured           = 0;
        FakeClock::now_us         = 0.0;
        FakeClock::fail_next_read = false;

        const auto result = hipblaslt_bench::search_candidates<FakeClock>(
            2,
            {},
            cpu_timing(1),
            nullptr,
            nullptr,
            nullptr,
            hipblaslt_bench::search_callbacks()
                .prepare([&](size_t) { ++prepared; })
                .launch([&](size_t, int64_t) { FakeClock::now_us += 10.0; })
                .measured([&](size_t, hipblaslt_bench::TimingResult&) {
                    ++measured;
                    failed = true;
                    return eligible;
                }),
            [&] { return failed; });

        EXPECT_TRUE(result.aborted);
        EXPECT_EQ(prepared, 1u);
        EXPECT_EQ(measured, 1u);
        EXPECT_EQ(result.best, std::numeric_limits<size_t>::max());
    }
}

// should_stop is asked before each candidate; the search ends at the first yes and
// reports so, keeping the best of the candidates it reached.
TEST(CandidateSearch_pre_checkin, ShouldStopEndsTheSearchBetweenCandidates)
{
    HostSearch host;
    host.launch_us   = {300, 200, 50};
    size_t asked     = 0;
    host.should_stop = [&] { return asked++ == 2; };

    const auto result = host.run(hipblaslt_bench::SearchConfig{}, cpu_timing(2));
    EXPECT_TRUE(result.stopped);
    EXPECT_EQ(result.best, 1u);
    EXPECT_EQ(host.count(entry("prepare", 2)), 0);
}
