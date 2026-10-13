// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "candidate_search.hpp"

#include <algorithm>
#include <chrono>
#include <functional>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace
{
    // Holds the host for about `us` microseconds. These cases time on the CPU, so a
    // launch that spins is measured at roughly its spin.
    void spin_us(double us)
    {
        const auto end
            = std::chrono::steady_clock::now() + std::chrono::duration<double, std::micro>(us);
        while(std::chrono::steady_clock::now() < end)
        {
        }
    }

    std::string entry(const char* what, size_t sol)
    {
        return std::string(what) + " " + std::to_string(sol);
    }

    std::string entry(const char* what, size_t sol, int64_t i)
    {
        return entry(what, sol) + " " + std::to_string(i);
    }

    // A search over host-only candidates, each launch spinning for its candidate's
    // time, that records every callback.
    struct HostSearch
    {
        std::vector<double>         cold_us;
        std::vector<double>         launch_us;
        std::vector<std::string>    log;
        std::function<bool()>       should_abort;
        std::function<void(size_t)> on_prepare;

        hipblaslt_bench::SearchResult run(const hipblaslt_bench::SearchConfig& search,
                                          const hipblaslt_bench::TimingConfig& timing)
        {
            hipEvent_t  start  = nullptr;
            hipEvent_t  stop   = nullptr;
            hipStream_t stream = nullptr;
            return hipblaslt_bench::search_candidates(
                launch_us.size(),
                search,
                timing,
                start,
                stop,
                stream,
                [&](size_t sol) {
                    log.push_back(entry("prepare", sol));
                    if(on_prepare)
                        on_prepare(sol);
                },
                [&](size_t sol, int32_t i) {
                    log.push_back(entry("cold", sol, i));
                    spin_us(cold_us.empty() ? 0.0 : cold_us[sol]);
                },
                [&](size_t sol, int64_t i) {
                    log.push_back(entry("launch", sol, i));
                    spin_us(launch_us[sol]);
                },
                [&](size_t sol, double, double) { log.push_back(entry("skipped", sol)); },
                [&](size_t sol) { log.push_back(entry("measuring", sol)); },
                [&](size_t sol, hipblaslt_bench::TimingResult&) {
                    log.push_back(entry("measured", sol));
                },
                should_abort);
        }
    };

    hipblaslt_bench::TimingConfig cpu_timing(int32_t iters)
    {
        hipblaslt_bench::TimingConfig timing;
        timing.iters         = iters;
        timing.use_gpu_timer = false;
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
    EXPECT_GT(result.best_timing.median_us, 0.0);

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

    auto count = [&](const std::string& line) {
        return std::count(host.log.begin(), host.log.end(), line);
    };
    EXPECT_EQ(count(entry("skipped", 0)), 0);
    EXPECT_EQ(count(entry("skipped", 1)), 1);
    EXPECT_EQ(count(entry("skipped", 2)), 0);
    EXPECT_EQ(count(entry("measuring", 1)), 0);
    EXPECT_EQ(count(entry("measured", 0)), 1);
    EXPECT_EQ(count(entry("measured", 2)), 1);
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
