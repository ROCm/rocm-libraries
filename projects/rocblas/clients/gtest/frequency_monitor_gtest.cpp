// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The clock monitor samples with AMD SMI on Linux only; on Windows it is a stub that is never
// enabled, and setenv/unsetenv do not exist
#ifndef _WIN32

#include "testing_common.hpp"

#include <chrono>
#include <cstdlib>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace
{
    // The monitor reads ROCBLAS_BENCH_FREQ, ROCBLAS_BENCH_FREQ_ALL and ROCBLAS_BENCH_PERF when a
    // thread first uses it (one monitor per thread), so each case runs f in a new thread with only
    // the variables in set defined, and restores the environment afterwards.
    template <typename F>
    void run_with_bench_env(std::vector<const char*> set, F f)
    {
        const char* names[]
            = {"ROCBLAS_BENCH_FREQ", "ROCBLAS_BENCH_FREQ_ALL", "ROCBLAS_BENCH_PERF"};
        std::vector<std::pair<const char*, std::optional<std::string>>> saved;
        for(const char* name : names)
        {
            const char* value = std::getenv(name);
            saved.emplace_back(name, value ? std::optional<std::string>(value) : std::nullopt);
            unsetenv(name);
        }
        for(const char* name : set)
            setenv(name, "1", 1);

        std::thread(f).join();

        for(auto& [name, value] : saved)
        {
            if(value)
                setenv(name, value->c_str(), 1);
            else
                unsetenv(name);
        }
    }

    std::string clock_columns()
    {
        rocblas_internal_ostream name_line;
        rocblas_internal_ostream val_line;
        ArgumentModel_log_frequencies(name_line, val_line);
        return name_line.str();
    }

    TEST(quick_frequency_monitor, columns_follow_the_environment)
    {
        run_with_bench_env({}, [] {
            EXPECT_FALSE(getFrequencyMonitor().enabled());
            EXPECT_FALSE(getFrequencyMonitor().efficiencyReport());
            EXPECT_EQ("", clock_columns());
        });
        run_with_bench_env({"ROCBLAS_BENCH_FREQ"}, [] {
            EXPECT_TRUE(getFrequencyMonitor().enabled());
            EXPECT_FALSE(getFrequencyMonitor().efficiencyReport());
            EXPECT_EQ(",lowest-avg-freq,lowest-median-freq,avg-MCLK,median-MCLK", clock_columns());
        });
        run_with_bench_env({"ROCBLAS_BENCH_PERF"}, [] {
            EXPECT_TRUE(getFrequencyMonitor().enabled());
            EXPECT_TRUE(getFrequencyMonitor().efficiencyReport());
        });
    }

    // A timing loop reports the clocks sampled during its own run, never those of an earlier
    // function: a run too short to be sampled reports 0. Where AMD SMI cannot read the clocks
    // the first run reports 0 as well.
    TEST(quick_frequency_monitor, every_run_starts_fresh)
    {
        run_with_bench_env({"ROCBLAS_BENCH_FREQ"}, [] {
            FrequencyMonitor& monitor = getFrequencyMonitor();
            monitor.start();
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            monitor.stop();
            rocblas_cout << "sampled run: " << monitor.getLowestAverageSYSCLK() << " MHz engine, "
                         << monitor.getAverageMEMCLK() << " MHz memory" << std::endl;

            monitor.start();
            monitor.stop();
            EXPECT_EQ(0.0, monitor.getLowestAverageSYSCLK());
            EXPECT_EQ(0.0, monitor.getLowestMedianSYSCLK());
            EXPECT_EQ(0.0, monitor.getAverageMEMCLK());
            EXPECT_EQ(0.0, monitor.getMedianMEMCLK());
        });
    }
}

#endif // _WIN32
