// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "BenchmarkTimer.hpp"
#include "ProgramOptions.hpp"
#include "ResultReporter.hpp"
#include "TimingEvents.hpp"

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>

#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace TensileLite;
using namespace TensileLite::Client;

namespace
{
    template <typename T>
    void setOption(po::variables_map& args, std::string const& name, T value)
    {
        args[name].value() = std::move(value);
    }

    po::variables_map makeTimerArgs(int enqueuesPerSync, int syncsPerBenchmark)
    {
        po::variables_map args;
        setOption(args, "num-warmups", 0);
        setOption(args, "sync-after-warmups", true);
        setOption(args, "num-benchmarks", 1);
        setOption(args, "num-enqueues-per-sync", enqueuesPerSync);
        setOption(args, "max-enqueues-per-sync", -1);
        setOption(args, "min-flops-per-sync", size_t(0));
        setOption(args, "num-syncs-per-benchmark", syncsPerBenchmark);
        setOption(args, "use-gpu-timer", true);
        setOption(args, "sleep-percent", 0);
        setOption(args, "skip-slow-solution-ratio", 0.0f);
        setOption(args, "prob-sol-map", std::map<int, int>{});
        return args;
    }

    class CapturingReporter : public ResultReporter
    {
    public:
        void reportValue_string(std::string const& key, std::string const& value) override {}
        void reportValue_uint(std::string const& key, uint64_t value) override {}
        void reportValue_int(std::string const& key, int64_t value) override {}
        void reportValue_double(std::string const& key, double value) override
        {
            doubles[key] = value;
        }
        void reportValue_sizes(std::string const& key, std::vector<size_t> const& value) override {}
        void reportValue_vecOfSizes(std::string const&                      key,
                                    std::vector<std::vector<size_t>> const& value) override
        {
        }
        void finalizeReport() override {}

        std::map<std::string, double> doubles;
    };

    class ScriptedTimer : public BenchmarkTimer
    {
    public:
        using BenchmarkTimer::BenchmarkTimer;
        std::vector<double> batchMs;

    protected:
        double takeBatchTimeMs() override
        {
            double ms = batchMs.front();
            batchMs.erase(batchMs.begin());
            return ms;
        }
    };

    class BenchmarkTimerTest : public ::testing::Test
    {
    protected:
        BenchmarkTimerTest()
            : device(AMDGPU::Processor::gfx950, 256, "test-gfx950")
            , problem(ContractionProblemGemm::GEMM(
                  false, false, 256, 256, 256, 256, 256, 256, 1.0, false, 1))
            , reporter(std::make_shared<CapturingReporter>())
        {
            solution.sizeMapping.macroTile     = TensileLite::dim3(128, 128, 1);
            solution.sizeMapping.workGroupSize = TensileLite::dim3(256, 1, 1);
            solution.sizeMapping.threadTile    = TensileLite::dim3(1, 1, 1);
            solution.sizeMapping.depthU        = 64;
            solution.sizeMapping.globalSplitU  = 1;
        }

        void start(BenchmarkTimer& timer)
        {
            timer.setReporter(reporter);
            timer.preProblem(&problem);
        }

        void runSolution(ScriptedTimer& timer, std::vector<double> const& timesUs)
        {
            timer.preSolution(&solution);
            for(double t : timesUs)
                runBatch(timer, t);
            timer.postSolution();
        }

        void runBatch(ScriptedTimer& timer, double timeUs)
        {
            TimingEvents events({}, 1, 0);
            timer.batchMs.push_back(timeUs / 1000.0);
            timer.preEnqueues();
            timer.postEnqueues(events, events);
            timer.validateEnqueues({0}, events, events);
        }

        double reported(std::string const& key)
        {
            return reporter->doubles.at(key);
        }

        AMDGPU                             device;
        ContractionProblemGemm             problem;
        ContractionSolution                solution;
        std::shared_ptr<CapturingReporter> reporter;
    };
}

TEST_F(BenchmarkTimerTest, ReportsMeanOfEnqueueTimes)
{
    ScriptedTimer timer(makeTimerArgs(10, 1), device, 0.0f, {});
    start(timer);
    runSolution(timer, {10, 1, 9, 2, 8, 3, 7, 4, 6, 5});
    EXPECT_NEAR(reported(ResultKey::TimeUS), 5.5, 1e-9);
}

TEST_F(BenchmarkTimerTest, SubtractsFlushTime)
{
    ScriptedTimer timer(makeTimerArgs(1, 1), device, 0.5f, {});
    start(timer);
    runSolution(timer, {4.0});
    EXPECT_NEAR(reported(ResultKey::TimeUS), 3.5, 1e-9);
}

TEST_F(BenchmarkTimerTest, TimesDoNotLeakAcrossSolutions)
{
    ScriptedTimer timer(makeTimerArgs(3, 1), device, 0.0f, {});
    start(timer);
    runSolution(timer, {100.0, 100.0, 100.0});
    runSolution(timer, {1.0, 2.0, 3.0});
    EXPECT_NEAR(reported(ResultKey::TimeUS), 2.0, 1e-9);
}

TEST_F(BenchmarkTimerTest, EnqueueTimesCountTowardRunsInSolution)
{
    ScriptedTimer timer(makeTimerArgs(2, 2), device, 0.0f, {});
    start(timer);
    timer.preSolution(&solution);
    for(int i = 0; i < 3; i++)
        runBatch(timer, 1.0);
    EXPECT_TRUE(timer.needMoreRunsInSolution());
    runBatch(timer, 1.0);
    EXPECT_FALSE(timer.needMoreRunsInSolution());
}
