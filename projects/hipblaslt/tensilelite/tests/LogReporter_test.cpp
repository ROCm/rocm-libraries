// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "LogReporter.hpp"
#include "MetaResultReporter.hpp"

#include <memory>
#include <sstream>
#include <string>

using namespace TensileLite::Client;

namespace
{
    std::shared_ptr<LogReporter> makeLogReporter(LogLevel level, std::ostream& stream)
    {
        return std::make_shared<LogReporter>(
            level,
            std::initializer_list<std::string>{ResultKey::Validation,
                                               ResultKey::TimeUS,
                                               ResultKey::SpeedGFlops,
                                               ResultKey::TotalGranularity},
            stream,
            false,
            false);
    }

    // Reports TotalGranularity into the reporter that asks, and counts the calls.
    struct CountingProvider
    {
        int calls = 0;

        ResultReporter::RowValueProvider get()
        {
            return [this](ResultReporter& reporter) {
                ++calls;
                reporter.report(ResultKey::TotalGranularity, 0.5);
            };
        }
    };

    // One solution's row: optional provider, then the verdict and time values.
    void runSolution(ResultReporter&                   reporter,
                     ResultReporter::RowValueProvider* provider,
                     std::string const&                validation)
    {
        reporter.preSolution(nullptr);
        if(provider)
            reporter.setRowValueProvider(*provider);
        reporter.report(ResultKey::Validation, validation);
        reporter.report(ResultKey::TimeUS, 1.0);
        reporter.report(ResultKey::SpeedGFlops, 2.0);
        reporter.postSolution();
    }
} // namespace

// A LogReporter asks for the deferred row values only when it prints the row, and the
// provider does not outlive its solution.
TEST(LogReporterRowValueProvider, OnlyForPrintedRows)
{
    std::ostringstream out;
    auto               reporter = makeLogReporter(LogLevel::Error, out);
    CountingProvider   counter;
    auto               provider = counter.get();

    reporter->preProblem(nullptr);
    runSolution(*reporter, &provider, "PASSED"); // suppressed at Error
    EXPECT_EQ(counter.calls, 0);
    runSolution(*reporter, &provider, "FAILED"); // printed at every level
    EXPECT_EQ(counter.calls, 1);
    runSolution(*reporter, nullptr, "FAILED"); // no provider for this solution
    EXPECT_EQ(counter.calls, 1);
    reporter->postProblem();

    EXPECT_EQ(out.str(),
              "validation,time-us,gflops,total-gran\n"
              "FAILED,1,2,0.5\n"
              "FAILED,1,2,\n");
}

// MetaResultReporter forwards the provider to every reporter; each one that prints the row
// (here the Normal one, as main.cpp's log-file reporter) fills its own row.
TEST(LogReporterRowValueProvider, ForwardedToEveryReporter)
{
    std::ostringstream errorOut, normalOut;
    MetaResultReporter meta;
    meta.addReporter(makeLogReporter(LogLevel::Error, errorOut));
    meta.addReporter(makeLogReporter(LogLevel::Normal, normalOut));
    CountingProvider counter;
    auto             provider = counter.get();

    meta.preProblem(nullptr);
    runSolution(meta, &provider, "PASSED");
    meta.postProblem();

    EXPECT_EQ(counter.calls, 1);
    EXPECT_EQ(errorOut.str(), "validation,time-us,gflops,total-gran\n");
    EXPECT_EQ(normalOut.str(),
              "validation,time-us,gflops,total-gran\n"
              "PASSED,1,2,0.5\n");
}
