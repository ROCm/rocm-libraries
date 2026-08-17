// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "ResultReporter.hpp"
#include "SynchronizerValidator.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace TensileLite::Client;

namespace
{
    // Captures every reportValue_string call; the FAILED verdict is reported
    // through this overload.
    class FakeResultReporter : public ResultReporter
    {
    public:
        std::vector<std::pair<std::string, std::string>> stringReports;

        void reportValue_string(std::string const& key, std::string const& value) override
        {
            stringReports.emplace_back(key, value);
        }
        void reportValue_uint(std::string const&, uint64_t) override { }
        void reportValue_int(std::string const&, int64_t) override { }
        void reportValue_double(std::string const&, double) override { }
        void reportValue_sizes(std::string const&, std::vector<size_t> const&) override { }
        void reportValue_vecOfSizes(std::string const&,
                                    std::vector<std::vector<size_t>> const&) override
        {
        }
        void finalizeReport() override { }
    };

    // Exposes the protected reporting state so the test can drive
    // preSolution()/postSolution() without a GPU-backed dirty buffer.
    class TestableSynchronizerValidator : public SynchronizerValidator
    {
    public:
        using SynchronizerValidator::SynchronizerValidator;
        void markDirty()
        {
            m_dirtyInSolution = true;
        }
    };

    po::variables_map enabledArgs()
    {
        po::variables_map vm;
        vm["check-streamk-sync"].value() = true;
        return vm;
    }
}

TEST(SynchronizerValidatorReporting, CleanSolutionReportsNothing)
{
    TestableSynchronizerValidator validator(enabledArgs());
    auto                          reporter = std::make_shared<FakeResultReporter>();
    validator.setReporter(reporter);

    validator.preSolution(nullptr);
    validator.postSolution();

    EXPECT_EQ(validator.error(), 0);
    EXPECT_TRUE(reporter->stringReports.empty());
}

TEST(SynchronizerValidatorReporting, DirtySolutionReportsFailureOnce)
{
    TestableSynchronizerValidator validator(enabledArgs());
    auto                          reporter = std::make_shared<FakeResultReporter>();
    validator.setReporter(reporter);

    validator.preSolution(nullptr);
    validator.markDirty();
    validator.postSolution();

    EXPECT_EQ(validator.error(), 1);
    ASSERT_EQ(reporter->stringReports.size(), 1u);
    EXPECT_EQ(reporter->stringReports[0].first, ResultKey::Validation);
    EXPECT_EQ(reporter->stringReports[0].second, "FAILED");

    // preSolution() resets the flag, so the next (clean) solution does not
    // re-report the previous one's failure.
    validator.preSolution(nullptr);
    validator.postSolution();
    EXPECT_EQ(validator.error(), 1);
    EXPECT_EQ(reporter->stringReports.size(), 1u);
}
