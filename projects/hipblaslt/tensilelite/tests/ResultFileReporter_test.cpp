// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "ResultFileReporter.hpp"

// Not self-contained: needs ResultReporter from the header above.
#include "MetaResultReporter.hpp"

#include <filesystem>
#include <string>
#include <system_error>

#include <unistd.h>

using namespace TensileLite::Client;
namespace fs = std::filesystem;

namespace
{
    struct ResultFileReporterDefault : public ::testing::Test
    {
        fs::path tmpDir;

        void SetUp() override
        {
            tmpDir = fs::temp_directory_path()
                     / fs::path("tensilelite-results-file-"
                                + std::to_string(static_cast<long long>(getpid())) + "-"
                                + ::testing::UnitTest::GetInstance()->current_test_info()->name());
            fs::create_directories(tmpDir);
        }

        void TearDown() override
        {
            std::error_code ec;
            fs::remove_all(tmpDir, ec);
        }

        po::variables_map args(bool noResultsFile) const
        {
            po::variables_map args;
            args["results-file"].value()            = (tmpDir / "results.csv").string();
            args["no-results-file"].value()         = noResultsFile;
            args["csv-export-extra-cols"].value()   = false;
            args["csv-merge-same-problems"].value() = false;
            args["performance-metric"].value()      = TensileLite::PerformanceMetric::Auto;
            return args;
        }

        // One problem with one solution row, reported the way main.cpp does.
        void runOneProblem(bool noResultsFile) const
        {
            auto reporters = std::make_shared<MetaResultReporter>();
            reporters->addReporter(ResultFileReporter::Default(args(noResultsFile)));
            reporters->preProblem(nullptr);
            reporters->preSolution(nullptr);
            reporters->report(ResultKey::Validation, std::string("PASSED"));
            reporters->report(ResultKey::SpeedGFlops, 1.0);
            reporters->postSolution();
            reporters->postProblem();
            reporters->finalizeReport();
        }
    };
} // namespace

// no-results-file constructs no reporter, so nothing is written, whatever results-file says.
TEST_F(ResultFileReporterDefault, NoResultsFileWritesNothing)
{
    EXPECT_EQ(ResultFileReporter::Default(args(true)), nullptr);

    runOneProblem(true);
    EXPECT_TRUE(fs::is_empty(tmpDir));
}

// Without it, results-file is written as before.
TEST_F(ResultFileReporterDefault, WritesResultsFile)
{
    EXPECT_NE(ResultFileReporter::Default(args(false)), nullptr);

    runOneProblem(false);
    EXPECT_TRUE(fs::exists(tmpDir / "results.csv"));
    EXPECT_GT(fs::file_size(tmpDir / "results.csv"), 0u);
}
