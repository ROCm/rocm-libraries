// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "DatabaseReporter.hpp"
#include "SqliteDatabase.hpp"

#include <Tensile/ContractionProblem.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

using namespace TensileLite;
using namespace TensileLite::Client;

namespace
{
    // Unique temporary directory, removed with the -wal and -shm files it holds.
    class TempDb
    {
    public:
        TempDb()
        {
            std::string pattern
                = (std::filesystem::temp_directory_path() / "tensilelite_db_XXXXXX").string();
            char* created = mkdtemp(pattern.data());
            if(!created)
                throw std::runtime_error("cannot create temporary directory");
            m_dir = created;
        }

        ~TempDb()
        {
            std::error_code error;
            std::filesystem::remove_all(m_dir, error);
        }

        TempDb(TempDb const&)            = delete;
        TempDb& operator=(TempDb const&) = delete;

        std::string path() const
        {
            return (m_dir / "bench.db").string();
        }

    private:
        std::filesystem::path m_dir;
    };

    class DatabaseReporterTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            database = std::make_shared<BenchmarkDatabase>(
                tmp.path(),
                std::vector<SolutionRecord>{{10, "SolA", "specA", "msgpack"},
                                            {20, "SolB", "specB", "msgpack"}},
                "count=1\n",
                std::vector<InitConfigRecord>{{0}, {1}, {2}});
            database->beginSession(1, 2);
            reader.emplace(tmp.path());
            reporter = std::make_shared<DatabaseReporter>(database);
        }

        // Starts a problem and a solution the way main does.
        void startPair(int64_t problem, int64_t solution, bool newProblem = true)
        {
            if(newProblem)
                reporter->report(ResultKey::ProblemIndex, problem);
            reporter->preSolution(nullptr);
            reporter->beginRun(problem, solution);
        }

        static std::string where(int64_t problem, int64_t solution)
        {
            return " WHERE problem_id = (SELECT id FROM init_configs WHERE problem_index = "
                   + std::to_string(problem)
                   + ") AND solution_id = (SELECT id FROM solutions WHERE library_index = "
                   + std::to_string(solution) + ")";
        }

        std::optional<std::string> value(std::string const& table,
                                         int64_t             problem,
                                         int64_t             solution,
                                         std::string const&  key)
        {
            SqliteStatement statement(
                *reader, "SELECT value FROM " + table + where(problem, solution) + " AND key = ?1");
            statement.bindText(1, key);
            if(!statement.step())
                return std::nullopt;
            return statement.columnText(0);
        }

        std::optional<std::string> runValue(int64_t problem, int64_t solution, std::string const& key)
        {
            return value("run_values", problem, solution, key);
        }

        std::optional<std::string> problemValue(int64_t problem, std::string const& key)
        {
            SqliteStatement statement(
                *reader,
                "SELECT value FROM problem_values WHERE problem_id = (SELECT id FROM "
                "init_configs WHERE problem_index = "
                    + std::to_string(problem) + ") AND key = ?1");
            statement.bindText(1, key);
            if(!statement.step())
                return std::nullopt;
            return statement.columnText(0);
        }

        int64_t count(std::string const& sql)
        {
            SqliteStatement statement(*reader, sql);
            EXPECT_TRUE(statement.step());
            return statement.columnInt(0);
        }

        // Runs a pair that reports one validation verdict and returns its stored status.
        RunStatus runWithValidation(std::vector<std::string> const& verdicts)
        {
            startPair(0, 10);
            for(std::string const& verdict : verdicts)
                reporter->report(ResultKey::Validation, verdict);
            reporter->endRun(0, 10);
            return database->runStatus(0, 10);
        }

        TempDb                             tmp;
        std::shared_ptr<BenchmarkDatabase> database;
        // A second connection that reads what the database wrote.
        std::optional<SqliteConnection>   reader;
        std::shared_ptr<DatabaseReporter> reporter;
    };
}

template <typename T>
static std::string streamed(T const& value)
{
    std::ostringstream oss;
    oss << value;
    return oss.str();
}

TEST_F(DatabaseReporterTest, ValueFormatting)
{
    reporter->report(ResultKey::ProblemIndex, 0);
    reporter->report(ResultKey::ProblemSizes, std::vector<size_t>{512, 512, 1, 1024});
    reporter->preSolution(nullptr);
    reporter->beginRun(0, 10);

    reporter->report(ResultKey::SolutionName, std::string("Cijk_name"));
    reporter->report(ResultKey::TotalFlops, uint64_t(268435456));
    reporter->report(ResultKey::LDD, 512);
    reporter->report(ResultKey::SolutionIndex, int64_t(-7));
    reporter->report(ResultKey::TimeUS, 1234.5678);
    reporter->report(ResultKey::SpeedGFlops, 0.1 + 0.2);
    reporter->report(ResultKey::GbpsBW, std::numeric_limits<double>::quiet_NaN());
    reporter->report(ResultKey::ASizes, std::vector<size_t>{1, 2, 3});
    reporter->report(ResultKey::DSizes, std::vector<std::vector<size_t>>{{1, 2}, {3, 4}});
    reporter->endRun(0, 10);

    EXPECT_EQ(runValue(0, 10, ResultKey::SolutionName), "Cijk_name");
    EXPECT_EQ(runValue(0, 10, ResultKey::TotalFlops), "268435456");
    EXPECT_EQ(runValue(0, 10, ResultKey::LDD), "512");
    EXPECT_EQ(runValue(0, 10, ResultKey::SolutionIndex), "-7");
    EXPECT_EQ(runValue(0, 10, ResultKey::TimeUS), streamed(1234.5678));
    EXPECT_EQ(runValue(0, 10, ResultKey::TimeUS), "1234.57");
    EXPECT_EQ(runValue(0, 10, ResultKey::SpeedGFlops), streamed(0.1 + 0.2));
    EXPECT_EQ(runValue(0, 10, ResultKey::GbpsBW),
              streamed(std::numeric_limits<double>::quiet_NaN()));
    EXPECT_EQ(runValue(0, 10, ResultKey::ASizes), "(1,2,3)");
    EXPECT_EQ(runValue(0, 10, ResultKey::DSizes), "((1,2),(3,4))");

    EXPECT_EQ(problemValue(0, ResultKey::ProblemSizes), "(512,512,1,1024)");
    EXPECT_EQ(problemValue(0, "SizeI"), "512");
    EXPECT_EQ(problemValue(0, "SizeJ"), "512");
    EXPECT_EQ(problemValue(0, "SizeK"), "1");
    EXPECT_EQ(problemValue(0, "SizeL"), "1024");
    EXPECT_EQ(problemValue(0, ResultKey::ProblemIndex), "0");
}

TEST_F(DatabaseReporterTest, VecOfSizesProblemSizesStoresSizeKeys)
{
    reporter->report(ResultKey::ProblemIndex, 0);
    reporter->report(ResultKey::ProblemSizes,
                     std::vector<std::vector<size_t>>{{8, 16}, {32, 64}});
    startPair(0, 10, false);
    reporter->endRun(0, 10);

    EXPECT_EQ(problemValue(0, ResultKey::ProblemSizes), "((8,16),(32,64))");
    EXPECT_EQ(problemValue(0, "SizeI"), "8");
    EXPECT_EQ(problemValue(0, "SizeJ"), "16");
    EXPECT_EQ(problemValue(0, "SizeK"), std::nullopt);
}

TEST_F(DatabaseReporterTest, Scoping)
{
    reporter->report(ResultKey::ProblemCount, uint64_t(3));
    reporter->report(ResultKey::ProblemIndex, 0);
    reporter->report(ResultKey::ProblemProgress, std::string("0/2"));
    reporter->preSolution(nullptr);
    reporter->report(ResultKey::SolutionName, std::string("SolA"));
    reporter->beginRun(0, 10);
    reporter->endRun(0, 10);

    EXPECT_EQ(problemValue(0, ResultKey::ProblemCount), std::nullopt);
    EXPECT_EQ(problemValue(0, ResultKey::ProblemProgress), "0/2");
    EXPECT_EQ(problemValue(0, ResultKey::SolutionName), std::nullopt);
    EXPECT_EQ(runValue(0, 10, ResultKey::SolutionName), "SolA");
    EXPECT_EQ(runValue(0, 10, ResultKey::ProblemProgress), std::nullopt);

    // A second pair of the same problem keeps the stored problem values.
    reporter->preSolution(nullptr);
    reporter->report(ResultKey::SolutionName, std::string("SolB"));
    reporter->beginRun(0, 20);
    reporter->endRun(0, 20);
    EXPECT_EQ(problemValue(0, ResultKey::ProblemProgress), "0/2");
    EXPECT_EQ(runValue(0, 20, ResultKey::SolutionName), "SolB");
    EXPECT_EQ(runValue(0, 10, ResultKey::SolutionName), "SolA");

    // A new problem drops the previous problem's values.
    reporter->report(ResultKey::ProblemIndex, 1);
    reporter->report(ResultKey::LDA, 5);
    startPair(1, 10, false);
    reporter->endRun(1, 10);
    EXPECT_EQ(problemValue(1, ResultKey::LDA), "5");
    EXPECT_EQ(problemValue(1, ResultKey::ProblemProgress), std::nullopt);
    EXPECT_EQ(problemValue(0, ResultKey::ProblemProgress), "0/2");
}

TEST_F(DatabaseReporterTest, ProblemValuesAreWrittenOnce)
{
    reporter->report(ResultKey::ProblemIndex, 0);
    reporter->report(ResultKey::LDA, 5);
    startPair(0, 10, false);
    reporter->endRun(0, 10);

    // A value reported later in the problem must not rewrite the stored ones.
    reporter->report(ResultKey::LDB, 6);
    startPair(0, 20, false);
    reporter->endRun(0, 20);
    EXPECT_EQ(problemValue(0, ResultKey::LDA), "5");
    EXPECT_EQ(count("SELECT COUNT(*) FROM problem_values"), 2);
}

TEST_F(DatabaseReporterTest, ProblemIndexStartsANewProblem)
{
    reporter->report(ResultKey::ProblemIndex, 0);
    reporter->report(ResultKey::LDA, 5);
    reporter->report(ResultKey::LDB, 6);
    startPair(0, 10, false);
    reporter->endRun(0, 10);

    reporter->report(ResultKey::ProblemIndex, 1);
    reporter->report(ResultKey::LDA, 7);
    startPair(1, 10, false);
    reporter->endRun(1, 10);

    EXPECT_EQ(problemValue(0, ResultKey::LDA), "5");
    EXPECT_EQ(problemValue(0, ResultKey::LDB), "6");
    EXPECT_EQ(problemValue(1, ResultKey::LDA), "7");
    EXPECT_EQ(problemValue(1, ResultKey::LDB), std::nullopt);
    EXPECT_EQ(problemValue(1, ResultKey::ProblemIndex), "1");
}

TEST_F(DatabaseReporterTest, ValuesBeforeTheFirstProblemAreNotStored)
{
    reporter->report(ResultKey::ProblemCount, uint64_t(3));
    reporter->preBenchmarkRun();
    reporter->report(ResultKey::BenchmarkRunNumber, 0);
    reporter->report(ResultKey::ProblemIndex, 0);
    startPair(0, 10, false);
    reporter->endRun(0, 10);

    EXPECT_EQ(problemValue(0, ResultKey::ProblemCount), std::nullopt);
    EXPECT_EQ(problemValue(0, ResultKey::BenchmarkRunNumber), std::nullopt);
    EXPECT_EQ(problemValue(0, ResultKey::ProblemIndex), "0");
}

TEST_F(DatabaseReporterTest, Iterations)
{
    startPair(0, 10);
    reporter->report(ResultKey::SyncEnqueues, int64_t(2));
    reporter->report(ResultKey::SyncTimeUS, 12.3456789);
    reporter->report(ResultKey::SyncEnqueues, int64_t(4));
    reporter->report(ResultKey::SyncTimeUS, 0.1 + 0.2);
    reporter->endRun(0, 10);

    SqliteStatement statement(*reader,
                              "SELECT sync_index, time_us, enqueues FROM iterations" + where(0, 10)
                                  + " ORDER BY sync_index");
    ASSERT_TRUE(statement.step());
    EXPECT_EQ(statement.columnInt(0), 0);
    EXPECT_EQ(statement.columnDouble(1), 12.3456789);
    EXPECT_EQ(statement.columnInt(2), 2);
    ASSERT_TRUE(statement.step());
    EXPECT_EQ(statement.columnInt(0), 1);
    EXPECT_EQ(statement.columnDouble(1), 0.1 + 0.2);
    EXPECT_EQ(statement.columnInt(2), 4);
    EXPECT_FALSE(statement.step());

    EXPECT_EQ(runValue(0, 10, ResultKey::SyncEnqueues), std::nullopt);
    EXPECT_EQ(runValue(0, 10, ResultKey::SyncTimeUS), std::nullopt);
}

TEST_F(DatabaseReporterTest, IterationsRestartForEachSolution)
{
    startPair(0, 10);
    reporter->report(ResultKey::SyncEnqueues, int64_t(2));
    reporter->report(ResultKey::SyncTimeUS, 1.0);
    reporter->endRun(0, 10);

    startPair(0, 20, false);
    reporter->report(ResultKey::SyncEnqueues, int64_t(3));
    reporter->report(ResultKey::SyncTimeUS, 2.0);
    reporter->endRun(0, 20);

    EXPECT_EQ(count("SELECT COUNT(*) FROM iterations" + where(0, 10)), 1);
    EXPECT_EQ(count("SELECT COUNT(*) FROM iterations" + where(0, 20)), 1);
    EXPECT_EQ(count("SELECT enqueues FROM iterations" + where(0, 20)), 3);
}

TEST_F(DatabaseReporterTest, StatusFromValidation)
{
    EXPECT_EQ(runWithValidation({"PASSED"}), RunStatus::Pass);
}

TEST_F(DatabaseReporterTest, StatusNoCheckPasses)
{
    EXPECT_EQ(runWithValidation({"NO_CHECK"}), RunStatus::Pass);
}

TEST_F(DatabaseReporterTest, StatusFailedFails)
{
    EXPECT_EQ(runWithValidation({"FAILED"}), RunStatus::Failure);
    EXPECT_EQ(count("SELECT COUNT(*) FROM runs" + where(0, 10) + " AND validation = 'FAILED'"), 1);
}

TEST_F(DatabaseReporterTest, StatusFailureIsStickyAfterPassing)
{
    EXPECT_EQ(runWithValidation({"FAILED", "PASSED"}), RunStatus::Failure);
    EXPECT_EQ(runValue(0, 10, ResultKey::Validation), "FAILED");
}

TEST_F(DatabaseReporterTest, StatusPassingThenFailureFails)
{
    EXPECT_EQ(runWithValidation({"PASSED", "FAILED"}), RunStatus::Failure);
    EXPECT_EQ(runValue(0, 10, ResultKey::Validation), "FAILED");
}

TEST_F(DatabaseReporterTest, StatusWithoutValidationPasses)
{
    EXPECT_EQ(runWithValidation({}), RunStatus::Pass);
    EXPECT_EQ(count("SELECT COUNT(*) FROM runs" + where(0, 10) + " AND validation = ''"), 1);
}

TEST_F(DatabaseReporterTest, FailedVerdictSpreadsToTheSolution)
{
    EXPECT_EQ(runWithValidation({"FAILED"}), RunStatus::Failure);
    EXPECT_EQ(database->runStatus(1, 10), RunStatus::IndirectVerificationFailure);
    EXPECT_EQ(database->runStatus(2, 10), RunStatus::IndirectVerificationFailure);
    EXPECT_EQ(database->runStatus(0, 20), RunStatus::NotBenchmarked);
}

TEST_F(DatabaseReporterTest, UnsatisfiedAssertsDoNotSpread)
{
    EXPECT_EQ(runWithValidation({"DID_NOT_SATISFY_ASSERTS"}), RunStatus::Failure);
    EXPECT_EQ(database->runStatus(1, 10), RunStatus::NotBenchmarked);
}

TEST_F(DatabaseReporterTest, WrongHardwareFails)
{
    EXPECT_EQ(runWithValidation({"WRONG_HARDWARE"}), RunStatus::Failure);
    EXPECT_EQ(database->runStatus(1, 10), RunStatus::NotBenchmarked);
}

TEST_F(DatabaseReporterTest, UnsupportedXcdTopologyFails)
{
    EXPECT_EQ(runWithValidation({"UNSUPPORTED_XCD_TOPOLOGY"}), RunStatus::Failure);
}

TEST_F(DatabaseReporterTest, InvalidDoesNotSpread)
{
    EXPECT_EQ(runWithValidation({"INVALID"}), RunStatus::Failure);
    EXPECT_EQ(database->runStatus(1, 10), RunStatus::NotBenchmarked);
}

TEST_F(DatabaseReporterTest, BeginRunLeavesPairUnknown)
{
    startPair(0, 10);
    EXPECT_EQ(database->runStatus(0, 10), RunStatus::Unknown);
    EXPECT_EQ(count("SELECT COUNT(*) FROM runs" + where(0, 10)
                    + " AND start_time IS NOT NULL AND end_time IS NULL"),
              1);
}

TEST_F(DatabaseReporterTest, SkippedPairIsUntouched)
{
    startPair(0, 10);
    reporter->report(ResultKey::Validation, std::string("PASSED"));
    reporter->endRun(0, 10);

    // Main calls preSolution for a skipped pair but never beginRun or endRun.
    reporter->preSolution(nullptr);
    reporter->report(ResultKey::Validation, std::string("SKIPPED_BY_DB"));

    EXPECT_EQ(database->runStatus(0, 20), RunStatus::NotBenchmarked);
    EXPECT_EQ(count("SELECT COUNT(*) FROM run_values" + where(0, 20)), 0);
    EXPECT_EQ(count("SELECT COUNT(*) FROM iterations" + where(0, 20)), 0);
    EXPECT_EQ(count("SELECT COUNT(*) FROM runs" + where(0, 20)
                    + " AND start_time IS NULL AND validation IS NULL"),
              1);
}
