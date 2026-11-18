// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "BenchmarkRecording.hpp"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#ifdef __linux__
#include <signal.h>
#include <sys/prctl.h>
#endif

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

    class BenchmarkRecordingTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            database = std::make_shared<BenchmarkDatabase>(
                tmp.path(),
                std::vector<SolutionRecord>{{10, "SolA", "specA", "msgpack"},
                                            {20, "SolB", "specB", "msgpack"}},
                "count=1\n",
                std::vector<InitConfigRecord>{{0}, {1}});
            reader.emplace(tmp.path());
            reporter = std::make_shared<DatabaseReporter>(database);
        }

        std::unique_ptr<BenchmarkSession> openSession()
        {
            return std::make_unique<BenchmarkSession>(database, 1, 2, IndirectFailurePolicy());
        }

        // Starts a problem and a solution the way main does.
        void startPair(int64_t problem)
        {
            reporter->report(ResultKey::ProblemIndex, problem);
            reporter->preSolution(nullptr);
        }

        // Whether the latest session has not ended.
        bool sessionOpen()
        {
            SqliteStatement statement(
                *reader, "SELECT end_time IS NULL FROM sessions ORDER BY id DESC LIMIT 1");
            EXPECT_TRUE(statement.step());
            return statement.columnInt(0) != 0;
        }

        // Fails every update of the table.
        void failUpdatesOf(std::string const& table)
        {
            reader->execute("CREATE TRIGGER fail_update BEFORE UPDATE ON " + table
                            + " BEGIN SELECT RAISE(ABORT, 'injected'); END");
        }

        TempDb                             tmp;
        std::shared_ptr<BenchmarkDatabase> database;
        // A second connection that reads the state and injects failures.
        std::optional<SqliteConnection>   reader;
        std::shared_ptr<DatabaseReporter> reporter;
    };
}

TEST_F(BenchmarkRecordingTest, PairRecorderMarksUnknownThenRecordsVerdict)
{
    std::unique_ptr<BenchmarkSession> session = openSession();
    startPair(0);

    PairRecorder recorder(*reporter, 0, 10);
    EXPECT_EQ(database->runStatus(0, 10), RunStatus::Unknown);

    reporter->report(ResultKey::Validation, std::string("PASSED"));
    recorder.finish();
    EXPECT_EQ(database->runStatus(0, 10), RunStatus::Pass);
}

TEST_F(BenchmarkRecordingTest, ExceptionLeavesPairUnknownAndSessionOpen)
{
    try
    {
        BenchmarkSession session(database, 1, 2, IndirectFailurePolicy());
        startPair(0);
        PairRecorder recorder(*reporter, 0, 10);
        throw std::runtime_error("kernel fault");
    }
    catch(std::runtime_error const&)
    {
    }

    EXPECT_EQ(database->runStatus(0, 10), RunStatus::Unknown);
    EXPECT_TRUE(sessionOpen());
}

TEST_F(BenchmarkRecordingTest, SessionEndsOnScopeExit)
{
    {
        BenchmarkSession session(database, 1, 2, IndirectFailurePolicy());
        EXPECT_TRUE(sessionOpen());
    }
    EXPECT_FALSE(sessionOpen());
}

TEST_F(BenchmarkRecordingTest, SessionEndsOnEarlyReturn)
{
    auto earlyReturn = [&]() -> int32_t {
        BenchmarkSession session(database, 1, 2, IndirectFailurePolicy());
        return 3;
    };
    EXPECT_EQ(earlyReturn(), 3);
    EXPECT_FALSE(sessionOpen());
}

TEST_F(BenchmarkRecordingTest, SessionStaysOpenWhenAnExceptionUnwinds)
{
    auto throwing = [&]() {
        BenchmarkSession session(database, 1, 2, IndirectFailurePolicy());
        throw std::runtime_error("boom");
    };
    EXPECT_THROW(throwing(), std::runtime_error);
    EXPECT_TRUE(sessionOpen());
}

TEST_F(BenchmarkRecordingTest, ExplicitEndIsIdempotentAndDestructionDoesNotEndAgain)
{
    testing::internal::CaptureStderr();
    {
        BenchmarkSession session(database, 1, 2, IndirectFailurePolicy());
        EXPECT_NO_THROW(session.end());
        EXPECT_FALSE(sessionOpen());
        EXPECT_NO_THROW(session.end());
    }
    EXPECT_EQ(testing::internal::GetCapturedStderr(), "");
    EXPECT_FALSE(sessionOpen());
}

TEST_F(BenchmarkRecordingTest, FailingEndThrowsExplicitlyAndIsSwallowedByDestructor)
{
    {
        BenchmarkSession session(database, 1, 2, IndirectFailurePolicy());
        EXPECT_NO_THROW(session.end());
    }
    EXPECT_FALSE(sessionOpen());

    failUpdatesOf("sessions");
    // The destructor tries to end the session again and reports the failure.
    testing::internal::CaptureStderr();
    {
        BenchmarkSession session(database, 1, 2, IndirectFailurePolicy());
        EXPECT_THROW(session.end(), SqliteError);
    }
    std::string const destructorMessage = testing::internal::GetCapturedStderr();
    EXPECT_NE(destructorMessage.find("failed to end the benchmark session"), std::string::npos)
        << destructorMessage;
    EXPECT_TRUE(sessionOpen());

    testing::internal::CaptureStderr();
    EXPECT_NO_THROW({ BenchmarkSession session(database, 1, 2, IndirectFailurePolicy()); });
    std::string const message = testing::internal::GetCapturedStderr();
    EXPECT_NE(message.find("failed to end the benchmark session"), std::string::npos) << message;
    EXPECT_TRUE(sessionOpen());
}

TEST_F(BenchmarkRecordingTest, ClassifyPairWithoutDatabaseIsDisabled)
{
    EXPECT_EQ(classifyPair(nullptr, 0, 10), PairRecording::Disabled);
}

// Only a pair that is not benchmarked is recorded; the database already holds every other status.
TEST_F(BenchmarkRecordingTest, ClassifyPairRecordsOnlyNotBenchmarkedPairs)
{
    // A session that dies with (0, 10) running, then a session that marks (1, 10).
    database->beginSession(1, 2, IndirectFailurePolicy());
    database->beginRun(0, 10);
    database->beginSession(1, 3, IndirectFailurePolicy());
    database->beginRun(0, 20);
    database->endRun(0, 20, RunStatus::Pass, "PASSED", {}, {});

    EXPECT_EQ(database->runStatus(0, 10), RunStatus::Unknown);
    EXPECT_EQ(classifyPair(database.get(), 0, 10), PairRecording::SkippedByDatabase);
    EXPECT_EQ(database->runStatus(1, 10), RunStatus::IndirectlyFailed);
    EXPECT_EQ(classifyPair(database.get(), 1, 10), PairRecording::SkippedByDatabase);
    EXPECT_EQ(database->runStatus(0, 20), RunStatus::Pass);
    EXPECT_EQ(classifyPair(database.get(), 0, 20), PairRecording::SkippedByDatabase);
    EXPECT_EQ(database->runStatus(1, 20), RunStatus::NotBenchmarked);
    EXPECT_EQ(classifyPair(database.get(), 1, 20), PairRecording::Record);
}

TEST_F(BenchmarkRecordingTest, ClassifyPairSkipsFailuresAndIndirectVerificationFailures)
{
    database->beginSession(1, 2, IndirectFailurePolicy());
    database->beginRun(0, 20);
    database->endRun(0, 20, RunStatus::Failure, "FAILED", {}, {});

    EXPECT_EQ(database->runStatus(0, 20), RunStatus::Failure);
    EXPECT_EQ(classifyPair(database.get(), 0, 20), PairRecording::SkippedByDatabase);
    EXPECT_EQ(database->runStatus(1, 20), RunStatus::IndirectVerificationFailure);
    EXPECT_EQ(classifyPair(database.get(), 1, 20), PairRecording::SkippedByDatabase);
    EXPECT_EQ(database->runStatus(0, 10), RunStatus::NotBenchmarked);
    EXPECT_EQ(classifyPair(database.get(), 0, 10), PairRecording::Record);
}

TEST(BenchmarkRecordingCatch, ReturnsTheClientValue)
{
    EXPECT_EQ(runCatchingBenchmarkFailures([]() { return 5; }), 5);
}

TEST(BenchmarkRecordingCatch, DeviceLostReturnsItsExitCodeSilently)
{
    testing::internal::CaptureStderr();
    int32_t const result
        = runCatchingBenchmarkFailures([]() -> int32_t { throw DeviceLostError(7); });
    EXPECT_EQ(result, 7);
    EXPECT_EQ(testing::internal::GetCapturedStderr(), "");
}

TEST(BenchmarkRecordingCatch, DatabaseErrorsReturnOne)
{
    testing::internal::CaptureStderr();
    int32_t result = runCatchingBenchmarkFailures(
        []() -> int32_t { throw BenchmarkDatabaseError("inconsistent"); });
    std::string message = testing::internal::GetCapturedStderr();
    EXPECT_EQ(result, 1);
    EXPECT_NE(message.find("benchmark database error: inconsistent"), std::string::npos);

    testing::internal::CaptureStderr();
    result  = runCatchingBenchmarkFailures([]() -> int32_t { throw SqliteError("disk full"); });
    message = testing::internal::GetCapturedStderr();
    EXPECT_EQ(result, 1);
    EXPECT_NE(message.find("benchmark database error: disk full"), std::string::npos);
}

TEST(BenchmarkRecordingCatch, OtherExceptionsPropagate)
{
    EXPECT_THROW(runCatchingBenchmarkFailures([]() -> int32_t { throw std::logic_error("bug"); }),
                 std::logic_error);
    EXPECT_THROW(
        runCatchingBenchmarkFailures([]() -> int32_t { throw std::runtime_error("other"); }),
        std::runtime_error);
}

TEST_F(BenchmarkRecordingTest, FailingRunUpdateBecomesDiagnosticAndExitCodeOne)
{
    BenchmarkSession session(database, 1, 2, IndirectFailurePolicy());
    // The PairRecorder constructor fails: it commits Unknown with an update of runs.
    failUpdatesOf("runs");
    startPair(0);

    testing::internal::CaptureStderr();
    int32_t const     result  = runCatchingBenchmarkFailures([&]() -> int32_t {
        PairRecorder recorder(*reporter, 0, 10);
        return 0;
    });
    std::string const message = testing::internal::GetCapturedStderr();
    EXPECT_EQ(result, 1);
    EXPECT_NE(message.find("benchmark database error"), std::string::npos) << message;
}

TEST_F(BenchmarkRecordingTest, FailingSessionEndBecomesNonZeroExit)
{
    failUpdatesOf("sessions");

    testing::internal::CaptureStderr();
    int32_t const     result  = runCatchingBenchmarkFailures([&]() -> int32_t {
        BenchmarkSession session(database, 1, 2, IndirectFailurePolicy());
        session.end();
        return 0;
    });
    std::string const message = testing::internal::GetCapturedStderr();
    EXPECT_NE(result, 0);
    EXPECT_NE(message.find("benchmark database error"), std::string::npos) << message;
    EXPECT_TRUE(sessionOpen());
}

#ifdef __linux__
namespace
{
    // Restores the parent-death signal that the test binary had on entry.
    class PdeathsigGuard
    {
    public:
        PdeathsigGuard()
        {
            prctl(PR_GET_PDEATHSIG, &m_signal);
        }

        ~PdeathsigGuard()
        {
            prctl(PR_SET_PDEATHSIG, m_signal);
        }

        PdeathsigGuard(PdeathsigGuard const&)            = delete;
        PdeathsigGuard& operator=(PdeathsigGuard const&) = delete;

    private:
        int32_t m_signal = 0;
    };
}

TEST(DieWithParentTest, RequestsSigkillOnParentDeath)
{
    // Do not let the test binary die with its own parent.
    PdeathsigGuard guard;

    EXPECT_TRUE(dieWithParent());

    int32_t signalNumber = 0;
    ASSERT_EQ(prctl(PR_GET_PDEATHSIG, &signalNumber), 0);
    EXPECT_EQ(signalNumber, SIGKILL);
}
#endif

// ---------------------------------------------------------------------------
// Pairs the iterator does not select
// ---------------------------------------------------------------------------

namespace
{
    // Replays what main does for one problem: the selected solutions run and are recorded, then the
    // problem finishes and its other pairs become NotSelected.
    class NotSelectedRecordingTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            openDatabase();
        }

        void openDatabase()
        {
            reporter.reset();
            database.reset();
            database = std::make_shared<BenchmarkDatabase>(
                tmp.path(),
                std::vector<SolutionRecord>{{10, "SolA", "specA", "msgpack"},
                                            {20, "SolB", "specB", "msgpack"},
                                            {30, "SolC", "specC", "msgpack"},
                                            {40, "SolD", "specD", "msgpack"}},
                "count=1\n",
                std::vector<InitConfigRecord>{{0}, {1}, {2}});
            reporter = std::make_shared<DatabaseReporter>(database);
        }

        std::unique_ptr<BenchmarkSession> openSession()
        {
            return std::make_unique<BenchmarkSession>(database, 1, 2, IndirectFailurePolicy());
        }

        // Runs one solution the way main does and returns what the database decided.
        PairRecording visit(int64_t problem, int64_t solution, bool finish = true)
        {
            PairRecording recording = classifyPair(database.get(), problem, solution);
            if(recording != PairRecording::Record)
                return recording;
            reporter->report(ResultKey::ProblemIndex, problem);
            reporter->preSolution(nullptr);
            PairRecorder recorder(*reporter, problem, solution);
            if(finish)
            {
                reporter->report(ResultKey::Validation, std::string("PASSED"));
                recorder.finish();
            }
            return recording;
        }

        void visitAll(int64_t problem, std::vector<int64_t> const& solutions)
        {
            for(int64_t solution : solutions)
                visit(problem, solution);
        }

        void expectStatuses(int64_t problem, std::vector<RunStatus> const& statuses)
        {
            ASSERT_EQ(statuses.size(), solutionIndices.size());
            for(size_t i = 0; i < statuses.size(); i++)
                EXPECT_EQ(database->runStatus(problem, solutionIndices[i]), statuses[i])
                    << "problem " << problem << " solution " << solutionIndices[i];
        }

        static constexpr RunStatus pass = RunStatus::Pass;
        static constexpr RunStatus notSelected = RunStatus::NotSelected;
        static constexpr RunStatus todo = RunStatus::NotBenchmarked;

        std::vector<int64_t>               solutionIndices{10, 20, 30, 40};
        TempDb                             tmp;
        std::shared_ptr<BenchmarkDatabase> database;
        std::shared_ptr<DatabaseReporter>  reporter;
    };
}

TEST_F(NotSelectedRecordingTest, FinishedProblemMarksTheUnselectedPairs)
{
    std::unique_ptr<BenchmarkSession> session = openSession();
    visitAll(0, {10, 30});
    expectStatuses(0, {pass, todo, pass, todo});

    database->markNotSelected(0);
    expectStatuses(0, {pass, notSelected, pass, notSelected});
    expectStatuses(1, {todo, todo, todo, todo});
    EXPECT_FALSE(database->hasPendingRuns(0));
    EXPECT_TRUE(database->hasPendingRuns(1));
}

TEST_F(NotSelectedRecordingTest, ProblemCutShortKeepsItsPairsNotBenchmarkedAndResumes)
{
    // Unwinding leaves the session open, as a crash does; a normal scope exit would end it.
    try
    {
        BenchmarkSession session(database, 1, 2, IndirectFailurePolicy());
        visit(0, 10);
        throw std::runtime_error("the client dies before the problem ends");
    }
    catch(std::runtime_error const&)
    {
    }
    expectStatuses(0, {pass, todo, todo, todo});

    // The database is reopened and the iterator selects the same subset again.
    openDatabase();
    std::unique_ptr<BenchmarkSession> session = openSession();
    EXPECT_EQ(classifyPair(database.get(), 0, 10), PairRecording::SkippedByDatabase);
    EXPECT_EQ(visit(0, 10), PairRecording::SkippedByDatabase);
    EXPECT_EQ(visit(0, 30), PairRecording::Record);
    database->markNotSelected(0);
    expectStatuses(0, {pass, notSelected, pass, notSelected});
}

TEST_F(NotSelectedRecordingTest, ResumeNeitherRerunsNorRevivesNotSelectedPairs)
{
    {
        std::unique_ptr<BenchmarkSession> session = openSession();
        visitAll(0, {10, 30});
        database->markNotSelected(0);
        session->end();
    }

    openDatabase();
    std::unique_ptr<BenchmarkSession> session = openSession();
    EXPECT_FALSE(database->hasPendingRuns(0));
    // Even a selection that now includes a NotSelected solution does not run it again.
    for(int64_t solution : {10, 20, 30, 40})
        EXPECT_EQ(visit(0, solution), PairRecording::SkippedByDatabase);
    database->markNotSelected(0);
    expectStatuses(0, {pass, notSelected, pass, notSelected});
}
