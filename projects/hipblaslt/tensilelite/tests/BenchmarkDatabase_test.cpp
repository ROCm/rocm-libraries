// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "BenchmarkDatabase.hpp"
#include "SqliteDatabase.hpp"

#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

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
}

static std::vector<SolutionRecord> makeSolutions()
{
    // The spec includes a NUL and a high byte: it is binary.
    return {{10, "SolA", std::string("specA\0\xff", 7), "msgpack"},
            {20, "SolB", std::string("specB"), "msgpack"}};
}

static std::string makeClientConfig()
{
    return "num-syncs-per-benchmark=2\nproblem-size=512,512,1,1024\n";
}

static std::vector<InitConfigRecord> makeConfigs()
{
    return {{0}, {1}, {2}};
}

namespace
{
    // Remembers the file so that tests can inspect it through a separate connection.
    class TestDatabase : public BenchmarkDatabase
    {
    public:
        TestDatabase(std::string const&                   path,
                     std::vector<SolutionRecord> const&   solutions,
                     std::string const&                   clientConfig,
                     std::vector<InitConfigRecord> const& initConfigs)
            : BenchmarkDatabase(path, solutions, clientConfig, initConfigs)
            , m_path(path)
        {
        }

        std::string const& path() const
        {
            return m_path;
        }

    private:
        std::string m_path;
    };
}

// Runs a statement on a separate connection, as an external tool would.
static void executeSql(std::string const& path, std::string const& sql)
{
    SqliteConnection connection(path);
    connection.execute("PRAGMA foreign_keys=ON");
    connection.execute(sql);
}

static int64_t queryInt(std::string const& path, std::string const& sql)
{
    SqliteConnection connection(path);
    SqliteStatement  statement(connection, sql);
    EXPECT_TRUE(statement.step());
    return statement.columnInt(0);
}

static std::string queryText(std::string const& path, std::string const& sql)
{
    SqliteConnection connection(path);
    SqliteStatement  statement(connection, sql);
    EXPECT_TRUE(statement.step());
    return statement.columnText(0);
}

static int64_t queryInt(TestDatabase const& db, std::string const& sql)
{
    return queryInt(db.path(), sql);
}

static std::string queryText(TestDatabase const& db, std::string const& sql)
{
    return queryText(db.path(), sql);
}

static std::string runWhere(int64_t problemIndex, int64_t solutionIndex)
{
    return " FROM runs WHERE problem_id = (SELECT id FROM init_configs WHERE problem_index = "
           + std::to_string(problemIndex)
           + ") AND solution_id = (SELECT id FROM solutions WHERE library_index = "
           + std::to_string(solutionIndex) + ")";
}

static int64_t runStatusFromSql(TestDatabase& db, int64_t problem, int64_t solution)
{
    return queryInt(db, "SELECT status" + runWhere(problem, solution));
}

TEST(BenchmarkDatabase, FreshCreatePopulateReopenMatches)
{
    TempDb tmp;
    {
        TestDatabase db(tmp.path(), makeSolutions(), makeClientConfig(), makeConfigs());
        EXPECT_EQ(queryInt(db, "SELECT COUNT(*) FROM solutions"), 2);
        EXPECT_EQ(queryInt(db, "SELECT COUNT(*) FROM init_configs"), 3);
        EXPECT_EQ(queryInt(db, "SELECT COUNT(*) FROM runs"), 6);
        EXPECT_EQ(queryInt(db, "SELECT COUNT(*) FROM runs WHERE status = 0"), 6);
        EXPECT_EQ(queryInt(db, "PRAGMA user_version"), 1);
        EXPECT_EQ(queryText(db, "PRAGMA journal_mode"), "wal");
    }
    TestDatabase db(tmp.path(), makeSolutions(), makeClientConfig(), makeConfigs());
    EXPECT_EQ(queryInt(db, "SELECT COUNT(*) FROM runs WHERE status = 0"), 6);
    EXPECT_EQ(queryInt(db, "SELECT COUNT(*) FROM solutions"), 2);
}

TEST(BenchmarkDatabase, MismatchIsRejected)
{
    TempDb tmp;
    {
        TestDatabase db(tmp.path(), makeSolutions(), makeClientConfig(), makeConfigs());
    }

    std::vector<SolutionRecord> renamed = makeSolutions();
    renamed[1].name                     = "Other";
    EXPECT_THROW(BenchmarkDatabase(tmp.path(), renamed, makeClientConfig(), makeConfigs()),
                 BenchmarkDatabaseMismatch);
    try
    {
        TestDatabase db(tmp.path(), renamed, makeClientConfig(), makeConfigs());
        FAIL() << "expected a mismatch";
    }
    catch(BenchmarkDatabaseMismatch const& error)
    {
        EXPECT_NE(std::string(error.what()).find("solution 20 is named 'SolB'"), std::string::npos)
            << error.what();
    }

    std::vector<SolutionRecord> specByte = makeSolutions();
    specByte[0].spec[5]                  = 'x';
    EXPECT_THROW(BenchmarkDatabase(tmp.path(), specByte, makeClientConfig(), makeConfigs()),
                 BenchmarkDatabaseMismatch);

    std::vector<SolutionRecord> format = makeSolutions();
    format[1].specFormat               = "yaml";
    EXPECT_THROW(BenchmarkDatabase(tmp.path(), format, makeClientConfig(), makeConfigs()),
                 BenchmarkDatabaseMismatch);

    EXPECT_THROW(BenchmarkDatabase(tmp.path(),
                                   makeSolutions(),
                                   "num-syncs-per-benchmark=3\nproblem-size=512,512,1,1024\n",
                                   makeConfigs()),
                 BenchmarkDatabaseMismatch);

    std::vector<InitConfigRecord> otherIndex = makeConfigs();
    otherIndex[2].problemIndex               = 7;
    EXPECT_THROW(BenchmarkDatabase(tmp.path(), makeSolutions(), makeClientConfig(), otherIndex),
                 BenchmarkDatabaseMismatch);

    std::vector<SolutionRecord> fewerSolutions = makeSolutions();
    fewerSolutions.pop_back();
    EXPECT_THROW(BenchmarkDatabase(tmp.path(), fewerSolutions, makeClientConfig(), makeConfigs()),
                 BenchmarkDatabaseMismatch);

    std::vector<InitConfigRecord> moreConfigs = makeConfigs();
    moreConfigs.push_back({3});
    EXPECT_THROW(BenchmarkDatabase(tmp.path(), makeSolutions(), makeClientConfig(), moreConfigs),
                 BenchmarkDatabaseMismatch);
}

TEST(BenchmarkDatabase, SpecsAndConfigRoundTrip)
{
    TempDb       tmp;
    TestDatabase db(tmp.path(), makeSolutions(), makeClientConfig(), makeConfigs());

    SqliteConnection reader(db.path());
    SqliteStatement  solutions(reader,
                               "SELECT spec, spec_format FROM solutions ORDER BY library_index");
    ASSERT_TRUE(solutions.step());
    EXPECT_EQ(solutions.columnBlob(0), makeSolutions()[0].spec);
    EXPECT_EQ(solutions.columnText(1), "msgpack");

    EXPECT_EQ(queryText(db, "SELECT ini FROM client_configs"), makeClientConfig());
    EXPECT_EQ(queryInt(db, "SELECT COUNT(*) FROM client_configs"), 1);
    EXPECT_EQ(queryInt(db,
                       "SELECT COUNT(*) FROM init_configs i JOIN client_configs c "
                       "ON i.config_id = c.id"),
              3);
}

TEST(BenchmarkDatabase, SchemaVersionMismatchIsRejected)
{
    TempDb tmp;
    {
        TestDatabase db(tmp.path(), makeSolutions(), makeClientConfig(), makeConfigs());
    }
    executeSql(tmp.path(), "PRAGMA user_version = 2");
    try
    {
        TestDatabase db(tmp.path(), makeSolutions(), makeClientConfig(), makeConfigs());
        FAIL() << "expected a mismatch";
    }
    catch(BenchmarkDatabaseMismatch const& error)
    {
        EXPECT_NE(std::string(error.what()).find("schema version is 2"), std::string::npos)
            << error.what();
    }
}

static std::string readBytes(std::string const& path)
{
    std::ifstream file(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

static void expectNoJournalFiles(std::string const& path)
{
    EXPECT_FALSE(std::filesystem::exists(path + "-wal"));
    EXPECT_FALSE(std::filesystem::exists(path + "-shm"));
}

static void expectMismatchWithoutModifying(std::string const& path, std::string const& before)
{
    EXPECT_THROW(BenchmarkDatabase(path, makeSolutions(), makeClientConfig(), makeConfigs()),
                 BenchmarkDatabaseMismatch);
    EXPECT_EQ(readBytes(path), before);
    expectNoJournalFiles(path);
}

TEST(BenchmarkDatabase, ForeignFileIsRejectedAndUntouched)
{
    TempDb tmp;
    {
        SqliteConnection raw(tmp.path());
        raw.execute("CREATE TABLE foo(x)");
    }
    std::string before = readBytes(tmp.path());
    ASSERT_FALSE(before.empty());
    expectMismatchWithoutModifying(tmp.path(), before);
}

TEST(BenchmarkDatabase, PopulatedDatabaseWithoutVersionIsRejectedAndUntouched)
{
    TempDb tmp;
    {
        TestDatabase db(tmp.path(), makeSolutions(), makeClientConfig(), makeConfigs());
    }
    executeSql(tmp.path(), "PRAGMA user_version = 0");
    // Closing the last connection checkpointed the WAL into the file.
    expectNoJournalFiles(tmp.path());
    expectMismatchWithoutModifying(tmp.path(), readBytes(tmp.path()));
}

TEST(BenchmarkDatabase, EmptyExistingFileIsPopulated)
{
    TempDb tmp;
    std::ofstream(tmp.path()).close();
    TestDatabase db(tmp.path(), makeSolutions(), makeClientConfig(), makeConfigs());
    EXPECT_EQ(queryInt(db, "SELECT COUNT(*) FROM runs"), 6);
    EXPECT_EQ(queryInt(db, "PRAGMA user_version"), 1);
}

TEST(BenchmarkDatabase, DatabaseEmptiedOfTablesIsPopulated)
{
    TempDb tmp;
    {
        SqliteConnection raw(tmp.path());
        raw.execute("CREATE TABLE foo(x); DROP TABLE foo");
    }
    TestDatabase db(tmp.path(), makeSolutions(), makeClientConfig(), makeConfigs());
    EXPECT_EQ(queryInt(db, "SELECT COUNT(*) FROM runs"), 6);
    EXPECT_EQ(queryInt(db, "PRAGMA user_version"), 1);
}

TEST(BenchmarkDatabase, FailedCreationLeavesNoTables)
{
    TempDb tmp;
    // Duplicate problem indices violate a UNIQUE constraint after the schema was created.
    EXPECT_THROW(BenchmarkDatabase(tmp.path(), makeSolutions(), makeClientConfig(), {{0}, {0}}),
                 SqliteError);
    {
        SqliteConnection raw(tmp.path());
        SqliteStatement  tables(raw, "SELECT COUNT(*) FROM sqlite_master");
        ASSERT_TRUE(tables.step());
        EXPECT_EQ(tables.columnInt(0), 0);
        SqliteStatement version(raw, "PRAGMA user_version");
        ASSERT_TRUE(version.step());
        EXPECT_EQ(version.columnInt(0), 0);
    }

    TestDatabase db(tmp.path(), makeSolutions(), makeClientConfig(), makeConfigs());
    EXPECT_EQ(queryInt(db, "SELECT COUNT(*) FROM runs"), 6);
    EXPECT_EQ(queryInt(db, "PRAGMA user_version"), 1);
}

TEST(BenchmarkDatabase, StateTransitions)
{
    TempDb       tmp;
    TestDatabase db(tmp.path(), makeSolutions(), makeClientConfig(), makeConfigs());

    EXPECT_THROW(db.beginRun(0, 10), BenchmarkDatabaseError);

    int64_t session = db.beginSession(1234, 7);
    db.beginRun(0, 10);
    EXPECT_EQ(db.runStatus(0, 10), RunStatus::Unknown);
    EXPECT_EQ(queryInt(db, "SELECT session_id" + runWhere(0, 10)), session);
    EXPECT_GT(queryInt(db, "SELECT CAST(start_time AS INTEGER)" + runWhere(0, 10)), 1000000000);
    EXPECT_EQ(queryInt(db, "SELECT end_time IS NULL" + runWhere(0, 10)), 1);

    EXPECT_THROW(db.beginRun(0, 10), BenchmarkDatabaseError);
    EXPECT_THROW(db.endRun(0, 20, RunStatus::Pass, "PASSED", {}, {}), BenchmarkDatabaseError);
    EXPECT_THROW(db.endRun(0, 10, RunStatus::Unknown, "x", {}, {}), BenchmarkDatabaseError);
    EXPECT_THROW(db.endRun(0, 10, RunStatus::NotBenchmarked, "x", {}, {}), BenchmarkDatabaseError);
    EXPECT_EQ(db.runStatus(0, 10), RunStatus::Unknown);

    db.endRun(0, 10, RunStatus::Pass, "PASSED", {}, {});
    EXPECT_EQ(db.runStatus(0, 10), RunStatus::Pass);
    EXPECT_EQ(queryText(db, "SELECT validation" + runWhere(0, 10)), "PASSED");
    EXPECT_EQ(queryInt(db, "SELECT end_time >= start_time" + runWhere(0, 10)), 1);
    EXPECT_THROW(db.endRun(0, 10, RunStatus::Pass, "PASSED", {}, {}), BenchmarkDatabaseError);

    db.beginRun(1, 20);
    db.endRun(1, 20, RunStatus::Failure, "FAILED", {}, {});
    EXPECT_EQ(db.runStatus(1, 20), RunStatus::Failure);
    EXPECT_EQ(queryText(db, "SELECT validation" + runWhere(1, 20)), "FAILED");
    // A verification failure spreads to the other pending pairs of the solution.
    EXPECT_EQ(db.runStatus(2, 20), RunStatus::IndirectVerificationFailure);

    EXPECT_THROW(db.beginRun(99, 10), BenchmarkDatabaseError);
    EXPECT_THROW(db.beginRun(0, 99), BenchmarkDatabaseError);
    EXPECT_THROW(db.runStatus(99, 10), BenchmarkDatabaseError);

    // A failed endRun must not leave partial rows behind.
    db.beginRun(2, 10);
    EXPECT_THROW(db.endRun(2, 10, RunStatus::Pass, "PASSED", {{"k", "v"}, {"k", "w"}}, {}),
                 SqliteError);
    EXPECT_EQ(db.runStatus(2, 10), RunStatus::Unknown);
    EXPECT_EQ(queryInt(db, "SELECT COUNT(*) FROM run_values"), 0);
}

TEST(BenchmarkDatabase, SessionWithoutEnd)
{
    TempDb tmp;
    {
        TestDatabase db(tmp.path(), makeSolutions(), makeClientConfig(), makeConfigs());
        EXPECT_THROW(db.endSession(), BenchmarkDatabaseError);
        db.beginSession(4321, 987654321);
    }
    TestDatabase db(tmp.path(), makeSolutions(), makeClientConfig(), makeConfigs());
    EXPECT_EQ(queryInt(db, "SELECT COUNT(*) FROM sessions"), 1);
    EXPECT_EQ(queryInt(db, "SELECT end_time IS NULL FROM sessions"), 1);
    EXPECT_EQ(queryInt(db, "SELECT init_seed FROM sessions"), 987654321);
    EXPECT_EQ(queryInt(db, "SELECT pid FROM sessions"), 4321);

    db.beginSession(4322, 1);
    db.endSession();
    EXPECT_EQ(queryInt(db, "SELECT COUNT(*) FROM sessions"), 2);
    EXPECT_EQ(queryInt(db, "SELECT COUNT(*) FROM sessions WHERE end_time IS NOT NULL"), 1);
    EXPECT_EQ(queryInt(db, "SELECT COUNT(*) FROM sessions WHERE end_time IS NULL"), 1);
}

TEST(BenchmarkDatabase, PendingQueries)
{
    TempDb       tmp;
    TestDatabase db(tmp.path(), makeSolutions(), makeClientConfig(), makeConfigs());
    db.beginSession(1, 1);

    EXPECT_TRUE(db.isPending(0, 10));
    EXPECT_TRUE(db.hasPendingRuns(0));
    EXPECT_THROW(db.isPending(99, 10), BenchmarkDatabaseError);

    db.beginRun(0, 10);
    EXPECT_FALSE(db.isPending(0, 10));
    EXPECT_TRUE(db.hasPendingRuns(0));

    db.endRun(0, 10, RunStatus::Pass, "PASSED", {}, {});
    EXPECT_FALSE(db.isPending(0, 10));
    EXPECT_TRUE(db.hasPendingRuns(0));

    db.beginRun(0, 20);
    EXPECT_FALSE(db.hasPendingRuns(0));
    db.endRun(0, 20, RunStatus::Failure, "DID_NOT_SATISFY_ASSERTS", {}, {});
    EXPECT_FALSE(db.isPending(0, 20));
    EXPECT_FALSE(db.hasPendingRuns(0));

    EXPECT_TRUE(db.hasPendingRuns(1));
    EXPECT_TRUE(db.isPending(1, 10));
}

TEST(BenchmarkDatabase, ValuesAndIterationsRoundTrip)
{
    TempDb       tmp;
    TestDatabase db(tmp.path(), makeSolutions(), makeClientConfig(), makeConfigs());
    db.beginSession(1, 1);

    EXPECT_EQ(queryInt(db, "SELECT COUNT(*) FROM meta"), 0);
    db.setMeta("performance_metric", "DeviceEfficiency");
    db.setMeta("performance_metric", "CUEfficiency");
    EXPECT_EQ(queryText(db, "SELECT value FROM meta WHERE key = 'performance_metric'"),
              "CUEfficiency");
    EXPECT_EQ(queryInt(db, "SELECT COUNT(*) FROM meta"), 1);

    db.setProblemValues(1, {{"size-m", "128"}, {"alpha", "1.5e+06"}});
    db.setProblemValues(1, {{"size-m", "256"}});
    EXPECT_EQ(queryText(db, "SELECT value FROM problem_values WHERE key = 'size-m'"), "256");
    EXPECT_EQ(queryText(db, "SELECT value FROM problem_values WHERE key = 'alpha'"), "1.5e+06");

    db.beginRun(1, 20);
    db.endRun(1,
              20,
              RunStatus::Pass,
              "PASSED",
              {{"speed", "-nan"}, {"time", "1.5e+06"}, {"empty", ""}},
              {{0, 12.375, 10}, {1, 0.1, 20}, {2, 1234567.890625, 30}});

    EXPECT_EQ(queryText(db, "SELECT value FROM run_values WHERE key = 'speed'"), "-nan");
    EXPECT_EQ(queryText(db, "SELECT value FROM run_values WHERE key = 'time'"), "1.5e+06");
    EXPECT_EQ(queryText(db, "SELECT value FROM run_values WHERE key = 'empty'"), "");
    EXPECT_EQ(queryInt(db, "SELECT COUNT(*) FROM run_values"), 3);

    SqliteConnection             reader(db.path());
    SqliteStatement              statement(reader,
                                           "SELECT sync_index, time_us, enqueues FROM iterations "
                                           "ORDER BY sync_index");
    std::vector<IterationRecord> expected
        = {{0, 12.375, 10}, {1, 0.1, 20}, {2, 1234567.890625, 30}};
    for(IterationRecord const& record : expected)
    {
        ASSERT_TRUE(statement.step());
        EXPECT_EQ(statement.columnInt(0), record.syncIndex);
        EXPECT_DOUBLE_EQ(statement.columnDouble(1), record.timeUs);
        EXPECT_EQ(statement.columnInt(2), record.enqueues);
    }
    EXPECT_FALSE(statement.step());
}

TEST(BenchmarkDatabase, AbandonedRunSurvivesProcessDeath)
{
    TempDb tmp;
    {
        TestDatabase db(tmp.path(), makeSolutions(), makeClientConfig(), makeConfigs());
    }

    pid_t child = fork();
    ASSERT_GE(child, 0);
    if(child == 0)
    {
        // No sqlite handle crosses the fork; the child dies mid-run on purpose. It must never
        // return into the gtest runner, so every path leaves through _exit.
        try
        {
            TestDatabase db(tmp.path(), makeSolutions(), makeClientConfig(), makeConfigs());
            db.beginSession(getpid(), 5);
            db.beginRun(1, 20);
            raise(SIGKILL);
        }
        catch(...)
        {
            _exit(2);
        }
        _exit(1);
    }

    int32_t status = 0;
    ASSERT_EQ(waitpid(child, &status, 0), child);
    ASSERT_TRUE(WIFSIGNALED(status));
    ASSERT_EQ(WTERMSIG(status), SIGKILL);

    TestDatabase db(tmp.path(), makeSolutions(), makeClientConfig(), makeConfigs());
    EXPECT_EQ(db.runStatus(1, 20), RunStatus::Unknown);
    EXPECT_EQ(queryInt(db, "SELECT start_time IS NOT NULL" + runWhere(1, 20)), 1);
    EXPECT_EQ(queryInt(db, "SELECT end_time IS NULL" + runWhere(1, 20)), 1);
    EXPECT_EQ(queryInt(db, "SELECT end_time IS NULL FROM sessions"), 1);
    EXPECT_FALSE(db.isPending(1, 20));
    EXPECT_EQ(queryInt(db, "SELECT COUNT(*) FROM runs WHERE status = 0"), 5);
    EXPECT_EQ(runStatusFromSql(db, 0, 10), 0);
}

// ---------------------------------------------------------------------------
// Indirect failures
// ---------------------------------------------------------------------------

static int64_t causedByProblemIndex(TestDatabase& db, int64_t problem, int64_t solution)
{
    return queryInt(db,
                    "SELECT IFNULL((SELECT problem_index FROM init_configs WHERE id = "
                    "caused_by_problem_id), -1)"
                        + runWhere(problem, solution));
}

static int64_t initConfigId(TestDatabase& db, int64_t problem)
{
    return queryInt(db,
                    "SELECT id FROM init_configs WHERE problem_index = " + std::to_string(problem));
}

static int64_t causedByTableId(TestDatabase& db, int64_t problem, int64_t solution)
{
    return queryInt(db, "SELECT caused_by_problem_id" + runWhere(problem, solution));
}

static void expectUntouched(TestDatabase& db, int64_t problem, int64_t solution)
{
    EXPECT_EQ(db.runStatus(problem, solution), RunStatus::NotBenchmarked);
    EXPECT_EQ(queryInt(db,
                       "SELECT session_id IS NULL AND start_time IS NULL AND end_time IS NULL "
                       "AND validation IS NULL AND caused_by_problem_id IS NULL"
                           + runWhere(problem, solution)),
              1);
}

static void expectMarked(TestDatabase&      db,
                         int64_t            problem,
                         int64_t            solution,
                         RunStatus          status,
                         std::string const& validation,
                         int64_t            session,
                         int64_t            causedBy)
{
    EXPECT_EQ(db.runStatus(problem, solution), status);
    EXPECT_EQ(queryText(db, "SELECT validation" + runWhere(problem, solution)), validation);
    EXPECT_EQ(queryInt(db, "SELECT session_id" + runWhere(problem, solution)), session);
    EXPECT_EQ(causedByProblemIndex(db, problem, solution), causedBy);
    EXPECT_EQ(queryInt(db,
                       "SELECT start_time IS NULL AND end_time IS NULL"
                           + runWhere(problem, solution)),
              1);
    EXPECT_FALSE(db.isPending(problem, solution));
}

TEST(BenchmarkDatabase, CrashMarksRemainingPairsOfTheSolutionAtSessionStart)
{
    TempDb       tmp;
    TestDatabase db(tmp.path(), makeSolutions(), makeClientConfig(), makeConfigs());
    db.beginSession(1, 1);
    db.beginRun(0, 10);
    db.endRun(0, 10, RunStatus::Pass, "PASSED", {}, {});
    db.beginRun(1, 10);
    // The session dies here, leaving the pair Unknown.

    int64_t session = db.beginSession(2, 2);
    expectMarked(
        db, 2, 10, RunStatus::IndirectlyFailed, "INDIRECTLY_FAILED", session, 1);
    EXPECT_EQ(causedByTableId(db, 2, 10), initConfigId(db, 1));
    EXPECT_EQ(db.runStatus(1, 10), RunStatus::Unknown);
    EXPECT_EQ(db.runStatus(0, 10), RunStatus::Pass);
    for(int64_t problem : {0, 1, 2})
        expectUntouched(db, problem, 20);

    EXPECT_TRUE(db.hasPendingRuns(2));
    db.beginRun(2, 20);
    db.endRun(2, 20, RunStatus::Pass, "PASSED", {}, {});
    EXPECT_FALSE(db.hasPendingRuns(2));
}

TEST(BenchmarkDatabase, CrashCausedByIsTheSmallestUnknownProblem)
{
    TempDb       tmp;
    TestDatabase db(tmp.path(), makeSolutions(), makeClientConfig(), makeConfigs());
    db.beginSession(1, 1);
    db.beginRun(2, 10);
    db.beginRun(1, 10);

    int64_t session = db.beginSession(2, 2);
    expectMarked(
        db, 0, 10, RunStatus::IndirectlyFailed, "INDIRECTLY_FAILED", session, 1);
}

TEST(BenchmarkDatabase, VerificationFailureMarksRemainingPairsImmediately)
{
    TempDb       tmp;
    TestDatabase db(tmp.path(), makeSolutions(), makeClientConfig(), makeConfigs());
    int64_t      session = db.beginSession(1, 1);
    db.beginRun(0, 10);
    db.endRun(0, 10, RunStatus::Pass, "PASSED", {}, {});
    db.beginRun(1, 10);
    db.endRun(1, 10, RunStatus::Failure, "FAILED", {}, {});

    EXPECT_EQ(db.runStatus(0, 10), RunStatus::Pass);
    EXPECT_EQ(db.runStatus(1, 10), RunStatus::Failure);
    EXPECT_EQ(queryText(db, "SELECT validation" + runWhere(1, 10)), "FAILED");
    expectMarked(db,
                 2,
                 10,
                 RunStatus::IndirectVerificationFailure,
                 "INDIRECT_VERIFICATION_FAILURE",
                 session,
                 1);
    EXPECT_EQ(causedByTableId(db, 2, 10), initConfigId(db, 1));
    for(int64_t problem : {0, 1, 2})
        expectUntouched(db, problem, 20);
}

TEST(BenchmarkDatabase, OnlyTheFailedVerdictSpreads)
{
    for(std::string const& validation :
        {"DID_NOT_SATISFY_ASSERTS", "WRONG_HARDWARE", "UNSUPPORTED_XCD_TOPOLOGY", "", "INVALID"})
    {
        TempDb       tmp;
        TestDatabase db(tmp.path(), makeSolutions(), makeClientConfig(), makeConfigs());
        db.beginSession(1, 1);
        db.beginRun(0, 10);
        db.endRun(0, 10, RunStatus::Failure, validation, {}, {});
        expectUntouched(db, 1, 10);
        expectUntouched(db, 2, 10);
    }

    TempDb       tmp;
    TestDatabase db(tmp.path(), makeSolutions(), makeClientConfig(), makeConfigs());
    db.beginSession(1, 1);
    db.beginRun(0, 10);
    db.endRun(0, 10, RunStatus::Pass, "FAILED", {}, {});
    expectUntouched(db, 1, 10);
}

TEST(BenchmarkDatabase, VerificationFailureDoesNotSpreadWhenDisabled)
{
    TempDb       tmp;
    TestDatabase db(tmp.path(), makeSolutions(), makeClientConfig(), makeConfigs());
    db.beginSession(1, 1, IndirectFailurePolicy{true, false});
    db.beginRun(0, 10);
    db.endRun(0, 10, RunStatus::Failure, "FAILED", {}, {});
    expectUntouched(db, 1, 10);
}

TEST(BenchmarkDatabase, SessionAfterADeadOneDoesNotSpreadVerificationFailures)
{
    TempDb       tmp;
    TestDatabase db(tmp.path(), makeSolutions(), makeClientConfig(), makeConfigs());
    db.beginSession(1, 1);
    db.beginRun(0, 20);
    // Session 1 dies; session 2 starts tainted but still propagates the crash.
    int64_t second = db.beginSession(2, 2);
    expectMarked(
        db, 1, 20, RunStatus::IndirectlyFailed, "INDIRECTLY_FAILED", second, 0);

    db.beginRun(0, 10);
    db.endRun(0, 10, RunStatus::Failure, "FAILED", {}, {});
    expectUntouched(db, 1, 10);
    db.endSession();

    // A session after a cleanly ended one is trusted again.
    int64_t third = db.beginSession(3, 3);
    db.beginRun(1, 10);
    db.endRun(1, 10, RunStatus::Failure, "FAILED", {}, {});
    expectMarked(db,
                 2,
                 10,
                 RunStatus::IndirectVerificationFailure,
                 "INDIRECT_VERIFICATION_FAILURE",
                 third,
                 1);
}

TEST(BenchmarkDatabase, DisabledPoliciesResetMarkedPairs)
{
    TempDb       tmp;
    TestDatabase db(tmp.path(), makeSolutions(), makeClientConfig(), makeConfigs());
    db.beginSession(1, 1);
    db.beginRun(0, 10);
    db.endRun(0, 10, RunStatus::Failure, "FAILED", {}, {});
    db.endSession();
    db.beginSession(2, 2);
    db.beginRun(0, 20);
    // Session 2 dies with solution 20 crashed; session 3 marks it.
    int64_t third = db.beginSession(3, 3);
    expectMarked(
        db, 1, 20, RunStatus::IndirectlyFailed, "INDIRECTLY_FAILED", third, 0);
    EXPECT_EQ(db.runStatus(1, 10), RunStatus::IndirectVerificationFailure);
    db.endSession();

    // Turning crashes off resets only the crash marks, and nothing is marked again.
    db.beginSession(4, 4, IndirectFailurePolicy{false, true});
    expectUntouched(db, 1, 20);
    expectUntouched(db, 2, 20);
    EXPECT_EQ(db.runStatus(1, 10), RunStatus::IndirectVerificationFailure);
    db.endSession();

    // Turning verification failures off resets those marks.
    db.beginSession(5, 5, IndirectFailurePolicy{false, false});
    expectUntouched(db, 1, 10);
    expectUntouched(db, 2, 10);
    EXPECT_EQ(db.runStatus(0, 20), RunStatus::Unknown);
    db.endSession();

    // Turning crashes back on marks the solution again.
    int64_t sixth = db.beginSession(6, 6);
    expectMarked(
        db, 1, 20, RunStatus::IndirectlyFailed, "INDIRECTLY_FAILED", sixth, 0);
}

TEST(BenchmarkDatabase, DisabledCrashesLeaveDeadPairsAlone)
{
    TempDb       tmp;
    TestDatabase db(tmp.path(), makeSolutions(), makeClientConfig(), makeConfigs());
    db.beginSession(1, 1);
    db.beginRun(0, 10);
    db.beginSession(2, 2, IndirectFailurePolicy{false, true});
    expectUntouched(db, 1, 10);
    EXPECT_TRUE(db.isPending(2, 10));
}

// ---------------------------------------------------------------------------
// Schema constraints
// ---------------------------------------------------------------------------

// Table ids follow the insertion order, so problem_index 0 is id 1 and solution 10 is id 1.
static std::string const firstPairWhere = " WHERE problem_id = 1 AND solution_id = 1";

TEST(BenchmarkDatabase, RunStatusIsLimitedToTheKnownValues)
{
    TempDb       tmp;
    TestDatabase db(tmp.path(), makeSolutions(), makeClientConfig(), makeConfigs());
    EXPECT_THROW(executeSql(db.path(), "UPDATE runs SET status = 7" + firstPairWhere), SqliteError);
    EXPECT_THROW(executeSql(db.path(), "UPDATE runs SET status = -1" + firstPairWhere),
                 SqliteError);
    EXPECT_NO_THROW(executeSql(db.path(), "UPDATE runs SET status = 3" + firstPairWhere));
    EXPECT_NO_THROW(executeSql(db.path(), "UPDATE runs SET status = 6" + firstPairWhere));
}

TEST(BenchmarkDatabase, CausedByExistsExactlyForIndirectStatuses)
{
    TempDb       tmp;
    TestDatabase db(tmp.path(), makeSolutions(), makeClientConfig(), makeConfigs());
    for(int32_t status : {4, 5})
        EXPECT_THROW(
            executeSql(db.path(),
                       "UPDATE runs SET status = " + std::to_string(status) + firstPairWhere),
            SqliteError);
    EXPECT_THROW(
        executeSql(db.path(),
                   "UPDATE runs SET status = 2, caused_by_problem_id = 2" + firstPairWhere),
        SqliteError);
    EXPECT_THROW(executeSql(db.path(), "UPDATE runs SET caused_by_problem_id = 2" + firstPairWhere),
                 SqliteError);
    EXPECT_THROW(
        executeSql(db.path(),
                   "UPDATE runs SET status = 6, caused_by_problem_id = 2" + firstPairWhere),
        SqliteError);

    EXPECT_NO_THROW(executeSql(
        db.path(), "UPDATE runs SET status = 5, caused_by_problem_id = 2" + firstPairWhere));
    EXPECT_NO_THROW(executeSql(
        db.path(), "UPDATE runs SET status = 0, caused_by_problem_id = NULL" + firstPairWhere));
}

TEST(BenchmarkDatabase, ThereIsASingleClientConfig)
{
    TempDb       tmp;
    TestDatabase db(tmp.path(), makeSolutions(), makeClientConfig(), makeConfigs());
    EXPECT_THROW(executeSql(db.path(), "INSERT INTO client_configs(id, ini) VALUES(2, 'x')"),
                 SqliteError);
    EXPECT_THROW(executeSql(db.path(), "INSERT INTO client_configs(ini) VALUES('x')"), SqliteError);
    EXPECT_EQ(queryInt(db, "SELECT COUNT(*) FROM client_configs"), 1);
}

// ---------------------------------------------------------------------------
// Not selected pairs
// ---------------------------------------------------------------------------

// A NotSelected pair keeps the session that marked it and nothing else.
static void expectNotSelected(TestDatabase& db, int64_t problem, int64_t solution, int64_t session)
{
    EXPECT_EQ(db.runStatus(problem, solution), RunStatus::NotSelected);
    EXPECT_EQ(queryInt(db,
                       "SELECT session_id = " + std::to_string(session)
                           + " AND start_time IS NULL AND end_time IS NULL AND validation IS NULL "
                             "AND caused_by_problem_id IS NULL"
                           + runWhere(problem, solution)),
              1);
    EXPECT_FALSE(db.isPending(problem, solution));
}

TEST(BenchmarkDatabase, MarkNotSelectedChangesOnlyPendingPairsOfTheProblem)
{
    TempDb       tmp;
    TestDatabase db(tmp.path(), makeSolutions(), makeClientConfig(), makeConfigs());
    int64_t      session = db.beginSession(1, 1);
    db.beginRun(0, 10);
    db.endRun(0, 10, RunStatus::Pass, "PASSED", {}, {});

    db.markNotSelected(0);
    EXPECT_EQ(db.runStatus(0, 10), RunStatus::Pass);
    expectNotSelected(db, 0, 20, session);
    for(int64_t solution : {10, 20})
    {
        expectUntouched(db, 1, solution);
        expectUntouched(db, 2, solution);
    }
    EXPECT_FALSE(db.hasPendingRuns(0));
    EXPECT_TRUE(db.hasPendingRuns(1));

    // Marking again is a no-op.
    db.markNotSelected(0);
    expectNotSelected(db, 0, 20, session);
    EXPECT_EQ(db.runStatus(0, 10), RunStatus::Pass);
}

TEST(BenchmarkDatabase, MarkNotSelectedKeepsEveryOtherStatus)
{
    TempDb       tmp;
    TestDatabase db(tmp.path(), makeSolutions(), makeClientConfig(), makeConfigs());
    db.beginSession(1, 1);
    db.beginRun(0, 10);
    db.endRun(0, 10, RunStatus::Failure, "DID_NOT_SATISFY_ASSERTS", {}, {});
    db.beginRun(1, 10);
    db.endRun(1, 10, RunStatus::Pass, "PASSED", {}, {});
    db.beginRun(2, 10);
    // Solution 10 on problem 2 stays Unknown, as after a crash.
    db.beginRun(0, 20);
    db.endRun(0, 20, RunStatus::Failure, "FAILED", {}, {});
    // Solution 20 failed verification, so its pairs of problems 1 and 2 are indirect.
    EXPECT_EQ(db.runStatus(1, 20), RunStatus::IndirectVerificationFailure);

    for(int64_t problem : {0, 1, 2})
        db.markNotSelected(problem);
    EXPECT_EQ(db.runStatus(0, 10), RunStatus::Failure);
    EXPECT_EQ(db.runStatus(1, 10), RunStatus::Pass);
    EXPECT_EQ(db.runStatus(2, 10), RunStatus::Unknown);
    EXPECT_EQ(db.runStatus(0, 20), RunStatus::Failure);
    EXPECT_EQ(db.runStatus(1, 20), RunStatus::IndirectVerificationFailure);
    EXPECT_EQ(db.runStatus(2, 20), RunStatus::IndirectVerificationFailure);
    EXPECT_EQ(causedByProblemIndex(db, 1, 20), 0);
    EXPECT_EQ(queryInt(db, "SELECT COUNT(*) FROM runs WHERE status = 6"), 0);
}

TEST(BenchmarkDatabase, MarkNotSelectedKeepsCrashMarkedPairs)
{
    TempDb       tmp;
    TestDatabase db(tmp.path(), makeSolutions(), makeClientConfig(), makeConfigs());
    db.beginSession(1, 1);
    db.beginRun(0, 10);
    int64_t session = db.beginSession(2, 2);
    expectMarked(db, 1, 10, RunStatus::IndirectlyFailed, "INDIRECTLY_FAILED", session, 0);

    db.markNotSelected(1);
    expectMarked(db, 1, 10, RunStatus::IndirectlyFailed, "INDIRECTLY_FAILED", session, 0);
    expectNotSelected(db, 1, 20, session);
}

TEST(BenchmarkDatabase, MarkNotSelectedNeedsASession)
{
    TempDb       tmp;
    TestDatabase db(tmp.path(), makeSolutions(), makeClientConfig(), makeConfigs());
    EXPECT_THROW(db.markNotSelected(0), BenchmarkDatabaseError);
    expectUntouched(db, 0, 10);

    db.beginSession(1, 1);
    EXPECT_THROW(db.markNotSelected(99), BenchmarkDatabaseError);
}

TEST(BenchmarkDatabase, NotSelectedIsNotPending)
{
    TempDb       tmp;
    TestDatabase db(tmp.path(), makeSolutions(), makeClientConfig(), makeConfigs());
    db.beginSession(1, 1);
    db.beginRun(0, 10);
    db.endRun(0, 10, RunStatus::Pass, "PASSED", {}, {});
    EXPECT_TRUE(db.hasPendingRuns(0));
    db.markNotSelected(0);

    EXPECT_FALSE(db.isPending(0, 20));
    EXPECT_FALSE(db.hasPendingRuns(0));
    EXPECT_THROW(db.beginRun(0, 20), BenchmarkDatabaseError);
    EXPECT_EQ(db.runStatus(0, 20), RunStatus::NotSelected);

    // The problem is pending again as soon as one pair is NotBenchmarked.
    executeSql(db.path(),
               "UPDATE runs SET status = 0, session_id = NULL" + firstPairWhere);
    EXPECT_TRUE(db.hasPendingRuns(0));
}

// Every session start reviews the indirect markings, which must leave NotSelected alone.
TEST(BenchmarkDatabase, SessionStartLeavesNotSelectedPairsAlone)
{
    TempDb       tmp;
    TestDatabase db(tmp.path(), makeSolutions(), makeClientConfig(), makeConfigs());
    int64_t      first = db.beginSession(1, 1);
    db.beginRun(0, 10);
    db.endRun(0, 10, RunStatus::Pass, "PASSED", {}, {});
    db.markNotSelected(0);
    db.beginRun(1, 10);
    db.endRun(1, 10, RunStatus::Failure, "FAILED", {}, {});
    db.markNotSelected(2);
    EXPECT_EQ(db.runStatus(2, 10), RunStatus::IndirectVerificationFailure);
    db.beginRun(1, 20);
    // The session dies here: solution 20 crashed.

    // Crash marking must skip the NotSelected pairs of solution 20.
    db.beginSession(2, 2);
    expectNotSelected(db, 0, 20, first);
    expectNotSelected(db, 2, 20, first);
    EXPECT_EQ(db.runStatus(1, 20), RunStatus::Unknown);
    db.endSession();

    // Disabling either kind resets only its own marks.
    for(IndirectFailurePolicy policy : {IndirectFailurePolicy{false, true},
                                        IndirectFailurePolicy{true, false},
                                        IndirectFailurePolicy{false, false},
                                        IndirectFailurePolicy{true, true}})
    {
        db.beginSession(3, 3, policy);
        expectNotSelected(db, 0, 20, first);
        expectNotSelected(db, 2, 20, first);
        db.endSession();
    }
}

TEST(BenchmarkDatabase, VerificationFailurePropagationSkipsNotSelectedPairs)
{
    TempDb       tmp;
    TestDatabase db(tmp.path(), makeSolutions(), makeClientConfig(), makeConfigs());
    int64_t      session = db.beginSession(1, 1);
    db.markNotSelected(2);
    db.beginRun(0, 10);
    db.endRun(0, 10, RunStatus::Failure, "FAILED", {}, {});

    expectNotSelected(db, 2, 10, session);
    expectNotSelected(db, 2, 20, session);
    expectMarked(db,
                 1,
                 10,
                 RunStatus::IndirectVerificationFailure,
                 "INDIRECT_VERIFICATION_FAILURE",
                 session,
                 0);
}

TEST(BenchmarkDatabase, NotSelectedPairsSurviveReopening)
{
    TempDb  tmp;
    int64_t session = 0;
    {
        TestDatabase db(tmp.path(), makeSolutions(), makeClientConfig(), makeConfigs());
        session = db.beginSession(1, 1);
        db.beginRun(0, 10);
        db.endRun(0, 10, RunStatus::Pass, "PASSED", {}, {});
        db.markNotSelected(0);
        db.endSession();
    }

    TestDatabase db(tmp.path(), makeSolutions(), makeClientConfig(), makeConfigs());
    expectNotSelected(db, 0, 20, session);
    EXPECT_EQ(db.runStatus(0, 10), RunStatus::Pass);
    db.beginSession(2, 2);
    expectNotSelected(db, 0, 20, session);
    EXPECT_FALSE(db.hasPendingRuns(0));
    EXPECT_TRUE(db.hasPendingRuns(1));
}

TEST(BenchmarkDatabase, ValuesAndIterationsNeedARun)
{
    TempDb       tmp;
    TestDatabase db(tmp.path(), makeSolutions(), makeClientConfig(), makeConfigs());
    EXPECT_THROW(executeSql(db.path(),
                            "INSERT INTO run_values(problem_id, solution_id, key, value) "
                            "VALUES(99, 1, 'k', 'v')"),
                 SqliteError);
    EXPECT_THROW(executeSql(db.path(),
                            "INSERT INTO iterations(problem_id, solution_id, sync_index, time_us, "
                            "enqueues) VALUES(1, 99, 0, 1.0, 1)"),
                 SqliteError);

    EXPECT_NO_THROW(executeSql(
        db.path(),
        "INSERT INTO run_values(problem_id, solution_id, key, value) VALUES(1, 1, 'k', 'v')"));
    EXPECT_NO_THROW(
        executeSql(db.path(),
                   "INSERT INTO iterations(problem_id, solution_id, sync_index, time_us, enqueues) "
                   "VALUES(1, 1, 0, NULL, 1)"));
}

TEST(BenchmarkDatabase, NaNIterationTimeIsStoredAsNull)
{
    TempDb       tmp;
    TestDatabase db(tmp.path(), makeSolutions(), makeClientConfig(), makeConfigs());
    db.beginSession(1, 1);
    db.beginRun(0, 10);
    db.endRun(0,
              10,
              RunStatus::Pass,
              "PASSED",
              {},
              {{0, std::numeric_limits<double>::quiet_NaN(), 1}, {1, 2.5, 2}});

    EXPECT_EQ(queryInt(db, "SELECT time_us IS NULL FROM iterations WHERE sync_index = 0"), 1);
    EXPECT_EQ(queryInt(db, "SELECT time_us IS NULL FROM iterations WHERE sync_index = 1"), 0);
}

TEST(BenchmarkDatabase, EndRunNeedsASession)
{
    TempDb       tmp;
    TestDatabase db(tmp.path(), makeSolutions(), makeClientConfig(), makeConfigs());
    EXPECT_THROW(db.endRun(0, 10, RunStatus::Pass, "PASSED", {}, {}), BenchmarkDatabaseError);
}

TEST(BenchmarkDatabase, NaNIterationTimeKeepsTheVerdict)
{
    double const nan = std::numeric_limits<double>::quiet_NaN();
    TempDb       tmp;
    TestDatabase db(tmp.path(), makeSolutions(), makeClientConfig(), makeConfigs());
    db.beginSession(1, 1);

    db.beginRun(0, 10);
    db.endRun(0, 10, RunStatus::Pass, "PASSED", {}, {{0, nan, 1}});
    EXPECT_EQ(db.runStatus(0, 10), RunStatus::Pass);
    EXPECT_EQ(queryText(db, "SELECT typeof(time_us) FROM iterations"), "null");

    db.beginRun(0, 20);
    db.endRun(0, 20, RunStatus::Failure, "INVALID", {}, {{0, nan, 1}});
    EXPECT_EQ(db.runStatus(0, 20), RunStatus::Failure);
    EXPECT_EQ(queryInt(db, "SELECT COUNT(*) FROM iterations WHERE time_us IS NULL"), 2);
}

TEST(BenchmarkDatabase, WriteAheadLoggingIsRequired)
{
    // An in-memory database reports the journal mode 'memory' and never enters WAL.
    try
    {
        BenchmarkDatabase db(":memory:", makeSolutions(), makeClientConfig(), makeConfigs());
        FAIL() << "expected an error";
    }
    catch(BenchmarkDatabaseError const& error)
    {
        EXPECT_NE(std::string(error.what()).find("write-ahead logging"), std::string::npos)
            << error.what();
        EXPECT_NE(std::string(error.what()).find("'memory'"), std::string::npos) << error.what();
    }
}

// ---------------------------------------------------------------------------
// Scaling
// ---------------------------------------------------------------------------

TEST(BenchmarkDatabase, CrashMarkingScalesWithTheNumberOfPairs)
{
    int64_t const solutionCount = 200;
    int64_t const problemCount  = 200;
    int64_t const crashedCount  = 5;

    std::vector<SolutionRecord>   solutions;
    std::vector<InitConfigRecord> configs;
    for(int64_t i = 0; i < solutionCount; i++)
        solutions.push_back({i, "s" + std::to_string(i), "x", "msgpack"});
    for(int64_t i = 0; i < problemCount; i++)
        configs.push_back({i});

    TempDb       tmp;
    TestDatabase db(tmp.path(), solutions, "c", configs);
    // The first problem of the first solutions was running when the previous session died.
    executeSql(db.path(),
               "UPDATE runs SET status = 1 WHERE problem_id = 1 AND solution_id <= "
                   + std::to_string(crashedCount));

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    db.beginSession(1, 1, IndirectFailurePolicy{true, true});
    std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;

    int64_t pairs = solutionCount * problemCount;
    RecordProperty("beginSessionSeconds", std::to_string(elapsed.count()));
    // Generous limit: it only has to catch quadratic behaviour, not machine speed.
    EXPECT_LT(elapsed.count(), 50.0);

    EXPECT_EQ(queryInt(db, "SELECT COUNT(*) FROM runs WHERE status = 1"), crashedCount);
    EXPECT_EQ(queryInt(db, "SELECT COUNT(*) FROM runs WHERE status = 4"),
              crashedCount * (problemCount - 1));
    EXPECT_EQ(queryInt(db, "SELECT COUNT(*) FROM runs WHERE status = 0"),
              pairs - crashedCount * problemCount);
}
