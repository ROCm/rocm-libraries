// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// GPU-free parity check between the client's results CSV (ResultFileReporter) and the benchmark
// database (DatabaseReporter), driven by the checked-in fixtures in tests/data/benchmark_db. Every
// scenario has a golden <name>.csv and a canonical database dump <name>.db.txt. This test checks
// that the CSV that ResultFileReporter writes for a scripted run equals <name>.csv byte for byte,
// and that the database that the DatabaseReporter fills in the same run equals <name>.db.txt.
// So the golden CSV is what the client writes for the run, and <name>.db.txt is the database that
// holds the same results.
// Tests/unit/test_benchmark_csv_parity.py checks the same fixtures against the Python writer
// (tensilelite.BenchmarkDatabase.writeResultsCsv), which writes -1 for pairs that were never
// reached and an empty cell for NotSelected pairs. It visits the solutions of a problem in library
// index order, as the scripted run does.
//
// The scripted run mimics the per-pair loop of client/main.cpp with the real recording API
// (BenchmarkSession, classifyPair, PairRecorder), and sends every report to both reporters:
//
//  - Pairs the loop launches report their real values to ResultFileReporter.
//  - Pairs classifyPair does not return Record for (Unknown, IndirectlyFailed and
//    IndirectVerificationFailure) report validation SKIPPED_BY_DB after preSolution, as main does,
//    which gives the same -1 cell.
//  - Pairs the iterator does not select are not reported at all, as the real client never visits
//    them. When the problem finishes they become NotSelected, as main does, and their CSV cell is
//    empty.
//  - Pairs of a problem cut short (an early exit) stay NotBenchmarked and are replayed with the
//    SKIPPED_BY_DB report. This is the one deliberate deviation: a real client leaves the cell of
//    a pair it never reached empty, but the golden CSV holds -1 for them, so the replay makes the
//    cell match. Only these never-reached pairs keep the -1.
//
// Canonical dump: one tab separated record per line, in this order, with no header or comment.
//
//   meta          <key> <value>                   the three csv_* and metric keys, by key
//   solution      <libraryIndex> <name>           by library index
//   problem       <problemIndex>                  every init config
//   problem_value <problemIndex> <key> <value>    by problem, key
//   run           <problemIndex> <libraryIndex> <STATUS> <validation> <causedByProblemIndex>
//   run_value     <problemIndex> <libraryIndex> <key> <value>
//
// STATUS uses the upper-case names of RunStatus (for example PASS), and an absent validation or
// cause is empty. Sessions, timestamps, iterations and the spec and ini blobs are left out: they
// are not deterministic or do not affect the CSV.
//
// On a mismatch the actual text is printed and nothing is regenerated. To update a fixture
// deliberately, review the printed text and copy it over the file.

#include <gtest/gtest.h>

#include "BenchmarkDatabaseSetup.hpp"
#include "BenchmarkRecording.hpp"
#include "MetaResultReporter.hpp"
#include "ResultFileReporter.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#ifndef BENCHMARK_DB_FIXTURE_DIR
#error "BENCHMARK_DB_FIXTURE_DIR must name the fixture directory"
#endif

using namespace TensileLite;
using namespace TensileLite::Client;

namespace
{
    // How a scripted pair ends. Skipped pairs are not launched because the database already holds
    // them; their status comes from the database history, not from the script.
    enum class Outcome
    {
        Pass,
        // No validation is reported, which the client stores as PASS.
        NoVerdict,
        // The solution was skipped as too slow: no timing, a NaN time and zero rates.
        NanTime,
        Failed,
        // The launch threw, so main reports INVALID.
        Invalid,
        Skipped,
        // Pending at the end of the run, as after an early exit. A problem with such a pair did
        // not finish, so none of its pairs is marked NotSelected.
        NotLaunched,
        // The iterator did not select the pair, so the client never visits it.
        NotSelected,
    };

    struct PairScript
    {
        Outcome outcome;
        double  timeUs;
        double  tilesPerCu;
        double  totalGranularity;
    };

    struct ProblemScript
    {
        int64_t index;
        size_t  m;
        size_t  n;
        size_t  k;
        // One entry per solution, in library index order.
        std::vector<PairScript> pairs;
    };

    struct Scenario
    {
        std::string       name;
        PerformanceMetric metric;
        bool              extraCols;
        bool              mergeSameProblems;
        // The problems of the run.
        std::vector<ProblemScript> (*makeProblems)();
    };

    struct ActualOutput
    {
        std::string csv;
        std::string dump;
    };

    // Unique temporary directory, removed with the -wal and -shm files it holds.
    class TempDir
    {
    public:
        TempDir()
        {
            std::string pattern
                = (std::filesystem::temp_directory_path() / "tensilelite_parity_XXXXXX").string();
            char* created = mkdtemp(pattern.data());
            if(!created)
                throw std::runtime_error("cannot create temporary directory");
            m_dir = created;
        }

        ~TempDir()
        {
            std::error_code error;
            std::filesystem::remove_all(m_dir, error);
        }

        TempDir(TempDir const&)            = delete;
        TempDir& operator=(TempDir const&) = delete;

        std::string file(std::string const& name) const
        {
            return (m_dir / name).string();
        }

    private:
        std::filesystem::path m_dir;
    };
}

static constexpr int32_t cuCount         = 8;
static constexpr int64_t syncCount       = 2;
static constexpr int64_t enqueuesPerSync = 10;
static constexpr double  nanValue        = std::numeric_limits<double>::quiet_NaN();

// The pair that the dead session leaves Unknown, which makes the rest of its solution
// IndirectlyFailed.
static constexpr int64_t crashedProblemIndex  = 0;
static constexpr int64_t crashedSolutionIndex = 3;

static std::vector<SolutionRecord> makeSolutions()
{
    // The names with a comma and a double quote exercise the escaping of the header and of the
    // winner name column.
    std::vector<std::string> names
        = {"SolA", "SolB,comma", "SolC\"quote\"", "SolD", "SolE", "SolF"};
    std::vector<SolutionRecord> solutions;
    for(int64_t index = 0; index < static_cast<int64_t>(names.size()); index++)
        solutions.push_back(SolutionRecord{index, names[index], "spec", "yaml"});
    return solutions;
}

// Solutions: A..F. SolD is the crashed solution, SolC fails verification on the second problem, so
// its remaining pairs become IndirectVerificationFailure, and SolF never runs on some problems.
// Every problem has launched pairs, so every problem has stored values.
static std::vector<ProblemScript> makeProblems()
{
    // clang-format off
    return {
        {0, 1024, 1024, 1024, {
            {Outcome::Pass,        100.0, 0.25, 0.90},
            {Outcome::Pass,         80.0, 3.00, 0.95},
            {Outcome::Pass,         70.0, 0.75, 0.70},
            {Outcome::Skipped,       0.0, 2.00, 0.80},
            {Outcome::Pass,        150.0, 4.00, 0.60},
            {Outcome::NotLaunched,   0.0, 1.00, 0.50}}},
        {1, 512, 2048, 256, {
            {Outcome::Pass,         60.0, 0.50, 0.55},
            {Outcome::NoVerdict,    50.0, 2.00, 0.65},
            {Outcome::Failed,       20.0, 5.00, 0.99},
            {Outcome::Skipped,       0.0, 2.00, 0.80},
            {Outcome::Invalid,       0.0, 1.00, 0.40},
            {Outcome::Pass,         65.0, 1.25, 0.75}}},
        {2, 4096, 128, 64, {
            {Outcome::NanTime,       0.0, 3.50, 0.85},
            {Outcome::Pass,         40.0, 6.00, 0.50},
            {Outcome::Skipped,       0.0, 1.00, 0.50},
            {Outcome::Skipped,       0.0, 2.00, 0.80},
            {Outcome::Pass,         45.0, 2.00, 0.90},
            {Outcome::NotLaunched,   0.0, 1.00, 0.50}}},
        {5, 100, 200, 300, {
            {Outcome::Failed,       30.0, 1.00, 0.80},
            {Outcome::Pass,         35.0, 1.50, 0.70},
            {Outcome::Skipped,       0.0, 1.00, 0.50},
            {Outcome::Skipped,       0.0, 2.00, 0.80},
            {Outcome::Pass,         50.0, 0.25, 0.40},
            {Outcome::Pass,         33.0, 2.00, 0.95}}},
    };
    // clang-format on
}

// Selection by performance prediction: every problem selects another subset of the solutions, and
// the other pairs become NotSelected when the problem finishes. SolD is the crashed solution and
// SolC fails verification on the second problem. The crash marking runs at the start of the
// session, before any selection is known, so SolD is IndirectlyFailed on the later problems
// although the prediction selects it on none of them. SolE is never selected, so it never gets a
// CSV column. The first problem visits every solution that a later problem selects: the CSV
// writer takes its header from the solutions of the problems it has seen.
static std::vector<ProblemScript> makePredictionProblems()
{
    // clang-format off
    return {
        {0, 1024, 1024, 1024, {
            {Outcome::Pass,         100.0, 0.25, 0.90},
            {Outcome::Pass,          80.0, 3.00, 0.95},
            {Outcome::Pass,          70.0, 0.75, 0.70},
            {Outcome::Skipped,        0.0, 2.00, 0.80},
            {Outcome::NotSelected,    0.0, 1.00, 0.50},
            {Outcome::Pass,          90.0, 2.00, 0.60}}},
        {1, 512, 2048, 256, {
            {Outcome::NotSelected,    0.0, 1.00, 0.50},
            {Outcome::Pass,          60.0, 0.50, 0.55},
            {Outcome::Failed,        20.0, 5.00, 0.99},
            {Outcome::Skipped,        0.0, 2.00, 0.80},
            {Outcome::NotSelected,    0.0, 1.00, 0.50},
            {Outcome::NotSelected,    0.0, 1.00, 0.50}}},
        {2, 4096, 128, 64, {
            {Outcome::Pass,          40.0, 6.00, 0.50},
            {Outcome::NotSelected,    0.0, 1.00, 0.50},
            {Outcome::Skipped,        0.0, 1.00, 0.50},
            {Outcome::Skipped,        0.0, 2.00, 0.80},
            {Outcome::NotSelected,    0.0, 1.00, 0.50},
            {Outcome::Pass,          45.0, 2.00, 0.90}}},
    };
    // clang-format on
}

// The client adds a CSV column the first time it visits a solution, and writes every row with the
// columns known at that time. A solution that only a later problem selects is therefore missing
// from the header, and the rows written before it appeared are shorter: later problems add C and F,
// and the last one adds E. The pairs the iterator did not select have empty cells.
static std::vector<ProblemScript> makeLateColumnProblems()
{
    // clang-format off
    return {
        {0, 1024, 1024, 1024, {
            {Outcome::Pass,         100.0, 0.25, 0.90},
            {Outcome::Pass,          80.0, 3.00, 0.95},
            {Outcome::NotSelected,    0.0, 1.00, 0.50},
            {Outcome::Skipped,        0.0, 2.00, 0.80},
            {Outcome::NotSelected,    0.0, 1.00, 0.50},
            {Outcome::NotSelected,    0.0, 1.00, 0.50}}},
        {1, 512, 2048, 256, {
            {Outcome::Pass,          60.0, 0.50, 0.55},
            {Outcome::NotSelected,    0.0, 1.00, 0.50},
            {Outcome::Pass,          50.0, 2.00, 0.65},
            {Outcome::Skipped,        0.0, 2.00, 0.80},
            {Outcome::NotSelected,    0.0, 1.00, 0.50},
            {Outcome::Pass,          65.0, 1.25, 0.75}}},
        {2, 4096, 128, 64, {
            {Outcome::NotSelected,    0.0, 1.00, 0.50},
            {Outcome::Pass,          40.0, 6.00, 0.50},
            {Outcome::NotSelected,    0.0, 1.00, 0.50},
            {Outcome::Skipped,        0.0, 2.00, 0.80},
            {Outcome::Pass,          45.0, 2.00, 0.90},
            {Outcome::NotSelected,    0.0, 1.00, 0.50}}},
    };
    // clang-format on
}

static size_t flopCount(ProblemScript const& problem)
{
    return 2 * problem.m * problem.n * problem.k;
}

// Bytes moved by a half precision GEMM; only the bandwidth depends on it.
static double byteCount(ProblemScript const& problem)
{
    return 2.0
           * static_cast<double>(problem.m * problem.k + problem.k * problem.n
                                 + 2 * problem.m * problem.n);
}

namespace
{
    // The values BenchmarkTimer reports after a solution.
    struct Timing
    {
        double timeUs;
        double gflopsPerCu;
        double gflops;
        double bandwidthGbps;
    };
}

static Timing measure(ProblemScript const& problem, PairScript const& script)
{
    if(script.outcome == Outcome::NanTime)
        return {nanValue, 0.0, 0.0, 0.0};
    // An unstarted launch divides zero by zero. The sign of that NaN depends on the platform, so a
    // fixed one is used.
    if(script.outcome == Outcome::Invalid)
        return {nanValue, nanValue, nanValue, nanValue};

    double  gflops    = static_cast<double>(flopCount(problem)) / script.timeUs / 1000.0;
    int32_t tiles     = static_cast<int32_t>(script.tilesPerCu * cuCount);
    int32_t usedCus   = std::min(tiles, cuCount);
    double  bandwidth = byteCount(problem) / script.timeUs / 1e3;
    return {script.timeUs, gflops / usedCus, gflops, bandwidth};
}

static std::optional<std::string> verdictOf(Outcome outcome)
{
    switch(outcome)
    {
    case Outcome::Pass:
        return "PASSED";
    case Outcome::Failed:
        return "FAILED";
    case Outcome::Invalid:
        return "INVALID";
    default:
        return std::nullopt;
    }
}

namespace
{
    // Plays the reports of main's per-pair loop into the client's result file reporter and the
    // database reporter, like main does through its MetaResultReporter. It mirrors main's per-pair
    // sequence by hand, so this test does not detect a drift against client/main.cpp.
    class SyntheticClient
    {
    public:
        SyntheticClient(std::shared_ptr<BenchmarkDatabase>  database,
                        std::shared_ptr<ResultFileReporter> fileReporter,
                        std::vector<SolutionRecord> const&  solutions)
            : m_database(std::move(database))
            , m_databaseReporter(std::make_shared<DatabaseReporter>(m_database))
            , m_reporters(std::make_shared<MetaResultReporter>())
            , m_solutions(solutions)
        {
            m_reporters->addReporter(fileReporter);
            m_reporters->addReporter(m_databaseReporter);
        }

        void run(std::vector<ProblemScript> const& problems)
        {
            {
                BenchmarkSession session(m_database, 3, 0, IndirectFailurePolicy());
                m_reporters->report(ResultKey::ProblemCount, problems.size());
                for(ProblemScript const& problem : problems)
                    runProblem(problem, problems.back().index);
                session.end();
            }
            m_reporters->finalizeReport();
        }

    private:
        void reportProblem(ProblemScript const& problem, int64_t lastProblemIndex)
        {
            m_reporters->report(ResultKey::ProblemIndex, problem.index);
            m_reporters->report(ResultKey::ProblemProgress,
                                std::to_string(problem.index) + "/"
                                    + std::to_string(lastProblemIndex));
            m_reporters->report(ResultKey::OperationIdentifier,
                                std::string("Contraction_l_Alik_Bljk_Cijk_Dijk"));
            m_reporters->report(ResultKey::TotalFlops, flopCount(problem));
            m_reporters->report(ResultKey::LDA, problem.m);
            m_reporters->report(ResultKey::LDB, problem.k);
            m_reporters->report(ResultKey::LDC, problem.m);
            m_reporters->report(ResultKey::LDD, problem.m);
            m_reporters->report(ResultKey::BiasType, std::string("None"));
            m_reporters->report(ResultKey::ActivationType, std::string("none"));
            m_reporters->report(ResultKey::ProblemSizes,
                                std::vector<size_t>{problem.m, problem.n, 1, problem.k});
        }

        // The keys the CSV does not use, like the other granularities, are left out.
        void reportPreSolution(SolutionRecord const& solution, PairScript const& script)
        {
            m_reporters->preSolution(nullptr);
            m_reporters->report(ResultKey::SolutionLibraryIndex, solution.libraryIndex);
            m_reporters->report(ResultKey::SolutionIndex, solution.libraryIndex);
            m_reporters->report(ResultKey::SolutionName, solution.name);
            m_reporters->report(ResultKey::TotalGranularity, script.totalGranularity);
            m_reporters->report(ResultKey::NumCus, static_cast<int64_t>(cuCount));
            m_reporters->report(ResultKey::TilesPerCu, script.tilesPerCu);
        }

        void reportSyncs(double timeUs)
        {
            for(int64_t sync = 0; sync < syncCount; sync++)
            {
                // The enqueue count comes first, as DatabaseReporter requires.
                m_reporters->report(ResultKey::SyncEnqueues, enqueuesPerSync);
                m_reporters->report(ResultKey::SyncTimeUS,
                                    timeUs * static_cast<double>(enqueuesPerSync));
            }
        }

        void reportLaunchedPair(ProblemScript const& problem, PairScript const& script)
        {
            std::optional<std::string> verdict = verdictOf(script.outcome);
            if(verdict)
                m_reporters->report(ResultKey::Validation, *verdict);

            bool timed = script.outcome != Outcome::NanTime && script.outcome != Outcome::Invalid;
            if(timed)
                reportSyncs(script.timeUs);

            Timing timing = measure(problem, script);
            m_reporters->report(ResultKey::TimeUS, timing.timeUs);
            m_reporters->report(ResultKey::SpeedGFlopsPerCu, timing.gflopsPerCu);
            m_reporters->report(ResultKey::SpeedGFlops, timing.gflops);
            m_reporters->report(ResultKey::GbpsBW, timing.bandwidthGbps);
        }

        void runPair(ProblemScript const&  problem,
                     SolutionRecord const& solution,
                     PairScript const&     script)
        {
            PairRecording recording
                = classifyPair(m_database.get(), problem.index, solution.libraryIndex);
            EXPECT_EQ(recording == PairRecording::SkippedByDatabase,
                      script.outcome == Outcome::Skipped)
                << "problem " << problem.index << ", solution " << solution.name;

            // The real client never visits a pair the iterator did not select.
            if(script.outcome == Outcome::NotSelected)
            {
                EXPECT_EQ(recording, PairRecording::Record)
                    << "problem " << problem.index << ", solution " << solution.name;
                return;
            }

            reportPreSolution(solution, script);

            // BenchmarkTimer would report NaN timings after a pair that did not run, which both
            // reporters ignore after the SKIPPED_BY_DB verdict.
            if(recording == PairRecording::SkippedByDatabase
               || script.outcome == Outcome::NotLaunched)
            {
                m_reporters->report(ResultKey::Validation, std::string("SKIPPED_BY_DB"));
                m_reporters->postSolution();
                return;
            }

            PairRecorder recorder(*m_databaseReporter, problem.index, solution.libraryIndex);
            reportLaunchedPair(problem, script);
            m_reporters->postSolution();
            recorder.finish();
        }

        void runProblem(ProblemScript const& problem, int64_t lastProblemIndex)
        {
            if(!m_database->hasPendingRuns(problem.index))
                return;

            reportProblem(problem, lastProblemIndex);
            for(size_t solution = 0; solution < m_solutions.size(); solution++)
                runPair(problem, m_solutions[solution], problem.pairs[solution]);
            m_reporters->postProblem();
            // As in main, only a problem that ended normally marks its unselected pairs.
            bool cutShort = std::any_of(
                problem.pairs.begin(), problem.pairs.end(), [](PairScript const& pair) {
                    return pair.outcome == Outcome::NotLaunched;
                });
            if(!cutShort)
                m_database->markNotSelected(problem.index);
        }

        std::shared_ptr<BenchmarkDatabase>  m_database;
        std::shared_ptr<DatabaseReporter>   m_databaseReporter;
        std::shared_ptr<MetaResultReporter> m_reporters;
        std::vector<SolutionRecord>         m_solutions;
    };
}

static std::shared_ptr<BenchmarkDatabase> openDatabase(std::string const&                 path,
                                                       Scenario const&                    scenario,
                                                       std::vector<ProblemScript> const&  problems,
                                                       std::vector<SolutionRecord> const& solutions)
{
    std::vector<InitConfigRecord> configs;
    for(ProblemScript const& problem : problems)
        configs.push_back(InitConfigRecord{problem.index});

    std::shared_ptr<BenchmarkDatabase> database
        = std::make_shared<BenchmarkDatabase>(path, solutions, "count=1\n", configs);

    recordCsvMeta(*database, scenario.metric, scenario.extraCols, scenario.mergeSameProblems);
    return database;
}

// A client that is killed with one pair launched leaves it Unknown, and the next session marks the
// other pairs of its solution IndirectlyFailed. A verification failure only spreads in a session
// whose predecessor ended, so a resumed session that ends at once separates the crash from the
// session that runs the problems.
static std::shared_ptr<BenchmarkDatabase>
    resumeAfterCrash(std::string const&                 path,
                     Scenario const&                    scenario,
                     std::vector<ProblemScript> const&  problems,
                     std::vector<SolutionRecord> const& solutions)
{
    {
        std::shared_ptr<BenchmarkDatabase> crashed
            = openDatabase(path, scenario, problems, solutions);
        crashed->beginSession(1, 0);
        crashed->beginRun(crashedProblemIndex, crashedSolutionIndex);
        // Destroyed without endSession, like a killed process.
    }
    std::shared_ptr<BenchmarkDatabase> resumed = openDatabase(path, scenario, problems, solutions);
    resumed->beginSession(2, 0);
    resumed->endSession();
    return resumed;
}

static std::string readFile(std::string const& path)
{
    std::ifstream stream(path, std::ios::binary);
    if(!stream)
        throw std::runtime_error("cannot read '" + path + "'");
    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

static std::string checkedField(std::string const& value)
{
    if(value.find_first_of("\t\n") != std::string::npos)
        throw std::runtime_error("dump value contains a tab or a newline: " + value);
    return value;
}

// Appends one record per row of the query, prefixed by the record kind.
static void dumpRecords(SqliteConnection&  db,
                        std::string const& kind,
                        std::string const& sql,
                        int32_t            columns,
                        std::string&       dump)
{
    SqliteStatement statement(db, sql);
    while(statement.step())
    {
        dump += kind;
        for(int32_t column = 0; column < columns; column++)
            dump += "\t" + checkedField(statement.columnText(column));
        dump += "\n";
    }
}

static std::string dumpDatabase(std::string const& path)
{
    SqliteConnection db(path);
    std::string      dump;
    dumpRecords(db,
                "meta",
                "SELECT key, value FROM meta WHERE key IN ('csv_export_extra_cols', "
                "'csv_merge_same_problems', 'performance_metric') ORDER BY key",
                2,
                dump);
    dumpRecords(db,
                "solution",
                "SELECT library_index, name FROM solutions ORDER BY library_index",
                2,
                dump);
    dumpRecords(
        db, "problem", "SELECT problem_index FROM init_configs ORDER BY problem_index", 1, dump);
    dumpRecords(db,
                "problem_value",
                "SELECT i.problem_index, v.key, v.value FROM problem_values v "
                "JOIN init_configs i ON i.id = v.problem_id ORDER BY i.problem_index, v.key",
                3,
                dump);
    dumpRecords(db,
                "run",
                "SELECT i.problem_index, s.library_index, "
                "CASE r.status WHEN 0 THEN 'NOT_BENCHMARKED' WHEN 1 THEN 'UNKNOWN' "
                "WHEN 2 THEN 'PASS' WHEN 3 THEN 'FAILURE' WHEN 4 THEN 'INDIRECTLY_FAILED' "
                "WHEN 5 THEN 'INDIRECT_VERIFICATION_FAILURE' WHEN 6 THEN 'NOT_SELECTED' END, "
                "COALESCE(r.validation, ''), COALESCE(CAST(c.problem_index AS TEXT), '') "
                "FROM runs r JOIN init_configs i ON i.id = r.problem_id "
                "JOIN solutions s ON s.id = r.solution_id "
                "LEFT JOIN init_configs c ON c.id = r.caused_by_problem_id "
                "ORDER BY i.problem_index, s.library_index",
                5,
                dump);
    dumpRecords(db,
                "run_value",
                "SELECT i.problem_index, s.library_index, v.key, v.value FROM run_values v "
                "JOIN init_configs i ON i.id = v.problem_id "
                "JOIN solutions s ON s.id = v.solution_id "
                "ORDER BY i.problem_index, s.library_index, v.key",
                4,
                dump);
    return dump;
}

static ActualOutput runScenario(Scenario const& scenario)
{
    TempDir                     dir;
    std::vector<SolutionRecord> solutions = makeSolutions();
    std::vector<ProblemScript>  problems  = scenario.makeProblems();

    std::shared_ptr<BenchmarkDatabase> database
        = resumeAfterCrash(dir.file("bench.db"), scenario, problems, solutions);
    {
        // The reporter flushes and closes its file when it is destroyed.
        std::shared_ptr<ResultFileReporter> fileReporter
            = std::make_shared<ResultFileReporter>(dir.file("results.csv"),
                                                   scenario.extraCols,
                                                   scenario.mergeSameProblems,
                                                   scenario.metric,
                                                   false);
        SyntheticClient client(database, fileReporter, solutions);
        client.run(problems);
    }
    return {readFile(dir.file("results.csv")), dumpDatabase(dir.file("bench.db"))};
}

static void expectMatchesFixture(std::string const& fileName, std::string const& actual)
{
    std::filesystem::path path = std::filesystem::path(BENCHMARK_DB_FIXTURE_DIR) / fileName;
    std::ifstream         stream(path, std::ios::binary);
    if(!stream)
    {
        ADD_FAILURE() << "fixture " << path << " is missing; the actual text is\n"
                      << "----- begin " << fileName << " -----\n"
                      << actual << "----- end " << fileName << " -----";
        return;
    }

    std::string expected((std::istreambuf_iterator<char>(stream)),
                         std::istreambuf_iterator<char>());
    if(expected == actual)
        return;
    ADD_FAILURE() << "fixture " << path << " differs from the actual text below; copy it over "
                  << "the fixture to update it deliberately\n"
                  << "----- begin " << fileName << " -----\n"
                  << actual << "----- end " << fileName << " -----";
}

class BenchmarkCsvParityTest : public ::testing::TestWithParam<Scenario>
{
};

TEST_P(BenchmarkCsvParityTest, ClientOutputMatchesFixtures)
{
    Scenario const& scenario = GetParam();
    ActualOutput    actual   = runScenario(scenario);
    expectMatchesFixture(scenario.name + ".csv", actual.csv);
    expectMatchesFixture(scenario.name + ".db.txt", actual.dump);
}

// The merge scenario also exports the extra columns, so that the winner name, which holds the
// comma and quote names, is a value that the merged rows escape twice.
INSTANTIATE_TEST_SUITE_P(
    BenchmarkCsvParity,
    BenchmarkCsvParityTest,
    ::testing::Values(
        Scenario{"default", PerformanceMetric::DeviceEfficiency, false, false, makeProblems},
        Scenario{"merge", PerformanceMetric::DeviceEfficiency, true, true, makeProblems},
        Scenario{"extra_cols", PerformanceMetric::DeviceEfficiency, true, false, makeProblems},
        Scenario{"cu_efficiency", PerformanceMetric::CUEfficiency, true, false, makeProblems},
        // Pairs that the iterator does not select are NotSelected, and their CSV cells are empty.
        Scenario{
            "prediction", PerformanceMetric::DeviceEfficiency, true, false, makePredictionProblems},
        // Solutions that only a later problem visits, with and without merging the rows.
        Scenario{"late_columns",
                 PerformanceMetric::DeviceEfficiency,
                 true,
                 false,
                 makeLateColumnProblems},
        Scenario{"late_columns_merge",
                 PerformanceMetric::DeviceEfficiency,
                 true,
                 true,
                 makeLateColumnProblems}),
    [](::testing::TestParamInfo<Scenario> const& info) { return info.param.name; });
