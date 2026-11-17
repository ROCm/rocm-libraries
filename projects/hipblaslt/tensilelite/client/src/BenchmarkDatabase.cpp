// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "BenchmarkDatabase.hpp"

#include <algorithm>
#include <chrono>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace TensileLite::Client;

// Stored in PRAGMA user_version, which a foreign or unversioned file leaves at 0.
static constexpr int32_t schemaVersion = 1;

// The only client_configs row; the schema's CHECK(id = 1) keeps it the only one.
static constexpr int64_t clientConfigId = 1;

// The status CHECKs below mirror RunStatus.
// Only applied to an empty database, so the tables are created without IF NOT EXISTS.
static char const* const schemaSql = R"sql(
CREATE TABLE meta(
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL);
CREATE TABLE solutions(
    id INTEGER PRIMARY KEY,
    library_index INTEGER NOT NULL UNIQUE,
    name TEXT NOT NULL,
    spec BLOB NOT NULL,
    spec_format TEXT NOT NULL);
CREATE TABLE client_configs(
    id INTEGER PRIMARY KEY CHECK(id = 1),
    ini TEXT NOT NULL);
CREATE TABLE init_configs(
    id INTEGER PRIMARY KEY,
    config_id INTEGER NOT NULL REFERENCES client_configs(id),
    problem_index INTEGER NOT NULL UNIQUE);
CREATE TABLE problem_values(
    problem_id INTEGER NOT NULL REFERENCES init_configs(id),
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY(problem_id, key));
CREATE TABLE sessions(
    id INTEGER PRIMARY KEY,
    pid INTEGER NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL,
    init_seed INTEGER NOT NULL);
CREATE TABLE runs(
    problem_id INTEGER NOT NULL REFERENCES init_configs(id),
    solution_id INTEGER NOT NULL REFERENCES solutions(id),
    status INTEGER NOT NULL DEFAULT 0 CHECK(status BETWEEN 0 AND 6),
    session_id INTEGER REFERENCES sessions(id),
    start_time REAL,
    end_time REAL,
    validation TEXT,
    caused_by_problem_id INTEGER REFERENCES init_configs(id),
    PRIMARY KEY(problem_id, solution_id),
    -- Only the indirect statuses (4 and 5) have a causing problem.
    CHECK((caused_by_problem_id IS NOT NULL) = (status IN (4, 5))));
-- Serves the per-solution status lookups of the crash marking.
CREATE INDEX runs_solution_status ON runs(solution_id, status);
CREATE TABLE run_values(
    problem_id INTEGER NOT NULL,
    solution_id INTEGER NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY(problem_id, solution_id, key),
    FOREIGN KEY(problem_id, solution_id) REFERENCES runs(problem_id, solution_id));
CREATE TABLE iterations(
    problem_id INTEGER NOT NULL,
    solution_id INTEGER NOT NULL,
    sync_index INTEGER NOT NULL,
    -- NULL when the measured time was NaN, which sqlite stores as NULL.
    time_us REAL,
    enqueues INTEGER NOT NULL,
    PRIMARY KEY(problem_id, solution_id, sync_index),
    FOREIGN KEY(problem_id, solution_id) REFERENCES runs(problem_id, solution_id));
)sql";

// Only a verification verdict spreads to the other pairs of a solution; other failures are
// per-size rejections (including INVALID, which also covers environment errors).
static char const* const verificationFailedVerdict   = "FAILED";
static char const* const indirectCrashVerdict        = "INDIRECTLY_FAILED";
static char const* const indirectVerificationVerdict = "INDIRECT_VERIFICATION_FAILURE";

static double nowEpochSeconds()
{
    using Seconds = std::chrono::duration<double>;
    return std::chrono::duration_cast<Seconds>(std::chrono::system_clock::now().time_since_epoch())
        .count();
}

// Runs a statement that returns no rows and leaves it ready for reuse.
static void executeOnce(SqliteStatement& statement)
{
    statement.step();
    statement.reset();
}

[[noreturn]] static void throwMismatch(std::string const& what)
{
    throw BenchmarkDatabaseMismatch("benchmark database mismatch: " + what);
}

[[noreturn]] static void throwDatabaseError(std::string const& message)
{
    throw BenchmarkDatabaseError("benchmark database error: " + message);
}

static std::string describePair(int64_t problemIndex, int64_t solutionIndex)
{
    return "problem " + std::to_string(problemIndex) + " and solution "
           + std::to_string(solutionIndex);
}

static void bindStatus(SqliteStatement& statement, int32_t index, RunStatus status)
{
    statement.bindInt(index, static_cast<int32_t>(status));
}

// Resolves a public index to the table id that the other tables reference.
static int64_t
    lookupId(SqliteConnection& db, std::string const& sql, std::string const& kind, int64_t index)
{
    SqliteStatement statement(db, sql);
    statement.bindInt(1, index);
    if(!statement.step())
        throwDatabaseError("no " + kind + " " + std::to_string(index));
    return statement.columnInt(0);
}

static int64_t problemId(SqliteConnection& db, int64_t problemIndex)
{
    return lookupId(
        db, "SELECT id FROM init_configs WHERE problem_index = ?1", "problem", problemIndex);
}

static int64_t solutionId(SqliteConnection& db, int64_t solutionIndex)
{
    return lookupId(
        db, "SELECT id FROM solutions WHERE library_index = ?1", "solution", solutionIndex);
}

static bool hasTables(SqliteConnection& db)
{
    SqliteStatement statement(db,
                              "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'table')");
    statement.step();
    return statement.columnInt(0) != 0;
}

static int64_t userVersion(SqliteConnection& db)
{
    SqliteStatement statement(db, "PRAGMA user_version");
    statement.step();
    return statement.columnInt(0);
}

// A file with tables but another version is either an older database or not ours at all.
static void checkSchemaVersion(SqliteConnection& db)
{
    int64_t found = userVersion(db);
    if(found == schemaVersion)
        return;
    throwMismatch("schema version is " + std::to_string(found) + " in the database but "
                  + std::to_string(schemaVersion)
                  + " is required; it is not a benchmark database of this client");
}

// Concurrent readers and the crash-safe commit protocol need WAL, which a filesystem without
// shared memory cannot provide; sqlite then silently keeps another mode.
static void enableWriteAheadLogging(SqliteConnection& db, std::string const& path)
{
    SqliteStatement statement(db, "PRAGMA journal_mode=WAL");
    statement.step();
    std::string mode = statement.columnText(0);
    if(mode == "wal")
        return;
    throwDatabaseError("cannot enable write-ahead logging on '" + path + "' (journal mode is '"
                       + mode + "'); the filesystem may not support shared memory");
}

// The journal mode is only changed after the file is known to be ours or empty, because
// switching it writes the header of whatever file is there.
static SqliteConnection openDatabase(std::string const& path)
{
    SqliteConnection db(path);
    db.execute("PRAGMA foreign_keys=ON");
    if(hasTables(db))
        checkSchemaVersion(db);
    enableWriteAheadLogging(db, path);
    // A GPU fault can reset the node, and NORMAL may then lose the committed Unknown marker.
    db.execute("PRAGMA synchronous=FULL");
    return db;
}

static void createSchema(SqliteConnection& db)
{
    db.execute(schemaSql);
    // The pragma takes no parameters; it is transactional, so it rolls back with the schema.
    db.execute("PRAGMA user_version = " + std::to_string(schemaVersion));
}

static void checkCount(std::string const& kind, int64_t inDatabase, int64_t inRun)
{
    if(inDatabase == inRun)
        return;
    throwMismatch(kind + " count differs: database has " + std::to_string(inDatabase) + ", run has "
                  + std::to_string(inRun));
}

// Runs inside the transaction that creates the schema.
static void populate(SqliteConnection&                    db,
                     std::vector<SolutionRecord> const&   solutions,
                     std::string const&                   clientConfig,
                     std::vector<InitConfigRecord> const& initConfigs)
{
    SqliteStatement insertSolution(
        db,
        "INSERT INTO solutions(library_index, name, spec, spec_format) VALUES(?1, ?2, ?3, ?4)");
    for(SolutionRecord const& solution : solutions)
    {
        insertSolution.bindInt(1, solution.libraryIndex);
        insertSolution.bindText(2, solution.name);
        insertSolution.bindBlob(3, solution.spec);
        insertSolution.bindText(4, solution.specFormat);
        executeOnce(insertSolution);
    }

    SqliteStatement insertClientConfig(db, "INSERT INTO client_configs(id, ini) VALUES(?1, ?2)");
    insertClientConfig.bindInt(1, clientConfigId);
    insertClientConfig.bindText(2, clientConfig);
    executeOnce(insertClientConfig);

    SqliteStatement insertConfig(
        db, "INSERT INTO init_configs(config_id, problem_index) VALUES(?1, ?2)");
    for(InitConfigRecord const& config : initConfigs)
    {
        insertConfig.bindInt(1, clientConfigId);
        insertConfig.bindInt(2, config.problemIndex);
        executeOnce(insertConfig);
    }

    db.execute("INSERT INTO runs(problem_id, solution_id) "
               "SELECT i.id, s.id FROM init_configs i CROSS JOIN solutions s");
}

static void validateSolutions(SqliteConnection& db, std::vector<SolutionRecord> const& solutions)
{
    std::map<int64_t, SolutionRecord> stored;
    SqliteStatement statement(db, "SELECT library_index, name, spec, spec_format FROM solutions");
    while(statement.step())
    {
        int64_t index = statement.columnInt(0);
        stored[index] = SolutionRecord{
            index, statement.columnText(1), statement.columnBlob(2), statement.columnText(3)};
    }

    checkCount(
        "solution", static_cast<int64_t>(stored.size()), static_cast<int64_t>(solutions.size()));
    for(SolutionRecord const& solution : solutions)
    {
        std::string label = "solution " + std::to_string(solution.libraryIndex);
        auto        it    = stored.find(solution.libraryIndex);
        if(it == stored.end())
            throwMismatch(label + " is missing from the database");
        if(it->second.name != solution.name)
            throwMismatch(label + " is named '" + it->second.name + "' in the database but '"
                          + solution.name + "' in this run");
        if(it->second.specFormat != solution.specFormat)
            throwMismatch(label + " spec is stored as " + it->second.specFormat
                          + " in the database but this run writes " + solution.specFormat);
        if(it->second.spec != solution.spec)
            throwMismatch(label + " spec differs");
    }
}

static std::vector<std::string> splitLines(std::string const& text)
{
    std::vector<std::string> lines;
    std::istringstream       stream(text);
    for(std::string line; std::getline(stream, line);)
        lines.push_back(line);
    return lines;
}

static void validateClientConfig(SqliteConnection& db, std::string const& clientConfig)
{
    SqliteStatement statement(db, "SELECT ini FROM client_configs");
    std::vector<std::string> stored;
    while(statement.step())
        stored.push_back(statement.columnText(0));

    checkCount("client config", static_cast<int64_t>(stored.size()), 1);
    if(stored.front() == clientConfig)
        return;

    std::vector<std::string> inDatabase = splitLines(stored.front());
    std::vector<std::string> inRun      = splitLines(clientConfig);
    int64_t lineCount = static_cast<int64_t>(std::max(inDatabase.size(), inRun.size()));
    for(int64_t i = 0; i < lineCount; i++)
    {
        std::string const& a
            = i < static_cast<int64_t>(inDatabase.size()) ? inDatabase[i] : std::string();
        std::string const& b = i < static_cast<int64_t>(inRun.size()) ? inRun[i] : std::string();
        if(a == b)
            continue;
        std::string const& line = a.empty() ? b : a;
        throwMismatch("client configuration differs at '" + line.substr(0, line.find('=')) + "'");
    }
    throwMismatch("client configuration differs");
}

static void validateInitConfigs(SqliteConnection&                    db,
                                std::vector<InitConfigRecord> const& initConfigs)
{
    std::set<int64_t> stored;
    SqliteStatement   statement(db, "SELECT problem_index FROM init_configs");
    while(statement.step())
        stored.insert(statement.columnInt(0));

    checkCount("init config",
               static_cast<int64_t>(stored.size()),
               static_cast<int64_t>(initConfigs.size()));
    for(InitConfigRecord const& config : initConfigs)
        if(!stored.count(config.problemIndex))
            throwMismatch("init config " + std::to_string(config.problemIndex)
                          + " is missing from the database");
}

// ---------------------------------------------------------------------------
// BenchmarkDatabase
// ---------------------------------------------------------------------------

BenchmarkDatabase::BenchmarkDatabase(std::string const&                   path,
                                     std::vector<SolutionRecord> const&   solutions,
                                     std::string const&                   clientConfig,
                                     std::vector<InitConfigRecord> const& initConfigs)
    : m_db(openDatabase(path))
{
    // Classified again under the write lock, because another process may have created the
    // database since it was opened.
    SqliteTransaction transaction(m_db);
    if(!hasTables(m_db))
    {
        createSchema(m_db);
        populate(m_db, solutions, clientConfig, initConfigs);
        transaction.commit();
        return;
    }
    checkSchemaVersion(m_db);
    validateSolutions(m_db, solutions);
    validateClientConfig(m_db, clientConfig);
    validateInitConfigs(m_db, initConfigs);
    transaction.commit();
}

void BenchmarkDatabase::setMeta(std::string const& key, std::string const& value)
{
    SqliteStatement statement(m_db, "INSERT OR REPLACE INTO meta(key, value) VALUES(?1, ?2)");
    statement.bindText(1, key);
    statement.bindText(2, value);
    executeOnce(statement);
}

// Whether the most recent session never ended.
static bool lastSessionDied(SqliteConnection& db)
{
    SqliteStatement statement(db, "SELECT end_time IS NULL FROM sessions ORDER BY id DESC LIMIT 1");
    return statement.step() && statement.columnInt(0) != 0;
}

static int64_t insertSession(SqliteConnection& db, int64_t pid, int64_t initSeed)
{
    SqliteStatement statement(
        db, "INSERT INTO sessions(pid, start_time, init_seed) VALUES(?1, ?2, ?3)");
    statement.bindInt(1, pid);
    statement.bindDouble(2, nowEpochSeconds());
    statement.bindInt(3, initSeed);
    executeOnce(statement);
    return db.lastInsertRowId();
}

// The marked pairs never ran, so resetting them restores the pending state.
static void resetMarkedPairs(SqliteConnection& db, RunStatus marked)
{
    SqliteStatement statement(
        db,
        "UPDATE runs SET status = ?1, session_id = NULL, start_time = NULL, end_time = NULL, "
        "validation = NULL, caused_by_problem_id = NULL WHERE status = ?2");
    bindStatus(statement, 1, RunStatus::NotBenchmarked);
    bindStatus(statement, 2, marked);
    executeOnce(statement);
}

// Every Unknown pair belongs to a dead session because the new one has not begun a run.
static void markCrashedSolutions(SqliteConnection& db, int64_t sessionId)
{
    SqliteStatement statement(db,
                              "UPDATE runs SET status = ?1, session_id = ?2, validation = ?3, "
                              "caused_by_problem_id = (SELECT MIN(u.problem_id) FROM runs u "
                              "WHERE u.solution_id = runs.solution_id AND u.status = ?4) "
                              "WHERE status = ?5 AND EXISTS (SELECT 1 FROM runs u "
                              "WHERE u.solution_id = runs.solution_id AND u.status = ?4)");
    bindStatus(statement, 1, RunStatus::IndirectlyFailed);
    statement.bindInt(2, sessionId);
    statement.bindText(3, indirectCrashVerdict);
    bindStatus(statement, 4, RunStatus::Unknown);
    bindStatus(statement, 5, RunStatus::NotBenchmarked);
    executeOnce(statement);
}

int64_t BenchmarkDatabase::beginSession(int64_t                      pid,
                                        int64_t                      initSeed,
                                        IndirectFailurePolicy const& policy)
{
    SqliteTransaction transaction(m_db);
    bool              tainted   = lastSessionDied(m_db);
    int64_t           sessionId = insertSession(m_db, pid, initSeed);

    if(!policy.crashes)
        resetMarkedPairs(m_db, RunStatus::IndirectlyFailed);
    if(!policy.verificationFailures)
        resetMarkedPairs(m_db, RunStatus::IndirectVerificationFailure);
    if(policy.crashes)
        markCrashedSolutions(m_db, sessionId);
    transaction.commit();

    m_sessionId = sessionId;
    m_policy    = policy;
    m_tainted   = tainted;
    return sessionId;
}

void BenchmarkDatabase::endSession()
{
    if(!m_sessionId)
        throwDatabaseError("no session to end");

    SqliteStatement statement(m_db, "UPDATE sessions SET end_time = ?1 WHERE id = ?2");
    statement.bindDouble(1, nowEpochSeconds());
    statement.bindInt(2, *m_sessionId);
    executeOnce(statement);
}

void BenchmarkDatabase::setProblemValues(int64_t problemIndex, KeyValues const& values)
{
    int64_t           id = problemId(m_db, problemIndex);
    SqliteTransaction transaction(m_db);
    SqliteStatement   statement(
        m_db, "INSERT OR REPLACE INTO problem_values(problem_id, key, value) VALUES(?1, ?2, ?3)");
    for(auto const& [key, value] : values)
    {
        statement.bindInt(1, id);
        statement.bindText(2, key);
        statement.bindText(3, value);
        executeOnce(statement);
    }
    transaction.commit();
}

void BenchmarkDatabase::beginRun(int64_t problemIndex, int64_t solutionIndex)
{
    if(!m_sessionId)
        throwDatabaseError("no session to begin a run in");

    int64_t           problem  = problemId(m_db, problemIndex);
    int64_t           solution = solutionId(m_db, solutionIndex);
    SqliteTransaction transaction(m_db);
    SqliteStatement   statement(m_db,
                                "UPDATE runs SET status = ?1, session_id = ?2, start_time = ?3 "
                                "WHERE problem_id = ?4 AND solution_id = ?5 AND status = ?6");
    bindStatus(statement, 1, RunStatus::Unknown);
    statement.bindInt(2, *m_sessionId);
    statement.bindDouble(3, nowEpochSeconds());
    statement.bindInt(4, problem);
    statement.bindInt(5, solution);
    bindStatus(statement, 6, RunStatus::NotBenchmarked);
    executeOnce(statement);
    if(m_db.changes() == 0)
        throwDatabaseError("run of " + describePair(problemIndex, solutionIndex)
                           + " is not pending");
    transaction.commit();
}

static void insertRunValues(SqliteConnection& db,
                            int64_t           problem,
                            int64_t           solution,
                            KeyValues const&  values)
{
    SqliteStatement statement(
        db, "INSERT INTO run_values(problem_id, solution_id, key, value) VALUES(?1, ?2, ?3, ?4)");
    for(auto const& [key, value] : values)
    {
        statement.bindInt(1, problem);
        statement.bindInt(2, solution);
        statement.bindText(3, key);
        statement.bindText(4, value);
        executeOnce(statement);
    }
}

static void insertIterations(SqliteConnection&                   db,
                             int64_t                             problem,
                             int64_t                             solution,
                             std::vector<IterationRecord> const& iterations)
{
    SqliteStatement statement(
        db,
        "INSERT INTO iterations(problem_id, solution_id, sync_index, time_us, enqueues) "
        "VALUES(?1, ?2, ?3, ?4, ?5)");
    for(IterationRecord const& iteration : iterations)
    {
        statement.bindInt(1, problem);
        statement.bindInt(2, solution);
        statement.bindInt(3, iteration.syncIndex);
        statement.bindDouble(4, iteration.timeUs);
        statement.bindInt(5, iteration.enqueues);
        executeOnce(statement);
    }
}

static void markVerificationFailedSolution(SqliteConnection& db,
                                           int64_t           problem,
                                           int64_t           solution,
                                           int64_t           sessionId)
{
    SqliteStatement statement(db,
                              "UPDATE runs SET status = ?1, session_id = ?2, validation = ?3, "
                              "caused_by_problem_id = ?4 WHERE solution_id = ?5 AND status = ?6");
    bindStatus(statement, 1, RunStatus::IndirectVerificationFailure);
    statement.bindInt(2, sessionId);
    statement.bindText(3, indirectVerificationVerdict);
    statement.bindInt(4, problem);
    statement.bindInt(5, solution);
    bindStatus(statement, 6, RunStatus::NotBenchmarked);
    executeOnce(statement);
}

void BenchmarkDatabase::endRun(int64_t                             problemIndex,
                               int64_t                             solutionIndex,
                               RunStatus                           status,
                               std::string const&                  validation,
                               KeyValues const&                    values,
                               std::vector<IterationRecord> const& iterations)
{
    if(!m_sessionId)
        throwDatabaseError("no session to end a run in");
    if(status != RunStatus::Pass && status != RunStatus::Failure)
        throwDatabaseError("a run must end as pass or failure");

    int64_t           problem  = problemId(m_db, problemIndex);
    int64_t           solution = solutionId(m_db, solutionIndex);
    SqliteTransaction transaction(m_db);
    SqliteStatement   statement(m_db,
                                "UPDATE runs SET status = ?1, end_time = ?2, validation = ?3 "
                                "WHERE problem_id = ?4 AND solution_id = ?5 AND status = ?6");
    bindStatus(statement, 1, status);
    statement.bindDouble(2, nowEpochSeconds());
    statement.bindText(3, validation);
    statement.bindInt(4, problem);
    statement.bindInt(5, solution);
    bindStatus(statement, 6, RunStatus::Unknown);
    executeOnce(statement);
    if(m_db.changes() == 0)
        throwDatabaseError("run of " + describePair(problemIndex, solutionIndex)
                           + " was not started");

    insertRunValues(m_db, problem, solution, values);
    insertIterations(m_db, problem, solution, iterations);
    if(status == RunStatus::Failure && validation == verificationFailedVerdict
       && m_policy.verificationFailures && !m_tainted)
        markVerificationFailedSolution(m_db, problem, solution, *m_sessionId);
    transaction.commit();
}

void BenchmarkDatabase::markNotSelected(int64_t problemIndex)
{
    if(!m_sessionId)
        throwDatabaseError("no session to mark pairs not selected in");

    int64_t           problem = problemId(m_db, problemIndex);
    SqliteTransaction transaction(m_db);
    // Only the session id is recorded, like the indirect markings: the pair never ran.
    SqliteStatement statement(
        m_db, "UPDATE runs SET status = ?1, session_id = ?2 WHERE problem_id = ?3 AND status = ?4");
    bindStatus(statement, 1, RunStatus::NotSelected);
    statement.bindInt(2, *m_sessionId);
    statement.bindInt(3, problem);
    bindStatus(statement, 4, RunStatus::NotBenchmarked);
    executeOnce(statement);
    transaction.commit();
}

RunStatus BenchmarkDatabase::runStatus(int64_t problemIndex, int64_t solutionIndex) const
{
    int64_t         problem  = problemId(m_db, problemIndex);
    int64_t         solution = solutionId(m_db, solutionIndex);
    SqliteStatement statement(m_db,
                              "SELECT status FROM runs WHERE problem_id = ?1 AND solution_id = ?2");
    statement.bindInt(1, problem);
    statement.bindInt(2, solution);
    if(!statement.step())
        throwDatabaseError("no run for " + describePair(problemIndex, solutionIndex));
    return static_cast<RunStatus>(statement.columnInt(0));
}

bool BenchmarkDatabase::isPending(int64_t problemIndex, int64_t solutionIndex) const
{
    return runStatus(problemIndex, solutionIndex) == RunStatus::NotBenchmarked;
}

bool BenchmarkDatabase::hasPendingRuns(int64_t problemIndex) const
{
    int64_t         problem = problemId(m_db, problemIndex);
    SqliteStatement statement(
        m_db, "SELECT EXISTS(SELECT 1 FROM runs WHERE status = ?1 AND problem_id = ?2)");
    bindStatus(statement, 1, RunStatus::NotBenchmarked);
    statement.bindInt(2, problem);
    statement.step();
    return statement.columnInt(0) != 0;
}
