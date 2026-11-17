// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "SqliteDatabase.hpp"

#include <sqlite3.h>

#include <utility>

using namespace TensileLite::Client;

static constexpr int32_t busyTimeoutMs = 5000;

// A null handle means sqlite could not allocate the connection.
static std::string errorMessage(sqlite3* db)
{
    return db ? sqlite3_errmsg(db) : "out of memory";
}

[[noreturn]] static void throwSqliteError(sqlite3* db, std::string const& context)
{
    throw SqliteError("sqlite error: " + context + ": " + errorMessage(db));
}

static void checkResult(sqlite3* db, int32_t result, std::string const& context)
{
    if(result != SQLITE_OK)
        throwSqliteError(db, context);
}

SqliteConnection::SqliteConnection(std::string const& path)
{
    int32_t result
        = sqlite3_open_v2(path.c_str(), &m_db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr);
    if(result == SQLITE_OK)
        result = sqlite3_busy_timeout(m_db, busyTimeoutMs);
    if(result == SQLITE_OK)
        return;

    // A throwing constructor does not run the destructor, so the handle is closed here.
    std::string message = errorMessage(m_db);
    sqlite3_close_v2(m_db);
    m_db = nullptr;
    throw SqliteError("sqlite error: cannot open '" + path + "': " + message);
}

SqliteConnection::~SqliteConnection()
{
    sqlite3_close_v2(m_db);
}

SqliteConnection::SqliteConnection(SqliteConnection&& other) noexcept
    : m_db(std::exchange(other.m_db, nullptr))
{
}

SqliteConnection& SqliteConnection::operator=(SqliteConnection&& other) noexcept
{
    if(this == &other)
        return *this;
    sqlite3_close_v2(m_db);
    m_db = std::exchange(other.m_db, nullptr);
    return *this;
}

sqlite3* SqliteConnection::get() const
{
    return m_db;
}

void SqliteConnection::execute(std::string const& sql)
{
    char*   errorMessage = nullptr;
    int32_t result       = sqlite3_exec(m_db, sql.c_str(), nullptr, nullptr, &errorMessage);
    if(result == SQLITE_OK)
        return;

    std::string message = errorMessage ? errorMessage : sqlite3_errmsg(m_db);
    sqlite3_free(errorMessage);
    throw SqliteError("sqlite error: execute: " + message);
}

int64_t SqliteConnection::lastInsertRowId() const
{
    return sqlite3_last_insert_rowid(m_db);
}

int32_t SqliteConnection::changes() const
{
    return sqlite3_changes(m_db);
}

SqliteStatement::SqliteStatement(SqliteConnection& db, std::string const& sql)
    : m_db(db.get())
{
    checkResult(m_db,
                sqlite3_prepare_v2(m_db, sql.c_str(), -1, &m_stmt, nullptr),
                "cannot prepare statement");
}

SqliteStatement::~SqliteStatement()
{
    sqlite3_finalize(m_stmt);
}

void SqliteStatement::bindInt(int32_t index, int64_t value)
{
    checkResult(m_db, sqlite3_bind_int64(m_stmt, index, value), "bind integer");
}

void SqliteStatement::bindDouble(int32_t index, double value)
{
    checkResult(m_db, sqlite3_bind_double(m_stmt, index, value), "bind real");
}

void SqliteStatement::bindText(int32_t index, std::string const& value)
{
    checkResult(m_db,
                sqlite3_bind_text64(
                    m_stmt, index, value.data(), value.size(), SQLITE_TRANSIENT, SQLITE_UTF8),
                "bind text");
}

void SqliteStatement::bindBlob(int32_t index, std::string const& value)
{
    checkResult(m_db,
                sqlite3_bind_blob64(m_stmt, index, value.data(), value.size(), SQLITE_TRANSIENT),
                "bind blob");
}

bool SqliteStatement::step()
{
    int32_t result = sqlite3_step(m_stmt);
    if(result == SQLITE_ROW)
        return true;
    if(result == SQLITE_DONE)
        return false;
    throwSqliteError(m_db, "step");
}

int64_t SqliteStatement::columnInt(int32_t col) const
{
    return sqlite3_column_int64(m_stmt, col);
}

double SqliteStatement::columnDouble(int32_t col) const
{
    return sqlite3_column_double(m_stmt, col);
}

std::string SqliteStatement::columnText(int32_t col) const
{
    char const* text = reinterpret_cast<char const*>(sqlite3_column_text(m_stmt, col));
    if(!text)
        return {};
    return std::string(text, static_cast<size_t>(sqlite3_column_bytes(m_stmt, col)));
}

std::string SqliteStatement::columnBlob(int32_t col) const
{
    char const* data = reinterpret_cast<char const*>(sqlite3_column_blob(m_stmt, col));
    if(!data)
        return {};
    return std::string(data, static_cast<size_t>(sqlite3_column_bytes(m_stmt, col)));
}

void SqliteStatement::reset()
{
    // The result only repeats an error already reported by step.
    sqlite3_reset(m_stmt);
    sqlite3_clear_bindings(m_stmt);
}

SqliteTransaction::SqliteTransaction(SqliteConnection& db)
    : m_db(db)
{
    m_db.execute("BEGIN IMMEDIATE");
}

SqliteTransaction::~SqliteTransaction()
{
    if(m_committed)
        return;
    // A destructor must not throw, so a failed rollback is ignored; sqlite rolls back on close.
    sqlite3_exec(m_db.get(), "ROLLBACK", nullptr, nullptr, nullptr);
}

void SqliteTransaction::commit()
{
    m_db.execute("COMMIT");
    m_committed = true;
}
