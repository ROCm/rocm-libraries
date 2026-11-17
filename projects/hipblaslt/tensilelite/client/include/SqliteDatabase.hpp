// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef TENSILELITE_CLIENT_SQLITE_DATABASE_HPP
#define TENSILELITE_CLIENT_SQLITE_DATABASE_HPP

#include <cstdint>
#include <stdexcept>
#include <string>

// Forward declared so that users of this header do not need sqlite3.h.
struct sqlite3;
struct sqlite3_stmt;

namespace TensileLite
{
    namespace Client
    {
        // Thrown for any sqlite failure; the message includes sqlite's error message.
        class SqliteError : public std::runtime_error
        {
        public:
            using std::runtime_error::runtime_error;
        };

        class SqliteStatement;
        class SqliteTransaction;

        // Move-only RAII owner of a sqlite3 connection.
        class SqliteConnection
        {
        public:
            explicit SqliteConnection(std::string const& path);
            ~SqliteConnection();

            SqliteConnection(SqliteConnection&& other) noexcept;
            SqliteConnection& operator=(SqliteConnection&& other) noexcept;
            SqliteConnection(SqliteConnection const&)            = delete;
            SqliteConnection& operator=(SqliteConnection const&) = delete;

            void    execute(std::string const& sql);
            int64_t lastInsertRowId() const;
            int32_t changes() const;

        private:
            friend class SqliteStatement;
            friend class SqliteTransaction;

            sqlite3* get() const;

            sqlite3* m_db = nullptr;
        };

        // Non-copyable RAII owner of a prepared statement. Parameter and column indices
        // follow sqlite: parameters are 1-based, columns are 0-based.
        class SqliteStatement
        {
        public:
            SqliteStatement(SqliteConnection& db, std::string const& sql);
            ~SqliteStatement();

            SqliteStatement(SqliteStatement const&)            = delete;
            SqliteStatement& operator=(SqliteStatement const&) = delete;

            void bindInt(int32_t index, int64_t value);
            void bindDouble(int32_t index, double value);
            void bindText(int32_t index, std::string const& value);
            void bindBlob(int32_t index, std::string const& value);

            // Returns true on a row and false when done; throws on any other result.
            bool step();

            int64_t     columnInt(int32_t col) const;
            double      columnDouble(int32_t col) const;
            std::string columnText(int32_t col) const;
            std::string columnBlob(int32_t col) const;

            // Resets the statement and clears its bindings.
            void reset();

        private:
            sqlite3*      m_db   = nullptr;
            sqlite3_stmt* m_stmt = nullptr;
        };

        // BEGIN IMMEDIATE ... COMMIT, rolling back if commit was not reached.
        class SqliteTransaction
        {
        public:
            explicit SqliteTransaction(SqliteConnection& db);
            ~SqliteTransaction();

            SqliteTransaction(SqliteTransaction const&)            = delete;
            SqliteTransaction& operator=(SqliteTransaction const&) = delete;

            void commit();

        private:
            SqliteConnection& m_db;
            bool              m_committed = false;
        };
    }
}

#endif // TENSILELITE_CLIENT_SQLITE_DATABASE_HPP
