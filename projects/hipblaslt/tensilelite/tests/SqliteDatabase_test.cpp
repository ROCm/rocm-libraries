// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "SqliteDatabase.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

using namespace TensileLite::Client;

static int64_t countRows(SqliteConnection& db)
{
    SqliteStatement statement(db, "SELECT COUNT(*) FROM t");
    EXPECT_TRUE(statement.step());
    return statement.columnInt(0);
}

TEST(SqliteDatabase, FailingStatementThrowsSqliteError)
{
    SqliteConnection db(":memory:");
    EXPECT_THROW(db.execute("SELECT * FROM missing"), SqliteError);
    EXPECT_THROW(SqliteStatement(db, "NOT SQL"), SqliteError);
}

TEST(SqliteDatabase, UncommittedTransactionRollsBack)
{
    SqliteConnection db(":memory:");
    db.execute("CREATE TABLE t(x)");
    {
        SqliteTransaction transaction(db);
        db.execute("INSERT INTO t VALUES(1)");
    }
    EXPECT_EQ(countRows(db), 0);
    {
        SqliteTransaction transaction(db);
        db.execute("INSERT INTO t VALUES(1)");
        transaction.commit();
    }
    EXPECT_EQ(countRows(db), 1);
}

TEST(SqliteDatabase, MovedFromConnectionIsSafe)
{
    SqliteConnection first(":memory:");
    first.execute("CREATE TABLE t(x)");
    SqliteConnection second(std::move(first));
    second.execute("INSERT INTO t VALUES(1)");
    EXPECT_EQ(countRows(second), 1);

    SqliteConnection third(":memory:");
    third = std::move(second);
    EXPECT_EQ(countRows(third), 1);
}

TEST(SqliteDatabase, ValuesRoundTrip)
{
    SqliteConnection db(":memory:");
    db.execute("CREATE TABLE t(i, d, s, b, n)");

    // Embedded NUL and non-ASCII bytes must survive in both text-like columns.
    std::string const blob("a\0b\xff", 4);
    int64_t const     bigInt = std::numeric_limits<int64_t>::min();
    {
        SqliteStatement insert(db, "INSERT INTO t(i, d, s, b) VALUES(?1, ?2, ?3, ?4)");
        insert.bindInt(1, bigInt);
        insert.bindDouble(2, 0.1 + 0.2);
        insert.bindText(3, "text");
        insert.bindBlob(4, blob);
        EXPECT_FALSE(insert.step());
    }
    EXPECT_EQ(db.changes(), 1);
    EXPECT_EQ(db.lastInsertRowId(), 1);

    SqliteStatement select(db, "SELECT i, d, s, b, n FROM t");
    ASSERT_TRUE(select.step());
    EXPECT_EQ(select.columnInt(0), bigInt);
    EXPECT_EQ(select.columnDouble(1), 0.1 + 0.2);
    EXPECT_EQ(select.columnText(2), "text");
    EXPECT_EQ(select.columnBlob(3), blob);
    EXPECT_EQ(select.columnBlob(3).size(), 4u);
    // A NULL column reads back as empty text, a zero integer and a zero double.
    EXPECT_EQ(select.columnText(4), "");
    EXPECT_EQ(select.columnInt(4), 0);
    EXPECT_EQ(select.columnDouble(4), 0.0);
}

TEST(SqliteDatabase, StepReportsEndAndResetAllowsRebinding)
{
    SqliteConnection db(":memory:");
    db.execute("CREATE TABLE t(x)");
    db.execute("INSERT INTO t VALUES(1), (2), (3)");

    SqliteStatement select(db, "SELECT x FROM t WHERE x >= ?1 ORDER BY x");
    select.bindInt(1, 2);
    ASSERT_TRUE(select.step());
    EXPECT_EQ(select.columnInt(0), 2);
    ASSERT_TRUE(select.step());
    EXPECT_EQ(select.columnInt(0), 3);
    EXPECT_FALSE(select.step());

    select.reset();
    select.bindInt(1, 3);
    ASSERT_TRUE(select.step());
    EXPECT_EQ(select.columnInt(0), 3);
    EXPECT_FALSE(select.step());

    // Reset clears the bindings: the unbound parameter is NULL and matches nothing.
    select.reset();
    EXPECT_FALSE(select.step());
}

TEST(SqliteDatabase, OpeningAnInvalidPathThrowsSqliteError)
{
    EXPECT_THROW(SqliteConnection("/nonexistent_directory_for_test/db.sqlite"), SqliteError);
}

TEST(SqliteDatabase, TransactionLeftByExceptionRollsBack)
{
    SqliteConnection db(":memory:");
    db.execute("CREATE TABLE t(x)");
    try
    {
        SqliteTransaction transaction(db);
        db.execute("INSERT INTO t VALUES(1)");
        throw std::runtime_error("scope left");
    }
    catch(std::runtime_error const&)
    {
    }
    EXPECT_EQ(countRows(db), 0);

    // The rollback released the write lock, so a new transaction can start and commit.
    SqliteTransaction transaction(db);
    db.execute("INSERT INTO t VALUES(2)");
    transaction.commit();
    EXPECT_EQ(countRows(db), 1);
}

TEST(SqliteDatabase, CommitThenNewTransactionWorks)
{
    SqliteConnection db(":memory:");
    db.execute("CREATE TABLE t(x)");
    {
        SqliteTransaction transaction(db);
        db.execute("INSERT INTO t VALUES(1)");
        transaction.commit();
    }
    {
        SqliteTransaction transaction(db);
        db.execute("INSERT INTO t VALUES(2)");
        transaction.commit();
    }
    EXPECT_EQ(countRows(db), 2);
}
