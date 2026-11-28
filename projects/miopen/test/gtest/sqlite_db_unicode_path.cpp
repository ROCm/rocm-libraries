// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// MIOpen must open its SQLite databases (kernel cache, user and system databases) in directories
// whose names have non-ASCII characters, such as a Windows user profile. sqlite3_open_v2() takes a
// UTF-8 file name, so on Windows the path must not be passed in the ANSI code page.
//
// The non-ASCII cases fail without the fix only on Windows with a non-UTF-8 ANSI code page (for
// example 1252 or 936); elsewhere the native narrow encoding is already UTF-8 and they always pass.
// SQLite also returns the file name in UTF-8 on every query (sqlite3_db_filename()). The last two
// names are for code page 936: their UTF-8 bytes are not a valid GBK sequence, so reading that
// name back through the ANSI code page throws on the first query after the open.

#include <miopen/config.h>

#if MIOPEN_ENABLE_SQLITE && !MIOPEN_EMBED_DB

#include <miopen/db_disable.hpp>
#include <miopen/filesystem.hpp>
#include <miopen/kern_db.hpp>
#include <miopen/sqlite_db.hpp>
#include <gtest/gtest.h>

#include <ostream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <process.h>
#define MIOPEN_TEST_GETPID _getpid
#else
#include <unistd.h>
#define MIOPEN_TEST_GETPID getpid
#endif

namespace fs = miopen::fs;

namespace {

// gtest prints each test parameter. Printing a std::u8string needs gtest itself built as C++20
// (testing::internal::PrintU8StringTo), so the parameter is a struct with its own PrintTo.
struct DirName
{
    const char* label;
    std::u8string utf8;
};

void PrintTo(const DirName& name, std::ostream* os) { *os << name.label; }

class UnicodeDir : public ::testing::TestWithParam<DirName>
{
protected:
    void SetUp() override
    {
        dir =
            fs::temp_directory_path() / (u8"miopen_sqlite_" + GetParam().utf8 + u8"_" +
                                         fs::path(std::to_string(MIOPEN_TEST_GETPID())).u8string());
        fs::remove_all(dir);
        ASSERT_TRUE(fs::create_directories(dir));
    }

    void TearDown() override
    {
        std::error_code ec;
        fs::remove_all(dir, ec);
    }

    fs::path dir;
};

class CPU_SQLiteUnicodePath_NONE : public UnicodeDir
{
};

class CPU_KernDbUnicodePath_NONE : public UnicodeDir
{
};

const auto kDirNames =
    testing::Values(DirName{"ascii", u8"ascii"},
                    DirName{"U+00E9", u8"Jos\u00e9"},             // in the Windows-1252 code page
                    DirName{"CJK", u8"\u9648\u9e4f\u5b87"},       // CJK, as in issue #11903
                    DirName{"U+674E_1", u8"\u674e1"},             // UTF-8 not valid as GBK
                    DirName{"U+738B_U+4F1F", u8"\u738b \u4f1f"}); // UTF-8 not valid as GBK

} // namespace

TEST_P(CPU_SQLiteUnicodePath_NONE, CreateAndReopen)
{
    const auto file = dir / "test.ukdb";
    {
        auto sql = miopen::SQLite{file, false};
        ASSERT_TRUE(sql.Valid()) << sql.ErrorMessage();
        sql.Exec("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, v TEXT);"
                 "INSERT INTO t (v) VALUES ('miopen');");
    }
    // The database must be created at the requested path, not at a mangled one
    ASSERT_TRUE(fs::exists(file));

    auto sql = miopen::SQLite{file, true};
    ASSERT_TRUE(sql.Valid()) << sql.ErrorMessage();
    const auto rows = sql.Exec("SELECT v FROM t;");
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_EQ(rows[0].at("v"), "miopen");
}

// The kernel cache goes through SQLiteBase (directory creation, error handling) and closes the
// database through SQLiteCloser, which reads the file name back from SQLite.
TEST_P(CPU_KernDbUnicodePath_NONE, StoreAndFind)
{
    // The kernel cache is a user database: with the user databases disabled it stores nothing
    if(miopen::IsUserDbDisabled())
        GTEST_SKIP() << "user databases are disabled";
    const auto file = dir / "cache" / "gfx000_1.ukdb";
    miopen::KernelConfig cfg;
    cfg.kernel_name = "kernel1";
    cfg.kernel_args = "-DNUM=1";
    cfg.kernel_blob = std::vector<char>(4096, 'x');
    {
        miopen::KernDb db(miopen::DbKinds::KernelDb, file, false);
        ASSERT_TRUE(db.StoreRecordUnsafe(cfg));
    }
    ASSERT_TRUE(fs::exists(file));

    miopen::KernDb db(miopen::DbKinds::KernelDb, file, false);
    const auto blob = db.FindRecordUnsafe(cfg);
    ASSERT_TRUE(blob);
    EXPECT_EQ(blob.value(), cfg.kernel_blob);
}

INSTANTIATE_TEST_SUITE_P(Smoke, CPU_SQLiteUnicodePath_NONE, kDirNames);
INSTANTIATE_TEST_SUITE_P(Smoke, CPU_KernDbUnicodePath_NONE, kDirNames);

#endif // MIOPEN_ENABLE_SQLITE && !MIOPEN_EMBED_DB
