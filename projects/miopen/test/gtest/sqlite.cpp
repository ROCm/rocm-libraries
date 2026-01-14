// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <fstream>
#include <gtest/gtest.h>
#include <miopen/sqlite_db.hpp>
#include <miopen/temp_file.hpp>

const char* const lfs_db = R"(version https://git-lfs.github.com/spec/v1
oid sha256:cc45c32e44560074b5e4b0c0e48472a86e6b3bb1c73c189580f950f098d2a8d7
size 357490688)";

struct DummyDB
{
};

bool test_lfs_db(bool is_system)
{
    miopen::TempFile tmp_db{"test_lfs_db"};
    // write file to temp file
    std::ofstream tmp_db_file(tmp_db.Path());
    tmp_db_file << lfs_db;
    tmp_db_file.close();
    // construct a db out of it
    miopen::SQLiteBase<DummyDB> lfs_sqdb{miopen::DbKinds::PerfDb, tmp_db, is_system};
    return lfs_sqdb.dbInvalid;
}

struct CPU_Sqlite_NONE : public ::testing::TestWithParam<int>
{
};

TEST_P(CPU_Sqlite_NONE, LfsDbSystem)
{
    (void)GetParam(); // Parameter unused but required for TEST_P pattern
    // System DB should pass, since the lfs file was installed in the sys directory
    EXPECT_TRUE(test_lfs_db(true));
}

#if !MIOPEN_EMBED_DB
TEST_P(CPU_Sqlite_NONE, LfsDbUser)
{
    (void)GetParam(); // Parameter unused but required for TEST_P pattern
    // User db should fail since MIOpen should not create such a file
    // ever, if it exists its a corrupt file which should be reported.
    EXPECT_THROW(test_lfs_db(false), std::exception);
}
#endif

INSTANTIATE_TEST_SUITE_P(Smoke, CPU_Sqlite_NONE, testing::Values(0));
