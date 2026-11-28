// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <miopen/filesystem.hpp>
#include <miopen/readonlyramdb.hpp>

#include <fstream>
#include <random>
#include <string>
#include <string_view>
#include <utility>

namespace {

namespace fs = miopen::fs;

struct CPU_ReadonlyRamDb_NONE : public ::testing::TestWithParam<int>
{
    void SetUp() override
    {
        // Builds that embed the system dbs (MIOPEN_EMBED_DB) would look these files up in the
        // embedded filesystem. Read them from disk instead.
        saved_embed_fs_override = std::exchange(miopen::debug::rordb_embed_fs_override(), true);
        std::random_device rd;
        dir = fs::temp_directory_path() /
              ("miopen_rordb_" + std::to_string(rd()) + std::to_string(rd()));
        fs::create_directories(dir);
    }

    void TearDown() override
    {
        std::error_code ec;
        fs::remove_all(dir, ec);
        miopen::debug::rordb_embed_fs_override() = saved_embed_fs_override;
    }

    fs::path WriteDb(const std::string& name, std::string_view bytes) const
    {
        const auto path = dir / name;
        std::ofstream out{path, std::ios::binary};
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        out.close();
        EXPECT_TRUE(fs::is_regular_file(path));
        return path;
    }

    // Every test uses its own path, as GetCached keeps one instance per path until the process
    // exits.
    static const miopen::ReadonlyRamDb& Load(const fs::path& path)
    {
        return miopen::ReadonlyRamDb::GetCached(miopen::DbKinds::FindDb, path, false);
    }

    static void ExpectRecord(const miopen::ReadonlyRamDb& db,
                             std::string_view key,
                             std::string_view content,
                             int line)
    {
        const auto& map = db.GetCacheMap();
        const auto it   = map.find(key);
        ASSERT_NE(it, map.end()) << "key not loaded: " << key;
        EXPECT_EQ(it->second.content, content) << "key: " << key;
        EXPECT_EQ(it->second.line, line) << "key: " << key;
    }

    fs::path dir;
    bool saved_embed_fs_override = false;
};

} // namespace

TEST_P(CPU_ReadonlyRamDb_NONE, ParsesLinesLikeTheStreamReader)
{
    // clang-format off
    const auto path = WriteDb("lines.db.txt",
        "crlf=a:1;b:2\r\n"      // 1: CRLF line, the carriage return is not part of the content
        "\r\n"                  // 2: CR only line, no record
        "lf=c:3\n"              // 3: plain line
        "dup=first:1\n"         // 4
        "dup=second:2\n"        // 5: duplicate key, the first record wins
        "\n"                    // 6: empty line
        "no_separator\n"        // 7: ill-formed, no '='
        "=no_key\n"             // 8: ill-formed, empty key
        "empty_value=\n"        // 9: a record with no content
        "last=d:4;e:5");        // 10: unterminated last line
    // clang-format on

    const auto& db = Load(path);

    EXPECT_EQ(db.GetCacheMap().size(), 5u);
    ExpectRecord(db, "crlf", "a:1;b:2", 1);
    ExpectRecord(db, "lf", "c:3", 3);
    ExpectRecord(db, "dup", "first:1", 4);
    ExpectRecord(db, "empty_value", "", 9);
    ExpectRecord(db, "last", "d:4;e:5", 10);

    // The record is found through the public lookup as well, and the same path gives the same
    // instance.
    EXPECT_TRUE(db.FindRecord(std::string{"crlf"}).has_value());
    EXPECT_FALSE(db.FindRecord(std::string{"no_separator"}).has_value());
    EXPECT_EQ(&db, &Load(path));
}

TEST_P(CPU_ReadonlyRamDb_NONE, UnterminatedCrLastLine)
{
    const auto path = WriteDb("cr_last.db.txt", "first=a:1\r\nlast=b:2\r");

    const auto& db = Load(path);

    EXPECT_EQ(db.GetCacheMap().size(), 2u);
    ExpectRecord(db, "first", "a:1", 1);
    ExpectRecord(db, "last", "b:2", 2);
}

TEST_P(CPU_ReadonlyRamDb_NONE, EmptyFileLoadsNothing)
{
    const auto& db = Load(WriteDb("empty.db.txt", ""));

    EXPECT_TRUE(db.GetCacheMap().empty());
}

TEST_P(CPU_ReadonlyRamDb_NONE, MissingFileLoadsNothing)
{
    const auto& db = Load(dir / "missing.db.txt");

    EXPECT_TRUE(db.GetCacheMap().empty());
}

TEST_P(CPU_ReadonlyRamDb_NONE, DirectoryAtDbPathLoadsNothing)
{
    // A path that can't be sized or opened, a directory for instance, is as unreadable as a
    // missing file: it loads nothing and does not throw.
    const auto path = dir / "directory.db.txt";
    fs::create_directories(path);

    const miopen::ReadonlyRamDb* db = nullptr;
    ASSERT_NO_THROW(db = &Load(path));
    EXPECT_TRUE(db->GetCacheMap().empty());
    EXPECT_FALSE(db->FindRecord(std::string{"any"}).has_value());

    // The instance GetCached keeps for that path is complete: a second call does not throw and does
    // not load anything either.
    ASSERT_NO_THROW(db = &Load(path));
    EXPECT_TRUE(db->GetCacheMap().empty());
}

INSTANTIATE_TEST_SUITE_P(Quick, CPU_ReadonlyRamDb_NONE, testing::Values(0));
