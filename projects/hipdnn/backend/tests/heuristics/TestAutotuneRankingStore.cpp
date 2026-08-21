// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file TestAutotuneRankingStore.cpp
 * @brief Contract tests for the exact-match ranking store seam (`IAutotuneRankingStore`
 *        and its file-backed implementation, `FileAutotuneRankingStore`).
 */

#include "heuristics/config/AutotuneRankingStore.hpp"

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/LineStore.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sys/stat.h>
#endif

#if defined(HIPDNN_AUTOTUNE_CROSS_PROCESS_HELPER_PATH) && !defined(_WIN32)
#include <sys/wait.h>
#endif

using namespace hipdnn_backend::heuristics::config;
using hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter;

namespace
{

std::filesystem::path makeUniqueCacheDir()
{
    static std::atomic<int> s_counter{0};
    const auto unique = std::to_string(::testing::UnitTest::GetInstance()->random_seed()) + "_"
                        + std::to_string(s_counter++);
    return std::filesystem::temp_directory_path() / ("hipdnn_test_rankingstore_" + unique);
}

} // namespace

/// Each test gets its own `HIPDNN_CACHE_DIR`: the store has no per-test isolation of
/// its own, since real cross-process persistence is the entire point of it.
class TestAutotuneRankingStore : public ::testing::Test
{
protected:
    void SetUp() override
    {
        _cacheDir = makeUniqueCacheDir();
        _cacheDirEnv = std::make_unique<ScopedEnvironmentVariableSetter>("HIPDNN_CACHE_DIR",
                                                                         _cacheDir.string());
    }

    void TearDown() override
    {
        _cacheDirEnv.reset();
        std::error_code ignored;
        std::filesystem::remove_all(_cacheDir, ignored);
    }

    std::filesystem::path _cacheDir;
    std::unique_ptr<ScopedEnvironmentVariableSetter> _cacheDirEnv;
};

TEST_F(TestAutotuneRankingStore, PutThenGetReturnsTheSameEntry)
{
    FileAutotuneRankingStore store;
    const std::vector<uint8_t> key{1, 2, 3};
    const std::vector<uint8_t> deviceKey{9, 9};
    const std::vector<int64_t> sampledEngineIds{10, 20, 30};
    const std::vector<int64_t> order{30, 10, 20};

    store.put(key, deviceKey, sampledEngineIds, order);

    RankingLookupStatus status = RankingLookupStatus::UNAVAILABLE;
    const auto entry = store.get(key, deviceKey, &status);

    ASSERT_TRUE(entry.has_value());
    EXPECT_EQ(status, RankingLookupStatus::HIT);
    EXPECT_EQ(entry->sampledEngineIds, sampledEngineIds);
    EXPECT_EQ(entry->order, order);
}

TEST_F(TestAutotuneRankingStore, GetOnAbsentKeyReturnsNulloptWithMissStatus)
{
    const FileAutotuneRankingStore store;
    const std::vector<uint8_t> key{1, 2, 3};
    const std::vector<uint8_t> deviceKey{9, 9};

    RankingLookupStatus status = RankingLookupStatus::UNAVAILABLE;
    EXPECT_FALSE(store.get(key, deviceKey, &status).has_value());
    EXPECT_EQ(status, RankingLookupStatus::MISS);
}

TEST_F(TestAutotuneRankingStore, DifferentDeviceKeySameGraphKeyDoesNotCollide)
{
    FileAutotuneRankingStore store;
    const std::vector<uint8_t> key{1, 2, 3};
    const std::vector<uint8_t> deviceKeyA{0xAA};
    const std::vector<uint8_t> deviceKeyB{0xBB};
    const std::vector<int64_t> sampledEngineIds{10, 20};
    const std::vector<int64_t> order{10, 20};

    store.put(key, deviceKeyA, sampledEngineIds, order);

    EXPECT_TRUE(store.get(key, deviceKeyA).has_value());
    EXPECT_FALSE(store.get(key, deviceKeyB).has_value());
}

TEST_F(TestAutotuneRankingStore, PutTwiceForSameKeyLastWriteWins)
{
    // put() declines a second write once any record for the key exists; the first
    // write is what a reader observes.
    FileAutotuneRankingStore store;
    const std::vector<uint8_t> key{7, 7};
    const std::vector<uint8_t> deviceKey{1};

    store.put(key, deviceKey, {1, 2}, {2, 1});
    store.put(key, deviceKey, {1, 2, 3}, {3, 1, 2});

    const auto entry = store.get(key, deviceKey);
    ASSERT_TRUE(entry.has_value());
    EXPECT_EQ(entry->sampledEngineIds, (std::vector<int64_t>{1, 2}));
    EXPECT_EQ(entry->order, (std::vector<int64_t>{2, 1}));
}

TEST_F(TestAutotuneRankingStore, VersionMismatchReportsUnavailableNotMiss)
{
    // A shard whose version line does not match the store's compiled-in version must
    // report UNAVAILABLE, distinguishable from an ordinary miss.
    FileAutotuneRankingStore store;
    const std::vector<uint8_t> key{4, 4, 4};
    const std::vector<uint8_t> deviceKey{};

    store.put(key, deviceKey, {1}, {1});

    bool found = false;
    std::filesystem::path shardPath;
    for(const auto& entry :
        std::filesystem::recursive_directory_iterator(_cacheDir / "autotune-rankings"))
    {
        if(entry.is_regular_file() && entry.path().extension() == ".jsonl")
        {
            shardPath = entry.path();
            found = true;
            break;
        }
    }
    ASSERT_TRUE(found) << "expected put() to have created a shard file";

    {
        std::ofstream rewritten(shardPath, std::ios::trunc);
        rewritten << "not-a-real-version\n";
    }

    RankingLookupStatus status = RankingLookupStatus::MISS;
    const auto entry = store.get(key, deviceKey, &status);

    EXPECT_FALSE(entry.has_value());
    EXPECT_EQ(status, RankingLookupStatus::UNAVAILABLE);
}

TEST_F(TestAutotuneRankingStore, MalformedLineIsSkippedNotFatal)
{
    // A malformed line must not prevent other records in the same or a different shard
    // from being read.
    FileAutotuneRankingStore store;
    const std::vector<uint8_t> keyA{1};
    const std::vector<uint8_t> keyB{2};
    const std::vector<uint8_t> deviceKey{};

    store.put(keyA, deviceKey, {1, 2}, {2, 1});

    std::filesystem::path shardPath;
    for(const auto& entry :
        std::filesystem::recursive_directory_iterator(_cacheDir / "autotune-rankings"))
    {
        if(entry.is_regular_file() && entry.path().extension() == ".jsonl")
        {
            shardPath = entry.path();
            break;
        }
    }
    ASSERT_FALSE(shardPath.empty());

    {
        std::ofstream appended(shardPath, std::ios::app);
        appended << "{ this is not valid json\n";
    }

    const auto entryA = store.get(keyA, deviceKey);
    ASSERT_TRUE(entryA.has_value());
    EXPECT_EQ(entryA->sampledEngineIds, (std::vector<int64_t>{1, 2}));

    store.put(keyB, deviceKey, {5}, {5});
    const auto entryB = store.get(keyB, deviceKey);
    ASSERT_TRUE(entryB.has_value());
    EXPECT_EQ(entryB->sampledEngineIds, (std::vector<int64_t>{5}));
}

#if defined(__linux__)
TEST_F(TestAutotuneRankingStore, UnwritableCacheRootDeclinesReadAndWriteWithoutThrowing)
{
    // cacheRoot() fails soft under a read-only parent; put()/get() must decline cleanly.
    _cacheDirEnv.reset();
    std::error_code ignored;
    std::filesystem::remove_all(_cacheDir, ignored);
    std::filesystem::create_directories(_cacheDir);
    ::chmod(_cacheDir.c_str(), 0500); // read + execute only, no write

    const auto target = _cacheDir / "subcache";
    _cacheDirEnv
        = std::make_unique<ScopedEnvironmentVariableSetter>("HIPDNN_CACHE_DIR", target.string());

    FileAutotuneRankingStore store;
    const std::vector<uint8_t> key{9};
    const std::vector<uint8_t> deviceKey{};

    EXPECT_NO_THROW(store.put(key, deviceKey, {1}, {1}));

    RankingLookupStatus status = RankingLookupStatus::HIT;
    std::optional<CachedEntry> entry;
    EXPECT_NO_THROW(entry = store.get(key, deviceKey, &status));
    EXPECT_FALSE(entry.has_value());
    EXPECT_EQ(status, RankingLookupStatus::UNAVAILABLE);

    ::chmod(_cacheDir.c_str(), 0700);
}
#endif // defined(__linux__)

// --- Cross-process oracle -------------------------------------------------------------
//
// Persistence across process boundaries is unobservable within a single test process,
// so these drive a real second OS process via AutotuneCrossProcessHelper (path baked in
// as HIPDNN_AUTOTUNE_CROSS_PROCESS_HELPER_PATH). They also prove that two structurally
// identical graphs with differently-numbered tensors derive the same cache key.
#if defined(HIPDNN_AUTOTUNE_CROSS_PROCESS_HELPER_PATH) && !defined(_WIN32)
namespace
{
/// Runs the helper and returns {exit code, stdout}.
std::pair<int, std::string>
    runHelper(const std::string& mode, int64_t uid, int64_t dim, const std::string& engineIdsCsv)
{
    const std::string command = std::string(HIPDNN_AUTOTUNE_CROSS_PROCESS_HELPER_PATH) + " " + mode
                                + " " + std::to_string(uid) + " " + std::to_string(dim) + " '"
                                + engineIdsCsv + "'";

    std::string output;
    FILE* pipe = ::popen(command.c_str(), "r");
    if(pipe == nullptr)
    {
        return {-1, output};
    }
    std::array<char, 256> buffer{};
    while(std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr)
    {
        output += buffer.data();
    }
    const int status = ::pclose(pipe);
    return {WIFEXITED(status) ? WEXITSTATUS(status) : -1, output};
}
} // namespace

TEST_F(TestAutotuneRankingStore, RankingWrittenByOneProcessIsReadBackByAnother)
{
    const auto written = runHelper("write", 7, 64, "101,202,303");
    ASSERT_EQ(written.first, 0) << "helper failed to write the ranking";

    const auto read = runHelper("read", 7, 64, "");
    ASSERT_EQ(read.first, 0) << "helper failed to read the ranking back (3 = miss)";
    EXPECT_EQ(read.second, "101,202,303");
}

TEST_F(TestAutotuneRankingStore, RenumberedGraphHitsAcrossProcesses)
{
    // Same graph content, different tensor uid: only the uid's ordinal folds into the
    // key, so a renumbered graph must still hit.
    const auto written = runHelper("write", 11, 128, "404,505");
    ASSERT_EQ(written.first, 0) << "helper failed to write the ranking";

    const auto read = runHelper("read", 99, 128, "");
    ASSERT_EQ(read.first, 0) << "renumbered graph missed the cache across processes";
    EXPECT_EQ(read.second, "404,505");
}

TEST_F(TestAutotuneRankingStore, DifferentGraphMissesAcrossProcesses)
{
    // The negative half: a graph differing in a folded field (dims) must not hit.
    const auto written = runHelper("write", 7, 64, "606");
    ASSERT_EQ(written.first, 0) << "helper failed to write the ranking";

    const auto read = runHelper("read", 7, 4096, "");
    EXPECT_EQ(read.first, 3) << "a structurally different graph must miss, not hit";
}
#endif // HIPDNN_AUTOTUNE_CROSS_PROCESS_HELPER_PATH && !_WIN32
