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

    /// Counts record lines in this test's shard, reading the file directly rather than through
    /// the store.
    ///
    /// A store-mediated count cannot see a duplicate append at all: get() collapses several
    /// lines for one key to the last, so the very thing these tests pin -- whether a second line
    /// was written -- is invisible from that side. The version line is line 0 and is excluded.
    size_t countRecordLines(const std::vector<uint8_t>& key,
                            const std::vector<uint8_t>& deviceKey) const
    {
        (void)key;
        (void)deviceKey;
        size_t lines = 0;
        std::error_code ignored;
        for(const auto& entry : std::filesystem::recursive_directory_iterator(_cacheDir, ignored))
        {
            if(!entry.is_regular_file() || entry.path().extension() != ".jsonl")
            {
                continue;
            }
            std::ifstream shard(entry.path());
            std::string line;
            bool first = true;
            while(std::getline(shard, line))
            {
                if(first)
                {
                    // The version stamp, not a record.
                    first = false;
                    continue;
                }
                if(!line.empty())
                {
                    ++lines;
                }
            }
        }
        return lines;
    }
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
    // A record is not permanent: a later sweep that measured something different supersedes it,
    // resolved last-line-wins by get(). Without this, an engine added after the first tune would
    // make the stored record fail the read path's C\S check on every lookup, forever, with
    // re-tuning unable to repair it.
    //
    // Falsifying mutation: restore put()'s early return whenever a record for the key exists.
    FileAutotuneRankingStore store;
    const std::vector<uint8_t> key{7, 7};
    const std::vector<uint8_t> deviceKey{1};

    EXPECT_EQ(store.put(key, deviceKey, {1, 2}, {2, 1}), RankingWriteStatus::WRITTEN);
    EXPECT_EQ(store.put(key, deviceKey, {1, 2, 3}, {3, 1, 2}), RankingWriteStatus::WRITTEN);

    const auto entry = store.get(key, deviceKey);
    ASSERT_TRUE(entry.has_value());
    EXPECT_EQ(entry->sampledEngineIds, (std::vector<int64_t>{1, 2, 3}));
    EXPECT_EQ(entry->order, (std::vector<int64_t>{3, 1, 2}));
}

TEST_F(TestAutotuneRankingStore, PutOfAnIdenticalRankingWritesNothing)
{
    // The racing-writer case the re-read under the lock exists for: two processes that raced the
    // same miss measure the same engines and produce the same order, so the loser has nothing to
    // add. Reported as UNCHANGED rather than WRITTEN, since nothing reached the shard.
    //
    // Falsifying mutation: drop the equality check and always append -- the shard gains a second
    // line and the status becomes WRITTEN.
    FileAutotuneRankingStore store;
    const std::vector<uint8_t> key{8, 8};
    const std::vector<uint8_t> deviceKey{1};

    EXPECT_EQ(store.put(key, deviceKey, {1, 2}, {2, 1}), RankingWriteStatus::WRITTEN);
    EXPECT_EQ(store.put(key, deviceKey, {1, 2}, {2, 1}), RankingWriteStatus::UNCHANGED);

    EXPECT_EQ(countRecordLines(key, deviceKey), 1U);
}

TEST_F(TestAutotuneRankingStore, PutOfANewOrderOverTheSameEngineSetSupersedes)
{
    // The case set-equality on sampledEngineIds would wrongly decline. A re-tune of the same
    // engines can legitimately produce a different ranking -- priming succeeded this time, a
    // plugin shipped a faster kernel, the clock regime changed -- and that is new information.
    //
    // Falsifying mutation: compare only sampledEngineIds instead of the order too.
    FileAutotuneRankingStore store;
    const std::vector<uint8_t> key{9, 9};
    const std::vector<uint8_t> deviceKey{1};

    EXPECT_EQ(store.put(key, deviceKey, {1, 2, 3}, {1, 2, 3}), RankingWriteStatus::WRITTEN);
    EXPECT_EQ(store.put(key, deviceKey, {1, 2, 3}, {3, 2, 1}), RankingWriteStatus::WRITTEN);

    const auto entry = store.get(key, deviceKey);
    ASSERT_TRUE(entry.has_value());
    EXPECT_EQ(entry->order, (std::vector<int64_t>{3, 2, 1}));
}

TEST_F(TestAutotuneRankingStore, PutOfAStrictSubsetIsDeclined)
{
    // A deliberately narrowed sweep (engineIdFilter) measured fewer engines than the stored
    // record covers. Letting it win would replace a usable full-coverage ranking with one that
    // rejects on every later lookup -- permanently de-optimising the machine through an
    // otherwise legitimate API call.
    //
    // Falsifying mutation: remove the isStrictSubset() guard.
    FileAutotuneRankingStore store;
    const std::vector<uint8_t> key{10, 10};
    const std::vector<uint8_t> deviceKey{1};

    EXPECT_EQ(store.put(key, deviceKey, {1, 2, 3}, {1, 2, 3}), RankingWriteStatus::WRITTEN);
    EXPECT_EQ(store.put(key, deviceKey, {1, 2}, {2, 1}), RankingWriteStatus::UNCHANGED);

    const auto entry = store.get(key, deviceKey);
    ASSERT_TRUE(entry.has_value());
    EXPECT_EQ(entry->sampledEngineIds, (std::vector<int64_t>{1, 2, 3}));
    EXPECT_EQ(countRecordLines(key, deviceKey), 1U);
}

TEST_F(TestAutotuneRankingStore, PutComparesAgainstTheLastLineNotTheFirst)
{
    // Once a shard can hold a superseded line, the first match is stale. put() must compare
    // against whatever get() would return -- the last line -- or it adopts a record the reader
    // has already replaced, and a genuine re-measurement is silently dropped.
    //
    // Falsifying mutation: break out of the scan on the first matching key. The third put then
    // compares {1,2} against the stale first line, finds them equal, and reports UNCHANGED.
    FileAutotuneRankingStore store;
    const std::vector<uint8_t> key{11, 11};
    const std::vector<uint8_t> deviceKey{1};

    EXPECT_EQ(store.put(key, deviceKey, {1, 2}, {1, 2}), RankingWriteStatus::WRITTEN);
    EXPECT_EQ(store.put(key, deviceKey, {1, 2}, {2, 1}), RankingWriteStatus::WRITTEN);
    ASSERT_EQ(countRecordLines(key, deviceKey), 2U);

    // Byte-identical to the FIRST line, different from the last: must be written.
    EXPECT_EQ(store.put(key, deviceKey, {1, 2}, {1, 2}), RankingWriteStatus::WRITTEN);

    const auto entry = store.get(key, deviceKey);
    ASSERT_TRUE(entry.has_value());
    EXPECT_EQ(entry->order, (std::vector<int64_t>{1, 2}));
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
