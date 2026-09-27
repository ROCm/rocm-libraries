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
#include <unistd.h> // geteuid: the read-only-parent case is a no-op as root
#endif

#if defined(HIPDNN_AUTOTUNE_CROSS_PROCESS_HELPER_NAME) && !defined(_WIN32)
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

/// A cache root that cannot be created must make put()/get() decline cleanly rather than
/// throw. Two independent ways to make it uncreatable, because they fail differently:
///
///  - a path whose parent is a regular file: mkdir returns ENOTDIR, which is a type
///    error and is enforced against every uid including root. This is the case CI
///    actually exercises, since its containers run as root (`--user 0:0`).
///  - a read-only parent directory: mkdir returns EACCES, which root bypasses entirely.
///    Skipped under root rather than silently asserting nothing -- a run as root would
///    otherwise see the write succeed and the lookup hit, which is exactly how this test
///    failed in CI while passing on every developer machine.
TEST_F(TestAutotuneRankingStore, CacheRootUnderARegularFileDeclinesReadAndWriteWithoutThrowing)
{
    _cacheDirEnv.reset();
    std::error_code ignored;
    std::filesystem::remove_all(_cacheDir, ignored);
    std::filesystem::create_directories(_cacheDir);

    const auto occupied = _cacheDir / "not_a_directory";
    {
        std::ofstream(occupied) << "occupied";
    }

    // Parent is a regular file, so this can never be created -- as root or otherwise.
    const auto target = occupied / "subcache";
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
}

#if defined(__linux__)
TEST_F(TestAutotuneRankingStore, UnwritableCacheRootDeclinesReadAndWriteWithoutThrowing)
{
    if(::geteuid() == 0)
    {
        GTEST_SKIP() << "runs as root, which ignores the directory permissions this case "
                        "depends on; CacheRootUnderARegularFileDeclines... covers the same "
                        "contract in a way root cannot bypass";
    }

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

// --- A lookup must not create state --------------------------------------------------
//
// The read path runs for every graph on the default heuristic policy list, over an
// unbounded key space: one shard per distinct (graph, device) pair. Were a miss to
// materialize the shard it failed to find, a process finalizing many distinct graphs
// would leave an empty file for each, plus an open descriptor -- the registry never
// closes an entry, since POSIX locks are per (process, inode). Creating state belongs to
// the write path.

TEST_F(TestAutotuneRankingStore, MissDoesNotCreateAShard)
{
    const FileAutotuneRankingStore store;
    const std::vector<uint8_t> key{4, 4, 4};
    const std::vector<uint8_t> deviceKey{1};

    RankingLookupStatus status = RankingLookupStatus::HIT;
    const auto entry = store.get(key, deviceKey, &status);

    EXPECT_FALSE(entry.has_value());
    EXPECT_EQ(status, RankingLookupStatus::MISS)
        << "an absent shard is an ordinary miss, not an unavailable cache";

    size_t created = 0;
    std::error_code ignored;
    for(const auto& e : std::filesystem::recursive_directory_iterator(_cacheDir, ignored))
    {
        if(e.is_regular_file())
        {
            ++created;
        }
    }
    EXPECT_EQ(created, 0u) << "a pure lookup created " << created << " file(s) under " << _cacheDir;
}

// Files scale with distinct keys looked up, so a read-only workload must create none
// however many keys it touches.
TEST_F(TestAutotuneRankingStore, ManyDistinctMissesCreateNothing)
{
    const FileAutotuneRankingStore store;
    constexpr int DISTINCT_KEYS = 64;

    for(int i = 0; i < DISTINCT_KEYS; ++i)
    {
        const std::vector<uint8_t> key{static_cast<uint8_t>(i), 0xAB};
        RankingLookupStatus status = RankingLookupStatus::HIT;
        EXPECT_FALSE(store.get(key, {}, &status).has_value());
        EXPECT_EQ(status, RankingLookupStatus::MISS);
    }

    size_t created = 0;
    std::error_code ignored;
    for(const auto& e : std::filesystem::recursive_directory_iterator(_cacheDir, ignored))
    {
        if(e.is_regular_file())
        {
            ++created;
        }
    }
    EXPECT_EQ(created, 0u) << DISTINCT_KEYS << " distinct misses created " << created << " file(s)";
}

#if defined(__linux__)
// States the resource bound directly, via /proc/self/fd. The file-count assertions above
// would still pass if the store opened a descriptor per key and leaked it without
// creating anything -- the half that reaches RLIMIT_NOFILE and then fails unrelated
// dlopen/hipModuleLoad calls.
TEST_F(TestAutotuneRankingStore, DistinctMissesDoNotAccumulateDescriptors)
{
    const auto openDescriptors = [] {
        size_t count = 0;
        std::error_code ignored;
        for(const auto& e : std::filesystem::directory_iterator("/proc/self/fd", ignored))
        {
            (void)e;
            ++count;
        }
        return count;
    };

    const FileAutotuneRankingStore store;
    const size_t before = openDescriptors();

    for(int i = 0; i < 64; ++i)
    {
        const std::vector<uint8_t> key{static_cast<uint8_t>(i), 0xCD};
        RankingLookupStatus status = RankingLookupStatus::HIT;
        EXPECT_FALSE(store.get(key, {}, &status).has_value());
    }

    EXPECT_EQ(openDescriptors(), before)
        << "64 distinct-key misses changed this process's open descriptor count";
}
#endif // defined(__linux__)

// Not creating on read must not stop a written record from being found: a get() that
// always missed would satisfy the assertions above.
TEST_F(TestAutotuneRankingStore, WriteThenReadStillHitsAfterNonCreatingRead)
{
    FileAutotuneRankingStore store;
    const std::vector<uint8_t> key{5, 5};
    const std::vector<uint8_t> deviceKey{2};

    RankingLookupStatus status = RankingLookupStatus::HIT;
    EXPECT_FALSE(store.get(key, deviceKey, &status).has_value());
    EXPECT_EQ(status, RankingLookupStatus::MISS);

    ASSERT_EQ(store.put(key, deviceKey, {1, 2, 3}, {3, 1, 2}), RankingWriteStatus::WRITTEN);

    const auto entry = store.get(key, deviceKey, &status);
    ASSERT_TRUE(entry.has_value());
    EXPECT_EQ(status, RankingLookupStatus::HIT);
    EXPECT_EQ(entry->order, (std::vector<int64_t>{3, 1, 2}));
    EXPECT_EQ(entry->sampledEngineIds, (std::vector<int64_t>{1, 2, 3}));
}

// --- Cross-process oracle -------------------------------------------------------------
//
// Persistence across process boundaries is unobservable within a single test process,
// so these drive a real second OS process via the cross-process helper. They also prove
// that two structurally identical graphs with differently-numbered tensors derive the
// same cache key.
#if defined(HIPDNN_AUTOTUNE_CROSS_PROCESS_HELPER_NAME) && !defined(_WIN32)
namespace
{
/// Resolves the helper beside this test binary (see CMakeLists.txt for why only the
/// filename is baked in as HIPDNN_AUTOTUNE_CROSS_PROCESS_HELPER_NAME: an absolute build
/// tree path does not survive CI installing the tests and running them from the prefix).
std::filesystem::path resolveHelperPath()
{
    std::error_code exeError;
    const auto selfPath = std::filesystem::read_symlink("/proc/self/exe", exeError);
    if(exeError)
    {
        return {};
    }
    return selfPath.parent_path() / HIPDNN_AUTOTUNE_CROSS_PROCESS_HELPER_NAME;
}

/// Runs the helper and returns {exit code, stdout}.
std::pair<int, std::string>
    runHelper(const std::string& mode, int64_t uid, int64_t dim, const std::string& engineIdsCsv)
{
    const auto helperPath = resolveHelperPath();
    if(helperPath.empty() || !std::filesystem::exists(helperPath))
    {
        // Distinguishable from a helper that ran and failed: -2 means never launched.
        return {-2, {}};
    }

    const std::string command = helperPath.string() + " " + mode + " " + std::to_string(uid) + " "
                                + std::to_string(dim) + " '" + engineIdsCsv + "'";

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
#endif // HIPDNN_AUTOTUNE_CROSS_PROCESS_HELPER_NAME && !_WIN32
