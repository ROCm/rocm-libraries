// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/LineStore.hpp>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <sys/wait.h>
#include <unistd.h>
#endif

using namespace hipdnn_data_sdk::utilities;

namespace
{

std::filesystem::path makeUniqueShardPath()
{
    static std::atomic<int> s_counter{0};
    const auto unique = std::to_string(::testing::UnitTest::GetInstance()->random_seed()) + "_"
                        + std::to_string(s_counter++);
    return std::filesystem::temp_directory_path() / ("hipdnn_test_linestore_" + unique + ".txt");
}

std::optional<std::string> parseLine(std::string_view line)
{
    if(line.rfind("BAD", 0) == 0)
    {
        return std::nullopt;
    }
    return std::string(line);
}

} // namespace

class TestLineStore : public ::testing::Test
{
protected:
    void SetUp() override
    {
        _shardPath = makeUniqueShardPath();
    }

    void TearDown() override
    {
        std::error_code ignored;
        std::filesystem::remove(_shardPath, ignored);
    }

    std::filesystem::path _shardPath;
};

TEST_F(TestLineStore, AppendThenReadBackRoundTrip)
{
    auto [shard, openStatus] = openLineStore(_shardPath, "v1");
    ASSERT_EQ(openStatus, LineStoreStatus::OK);
    ASSERT_TRUE(shard.has_value());

    ASSERT_EQ(lockLineStore(*shard), LineStoreStatus::OK);
    EXPECT_EQ(appendLine(*shard, "hello"), LineStoreStatus::OK);
    EXPECT_EQ(appendLine(*shard, "world"), LineStoreStatus::OK);
    unlockLineStore(*shard);

    const auto [records, readStatus] = readAllLines(*shard, parseLine);
    EXPECT_EQ(readStatus, LineStoreStatus::OK);
    ASSERT_EQ(records.size(), 2u);
    EXPECT_EQ(records[0], "hello");
    EXPECT_EQ(records[1], "world");
}

TEST_F(TestLineStore, MalformedLineIsSkippedNotFatal)
{
    auto [shard, openStatus] = openLineStore(_shardPath, "v1");
    ASSERT_EQ(openStatus, LineStoreStatus::OK);
    ASSERT_TRUE(shard.has_value());

    ASSERT_EQ(lockLineStore(*shard), LineStoreStatus::OK);
    EXPECT_EQ(appendLine(*shard, "good1"), LineStoreStatus::OK);
    EXPECT_EQ(appendLine(*shard, "BADline"), LineStoreStatus::OK);
    EXPECT_EQ(appendLine(*shard, "good2"), LineStoreStatus::OK);
    unlockLineStore(*shard);

    const auto [records, readStatus] = readAllLines(*shard, parseLine);
    EXPECT_EQ(readStatus, LineStoreStatus::OK);
    ASSERT_EQ(records.size(), 2u);
    EXPECT_EQ(records[0], "good1");
    EXPECT_EQ(records[1], "good2");
}

TEST_F(TestLineStore, VersionMismatchOnReadIsADeclineNotAThrow)
{
    {
        auto [shard, openStatus] = openLineStore(_shardPath, "v1");
        ASSERT_EQ(openStatus, LineStoreStatus::OK);
    }

    std::pair<std::optional<LineStoreShard>, LineStoreStatus> reopened;
    EXPECT_NO_THROW(reopened = openLineStore(_shardPath, "v2-does-not-match"));

    EXPECT_FALSE(reopened.first.has_value());
    EXPECT_EQ(reopened.second, LineStoreStatus::VERSION_MISMATCH);
}

TEST_F(TestLineStore, MultipleThreadsInOneProcessAppendWithoutCorruption)
{
    // Only tests in-process append safety: POSIX fcntl() record locks are per-process, so
    // threads in one process never contend on the lock the way two processes do (see
    // TwoProcessesRacingTheSameKeysAppendEachKeyExactlyOnce below).
    {
        auto [shard, openStatus] = openLineStore(_shardPath, "v1");
        ASSERT_EQ(openStatus, LineStoreStatus::OK);
    }

    constexpr int THREAD_COUNT = 8;
    constexpr int LINES_PER_THREAD = 20;
    std::vector<std::thread> threads;
    threads.reserve(THREAD_COUNT);

    for(int t = 0; t < THREAD_COUNT; ++t)
    {
        threads.emplace_back([this, t]() {
            for(int i = 0; i < LINES_PER_THREAD; ++i)
            {
                auto [shard, openStatus] = openLineStore(_shardPath, "v1");
                ASSERT_EQ(openStatus, LineStoreStatus::OK);
                ASSERT_EQ(lockLineStore(*shard), LineStoreStatus::OK);
                appendLine(*shard, "t" + std::to_string(t) + "-l" + std::to_string(i));
                unlockLineStore(*shard);
            }
        });
    }
    for(auto& thread : threads)
    {
        thread.join();
    }

    auto [shard, openStatus] = openLineStore(_shardPath, "v1");
    ASSERT_EQ(openStatus, LineStoreStatus::OK);
    const auto [records, readStatus] = readAllLines(*shard, parseLine);
    EXPECT_EQ(readStatus, LineStoreStatus::OK);
    EXPECT_EQ(records.size(), static_cast<size_t>(THREAD_COUNT * LINES_PER_THREAD));

    // No line is torn or interleaved: every line matches the "tN-lM" shape in full.
    for(const auto& record : records)
    {
        EXPECT_NE(record.find('t'), std::string::npos);
        EXPECT_NE(record.find("-l"), std::string::npos);
    }
}

/// Uses a directory as the unopenable path: chmod-based permission tricks are unreliable
/// when running as root.
TEST_F(TestLineStore, AnUnopenablePathDeclines)
{
    const auto directoryPath = _shardPath.parent_path() / "not-a-file";
    std::filesystem::create_directories(directoryPath);

    auto [shard, status] = openLineStore(directoryPath, "v1");

    EXPECT_FALSE(shard.has_value());
    EXPECT_EQ(status, LineStoreStatus::OPEN_FAILED);
}

/// Not a double release: the write-back path calls unlockLineStore() on early-return
/// branches where the lock may not be held.
TEST_F(TestLineStore, UnlockingAnUnlockedShardIsANoOp)
{
    auto [shard, status] = openLineStore(_shardPath, "v1");
    ASSERT_EQ(status, LineStoreStatus::OK);

    EXPECT_NO_THROW(unlockLineStore(*shard));
    EXPECT_EQ(lockLineStore(*shard), LineStoreStatus::OK);
    EXPECT_NO_THROW(unlockLineStore(*shard));
}

/// Otherwise the next opener would block forever on a lock whose holder already exited.
TEST_F(TestLineStore, DestroyingALockedShardReleasesTheLock)
{
    {
        auto [shard, status] = openLineStore(_shardPath, "v1");
        ASSERT_EQ(status, LineStoreStatus::OK);
        ASSERT_EQ(lockLineStore(*shard), LineStoreStatus::OK);
        EXPECT_EQ(appendLine(*shard, "written-under-a-lock-never-released"), LineStoreStatus::OK);
    }

    auto [shard, status] = openLineStore(_shardPath, "v1");
    ASSERT_EQ(status, LineStoreStatus::OK);
    EXPECT_EQ(lockLineStore(*shard), LineStoreStatus::OK)
        << "the previous shard's destructor did not release the lock";
    unlockLineStore(*shard);
}

TEST_F(TestLineStore, MoveAssignmentTakesOverTheHandleAndClosesTheOldOne)
{
    const auto otherPath = _shardPath.parent_path() / "move-assignment-target.jsonl";
    std::filesystem::remove(otherPath);

    auto [first, firstStatus] = openLineStore(_shardPath, "v1");
    ASSERT_EQ(firstStatus, LineStoreStatus::OK);
    ASSERT_EQ(appendLine(*first, "first-shard-line"), LineStoreStatus::OK);

    auto [second, secondStatus] = openLineStore(otherPath, "v1");
    ASSERT_EQ(secondStatus, LineStoreStatus::OK);
    ASSERT_EQ(appendLine(*second, "second-shard-line"), LineStoreStatus::OK);

    *first = std::move(*second);

    const auto [records, readStatus] = readAllLines(*first, parseLine);
    ASSERT_EQ(readStatus, LineStoreStatus::OK);
    ASSERT_EQ(records.size(), 1U);
    EXPECT_EQ(records.front(), "second-shard-line");
}

#if defined(__linux__)
TEST_F(TestLineStore, TwoProcessesRacingTheSameKeysAppendEachKeyExactlyOnce)
{
    // Exercises the concurrent-miss race the lock protects (see LineStoreLockHelper.cpp).
    // Uses spawned helpers rather than fork-plus-parent, since a forked parent starts
    // looping before the child finishes paying execl() startup and the two barely overlap.
    {
        auto [shard, openStatus] = openLineStore(_shardPath, "v1");
        ASSERT_EQ(openStatus, LineStoreStatus::OK);
    }

    constexpr int APPENDS_PER_PROCESS = 40;
    constexpr const char* APPENDS_PER_PROCESS_ARG = "40";

    // Both helpers spin until this instant before touching the shard; without a shared
    // start, one helper's execl() lag lets a broken lock go undetected.
    const auto startInstant
        = std::chrono::duration_cast<std::chrono::microseconds>(
              (std::chrono::system_clock::now() + std::chrono::milliseconds(300))
                  .time_since_epoch())
              .count();
    const std::string startInstantArg = std::to_string(startInstant);

    // Resolve the helper beside this binary (see CMakeLists.txt for why only the filename
    // is baked in).
    std::error_code exeError;
    const auto selfPath = std::filesystem::read_symlink("/proc/self/exe", exeError);
    ASSERT_FALSE(exeError) << "could not resolve this test binary's own path";
    const auto helperPath = selfPath.parent_path() / HIPDNN_LINESTORE_LOCK_HELPER_NAME;
    ASSERT_TRUE(std::filesystem::exists(helperPath))
        << "LineStoreLockHelper is not beside the test binary: " << helperPath;
    const std::string helper = helperPath.string();

    std::array<pid_t, 2> pids{};
    for(auto& pid : pids)
    {
        pid = fork();
        ASSERT_NE(pid, -1) << "fork() failed";
        if(pid == 0)
        {
            execl(helper.c_str(),
                  helper.c_str(),
                  _shardPath.c_str(),
                  "v1",
                  "shared-key",
                  APPENDS_PER_PROCESS_ARG,
                  startInstantArg.c_str(),
                  static_cast<char*>(nullptr));
            _exit(127); // execl() only returns on failure.
        }
    }

    for(const auto pid : pids)
    {
        int childStatus = 0;
        ASSERT_EQ(waitpid(pid, &childStatus, 0), pid);
        ASSERT_TRUE(WIFEXITED(childStatus));
        ASSERT_EQ(WEXITSTATUS(childStatus), 0) << "LineStoreLockHelper failed";
    }

    auto [shard, openStatus] = openLineStore(_shardPath, "v1");
    ASSERT_EQ(openStatus, LineStoreStatus::OK);
    const auto [records, readStatus] = readAllLines(*shard, parseLine);
    ASSERT_EQ(readStatus, LineStoreStatus::OK);

    std::map<std::string, int> counts;
    for(const auto& record : records)
    {
        ++counts[record];
    }
    for(int i = 0; i < APPENDS_PER_PROCESS; ++i)
    {
        const std::string key = "shared-key-" + std::to_string(i);
        EXPECT_EQ(counts[key], 1) << "key '" << key << "' appeared " << counts[key]
                                  << " times; the lock failed to serialize read-then-append";
    }
    EXPECT_EQ(records.size(), static_cast<size_t>(APPENDS_PER_PROCESS))
        << "extra lines present: the check-then-append critical section was not atomic";
}
#endif // defined(__linux__)
