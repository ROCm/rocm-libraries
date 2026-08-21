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

// A minimal parse callback: rejects any line beginning with "BAD", otherwise returns the
// line verbatim. Used to exercise the skip-malformed-line contract without depending on
// any real record format (LineStore itself is format-agnostic).
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

    // The malformed line is skipped -- it is neither returned nor does it stop later
    // lines from being read, and the call as a whole still reports success.
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
    // NOTE: this exercises only the in-process append path, not the lock's actual
    // contention behavior. POSIX fcntl() record locks are scoped per PROCESS, not per
    // thread/file-descriptor-open -- two threads in the same process holding "the lock"
    // never contend on the underlying primitive the way two processes do. What this test
    // does prove: appendLine()/lockLineStore()/unlockLineStore() are safe to call
    // concurrently from multiple threads against one shard with no torn or lost lines.
    // The two-PROCESS case below (TwoProcessesAppendingConcurrentlyProducesNoTornLines) is
    // the one that actually exercises fcntl() contention.
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

#if defined(__linux__)
TEST_F(TestLineStore, TwoProcessesRacingTheSameKeysAppendEachKeyExactlyOnce)
{
    // The only case in this suite that exercises the lock's real job. It is NOT about torn
    // lines: the shard is opened O_APPEND, so a single write() is already atomic and an
    // append-only test passes with the lock removed entirely (verified by mutating
    // acquireLineStoreLock() to a no-op). What the lock actually protects is D3's
    // concurrent-miss race -- the read-then-decide-then-append sequence. Two processes can
    // both read a key as absent and both append it, leaving a duplicate.
    //
    // Both sides are spawned helpers rather than parent-plus-child: a forked parent begins
    // its loop immediately while the child still pays execl() startup, so the two barely
    // overlap and the race is missed. Two helpers started back to back overlap for
    // essentially their whole run.
    {
        auto [shard, openStatus] = openLineStore(_shardPath, "v1");
        ASSERT_EQ(openStatus, LineStoreStatus::OK);
    }

    constexpr int APPENDS_PER_PROCESS = 40;
    constexpr const char* APPENDS_PER_PROCESS_ARG = "40";

    // Both helpers spin until this instant before touching the shard, so neither is
    // still paying execl() startup while the other is already looping. Without the
    // barrier the overlap is partial and a no-op lock escapes detection on some runs.
    const auto startInstant
        = std::chrono::duration_cast<std::chrono::microseconds>(
              (std::chrono::system_clock::now() + std::chrono::milliseconds(300))
                  .time_since_epoch())
              .count();
    const std::string startInstantArg = std::to_string(startInstant);

    // The helper ships beside this binary, so resolve it from this process's own
    // location rather than from a path baked at configure time: CI installs the test
    // binaries and runs them from the install prefix, where the build tree no longer
    // exists and execl() would fail with 127.
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

    // Each key exactly once. A duplicate means both processes observed it as absent
    // inside the same unserialized read-modify-write window.
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
