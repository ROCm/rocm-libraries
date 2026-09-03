// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/LineStore.hpp>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <cerrno>
#include <csignal>
#include <poll.h>
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

/// Aborts the process with a named diagnosis if the guarded scope does not finish within
/// @p limit. A deadlock on the calling thread cannot be reported by that thread, and
/// ctest's default timeout names only "timeout" -- this names the invariant that broke.
class DeadlineWatchdog
{
public:
    DeadlineWatchdog(std::chrono::milliseconds limit, std::string diagnosis)
        : _diagnosis(std::move(diagnosis))
    {
        _worker = std::thread([this, limit]() {
            std::unique_lock<std::mutex> guard(_mutex);
            if(!_finish.wait_for(guard, limit, [this]() { return _finished; }))
            {
                // Flushed explicitly: abort() below skips every stream destructor.
                std::cerr << "DEADLINE EXCEEDED: " << _diagnosis << '\n' << std::flush;
                std::abort();
            }
        });
    }

    DeadlineWatchdog(const DeadlineWatchdog&) = delete;
    DeadlineWatchdog& operator=(const DeadlineWatchdog&) = delete;
    DeadlineWatchdog(DeadlineWatchdog&&) = delete;
    DeadlineWatchdog& operator=(DeadlineWatchdog&&) = delete;

    ~DeadlineWatchdog()
    {
        {
            const std::lock_guard<std::mutex> guard(_mutex);
            _finished = true;
        }
        _finish.notify_one();
        _worker.join();
    }

private:
    std::string _diagnosis;
    std::mutex _mutex;
    std::condition_variable _finish;
    bool _finished = false;
    std::thread _worker;
};

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

/// A record carrying a newline would be read back as two records, silently splitting one
/// caller's entry into a valid line plus a fragment the parse callback either rejects or,
/// worse, accepts as a different record. appendLine() rejects it at the boundary instead.
///
/// Falsifying mutation: drop the `line.find('\n')` guard in appendLine() -- the append
/// reports OK and the read-back returns three records instead of one.
TEST_F(TestLineStore, ALineCarryingANewlineIsRejected)
{
    auto [shard, openStatus] = openLineStore(_shardPath, "v1");
    ASSERT_EQ(openStatus, LineStoreStatus::OK);
    ASSERT_TRUE(shard.has_value());

    ASSERT_EQ(lockLineStore(*shard), LineStoreStatus::OK);
    EXPECT_EQ(appendLine(*shard, "front\nback"), LineStoreStatus::IO_ERROR);
    EXPECT_EQ(appendLine(*shard, "trailing\n"), LineStoreStatus::IO_ERROR);
    EXPECT_EQ(appendLine(*shard, "clean"), LineStoreStatus::OK);
    unlockLineStore(*shard);

    const auto [records, readStatus] = readAllLines(*shard, parseLine);
    EXPECT_EQ(readStatus, LineStoreStatus::OK);
    ASSERT_EQ(records.size(), 1U)
        << "a rejected line still reached the file and split into extra records";
    EXPECT_EQ(records.front(), "clean");
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

/// A first write interrupted before its newline leaves bytes that hold no complete line.
/// Stamping the version line after that fragment would make the fragment line 0, so every
/// later open reports VERSION_MISMATCH with nothing able to repair it.
///
/// Falsifying mutation: drop the truncateLineStoreToEmpty() call in openLineStore(). The
/// below still reports OK; the second one reports VERSION_MISMATCH.
TEST_F(TestLineStore, ATornFirstWriteIsRepairedNotInheritedForever)
{
    {
        std::ofstream fragment(_shardPath, std::ios::binary);
        ASSERT_TRUE(fragment.is_open());
        // The interrupted first write: a partial version line, no newline.
        fragment << "v";
    }

    {
        auto [shard, openStatus] = openLineStore(_shardPath, "v1");
        ASSERT_EQ(openStatus, LineStoreStatus::OK);
        ASSERT_TRUE(shard.has_value());
        ASSERT_EQ(lockLineStore(*shard), LineStoreStatus::OK);
        EXPECT_EQ(appendLine(*shard, "after-repair"), LineStoreStatus::OK);
        unlockLineStore(*shard);
    }

    auto [shard, openStatus] = openLineStore(_shardPath, "v1");
    EXPECT_EQ(openStatus, LineStoreStatus::OK)
        << "the torn fragment survived as line 0, so this shard is unreadable for good";
    ASSERT_TRUE(shard.has_value());
    const auto [records, readStatus] = readAllLines(*shard, parseLine);
    EXPECT_EQ(readStatus, LineStoreStatus::OK);
    ASSERT_EQ(records.size(), 1U);
    EXPECT_EQ(records.front(), "after-repair");
}

/// Every reader of a shard shares the registry's single descriptor, and with it one file
/// offset. Readers must therefore serialize in-process: interleaved seek-then-read pairs
/// hand each reader a truncated or duplicated view of the file.
///
/// Falsifying mutation: make the shared acquisition in `acquireLineStoreLock()` take
/// `accessMutex` with a shared_mutex's `lock_shared()`. Readers then run concurrently and
/// the counts below come back short. The shard is sized past the 64 KiB read buffer so a
/// read takes several syscalls, which widens the window enough to catch reliably.
TEST_F(TestLineStore, ConcurrentReadersEachSeeTheWholeShard)
{
    constexpr int LINE_COUNT = 8000;
    {
        auto [shard, openStatus] = openLineStore(_shardPath, "v1");
        ASSERT_EQ(openStatus, LineStoreStatus::OK);
        ASSERT_TRUE(shard.has_value());
        ASSERT_EQ(lockLineStore(*shard), LineStoreStatus::OK);
        for(int i = 0; i < LINE_COUNT; ++i)
        {
            ASSERT_EQ(appendLine(*shard, "line-" + std::to_string(i)), LineStoreStatus::OK);
        }
        unlockLineStore(*shard);
    }

    constexpr int READER_COUNT = 8;
    std::vector<size_t> counts(READER_COUNT, 0);
    std::vector<std::thread> readers;
    readers.reserve(READER_COUNT);
    for(int t = 0; t < READER_COUNT; ++t)
    {
        readers.emplace_back([this, t, &counts]() {
            auto [shard, openStatus] = openLineStore(_shardPath, "v1");
            // A fatal assertion only returns from this lambda, so every check here is an
            // EXPECT plus an explicit return -- an ASSERT would fall through to *shard.
            EXPECT_EQ(openStatus, LineStoreStatus::OK);
            if(!shard.has_value())
            {
                ADD_FAILURE() << "reader " << t << " could not open the shard";
                return;
            }
            const auto [records, readStatus] = readAllLines(*shard, parseLine);
            EXPECT_EQ(readStatus, LineStoreStatus::OK);
            counts[static_cast<size_t>(t)] = records.size();
        });
    }
    for(auto& reader : readers)
    {
        reader.join();
    }

    for(int t = 0; t < READER_COUNT; ++t)
    {
        EXPECT_EQ(counts[static_cast<size_t>(t)], static_cast<size_t>(LINE_COUNT))
            << "reader " << t
            << " saw a partial shard: concurrent readers shared one "
               "file offset";
    }
}

/// The registry descriptor's file pointer sits at end-of-file only by accident: any other
/// writer that grows the shard leaves it stale. POSIX pins every write to the end through
/// O_APPEND, but Win32 cannot combine that with the write access the torn-write repair
/// needs, so appendRawLineStoreLine() seeks to FILE_END itself before writing.
///
/// Falsifying mutation: drop the SetFilePointerEx(FILE_END) call in
/// appendRawLineStoreLine(). The append below then lands at the stale offset and
/// overwrites the external line instead of following it. This runs on both platforms and
/// only Win32 can fail it, which is the point -- that branch has no CI to execute it.
TEST_F(TestLineStore, AnAppendAfterAnotherWriterGrewTheFileGoesToTheEnd)
{
    auto [shard, openStatus] = openLineStore(_shardPath, "v1");
    ASSERT_EQ(openStatus, LineStoreStatus::OK);
    ASSERT_TRUE(shard.has_value());

    // Grow the file behind the shard's back while nothing holds the lock, which leaves
    // the registry descriptor's pointer short of the new end.
    {
        std::ofstream external(_shardPath, std::ios::binary | std::ios::app);
        ASSERT_TRUE(external.is_open());
        external << "written-by-another-writer\n";
    }

    ASSERT_EQ(lockLineStore(*shard), LineStoreStatus::OK);
    EXPECT_EQ(appendLine(*shard, "written-through-the-shard"), LineStoreStatus::OK);
    unlockLineStore(*shard);

    const auto [records, readStatus] = readAllLines(*shard, parseLine);
    ASSERT_EQ(readStatus, LineStoreStatus::OK);
    ASSERT_EQ(records.size(), static_cast<size_t>(2))
        << "the shard's append landed at a stale file offset and overwrote the line "
           "another writer had already added";
    EXPECT_EQ(records[0], "written-by-another-writer");
    EXPECT_EQ(records[1], "written-through-the-shard");
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
/// branches where the lock may not be held. A same-thread call alone only proves
/// EXPECT_NO_THROW; the real falsifiable case -- a SECOND `LineStoreShard` object on
/// the same shard whose no-op unlock must not release the FIRST object's still-held
/// lock -- needs a real second process to observe (this process's registry entry
/// state is not otherwise externally visible); see
/// `UnlockingAnUnlockedShardIsANoOpAcrossProcesses` in the Linux cross-process section
/// below.
TEST_F(TestLineStore, UnlockingAnUnlockedShardIsANoOp)
{
    auto [shard, status] = openLineStore(_shardPath, "v1");
    ASSERT_EQ(status, LineStoreStatus::OK);

    EXPECT_NO_THROW(unlockLineStore(*shard));
    EXPECT_EQ(lockLineStore(*shard), LineStoreStatus::OK);
    EXPECT_NO_THROW(unlockLineStore(*shard));
}

/// Otherwise the next opener would block forever on a lock whose holder already
/// exited. This same-process shape only catches a destructor that is a COMPLETE
/// no-op (see the coarser mutation below); it cannot distinguish that from a
/// destructor that resets bookkeeping but leaves the native OS lock held, since this
/// process's own registry entry is never erased and POSIX locks never block a process
/// against itself -- see `DestroyingALockedShardReleasesTheLockAcrossProcesses` in the
/// Linux cross-process section below for that deeper case.
///
/// Falsifying mutation: make `LineStoreShard::releaseIfLocked()` a complete no-op
/// (delete its body). The second `openLineStore()`/`lockLineStore()` pair below would
/// then observe the SAME registry entry still marked exclusively held by this
/// process's own bookkeeping and, depending on platform lock semantics, either fail or
/// silently re-enter; on this codebase's actual (thread, not process) nesting tracker
/// it fails LOCK_FAILED because a different value is never re-derived -- reproduces as
/// a non-OK status here.
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
namespace
{

/// Resolves LineStoreLockHelper beside this test binary (see CMakeLists.txt for why
/// only the filename is baked in as HIPDNN_LINESTORE_LOCK_HELPER_NAME).
std::filesystem::path resolveLockHelperPath()
{
    std::error_code exeError;
    const auto selfPath = std::filesystem::read_symlink("/proc/self/exe", exeError);
    if(exeError)
    {
        return {};
    }
    return selfPath.parent_path() / HIPDNN_LINESTORE_LOCK_HELPER_NAME;
}

/// A LineStoreLockHelper "probe" child in flight: `pid` plus the read end of the pipe
/// its stdout was redirected to.
struct LockProbeHandle
{
    pid_t pid = -1;
    int readFd = -1;
};

/// Spawns `LineStoreLockHelper probe <path> <version>` and returns immediately with a
/// handle to await later. MUST be called before the caller starts any other thread:
/// fork() in a multi-threaded process only carries the calling thread into the child,
/// so a helper thread that is mid-syscall (or holding a lock) at fork time can leave
/// the child process wedged before it ever reaches execl(). Callers hold their lock
/// SYNCHRONOUSLY (a plain sleep on the calling thread, no background thread) between
/// spawn and release, which keeps the whole process single-threaded throughout.
std::optional<LockProbeHandle> spawnLockProbe(const std::filesystem::path& helperPath,
                                              const std::filesystem::path& shardPath,
                                              const std::string& version)
{
    const std::string helper = helperPath.string();
    const std::string shard = shardPath.string();

    std::array<int, 2> pipeFds{-1, -1};
    if(::pipe(pipeFds.data()) != 0)
    {
        return std::nullopt;
    }

    const pid_t pid = fork();
    if(pid == -1)
    {
        ::close(pipeFds[0]);
        ::close(pipeFds[1]);
        return std::nullopt;
    }
    if(pid == 0)
    {
        ::close(pipeFds[0]);
        ::dup2(pipeFds[1], STDOUT_FILENO);
        ::close(pipeFds[1]);
        execl(helper.c_str(),
              helper.c_str(),
              "probe",
              shard.c_str(),
              version.c_str(),
              static_cast<char*>(nullptr));
        _exit(127); // execl() only returns on failure.
    }

    ::close(pipeFds[1]);
    return LockProbeHandle{pid, pipeFds[0]};
}

/// Waits for a spawned probe to finish and returns its reported elapsedMs, or nullopt on
/// any usage/open/lock failure -- including the probe never finishing.
///
/// Every wait here is bounded. A nesting or release regression can leave the probe
/// blocked on the shard forever, and an unbounded read-to-EOF plus waitpid() would turn
/// that into a CTest timeout minutes later, naming the clock instead of the invariant.
/// On the deadline the probe is killed and reaped, and the caller sees a decline.
std::optional<long long> collectLockProbe(const LockProbeHandle& handle)
{
    constexpr auto PROBE_DEADLINE = std::chrono::seconds(30);
    const auto deadline = std::chrono::steady_clock::now() + PROBE_DEADLINE;

    std::string output;
    std::array<char, 256> buffer{};
    bool timedOut = false;
    for(;;)
    {
        pollfd waitFor{handle.readFd, POLLIN, 0};
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now());
        if(remaining.count() <= 0)
        {
            timedOut = true;
            break;
        }
        const int ready = ::poll(&waitFor, 1, static_cast<int>(remaining.count()));
        if(ready < 0 && errno == EINTR)
        {
            continue;
        }
        if(ready <= 0)
        {
            timedOut = ready == 0;
            break;
        }
        const ssize_t count = ::read(handle.readFd, buffer.data(), buffer.size());
        if(count < 0 && errno == EINTR)
        {
            continue;
        }
        if(count <= 0)
        {
            break; // EOF: the probe closed its stdout, so it is exiting.
        }
        output.append(buffer.data(), static_cast<size_t>(count));
    }
    ::close(handle.readFd);

    if(timedOut)
    {
        ::kill(handle.pid, SIGKILL);
    }

    int childStatus = 0;
    ::waitpid(handle.pid, &childStatus, 0);
    if(timedOut || !WIFEXITED(childStatus) || WEXITSTATUS(childStatus) != 0)
    {
        return std::nullopt;
    }

    const auto marker = output.find("elapsedMs=");
    if(marker == std::string::npos)
    {
        return std::nullopt;
    }
    return std::atoll(output.c_str() + marker + std::string("elapsedMs=").size());
}

} // namespace

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

/// The direct oracle for A2's fcntl semantics: the parent holds the exclusive lock,
/// and a real second PROCESS (spawned via the probe helper) must block for a
/// non-trivial duration before it can acquire it. POSIX record locks never block a
/// process against itself, so this is the only shape that can observe fcntl blocking
/// semantics at all -- an in-process, multi-thread version would pass even with the
/// lock entirely removed (see TestWinnerCache.cpp's
/// ConcurrentRecordWinnerOnOneKeyAppendsExactlyOneLine, which guards the in-process
/// mutex half only). The probe is spawned (fork()) BEFORE the hold begins and the hold
/// is a plain synchronous sleep on the test's only thread: forking a multi-threaded
/// process only carries the calling thread into the child, so introducing a holder
/// thread before the fork can leave the child wedged before it ever reaches execl().
///
/// Falsifying mutation: make `unlockLineStore()`'s underlying release a no-op for the
/// exclusive case (e.g. skip `releaseNativeLineStoreLock()` in `releaseLineStoreLock()`
/// when `exclusive` is true), or more simply, delete the `lockLineStore(*shard)` call
/// below so the parent never actually holds anything. Either way the probe process's
/// `elapsedMs` collapses to near 0.
TEST_F(TestLineStore, AProcessHoldingTheExclusiveLockBlocksASecondProcessForAWhile)
{
    const auto helperPath = resolveLockHelperPath();
    ASSERT_TRUE(std::filesystem::exists(helperPath))
        << "LineStoreLockHelper is not beside the test binary: " << helperPath;

    auto [shard, openStatus] = openLineStore(_shardPath, "v1");
    ASSERT_EQ(openStatus, LineStoreStatus::OK);
    ASSERT_EQ(lockLineStore(*shard), LineStoreStatus::OK);

    const auto probe = spawnLockProbe(helperPath, _shardPath, "v1");
    ASSERT_TRUE(probe.has_value()) << "failed to spawn the probe helper";

    constexpr auto HOLD_DURATION = std::chrono::milliseconds(300);
    std::this_thread::sleep_for(HOLD_DURATION);
    unlockLineStore(*shard);

    const auto elapsedMs = collectLockProbe(*probe);
    ASSERT_TRUE(elapsedMs.has_value()) << "probe helper failed to report a timing";
    EXPECT_GE(*elapsedMs, 200)
        << "a second process acquired the exclusive lock almost instantly; the parent's "
           "hold did not block it";
}

/// Covers A2's (st_dev, st_ino) identity keying: two DIFFERENT paths to the SAME inode
/// (a hard link) must share ONE registry entry, so a lock taken through one path is
/// visible to a thread reaching the shard through the OTHER path -- and, since the
/// mutex is per-registry-entry, two threads racing a shared critical section through
/// the two DIFFERENT paths must still serialize exactly as if they shared one path.
/// In-process is the right shape here specifically because identity keying is a
/// registry-lookup concern, not an OS-lock concern: fcntl already resolves to the same
/// inode regardless of which path reached it, so a cross-process probe (as used
/// elsewhere in this file) cannot distinguish correct identity keying from broken
/// identity keying -- both let the OS lock block correctly either way. What differs is
/// whether the two paths share ONE in-process registry entry (one accessMutex, one
/// thread-local depth counter) or two, which only a same-process race can show.
///
/// Falsifying mutation: key the registry on the path string instead of (st_dev,
/// st_ino) -- the plan's explicitly rejected "registry keyed on path" alternative.
/// Two paths to one inode then get two DIFFERENT registry entries (two mutexes, two
/// depth counters), so the two threads below race the shard's read-then-append
/// critical section exactly like the pre-A2 bug, and the shared key's raw line count
/// exceeds 1.
TEST_F(TestLineStore, TwoPathsToOneInodeShareOneLock)
{
    const auto aliasPath = _shardPath.parent_path() / (_shardPath.filename().string() + "-alias");
    std::error_code ignored;
    std::filesystem::remove(aliasPath, ignored);

    {
        auto [shard, openStatus] = openLineStore(_shardPath, "v1");
        ASSERT_EQ(openStatus, LineStoreStatus::OK);
    }
    std::error_code linkError;
    std::filesystem::create_hard_link(_shardPath, aliasPath, linkError);
    ASSERT_FALSE(linkError) << "could not create a hard link for this filesystem";

    constexpr int THREADS = 8;
    constexpr int ITERATIONS_PER_THREAD = 30;
    std::vector<std::thread> threads;
    threads.reserve(THREADS);
    for(int t = 0; t < THREADS; ++t)
    {
        // Half the threads reach the shard via _shardPath, half via the hard-linked
        // alias -- same inode, two names.
        const auto& path = (t % 2 == 0) ? _shardPath : aliasPath;
        threads.emplace_back([path]() {
            auto [shard, openStatus] = openLineStore(path, "v1");
            ASSERT_EQ(openStatus, LineStoreStatus::OK);
            for(int i = 0; i < ITERATIONS_PER_THREAD; ++i)
            {
                ASSERT_EQ(lockLineStore(*shard), LineStoreStatus::OK);
                const auto [records, readStatus] = readAllLines(*shard, parseLine);
                ASSERT_EQ(readStatus, LineStoreStatus::OK);
                const bool present
                    = std::find(records.begin(), records.end(), "shared") != records.end();
                // Widen the race window: without this, two registry entries would
                // rarely interleave inside it and the test would pass even when broken.
                std::this_thread::sleep_for(std::chrono::microseconds(200));
                if(!present)
                {
                    appendLine(*shard, "shared");
                }
                unlockLineStore(*shard);
            }
        });
    }
    for(auto& thread : threads)
    {
        thread.join();
    }
    std::filesystem::remove(aliasPath, ignored);

    auto [shard, openStatus] = openLineStore(_shardPath, "v1");
    ASSERT_EQ(openStatus, LineStoreStatus::OK);
    const auto [records, readStatus] = readAllLines(*shard, parseLine);
    ASSERT_EQ(readStatus, LineStoreStatus::OK);

    int sharedCount = 0;
    for(const auto& record : records)
    {
        if(record == "shared")
        {
            ++sharedCount;
        }
    }
    EXPECT_EQ(sharedCount, 1)
        << "the shard was reached through two different paths to the SAME inode; if "
           "they do not share one registry entry, the two paths' threads race the "
           "check-then-append critical section independently -- got "
        << sharedCount << " lines instead of 1";
}

/// Covers A2's nesting no-op: `readAllLines()` under an already-held exclusive lock, on
/// the SAME thread, must not touch the underlying OS lock at all -- neither convert it
/// (POSIX) nor release it early (Win32's same-handle-overlap hazard). Proven the only
/// way that is externally observable: a second PROCESS still cannot acquire the shard
/// while the nested read is in flight and the exclusive hold continues afterward.
///
/// Falsifying mutation: remove the thread-local depth tracking in
/// `acquireLineStoreLock()`/`releaseLineStoreLock()` so a nested shared acquisition
/// takes and releases the native lock for real (reverting to the rejected revision-1
/// design, `Results/pr11101-plan-review`'s LockFixAudit item 1). On POSIX this converts
/// the exclusive lock to shared and the nested release drops it outright -- the probe
/// process's `elapsedMs` collapses to near 0 despite the parent still holding the outer
/// lock. The same mutation also self-deadlocks the calling thread on `accessMutex`, which
/// is not recursive; the DeadlineWatchdog below turns that hang into a named failure, so
/// either shape of the defect reports instead of wedging the suite until ctest times out.
TEST_F(TestLineStore, NestedReadAllLinesUnderAnExclusiveLockDoesNotReleaseIt)
{
    const auto helperPath = resolveLockHelperPath();
    ASSERT_TRUE(std::filesystem::exists(helperPath))
        << "LineStoreLockHelper is not beside the test binary: " << helperPath;

    auto [shard, openStatus] = openLineStore(_shardPath, "v1");
    ASSERT_EQ(openStatus, LineStoreStatus::OK);
    ASSERT_EQ(lockLineStore(*shard), LineStoreStatus::OK);

    // Same-thread nested read while the exclusive lock is held. Guarded, because the
    // mutation that breaks it deadlocks this very thread: an unguarded call would hang
    // the suite rather than name the defect.
    {
        const DeadlineWatchdog watchdog(std::chrono::seconds(30),
                                        "the nested readAllLines() never returned -- the "
                                        "per-thread nesting no-op was removed and the "
                                        "non-recursive accessMutex self-deadlocked");
        const auto [records, readStatus] = readAllLines(*shard, parseLine);
        EXPECT_EQ(readStatus, LineStoreStatus::OK) << "the nested read itself must succeed";
    }

    const auto probe = spawnLockProbe(helperPath, _shardPath, "v1");
    ASSERT_TRUE(probe.has_value()) << "failed to spawn the probe helper";

    constexpr auto HOLD_DURATION = std::chrono::milliseconds(300);
    std::this_thread::sleep_for(HOLD_DURATION);
    unlockLineStore(*shard);

    const auto elapsedMs = collectLockProbe(*probe);
    ASSERT_TRUE(elapsedMs.has_value()) << "probe helper failed to report a timing";
    EXPECT_GE(*elapsedMs, 200)
        << "a second process acquired the exclusive lock almost instantly; the nested "
           "readAllLines() call released or downgraded it";
}

/// The deeper cross-process case `DestroyingALockedShardReleasesTheLock` above cannot
/// reach: this process's registry entry is never erased and reuses the SAME
/// descriptor for process lifetime, so a same-process re-open after the shard's
/// destructor runs re-acquires the identical (process, inode) fcntl lock trivially --
/// POSIX record locks never block a process against itself. Only a real second
/// process, spawned fresh, can tell "the destructor genuinely released the OS lock"
/// apart from "the destructor reset only its own bookkeeping and left the native fcntl
/// lock held."
///
/// Falsifying mutation: in `releaseLineStoreLock()`, clear this thread's depth entry
/// and release `accessMutex` but skip the `releaseNativeLineStoreLock()` call --
/// bookkeeping looks released; the native lock is not. This process's own re-open (the
/// test above) passes regardless; a fresh second process's `elapsedMs` here does not.
TEST_F(TestLineStore, DestroyingALockedShardReleasesTheLockAcrossProcesses)
{
    const auto helperPath = resolveLockHelperPath();
    ASSERT_TRUE(std::filesystem::exists(helperPath))
        << "LineStoreLockHelper is not beside the test binary: " << helperPath;

    {
        auto [shard, status] = openLineStore(_shardPath, "v1");
        ASSERT_EQ(status, LineStoreStatus::OK);
        ASSERT_EQ(lockLineStore(*shard), LineStoreStatus::OK);
        EXPECT_EQ(appendLine(*shard, "written-under-a-lock-never-released"), LineStoreStatus::OK);
        // shard destructs here, at the closing brace, WITHOUT an explicit
        // unlockLineStore() call -- this is the exact case under test.
    }

    const auto probe = spawnLockProbe(helperPath, _shardPath, "v1");
    ASSERT_TRUE(probe.has_value()) << "failed to spawn the probe helper";
    const auto elapsedMs = collectLockProbe(*probe);

    ASSERT_TRUE(elapsedMs.has_value()) << "probe helper failed to report a timing";
    EXPECT_LT(*elapsedMs, 100)
        << "a fresh second process could not acquire the lock promptly after the "
           "previous shard object was destroyed; its destructor did not release the "
           "OS-level lock";
}

/// Falsifies `UnlockingAnUnlockedShardIsANoOp`'s same-process claim with the only
/// shape that can: a fresh second PROCESS, spawned via the probe helper, must still
/// block on the shard while THIS process holds it -- even after another
/// `LineStoreShard` object on the SAME shard (which never itself locked) calls
/// `unlockLineStore()`. Both objects reference one process-wide registry entry, so a
/// missing no-op guard would let the second object's call release the first's hold.
///
/// Falsifying mutation: delete the
/// `if(!detail::LineStoreAccess::locked(shard)) return;` guard at the top of
/// `unlockLineStore()`. `otherShard`'s call then drops `shard`'s exclusive hold for
/// real, and the probe process's `elapsedMs` collapses to near 0 despite `shard`
/// never having been explicitly unlocked.
TEST_F(TestLineStore, UnlockingAnUnlockedShardIsANoOpAcrossProcesses)
{
    const auto helperPath = resolveLockHelperPath();
    ASSERT_TRUE(std::filesystem::exists(helperPath))
        << "LineStoreLockHelper is not beside the test binary: " << helperPath;

    auto [shard, status] = openLineStore(_shardPath, "v1");
    ASSERT_EQ(status, LineStoreStatus::OK);
    auto [otherShard, otherStatus] = openLineStore(_shardPath, "v1");
    ASSERT_EQ(otherStatus, LineStoreStatus::OK);

    ASSERT_EQ(lockLineStore(*shard), LineStoreStatus::OK);
    // otherShard never locked; this must be a true no-op, not a release of shard's hold.
    unlockLineStore(*otherShard);

    const auto probe = spawnLockProbe(helperPath, _shardPath, "v1");
    ASSERT_TRUE(probe.has_value()) << "failed to spawn the probe helper";

    constexpr auto HOLD_DURATION = std::chrono::milliseconds(300);
    std::this_thread::sleep_for(HOLD_DURATION);
    unlockLineStore(*shard);

    const auto elapsedMs = collectLockProbe(*probe);
    ASSERT_TRUE(elapsedMs.has_value()) << "probe helper failed to report a timing";
    EXPECT_GE(*elapsedMs, 200)
        << "a second process acquired the exclusive lock almost instantly; "
           "otherShard's no-op unlock released shard's still-held exclusive lock";
}
#endif // defined(__linux__)
