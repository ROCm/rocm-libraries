// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Standalone helper for TestLineStore's cross-process lock cases. Two modes:
//
// Contention mode (5 positional args):
//   LineStoreLockHelper <shard-path> <version> <line> <repeat-count> <barrier-path>
// Prints "arm\n" and flushes, then spins until <barrier-path> exists before opening the
// shard and, <repeat-count> times, acquiring its lock, appending <line> suffixed with the
// iteration number, and releasing the lock.
//
// The repeat loop is what makes this a real contention test: a single append is one
// write() and lands atomically even with no lock held, but many interleaved
// lock/append/unlock cycles from two processes do not survive a missing lock. A
// single-process, multi-thread version cannot substitute for this: POSIX fcntl() record
// locks are per-process, so only a real second OS process exercises contention. The
// barrier file (created by the parent only after every helper has reported "arm") is what
// makes two helpers spawned close together start racing together, regardless of how long
// either one's process-creation actually took.
//
// Probe mode (3 args: "probe" <shard-path> <version>):
// Prints "arm\n" and flushes, then times a single lockLineStore() call, prints
// "elapsedMs=<N>\n" to stdout, releases the lock, and exits 0. This is the ONLY shape
// that can observe fcntl/LockFileEx blocking semantics at all -- POSIX record locks never
// block a process against itself, so an in-process test cannot substitute for a second
// real OS process here. Used by the parent test to prove a second process blocks for a
// non-trivial duration while the parent holds the shard's exclusive lock, including
// through a hard-linked alias path (the (st_dev, st_ino) registry key) and through a
// same-thread nested readAllLines() call under the held lock (the nesting no-op rule).
// The parent blocks on this "arm" line before starting its own hold-then-release timing,
// so the hold is known to begin only after this process has reached its own timed
// section, not "probably after" a fixed sleep.
//
// Exit codes: 0 on success; 1 for a usage error; 2-4 mirror LineStore's own failure
// modes, letting the parent test distinguish "never got the lock" from "write/read
// itself failed"; 5 if the barrier file never appeared within BARRIER_TIMEOUT (a backstop
// against spinning forever if the parent that would otherwise kill this process is gone).

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <hipdnn_data_sdk/utilities/LineStore.hpp>
#include <optional>
#include <string>
#include <thread>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#endif

using hipdnn_data_sdk::utilities::appendLine;
using hipdnn_data_sdk::utilities::LineStoreStatus;
using hipdnn_data_sdk::utilities::lockLineStore;
using hipdnn_data_sdk::utilities::openLineStore;
using hipdnn_data_sdk::utilities::readAllLines;
using hipdnn_data_sdk::utilities::unlockLineStore;

int main(int argc, char** argv)
{
#if defined(_WIN32)
    // The CRT's default text mode rewrites every "\n" this helper prints to "\r\n" on
    // its way through the pipe to the parent, which reads the raw bytes and sees a
    // trailing '\r' the parent never asked for. Switch stdout to binary mode once,
    // before the first print, so what this helper writes is what the parent reads.
    _setmode(_fileno(stdout), _O_BINARY);
#endif

    // Probe mode: "probe" <shard-path> <version> -- times a single lockLineStore() call.
    if(argc == 4 && std::string(argv[1]) == "probe")
    {
        const std::filesystem::path shardPath = argv[2];
        const std::string version = argv[3];

        // Printed before the timed section starts, so the parent can block until this
        // process has actually reached the point where timing begins, instead of hoping
        // a fixed sleep covered process-creation latency.
        std::printf("arm\n");
        std::fflush(stdout);

        // openLineStore() itself takes the shard's exclusive lock internally (for its
        // version-line check) before this helper ever calls lockLineStore(), so the
        // timer must wrap the WHOLE sequence -- timing only the explicit
        // lockLineStore() call would silently swallow the blocking that happens inside
        // openLineStore() and report a near-zero elapsed time regardless of whether the
        // shard is actually held.
        const auto start = std::chrono::steady_clock::now();
        auto [shard, openStatus] = openLineStore(shardPath, version);
        if(openStatus != LineStoreStatus::OK || !shard)
        {
            std::fprintf(stderr, "openLineStore failed, status=%d\n", static_cast<int>(openStatus));
            return 2;
        }

        const auto lockStatus = lockLineStore(*shard);
        const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                   std::chrono::steady_clock::now() - start)
                                   .count();
        if(lockStatus != LineStoreStatus::OK)
        {
            std::fprintf(stderr, "lockLineStore failed, status=%d\n", static_cast<int>(lockStatus));
            return 3;
        }
        unlockLineStore(*shard);

        std::printf("elapsedMs=%lld\n", static_cast<long long>(elapsedMs));
        std::fflush(stdout);
        return 0;
    }

    if(argc != 6)
    {
        std::fprintf(stderr,
                     "usage: %s <shard-path> <version> <line> <repeat-count> <barrier-path>\n"
                     "       %s probe <shard-path> <version>\n",
                     argv[0],
                     argv[0]);
        return 1;
    }
    const std::filesystem::path shardPath = argv[1];
    const std::string version = argv[2];
    const std::string line = argv[3];
    const int repeatCount = std::atoi(argv[4]);
    const std::filesystem::path barrierPath = argv[5];

    // Printed before the spin-wait below, so the parent knows this process has parsed its
    // arguments and is ready to race, before it creates the barrier file every helper is
    // waiting on.
    std::printf("arm\n");
    std::fflush(stdout);

    // Spin until the parent creates the barrier file so every helper enters the loop
    // together, regardless of how long process creation took for any of them. Bounded:
    // the parent's own ChildProcess RAII kills an orphaned helper on any early test
    // failure, but a bounded self-timeout is a second, independent backstop against a
    // helper spinning forever if the parent is itself killed (e.g. SIGKILL) before its
    // destructors run.
    constexpr auto BARRIER_TIMEOUT = std::chrono::seconds(60);
    const auto barrierDeadline = std::chrono::steady_clock::now() + BARRIER_TIMEOUT;
    while(!std::filesystem::exists(barrierPath))
    {
        if(std::chrono::steady_clock::now() >= barrierDeadline)
        {
            std::fprintf(stderr, "barrier file never appeared: %s\n", barrierPath.string().c_str());
            return 5;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    auto [shard, openStatus] = openLineStore(shardPath, version);
    if(openStatus != LineStoreStatus::OK || !shard)
    {
        std::fprintf(stderr, "openLineStore failed, status=%d\n", static_cast<int>(openStatus));
        return 2;
    }

    // The concurrent-miss race this lock protects against: read the shard under the
    // lock, append only if the key is still absent, release. The append is already
    // torn-line-safe via O_APPEND, what needs the lock is the read-then-decide-then-write
    // sequence, where two processes could both observe "absent" and both append.
    const auto parseLine
        = [](std::string_view raw) -> std::optional<std::string> { return std::string(raw); };

    for(int i = 0; i < repeatCount; ++i)
    {
        const std::string key = line + "-" + std::to_string(i);

        if(lockLineStore(*shard) != LineStoreStatus::OK)
        {
            std::fprintf(stderr, "lockLineStore failed\n");
            return 3;
        }

        const auto [records, readStatus] = readAllLines(*shard, parseLine);
        if(readStatus != LineStoreStatus::OK)
        {
            unlockLineStore(*shard);
            std::fprintf(stderr, "readAllLines failed\n");
            return 4;
        }

        const bool present = std::find(records.begin(), records.end(), key) != records.end();

        // Widen the read-modify-write window: without a pause here, two processes rarely
        // interleave inside it and the test would pass even with a broken lock.
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        LineStoreStatus appendStatus = LineStoreStatus::OK;
        if(!present)
        {
            appendStatus = appendLine(*shard, key);
        }
        unlockLineStore(*shard);

        if(appendStatus != LineStoreStatus::OK)
        {
            std::fprintf(stderr, "appendLine failed, status=%d\n", static_cast<int>(appendStatus));
            return 4;
        }
    }

    return 0;
}
