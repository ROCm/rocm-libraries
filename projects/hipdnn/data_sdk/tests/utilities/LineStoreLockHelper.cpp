// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Standalone helper for TestLineStore's cross-process lock cases. Two modes:
//
// Contention mode (5 positional args):
//   LineStoreLockHelper <shard-path> <version> <line> <repeat-count> <start-epoch-us>
// Opens/creates the shard, then <repeat-count> times acquires its lock, appends <line>
// suffixed with the iteration number, and releases the lock.
//
// The repeat loop is what makes this a real contention test: a single append is one
// write() and lands atomically even with no lock held, but many interleaved
// lock/append/unlock cycles from two processes do not survive a missing lock. A
// single-process, multi-thread version cannot substitute for this: POSIX fcntl() record
// locks are per-process, so only a real second OS process exercises contention.
//
// Probe mode (3 args: "probe" <shard-path> <version>):
// Opens the shard, times a single lockLineStore() call, prints
// "elapsedMs=<N>\n" to stdout, releases the lock, and exits 0. This is the ONLY shape
// that can observe fcntl blocking semantics at all -- POSIX record locks never block a
// process against itself, so an in-process test cannot substitute for a second real OS
// process here. Used by the parent test to prove a second process blocks for a
// non-trivial duration while the parent holds the shard's exclusive lock, including
// through a hard-linked alias path (the (st_dev, st_ino) registry key) and through a
// same-thread nested readAllLines() call under the held lock (the nesting no-op rule).
//
// Exit codes: 0 on success; 1 for a usage error; 2-4 mirror LineStore's own failure
// modes, letting the parent test distinguish "never got the lock" from "write/read
// itself failed."

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <hipdnn_data_sdk/utilities/LineStore.hpp>
#include <optional>
#include <string>
#include <thread>

using hipdnn_data_sdk::utilities::appendLine;
using hipdnn_data_sdk::utilities::LineStoreStatus;
using hipdnn_data_sdk::utilities::lockLineStore;
using hipdnn_data_sdk::utilities::openLineStore;
using hipdnn_data_sdk::utilities::readAllLines;
using hipdnn_data_sdk::utilities::unlockLineStore;

int main(int argc, char** argv)
{
    // Probe mode: "probe" <shard-path> <version> -- times a single lockLineStore() call.
    if(argc == 4 && std::string(argv[1]) == "probe")
    {
        const std::filesystem::path shardPath = argv[2];
        const std::string version = argv[3];

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
        return 0;
    }

    if(argc != 6)
    {
        std::fprintf(stderr,
                     "usage: %s <shard-path> <version> <line> <repeat-count> <start-epoch-us>\n"
                     "       %s probe <shard-path> <version>\n",
                     argv[0],
                     argv[0]);
        return 1;
    }
    const std::filesystem::path shardPath = argv[1];
    const std::string version = argv[2];
    const std::string line = argv[3];
    const int repeatCount = std::atoi(argv[4]);
    const long long startEpochUs = std::atoll(argv[5]);

    // Spin until the shared start instant so both helpers enter the loop together.
    while(std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count()
          < startEpochUs)
    {
        std::this_thread::yield();
    }

    auto [shard, openStatus] = openLineStore(shardPath, version);
    if(openStatus != LineStoreStatus::OK || !shard)
    {
        std::fprintf(stderr, "openLineStore failed, status=%d\n", static_cast<int>(openStatus));
        return 2;
    }

    // The concurrent-miss race this lock protects against: read the shard under the
    // lock, append only if the key is still absent, release. The append is already
    // torn-line-safe via O_APPEND; what needs the lock is the read-then-decide-then-write
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
