// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Standalone helper process for TestLineStore's two-process lock contention case.
//
// Invoked as: LineStoreLockHelper <shard-path> <version> <line> <repeat-count> <start-epoch-us>
//
// Opens the shard at <shard-path> (creating it with <version> as its first line if it
// does not exist), then <repeat-count> times acquires the shard's advisory lock, appends
// <line> suffixed with the iteration number, and releases the lock.
//
// The repeat loop is what makes this a real contention test. A single append of a short
// line is one write() and lands atomically whether or not the lock is held, so a
// one-append-per-process version passes even with the lock removed entirely (verified by
// mutation). Many interleaved lock/append/unlock cycles from two processes do not. This is the only way this plan exercises the file lock's actual contention
// path -- two threads in one process never contend on a POSIX fcntl() record lock, since
// those locks are per-process, so a real second OS process is required.
//
// Exit codes: 0 on success; 1 for a usage error; 2-4 mirror LineStore's own failure modes
// (open/lock/append) so the parent test can distinguish "helper never got the lock" from
// "helper's write itself failed."

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
    if(argc != 6)
    {
        std::fprintf(stderr,
                     "usage: %s <shard-path> <version> <line> <repeat-count> <start-epoch-us>\n",
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

    // D3's concurrent-miss race, which is what the lock actually protects: read the
    // shard under the lock, append only if this key is still absent, release. The append
    // itself is already torn-line-safe via O_APPEND -- what is NOT safe without the lock
    // is the read-then-decide-then-write sequence, where two processes can both observe
    // "absent" and both append, leaving a duplicate.
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

        // Widen the read-modify-write window deliberately. A real caller spends time here
        // parsing and comparing records; without a pause the window is a few microseconds
        // and two processes essentially never interleave inside it, which would make this
        // test pass whether or not the lock works.
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
