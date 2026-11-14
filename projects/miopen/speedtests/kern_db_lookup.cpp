// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Measures the kernel-cache lookup path in isolation from kernel compilation and loading: one
// KernDb instance is populated, then several threads look records up through it, separately for
// misses (the lookups made before every compilation) and hits. Kernel arguments are padded to a
// realistic length, since they are part of the key. Run it under strace to count file system
// calls, e.g. on a local disk and on NFS:
//
//   strace -f -c speedtest_kern_db_lookup /mnt/nfs/kdb.ukdb 1024 20000 8 16

#include <miopen/kern_db.hpp>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#if MIOPEN_ENABLE_SQLITE
namespace {
std::vector<miopen::KernelConfig>
MakeConfigs(std::size_t count, std::size_t blob_kib, const std::string& prefix)
{
    // Code objects compress roughly 3-5x; mimic that with a small alphabet.
    std::mt19937 gen{42};
    std::uniform_int_distribution<int> dist{0, 15};
    std::vector<miopen::KernelConfig> cfgs(count);
    for(std::size_t i = 0; i < count; ++i)
    {
        cfgs[i].kernel_name = prefix + std::to_string(i) + ".s.o";
        cfgs[i].kernel_args = "-mcpu=gfx950 -DCONFIG=" + std::to_string(i);
        cfgs[i].kernel_args.resize(600, 'x');
        cfgs[i].kernel_blob.resize(blob_kib * 1024);
        for(auto& c : cfgs[i].kernel_blob)
            c = static_cast<char>('a' + dist(gen));
    }
    return cfgs;
}

// Returns the mean wall time per lookup in microseconds, or a negative value if a lookup returned
// the wrong result.
double TimeLookups(miopen::KernDb& db,
                   const std::vector<miopen::KernelConfig>& cfgs,
                   std::size_t lookups,
                   std::size_t threads,
                   bool expect_hit)
{
    std::vector<int> errors(threads, 0);
    const auto start = std::chrono::steady_clock::now();
    std::vector<std::thread> workers;
    workers.reserve(threads);
    for(std::size_t t = 0; t < threads; ++t)
    {
        workers.emplace_back([&, t]() {
            for(std::size_t i = t; i < lookups; i += threads)
            {
                const auto& cfg   = cfgs[i % cfgs.size()];
                const auto record = db.FindRecordUnsafe(cfg);
                if(record.has_value() != expect_hit || (expect_hit && *record != cfg.kernel_blob))
                    ++errors[t];
            }
        });
    }
    for(auto& w : workers)
        w.join();
    const auto elapsed =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start);
    for(const auto e : errors)
        if(e != 0)
            return -1.0;
    return elapsed.count() / static_cast<double>(lookups);
}
} // namespace

int main(int argc, const char* argv[])
{
    if(argc < 2)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <db file> [records=1024] [lookups=20000] [threads=1] [blob KiB=16]\n";
        return EXIT_FAILURE;
    }
    const miopen::fs::path file = argv[1];
    const std::size_t records   = argc > 2 ? std::stoul(argv[2]) : 1024;
    const std::size_t lookups   = argc > 3 ? std::stoul(argv[3]) : 20000;
    const std::size_t threads   = argc > 4 ? std::stoul(argv[4]) : 1;
    const std::size_t blob_kib  = argc > 5 ? std::stoul(argv[5]) : 16;

    for(const auto* suffix : {"", "-wal", "-shm", "-journal"})
        miopen::fs::remove(file.string() + suffix);

    const auto stored  = MakeConfigs(records, blob_kib, "kernel_");
    const auto missing = MakeConfigs(records, 0, "missing_");

    miopen::KernDb db(miopen::DbKinds::KernelDb, file, false);
    for(const auto& cfg : stored)
        db.StoreRecordUnsafe(cfg);

    const auto miss_us = TimeLookups(db, missing, lookups, threads, false);
    const auto hit_us  = TimeLookups(db, stored, lookups, threads, true);
    if(miss_us < 0 || hit_us < 0)
    {
        std::cerr << "A lookup returned a wrong result\n";
        return EXIT_FAILURE;
    }

    const auto journal = db.sql.Exec("PRAGMA journal_mode;");
    std::cout << "records=" << records << " lookups=" << lookups << " threads=" << threads
              << " blob_kib=" << blob_kib << " journal_mode=" << journal.at(0).at("journal_mode")
              << " miss_us=" << miss_us << " hit_us=" << hit_us << std::endl;
    return EXIT_SUCCESS;
}
#else
int main()
{
    std::cerr << "Requires MIOPEN_ENABLE_SQLITE\n";
    return EXIT_FAILURE;
}
#endif
