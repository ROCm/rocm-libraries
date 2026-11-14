// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Measures the kernel-cache store path in isolation from kernel compilation: several threads
// store prepared kernel binaries into a fresh user kernel database, either one commit per binary
// or inside a KernelCacheBatch. Run it under strace to count synchronization and write calls,
// e.g. on a local disk and on NFS:
//
//   strace -f -c -e trace=fsync,fdatasync,pwrite64,write \
//       speedtest_kern_db_batch /mnt/nfs/kdb.ukdb 512 64 8 batch
//
// MIOPEN_LOG_LEVEL=6 logs the effective journal settings and every committed batch.

#include <miopen/binary_cache.hpp>
#include <miopen/kern_db.hpp>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#if MIOPEN_ENABLE_SQLITE_KERN_CACHE
int main(int argc, const char* argv[])
{
    if(argc < 2)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <db file> [records=256] [blob KiB=64] [threads=8] [batch|single=batch]\n";
        return EXIT_FAILURE;
    }
    const miopen::fs::path file = argv[1];
    const std::size_t records   = argc > 2 ? std::stoul(argv[2]) : 256;
    const std::size_t blob_kib  = argc > 3 ? std::stoul(argv[3]) : 64;
    const std::size_t threads   = argc > 4 ? std::stoul(argv[4]) : 8;
    const bool batch            = argc > 5 ? std::string{argv[5]} != "single" : true;

    for(const auto* suffix : {"", "-wal", "-shm", "-journal"})
        miopen::fs::remove(file.string() + suffix);

    // Code objects compress roughly 3-5x; mimic that with a small alphabet.
    std::vector<miopen::KernelConfig> cfgs(records);
    std::mt19937 gen{42};
    std::uniform_int_distribution<int> dist{0, 15};
    for(std::size_t i = 0; i < records; ++i)
    {
        cfgs[i].kernel_name = "kernel_" + std::to_string(i) + ".s.o";
        cfgs[i].kernel_args = "-mcpu=gfx942 -DCONFIG=" + std::to_string(i);
        cfgs[i].kernel_blob.resize(blob_kib * 1024);
        for(auto& c : cfgs[i].kernel_blob)
            c = static_cast<char>('a' + dist(gen));
    }

    miopen::KernDb db(miopen::DbKinds::KernelDb, file, false);

    const auto start = std::chrono::steady_clock::now();
    {
        std::optional<miopen::KernelCacheBatch> scope;
        if(batch)
            scope.emplace();
        std::vector<std::thread> workers;
        workers.reserve(threads);
        for(std::size_t t = 0; t < threads; ++t)
        {
            workers.emplace_back([&, t]() {
                for(std::size_t i = t; i < records; i += threads)
                    db.StoreRecordUnsafe(cfgs[i]);
            });
        }
        for(auto& w : workers)
            w.join();
        if(scope)
            scope->Commit();
    }
    const auto elapsed =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start);

    const auto journal     = db.sql.Exec("PRAGMA journal_mode;");
    const auto synchronous = db.sql.Exec("PRAGMA synchronous;");
    const auto limits      = miopen::KernDb::DefaultBatchLimits();
    std::cout << "mode=" << (batch ? "batch" : "single") << " records=" << records
              << " blob_kib=" << blob_kib << " threads=" << threads
              << " journal_mode=" << journal.at(0).at("journal_mode")
              << " synchronous=" << synchronous.at(0).at("synchronous")
              << " max_records=" << limits.max_records << " max_bytes=" << limits.max_bytes
              << " max_age_ms=" << limits.max_age.count() << " wall_ms=" << elapsed.count()
              << std::endl;
    return EXIT_SUCCESS;
}
#else
int main()
{
    std::cerr << "Requires MIOPEN_ENABLE_SQLITE_KERN_CACHE\n";
    return EXIT_FAILURE;
}
#endif
