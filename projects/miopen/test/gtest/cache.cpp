/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include <gtest/gtest.h>

#include <miopen/binary_cache.hpp>
#include <miopen/bz2.hpp>
#include <miopen/db.hpp>
#include <miopen/kern_db.hpp>
#include <miopen/temp_file.hpp>

#include "test.hpp"
#include "random.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <type_traits>
#include <vector>

#if MIOPEN_ENABLE_SQLITE
std::vector<char> random_bytes(size_t length)
{
    auto randchar = []() -> char {
        const char charset[]   = "0123456789"
                                 "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                 "abcdefghijklmnopqrstuvwxyz";
        const size_t max_index = (sizeof(charset) - 1);
        return charset[prng::gen_0_to_B(max_index)];
    };
    std::vector<char> v(length, 0);
    std::generate_n(v.begin(), length, randchar);
    return v;
}

TEST(CPU_Cache_NONE, check_bz2_compress)
{
    std::vector<char> to_compress;
    bool success = false;
    std::vector<char> cmprsd;

    EXPECT_TRUE(throws([&]() { cmprsd = miopen::compress(to_compress, &success); }));

    to_compress = random_bytes(4096);
    // if the following throws the test will fail
    cmprsd = miopen::compress(to_compress, nullptr);
    ASSERT_TRUE(!(cmprsd.empty()));
    cmprsd = miopen::compress(to_compress, &success);
    ASSERT_TRUE(success);
    ASSERT_TRUE(cmprsd.size() < to_compress.size());
}

TEST(CPU_Cache_NONE, check_bz2_decompress)
{
    std::vector<char> empty;

    std::vector<char> decompressed;

    EXPECT_TRUE(throws([&]() { decompressed = miopen::decompress(empty, 0); }));

    auto original = random_bytes(4096);
    bool success  = false;
    std::vector<char> compressed;
    compressed = miopen::compress(original, &success);
    ASSERT_TRUE(success);

    decompressed = miopen::decompress(compressed, original.size());
    ASSERT_TRUE(decompressed == original);

    EXPECT_TRUE(throws([&]() { decompressed = miopen::decompress(compressed, 10); }));

    ASSERT_TRUE(decompressed == miopen::decompress(compressed, original.size() + 10));
}

TEST(CPU_Cache_NONE, check_kern_db)
{
    miopen::KernelConfig cfg0;
    cfg0.kernel_name = "kernel1";
    cfg0.kernel_args = {random_bytes(512).data(), 512};
    cfg0.kernel_blob = random_bytes(8192);

    miopen::KernDb empty_db(miopen::DbKinds::KernelDb, "", false);
    EXPECT_TRUE(empty_db.RemoveRecordUnsafe(cfg0)); // for empty file, remove should succeed
    EXPECT_FALSE(empty_db.FindRecordUnsafe(cfg0));  // no record in empty database
    EXPECT_FALSE(empty_db.StoreRecordUnsafe(cfg0)); // storing in an empty database should fail

    {
        miopen::TempFile temp_file("tmp-kerndb");
        miopen::KernDb clean_db(miopen::DbKinds::KernelDb, temp_file, false);

        EXPECT_TRUE(clean_db.StoreRecordUnsafe(cfg0));
        auto readout = clean_db.FindRecordUnsafe(cfg0);
        EXPECT_TRUE(readout);
        EXPECT_TRUE(readout.value() == cfg0.kernel_blob);
        EXPECT_TRUE(clean_db.RemoveRecordUnsafe(cfg0));
        EXPECT_FALSE(clean_db.FindRecordUnsafe(cfg0));
    }

    {
        miopen::TempFile temp_file("tmp-kerndb");
        miopen::KernDb err_db(
            miopen::DbKinds::KernelDb,
            temp_file,
            false,
            [](const std::vector<char>&, bool* success) {
                *success = false;
                return std::vector<char>{};
            },
            [](const std::vector<char>&, unsigned int) -> std::vector<char> {
                throw;
            }); // error compressing
        // Even if compression fails, it should still work
        EXPECT_TRUE(err_db.StoreRecordUnsafe(cfg0));
        // In which case decompresion should not be called
        EXPECT_TRUE(err_db.FindRecordUnsafe(cfg0));
        EXPECT_TRUE(err_db.RemoveRecordUnsafe(cfg0));
    }
}

// Compile-time check: GetDbInstance<KernDb> must resolve to the rank<1> caching overload
// (returning a reference). If the SFINAE probe drifts again, this fires at build time.
static_assert(std::is_lvalue_reference_v<decltype(miopen::GetDbInstance<miopen::KernDb>(
                  miopen::DbKinds::KernelDb, {}, true))>,
              "GetDbInstance<KernDb> should return an lvalue reference (cached overload)");

TEST(CPU_Cache_NONE, check_kern_db_cached_reuse)
{
    miopen::TempFile temp_file("tmp-kerndb-cached");
    auto& db1 = miopen::KernDb::GetCached(miopen::DbKinds::KernelDb, temp_file, false);
    auto& db2 = miopen::KernDb::GetCached(miopen::DbKinds::KernelDb, temp_file, false);
    EXPECT_EQ(&db1, &db2);
    miopen::KernDb::EvictCached(temp_file, false);
}

TEST(CPU_Cache_NONE, check_kern_db_cached_distinct_paths)
{
    miopen::TempFile temp_file_a("tmp-kerndb-cached-a");
    miopen::TempFile temp_file_b("tmp-kerndb-cached-b");
    auto& db_a = miopen::KernDb::GetCached(miopen::DbKinds::KernelDb, temp_file_a, false);
    auto& db_b = miopen::KernDb::GetCached(miopen::DbKinds::KernelDb, temp_file_b, false);
    EXPECT_NE(&db_a, &db_b);
    miopen::KernDb::EvictCached(temp_file_a, false);
    miopen::KernDb::EvictCached(temp_file_b, false);
}

TEST(CPU_Cache_NONE, check_kern_db_cached_thread_safety)
{
    miopen::TempFile temp_file("tmp-kerndb-cached-mt");
    auto& db = miopen::KernDb::GetCached(miopen::DbKinds::KernelDb, temp_file, false);

    constexpr int kNumThreads = 8;

    // Each thread gets its own config with a unique key and blob.
    struct ThreadData
    {
        miopen::KernelConfig cfg;
        std::vector<char> expected_blob;
    };
    std::vector<ThreadData> thread_data(kNumThreads);
    for(int i = 0; i < kNumThreads; ++i)
    {
        thread_data[i].cfg.kernel_name = "kernel_mt_" + std::to_string(i);
        thread_data[i].cfg.kernel_args = {random_bytes(512).data(), 512};
        thread_data[i].cfg.kernel_blob = random_bytes(8192);
        thread_data[i].expected_blob   = thread_data[i].cfg.kernel_blob;
    }

    // Use vector<int> instead of vector<bool> to avoid bit-packing data races.
    std::vector<int> results(kNumThreads, 0);
    std::vector<std::thread> threads;

    for(int i = 0; i < kNumThreads; ++i)
    {
        threads.emplace_back([&db, &thread_data, &results, i]() {
            // Each thread stores its own record, then reads it back.
            if(!db.StoreRecord(thread_data[i].cfg))
                return;
            auto rec = db.FindRecord(thread_data[i].cfg);
            if(rec && rec.value() == thread_data[i].expected_blob)
                results[i] = 1;
        });
    }

    for(auto& t : threads)
        t.join();

    for(int i = 0; i < kNumThreads; ++i)
        EXPECT_EQ(results[i], 1) << "Thread " << i << " failed to store/find its own record";

    miopen::KernDb::EvictCached(temp_file, false);
}

#if MIOPEN_ENABLE_SQLITE_KERN_CACHE
namespace {
miopen::KernelConfig MakeKernelConfig(const std::string& name, std::size_t blob_size = 4096)
{
    miopen::KernelConfig cfg;
    cfg.kernel_name = name;
    cfg.kernel_args = "-DKERNEL=" + name;
    cfg.kernel_blob = random_bytes(blob_size);
    return cfg;
}

// Limits that no test reaches unless it means to.
const miopen::KernDb::BatchLimits unbounded_batch{1000, 1ULL << 30, std::chrono::hours{1}};

// Reads through a separate connection, i.e. what another process would see.
bool IsCommitted(const miopen::fs::path& file, const miopen::KernelConfig& cfg)
{
    miopen::KernDb reader(miopen::DbKinds::KernelDb, file, false);
    const auto record = reader.FindRecordUnsafe(cfg);
    return record && record.value() == cfg.kernel_blob;
}

std::string IntegrityCheck(const miopen::KernDb& db)
{
    const auto res = db.sql.Exec("PRAGMA integrity_check;");
    return res.empty() ? "" : res[0].at("integrity_check");
}
} // namespace

TEST(CPU_Cache_NONE, check_kern_db_store_without_batch_commits_immediately)
{
    miopen::TempFile temp_file("tmp-kerndb-nobatch");
    miopen::KernDb db(miopen::DbKinds::KernelDb, temp_file, false);
    db.SetBatchLimits(unbounded_batch);
    const auto cfg = MakeKernelConfig("kernel_nobatch");

    ASSERT_FALSE(miopen::KernDb::IsBatchActive());
    EXPECT_TRUE(db.StoreRecordUnsafe(cfg));
    EXPECT_EQ(db.PendingCount(), 0u);
    EXPECT_TRUE(IsCommitted(temp_file, cfg));
}

TEST(CPU_Cache_NONE, check_kern_db_batch_multi_record_commit)
{
    miopen::TempFile temp_file("tmp-kerndb-batch");
    miopen::KernDb db(miopen::DbKinds::KernelDb, temp_file, false);
    db.SetBatchLimits({4, 1ULL << 30, std::chrono::hours{1}});

    std::vector<miopen::KernelConfig> cfgs;
    for(int i = 0; i < 6; ++i)
        cfgs.push_back(MakeKernelConfig("kernel_batch_" + std::to_string(i)));

    miopen::KernelCacheBatch batch;
    for(int i = 0; i < 3; ++i)
        EXPECT_TRUE(db.StoreRecordUnsafe(cfgs[i]));

    // Below the record limit nothing is committed, but this instance already serves the records.
    EXPECT_EQ(db.PendingCount(), 3u);
    for(int i = 0; i < 3; ++i)
    {
        const auto record = db.FindRecordUnsafe(cfgs[i]);
        ASSERT_TRUE(record);
        EXPECT_EQ(record.value(), cfgs[i].kernel_blob);
        EXPECT_FALSE(IsCommitted(temp_file, cfgs[i]));
    }

    // The fourth record reaches the limit and all four are committed together.
    EXPECT_TRUE(db.StoreRecordUnsafe(cfgs[3]));
    EXPECT_EQ(db.PendingCount(), 0u);
    for(int i = 0; i < 4; ++i)
        EXPECT_TRUE(IsCommitted(temp_file, cfgs[i]));

    // Ending the batch commits the remainder.
    EXPECT_TRUE(db.StoreRecordUnsafe(cfgs[4]));
    EXPECT_TRUE(db.StoreRecordUnsafe(cfgs[5]));
    EXPECT_EQ(db.PendingCount(), 2u);
    EXPECT_FALSE(IsCommitted(temp_file, cfgs[4]));
    batch.Commit();
    EXPECT_FALSE(miopen::KernDb::IsBatchActive());
    EXPECT_EQ(db.PendingCount(), 0u);
    for(const auto& cfg : cfgs)
        EXPECT_TRUE(IsCommitted(temp_file, cfg));
    EXPECT_EQ(IntegrityCheck(db), "ok");
}

TEST(CPU_Cache_NONE, check_kern_db_batch_byte_and_age_limits)
{
    miopen::TempFile temp_file("tmp-kerndb-batch-limits");
    miopen::KernDb db(miopen::DbKinds::KernelDb, temp_file, false);
    miopen::KernelCacheBatch batch;

    // Byte limit: any stored blob exceeds one byte.
    db.SetBatchLimits({1000, 1, std::chrono::hours{1}});
    const auto by_size = MakeKernelConfig("kernel_by_size");
    EXPECT_TRUE(db.StoreRecordUnsafe(by_size));
    EXPECT_EQ(db.PendingCount(), 0u);
    EXPECT_TRUE(IsCommitted(temp_file, by_size));

    // Age limit, enforced on demand by the batch owner.
    db.SetBatchLimits(unbounded_batch);
    const auto by_age = MakeKernelConfig("kernel_by_age");
    EXPECT_TRUE(db.StoreRecordUnsafe(by_age));
    batch.FlushExpired();
    EXPECT_EQ(db.PendingCount(), 1u);
    db.SetBatchLimits({1000, 1ULL << 30, std::chrono::milliseconds{0}});
    batch.FlushExpired();
    EXPECT_EQ(db.PendingCount(), 0u);
    EXPECT_TRUE(IsCommitted(temp_file, by_age));
}

TEST(CPU_Cache_NONE, check_kern_db_batch_destruction_commits)
{
    miopen::TempFile temp_file("tmp-kerndb-batch-dtor");
    const auto in_scope    = MakeKernelConfig("kernel_scope_end");
    const auto in_instance = MakeKernelConfig("kernel_instance_end");
    miopen::KernDb db(miopen::DbKinds::KernelDb, temp_file, false);
    db.SetBatchLimits(unbounded_batch);
    {
        miopen::KernelCacheBatch batch;
        EXPECT_TRUE(db.StoreRecordUnsafe(in_scope));

        // An instance destroyed while its records are pending commits them first.
        {
            miopen::KernDb short_lived(miopen::DbKinds::KernelDb, temp_file, false);
            short_lived.SetBatchLimits(unbounded_batch);
            EXPECT_TRUE(short_lived.StoreRecordUnsafe(in_instance));
            EXPECT_EQ(short_lived.PendingCount(), 1u);
        }
        EXPECT_TRUE(IsCommitted(temp_file, in_instance));
        EXPECT_FALSE(IsCommitted(temp_file, in_scope));
        // Leaving the scope without Commit(), e.g. by an exception, still ends the batch.
    }
    EXPECT_FALSE(miopen::KernDb::IsBatchActive());
    EXPECT_EQ(db.PendingCount(), 0u);
    EXPECT_TRUE(IsCommitted(temp_file, in_scope));
}

TEST(CPU_Cache_NONE, check_kern_db_batch_nested)
{
    miopen::TempFile temp_file("tmp-kerndb-batch-nested");
    miopen::KernDb db(miopen::DbKinds::KernelDb, temp_file, false);
    db.SetBatchLimits(unbounded_batch);
    const auto first  = MakeKernelConfig("kernel_nested_first");
    const auto second = MakeKernelConfig("kernel_nested_second");

    miopen::KernelCacheBatch outer;
    {
        miopen::KernelCacheBatch inner;
        EXPECT_TRUE(db.StoreRecordUnsafe(first));
        // Ending any batch commits everything stored so far.
        inner.Commit();
        EXPECT_TRUE(IsCommitted(temp_file, first));
    }
    // The outer batch is still active.
    EXPECT_TRUE(miopen::KernDb::IsBatchActive());
    EXPECT_TRUE(db.StoreRecordUnsafe(second));
    EXPECT_EQ(db.PendingCount(), 1u);
    outer.Commit();
    EXPECT_TRUE(IsCommitted(temp_file, second));
}

TEST(CPU_Cache_NONE, check_kern_db_batch_duplicate_and_remove)
{
    miopen::TempFile temp_file("tmp-kerndb-batch-dup");
    miopen::KernDb db(miopen::DbKinds::KernelDb, temp_file, false);
    db.SetBatchLimits(unbounded_batch);

    auto committed = MakeKernelConfig("kernel_dup");
    EXPECT_TRUE(db.StoreRecordUnsafe(committed));
    ASSERT_TRUE(IsCommitted(temp_file, committed));

    miopen::KernelCacheBatch batch;

    // A pending store shadows the committed record with the same key; the latest store wins.
    auto first_update         = committed;
    first_update.kernel_blob  = random_bytes(4096);
    auto second_update        = committed;
    second_update.kernel_blob = random_bytes(2048);
    EXPECT_TRUE(db.StoreRecordUnsafe(first_update));
    EXPECT_TRUE(db.StoreRecordUnsafe(second_update));
    EXPECT_EQ(db.PendingCount(), 1u);
    EXPECT_EQ(db.FindRecordUnsafe(committed).value(), second_update.kernel_blob);
    EXPECT_TRUE(IsCommitted(temp_file, committed));

    // Removing drops the pending record as well as the committed one.
    const auto removed = MakeKernelConfig("kernel_removed");
    EXPECT_TRUE(db.StoreRecordUnsafe(removed));
    EXPECT_TRUE(db.RemoveRecordUnsafe(removed));
    EXPECT_FALSE(db.FindRecordUnsafe(removed));

    batch.Commit();
    EXPECT_TRUE(IsCommitted(temp_file, second_update));
    EXPECT_FALSE(IsCommitted(temp_file, removed));
    EXPECT_EQ(db.sql.Exec("SELECT COUNT(*) AS n FROM kern_db;")[0].at("n"), "1");
}

TEST(CPU_Cache_NONE, check_kern_db_batch_failed_commit_rolls_back)
{
    miopen::TempFile temp_file("tmp-kerndb-batch-fail");
    // Store blobs uncompressed, so that an empty blob reaches SQLite as NULL and violates the
    // NOT NULL constraint on kernel_blob in the middle of the batch.
    miopen::KernDb db(
        miopen::DbKinds::KernelDb,
        temp_file,
        false,
        [](const std::vector<char>&, bool* success) {
            *success = false;
            return std::vector<char>{};
        },
        [](const std::vector<char>& blob, unsigned int) { return blob; });
    db.SetBatchLimits({3, 1ULL << 30, std::chrono::hours{1}});

    const auto before = MakeKernelConfig("kernel_before");
    EXPECT_TRUE(db.StoreRecordUnsafe(before));

    miopen::KernelCacheBatch batch;
    const auto good_1 = MakeKernelConfig("kernel_good_1");
    const auto bad    = MakeKernelConfig("kernel_bad", 0);
    const auto good_2 = MakeKernelConfig("kernel_good_2");
    EXPECT_TRUE(db.StoreRecordUnsafe(good_1));
    EXPECT_TRUE(db.StoreRecordUnsafe(bad));
    // The third store triggers the commit, which fails; the failure reaches the caller.
    EXPECT_ANY_THROW(db.StoreRecordUnsafe(good_2));

    // The whole batch was rolled back and discarded; earlier commits are untouched.
    EXPECT_EQ(db.PendingCount(), 0u);
    EXPECT_FALSE(db.sql.InTransaction());
    EXPECT_FALSE(IsCommitted(temp_file, good_1));
    EXPECT_FALSE(IsCommitted(temp_file, good_2));
    EXPECT_FALSE(db.FindRecordUnsafe(good_1));
    EXPECT_TRUE(IsCommitted(temp_file, before));
    EXPECT_EQ(IntegrityCheck(db), "ok");

    // The database remains writable.
    EXPECT_TRUE(db.StoreRecordUnsafe(good_1));
    batch.Commit();
    EXPECT_TRUE(IsCommitted(temp_file, good_1));

    // A failing final commit is reported by Commit() rather than swallowed.
    miopen::KernelCacheBatch failing_batch;
    EXPECT_TRUE(db.StoreRecordUnsafe(bad));
    EXPECT_ANY_THROW(failing_batch.Commit());
    EXPECT_EQ(db.PendingCount(), 0u);
    EXPECT_FALSE(miopen::KernDb::IsBatchActive());
}

TEST(CPU_Cache_NONE, check_kern_db_batch_concurrent_writers_and_reader)
{
    miopen::TempFile temp_file("tmp-kerndb-batch-mt");
    miopen::KernDb db(miopen::DbKinds::KernelDb, temp_file, false);
    db.SetBatchLimits({5, 1ULL << 30, std::chrono::hours{1}});

    constexpr int num_threads        = 8;
    constexpr int records_per_thread = 16;
    std::vector<std::vector<miopen::KernelConfig>> cfgs(num_threads);
    for(int t = 0; t < num_threads; ++t)
        for(int i = 0; i < records_per_thread; ++i)
            cfgs[t].push_back(
                MakeKernelConfig("kernel_mt_" + std::to_string(t) + "_" + std::to_string(i), 1024));

    miopen::KernelCacheBatch batch;
    std::atomic<bool> writing{true};
    std::atomic<int> reader_errors{0};
    // Another connection reads while batches commit; it must never see a corrupt record.
    std::thread reader([&]() {
        miopen::KernDb other(miopen::DbKinds::KernelDb, temp_file, false);
        while(writing)
        {
            for(const auto& cfg : cfgs[0])
            {
                try
                {
                    const auto record = other.FindRecordUnsafe(cfg);
                    if(record && record.value() != cfg.kernel_blob)
                        ++reader_errors;
                }
                catch(...)
                {
                    ++reader_errors;
                }
            }
        }
    });

    std::vector<int> results(num_threads, 0);
    std::vector<std::thread> writers;
    for(int t = 0; t < num_threads; ++t)
    {
        writers.emplace_back([&, t]() {
            for(const auto& cfg : cfgs[t])
            {
                if(!db.StoreRecordUnsafe(cfg))
                    return;
                const auto record = db.FindRecordUnsafe(cfg);
                if(!record || record.value() != cfg.kernel_blob)
                    return;
            }
            results[t] = 1;
        });
    }
    for(auto& w : writers)
        w.join();
    batch.Commit();
    writing = false;
    reader.join();

    for(int t = 0; t < num_threads; ++t)
        EXPECT_EQ(results[t], 1) << "Thread " << t << " could not read back its own records";
    EXPECT_EQ(reader_errors, 0);
    EXPECT_EQ(db.PendingCount(), 0u);
    for(const auto& per_thread : cfgs)
        for(const auto& cfg : per_thread)
            EXPECT_TRUE(IsCommitted(temp_file, cfg));
    EXPECT_FALSE(db.sql.InTransaction());
    EXPECT_EQ(IntegrityCheck(db), "ok");
}
#endif // MIOPEN_ENABLE_SQLITE_KERN_CACHE
#endif

TEST(CPU_Cache_NONE, check_cache_file)
{
    auto p = miopen::GetCacheFile("gfx", "base", "args");
    EXPECT_TRUE(p.filename() == miopen::make_object_file_name("base"));
}
