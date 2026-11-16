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

namespace {
miopen::KernelConfig
MakeKernelConfig(const std::string& name, const std::string& args, std::size_t blob_size = 4096)
{
    miopen::KernelConfig cfg;
    cfg.kernel_name = name;
    cfg.kernel_args = args;
    cfg.kernel_blob = random_bytes(blob_size);
    return cfg;
}

struct StatementRuns
{
    std::size_t live = 0;
    int select       = -1;
    int insert       = -1;
    int remove       = -1;
};

// The statements alive on the instance's connection and how many times each kind has completed a
// run (one or more steps followed by a reset). A kind that was never prepared reports -1.
StatementRuns GetStatementRuns(const miopen::KernDb& db)
{
    StatementRuns runs;
    for(const auto& stmt : db.sql.LiveStatements())
    {
        ++runs.live;
        if(stmt.sql.rfind("SELECT kernel_blob", 0) == 0)
            runs.select = stmt.runs;
        else if(stmt.sql.rfind("INSERT OR REPLACE", 0) == 0)
            runs.insert = stmt.runs;
        else if(stmt.sql.rfind("DELETE FROM", 0) == 0)
            runs.remove = stmt.runs;
    }
    return runs;
}
} // namespace

TEST(CPU_Cache_NONE, check_kern_db_keys_are_bound_not_interpolated)
{
    miopen::TempFile temp_file("tmp-kerndb-quoted");
    miopen::KernDb db(miopen::DbKinds::KernelDb, temp_file, false);

    // Keys containing SQL metacharacters are matched literally.
    const auto quoted =
        MakeKernelConfig("it's_a_kernel.s", R"(-DNAME='a' -DOTHER="b" -DPCT=%d?; -- tail)");
    const auto plain = MakeKernelConfig("plain.s", "-DNAME=a");

    ASSERT_TRUE(db.StoreRecordUnsafe(quoted));
    ASSERT_TRUE(db.StoreRecordUnsafe(plain));
    EXPECT_EQ(db.FindRecordUnsafe(quoted).value(), quoted.kernel_blob);
    EXPECT_EQ(db.FindRecordUnsafe(plain).value(), plain.kernel_blob);

    // A key that would widen an interpolated WHERE clause must not match or remove any record.
    const auto injected = MakeKernelConfig("plain.s", "x') OR ('1'='1", 0);
    EXPECT_FALSE(db.FindRecordUnsafe(injected));
    EXPECT_TRUE(db.RemoveRecordUnsafe(injected));
    EXPECT_TRUE(db.FindRecordUnsafe(plain));
    EXPECT_TRUE(db.FindRecordUnsafe(quoted));

    EXPECT_TRUE(db.RemoveRecordUnsafe(quoted));
    EXPECT_FALSE(db.FindRecordUnsafe(quoted));
    EXPECT_EQ(db.FindRecordUnsafe(plain).value(), plain.kernel_blob);
}

TEST(CPU_Cache_NONE, check_kern_db_statements_are_prepared_once)
{
    miopen::TempFile temp_file("tmp-kerndb-prepared");
    miopen::KernDb db(miopen::DbKinds::KernelDb, temp_file, false);
    EXPECT_EQ(GetStatementRuns(db).live, 0u);

    constexpr int n = 10;
    std::vector<miopen::KernelConfig> cfgs;
    for(int i = 0; i < n; ++i)
        cfgs.push_back(MakeKernelConfig("kernel_" + std::to_string(i), "-DI=" + std::to_string(i)));
    for(const auto& cfg : cfgs)
        ASSERT_TRUE(db.StoreRecordUnsafe(cfg));
    for(const auto& cfg : cfgs)
        EXPECT_EQ(db.FindRecordUnsafe(cfg).value(), cfg.kernel_blob);
    for(const auto& cfg : cfgs)
        EXPECT_TRUE(db.RemoveRecordUnsafe(cfg));
    for(const auto& cfg : cfgs)
        EXPECT_FALSE(db.FindRecordUnsafe(cfg));

    // One statement per kind, each prepared once and run (and reset) once per operation.
    const auto runs = GetStatementRuns(db);
    EXPECT_EQ(runs.live, 3u);
    EXPECT_EQ(runs.insert, n);
    EXPECT_EQ(runs.select, 2 * n);
    EXPECT_EQ(runs.remove, n);
}

TEST(CPU_Cache_NONE, check_kern_db_reused_statements_are_reset)
{
    miopen::TempFile temp_file("tmp-kerndb-reuse");
    miopen::KernDb db(miopen::DbKinds::KernelDb, temp_file, false);
    miopen::KernDb other(miopen::DbKinds::KernelDb, temp_file, false);

    std::vector<miopen::KernelConfig> cfgs;
    for(int i = 0; i < 4; ++i)
        cfgs.push_back(MakeKernelConfig(
            "kernel_reuse_" + std::to_string(i), "-DI=" + std::to_string(i), 1024 * (i + 1)));

    // Interleaved hits, misses, stores and removes on the same instance must each see only their
    // own key, never values bound or rows left over from the previous use.
    for(int round = 0; round < 3; ++round)
    {
        for(const auto& cfg : cfgs)
        {
            EXPECT_FALSE(db.FindRecordUnsafe(cfg));
            ASSERT_TRUE(db.StoreRecordUnsafe(cfg));
        }
        for(const auto& cfg : cfgs)
            EXPECT_EQ(db.FindRecordUnsafe(cfg).value(), cfg.kernel_blob);
        for(const auto& cfg : cfgs)
            EXPECT_TRUE(db.RemoveRecordUnsafe(cfg));
    }

    // A lookup that stopped at its row must not keep a read transaction open: another connection
    // can then checkpoint the whole WAL, and its later writes are visible to the next lookup.
    ASSERT_TRUE(db.StoreRecordUnsafe(cfgs[0]));
    ASSERT_TRUE(db.FindRecordUnsafe(cfgs[0]));
    ASSERT_TRUE(other.StoreRecordUnsafe(cfgs[1]));
    const auto checkpoint = other.sql.Exec("PRAGMA wal_checkpoint(TRUNCATE);");
    ASSERT_FALSE(checkpoint.empty());
    EXPECT_EQ(checkpoint[0].at("busy"), "0");
    EXPECT_EQ(db.FindRecordUnsafe(cfgs[1]).value(), cfgs[1].kernel_blob);
    EXPECT_TRUE(other.RemoveRecordUnsafe(cfgs[0]));
    EXPECT_FALSE(db.FindRecordUnsafe(cfgs[0]));
}

TEST(CPU_Cache_NONE, check_kern_db_reused_statements_recover_from_errors)
{
    miopen::TempFile temp_file("tmp-kerndb-errors");
    // Store blobs uncompressed, so that an empty blob reaches SQLite as NULL and makes the insert
    // fail on the NOT NULL constraint of kernel_blob.
    miopen::KernDb db(
        miopen::DbKinds::KernelDb,
        temp_file,
        false,
        [](const std::vector<char>&, bool* success) {
            *success = false;
            return std::vector<char>{};
        },
        [](const std::vector<char>& blob, unsigned int) { return blob; });
    miopen::KernDb other(miopen::DbKinds::KernelDb, temp_file, false);

    const auto good    = MakeKernelConfig("kernel_good", "-DGOOD");
    const auto bad     = MakeKernelConfig("kernel_bad", "-DBAD", 0);
    const auto corrupt = MakeKernelConfig("kernel_corrupt", "-DCORRUPT");

    // A failed insert leaves the reused statement usable, without the failed record's bindings.
    EXPECT_ANY_THROW(db.StoreRecordUnsafe(bad));
    EXPECT_TRUE(db.StoreRecordUnsafe(good));
    EXPECT_EQ(db.FindRecordUnsafe(good).value(), good.kernel_blob);
    EXPECT_FALSE(db.FindRecordUnsafe(bad));

    // A lookup that fails its integrity check still leaves the select statement usable.
    ASSERT_TRUE(db.StoreRecordUnsafe(corrupt));
    other.sql.Exec("UPDATE kern_db SET kernel_hash = 'bad' WHERE kernel_name = 'kernel_corrupt';");
    EXPECT_ANY_THROW(db.FindRecordUnsafe(corrupt));
    EXPECT_EQ(db.FindRecordUnsafe(good).value(), good.kernel_blob);
    EXPECT_TRUE(db.RemoveRecordUnsafe(corrupt));
    EXPECT_FALSE(db.FindRecordUnsafe(corrupt));

    // Errors neither re-prepare nor leak statements.
    const auto runs = GetStatementRuns(db);
    EXPECT_EQ(runs.live, 3u);
    EXPECT_EQ(runs.insert, 3);
    EXPECT_EQ(runs.select, 5);
    EXPECT_EQ(runs.remove, 1);
}

TEST(CPU_Cache_NONE, check_kern_db_busy_retry_keeps_bindings)
{
    miopen::TempFile temp_file("tmp-kerndb-busy");
    miopen::KernDb db(miopen::DbKinds::KernelDb, temp_file, false);
    miopen::KernDb other(miopen::DbKinds::KernelDb, temp_file, false);

    const auto committed = MakeKernelConfig("kernel_committed", "-DCOMMITTED");
    const auto blocked   = MakeKernelConfig("kernel_blocked", "-DBLOCKED");
    ASSERT_TRUE(db.StoreRecordUnsafe(committed));

    // Another connection holds the write lock far longer than the busy timeout, so the store
    // returns SQLITE_BUSY and is retried with its statement and bindings in place.
    other.sql.Exec("BEGIN IMMEDIATE;");
    const auto hold  = std::chrono::milliseconds{300};
    const auto start = std::chrono::steady_clock::now();
    std::thread writer([&]() { EXPECT_TRUE(db.StoreRecordUnsafe(blocked)); });
    std::this_thread::sleep_for(hold);
    other.sql.Exec("COMMIT;");
    writer.join();
    EXPECT_GE(std::chrono::steady_clock::now() - start, hold);

    EXPECT_EQ(db.FindRecordUnsafe(blocked).value(), blocked.kernel_blob);
    EXPECT_EQ(other.FindRecordUnsafe(blocked).value(), blocked.kernel_blob);
    EXPECT_EQ(db.FindRecordUnsafe(committed).value(), committed.kernel_blob);
    EXPECT_EQ(GetStatementRuns(db).live, 2u);
}

TEST(CPU_Cache_NONE, check_kern_db_statements_finalized_on_release)
{
    // Closing a connection fails while it still has unfinalized statements. When the last
    // connection to a WAL database closes cleanly, SQLite checkpoints and deletes the WAL file.
    const auto wal_removed_on_release = [](const miopen::fs::path& file, auto&& use_and_release) {
        const miopen::fs::path wal = file.string() + "-wal";
        use_and_release();
        return !miopen::fs::exists(wal);
    };
    const auto cfg = MakeKernelConfig("kernel_release", "-DRELEASE");

    miopen::TempFile direct_file("tmp-kerndb-release");
    EXPECT_TRUE(wal_removed_on_release(direct_file, [&]() {
        miopen::KernDb db(miopen::DbKinds::KernelDb, direct_file, false);
        ASSERT_EQ(db.sql.Exec("PRAGMA journal_mode;").at(0).at("journal_mode"), "wal");
        ASSERT_TRUE(db.StoreRecordUnsafe(cfg));
        ASSERT_TRUE(db.FindRecordUnsafe(cfg));
        ASSERT_TRUE(db.RemoveRecordUnsafe(cfg));
        ASSERT_EQ(GetStatementRuns(db).live, 3u);
    }));

    miopen::TempFile cached_file("tmp-kerndb-release-cached");
    EXPECT_TRUE(wal_removed_on_release(cached_file, [&]() {
        auto& db = miopen::KernDb::GetCached(miopen::DbKinds::KernelDb, cached_file, false);
        ASSERT_TRUE(db.StoreRecordUnsafe(cfg));
        ASSERT_TRUE(db.FindRecordUnsafe(cfg));
        ASSERT_EQ(GetStatementRuns(db).live, 2u);
        miopen::KernDb::EvictCached(cached_file, false);
    }));
}

TEST(CPU_Cache_NONE, check_kern_db_shared_statements_concurrent)
{
    miopen::TempFile temp_file("tmp-kerndb-shared-mt");
    auto& db = miopen::KernDb::GetCached(miopen::DbKinds::KernelDb, temp_file, false);

    constexpr int num_threads = 8;
    constexpr int iterations  = 25;
    std::vector<miopen::KernelConfig> shared;
    for(int i = 0; i < 4; ++i)
        shared.push_back(MakeKernelConfig("kernel_shared_" + std::to_string(i), "-DSHARED", 2048));
    for(const auto& cfg : shared)
        ASSERT_TRUE(db.StoreRecordUnsafe(cfg));

    std::vector<miopen::KernelConfig> owned;
    std::vector<miopen::KernelConfig> missing;
    for(int t = 0; t < num_threads; ++t)
    {
        owned.push_back(
            MakeKernelConfig("kernel_own_" + std::to_string(t), "-DT=" + std::to_string(t)));
        missing.push_back(
            MakeKernelConfig("kernel_missing_" + std::to_string(t), "-DT=" + std::to_string(t), 0));
    }

    // Every thread mixes hits, misses, stores and removes on the same statements; each result
    // must belong to the key that thread asked for.
    std::vector<int> errors(num_threads, 0);
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for(int t = 0; t < num_threads; ++t)
    {
        threads.emplace_back([&, t]() {
            const auto& own = owned[t];
            try
            {
                for(int i = 0; i < iterations; ++i)
                {
                    const auto& other = shared[(t + i) % shared.size()];
                    errors[t] += db.StoreRecordUnsafe(own) ? 0 : 1;
                    errors[t] += db.FindRecordUnsafe(own) == own.kernel_blob ? 0 : 1;
                    errors[t] += db.FindRecordUnsafe(other) == other.kernel_blob ? 0 : 1;
                    errors[t] += db.FindRecordUnsafe(missing[t]) ? 1 : 0;
                    errors[t] += db.RemoveRecordUnsafe(own) ? 0 : 1;
                    errors[t] += db.FindRecordUnsafe(own) ? 1 : 0;
                }
            }
            catch(...)
            {
                ++errors[t];
            }
        });
    }
    for(auto& thread : threads)
        thread.join();

    for(int t = 0; t < num_threads; ++t)
        EXPECT_EQ(errors[t], 0) << "Thread " << t;
    const auto runs = GetStatementRuns(db);
    EXPECT_EQ(runs.live, 3u);
    EXPECT_EQ(runs.insert, static_cast<int>(shared.size()) + num_threads * iterations);
    EXPECT_EQ(runs.select, 4 * num_threads * iterations);
    EXPECT_EQ(runs.remove, num_threads * iterations);

    miopen::KernDb::EvictCached(temp_file, false);
}
#endif

TEST(CPU_Cache_NONE, check_cache_file)
{
    auto p = miopen::GetCacheFile("gfx", "base", "args");
    EXPECT_TRUE(p.filename() == miopen::make_object_file_name("base"));
}
