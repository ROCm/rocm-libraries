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
#endif

TEST(CPU_Cache_NONE, check_cache_file)
{
    auto p = miopen::GetCacheFile("gfx", "base", "args");
    EXPECT_TRUE(p.filename() == miopen::make_object_file_name("base"));
}
