/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
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

// CPU-only unit tests for the per-Handle hipBLASLt GEMM descriptor cache
// (miopen::hipblaslt_gemm_cache). These exercise the shape-keyed lookup logic
// that decides whether the hot path reuses cached descriptors/algo or rebuilds
// them: key equality/hash, find-miss, insert-then-hit, key discrimination on
// every field, and insert idempotency. No GPU or hipBLASLt runtime is needed --
// a default-constructed cache entry holds only null handles, so its destructor
// is a no-op and the whole test runs on host.

#include <miopen/config.h>

#if MIOPEN_USE_HIPBLASLT

#include <miopen/hipblaslt_gemm.hpp>
#include <miopen/hipblaslt_gemm_impl.hpp>

#include <gtest/gtest.h>

#include <memory>
#include <vector>

namespace {

miopen::hipblaslt_gemm_cache_key MakeKey()
{
    // A representative shape (the Kokoro LSTM decode matvec that motivated the
    // cache). Field values are arbitrary but fixed so mutations below are the
    // only thing that changes the key.
    return miopen::hipblaslt_gemm_cache_key{/*transA*/ false,
                                            /*transB*/ true,
                                            /*m*/ 256,
                                            /*n*/ 1,
                                            /*k*/ 512,
                                            /*lda*/ 512,
                                            /*ldb*/ 512,
                                            /*ldc*/ 256,
                                            /*batch_count*/ 1,
                                            /*strideA*/ 0,
                                            /*strideB*/ 0,
                                            /*strideC*/ 0,
                                            /*skip_batches*/ false,
                                            /*type_AB*/ 0,
                                            /*type_C*/ 0};
}

std::unique_ptr<miopen::hipblaslt_gemm_cache_entry> MakeEntry()
{
    // All handles are null -> destructor is a no-op, so no GPU/hipBLASLt state
    // is created. The identity of the pointer is what the tests check.
    return std::make_unique<miopen::hipblaslt_gemm_cache_entry>();
}

} // namespace

// Two keys built from the same fields compare equal and hash equal.
TEST(CPU_HipblasLtGemmCache_NONE, KeyEqualityAndHash)
{
    const auto a = MakeKey();
    const auto b = MakeKey();
    EXPECT_EQ(a, b);

    const miopen::hipblaslt_gemm_cache_key_hash hasher;
    EXPECT_EQ(hasher(a), hasher(b));
}

// Changing any single field must produce a key that is not equal. Equal hashes
// are permitted (collisions are legal) but inequality of the key is required,
// because the cache relies on operator== for correctness.
TEST(CPU_HipblasLtGemmCache_NONE, KeyDiscriminatesEveryField)
{
    const auto base = MakeKey();

    std::vector<miopen::hipblaslt_gemm_cache_key> mutated;
    auto add = [&](auto mutate) {
        auto k = base;
        mutate(k);
        mutated.push_back(k);
    };

    add([](auto& k) { k.transA = !k.transA; });
    add([](auto& k) { k.transB = !k.transB; });
    add([](auto& k) { k.m += 1; });
    add([](auto& k) { k.n += 1; });
    add([](auto& k) { k.k += 1; });
    add([](auto& k) { k.lda += 1; });
    add([](auto& k) { k.ldb += 1; });
    add([](auto& k) { k.ldc += 1; });
    add([](auto& k) { k.batch_count += 1; });
    add([](auto& k) { k.strideA += 1; });
    add([](auto& k) { k.strideB += 1; });
    add([](auto& k) { k.strideC += 1; });
    add([](auto& k) { k.skip_batches = !k.skip_batches; });
    add([](auto& k) { k.type_AB += 1; });
    add([](auto& k) { k.type_C += 1; });

    for(const auto& k : mutated)
    {
        EXPECT_FALSE(k == base) << "a mutated key compared equal to the base key";
    }
}

// find() on an empty cache returns nullptr (forces the hot path to rebuild).
TEST(CPU_HipblasLtGemmCache_NONE, FindMissReturnsNull)
{
    const miopen::hipblaslt_gemm_cache cache;
    EXPECT_EQ(cache.find(MakeKey()), nullptr);
}

// After insert, find() returns the exact entry that was inserted.
TEST(CPU_HipblasLtGemmCache_NONE, InsertThenFindReturnsSameEntry)
{
    miopen::hipblaslt_gemm_cache cache;
    const auto key = MakeKey();

    auto entry                            = MakeEntry();
    miopen::hipblaslt_gemm_cache_entry* p = entry.get();

    miopen::hipblaslt_gemm_cache_entry* inserted = cache.insert(key, std::move(entry));
    EXPECT_EQ(inserted, p) << "insert into an empty slot should return the inserted entry";
    EXPECT_EQ(cache.find(key), p) << "find must return the previously inserted entry";
}

// A second insert with the same key does not replace the resident entry; the
// original pointer is returned and the second entry is dropped. This matches
// the concurrent find/insert race the cache is designed to tolerate.
TEST(CPU_HipblasLtGemmCache_NONE, InsertIsIdempotentForSameKey)
{
    miopen::hipblaslt_gemm_cache cache;
    const auto key = MakeKey();

    auto first                                  = MakeEntry();
    miopen::hipblaslt_gemm_cache_entry* first_p = first.get();
    ASSERT_EQ(cache.insert(key, std::move(first)), first_p);

    auto second                                  = MakeEntry();
    miopen::hipblaslt_gemm_cache_entry* second_p = second.get();
    miopen::hipblaslt_gemm_cache_entry* resident = cache.insert(key, std::move(second));

    EXPECT_EQ(resident, first_p) << "second insert must keep the first entry";
    EXPECT_NE(resident, second_p) << "second entry must not become resident";
    EXPECT_EQ(cache.find(key), first_p);
}

// Distinct shapes coexist without aliasing: each key returns its own entry.
TEST(CPU_HipblasLtGemmCache_NONE, DistinctKeysDoNotAlias)
{
    miopen::hipblaslt_gemm_cache cache;

    auto key1 = MakeKey();
    auto key2 = MakeKey();
    key2.n += 1; // different shape

    auto e1                                = MakeEntry();
    miopen::hipblaslt_gemm_cache_entry* p1 = e1.get();
    auto e2                                = MakeEntry();
    miopen::hipblaslt_gemm_cache_entry* p2 = e2.get();

    ASSERT_EQ(cache.insert(key1, std::move(e1)), p1);
    ASSERT_EQ(cache.insert(key2, std::move(e2)), p2);

    EXPECT_EQ(cache.find(key1), p1);
    EXPECT_EQ(cache.find(key2), p2);
    EXPECT_NE(cache.find(key1), cache.find(key2));
}

#endif // MIOPEN_USE_HIPBLASLT
