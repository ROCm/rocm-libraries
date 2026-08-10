// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// CPU-only unit tests for the AOT catalog engine's measure-and-cache store
// (Phase 2): the problemKey canonicalizer and the TuneCache store/lookup +
// JSON persistence round-trip. No GPU required.

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

#include "catalog/CatalogTypes.hpp"
#include "catalog/TuneCache.hpp"

namespace
{

using namespace aot_catalog_engine::catalog;
namespace fs = std::filesystem;

// A unique temp path for a test's cache file, removed by the fixture teardown.
std::string tempCachePath(const std::string& tag)
{
    return (fs::temp_directory_path() / ("hipdnn_aot_tunecache_test_" + tag + ".json")).string();
}

class TestTuneCache : public ::testing::Test
{
protected:
    void SetUp() override
    {
        _path = tempCachePath(::testing::UnitTest::GetInstance()->current_test_info()->name());
        std::error_code ec;
        fs::remove(_path, ec);
        fs::remove(_path + ".tmp", ec);
    }
    void TearDown() override
    {
        std::error_code ec;
        fs::remove(_path, ec);
        fs::remove(_path + ".tmp", ec);
    }
    std::string _path;
};

} // namespace

// The key is deterministic and independent of the order shape keys are inserted
// (ProblemShape is an ordered map), and renders each variant type predictably.
TEST(TestAotCatalogTuneCacheKey, IsDeterministicAndCanonical)
{
    ProblemShape a;
    a.emplace("dtype", ShapeValue{std::string("f16")});
    a.emplace("M", ShapeValue{static_cast<int64_t>(8)});
    a.emplace("N", ShapeValue{static_cast<int64_t>(2048)});

    // Same content, inserted in a different order.
    ProblemShape b;
    b.emplace("N", ShapeValue{static_cast<int64_t>(2048)});
    b.emplace("dtype", ShapeValue{std::string("f16")});
    b.emplace("M", ShapeValue{static_cast<int64_t>(8)});

    const std::string keyA = problemKey("rmsnorm2d_gfx1151_f16", a);
    const std::string keyB = problemKey("rmsnorm2d_gfx1151_f16", b);

    EXPECT_EQ(keyA, keyB);
    // Keys are ordered by shape-key name: M, N, dtype.
    EXPECT_EQ(keyA, "rmsnorm2d_gfx1151_f16|M=8,N=2048,dtype=f16");

    // A different family (different arch/dtype encoded in the name) never
    // collides even for the same problem.
    EXPECT_NE(keyA, problemKey("rmsnorm2d_gfx1151_bf16", a));
}

TEST(TestAotCatalogTuneCacheKey, RendersEachVariantType)
{
    ProblemShape p;
    p.emplace("causal", ShapeValue{true});
    p.emplace("count", ShapeValue{static_cast<int64_t>(-3)});
    p.emplace("dtype", ShapeValue{std::string("f16")});
    p.emplace("scale", ShapeValue{0.5});

    EXPECT_EQ(problemKey("fam", p), "fam|causal=1,count=-3,dtype=f16,scale=0.5");
}

// store() then lookup() returns the winning symbol; an unknown key misses.
TEST_F(TestTuneCache, StoreThenLookup)
{
    TuneCache cache(_path);
    EXPECT_FALSE(cache.lookup("k1").has_value());

    cache.store("k1", "kernel_fast", 0.42);
    const std::optional<std::string> got = cache.lookup("k1");
    ASSERT_TRUE(got.has_value());
    EXPECT_EQ(*got, "kernel_fast");

    EXPECT_FALSE(cache.lookup("other").has_value());
}

// store() persists to JSON; a fresh cache on the same path reloads the winner.
TEST_F(TestTuneCache, PersistsAndReloads)
{
    {
        TuneCache cache(_path);
        cache.store("k1", "kernel_a", 1.0);
        cache.store("k2", "kernel_b", 2.0);
    }
    ASSERT_TRUE(fs::exists(_path));

    const TuneCache reloaded(_path);
    const std::optional<std::string> a = reloaded.lookup("k1");
    const std::optional<std::string> b = reloaded.lookup("k2");
    ASSERT_TRUE(a.has_value());
    ASSERT_TRUE(b.has_value());
    EXPECT_EQ(*a, "kernel_a");
    EXPECT_EQ(*b, "kernel_b");
}

// Deleting the cache file forces a re-tune: a fresh cache on the now-missing
// path starts empty (this is how tests/users invalidate stale decisions).
TEST_F(TestTuneCache, DeletingFileForcesRetune)
{
    {
        TuneCache cache(_path);
        cache.store("k1", "kernel_a", 1.0);
    }
    ASSERT_TRUE(fs::exists(_path));

    std::error_code ec;
    fs::remove(_path, ec);
    ASSERT_FALSE(fs::exists(_path));

    const TuneCache reloaded(_path);
    EXPECT_FALSE(reloaded.lookup("k1").has_value());
}

// An empty path disables persistence entirely: store/lookup still work in
// memory, and nothing is written to disk.
TEST(TestAotCatalogTuneCacheNoPersist, EmptyPathIsMemoryOnly)
{
    TuneCache cache(std::string{});
    cache.store("k1", "kernel_a", 1.0);
    const std::optional<std::string> got = cache.lookup("k1");
    ASSERT_TRUE(got.has_value());
    EXPECT_EQ(*got, "kernel_a");
    EXPECT_TRUE(cache.path().empty());
}
