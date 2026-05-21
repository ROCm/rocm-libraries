// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <atomic>
#include <chrono>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <memory>
#include <stdexcept>

#include "CkDslContainer.hpp"
#include "python/CompileServiceBridge.hpp"
#include "runtime/HipModule.hpp"
#include "runtime/JitCache.hpp"
#include "runtime/KernelArtifact.hpp"

namespace {

using ck_dsl_provider::CkDslContainer;
using ck_dsl_provider::HipModule;
using ck_dsl_provider::JitCache;
using ck_dsl_provider::KernelArtifact;
using ck_dsl_provider::SignatureHash;

/// I-5 smoke test: prove the JitCache's miss-then-hit shape against
/// a real compile.
///
/// The "second call is sub-millisecond" assertion is the load-bearing
/// signal that the cache is actually preventing recompilation -- the
/// DSL's cold-cache compile is multi-second (comgr_bc dominates), so
/// a cache hit comes in 5+ orders of magnitude faster. We gate the
/// timing budget loosely (10 ms) so a heavily-loaded host doesn't
/// flake the test.
class JitCacheSmoke : public ::testing::Test {
   protected:
    void SetUp() override {
        int deviceCount = 0;
        hipError_t err = hipGetDeviceCount(&deviceCount);
        if (err != hipSuccess || deviceCount == 0) {
            GTEST_SKIP() << "JitCacheSmoke: no HIP-visible device (deviceCount=" << deviceCount
                         << ", hipError=" << static_cast<int>(err) << ")";
        }
        ASSERT_EQ(hipSetDevice(0), hipSuccess);
    }
};

TEST_F(JitCacheSmoke, MissThenHit) {
    CkDslContainer container;
    auto& bridge = container.compileServiceBridge();

    JitCache cache;
    constexpr SignatureHash kKey = 0xC0FFEEULL;

    std::atomic<int> loaderInvocations{0};
    auto loader = [&]() {
        loaderInvocations.fetch_add(1, std::memory_order_relaxed);
        return bridge.compileSmoke();
    };

    auto firstStart = std::chrono::steady_clock::now();
    std::shared_ptr<HipModule> first = cache.getOrLoad(kKey, loader);
    auto firstElapsed = std::chrono::steady_clock::now() - firstStart;

    ASSERT_NE(first, nullptr);
    EXPECT_EQ(loaderInvocations.load(), 1);
    EXPECT_EQ(cache.size(), 1u);
    EXPECT_TRUE(cache.contains(kKey));

    auto secondStart = std::chrono::steady_clock::now();
    std::shared_ptr<HipModule> second = cache.getOrLoad(kKey, loader);
    auto secondElapsed = std::chrono::steady_clock::now() - secondStart;

    EXPECT_EQ(loaderInvocations.load(), 1) << "loader must not be re-invoked on cache hit";
    EXPECT_EQ(second.get(), first.get()) << "cache must return the same HipModule on hit";
    EXPECT_EQ(cache.size(), 1u);

    auto secondUs = std::chrono::duration_cast<std::chrono::microseconds>(secondElapsed).count();
    auto firstMs = std::chrono::duration_cast<std::chrono::milliseconds>(firstElapsed).count();
    EXPECT_LT(secondUs, 10000) << "cache hit took " << secondUs
                               << " us; expected <10 ms (first compile took " << firstMs << " ms)";
}

TEST_F(JitCacheSmoke, DistinctKeysProduceDistinctModules) {
    CkDslContainer container;
    auto& bridge = container.compileServiceBridge();

    JitCache cache;
    auto loader = [&]() { return bridge.compileSmoke(); };

    auto a = cache.getOrLoad(SignatureHash{0xAAAA}, loader);
    auto b = cache.getOrLoad(SignatureHash{0xBBBB}, loader);

    EXPECT_NE(a.get(), b.get());
    EXPECT_EQ(cache.size(), 2u);
    EXPECT_TRUE(cache.contains(SignatureHash{0xAAAA}));
    EXPECT_TRUE(cache.contains(SignatureHash{0xBBBB}));
}

TEST(JitCache, EmptyLoaderRejected) {
    // Independent of GPU: the empty-callable guard runs before any HIP
    // call. Lets the host-only CI lane catch a regression where a
    // caller forgets to bind the loader.
    JitCache cache;
    EXPECT_THROW(cache.getOrLoad(SignatureHash{1}, JitCache::Loader{}),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_EQ(cache.size(), 0u);
}

TEST(JitCache, MissOnEmptyCache) {
    JitCache cache;
    EXPECT_FALSE(cache.contains(SignatureHash{1}));
    EXPECT_EQ(cache.size(), 0u);
}

}  // namespace
