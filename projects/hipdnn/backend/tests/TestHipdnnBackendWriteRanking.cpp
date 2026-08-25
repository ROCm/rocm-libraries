// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file TestHipdnnBackendWriteRanking.cpp
 * @brief Contract tests for hipdnnBackendWriteEngineRankingResults_ext, the write half of
 *        the exact-match autotune cache.
 */

#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/GraphTestUtils.hpp"
#include "handle/Handle.hpp"
#include "heuristics/DeviceProperties.hpp"
#include "heuristics/config/AutotuneCacheKey.hpp"
#include "heuristics/config/AutotuneRankingStore.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <system_error>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::heuristics::config;

namespace
{

constexpr const char* DISABLE_ENV = "HIPDNN_DISABLE_EXACT_ENGINE_CACHE";

// finalize() stores the handle attribute without dereferencing it, so an opaque fake
// pointer works for every case that declines before reaching queryDeviceProperties().
std::unique_ptr<HipdnnBackendDescriptor> makeFinalizedGraph()
{
    auto bundle = test_utilities::createDefaultConvOp();
    auto wrapper = test_utilities::createDescriptor<GraphDescriptor>();
    auto graphDesc = wrapper->asDescriptor<GraphDescriptor>();

    const std::array<HipdnnBackendDescriptor*, 1> ops = {bundle.convOp.get()};
    graphDesc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                            HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                            1,
                            static_cast<const void*>(ops.data()));

    auto fakeHandle = reinterpret_cast<hipdnnHandle_t>(0x12345678);
    graphDesc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                            HIPDNN_TYPE_HANDLE,
                            1,
                            static_cast<const void*>(&fakeHandle));
    graphDesc->finalize();

    static std::vector<test_utilities::ConvOpBundle> s_bundles;
    s_bundles.push_back(std::move(bundle));

    return wrapper;
}

} // namespace

class TestHipdnnBackendWriteRanking : public ::testing::Test
{
protected:
    void SetUp() override
    {
        _disableGuard
            = std::make_unique<hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter>(
                DISABLE_ENV);

        static std::atomic<uint64_t> s_cacheDirCounter{0};
        const uint64_t cacheDirId = s_cacheDirCounter.fetch_add(1);
        _cacheDir = std::filesystem::temp_directory_path()
                    / ("hipdnn_test_writeranking_cache_"
                       + std::to_string(::testing::UnitTest::GetInstance()->random_seed()) + "_"
                       + std::to_string(cacheDirId));
        _cacheDirGuard
            = std::make_unique<hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter>(
                "HIPDNN_CACHE_DIR", _cacheDir.string());
    }

    void TearDown() override
    {
        _cacheDirGuard.reset();
        std::error_code ignored;
        std::filesystem::remove_all(_cacheDir, ignored);
    }

    std::unique_ptr<hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter> _disableGuard;
    std::unique_ptr<hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter> _cacheDirGuard;
    std::filesystem::path _cacheDir;
};

TEST_F(TestHipdnnBackendWriteRanking, NullHandleReturnsBadParamNullPointer)
{
    auto graph = makeFinalizedGraph();
    const std::array<int64_t, 2> order = {1, 2};

    EXPECT_EQ(hipdnnBackendWriteEngineRankingResults_ext(
                  nullptr, graph.get(), order.data(), order.size(), nullptr),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestHipdnnBackendWriteRanking, UnfinalizedDescriptorDeclinesWithSuccess)
{
    auto fakeHandle = reinterpret_cast<hipdnnHandle_t>(0x12345678);
    auto wrapper = test_utilities::createDescriptor<GraphDescriptor>();
    const std::array<int64_t, 2> order = {1, 2};
    auto outcome = HIPDNN_AUTOTUNE_CACHE_WRITE_WRITTEN;

    EXPECT_EQ(hipdnnBackendWriteEngineRankingResults_ext(
                  fakeHandle, wrapper.get(), order.data(), order.size(), &outcome),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(outcome, HIPDNN_AUTOTUNE_CACHE_WRITE_DECLINED_UNKEYABLE_OR_UNFINALIZED);
}

TEST_F(TestHipdnnBackendWriteRanking, NullOrderDeclinesWithSuccess)
{
    auto fakeHandle = reinterpret_cast<hipdnnHandle_t>(0x12345678);
    auto graph = makeFinalizedGraph();
    auto outcome = HIPDNN_AUTOTUNE_CACHE_WRITE_WRITTEN;

    EXPECT_EQ(
        hipdnnBackendWriteEngineRankingResults_ext(fakeHandle, graph.get(), nullptr, 0, &outcome),
        HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(outcome, HIPDNN_AUTOTUNE_CACHE_WRITE_DECLINED_NO_ENGINES);
}

TEST_F(TestHipdnnBackendWriteRanking, EmptyOrderDeclinesWithSuccess)
{
    auto fakeHandle = reinterpret_cast<hipdnnHandle_t>(0x12345678);
    auto graph = makeFinalizedGraph();
    const std::array<int64_t, 1> order = {1};
    auto outcome = HIPDNN_AUTOTUNE_CACHE_WRITE_WRITTEN;

    EXPECT_EQ(hipdnnBackendWriteEngineRankingResults_ext(
                  fakeHandle, graph.get(), order.data(), 0, &outcome),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(outcome, HIPDNN_AUTOTUNE_CACHE_WRITE_DECLINED_NO_ENGINES);
}

TEST_F(TestHipdnnBackendWriteRanking, DisabledCacheDeclinesWithSuccess)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter enableDisable(DISABLE_ENV,
                                                                                    "1");

    auto fakeHandle = reinterpret_cast<hipdnnHandle_t>(0x12345678);
    auto graph = makeFinalizedGraph();
    const std::array<int64_t, 2> order = {5, 6};
    auto outcome = HIPDNN_AUTOTUNE_CACHE_WRITE_WRITTEN;

    EXPECT_EQ(hipdnnBackendWriteEngineRankingResults_ext(
                  fakeHandle, graph.get(), order.data(), order.size(), &outcome),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(outcome, HIPDNN_AUTOTUNE_CACHE_WRITE_DECLINED_DISABLED);
}

TEST_F(TestHipdnnBackendWriteRanking, NullOutcomePointerIsAccepted)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter enableDisable(DISABLE_ENV,
                                                                                    "1");
    auto fakeHandle = reinterpret_cast<hipdnnHandle_t>(0x12345678);
    auto graph = makeFinalizedGraph();
    const std::array<int64_t, 2> order = {7, 8};

    EXPECT_EQ(hipdnnBackendWriteEngineRankingResults_ext(
                  fakeHandle, graph.get(), order.data(), order.size(), nullptr),
              HIPDNN_STATUS_SUCCESS);
}

class TestGpuHipdnnBackendWriteRanking : public TestHipdnnBackendWriteRanking
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        TestHipdnnBackendWriteRanking::SetUp();
        ASSERT_EQ(hipStreamCreate(&_stream), hipSuccess);
        _handle.setStream(_stream);
    }

    void TearDown() override
    {
        if(_stream != nullptr)
        {
            EXPECT_EQ(hipStreamDestroy(_stream), hipSuccess);
            _stream = nullptr;
        }
        TestHipdnnBackendWriteRanking::TearDown();
    }

    hipdnnHandle _handle;
    hipStream_t _stream = nullptr;
};
TEST_F(TestGpuHipdnnBackendWriteRanking, DisabledCacheLeavesStoreUntouched)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter enableDisable(DISABLE_ENV,
                                                                                    "1");

    auto graph = makeFinalizedGraph();
    const std::array<int64_t, 2> order = {5, 6};
    auto outcome = HIPDNN_AUTOTUNE_CACHE_WRITE_WRITTEN;

    EXPECT_EQ(hipdnnBackendWriteEngineRankingResults_ext(
                  &_handle, graph.get(), order.data(), order.size(), &outcome),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(outcome, HIPDNN_AUTOTUNE_CACHE_WRITE_DECLINED_DISABLED);

    auto graphDesc = graph->asDescriptor<GraphDescriptor>();
    graphDesc->buildSerializedGraph();
    const auto serializedGraph = graphDesc->getSerializedGraph();

    const auto devProps = hipdnn_backend::heuristics::queryDeviceProperties(&_handle);
    const auto devicePropsSerialized
        = hipdnn_backend::heuristics::serializeDeviceProperties(devProps);
    const hipdnnPluginConstData_t devicePropsWrapper
        = hipdnn_backend::heuristics::wrapSerializedDeviceProperties(devicePropsSerialized);

    const auto cacheKey = deriveCacheKey(serializedGraph, devicePropsWrapper);
    ASSERT_TRUE(cacheKey.has_value());

    EXPECT_FALSE(exactCacheStore().get(*cacheKey, {}).has_value());
}

TEST_F(TestGpuHipdnnBackendWriteRanking, EnabledCacheReportsWritten)
{
    auto graph = makeFinalizedGraph();
    const std::array<int64_t, 2> order = {5, 6};
    auto outcome = HIPDNN_AUTOTUNE_CACHE_WRITE_DECLINED_NO_ENGINES;

    EXPECT_EQ(hipdnnBackendWriteEngineRankingResults_ext(
                  &_handle, graph.get(), order.data(), order.size(), &outcome),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(outcome, HIPDNN_AUTOTUNE_CACHE_WRITE_WRITTEN);
}

/// A second write of a ranking already on disk must report UNCHANGED, not WRITTEN.
///
/// The outcome enum exists so a caller can tell whether its measurement reached the cache;
/// reporting WRITTEN having written nothing defeats the only reason the parameter is there.
///
/// Falsifying mutation: assign HIPDNN_AUTOTUNE_CACHE_WRITE_WRITTEN unconditionally in
/// hipdnnBackendWriteEngineRankingResults_ext instead of mapping the store's status.
TEST_F(TestGpuHipdnnBackendWriteRanking, RewritingAnIdenticalRankingReportsUnchanged)
{
    auto graph = makeFinalizedGraph();
    const std::array<int64_t, 2> order = {5, 6};

    auto outcome = HIPDNN_AUTOTUNE_CACHE_WRITE_DECLINED_NO_ENGINES;
    ASSERT_EQ(hipdnnBackendWriteEngineRankingResults_ext(
                  &_handle, graph.get(), order.data(), order.size(), &outcome),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(outcome, HIPDNN_AUTOTUNE_CACHE_WRITE_WRITTEN);

    outcome = HIPDNN_AUTOTUNE_CACHE_WRITE_DECLINED_NO_ENGINES;
    EXPECT_EQ(hipdnnBackendWriteEngineRankingResults_ext(
                  &_handle, graph.get(), order.data(), order.size(), &outcome),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(outcome, HIPDNN_AUTOTUNE_CACHE_WRITE_UNCHANGED);
}

/// A ranking that differs from the stored one supersedes it and reports WRITTEN.
///
/// Falsifying mutation: restore put()'s early return whenever a record for the key exists --
/// the second call then reports UNCHANGED and the stored order stays {5, 6}.
TEST_F(TestGpuHipdnnBackendWriteRanking, RewritingADifferentRankingReportsWrittenAndSupersedes)
{
    auto graph = makeFinalizedGraph();
    const std::array<int64_t, 2> first = {5, 6};
    const std::array<int64_t, 3> second = {7, 5, 6};

    auto outcome = HIPDNN_AUTOTUNE_CACHE_WRITE_DECLINED_NO_ENGINES;
    ASSERT_EQ(hipdnnBackendWriteEngineRankingResults_ext(
                  &_handle, graph.get(), first.data(), first.size(), &outcome),
              HIPDNN_STATUS_SUCCESS);

    outcome = HIPDNN_AUTOTUNE_CACHE_WRITE_DECLINED_NO_ENGINES;
    EXPECT_EQ(hipdnnBackendWriteEngineRankingResults_ext(
                  &_handle, graph.get(), second.data(), second.size(), &outcome),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(outcome, HIPDNN_AUTOTUNE_CACHE_WRITE_WRITTEN);
    auto graphDesc = graph->asDescriptor<GraphDescriptor>();
    graphDesc->buildSerializedGraph();
    const auto serializedGraph = graphDesc->getSerializedGraph();

    const auto devProps = hipdnn_backend::heuristics::queryDeviceProperties(&_handle);
    const auto devicePropsSerialized
        = hipdnn_backend::heuristics::serializeDeviceProperties(devProps);
    const hipdnnPluginConstData_t devicePropsWrapper
        = hipdnn_backend::heuristics::wrapSerializedDeviceProperties(devicePropsSerialized);

    const auto cacheKey = deriveCacheKey(serializedGraph, devicePropsWrapper);
    ASSERT_TRUE(cacheKey.has_value());
    const auto entry = exactCacheStore().get(*cacheKey, {});
    ASSERT_TRUE(entry.has_value());
    EXPECT_EQ(entry->order, (std::vector<int64_t>{7, 5, 6}));
}

// The file-backed store makes cross-.so process-local statics irrelevant: both this
// binary and libhipdnn_backend.so read and write the same shard file on disk.
TEST_F(TestGpuHipdnnBackendWriteRanking, WrittenRankingIsVisibleThroughTheCExport)
{
    auto graph = makeFinalizedGraph();
    const std::array<int64_t, 3> order = {11, 22, 33};
    auto outcome = HIPDNN_AUTOTUNE_CACHE_WRITE_DECLINED_NO_ENGINES;

    ASSERT_EQ(hipdnnBackendWriteEngineRankingResults_ext(
                  &_handle, graph.get(), order.data(), order.size(), &outcome),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(outcome, HIPDNN_AUTOTUNE_CACHE_WRITE_WRITTEN);

    auto graphDesc = graph->asDescriptor<GraphDescriptor>();
    graphDesc->buildSerializedGraph();
    const auto serializedGraph = graphDesc->getSerializedGraph();

    const auto devProps = hipdnn_backend::heuristics::queryDeviceProperties(&_handle);
    const auto devicePropsSerialized
        = hipdnn_backend::heuristics::serializeDeviceProperties(devProps);
    const hipdnnPluginConstData_t devicePropsWrapper
        = hipdnn_backend::heuristics::wrapSerializedDeviceProperties(devicePropsSerialized);

    const auto cacheKey = deriveCacheKey(serializedGraph, devicePropsWrapper);
    ASSERT_TRUE(cacheKey.has_value());

    RankingLookupStatus status = RankingLookupStatus::MISS;
    const auto entry = exactCacheStore().get(*cacheKey, {}, &status);
    ASSERT_TRUE(entry.has_value());
    EXPECT_EQ(status, RankingLookupStatus::HIT);
    EXPECT_EQ(entry->sampledEngineIds, (std::vector<int64_t>{11, 22, 33}));
    EXPECT_EQ(entry->order, (std::vector<int64_t>{11, 22, 33}));
}
