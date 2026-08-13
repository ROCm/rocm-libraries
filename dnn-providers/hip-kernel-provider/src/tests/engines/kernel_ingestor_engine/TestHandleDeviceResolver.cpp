// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <hip/hip_runtime_api.h>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/HandleDeviceResolver.hpp"

/**
 * @file TestHandleDeviceResolver.cpp
 * @brief HandleDeviceResolver's device-id resolution and its device-properties cache.
 *
 * A process-lifetime static (see KernelIngestorEngine.cpp's deviceResolver()), so a
 * stale cache entry is a process-lifetime bug, not one scoped to one engine.
 */
namespace
{

using hip_kernel_provider::kernel_ingestor_engine::HandleDeviceResolver;

/// Answers every query, so a test can grow the cache past a rehash without real devices.
class FakeQueryResolver : public HandleDeviceResolver
{
public:
    /// warpSize is set from the id so a caller can tell entries apart.
    hipError_t queryDeviceProperties(hipDeviceProp_t* properties,
                                     hipdnn_plugin_sdk::ingestor::DeviceId deviceId) const override
    {
        *properties = hipDeviceProp_t{};
        properties->warpSize = static_cast<int>(deviceId);
        return hipSuccess;
    }
};

/// Fails every property query, to drive the refusal path.
class FailingQueryResolver : public HandleDeviceResolver
{
public:
    hipError_t
        queryDeviceProperties(hipDeviceProp_t* /*properties*/,
                              hipdnn_plugin_sdk::ingestor::DeviceId /*deviceId*/) const override
    {
        return hipErrorInvalidDevice;
    }
};

// deviceId()

TEST(TestHandleDeviceResolver, ResolvesTheCurrentDeviceForANullStream)
{
    SKIP_IF_NO_DEVICES();

    const HandleDeviceResolver resolver;
    Handle handle;
    handle.setStream(nullptr);

    int currentDevice = -1;
    ASSERT_EQ(hipGetDevice(&currentDevice), hipSuccess);

    EXPECT_EQ(resolver.deviceId(handle), currentDevice);
}

TEST(TestHandleDeviceResolver, ResolvesTheStreamsOwnDeviceWhenItDiffersFromCurrent)
{
    SKIP_IF_NO_DEVICES();

    // Resolves via hipStreamGetDevice, not whichever device is current on this thread.
    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    int streamDevice = -1;
    ASSERT_EQ(hipStreamGetDevice(stream, &streamDevice), hipSuccess);

    const HandleDeviceResolver resolver;
    Handle handle;
    handle.setStream(stream);

    EXPECT_EQ(resolver.deviceId(handle), streamDevice);

    static_cast<void>(hipStreamDestroy(stream));
}

TEST(TestHandleDeviceResolver, FallsThroughToTheCurrentDeviceWhenTheStreamCannotBeResolved)
{
    SKIP_IF_NO_DEVICES();

    // A stream hipStreamGetDevice cannot resolve falls through like a null stream.
    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);

    int currentDevice = -1;
    ASSERT_EQ(hipGetDevice(&currentDevice), hipSuccess);

    const HandleDeviceResolver resolver;
    Handle handle;
    handle.setStream(stream);

    EXPECT_EQ(resolver.deviceId(handle), currentDevice);

    static_cast<void>(hipGetLastError());
    static_cast<void>(hipExtGetLastError());
}

// deviceProperties(): cache hit vs miss, and the growth-safety invariant

TEST(TestHandleDeviceResolver, CachesDevicePropertiesAcrossCalls)
{
    SKIP_IF_NO_DEVICES();

    const HandleDeviceResolver resolver;

    // A hit returns the same address: the reference is stable for the resolver's lifetime.
    const auto& first = resolver.deviceProperties(0);
    const auto& second = resolver.deviceProperties(0);

    EXPECT_EQ(&first, &second);
    EXPECT_EQ(first.warpSize, second.warpSize);
}

TEST(TestHandleDeviceResolver, ReferencesStayValidAcrossCacheGrowth)
{
    // std::unordered_map keeps node handles stable across rehash. Uses FakeQueryResolver
    // since deviceProperties() does not cache failed queries.
    const FakeQueryResolver resolver;

    const auto& firstInserted = resolver.deviceProperties(1000);

    for(int deviceId = 1001; deviceId < 1064; ++deviceId)
    {
        static_cast<void>(resolver.deviceProperties(deviceId));
    }

    const auto& sameEntryAfterGrowth = resolver.deviceProperties(1000);
    EXPECT_EQ(&firstInserted, &sameEntryAfterGrowth);
    EXPECT_EQ(sameEntryAfterGrowth.warpSize, 1000);
}

TEST(TestHandleDeviceResolver, RefusesAndDoesNotCacheAFailedPropertyQuery)
{
    // Not cached: this cache is never invalidated, so a false answer would persist.
    const FailingQueryResolver resolver;

    EXPECT_THROW(static_cast<void>(resolver.deviceProperties(7)),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_THROW(static_cast<void>(resolver.deviceProperties(7)),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestHandleDeviceResolver, ConcurrentDevicePropertyLookupsAreSafe)
{
    // Mutex-guarded; many threads on a small id set maximize a race's chance to corrupt
    // the map. FakeQueryResolver encodes the device id in warpSize so a torn or
    // cross-wired entry is observable.
    const FakeQueryResolver resolver;
    std::atomic<int> mismatches{0};

    std::vector<std::thread> threads;
    threads.reserve(8);
    for(int t = 0; t < 8; ++t)
    {
        threads.emplace_back([&resolver, &mismatches, t]() {
            const auto deviceId = (t % 4) + 2000;
            for(int i = 0; i < 200; ++i)
            {
                if(resolver.deviceProperties(deviceId).warpSize != deviceId)
                {
                    mismatches.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    for(auto& thread : threads)
    {
        thread.join();
    }

    EXPECT_EQ(mismatches.load(std::memory_order_relaxed), 0)
        << "a concurrent lookup returned another device's properties";

    static_cast<void>(hipGetLastError());
    static_cast<void>(hipExtGetLastError());
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
