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
 * The resolver is a process-lifetime static (see KernelIngestorEngine.cpp's
 * deviceResolver()), so a stale or invalidated cache entry is a process-lifetime bug,
 * not one scoped to a single engine or container.
 *
 * deviceId() has four paths; the fourth (no usable HIP context) has no test hook here
 * and is documented but not exercised.
 */
namespace
{

using hip_kernel_provider::kernel_ingestor_engine::HandleDeviceResolver;

/// Answers every property query successfully, so a test can grow the cache past a
/// rehash without needing many physical devices.
class FakeQueryResolver : public HandleDeviceResolver
{
public:
    /// warpSize is set from the id so a caller can tell one faked entry from another.
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

// ---------------------------------------------------------------------------
// deviceId()
// ---------------------------------------------------------------------------

TEST(TestHandleDeviceResolver, ResolvesTheCurrentDeviceForANullStream)
{
    SKIP_IF_NO_DEVICES();

    // A null stream means the default stream, which belongs to the current device.
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

    // A stream hipStreamGetDevice cannot resolve (already destroyed) falls through like
    // a null stream. Destroying it leaves a pending HIP error, cleared below so it does
    // not fail an unrelated test.
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

// ---------------------------------------------------------------------------
// deviceProperties(): cache hit vs miss, and the growth-safety invariant
// ---------------------------------------------------------------------------

TEST(TestHandleDeviceResolver, CachesDevicePropertiesAcrossCalls)
{
    SKIP_IF_NO_DEVICES();

    const HandleDeviceResolver resolver;

    // A hit for the same device returns the same address: deviceProperties() promises a
    // reference stable for the resolver's lifetime.
    const auto& first = resolver.deviceProperties(0);
    const auto& second = resolver.deviceProperties(0);

    EXPECT_EQ(&first, &second);
    EXPECT_EQ(first.warpSize, second.warpSize);
}

TEST(TestHandleDeviceResolver, ReferencesStayValidAcrossCacheGrowth)
{
    // Verifies HandleDeviceResolver's stated invariant: references stay valid across
    // cache growth (std::unordered_map keeps node handles stable across rehash).
    //
    // Uses FakeQueryResolver rather than invalid device ids, since deviceProperties()
    // does not cache failed queries. CPU-only.
    const FakeQueryResolver resolver;

    const auto& firstInserted = resolver.deviceProperties(1000);

    // Enough insertions to force a rehash of the default bucket count.
    for(int deviceId = 1001; deviceId < 1064; ++deviceId)
    {
        static_cast<void>(resolver.deviceProperties(deviceId));
    }

    const auto& sameEntryAfterGrowth = resolver.deviceProperties(1000);
    EXPECT_EQ(&firstInserted, &sameEntryAfterGrowth);
    // Value must match too: a surviving address alone is not sufficient proof.
    EXPECT_EQ(sameEntryAfterGrowth.warpSize, 1000);
}

TEST(TestHandleDeviceResolver, RefusesAndDoesNotCacheAFailedPropertyQuery)
{
    // A zeroed hipDeviceProp_t is not cached: this cache is never invalidated, so a
    // false answer would persist for the process's life.
    const FailingQueryResolver resolver;

    EXPECT_THROW(static_cast<void>(resolver.deviceProperties(7)),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    // Still refused on the second ask: the failure was not remembered as an answer.
    EXPECT_THROW(static_cast<void>(resolver.deviceProperties(7)),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestHandleDeviceResolver, ConcurrentDevicePropertyLookupsAreSafe)
{
    // The cache is mutex-guarded (see class doc); many threads querying the same small
    // id set maximize a real race's chance to corrupt the map.
    //
    // Every result is checked rather than discarded. FakeQueryResolver encodes the
    // device id in warpSize precisely so a torn or cross-wired entry is observable, and
    // discarding the value left the test unable to fail on anything short of a crash.
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
