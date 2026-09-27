// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
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

hipDeviceProp_t validHipProperties()
{
    hipDeviceProp_t properties{};
    constexpr const char* ARCH = "gfx000";
    std::memcpy(properties.gcnArchName, ARCH, std::strlen(ARCH) + 1);
    properties.warpSize = 64;
    properties.multiProcessorCount = 48;
    properties.sharedMemPerBlock = 65536;
    return properties;
}

/// Supplies test device properties and counts queries without a GPU.
class FakeQueryResolver : public HandleDeviceResolver
{
public:
    hipDeviceProp_t properties = validHipProperties();
    hipError_t status = hipSuccess;
    bool distinguishDevices = false;
    bool omitLds = false;
    mutable std::atomic<int> queryCount{0};

    hipError_t queryDeviceProperties(hipDeviceProp_t* result,
                                     hipdnn_plugin_sdk::ingestor::DeviceId deviceId) const override
    {
        queryCount.fetch_add(1, std::memory_order_relaxed);
        if(status != hipSuccess)
        {
            return status;
        }

        const auto unwrittenLds = result->sharedMemPerBlock;
        *result = properties;
        if(omitLds)
        {
            result->sharedMemPerBlock = unwrittenLds;
        }
        else if(distinguishDevices)
        {
            result->sharedMemPerBlock += static_cast<size_t>(deviceId);
        }
        return hipSuccess;
    }
};

/// Distinguishes a concrete stream's owner from the caller's changing current device.
class ChangingCurrentDeviceResolver : public HandleDeviceResolver
{
public:
    static constexpr int STREAM_DEVICE = 7;
    int currentDevice = 3;

    hipError_t queryStreamDevice(hipStream_t /*stream*/, int* deviceId) const override
    {
        *deviceId = STREAM_DEVICE;
        return hipSuccess;
    }

    hipError_t queryCurrentDevice(int* deviceId) const override
    {
        *deviceId = currentDevice;
        return hipSuccess;
    }
};

/// Answers the fallthrough with an ordinal no single-device machine reports.
class UnwrittenStreamDeviceResolver : public HandleDeviceResolver
{
public:
    static constexpr int FALLTHROUGH_DEVICE = 3;

    hipError_t queryStreamDevice(hipStream_t /*stream*/, int* /*deviceId*/) const override
    {
        return hipSuccess;
    }

    hipError_t queryCurrentDevice(int* deviceId) const override
    {
        *deviceId = FALLTHROUGH_DEVICE;
        return hipSuccess;
    }
};

/// Reports success from the stream query alongside an ordinal no device can have.
class NegativeStreamDeviceResolver : public HandleDeviceResolver
{
public:
    static constexpr int BOGUS_DEVICE = -42;

    hipError_t queryStreamDevice(hipStream_t /*stream*/, int* deviceId) const override
    {
        *deviceId = BOGUS_DEVICE;
        return hipSuccess;
    }
};

/// What deviceId() must fall through to once a stream ordinal is rejected.
hipdnn_plugin_sdk::ingestor::DeviceId currentDeviceOrNone()
{
    int currentDevice = -1;
    if(hipGetDevice(&currentDevice) != hipSuccess)
    {
        return hipdnn_plugin_sdk::ingestor::NO_DEVICE;
    }
    return currentDevice;
}

/// A stream the overridden seam never dereferences; only its non-null-ness is read.
/// Backed by a real object rather than a literal address so the cast stays pointer-to-pointer.
hipStream_t unusedStream()
{
    static int s_placeholder = 0;
    return reinterpret_cast<hipStream_t>(&s_placeholder);
}

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

TEST(TestHandleDeviceResolver, ResolvesDefaultStreamsFromTheLiveCurrentDevice)
{
    ChangingCurrentDeviceResolver resolver;
    Handle handle;
    const std::array<hipStream_t, 3> defaultStreams
        = {nullptr, hipStreamLegacy, hipStreamPerThread};

    for(const auto stream : defaultStreams)
    {
        SCOPED_TRACE(stream);
        handle.setStream(stream);

        resolver.currentDevice = 3;
        EXPECT_EQ(resolver.deviceId(handle), 3);

        resolver.currentDevice = 5;
        EXPECT_EQ(resolver.deviceId(handle), 5);
    }
}

TEST(TestHandleDeviceResolver, KeepsTheConcreteStreamOwnerAcrossCurrentDeviceChanges)
{
    ChangingCurrentDeviceResolver resolver;
    Handle handle;
    handle.setStream(unusedStream());

    EXPECT_EQ(resolver.deviceId(handle), ChangingCurrentDeviceResolver::STREAM_DEVICE);

    resolver.currentDevice = 5;
    EXPECT_EQ(resolver.deviceId(handle), ChangingCurrentDeviceResolver::STREAM_DEVICE);
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

TEST(TestHandleDeviceResolver, RejectsAStreamOrdinalTheRuntimeNeverWrote)
{
    // hipSuccess with the out-parameter untouched must not read as device 0: the seed has
    // to be out of range so the guard rejects it and the fallthrough ordinal comes back.
    const UnwrittenStreamDeviceResolver resolver;
    Handle handle;
    handle.setStream(unusedStream());

    EXPECT_EQ(resolver.deviceId(handle), UnwrittenStreamDeviceResolver::FALLTHROUGH_DEVICE);
}

TEST(TestHandleDeviceResolver, RejectsANegativeStreamOrdinalReportedAsSuccess)
{
    // A negative ordinal is never a device, whatever status came back with it.
    const NegativeStreamDeviceResolver resolver;
    Handle handle;
    handle.setStream(unusedStream());

    EXPECT_NE(resolver.deviceId(handle), NegativeStreamDeviceResolver::BOGUS_DEVICE);
    EXPECT_EQ(resolver.deviceId(handle), currentDeviceOrNone());

    static_cast<void>(hipGetLastError());
    static_cast<void>(hipExtGetLastError());
}

// deviceProperties(): cache hit vs miss, and the growth-safety invariant

TEST(TestGpuHandleDeviceResolver, CachesCompletePropertiesFromTheCurrentDevice)
{
    SKIP_IF_NO_DEVICES();

    const HandleDeviceResolver resolver;
    const Handle handle;
    const auto deviceId = resolver.deviceId(handle);
    ASSERT_NE(deviceId, hipdnn_plugin_sdk::ingestor::NO_DEVICE);

    hipDeviceProp_t reported{};
    ASSERT_EQ(hipGetDeviceProperties(&reported, deviceId), hipSuccess);
    const auto& first = resolver.deviceProperties(deviceId);
    const auto& second = resolver.deviceProperties(deviceId);

    EXPECT_EQ(&first, &second);
    EXPECT_EQ(first.gcnArchName, reported.gcnArchName);
    EXPECT_EQ(first.warpSize, reported.warpSize);
    EXPECT_EQ(first.multiProcessorCount, reported.multiProcessorCount);
    ASSERT_LE(reported.sharedMemPerBlock,
              static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
    EXPECT_EQ(first.ldsSize, static_cast<int64_t>(reported.sharedMemPerBlock));
}

TEST(TestHandleDeviceResolver, CachesSuccessfulQueriesWithoutRepublishingProperties)
{
    FakeQueryResolver resolver;
    const auto& first = resolver.deviceProperties(7);

    // A cache hit must not query HIP again.
    resolver.status = hipErrorInvalidDevice;
    resolver.properties.sharedMemPerBlock = 32768;
    const auto& second = resolver.deviceProperties(7);

    EXPECT_EQ(&first, &second);
    EXPECT_EQ(first.ldsSize, 65536);
    EXPECT_EQ(resolver.queryCount.load(), 1);
}

TEST(TestHandleDeviceResolver, ReferencesStayValidAcrossCacheGrowth)
{
    FakeQueryResolver resolver;
    resolver.distinguishDevices = true;
    const auto& firstInserted = resolver.deviceProperties(1000);

    for(int deviceId = 1001; deviceId < 1064; ++deviceId)
    {
        static_cast<void>(resolver.deviceProperties(deviceId));
    }

    const auto& sameEntryAfterGrowth = resolver.deviceProperties(1000);
    EXPECT_EQ(&firstInserted, &sameEntryAfterGrowth);
    EXPECT_EQ(firstInserted.ldsSize, 65536 + 1000);
    EXPECT_EQ(resolver.queryCount.load(), 64);
}

TEST(TestHandleDeviceResolver, RetriesFailedQueriesAndCachesTheFirstSuccess)
{
    FakeQueryResolver resolver;
    resolver.status = hipErrorInvalidDevice;
    EXPECT_THROW(static_cast<void>(resolver.deviceProperties(7)),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_THROW(static_cast<void>(resolver.deviceProperties(7)),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_EQ(resolver.queryCount.load(), 2);

    resolver.status = hipSuccess;
    const auto& recovered = resolver.deviceProperties(7);
    EXPECT_EQ(recovered.ldsSize, 65536);
    EXPECT_EQ(&resolver.deviceProperties(7), &recovered);
    EXPECT_EQ(resolver.queryCount.load(), 3);
}

TEST(TestHandleDeviceResolver, RejectsInvalidDeviceFactsAndRetries)
{
    struct InvalidFacts
    {
        const char* name;
        const char* arch;
        int warpSize;
        int multiProcessorCount;
        uint64_t ldsSize;
    };
    const std::array<InvalidFacts, 6> cases = {{
        {"missing identity", "", 64, 48, 65536},
        {"zero warp", "gfx000", 0, 48, 65536},
        {"negative warp", "gfx000", -1, 48, 65536},
        {"zero count", "gfx000", 64, 0, 65536},
        {"negative count", "gfx000", 64, -1, 65536},
        {"capacity outside signed Int64", "gfx000", 64, 48, uint64_t{1} << 63},
    }};
    for(const auto& invalid : cases)
    {
        SCOPED_TRACE(invalid.name);
        FakeQueryResolver resolver;
        std::memcpy(resolver.properties.gcnArchName, invalid.arch, std::strlen(invalid.arch) + 1);
        resolver.properties.warpSize = invalid.warpSize;
        resolver.properties.multiProcessorCount = invalid.multiProcessorCount;
        resolver.properties.sharedMemPerBlock = invalid.ldsSize;

        EXPECT_THROW(static_cast<void>(resolver.deviceProperties(7)),
                     hipdnn_plugin_sdk::HipdnnPluginException);
        EXPECT_THROW(static_cast<void>(resolver.deviceProperties(7)),
                     hipdnn_plugin_sdk::HipdnnPluginException);
        EXPECT_EQ(resolver.queryCount.load(), 2);

        resolver.properties = validHipProperties();
        const auto& recovered = resolver.deviceProperties(7);
        EXPECT_EQ(recovered.ldsSize, 65536);
        EXPECT_EQ(&resolver.deviceProperties(7), &recovered);
        EXPECT_EQ(resolver.queryCount.load(), 3);
    }
}

TEST(TestHandleDeviceResolver, RejectsUnterminatedIdentityAndRetries)
{
    FakeQueryResolver resolver;
    std::memset(resolver.properties.gcnArchName, 'x', sizeof(resolver.properties.gcnArchName));
    EXPECT_THROW(static_cast<void>(resolver.deviceProperties(7)),
                 hipdnn_plugin_sdk::HipdnnPluginException);

    resolver.properties = validHipProperties();
    EXPECT_EQ(resolver.deviceProperties(7).gcnArchName, "gfx000");
    EXPECT_EQ(resolver.queryCount.load(), 2);
}

TEST(TestHandleDeviceResolver, AcceptsIdentityFillingTheArchBuffer)
{
    // The terminator search spans the whole buffer, so an identity reaching its last byte
    // is complete rather than truncated.
    FakeQueryResolver resolver;
    const auto archBufferSize = sizeof(resolver.properties.gcnArchName);
    std::memset(resolver.properties.gcnArchName, 'x', archBufferSize);
    resolver.properties.gcnArchName[archBufferSize - 1] = '\0';

    EXPECT_EQ(resolver.deviceProperties(7).gcnArchName, std::string(archBufferSize - 1, 'x'));
}

TEST(TestHandleDeviceResolver, RejectsUnwrittenLdsWithoutLosingReportedZero)
{
    FakeQueryResolver resolver;
    resolver.omitLds = true;
    EXPECT_THROW(static_cast<void>(resolver.deviceProperties(7)),
                 hipdnn_plugin_sdk::HipdnnPluginException);

    resolver.omitLds = false;
    resolver.properties.sharedMemPerBlock = 0;
    const auto& resolved = resolver.deviceProperties(7);
    EXPECT_EQ(resolved.ldsSize, 0);
    EXPECT_EQ(&resolver.deviceProperties(7), &resolved);
    EXPECT_EQ(resolver.queryCount.load(), 2);
}

TEST(TestHandleDeviceResolver, AcceptsMaximumSignedLdsCapacity)
{
    FakeQueryResolver resolver;
    resolver.properties.sharedMemPerBlock = std::numeric_limits<int64_t>::max();
    const auto& resolved = resolver.deviceProperties(7);
    EXPECT_EQ(resolved.ldsSize, std::numeric_limits<int64_t>::max());
    EXPECT_EQ(&resolver.deviceProperties(7), &resolved);
    EXPECT_EQ(resolver.queryCount.load(), 1);
}

TEST(TestHandleDeviceResolver, NoDeviceStaysUnresolvedWithoutQueryingHip)
{
    const FakeQueryResolver resolver;
    const auto& unresolved = resolver.deviceProperties(hipdnn_plugin_sdk::ingestor::NO_DEVICE);
    EXPECT_TRUE(unresolved.gcnArchName.empty());
    EXPECT_EQ(unresolved.warpSize, 0);
    EXPECT_EQ(unresolved.multiProcessorCount, 0);
    EXPECT_EQ(unresolved.ldsSize, -1);
    EXPECT_EQ(&resolver.deviceProperties(hipdnn_plugin_sdk::ingestor::NO_DEVICE), &unresolved);
    EXPECT_EQ(resolver.queryCount.load(), 0);
}

TEST(TestHandleDeviceResolver, ConcurrentDevicePropertyLookupsAreSafe)
{
    // Use different capacities to detect results from the wrong device.
    FakeQueryResolver resolver;
    resolver.distinguishDevices = true;
    std::atomic<int> mismatches{0};

    std::vector<std::thread> threads;
    threads.reserve(8);
    for(int t = 0; t < 8; ++t)
    {
        threads.emplace_back([&resolver, &mismatches, t]() {
            const auto deviceId = (t % 4) + 2000;
            for(int i = 0; i < 200; ++i)
            {
                const auto& properties = resolver.deviceProperties(deviceId);
                if(properties.ldsSize != 65536 + deviceId || properties.warpSize != 64
                   || properties.multiProcessorCount != 48 || properties.gcnArchName != "gfx000")
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

    EXPECT_EQ(resolver.queryCount.load(), 4);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
