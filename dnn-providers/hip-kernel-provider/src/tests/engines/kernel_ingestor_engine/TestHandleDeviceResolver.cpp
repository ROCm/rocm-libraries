// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <atomic>
#include <cstddef>
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

/// Supplies fake device properties and counts queries, so no GPU is needed.
class FakeQueryResolver : public HandleDeviceResolver
{
public:
    hipDeviceProp_t properties = validHipProperties();
    hipError_t status = hipSuccess;
    bool distinguishDevices = false;
    mutable std::atomic<int> queryCount{0};

    hipError_t queryDeviceProperties(hipDeviceProp_t* result,
                                     hipdnn_plugin_sdk::ingestor::DeviceId deviceId) const override
    {
        queryCount.fetch_add(1, std::memory_order_relaxed);
        if(status != hipSuccess)
        {
            return status;
        }

        *result = properties;
        if(distinguishDevices)
        {
            result->sharedMemPerBlock += static_cast<size_t>(deviceId);
        }
        return hipSuccess;
    }
};

/// The largest byte count the ingestor accepts: facts travel on as int64_t.
std::size_t maxByteCount()
{
    return static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max());
}

/// The message @p resolver refuses @p deviceId with, or a test failure if it resolves.
std::string refusalMessage(const HandleDeviceResolver& resolver,
                           hipdnn_plugin_sdk::ingestor::DeviceId deviceId)
{
    try
    {
        static_cast<void>(resolver.deviceProperties(deviceId));
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
    {
        return error.what();
    }
    ADD_FAILURE() << "device " << deviceId << " was not refused";
    return {};
}

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
    EXPECT_EQ(first.sharedMemPerBlock, reported.sharedMemPerBlock);
    EXPECT_EQ(first.totalGlobalMem, reported.totalGlobalMem);
    EXPECT_EQ(first.memoryBusWidth, reported.memoryBusWidth);
    EXPECT_EQ(first.memoryClockRate, reported.memoryClockRate);
}

TEST(TestHandleDeviceResolver, ACacheHitDoesNotQueryHipAgain)
{
    FakeQueryResolver resolver;
    const auto& first = resolver.deviceProperties(7);

    // Make a second query fail, so only a cache hit can succeed.
    resolver.status = hipErrorInvalidDevice;
    resolver.properties.sharedMemPerBlock = 32768;
    const auto& second = resolver.deviceProperties(7);

    EXPECT_EQ(&first, &second);
    EXPECT_EQ(first.sharedMemPerBlock, 65536U);
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
    EXPECT_EQ(firstInserted.sharedMemPerBlock, 65536U + 1000U);
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
    EXPECT_EQ(recovered.sharedMemPerBlock, 65536U);
    EXPECT_EQ(&resolver.deviceProperties(7), &recovered);
    EXPECT_EQ(resolver.queryCount.load(), 3);
}

TEST(TestHandleDeviceResolver, RemembersTheRefusalOfInvalidDeviceFacts)
{
    struct InvalidFacts
    {
        const char* name;
        const char* field;
        void (*corrupt)(hipDeviceProp_t&);
    };
    const std::array<InvalidFacts, 14> cases = {{
        {"missing identity", "gcnArchName", [](hipDeviceProp_t& p) { p.gcnArchName[0] = '\0'; }},
        {"unterminated identity",
         "gcnArchName",
         [](hipDeviceProp_t& p) { std::memset(p.gcnArchName, 'x', sizeof(p.gcnArchName)); }},
        {"zero warp", "warpSize", [](hipDeviceProp_t& p) { p.warpSize = 0; }},
        {"negative warp", "warpSize", [](hipDeviceProp_t& p) { p.warpSize = -1; }},
        {"zero count",
         "multiProcessorCount",
         [](hipDeviceProp_t& p) { p.multiProcessorCount = 0; }},
        {"negative count",
         "multiProcessorCount",
         [](hipDeviceProp_t& p) { p.multiProcessorCount = -1; }},
        {"zero LDS capacity",
         "sharedMemPerBlock",
         [](hipDeviceProp_t& p) { p.sharedMemPerBlock = 0; }},
        {"LDS capacity past int64",
         "sharedMemPerBlock",
         [](hipDeviceProp_t& p) { p.sharedMemPerBlock = maxByteCount() + 1; }},
        {"maximal LDS capacity",
         "sharedMemPerBlock",
         [](hipDeviceProp_t& p) { p.sharedMemPerBlock = std::numeric_limits<std::size_t>::max(); }},
        {"global memory past int64",
         "totalGlobalMem",
         [](hipDeviceProp_t& p) { p.totalGlobalMem = maxByteCount() + 1; }},
        {"maximal global memory",
         "totalGlobalMem",
         [](hipDeviceProp_t& p) { p.totalGlobalMem = std::numeric_limits<std::size_t>::max(); }},
        {"negative bus width", "memoryBusWidth", [](hipDeviceProp_t& p) { p.memoryBusWidth = -1; }},
        {"negative memory clock",
         "memoryClockRate",
         [](hipDeviceProp_t& p) { p.memoryClockRate = -1; }},
        {"most negative memory clock",
         "memoryClockRate",
         [](hipDeviceProp_t& p) { p.memoryClockRate = std::numeric_limits<int>::min(); }},
    }};
    for(const auto& invalid : cases)
    {
        SCOPED_TRACE(invalid.name);
        FakeQueryResolver resolver;
        invalid.corrupt(resolver.properties);

        const auto refusal = refusalMessage(resolver, 7);
        EXPECT_NE(refusal.find(invalid.field), std::string::npos) << refusal;

        // The device answered, so its refusal stands even once the runtime reports
        // valid facts: the same id is refused again without asking HIP.
        resolver.properties = validHipProperties();
        EXPECT_EQ(refusalMessage(resolver, 7), refusal);
        EXPECT_EQ(resolver.queryCount.load(), 1);

        // Another device is queried and cached on its own facts.
        const auto& other = resolver.deviceProperties(8);
        EXPECT_EQ(other.sharedMemPerBlock, 65536U);
        EXPECT_EQ(&resolver.deviceProperties(8), &other);
        EXPECT_EQ(resolver.queryCount.load(), 2);
    }
}

TEST(TestHandleDeviceResolver, AcceptsIdentityFillingTheArchBuffer)
{
    // The terminator may sit in the last byte, so a name filling the buffer is complete.
    FakeQueryResolver resolver;
    const auto archBufferSize = sizeof(resolver.properties.gcnArchName);
    std::memset(resolver.properties.gcnArchName, 'x', archBufferSize);
    resolver.properties.gcnArchName[archBufferSize - 1] = '\0';

    EXPECT_EQ(resolver.deviceProperties(7).gcnArchName, std::string(archBufferSize - 1, 'x'));
}

TEST(TestHandleDeviceResolver, AcceptsEachFactAtTheEndsOfItsRange)
{
    struct Boundary
    {
        const char* name;
        void (*apply)(hipDeviceProp_t&);
    };
    const std::array<Boundary, 10> cases = {{
        {"smallest LDS capacity", [](hipDeviceProp_t& p) { p.sharedMemPerBlock = 1; }},
        {"largest LDS capacity", [](hipDeviceProp_t& p) { p.sharedMemPerBlock = maxByteCount(); }},
        {"unknown global memory", [](hipDeviceProp_t& p) { p.totalGlobalMem = 0; }},
        {"largest global memory", [](hipDeviceProp_t& p) { p.totalGlobalMem = maxByteCount(); }},
        {"unknown bus width", [](hipDeviceProp_t& p) { p.memoryBusWidth = 0; }},
        {"largest bus width",
         [](hipDeviceProp_t& p) { p.memoryBusWidth = std::numeric_limits<int>::max(); }},
        {"unknown memory clock", [](hipDeviceProp_t& p) { p.memoryClockRate = 0; }},
        {"largest memory clock",
         [](hipDeviceProp_t& p) { p.memoryClockRate = std::numeric_limits<int>::max(); }},
        {"largest warp", [](hipDeviceProp_t& p) { p.warpSize = std::numeric_limits<int>::max(); }},
        {"largest count",
         [](hipDeviceProp_t& p) { p.multiProcessorCount = std::numeric_limits<int>::max(); }},
    }};
    for(const auto& boundary : cases)
    {
        SCOPED_TRACE(boundary.name);
        FakeQueryResolver resolver;
        boundary.apply(resolver.properties);

        const auto& resolved = resolver.deviceProperties(7);
        EXPECT_EQ(resolved.warpSize, resolver.properties.warpSize);
        EXPECT_EQ(resolved.multiProcessorCount, resolver.properties.multiProcessorCount);
        EXPECT_EQ(resolved.sharedMemPerBlock, resolver.properties.sharedMemPerBlock);
        EXPECT_EQ(resolved.totalGlobalMem, resolver.properties.totalGlobalMem);
        EXPECT_EQ(resolved.memoryBusWidth, resolver.properties.memoryBusWidth);
        EXPECT_EQ(resolved.memoryClockRate, resolver.properties.memoryClockRate);
        EXPECT_EQ(&resolver.deviceProperties(7), &resolved);
        EXPECT_EQ(resolver.queryCount.load(), 1);
    }
}

TEST(TestHandleDeviceResolver, NoDeviceStaysUnresolvedWithoutQueryingHip)
{
    const FakeQueryResolver resolver;
    const auto& unresolved = resolver.deviceProperties(hipdnn_plugin_sdk::ingestor::NO_DEVICE);
    EXPECT_TRUE(unresolved.gcnArchName.empty());
    EXPECT_EQ(unresolved.warpSize, 0);
    EXPECT_EQ(unresolved.multiProcessorCount, 0);
    EXPECT_EQ(unresolved.sharedMemPerBlock, 0U);
    EXPECT_EQ(&resolver.deviceProperties(hipdnn_plugin_sdk::ingestor::NO_DEVICE), &unresolved);
    EXPECT_EQ(resolver.queryCount.load(), 0);
}

TEST(TestHandleDeviceResolver, ConcurrentDevicePropertyLookupsAreSafe)
{
    // Give each device a different LDS size so a result from the wrong device shows up.
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
                if(properties.sharedMemPerBlock != 65536U + static_cast<std::size_t>(deviceId)
                   || properties.warpSize != 64 || properties.multiProcessorCount != 48
                   || properties.gcnArchName != "gfx000")
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
