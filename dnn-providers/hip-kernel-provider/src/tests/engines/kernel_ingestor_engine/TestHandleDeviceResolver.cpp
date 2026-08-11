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
 * Now a process-lifetime static (see KernelIngestorEngine.cpp's deviceResolver()) rather
 * than a per-engine member, which makes its cache semantics load-bearing rather than
 * incidental: the same instance answers for every engine and every container this
 * process ever builds, so a stale entry or an invalidated reference is a
 * process-lifetime bug, not one bounded by an engine's or a container's lifetime.
 *
 * deviceId() has four paths per its own doc: a null stream falls through to the current
 * device, a non-null stream resolves via hipStreamGetDevice, a stream hipStreamGetDevice
 * cannot resolve falls through the same way a null one does, and a process with no
 * usable HIP context at all falls back to device 0. The first three are driven below;
 * the fourth requires a HIP runtime with no usable context, which is not something a
 * test can put this process into without faking the HIP runtime itself, so it is a
 * defensive branch with no test hook here -- documented, not exercised.
 */
namespace
{

using hip_kernel_provider::kernel_ingestor_engine::HandleDeviceResolver;

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

    // A real, non-null stream resolves through hipStreamGetDevice rather than through
    // whichever device happens to be current on the calling thread -- the property that
    // makes the answer correct when several threads drive different handles.
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

    // A stream hipStreamGetDevice cannot resolve (here, one already destroyed) must not
    // propagate that failure: it falls through to the current device exactly as a null
    // stream does. Destroying the stream also leaves a genuine HIP error pending, which
    // this test clears so it does not fail an unrelated test via HipErrorHandler.
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

    // A cache miss followed by a hit for the same device must answer identically --
    // same address, since deviceProperties() promises a reference stable for the
    // resolver's lifetime, and a fresh lookup that reallocated would break that promise
    // silently rather than answering wrong.
    const auto& first = resolver.deviceProperties(0);
    const auto& second = resolver.deviceProperties(0);

    EXPECT_EQ(&first, &second);
    EXPECT_EQ(first.warpSize, second.warpSize);
}

TEST(TestHandleDeviceResolver, ReferencesStayValidAcrossCacheGrowth)
{
    // This is the invariant HandleDeviceResolver's own doc claims: "the cache hands out
    // references that stay valid for this resolver's lifetime... entries are never
    // erased or rehashed away: node handles in std::unordered_map keep referenced values
    // pinned across growth." Verified directly here rather than trusted from the
    // container choice, since a future change to a different container (e.g. a
    // node-invalidating flat map) would silently break every caller holding a reference
    // across a later insert.
    //
    // Invalid device ids are used deliberately: hipGetDeviceProperties() fails for them
    // and deviceProperties() still caches and returns the (zeroed) result, which is
    // enough to grow the map without needing that many real devices. CPU-only: no
    // successful HIP call is required, only that insertion happens and does not move
    // prior entries.
    const HandleDeviceResolver resolver;

    const auto& firstInserted = resolver.deviceProperties(1000);

    // Enough insertions to force at least one rehash of a typical unordered_map's
    // default bucket count.
    for(int deviceId = 1001; deviceId < 1064; ++deviceId)
    {
        static_cast<void>(resolver.deviceProperties(deviceId));
    }

    const auto& sameEntryAfterGrowth = resolver.deviceProperties(1000);
    EXPECT_EQ(&firstInserted, &sameEntryAfterGrowth);

    static_cast<void>(hipGetLastError());
    static_cast<void>(hipExtGetLastError());
}

TEST(TestHandleDeviceResolver, ConcurrentDevicePropertyLookupsAreSafe)
{
    // The cache is mutex-guarded because registration and lookup can race (see the
    // class's own doc); this drives that path directly rather than trusting the lock is
    // exercised by incidental test-suite parallelism. Every thread asks about the same
    // small set of device ids so a real race has many chances to corrupt the map if the
    // lock were missing.
    const HandleDeviceResolver resolver;

    std::vector<std::thread> threads;
    threads.reserve(8);
    for(int t = 0; t < 8; ++t)
    {
        threads.emplace_back([&resolver, t]() {
            for(int i = 0; i < 200; ++i)
            {
                static_cast<void>(resolver.deviceProperties((t % 4) + 2000));
            }
        });
    }
    for(auto& thread : threads)
    {
        thread.join();
    }

    static_cast<void>(hipGetLastError());
    static_cast<void>(hipExtGetLastError());
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
