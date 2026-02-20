// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "client/RotatingBuffer.hpp"
#include "CustomMatchers.hpp"
#include <algorithm>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cstddef>
#include <hip/hip_runtime.h>

using namespace rocRoller;

TEST_CASE("RotatingBuffer: Disabled rotation returns base pointer", "[gpu][utils]")
{
    std::vector<float>    hostData(16, 1.0f);
    RotatingBuffer<float> buf(hostData, 0);

    auto span1 = buf.next();
    auto span2 = buf.next();

    REQUIRE(span1.data() == span2.data());
    REQUIRE(span1.size() == hostData.size());

    REQUIRE_THAT(span1.data(), HasDeviceVectorEqualTo(hostData));
}

TEST_CASE("RotatingBuffer: Matrix smaller than cache rotates correctly", "[gpu][utils]")
{
    std::vector<int> hostData(4, 42);
    size_t           cacheBytes = 64;

    RotatingBuffer<int> buf(hostData, cacheBytes);

    auto span1 = buf.next();
    auto span2 = buf.next();

    REQUIRE(span1.size() == 4);
    REQUIRE(span2.size() == 4);

    // Rotated forward by numElems
    REQUIRE(span2.data() == span1.data() + 4);

    REQUIRE_THAT(span1.data(), HasDeviceVectorEqualTo(hostData));
    REQUIRE_THAT(span2.data(), HasDeviceVectorEqualTo(hostData));
}

TEST_CASE("RotatingBuffer: Matrix larger than cache gracefully falls back to single buffer",
          "[gpu][utils]")
{
    std::vector<double> hostData(1024, 3.14);
    size_t              cacheBytes = 128; // smaller than one matrix

    RotatingBuffer<double> buf(hostData, cacheBytes);

    auto span1 = buf.next();
    auto span2 = buf.next();

    // Both calls should return the same base (no rotation)
    REQUIRE(span1.data() == span2.data());
    REQUIRE(span1.size() == hostData.size());

    REQUIRE_THAT(span1.data(), HasDeviceVectorEqualTo(hostData));
}

TEST_CASE("RotatingBuffer: Data integrity across rotations", "[gpu][utils]")
{
    std::vector<int> hostData(8);
    for(int i = 0; i < 8; i++)
        hostData[i] = i;

    size_t              cacheBytes = 64; // can hold multiple instances
    RotatingBuffer<int> buf(hostData, cacheBytes);

    auto span1 = buf.next();
    auto span2 = buf.next(); // rotated

    // copied data must match too
    REQUIRE_THAT(span1.data(), HasDeviceVectorEqualTo(hostData));
    REQUIRE_THAT(span2.data(), HasDeviceVectorEqualTo(hostData));
}

TEST_CASE("RotatingBuffer: Empty host data throws FatalError", "[gpu][utils]")
{
    std::vector<float> hostData;
    REQUIRE_THROWS_AS(RotatingBuffer<float>(hostData, 32), FatalError);
}

TEST_CASE("RotatingBuffer: Small cacheBytes triggers graceful fallback to full buffer",
          "[gpu][utils]")
{
    std::vector<int> hostData(8, 7);
    size_t           cacheBytes = sizeof(int) * 4; // too small for one full copy

    RotatingBuffer<int> buf(hostData, cacheBytes);

    // Should fall back to full allocation
    auto span = buf.next();
    REQUIRE(span.size() == hostData.size());

    REQUIRE_THAT(span.data(), HasDeviceVectorEqualTo(hostData));
}

TEST_CASE("RotatingBuffer: Odd cache size falls back safely", "[gpu][utils]")
{
    std::vector<int> hostData(8);
    for(int i = 0; i < 8; i++)
        hostData[i] = i;

    size_t cacheBytes = 67; // not enough for 2 full copies

    RotatingBuffer<int> buf(hostData, cacheBytes);

    auto span1 = buf.next();
    auto span2 = buf.next(); // should advance by 8 elements (wrap to second copy)
    auto span3 = buf.next(); // wraps back to first copy

    REQUIRE(span1.size() == 8);
    REQUIRE(span2.size() == 8);
    REQUIRE(span3.size() == 8);

    // All values should remain consistent
    REQUIRE_THAT(span1.data(), HasDeviceVectorEqualTo(hostData));
    REQUIRE_THAT(span2.data(), HasDeviceVectorEqualTo(hostData));
    REQUIRE_THAT(span3.data(), HasDeviceVectorEqualTo(hostData));
}

// Exact-fit cache (== one tensor): should not rotate; spans remain identical.
TEST_CASE("RotatingBuffer: Exact-fit cache does not rotate", "[gpu][utils]")
{
    std::vector<int> hostData(8);
    for(int i = 0; i < 8; ++i)
        hostData[i] = i;

    const size_t        tensorBytes = hostData.size() * sizeof(int);
    RotatingBuffer<int> buf(hostData, tensorBytes); // exact fit

    auto s1 = buf.next();
    auto s2 = buf.next();

    REQUIRE(s1.size() == hostData.size());
    REQUIRE(s2.size() == hostData.size());
    REQUIRE(s2.data() == s1.data()); // no rotation

    REQUIRE_THAT(s1.data(), HasDeviceVectorEqualTo(hostData));
}

// Multi-copy rotation cycles deterministically (3 copies): addresses should cycle 0->+N->+2N->0...
TEST_CASE("RotatingBuffer: Multi-copy rotation cycles addresses deterministically", "[gpu][utils]")
{
    std::vector<int> hostData(8);
    for(int i = 0; i < 8; ++i)
        hostData[i] = i;

    const int    N          = static_cast<int>(hostData.size());
    const size_t copies     = 3;
    const size_t cacheBytes = copies * N * sizeof(int);

    RotatingBuffer<int> buf(hostData, cacheBytes);

    // Collect a few successive spans and check address cyc
    auto s0 = buf.next(); // offset N
    auto s1 = buf.next(); // offset 2N
    auto s2 = buf.next(); // offset 0 (wrap)
    auto s3 = buf.next(); // offset N

    REQUIRE(s0.size() == N);
    REQUIRE(s1.size() == N);
    REQUIRE(s2.size() == N);
    REQUIRE(s3.size() == N);

    //Discover the true base address (minimum of the three unique pointers)
    std::array<int*, 4> ptrs{s0.data(), s1.data(), s2.data(), s3.data()};
    auto                base = *std::min_element(ptrs.begin(), ptrs.end());

    // Helper to map a pointer to which copy it points at: 0 -> base, 1 -> base+N, 2 -> base+2N
    auto idxOf = [&](int* p) -> int {
        if(p == base)
            return 0;
        if(p == base + N)
            return 1;
        if(p == base + 2 * N)
            return 2;
        FAIL_CHECK("Span data not at an expected rotation slot");
        return -1;
    };

    REQUIRE(idxOf(s0.data()) == 1);
    REQUIRE(idxOf(s1.data()) == 2);
    REQUIRE(idxOf(s2.data()) == 0);
    REQUIRE(idxOf(s3.data()) == 1);

    REQUIRE_THAT(s0.data(), HasDeviceVectorEqualTo(hostData));
    REQUIRE_THAT(s1.data(), HasDeviceVectorEqualTo(hostData));
    REQUIRE_THAT(s2.data(), HasDeviceVectorEqualTo(hostData));
    REQUIRE_THAT(s3.data(), HasDeviceVectorEqualTo(hostData));
}

// Many iterations should never segfault (stress rotation & modulo logic)
TEST_CASE("RotatingBuffer: Many rotations do not segfault and data remains stable", "[gpu][utils]")
{
    std::vector<float> hostData(32);
    for(int i = 0; i < 32; ++i)
        hostData[i] = static_cast<float>(i);

    const size_t          cacheBytes = 5 * hostData.size() * sizeof(float);
    RotatingBuffer<float> buf(hostData, cacheBytes);

    // Cycle a bunch of times; verify size & contents each time
    for(int iter = 0; iter < 256; ++iter)
    {
        auto s = buf.next();
        REQUIRE(s.size() == hostData.size());
        REQUIRE_THAT(s.data(), HasDeviceVectorEqualTo(hostData));
    }
}

// Alloc/free churn: ensure deleter (hipFree) path is exercised without faults or leaks
TEST_CASE("RotatingBuffer: Allocator/deleter churn is safe", "[gpu][utils]")
{
    for(int rep = 0; rep < 64; ++rep)
    {
        std::vector<double> hostData(64, 2.5);
        // Alternate between disabled rotation and multi-copy to mix code paths
        size_t cacheBytes = (rep % 2 == 0) ? 0 : (3 * hostData.size() * sizeof(double));
        {
            RotatingBuffer<double> buf(hostData, cacheBytes);
            auto                   s = buf.next();
            REQUIRE(s.size() == hostData.size());
            REQUIRE_THAT(s.data(), HasDeviceVectorEqualTo(hostData));
        }
    }
    HIP_CHECK(hipDeviceSynchronize());
}
