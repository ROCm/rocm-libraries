/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
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

#include "client/RotatingBuffer.hpp"
#include <algorithm>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cstddef>
#include <hip/hip_runtime.h>

using namespace rocRoller;

template <typename T>
static std::vector<T> copyDeviceSpanToHost(std::span<T> dSpan)
{
    std::vector<T> h(dSpan.size());
    // hipMemcpy expects void*; cast the device pointer accordingly
    HIP_CHECK(hipMemcpy(h.data(), dSpan.data(), sizeof(T) * dSpan.size(), hipMemcpyDeviceToHost));
    return h;
}

TEST_CASE("Disabled rotation returns base pointer", "[RotatingBuffer]")
{
    std::vector<float>    hostData(16, 1.0f);
    RotatingBuffer<float> buf(hostData, 0);

    auto span1 = buf.next();
    auto span2 = buf.next();

    REQUIRE(span1.data() == span2.data());
    REQUIRE(span1.size() == hostData.size());

    auto h1 = copyDeviceSpanToHost(span1);
    for(auto v : h1)
    {
        REQUIRE(v == 1.0f);
    }
}

TEST_CASE("Matrix smaller than cache rotates correctly", "[RotatingBuffer]")
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

    auto h1 = copyDeviceSpanToHost(span1);
    auto h2 = copyDeviceSpanToHost(span2);
    for(int i = 0; i < 4; i++)
    {
        REQUIRE(h1[i] == 42);
        REQUIRE(h2[i] == 42);
    }
}

TEST_CASE("Matrix larger than cache gracefully falls back to single buffer", "[RotatingBuffer]")
{
    std::vector<double> hostData(1024, 3.14);
    size_t              cacheBytes = 128; // smaller than one matrix

    RotatingBuffer<double> buf(hostData, cacheBytes);

    auto span1 = buf.next();
    auto span2 = buf.next();

    // Both calls should return the same base (no rotation)
    REQUIRE(span1.data() == span2.data());
    REQUIRE(span1.size() == hostData.size());

    auto h1 = copyDeviceSpanToHost(span1);
    for(double v : h1)
        REQUIRE(v == 3.14);
}

TEST_CASE("Data integrity across rotations", "[RotatingBuffer]")
{
    std::vector<int> hostData(8);
    for(int i = 0; i < 8; i++)
        hostData[i] = i;

    size_t              cacheBytes = 64; // can hold multiple instances
    RotatingBuffer<int> buf(hostData, cacheBytes);

    auto span1 = buf.next();
    auto span2 = buf.next(); // rotated

    auto h1 = copyDeviceSpanToHost(span1);
    auto h2 = copyDeviceSpanToHost(span2);
    for(int i = 0; i < 8; i++)
    {
        REQUIRE(h1[i] == hostData[i]);
        REQUIRE(h2[i] == hostData[i]); // copied data must match too
    }
}

TEST_CASE("Empty host data throws FatalError", "[RotatingBuffer]")
{
    std::vector<float> hostData;
    REQUIRE_THROWS_AS(RotatingBuffer<float>(hostData, 32), FatalError);
}

TEST_CASE("Small cacheBytes triggers graceful fallback to full buffer", "[RotatingBuffer]")
{
    std::vector<int> hostData(8, 7);
    size_t           cacheBytes = sizeof(int) * 4; // too small for one full copy

    RotatingBuffer<int> buf(hostData, cacheBytes);

    auto span = buf.next();

    // Should fall back to full allocation
    REQUIRE(span.size() == hostData.size());

    auto h = copyDeviceSpanToHost(span);
    REQUIRE(std::all_of(h.begin(), h.end(), [](int v) { return v == 7; }));
}

TEST_CASE("Odd cache size falls back safely", "[RotatingBuffer]")
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

    auto h1 = copyDeviceSpanToHost(span1);
    auto h2 = copyDeviceSpanToHost(span2);
    auto h3 = copyDeviceSpanToHost(span3);

    // All values should remain consistent
    for(int i = 0; i < 8; i++)
    {
        REQUIRE(h1[i] == hostData[i]);
        REQUIRE(h2[i] == hostData[i]);
        REQUIRE(h3[i] == hostData[i]);
    }
}

// Exact-fit cache (== one tensor): should not rotate; spans remain identical.
TEST_CASE("Exact-fit cache does not rotate", "[RotatingBuffer]")
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

    auto h1 = copyDeviceSpanToHost(s1);
    for(int i = 0; i < 8; ++i)
        REQUIRE(h1[i] == hostData[i]);
}

// Multi-copy rotation cycles deterministically (3 copies): addresses should cycle 0->+N->+2N->0...
TEST_CASE("Multi-copy rotation cycles addresses deterministically", "[RotatingBuffer]")
{
    std::vector<int> hostData(8);
    for(int i = 0; i < 8; ++i)
        hostData[i] = i;

    const int    N          = static_cast<int>(hostData.size());
    const size_t copies     = 3;
    const size_t cacheBytes = copies * N * sizeof(int);

    RotatingBuffer<int> buf(hostData, cacheBytes);

    // Collect a few successive spans and check address cycle
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

    // Expected rotation order with pre-advance semantics: N, 2N, 0, N  -> indices {1,2,0,1}
    REQUIRE(idxOf(s0.data()) == 1);
    REQUIRE(idxOf(s1.data()) == 2);
    REQUIRE(idxOf(s2.data()) == 0);
    REQUIRE(idxOf(s3.data()) == 1);

    auto h0 = copyDeviceSpanToHost(s0);
    auto h1 = copyDeviceSpanToHost(s1);
    auto h2 = copyDeviceSpanToHost(s2);
    auto h3 = copyDeviceSpanToHost(s3);
    for(int i = 0; i < N; ++i)
    {
        REQUIRE(h0[i] == hostData[i]);
        REQUIRE(h1[i] == hostData[i]);
        REQUIRE(h2[i] == hostData[i]);
        REQUIRE(h3[i] == hostData[i]);
    }
}

// Many iterations should never segfault (stress rotation & modulo logic)
TEST_CASE("Many rotations do not segfault and data remains stable", "[RotatingBuffer]")
{
    std::vector<float> hostData(32);
    for(int i = 0; i < 32; ++i)
        hostData[i] = static_cast<float>(i);

    // 5 copies -> plenty of rotation states
    const size_t          cacheBytes = 5 * hostData.size() * sizeof(float);
    RotatingBuffer<float> buf(hostData, cacheBytes);

    // Cycle a bunch of times; verify size & contents each time
    for(int iter = 0; iter < 256; ++iter)
    {
        auto s = buf.next();
        REQUIRE(s.size() == hostData.size());
        auto h = copyDeviceSpanToHost(s);
        for(int i = 0; i < 32; ++i)
            REQUIRE(h[i] == hostData[i]);
    }
}

// Alloc/free churn: ensure deleter (hipFree) path is exercised without faults or leaks
TEST_CASE("Allocator/deleter churn is safe", "[RotatingBuffer]")
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
            auto h = copyDeviceSpanToHost(s);
            for(double v : h)
                REQUIRE(v == 2.5);
        } // buf goes out of scope -> hipFree via deleter runs
    }
    HIP_CHECK(hipDeviceSynchronize());
}
