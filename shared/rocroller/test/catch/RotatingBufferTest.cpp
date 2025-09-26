
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
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cstddef>

using namespace rocRoller;

TEST_CASE("Disabled rotation returns base pointer", "[RotatingBuffer]")
{
    std::vector<float>    hostData(16, 1.0f);
    RotatingBuffer<float> buf(hostData, 0); // rotation disabled

    auto span1 = buf.next();
    auto span2 = buf.next();

    REQUIRE(span1.data() == span2.data()); // always base pointer
    REQUIRE(span1.size() == hostData.size());
    for(auto v : span1)
    {
        REQUIRE(v == 1.0f);
    }
}

TEST_CASE("Matrix smaller than cache rotates correctly", "[RotatingBuffer]")
{
    std::vector<int> hostData(4, 42);
    size_t           cacheBytes = 64; // enough for multiple copies

    RotatingBuffer<int> buf(hostData, cacheBytes);

    auto span1 = buf.next();
    auto span2 = buf.next();

    REQUIRE(span1.size() == 4);
    REQUIRE(span2.size() == 4);

    // Rotated forward by numElems
    REQUIRE(span2.data() == span1.data() + 4);

    for(int i = 0; i < 4; i++)
    {
        REQUIRE(span1[i] == 42);
        REQUIRE(span2[i] == 42);
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

    // All values should remain correct
    for(double v : span1)
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

    for(int i = 0; i < 8; i++)
    {
        REQUIRE(span1[i] == hostData[i]);
        REQUIRE(span2[i] == hostData[i]); // copied data must match too
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
    REQUIRE(std::all_of(span.begin(), span.end(), [](int v) { return v == 7; }));
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

    // All values should remain consistent
    for(int i = 0; i < 8; i++)
    {
        REQUIRE(span1[i] == hostData[i]);
        REQUIRE(span2[i] == hostData[i]);
        REQUIRE(span3[i] == hostData[i]);
    }
}
