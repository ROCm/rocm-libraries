/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2017-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include <rocRoller/TensorDescriptor.hpp>

#include <rocRoller/Utilities/Random.hpp>

#include <cstddef>

using namespace rocRoller;

TEST_CASE("TensorDescriptor basic functionality", "[utils][tensor-descriptor]")
{
    TensorDescriptor t(DataType::Float, {11, 13, 17});

    CHECK(t.dimensions() == 3);
    CHECK(t.sizes() == std::vector<size_t>({11, 13, 17}));
    CHECK(t.strides() == std::vector<size_t>({1, 11, 11 * 13}));

    CHECK(t.totalLogicalElements() == 11 * 13 * 17);
    CHECK(t.totalAllocatedElements() == 11 * 13 * 17);
    CHECK(t.totalAllocatedBytes() == 11 * 13 * 17 * 4);

    for(int i = 0; i < 3; i++)
        CHECK(t.dimensionPadding(i) == 0);

    CHECK(t.index(3, 4, 1) == 3 + 4 * 11 + 11 * 13);
}

TEST_CASE("TensorDescriptor works with overlapping strides", "[utils][tensor-descriptor]")
{
    TensorDescriptor t(DataType::Float, {4, 6, 3}, {1, 4, 1});

    CHECK(t.dimensions() == 3);
    CHECK(t.sizes() == std::vector<size_t>({4, 6, 3}));
    CHECK(t.strides() == std::vector<size_t>({1, 4, 1}));

    CHECK(t.dimensionPadding(0) == 0);
    CHECK(t.dimensionPadding(1) == 0);

    CHECK(t.totalLogicalElements() == 4 * 6 * 3);
    CHECK(t.totalAllocatedElements() == 4 * 6 + (3 - 1));
    CHECK(t.totalAllocatedBytes() == (4 * 6 + (3 - 1)) * 4);
}

TEST_CASE("TensorDescriptor works with padding", "[utils][tensor-descriptor]")
{
    TensorDescriptor t(DataType::Float, {11, 13, 17, 4}, {1, 16, 16 * 13, 16 * 13 * 17});

    CHECK(t.dimensions() == 4);
    CHECK(t.sizes() == std::vector<size_t>({11, 13, 17, 4}));
    CHECK(t.strides() == std::vector<size_t>({1, 16, 16 * 13, 16 * 13 * 17}));

    CHECK(t.totalLogicalElements() == 11 * 13 * 17 * 4);
    CHECK(t.totalAllocatedElements()
          == 1 + 1 * (11 - 1) + 16 * (13 - 1) + (16 * 13) * (17 - 1) + (16 * 13 * 17) * (4 - 1));

    CHECK(t.dimensionPadding(0) == 0);
    CHECK(t.dimensionPadding(1) == 5);
    CHECK(t.dimensionPadding(2) == 0);
    CHECK(t.dimensionPadding(3) == 0);

    CHECK(t.index(3, 4, 1, 2) == 3 + 4 * 16 + 16 * 13 + 16 * 13 * 17 * 2);
}

TEST_CASE("TensorDescriptor works with simplified padding", "[utils][tensor-descriptor]")
{
    TensorDescriptor t(DataType::Float, {4, 5}, {1, 8});

    CHECK(t.dimensions() == 2);
    CHECK(t.sizes() == std::vector<size_t>({4, 5}));
    CHECK(t.strides() == std::vector<size_t>({1, 8})); // default 1,4

    CHECK(t.dimensionPadding(0) == 0);
    CHECK(t.dimensionPadding(1) == 4);

    CHECK(t.totalLogicalElements() == 4 * 5);
    CHECK(t.totalAllocatedElements() == 4 + 8 * (5 - 1));
    CHECK(t.totalAllocatedBytes() == (4 + 8 * (5 - 1)) * 4);
}

TEST_CASE("TensorDescriptor works with zero strides", "[utils][tensor-descriptor]")
{
    TensorDescriptor t(DataType::Float,
                       {4, 5, 6},
                       {TensorDescriptor::UseDefaultStride, TensorDescriptor::UseDefaultStride, 0});

    CHECK(t.dimensions() == 3);
    CHECK(t.sizes() == std::vector<size_t>({4, 5, 6}));
    CHECK(t.strides() == std::vector<size_t>({1, 4, 0})); // default 1,4

    CHECK(t.dimensionPadding(0) == 0);
    CHECK(t.dimensionPadding(1) == 0);
    CHECK(t.dimensionPadding(2) == -20);

    CHECK(t.totalLogicalElements() == 4 * 5 * 6);
    CHECK(t.totalAllocatedElements() == 4 * 5);
    CHECK(t.totalAllocatedBytes() == 4 * 5 * 4);
}

TEST_CASE("TensorDescriptor::CollapseDims works", "[utils][tensor-descriptor]")
{
    TensorDescriptor t(DataType::Float, {11, 13, 17, 4}, {1, 16, 16 * 13, 16 * 13 * 17});

    TensorDescriptor u = t;
    CHECK_THROWS_AS(u.collapseDims(0, 2), std::runtime_error);

    u.collapseDims(1, 3);

    CHECK(u.dimensions() == 3);
    CHECK(u.sizes() == std::vector<size_t>({11, 13 * 17, 4}));
    CHECK(u.strides() == std::vector<size_t>({1, 16, 16 * 13 * 17}));

    CHECK(u.totalLogicalElements() == t.totalLogicalElements());
    CHECK(u.totalAllocatedElements() == t.totalAllocatedElements());
    CHECK(u.totalAllocatedBytes() == t.totalAllocatedBytes());
}

TEST_CASE("TensorDescriptor::CollapseDims works part 2", "[utils][tensor-descriptor]")
{
    TensorDescriptor t(DataType::Float, {11, 13, 17, 4});

    SECTION("0,2")
    {
        TensorDescriptor u = t;
        u.collapseDims(0, 2);

        CHECK(u.dimensions() == 3);
        CHECK(u.sizes() == std::vector<size_t>({11 * 13, 17, 4}));
        CHECK(u.strides() == std::vector<size_t>({1, 11 * 13, 11 * 13 * 17}));

        CHECK(u.totalLogicalElements() == t.totalLogicalElements());
        CHECK(u.totalAllocatedElements() == t.totalAllocatedElements());
        CHECK(u.totalAllocatedBytes() == t.totalAllocatedBytes());
    }

    SECTION("0,4")
    {
        TensorDescriptor u = t;
        u.collapseDims(0, 4);

        CHECK(u.dimensions() == 1);
        CHECK(u.sizes() == std::vector<size_t>({11 * 13 * 17 * 4}));
        CHECK(u.strides() == std::vector<size_t>({1}));

        CHECK(u.totalLogicalElements() == t.totalLogicalElements());
        CHECK(u.totalAllocatedElements() == t.totalAllocatedElements());
        CHECK(u.totalAllocatedBytes() == t.totalAllocatedBytes());
    }

    SECTION("1,4")
    {
        TensorDescriptor u = t;
        u.collapseDims(1, 4);

        CHECK(u.dimensions() == 2);
        CHECK(u.sizes() == std::vector<size_t>({11, 13 * 17 * 4}));
        CHECK(u.strides() == std::vector<size_t>({1, 11}));

        CHECK(u.totalLogicalElements() == t.totalLogicalElements());
        CHECK(u.totalAllocatedElements() == t.totalAllocatedElements());
        CHECK(u.totalAllocatedBytes() == t.totalAllocatedBytes());
    }

    SECTION("1,3")
    {
        TensorDescriptor u = t;
        u.collapseDims(1, 3);

        CHECK(u.dimensions() == 3);
        CHECK(u.sizes() == std::vector<size_t>({11, 13 * 17, 4}));
        CHECK(u.strides() == std::vector<size_t>({1, 11, 11 * 13 * 17}));

        CHECK(u.totalLogicalElements() == t.totalLogicalElements());
        CHECK(u.totalAllocatedElements() == t.totalAllocatedElements());
        CHECK(u.totalAllocatedBytes() == t.totalAllocatedBytes());
    }
}

TEST_CASE("IncrementCoord works for 2 dimensions", "[utils][tensor-descriptor]")
{
    std::vector<size_t> dims{2, 4};
    std::vector<size_t> lastCoord{1, 3};
    std::vector<size_t> coordRef(2);
    std::vector<size_t> coordRun(2);

    for(coordRef[1] = 0; coordRef[1] < dims[1]; coordRef[1]++)
        for(coordRef[0] = 0; coordRef[0] < dims[0]; coordRef[0]++)
        {
            CHECK(coordRun == coordRef);

            bool continueIteration
                = IncrementCoord(coordRun.begin(), coordRun.end(), dims.begin(), dims.end());
            if(coordRef == lastCoord)
                CHECK(continueIteration == false);
            else
                CHECK(continueIteration == true);
        }

    coordRef = {0, 0};
    CHECK(coordRun == coordRef);

    CHECK(IncrementCoord(coordRun.begin(), coordRun.end(), dims.begin(), dims.end()));
}

TEST_CASE("Default strides work for TensorDescriptor.", "[utils][tensor-descriptor]")
{
    TensorDescriptor desc(DataType::Float, {4, 5, 6}, {static_cast<size_t>(-1), 5});
    CHECK(desc.dimensions() == 3);
    CHECK(desc.sizes() == std::vector<size_t>({4, 5, 6}));
    CHECK(desc.strides() == std::vector<size_t>({1, 5, 25}));
}

TEST_CASE("Specifying a subset of strides works for TensorDescriptor.",
          "[utils][tensor-descriptor]")
{
    TensorDescriptor desc(DataType::Float, {4, 5, 6}, {5});
    CHECK(desc.dimensions() == 3);
    CHECK(desc.sizes() == std::vector<size_t>({4, 5, 6}));
    CHECK(desc.strides() == std::vector<size_t>({5, 20, 100}));
}

TEST_CASE("ShuffleDims works in the no-op case", "[utils][tensor-descriptor]")
{
    auto dims = GENERATE(Catch::Generators::range(2, 10));
    DYNAMIC_SECTION(fmt::format("dims={}", dims))
    {
        auto seed = GENERATE(Catch::Generators::take(
            4, Catch::Generators::random(0, std::numeric_limits<int>::max())));

        DYNAMIC_SECTION(fmt::format("seed={}", seed))
        {
            auto sizes = RandomGenerator(seed).vector<size_t>(dims, 1, 12);
            CAPTURE(sizes);

            TensorDescriptor desc(DataType::Int32, sizes);

            auto input  = iota<int>(0, desc.totalAllocatedElements()).to<std::vector>();
            auto output = shuffleDims(input, desc, desc);

            CHECK(input == output);
        }
    }
}

TEST_CASE("ShuffleDims is reversible", "[utils][tensor-descriptor]")
{
    auto dims = GENERATE(Catch::Generators::range(2, 10));
    DYNAMIC_SECTION(fmt::format("dims={}", dims))
    {
        auto seed = GENERATE(Catch::Generators::take(
            4, Catch::Generators::random(0, std::numeric_limits<int>::max())));

        DYNAMIC_SECTION(fmt::format("seed={}", seed))
        {
            auto gen   = RandomGenerator(seed);
            auto sizes = gen.vector<size_t>(dims, 2, 12);
            CAPTURE(sizes);

            TensorDescriptor src(DataType::Int32, sizes);

            std::vector<size_t> order;
            {
                auto indices = iota<size_t>(0, dims).to<std::vector>();
                while(!indices.empty())
                {
                    auto idx = gen.next<size_t>(0, indices.size() - 1);
                    REQUIRE(idx < dims);
                    order.push_back(indices.at(idx));
                    indices.erase(next(indices.begin(), idx));
                }
            }

            auto dst = TensorDescriptor::ShuffledNoPadding(DataType::Int32, sizes, order);

            auto numNonUnitSizes = std::ranges::count_if(sizes, [](auto x) { return x > 1; });

            CAPTURE(src.strides());
            CAPTURE(dst.strides());

            auto input = iota<int>(0, src.totalAllocatedElements()).to<std::vector>();

            SECTION("a -> b -> a")
            {
                auto intermediate = shuffleDims(input, dst, src);
                CAPTURE(intermediate);
                if(numNonUnitSizes > 1 && src.strides() != dst.strides())
                    CHECK(input != intermediate);
                auto output = shuffleDims(intermediate, src, dst);
                REQUIRE(input == output);
            }

            SECTION("b -> a -> b")
            {
                auto intermediate = shuffleDims(input, src, dst);
                CAPTURE(intermediate);
                if(numNonUnitSizes > 1 && src.strides() != dst.strides())
                    CHECK(input != intermediate);
                auto output = shuffleDims(intermediate, dst, src);
                REQUIRE(input == output);
            }
        }
    }
}