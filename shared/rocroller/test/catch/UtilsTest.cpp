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

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Utilities/Random.hpp>
#include <rocRoller/Utilities/Utils.hpp>

#include <catch2/benchmark/catch_benchmark_all.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

TEST_CASE("concatenate_join joins different types", "[infrastructure][utils]")
{
    CHECK(rocRoller::concatenate_join(", ", "x", 5, rocRoller::DataType::Double) == "x, 5, Double");

    CHECK(rocRoller::concatenate_join(", ") == "");

    CHECK(rocRoller::concatenate_join(", ", 6) == "6");
}

TEST_CASE("mergeSets works", "[infrastructure][utils]")
{
    using vecSet = std::vector<std::set<int>>;

    auto checkPostconditions = [](vecSet input) {
        std::set<int> allInputValues;
        for(auto const& s : input)
            allInputValues.insert(s.begin(), s.end());

        auto output = rocRoller::mergeSets(input);

        std::set<int> seenOutputValues;

        for(auto const& s : output)
        {
            for(auto const& value : s)
                CHECK_FALSE(seenOutputValues.contains(value));

            seenOutputValues.insert(s.begin(), s.end());
        }

        CHECK(allInputValues == seenOutputValues);

        auto output2 = rocRoller::mergeSets(output);
        std::sort(output.begin(), output.end());
        std::sort(output2.begin(), output2.end());

        CHECK(output == output2);
    };

    SECTION("Empty set")
    {
        vecSet input;
        CHECK(rocRoller::mergeSets(input) == input);
        checkPostconditions(input);
    }

    SECTION("Empty sets 1")
    {
        vecSet input = {{}, {1}, {}};
        CHECK(rocRoller::mergeSets(input) == input);
        checkPostconditions(input);
    }

    SECTION("Empty sets 2")
    {
        vecSet input  = {{}, {1}, {}, {1, 2}};
        vecSet output = {{}, {1, 2}, {}};
        CHECK(rocRoller::mergeSets(input) == output);
        checkPostconditions(input);
    }

    SECTION("Distinct sets")
    {
        vecSet input = {{0, 1, 2}, {3, 4, 5}, {99}};

        CHECK(rocRoller::mergeSets(input) == input);
        checkPostconditions(input);
    }

    SECTION("Pairs")
    {
        vecSet input  = {{3, 9}, {2, 36}, {10, 37}, {16, 36}, {17, 37}};
        vecSet output = {{3, 9}, {2, 16, 36}, {10, 17, 37}};

        CHECK(rocRoller::mergeSets(input) == output);
        checkPostconditions(input);
    }

    SECTION("Repeated Pairs")
    {
        vecSet input  = {{3, 9}, {2, 36}, {10, 37}, {16, 36}, {17, 37}, {2, 29}};
        vecSet output = {{3, 9}, {2, 16, 29, 36}, {10, 17, 37}};

        CHECK(rocRoller::mergeSets(input) == output);
        checkPostconditions(input);
    }

    SECTION("Pathological case")
    {
        vecSet input  = {{1, 14},
                        {2, 3},
                        {3, 4},
                        {4, 5},
                        {5, 6},
                        {6, 7},
                        {7, 8},
                        {8, 9},
                        {9, 10},
                        {10, 11},
                        {11, 12},
                        {12, 13},
                        {13, 14}};
        vecSet output = {{1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}};

        CHECK(rocRoller::mergeSets(input) == output);
        checkPostconditions(input);
    }

    SECTION("Random")
    {
        vecSet input  = {{57, 36, 87, 93},
                        {48, 16, 27, 19, 89, 59, 49, 11, 88, 89, 31, 4, 42},
                        {22},
                        {46, 60},
                        {11, 50}};
        vecSet output = {{57, 36, 87, 93},
                         {48, 16, 27, 19, 89, 59, 50, 49, 11, 88, 89, 31, 4, 42},
                         {22},
                         {46, 60}};

        CHECK(rocRoller::mergeSets(input) == output);
        checkPostconditions(input);
    }

    SECTION("Variable random")
    {
        CAPTURE(Catch::getSeed());
        auto iter = GENERATE(range(0, 100));
        DYNAMIC_SECTION(iter)
        {
            rocRoller::RandomGenerator g(Catch::getSeed() + iter);
            auto                       numSets = g.next<int>(2, 30);

            vecSet input;
            for(int i = 0; i < numSets; i++)
            {
                auto numValues = g.next<int>(0, 50);
                auto values    = g.vector<int>(numValues, -100, 100);

                input.emplace_back(values.begin(), values.end());
            }
            CAPTURE(input);

            checkPostconditions(input);
        }
    }
}

TEST_CASE("mergeSets benchmark", "[infrastructure][utils][!benchmark]")
{
    using vecSet = std::vector<std::set<int>>;

    BENCHMARK("Variable random benchmark", i)
    {
        rocRoller::RandomGenerator g(648029 + i);

        auto numSets = 300;

        vecSet input;
        for(int i = 0; i < numSets; i++)
        {
            auto numValues = g.next<int>(0, 50);
            auto values    = g.vector<int>(numValues, -100, 100);

            input.emplace_back(values.begin(), values.end());
        }

        return rocRoller::mergeSets(std::move(input));
    };
}
