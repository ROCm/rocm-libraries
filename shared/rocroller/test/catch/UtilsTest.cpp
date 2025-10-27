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
#include <rocRoller/Utilities/Component.hpp>
#include <rocRoller/Utilities/Random.hpp>
#include <rocRoller/Utilities/Utils.hpp>

#include <catch2/benchmark/catch_benchmark_all.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <omp.h>

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

namespace
{
    using TestArgument = unsigned;

    struct TestComponentBase
    {
        using Argument = std::shared_ptr<TestArgument>;
        static const std::string Basename;
        virtual unsigned         getValue() = 0;
    };

    const std::string TestComponentBase::Basename = "TestComponentBase";

    static_assert(rocRoller::Component::ComponentBase<TestComponentBase>);

    template <unsigned ID>
    struct TestComponent : TestComponentBase
    {
        using Base = TestComponentBase;
        static const std::string Name;

        static bool Match(Argument arg)
        {
            return *arg == ID;
        }

        static std::shared_ptr<TestComponentBase> Build(Argument arg)
        {
            if(!Match(arg))
                return nullptr;
            return std::make_shared<TestComponent<ID>>();
        }

        virtual unsigned getValue() override
        {
            return ID;
        }
    };

    template <unsigned ID>
    const std::string TestComponent<ID>::Name = fmt::format("TestComponent{}", ID);

    static_assert(rocRoller::Component::Component<TestComponent<0>>);
    static_assert(rocRoller::Component::Component<TestComponent<1>>);

    unsigned const COMPONENT_TEST_COMPONENT_COUNT = 100;
    unsigned const COMPONENT_TEST_THREAD_COUNT    = 4;
}

namespace rocRoller::Component
{
    template <unsigned ID>
    struct RegisterTestComponent
    {
        template <typename Factory>
        void operator()(Factory&& factory)
        {
            factory.template registerComponent<TestComponent<ID>>();
            RegisterTestComponent<ID - 1>{}(factory);
        }
    };

    template <>
    struct RegisterTestComponent<0>
    {
        template <typename Factory>
        void operator()(Factory&& factory)
        {
            factory.template registerComponent<TestComponent<0>>();
        }
    };

    template <>
    void ComponentFactory<TestComponentBase>::registerImplementations()
    {
        RegisterTestComponent<COMPONENT_TEST_COMPONENT_COUNT>{}(*this);
    }
}

TEST_CASE("ComponentFactory tests", "[infrastructure][utils]")
{
    using namespace rocRoller::Component;
    using Factory = rocRoller::Component::ComponentFactory<TestComponentBase>;

    SECTION("ComponentFactory single thread test")
    {
        Factory::Instance().emptyCache();
        unsigned expectedResult = 0;
        unsigned actualResult   = 0;

        std::vector<std::shared_ptr<TestArgument>> args{std::make_shared<TestArgument>(0),
                                                        std::make_shared<TestArgument>(1),
                                                        std::make_shared<TestArgument>(2)};

        for(size_t i = 0; i < args.size(); ++i)
        {
            expectedResult += *args[i];
            actualResult += Get<TestComponentBase>(args[i])->getValue();
        }

        CHECK(expectedResult == actualResult);
    }

    SECTION("ComponentFactory concurrent test")
    {
        Factory::Instance().emptyCache();
        unsigned expectedResult = 0;
        unsigned actualResult   = 0;

        std::vector<std::shared_ptr<TestArgument>> args;
        args.reserve(COMPONENT_TEST_COMPONENT_COUNT);

        for(unsigned i = 0; i < COMPONENT_TEST_COMPONENT_COUNT; ++i)
        {
            args.push_back(std::make_shared<TestArgument>(i));
            expectedResult += +i;
        }

        std::array<unsigned, 64> primes{
            2,   3,   5,   7,   11,  13,  17,  19,  23,  29,  31,  37,  41,  43,  47,  53,
            59,  61,  67,  71,  73,  79,  83,  89,  97,  101, 103, 107, 109, 113, 127, 131,
            137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223,
            227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311};
        static_assert(COMPONENT_TEST_THREAD_COUNT < primes.size());

#pragma omp parallel for reduction(+ : actualResult) num_threads(COMPONENT_TEST_THREAD_COUNT)
        for(size_t i = 0; i < COMPONENT_TEST_COMPONENT_COUNT; ++i)
        {
            auto component = Get<TestComponentBase>(args[i]);
            actualResult += component->getValue();

            auto threadId = omp_get_thread_num();
            if((i + 1) % primes[threadId] == 0)
            {
                Factory::Instance().emptyCache();
            }
        }

        CHECK(expectedResult == actualResult);
    }
}
