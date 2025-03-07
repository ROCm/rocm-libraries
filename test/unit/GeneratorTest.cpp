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

#include <iterator>

#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Generator.hpp>

#include "SimpleFixture.hpp"

namespace rocRollerTest
{
    using namespace rocRoller;
    static_assert(std::input_iterator<typename Generator<int>::iterator>);
    static_assert(std::ranges::input_range<Generator<int>>);
    static_assert(CInputRangeOf<Generator<int>, int>);

    class GeneratorTest : public SimpleFixture
    {
    };

    template <typename T>
    rocRoller::Generator<T> Take(size_t n, Generator<T> gen)
    {
        auto it = gen.begin();
        for(size_t i = 0; i < n && it != gen.end(); ++i, ++it)
        {
            co_yield *it;
        }
    }

    template <std::integral T>
    Generator<T> range(T begin, T end, T inc)
    {
        for(T val = begin; val < end; val += inc)
        {
            co_yield val;
        }
    }

    template <std::integral T>
    Generator<T> range(T begin, T end)
    {
        co_yield range<T>(begin, end, 1);
    }

    template <std::integral T>
    Generator<T> range(T end)
    {
        co_yield range<T>(0, end);
    }

    template <std::integral T>
    Generator<T> range()
    {
        for(T i = 0; true; i++)
            co_yield i;
    }

    template <std::integral T>
    rocRoller::Generator<T> fibonacci()
    {
        T a = 1;
        T b = 1;
        co_yield a;
        co_yield b;

        while(true)
        {
            a = a + b;
            co_yield a;

            b = a + b;
            co_yield b;
        }
    }

    TEST_F(GeneratorTest, Fibonacci)
    {
        auto fibs = fibonacci<int>();
        auto iter = fibs.begin();

        EXPECT_EQ(1, *iter);
        ++iter;
        EXPECT_EQ(1, *iter);
        ++iter;
        EXPECT_EQ(2, *iter);
        ++iter;
        EXPECT_EQ(3, *iter);
        ++iter;
        EXPECT_EQ(5, *iter);
        ++iter;
        EXPECT_EQ(8, *iter);
        ++iter;
        EXPECT_EQ(13, *iter);
    }

    // Overload of yield_value in Generator::promise_type allows yielding a sequence directly.
    // Equivalent to the following code, but more efficient.
    // for(auto item: seq)
    //     co_yield item
    // Analogous to Python's `yield from` statement.
    template <typename T>
    Generator<T> identity(Generator<T> seq)
    {
        co_yield std::move(seq);
    }

    TEST_F(GeneratorTest, YieldFromGenerator)
    {
        auto fib1 = fibonacci<int>();
        auto fib2 = identity(fibonacci<int>());

        auto iter1 = fib1.begin();
        auto iter2 = fib2.begin();

        for(int i = 0; i < 10; ++i, iter1++, iter2++)
            EXPECT_EQ(*iter1, *iter2) << i;
    }

    TEST_F(GeneratorTest, YieldFromContainer)
    {
        auto foo = []() -> Generator<int> {
            std::vector<int> x;

            co_yield x;
            co_yield x;

            co_yield 5;

            x.push_back(1);
            co_yield x;
            AssertFatal(x.size() == 1, ShowValue(x.size()));

            x.push_back(2);
            co_yield x;
            AssertFatal(x.size() == 2, x.size());

            // It's not required for the implementation of Generator
            // to clear x here but the current implementation should.
            co_yield std::move(x);
            AssertFatal(x.size() == 0, x.size());

            x.push_back(29);

            co_yield x;

            co_yield Take(5, fibonacci<int>()).to<std::set>();

            co_yield {9, 8, 7, 6};

            co_yield std::vector<int>();
        };

        std::vector expected{5, 1, 1, 2, 1, 2, 29, 1, 2, 3, 5, 9, 8, 7, 6};

        EXPECT_EQ(expected, foo().to<std::vector>());

        {
            auto gen  = foo();
            auto iter = gen.begin();
            for(size_t i = 0; i < expected.size(); i++)
            {
                if(i % 3 == 0)
                    EXPECT_EQ(expected[i], *iter) << i;
                ++iter;
            }
        }
    }

    TEST_F(GeneratorTest, YieldFromSkippingDereference)
    {
        auto bar = []() -> Generator<int> {
            co_yield range<int>(5);
            co_yield range<int>(5, 7);
            co_yield range<int>(0);
            co_yield range<int>(0);
            co_yield 7;
            co_yield range<int>(0);
            co_yield range(8, 10);
            co_yield {10, 11, 12};
            co_yield range(13, 15);

            co_yield 15;
            co_yield std::vector<int>{};
            co_yield range<int>(16, 18);
            co_yield std::vector<int>{};
            co_yield std::vector<int>{};
            co_yield std::vector<int>{};
            co_yield 18;
            co_yield range<int>(19, 30);
        };
        const int rangeLimit = 30;
        ASSERT_EQ(range<int>(rangeLimit).to<std::vector>(), bar().to<std::vector>());

        for(int denominator = 1; denominator < rangeLimit; denominator++)
        {
            for(int remainder = 0; remainder < denominator; remainder++)
            {
                auto gen  = bar();
                auto iter = gen.begin();

                for(size_t i = 0; i < rangeLimit; i++)
                {
                    if(i % denominator == remainder)
                        EXPECT_EQ(i, *iter)
                            << ShowValue(denominator) << ShowValue(remainder) << ShowValue(i);
                    ++iter;
                }
                ++iter;
            }
        }

        auto gen  = bar();
        auto iter = gen.begin();
        for(size_t i = 0; i < rangeLimit; i++)
            ++iter;

        EXPECT_TRUE(iter == gen.end());
    }

    TEST_F(GeneratorTest, StateEnum)
    {
        for(int i = 0; i < static_cast<int>(GeneratorState::Count); i++)
        {
            auto enumVal = static_cast<GeneratorState>(i);
            EXPECT_EQ(enumVal, fromString<GeneratorState>(toString(enumVal)));
        }
    }

    TEST_F(GeneratorTest, Assignment)
    {
        auto fib = fibonacci<int>();

        EXPECT_EQ(GeneratorState::NoValue, fib.state());

        auto iter = fib.begin();

        EXPECT_EQ(GeneratorState::NoValue, fib.state());

        EXPECT_TRUE(iter != fib.end());

        EXPECT_EQ(GeneratorState::HasValue, fib.state());

        EXPECT_EQ(1, *iter);
        EXPECT_EQ(GeneratorState::HasValue, fib.state());
        ++iter;

        EXPECT_EQ(GeneratorState::NoValue, fib.state());

        EXPECT_EQ(1, *iter);
        ++iter;

        EXPECT_EQ(2, *iter);
        ++iter;
        EXPECT_EQ(3, *iter);

        fib = fibonacci<int>().take(4);

        iter = fib.begin();

        EXPECT_EQ(1, *iter);
        ++iter;

        EXPECT_EQ(1, *iter);
        ++iter;

        EXPECT_EQ(2, *iter);
        ++iter;

        EXPECT_EQ(3, *iter);
        ++iter;
        EXPECT_TRUE(iter == fib.end());
        EXPECT_EQ(GeneratorState::Done, fib.state());
        EXPECT_THROW(std::ignore = *iter, std::runtime_error);
    }

    TEST_F(GeneratorTest, Ranges)
    {
        auto             r = range(0, 5, 1);
        std::vector<int> v(r.begin(), r.end());
        std::vector<int> v2({0, 1, 2, 3, 4});
        EXPECT_EQ(v2, v);

        r  = range(1, 5);
        v  = std::vector(r.begin(), r.end());
        v2 = {1, 2, 3, 4};
        EXPECT_EQ(v2, v);

        r  = range(7);
        v  = std::vector(r.begin(), r.end());
        v2 = {0, 1, 2, 3, 4, 5, 6};
        EXPECT_EQ(v2, v);
    }

    TEST_F(GeneratorTest, IteratorSemantics)
    {
        auto fibs = fibonacci<int>();

        EXPECT_EQ(GeneratorState::NoValue, fibs.state());

        auto iter1 = fibs.begin();

        EXPECT_EQ(GeneratorState::NoValue, fibs.state());

        EXPECT_EQ(1, *iter1);

        EXPECT_EQ(GeneratorState::HasValue, fibs.state());

        ++iter1;
        EXPECT_EQ(GeneratorState::NoValue, fibs.state());

        ++iter1;
        EXPECT_EQ(2, *iter1);

        // Calling .begin() on the generator should not advance it.
        auto iter2 = fibs.begin();
        EXPECT_EQ(2, *iter2);

        auto iter3 = iter2++;

        EXPECT_EQ(3, *iter2);
        EXPECT_EQ(3, *iter1);
        EXPECT_EQ(2, *iter3);

        ++iter2;
        EXPECT_EQ(5, *iter2);
        EXPECT_EQ(5, *iter1);
        EXPECT_EQ(2, *iter3);

        auto iter4 = iter3++;

        EXPECT_EQ(2, *iter3);
        EXPECT_EQ(2, *iter4);

        ++iter4;

        EXPECT_THROW(*iter4, std::runtime_error);
    }

    /**
     * Yields the lowest value from the front of each generator.
     */
    template <std::integral T>
    Generator<T> MergeLess(std::vector<Generator<T>>&& gens)
    {
        if(gens.empty())
            co_return;
        std::vector<Generator<int>::iterator> iters;
        iters.reserve(gens.size());
        for(auto& g : gens)
        {
            // cppcheck-suppress useStlAlgorithm
            iters.push_back(g.begin());
        }

        bool any = true;
        while(any)
        {
            auto   theVal = std::numeric_limits<T>::max();
            size_t theIdx = 0;

            any = false;
            for(size_t i = 0; i < iters.size(); i++)
            {
                if(iters[i] != gens[i].end() && *iters[i] < theVal)
                {
                    theVal = *iters[i];
                    theIdx = i;
                    any    = true;
                }
            }

            if(any)
            {
                co_yield theVal;
                ++iters[theIdx];
            }
        }
    }

    TEST_F(GeneratorTest, MergeLess)
    {
        std::vector<Generator<int>> gens;
        gens.push_back(std::move(range(5)));
        gens.push_back(Take(5, fibonacci<int>()));

        auto ref = std::vector{0, 1, 1, 1, 2, 2, 3, 3, 4, 5};

        auto             g2 = MergeLess(std::move(gens));
        std::vector<int> v(g2.begin(), g2.end());
        EXPECT_EQ(ref, v);
    }

    TEST_F(GeneratorTest, ToContainer)
    {
        auto g = range(0, 5, 1);

        auto r = g.to<std::vector>();

        std::vector<int> v = {0, 1, 2, 3, 4};

        auto s = range(0, 5, 1).to<std::set>();

        EXPECT_EQ(r, v);
        EXPECT_EQ(s, std::set<int>(v.begin(), v.end()));

        auto fibs  = fibonacci<int>();
        auto fibs1 = Take(5, std::move(fibs)).to<std::vector>();

        std::vector<int> fibsE1 = {1, 1, 2, 3, 5};
        EXPECT_EQ(fibs1, fibsE1);
    }

    Generator<int> testWithRef(std::vector<int> const& v)
    {
        for(int i = 0; i < v.size(); i++)
        {
            co_yield v[i];
        }
    }

    // cppcheck-suppress passedByValue
    Generator<int> testWithoutRef(std::vector<int> v)
    {
        for(int i = 0; i < v.size(); i++)
        {
            co_yield v[i];
        }
    }

    TEST_F(GeneratorTest, TestVectorParams)
    {
        std::vector<int> a = {0, 1, 2};
        // Vectors as separate values
        {
            for(int i : testWithRef(a))
            {
                EXPECT_EQ(i, a.at(i));
            }

            for(int i : testWithoutRef(a))
            {
                EXPECT_EQ(i, a.at(i));
            }
        }

        // Vectors constructed in place
        {
            // FAILS on GCC
            // for(int i : testWithRef(std::vector<int>{0, 1, 2}))
            // {
            //     EXPECT_EQ(i, a.at(i));
            // }

            for(int i : testWithoutRef(std::vector<int>{0, 1, 2}))
            {
                EXPECT_EQ(i, a.at(i));
            }
        }

        // Initializer lists
        {
            // FAILS on GCC
            // for(int i : testWithRef({0, 1, 2}))
            // {
            //     EXPECT_EQ(i, a.at(i));
            // }

            for(int i : testWithoutRef({0, 1, 2}))
            {
                EXPECT_EQ(i, a.at(i));
            }
        }
    }

    TEST_F(GeneratorTest, GeneratorFilter)
    {
        auto func = []() -> Generator<int> {
            co_yield 3;
            co_yield 2;
            co_yield 9;
            co_yield 17;
            co_yield 4;
            co_yield 4;
        };

        EXPECT_EQ((std::vector{9, 17, 4, 4}),
                  filter([](int x) { return x > 3; }, func()).to<std::vector>());

        EXPECT_EQ((std::vector{3, 2}),
                  filter([](int x) { return x < 4; }, func()).to<std::vector>());

        EXPECT_EQ((std::vector<int>{}),
                  filter([](int x) { return x < 2; }, func()).to<std::vector>());

        EXPECT_EQ(
            (std::vector<int>{2, 8, 34, 144, 610, 2584}),
            filter([](int x) { return x % 2 == 0; }, Take(20, fibonacci<int>())).to<std::vector>());

        EXPECT_EQ(
            (std::vector<int>{1, 1, 3, 5, 13, 21, 55, 89, 233, 377, 987, 1597, 4181, 6765}),
            filter([](int x) { return x % 2 != 0; }, Take(20, fibonacci<int>())).to<std::vector>());
    }

    TEST_F(GeneratorTest, ConvenienceFunctions)
    {
        auto isEven     = [](auto x) { return x % 2 == 0; };
        auto isOdd      = [](auto x) { return x % 2 != 0; };
        auto isPositive = [](auto x) { return x > 0; };
        auto isNegative = [](auto x) { return x < 0; };

        auto square = [](auto x) { return x * x; };

        auto tenSquaresOfOdds = range<int>().filter(isOdd).map(square).take(10).to<std::vector>();

        std::vector expected = {1, 9, 25, 49, 81, 121, 169, 225, 289, 361};

        EXPECT_EQ(expected, tenSquaresOfOdds);

        auto fiveSquaresOfEvens
            = take(5, filter(isPositive, map(square, filter(isEven, range<int>()))))
                  .to<std::vector>();
        expected = {4, 16, 36, 64, 100};
        EXPECT_EQ(expected, fiveSquaresOfEvens);

        std::vector<int> threeSquares;
        auto             gen = range<int>().filter(isPositive).take(3);
        std::transform(gen.begin(), gen.end(), std::back_insert_iterator(threeSquares), square);

        expected = {1, 4, 9};
        EXPECT_EQ(expected, threeSquares);

        EXPECT_FALSE(empty(range<int>()));
        EXPECT_FALSE(range<int>().empty());

        EXPECT_TRUE(range(100).filter(isNegative).empty());
        EXPECT_TRUE(empty(range(100).filter(isNegative)));

        EXPECT_FALSE(only(range<int>()));
        EXPECT_FALSE(range<int>().only());
        EXPECT_FALSE(range(100).filter(isNegative).only());
        EXPECT_FALSE(only(range(100).filter(isNegative)));

        EXPECT_EQ(1, range(1, 2).only());
        EXPECT_EQ(1, only(range(1, 2)));
    }

    enum class ExceptionOrder
    {
        Beginning,
        Middle,
        End,
        Count
    };

    std::ostream& operator<<(std::ostream& stream, ExceptionOrder order)
    {
        switch(order)
        {
        case ExceptionOrder::Beginning:
            return stream << "Beginning";
        case ExceptionOrder::Middle:
            return stream << "Middle";
        case ExceptionOrder::End:
            return stream << "End";
        default:
        case ExceptionOrder::Count:
            return stream << "INVALID";
        }
    }

    class GeneratorExceptionTest
        : public ::testing::TestWithParam<std::tuple<ExceptionOrder, ExceptionOrder>>
    {
    };

    TEST_P(GeneratorExceptionTest, GeneratorExceptions)
    {
        auto genThatThrows = [&]() -> Generator<int> {
            auto throwOrder = std::get<0>(GetParam());
            if(throwOrder != ExceptionOrder::Beginning)
                co_yield 5;

            throw std::runtime_error("Foo");

            if(throwOrder != ExceptionOrder::End)
                co_yield 6;
        };

        auto genThatYields = [&]() -> Generator<int> {
            auto yieldOrder = std::get<1>(GetParam());
            if(yieldOrder != ExceptionOrder::Beginning)
                co_yield 7;

            co_yield genThatThrows();

            if(yieldOrder != ExceptionOrder::End)
                co_yield 8;
        };

        auto callThrowDirectly = [&]() { auto x = genThatThrows().to<std::vector>(); };
        auto callYieldFrom     = [&]() { auto x = genThatYields().to<std::vector>(); };

        auto iterateThroughDirectly = [&]() {
            auto gen  = genThatThrows();
            auto iter = gen.begin();
            for(int i = 0; i < 2; ++i)
                ++iter;
        };

        auto iterateThroughFrom = [&]() {
            auto gen  = genThatYields();
            auto iter = gen.begin();
            for(int i = 0; i < 4; ++i)
                ++iter;
        };

        EXPECT_THROW(callThrowDirectly(), std::runtime_error);
        EXPECT_THROW(callYieldFrom(), std::runtime_error);
        EXPECT_THROW(iterateThroughDirectly(), std::runtime_error);
        EXPECT_THROW(iterateThroughFrom(), std::runtime_error);
    }

    INSTANTIATE_TEST_SUITE_P(GeneratorTest,
                             GeneratorExceptionTest,
                             ::testing::Combine(::testing::Values(ExceptionOrder::Beginning,
                                                                  ExceptionOrder::Middle,
                                                                  ExceptionOrder::End),
                                                ::testing::Values(ExceptionOrder::Beginning,
                                                                  ExceptionOrder::Middle,
                                                                  ExceptionOrder::End)));
}
