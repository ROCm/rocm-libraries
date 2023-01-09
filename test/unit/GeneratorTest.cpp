
#include <gtest/gtest.h>

#include <iterator>

#include <rocRoller/Utilities/Generator.hpp>

namespace rocRollerTest
{
    static_assert(std::input_iterator<typename rocRoller::Generator<int>::iterator>);
    static_assert(std::ranges::input_range<rocRoller::Generator<int>>);

    using namespace rocRoller;
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

    TEST(GeneratorTest, Fibonacci)
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
        co_yield seq;
    }

    TEST(GeneratorTest, YieldFrom)
    {
        auto fib1 = fibonacci<int>();
        auto fib2 = identity(std::move(fibonacci<int>()));

        auto iter1 = fib1.begin();
        auto iter2 = fib2.begin();

        for(int i = 0; i < 10; ++i, iter1++, iter2++)
            EXPECT_EQ(*iter1, *iter2);
    }

    TEST(GeneratorTest, Assignment)
    {
        auto fib = fibonacci<int>();

        auto iter = fib.begin();

        EXPECT_EQ(1, *iter);
        ++iter;

        EXPECT_EQ(1, *iter);
        ++iter;

        EXPECT_EQ(2, *iter);
        ++iter;
        EXPECT_EQ(3, *iter);

        fib = fibonacci<int>();

        iter = fib.begin();

        EXPECT_EQ(1, *iter);
        ++iter;

        EXPECT_EQ(1, *iter);
        ++iter;

        EXPECT_EQ(2, *iter);
        ++iter;

        EXPECT_EQ(3, *iter);
    }

    template <std::integral T>
    Generator<T> range(T begin, T end, T inc)
    {
        //assert(std::sign(inc) == std::sign(end - begin));
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

    TEST(GeneratorTest, Ranges)
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

    /**
     * Yields the first `n` values from `gen`
     */
    template <typename T>
    Generator<T> Take(size_t n, Generator<T> gen)
    {
        auto it = gen.begin();
        for(size_t i = 0; i < n && it != gen.end(); ++i, ++it)
        {
            co_yield *it;
        }
    }

    TEST(GeneratorTest, MergeLess)
    {
        std::vector<Generator<int>> gens;
        gens.push_back(std::move(range(5)));
        gens.push_back(Take(5, fibonacci<int>()));

        auto ref = std::vector{0, 1, 1, 1, 2, 2, 3, 3, 4, 5};

        auto             g2 = MergeLess(std::move(gens));
        std::vector<int> v(g2.begin(), g2.end());
        EXPECT_EQ(ref, v);
    }

    Generator<int> Throws()
    {
        co_yield 5;
        throw std::runtime_error("Foo");
    }

    TEST(GeneratorTest, HandlesExceptions)
    {
        auto func = [&]() {
            std::vector<int> values;
            for(auto v : Throws())
            {
                // cppcheck-suppress useStlAlgorithm
                values.push_back(v);
            }
        };

        EXPECT_THROW(func(), std::runtime_error);
    }

    TEST(GeneratorTest, ToContainer)
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

    TEST(GeneratorTest, TestVectorParams)
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

    TEST(GeneratorTest, GeneratorFilter)
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
}
