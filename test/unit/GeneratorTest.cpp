
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

        fib = fibonacci<int>();

        iter = fib.begin();

        EXPECT_EQ(1, *iter);
        ++iter;

        EXPECT_EQ(1, *iter);
        ++iter;

        EXPECT_EQ(2, *iter);
        ++iter;
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

        v  = {};
        r  = range(1, 5);
        v  = std::vector(r.begin(), r.end());
        v2 = {1, 2, 3, 4};
        EXPECT_EQ(v2, v);

        v  = {};
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
            iters.push_back(g.begin());

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
            co_yield *it;
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
                values.push_back(v);
        };

        EXPECT_THROW(func(), std::runtime_error);
    }

}
