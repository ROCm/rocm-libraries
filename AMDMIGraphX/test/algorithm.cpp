/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */
#include <migraphx/algorithm.hpp>
#include <forward_list>
#include <list>
#include <functional>
#include <test.hpp>

// NOLINTNEXTLINE
#define MIGRAPHX_FORWARD_CONTAINER_TEST_CASE(name, type) \
    template <class Container>                           \
    static void name();                                  \
    TEST_CASE_REGISTER(name<std::vector<type>>);         \
    TEST_CASE_REGISTER(name<std::list<type>>);           \
    TEST_CASE_REGISTER(name<std::forward_list<type>>);   \
    template <class Container>                           \
    void name()

template <class Container, class Iterator>
static auto
erase_iterator(Container& c, Iterator pos, Iterator last) -> decltype(c.erase_after(pos, last))
{
    auto n  = std::distance(c.begin(), pos);
    auto it = n == 0 ? c.before_begin() : std::next(c.begin(), n - 1);
    return c.erase_after(it, last);
}

template <class Container, class Iterator>
static auto erase_iterator(Container& c, Iterator pos, Iterator last) -> decltype(c.erase(pos,
                                                                                          last))
{
    return c.erase(pos, last);
}

MIGRAPHX_FORWARD_CONTAINER_TEST_CASE(adjacent_remove_if1, int)
{
    Container v = {0, 1, 1, 1, 4, 2, 2, 4, 2};
    erase_iterator(v, migraphx::adjacent_remove_if(v.begin(), v.end(), std::equal_to<>{}), v.end());
    EXPECT(v == Container{0, 1, 4, 2, 4, 2});
}

MIGRAPHX_FORWARD_CONTAINER_TEST_CASE(adjacent_remove_if2, int)
{
    Container v = {0, 1, 1, 1, 4, 2, 2, 4, 2, 5, 5};
    erase_iterator(v, migraphx::adjacent_remove_if(v.begin(), v.end(), std::equal_to<>{}), v.end());
    EXPECT(v == Container{0, 1, 4, 2, 4, 2, 5});
}

MIGRAPHX_FORWARD_CONTAINER_TEST_CASE(adjacent_remove_if3, int)
{
    Container v = {0, 1, 1, 1, 4, 2, 2, 4, 2, 5, 5, 6};
    erase_iterator(v, migraphx::adjacent_remove_if(v.begin(), v.end(), std::equal_to<>{}), v.end());
    EXPECT(v == Container{0, 1, 4, 2, 4, 2, 5, 6});
}

MIGRAPHX_FORWARD_CONTAINER_TEST_CASE(adjacent_remove_if_non_equivalence, int)
{
    Container v = {0, 1, 1, 1, 4, 2, 2, 3, 4, 2, 5, 5, 6};
    auto pred   = [](int a, int b) { return (b - a) == 1; };
    erase_iterator(v, migraphx::adjacent_remove_if(v.begin(), v.end(), pred), v.end());
    EXPECT(v == Container{1, 1, 1, 4, 2, 4, 2, 5, 6});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
