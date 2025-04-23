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
 */
#include <migraphx/reduce_dims.hpp>
#include <migraphx/permutation.hpp>
#include <migraphx/ranges.hpp>
#include "test.hpp"

static migraphx::shape make_shape(std::vector<std::size_t> lens)
{
    return {migraphx::shape::float_type, std::move(lens)};
}

static migraphx::shape make_shape(std::vector<std::size_t> lens, std::vector<std::size_t> strides)
{
    return {migraphx::shape::float_type, std::move(lens), std::move(strides)};
}

static bool verify_shape(const migraphx::shape& s1, const migraphx::shape& s2)
{
    if(s1.elements() != s2.elements())
        return false;
    return migraphx::all_of(migraphx::range(s1.elements()),
                            [&](auto i) { return s1.index(i) == s2.index(i); });
}

template <class Range1, class Range2>
static bool verify_shapes(const Range1& r1, const Range2& r2)
{
    return migraphx::equal(
        r1, r2, [](const auto& s1, const auto& s2) { return verify_shape(s1, s2); });
}

TEST_CASE(same_standard)
{
    auto is                              = make_shape({64, 3, 7, 7});
    auto os                              = make_shape({64 * 3 * 7 * 7});
    std::vector<migraphx::shape> ishapes = {is, is, is};
    std::vector<migraphx::shape> eshapes = {os, os, os};
    auto rshapes                         = migraphx::reduce_dims(ishapes);
    EXPECT(verify_shapes(ishapes, rshapes));
    EXPECT(eshapes == rshapes);
}

TEST_CASE(same_broadcast1)
{
    auto is                              = make_shape({64, 3, 7, 7});
    auto os                              = make_shape({64, 3, 7 * 7});
    std::vector<migraphx::shape> ishapes = {is, make_shape({64, 3, 7, 7}, {0, 1, 0, 0}), is};
    std::vector<migraphx::shape> eshapes = {os, make_shape({64, 3, 7 * 7}, {0, 1, 0}), os};
    auto rshapes                         = migraphx::reduce_dims(ishapes);
    EXPECT(verify_shapes(ishapes, rshapes));
    EXPECT(eshapes == rshapes);
}

TEST_CASE(same_broadcast2)
{
    auto is                              = make_shape({64, 3, 8, 7, 7});
    auto os                              = make_shape({64, 8 * 3, 7 * 7});
    std::vector<migraphx::shape> ishapes = {is, make_shape({64, 3, 8, 7, 7}, {0, 8, 1, 0, 0}), is};
    std::vector<migraphx::shape> eshapes = {os, make_shape({64, 8 * 3, 7 * 7}, {0, 1, 0}), os};
    auto rshapes                         = migraphx::reduce_dims(ishapes);
    EXPECT(verify_shapes(ishapes, rshapes));
    EXPECT(eshapes == rshapes);
}

TEST_CASE(same_transposed)
{
    auto is                              = make_shape({64, 3, 7, 7});
    auto os                              = make_shape({64 * 3, 7, 7});
    std::vector<migraphx::shape> ishapes = {is, migraphx::reorder_shape(is, {0, 1, 3, 2}), is};
    std::vector<migraphx::shape> eshapes = {os, migraphx::reorder_shape(os, {0, 2, 1}), os};
    auto rshapes                         = migraphx::reduce_dims(ishapes);
    EXPECT(verify_shapes(ishapes, rshapes));
    EXPECT(eshapes == rshapes);
}

TEST_CASE(different_masked1)
{
    auto is                              = make_shape({64, 3, 7, 7});
    auto os                              = make_shape({64, 3, 7 * 7});
    std::vector<migraphx::shape> ishapes = {is, make_shape({1, 3, 1, 1}), is};
    std::vector<migraphx::shape> eshapes = {os, make_shape({1, 3, 1}), os};
    auto rshapes                         = migraphx::reduce_dims(ishapes);
    EXPECT(verify_shapes(ishapes, rshapes));
    EXPECT(eshapes == rshapes);
}

TEST_CASE(different_masked2)
{
    auto is                              = make_shape({64, 3, 7, 7});
    auto os                              = make_shape({64, 3, 7 * 7});
    std::vector<migraphx::shape> ishapes = {
        is, make_shape({1, 3, 1, 1}), make_shape({64, 1, 7, 7})};
    std::vector<migraphx::shape> eshapes = {os, make_shape({1, 3, 1}), make_shape({64, 1, 7 * 7})};
    auto rshapes                         = migraphx::reduce_dims(ishapes);
    EXPECT(verify_shapes(ishapes, rshapes));
    EXPECT(eshapes == rshapes);
}

TEST_CASE(different_incompatible)
{
    auto is                              = make_shape({64, 3, 7, 7});
    std::vector<migraphx::shape> ishapes = {is, make_shape({1, 3, 2, 1}), is};
    auto rshapes                         = migraphx::reduce_dims(ishapes);

    EXPECT(ishapes == rshapes);
}

TEST_CASE(different_ranks)
{
    auto is                              = make_shape({64, 3, 7, 7});
    std::vector<migraphx::shape> ishapes = {is, make_shape({1, 3}), is};
    auto rshapes                         = migraphx::reduce_dims(ishapes);

    EXPECT(ishapes == rshapes);
}

TEST_CASE(transposed1)
{
    std::vector<migraphx::shape> ishapes = {
        make_shape({8, 28, 4, 56, 56}),
        make_shape({8, 28, 4, 56, 56}, {351232, 3136, 87808, 56, 1})};
    std::vector<migraphx::shape> eshapes = {
        make_shape({8, 28, 4, 56 * 56}), make_shape({8, 28, 4, 56 * 56}, {351232, 3136, 87808, 1})};
    auto rshapes = migraphx::reduce_dims(ishapes);
    EXPECT(verify_shapes(ishapes, rshapes));
    EXPECT(eshapes == rshapes);
}

TEST_CASE(non_packed_empty1)
{
    std::vector<migraphx::shape> ishapes = {make_shape({1, 12}, {589824, 64})};
    std::vector<migraphx::shape> eshapes = {make_shape({12}, {64})};
    auto rshapes                         = migraphx::reduce_dims(ishapes);
    EXPECT(verify_shapes(ishapes, rshapes));
    EXPECT(eshapes == rshapes);
}

TEST_CASE(non_packed_empty2)
{
    std::vector<migraphx::shape> ishapes = {make_shape({12, 1}, {64, 589824})};
    std::vector<migraphx::shape> eshapes = {make_shape({12}, {64})};
    auto rshapes                         = migraphx::reduce_dims(ishapes);
    EXPECT(verify_shapes(ishapes, rshapes));
    EXPECT(eshapes == rshapes);
}

TEST_CASE(single_dim)
{
    std::vector<migraphx::shape> ishapes = {make_shape({1}, {1})};
    auto rshapes                         = migraphx::reduce_dims(ishapes);
    EXPECT(ishapes == rshapes);
}

TEST_CASE(step_broadcast_transpose)
{
    std::vector<migraphx::shape> ishapes = {make_shape({1, 2, 2, 1}, {0, 0, 3, 6}),
                                            make_shape({1, 2, 2, 1}, {4, 2, 1, 1})};
    std::vector<migraphx::shape> eshapes = {make_shape({2, 2}, {0, 3}), make_shape({2, 2}, {2, 1})};
    auto rshapes                         = migraphx::reduce_dims(ishapes);
    EXPECT(verify_shapes(ishapes, rshapes));
    EXPECT(eshapes == rshapes);
}

TEST_CASE(empty)
{
    auto rshapes = migraphx::reduce_dims({});
    EXPECT(rshapes.empty());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
