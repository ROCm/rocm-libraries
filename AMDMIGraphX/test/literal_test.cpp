/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <migraphx/literal.hpp>
#include <migraphx/serialize.hpp>
#include <sstream>
#include <string>
#include "test.hpp"

TEST_CASE(literal_test)
{
    EXPECT(migraphx::literal{1} == migraphx::literal{1});
    EXPECT(migraphx::literal{1} != migraphx::literal{2});
    EXPECT(migraphx::literal{} == migraphx::literal{});
    EXPECT(migraphx::literal{} != migraphx::literal{2});

    migraphx::literal l1{1};
    migraphx::literal l2 = l1; // NOLINT
    EXPECT(l1 == l2);
    EXPECT(l1.at<int>(0) == 1);
    EXPECT(not l1.empty());
    EXPECT(not l2.empty());

    migraphx::literal l3{};
    migraphx::literal l4{};
    EXPECT(l3 == l4);
    EXPECT(l3.empty());
    EXPECT(l4.empty());
}

TEST_CASE(literal_nstd_shape_vector)
{
    migraphx::shape nstd_shape{migraphx::shape::float_type, {1, 3, 2, 2}, {12, 1, 6, 3}};
    std::vector<float> data(12);
    std::iota(data.begin(), data.end(), 0);
    auto l0 = migraphx::literal{nstd_shape, data};

    // check data buffer is read in correctly
    std::vector<float> expected_buffer = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
    const auto* start                  = reinterpret_cast<const float*>(l0.data());
    std::vector<float> l0_data{start, start + 12};
    EXPECT(l0_data == expected_buffer);

    // check that using visit() (that uses a tensor view) gives data in correct order
    std::vector<float> results_vector(12);
    l0.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(results_vector == data);
}

TEST_CASE(literal_os1)
{
    migraphx::literal l{1};
    std::stringstream ss;
    ss << l;
    EXPECT(ss.str() == "1");
}

TEST_CASE(literal_os2)
{
    migraphx::literal l{};
    std::stringstream ss;
    ss << l;
    EXPECT(ss.str().empty());
}

TEST_CASE(literal_os3)
{
    migraphx::shape s{migraphx::shape::int64_type, {3}};
    migraphx::literal l{s, {1, 2, 3}};
    std::stringstream ss;
    ss << l;
    EXPECT(ss.str() == "1, 2, 3");
}

TEST_CASE(literal_visit_at)
{
    migraphx::literal x{1};
    bool visited = false;
    x.visit_at([&](int i) {
        visited = true;
        EXPECT(i == 1);
    });
    EXPECT(visited);
}

TEST_CASE(literal_visit)
{
    migraphx::literal x{1};
    migraphx::literal y{1};
    bool visited = false;
    x.visit([&](auto i) {
        y.visit([&](auto j) {
            visited = true;
            EXPECT(i == j);
        });
    });
    EXPECT(visited);
}

TEST_CASE(literal_visit_all)
{
    migraphx::literal x{1};
    migraphx::literal y{1};
    bool visited = false;
    migraphx::visit_all(x, y)([&](auto i, auto j) {
        visited = true;
        EXPECT(i == j);
    });
    EXPECT(visited);
}

TEST_CASE(literal_visit_mismatch_shape)
{
    migraphx::literal x{1};
    migraphx::shape s{migraphx::shape::int64_type, {3}};
    migraphx::literal y{s, {1, 2, 3}};
    bool visited = false;
    x.visit([&](auto i) {
        y.visit([&](auto j) {
            visited = true;
            EXPECT(i != j);
        });
    });
    EXPECT(visited);
}

TEST_CASE(literal_visit_all_mismatch_type)
{
    migraphx::shape s1{migraphx::shape::int32_type, {1}};
    migraphx::literal x{s1, {1}};
    migraphx::shape s2{migraphx::shape::int8_type, {1}};
    migraphx::literal y{s2, {1}};
    EXPECT(
        test::throws<migraphx::exception>([&] { migraphx::visit_all(x, y)([&](auto, auto) {}); }));
}

TEST_CASE(literal_visit_empty)
{
    migraphx::literal x{};
    EXPECT(test::throws([&] { x.visit([](auto) {}); }));
    EXPECT(test::throws([&] { x.visit_at([](auto) {}); }));
}

TEST_CASE(value_literal)
{
    migraphx::shape s{migraphx::shape::int64_type, {3}};
    migraphx::literal l1{s, {1, 2, 3}};
    auto v1 = migraphx::to_value(l1);
    migraphx::literal l2{1};
    auto v2 = migraphx::to_value(l2);
    EXPECT(v1 != v2);

    auto l3 = migraphx::from_value<migraphx::literal>(v1);
    EXPECT(l3 == l1);
    auto l4 = migraphx::from_value<migraphx::literal>(v2);
    EXPECT(l4 == l2);
}

TEST_CASE(literal_to_string_float_precision)
{
    migraphx::literal x{126.99993142003703f};
    EXPECT(x.to_string() != "127");
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
