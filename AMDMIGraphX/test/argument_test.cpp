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
#include <migraphx/argument.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/serialize.hpp>
#include <sstream>
#include <string>
#include <utility>
#include "test.hpp"

static migraphx::argument as_argument(migraphx::argument a) { return a; }
template <class T>
static migraphx::argument as_argument(T x)
{
    return migraphx::literal{x}.get_argument();
}
template <class... Ts>
static migraphx::argument make_tuple(Ts... xs)
{
    return migraphx::argument{{as_argument(std::move(xs))...}};
}

TEST_CASE(copy_eq)
{
    auto a1 = as_argument(3);
    auto a2 = as_argument(3);
    auto a3 = as_argument(1);
    auto a4 = a1; // NOLINT

    EXPECT(a1 == a2);
    EXPECT(a2 != a3);
    EXPECT(a1 == a4);
    EXPECT(a4 != a3);

    EXPECT(a1.get_sub_objects().empty());
    EXPECT(a2.get_sub_objects().empty());
    EXPECT(a3.get_sub_objects().empty());
    EXPECT(a4.get_sub_objects().empty());
}

TEST_CASE(default_construct)
{
    migraphx::argument a1{};
    migraphx::argument a2{};

    EXPECT(a1.empty());
    EXPECT(a2.empty());
    EXPECT(a1 == a2);

    EXPECT(a1.to_string().empty());
    EXPECT(a2.to_string().empty());

    EXPECT(a1.get_sub_objects().empty());
    EXPECT(a2.get_sub_objects().empty());
}

TEST_CASE(string_elems)
{
    migraphx::shape s{migraphx::shape::int64_type, {3}};
    migraphx::literal l{s, {1, 2, 3}};
    auto a = l.get_argument();

    EXPECT(a.to_string() == "1, 2, 3");
}

TEST_CASE(tuple)
{
    auto a1 = make_tuple(3, 3.0);

    EXPECT(a1.get_shape().type() == migraphx::shape::tuple_type);
    EXPECT(a1.get_sub_objects().size() == 2);
    EXPECT(a1.get_sub_objects()[0] == as_argument(3));
    EXPECT(a1.get_sub_objects()[1] == as_argument(3.0));

    auto a2 = make_tuple(3, 3.0);

    EXPECT(a1 == a2);
    EXPECT(a1.to_string() == a2.to_string());

    auto a3 = make_tuple(3, 4.0);
    EXPECT(a1 != a3);
    EXPECT(a1.to_string() != a3.to_string());
}

TEST_CASE(nested_tuple)
{
    auto a1 = make_tuple(3, make_tuple(5, 4));

    EXPECT(a1.get_shape().type() == migraphx::shape::tuple_type);
    EXPECT(a1.get_sub_objects().size() == 2);
    EXPECT(a1.get_sub_objects()[0] == as_argument(3));
    EXPECT(a1.get_sub_objects()[1] == make_tuple(5, 4));

    auto a2 = make_tuple(3, make_tuple(5, 4));

    EXPECT(a1 == a2);
    EXPECT(a1.to_string() == a2.to_string());

    auto a3 = make_tuple(3, make_tuple(5, 6));
    EXPECT(a1 != a3);
    EXPECT(a1.to_string() != a3.to_string());
}

TEST_CASE(tuple_construct)
{
    migraphx::shape s{{migraphx::shape{migraphx::shape::float_type, {4}},
                       migraphx::shape{migraphx::shape::int8_type, {3}}}};
    migraphx::argument a{s};
    EXPECT(a.get_sub_objects().size() == 2);
    EXPECT(a.get_shape() == s);

    auto b = a; // NOLINT
    EXPECT(a.get_shape() == b.get_shape());
    EXPECT(a.get_sub_objects().size() == 2);
    EXPECT(a.get_sub_objects()[0] == b.get_sub_objects()[0]);
    EXPECT(a.get_sub_objects()[1] == b.get_sub_objects()[1]);
    EXPECT(a == b);
}

TEST_CASE(tuple_visit)
{
    auto a1 = make_tuple(3, 3.0);
    EXPECT(test::throws([&] { a1.visit([](auto&&) {}); }));
    EXPECT(test::throws([&] { a1.at<float>(); }));

    bool reaches = false;
    a1.visit([&](auto&&) { EXPECT(false); },
             [&](auto&& xs) {
                 reaches = true;
                 EXPECT(xs.size() == 2);
                 EXPECT(xs[0] == as_argument(3));
                 EXPECT(xs[1] == as_argument(3.0));
             });
    EXPECT(reaches);
}

TEST_CASE(tuple_visit_all)
{
    auto a1 = make_tuple(3, 3.0);
    auto a2 = make_tuple(1, 2, 3);

    EXPECT(test::throws([&] { visit_all(a1, a2)([](auto&&, auto&&) {}); }));
    bool reaches = false;
    visit_all(a1, a2)([&](auto&&, auto&&) { EXPECT(false); },
                      [&](auto&& xs, auto&& ys) {
                          reaches = true;
                          EXPECT(xs.size() == 2);
                          EXPECT(xs[0] == as_argument(3));
                          EXPECT(xs[1] == as_argument(3.0));

                          EXPECT(ys.size() == 3);
                          EXPECT(ys[0] == as_argument(1));
                          EXPECT(ys[1] == as_argument(2));
                          EXPECT(ys[2] == as_argument(3));
                      });
    EXPECT(reaches);
}

TEST_CASE(value_argument)
{
    migraphx::shape s{migraphx::shape::int64_type, {3}};
    migraphx::literal l1{s, {1, 2, 3}};
    auto a1 = l1.get_argument();
    auto v1 = migraphx::to_value(a1);
    migraphx::literal l2{1};
    auto a2 = l2.get_argument();
    auto v2 = migraphx::to_value(a2);
    EXPECT(v1 != v2);

    auto a3 = migraphx::from_value<migraphx::argument>(v1);
    EXPECT(a3 == a1);
    auto a4 = migraphx::from_value<migraphx::argument>(v2);
    EXPECT(a4 == a2);
}

TEST_CASE(value_empty_argument)
{
    migraphx::argument a5;
    EXPECT(a5.empty());
    auto v3 = migraphx::to_value(a5);
    auto a6 = migraphx::from_value<migraphx::argument>(v3);
    EXPECT(a6 == a5);
}

TEST_CASE(value_tuple)
{
    auto a1 = make_tuple(3, 3.0, make_tuple(3, 4));
    auto a2 = make_tuple(1, 2, 3);

    auto v1 = migraphx::to_value(a1);
    auto v2 = migraphx::to_value(a2);
    EXPECT(v1 != v2);

    auto a3 = migraphx::from_value<migraphx::argument>(v1);
    EXPECT(a3 == a1);
    auto a4 = migraphx::from_value<migraphx::argument>(v2);
    EXPECT(a4 == a2);
}

TEST_CASE(argument_share)
{
    migraphx::shape s{migraphx::shape::int64_type, {3}};
    std::vector<char> buffer(s.bytes());
    migraphx::argument a1(s, [=]() mutable { return buffer.data(); });
    auto a2 = a1; // NOLINT
    EXPECT(a1.data() != a2.data());

    auto a3 = a1.share();
    EXPECT(a1.data() != a3.data());
    auto a4 = a3; // NOLINT
    EXPECT(a4.data() == a3.data());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
