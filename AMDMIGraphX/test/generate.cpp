/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/generate.hpp>
#include "test.hpp"

TEST_CASE(generate)
{
    migraphx::shape s{migraphx::shape::float_type, {4, 4, 1, 1}};
    EXPECT(migraphx::generate_literal(s, 1) == migraphx::generate_argument(s, 1));
    EXPECT(migraphx::generate_literal(s, 1) != migraphx::generate_argument(s, 0));
}

TEST_CASE(fill_tuple)
{
    migraphx::shape s0{migraphx::shape::float_type, {4, 4, 1, 1}};
    migraphx::shape s1{migraphx::shape::int32_type, {2, 3}};
    migraphx::shape s2{migraphx::shape::bool_type, {3, 2}};
    migraphx::shape s({s0, s1, s2});
    auto arg         = migraphx::fill_argument(s, 1);
    const auto& args = arg.get_sub_objects();
    EXPECT(args.at(0) == migraphx::fill_argument(s0, 1));
    EXPECT(args.at(1) == migraphx::fill_argument(s1, 1));
    EXPECT(args.at(2) == migraphx::fill_argument(s2, 1));
}

TEST_CASE(generate_tuple)
{
    migraphx::shape s0{migraphx::shape::float_type, {4, 4, 1, 1}};
    migraphx::shape s1{migraphx::shape::int32_type, {2, 3}};
    migraphx::shape s2{migraphx::shape::bool_type, {3, 2}};
    migraphx::shape s({s0, s1, s2});
    auto arg         = migraphx::generate_argument(s, 1);
    const auto& args = arg.get_sub_objects();
    EXPECT(args.at(0) == migraphx::generate_argument(s0, 1));
    EXPECT(args.at(1) == migraphx::generate_argument(s1, 1));
    EXPECT(args.at(2) == migraphx::generate_argument(s2, 1));

    EXPECT(args.at(0) != migraphx::generate_argument(s0, 0));
    EXPECT(args.at(1) != migraphx::generate_argument(s1, 2));
    EXPECT(args.at(2) != migraphx::generate_argument(s2, 0));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
