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
#include "test.hpp"
#include <migraphx/check_shapes.hpp>
#include <migraphx/make_op.hpp>

/*!
 * Tests for check_shapes object handling dynamic shapes
 */

using migraphx::shape;

static void create_shapes(bool dynamic_allowed)
{
    shape a{shape::int64_type, {3}};
    shape b{shape::float_type, {{3, 6}, {4, 4}}};
    migraphx::check_shapes{{a, b}, "", dynamic_allowed}.has(2);
}

TEST_CASE(allow_dynamic_shape)
{
    EXPECT(not test::throws([] { create_shapes(true); }));
}

TEST_CASE(fail_dynamic_shape)
{
    EXPECT(test::throws([] { create_shapes(false); }));
}

TEST_CASE(same_layout_fail)
{
    EXPECT(test::throws([] {
        shape a{shape::float_type, {2, 3}};
        shape b{shape::float_type, {2, 3}, {1, 2}};
        migraphx::check_shapes{{a, b}, ""}.same_layout();
    }));
}

TEST_CASE(same_layout_pass)
{
    EXPECT(not test::throws([] {
        shape a{shape::float_type, {2, 3}, {1, 2}};
        shape b{shape::float_type, {2, 3}, {1, 2}};
        migraphx::check_shapes{{a, b}, ""}.same_layout();
    }));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
