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
#include <migraphx/migraphx.h>
#include <migraphx/migraphx.hpp>
#include "test.hpp"

TEST_CASE(add_op)
{
    auto add_op = migraphx::operation("add");
    EXPECT(add_op.name() == "add");
}

TEST_CASE(reduce_mean_without_quotes)
{
    auto rm = migraphx::operation("reduce_mean", "{axes : [1, 2, 3, 4]}");
    EXPECT(rm.name() == "reduce_mean");
}

TEST_CASE(reduce_mean)
{
    auto rm = migraphx::operation("reduce_mean", "{\"axes\" : [1, 2, 3, 4]}");
    EXPECT(rm.name() == "reduce_mean");
}

TEST_CASE(reduce_mean_with_format)
{
    auto rm = migraphx::operation("reduce_mean", "{axes : [%i, %i, %i, %i]}", 1, 2, 3, 4);
    EXPECT(rm.name() == "reduce_mean");
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
