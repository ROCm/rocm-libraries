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

TEST_CASE(shape_assign)
{
    auto s1_cpp = migraphx::shape{migraphx_shape_float_type, {1, 3}};
    std::vector<size_t> lens{2, 3};

    // handle ptr is const, workaround to construct shape using C API
    migraphx_shape_t s2;
    migraphx_shape_create(&s2, migraphx_shape_float_type, lens.data(), lens.size());
    auto s2_cpp = migraphx::shape(s2, migraphx::own{});
    CHECK(bool{s1_cpp != s2_cpp});
    // use C++ API for assignment
    s1_cpp.assign_to_handle(s2);
    CHECK(bool{s1_cpp == s2_cpp});

    auto s3_cpp = migraphx::shape{migraphx_shape_float_type, lens};
    // use C API for assignment
    migraphx_shape_assign_to(s2, s3_cpp.get_handle_ptr());
    CHECK(bool{s2_cpp == s3_cpp});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
