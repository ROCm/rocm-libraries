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
#include <migraphx/migraphx.h>
#include <migraphx/migraphx.hpp>
#include "test.hpp"

TEST_CASE(create_dynamic_dimensions)
{
    migraphx::dynamic_dimension dd0{1, 4};
    EXPECT(not dd0.is_fixed());
    migraphx::dynamic_dimension dd1{4, 4};
    EXPECT(dd1.is_fixed());
    migraphx::optimals opts{1, 2, 4};
    migraphx::dynamic_dimension dd2{1, 4, opts};
    migraphx::dynamic_dimensions dyn_dims0{dd0, dd1, dd2};
    CHECK(bool{dyn_dims0[0] == dd0});
    CHECK(bool{dyn_dims0[1] == dd1});
    CHECK(bool{dyn_dims0[2] == dd2});
    CHECK(bool{dyn_dims0[2] != dd0});
    EXPECT(dyn_dims0.size() == 3);
}

TEST_CASE(create_dynamic_shape)
{
    migraphx::dynamic_dimensions dyn_dims(migraphx::dynamic_dimension{1, 4},
                                          migraphx::dynamic_dimension{78, 92},
                                          migraphx::dynamic_dimension{1, 4, {1, 4}});
    migraphx::shape dyn_shape{migraphx_shape_float_type, dyn_dims};
    CHECK(bool{dyn_shape.dynamic()});
    CHECK(bool{dyn_shape.dyn_dims()[0] == migraphx::dynamic_dimension{1, 4}});

    migraphx::shape static_shape{migraphx_shape_float_type, {3, 8}};
    EXPECT(not static_shape.dynamic());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
