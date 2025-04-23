/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <onnx_test.hpp>

TEST_CASE(dim_param_fixed_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 4}});
    mm->add_return({input});

    migraphx::onnx_options opt;
    opt.dim_params = {{"dim0", migraphx::shape::dynamic_dimension{2, 2}},
                      {"dim1", migraphx::shape::dynamic_dimension{4, 4}}};
    auto prog      = read_onnx("dim_param_test.onnx", opt);
    EXPECT(p == prog);
}

TEST_CASE(dim_param_dynamic_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("0",
                                   migraphx::shape{migraphx::shape::float_type,
                                                   {migraphx::shape::dynamic_dimension{1, 2},
                                                    migraphx::shape::dynamic_dimension{2, 4}}});
    mm->add_return({input});

    migraphx::onnx_options opt;
    opt.dim_params = {{"dim0", migraphx::shape::dynamic_dimension{1, 2}},
                      {"dim1", migraphx::shape::dynamic_dimension{2, 4}}};
    auto prog      = read_onnx("dim_param_test.onnx", opt);
    EXPECT(p == prog);
}
