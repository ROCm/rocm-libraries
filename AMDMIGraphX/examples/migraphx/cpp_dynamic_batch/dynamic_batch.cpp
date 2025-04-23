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
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>

// MIGraphX C++ API
#include <migraphx/migraphx.hpp>

int main(int argc, char** argv)
{
    migraphx::onnx_options o_options;
    migraphx::dynamic_dimensions dyn_dims = {migraphx::dynamic_dimension{1, 4, {2, 4}},
                                             migraphx::dynamic_dimension{3, 3},
                                             migraphx::dynamic_dimension{4, 4},
                                             migraphx::dynamic_dimension{5, 5}};
    o_options.set_dyn_input_parameter_shape("0", dyn_dims);
    auto p = migraphx::parse_onnx("../add_scalar_test.onnx", o_options);
    migraphx::compile_options c_options;
    c_options.set_offload_copy();
    p.compile(migraphx::target("gpu"), c_options);

    // batch size = 2
    std::vector<uint8_t> a(2 * 3 * 4 * 5, 3);
    std::vector<uint8_t> b = {2};
    migraphx::program_parameters pp;
    migraphx::shape s = migraphx::shape(migraphx_shape_uint8_type, {2, 3, 4, 5});
    pp.add("0", migraphx::argument(s, a.data()));
    pp.add("1", migraphx::argument(migraphx::shape(migraphx_shape_uint8_type, {1}, {0}), b.data()));
    auto outputs = p.eval(pp);
    auto result  = outputs[0];
    std::vector<uint8_t> c(2 * 3 * 4 * 5, 5);
    if(bool{result == migraphx::argument(s, c.data())})
    {
        std::cout << "Successfully executed dynamic batch add\n";
    }
    else
    {
        std::cout << "Failed dynamic batch add\n";
    }

    return 0;
}
