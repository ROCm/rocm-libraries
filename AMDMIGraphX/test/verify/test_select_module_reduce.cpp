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

#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_select_module_reduce : verify_program<test_select_module_reduce>
{
    migraphx::program create_program() const
    {
        migraphx::program p;

        // create batch submodules
        auto create_submodule = [&](std::size_t batch_size, const std::string& module_name) {
            auto* submod = p.create_module(module_name);
            migraphx::shape sm_shape{migraphx::shape::float_type, {batch_size, 2, 2}};
            auto sm_input = submod->add_parameter("data", sm_shape);
            auto reduce_ins =
                submod->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), sm_input);
            auto squeeze_ins =
                submod->add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), reduce_ins);
            submod->add_return({squeeze_ins});
            return submod;
        };
        auto* batch1 = create_submodule(1, "batch_1");
        auto* batch2 = create_submodule(2, "batch_2");
        auto* batch3 = create_submodule(3, "batch_3");
        auto* batch4 = create_submodule(4, "batch_4");

        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {2, 2}, {2, 2}}};
        auto input                              = mm->add_parameter("data", s);
        std::vector<migraphx::shape> sub_shapes = {};
        sub_shapes.push_back(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {2, 2}}});
        migraphx::shape out_attr = migraphx::shape{sub_shapes};
        auto sm_ins              = mm->add_instruction(
            migraphx::make_op("select_module",
                              {{"output_dyn_shapes", migraphx::to_value(out_attr)}}),
            {input},
            {batch1, batch2, batch3, batch4});
        auto ret = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), sm_ins);
        mm->add_return({ret});

        return p;
    }

    std::string section() const { return "reduce"; }
};
