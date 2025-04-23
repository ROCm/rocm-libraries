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

#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_select_module_add : verify_program<test_select_module_add>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {1}}};
        auto literal_ins = mm->add_literal(migraphx::literal{lit_s, {6}});

        // create batch submodules
        auto create_submodule = [&](std::size_t batch_size, const std::string& module_name) {
            auto* submod = p.create_module(module_name);
            migraphx::shape sm_shape{migraphx::shape::float_type, {batch_size, 4}};
            auto sm_input = submod->add_parameter("data", sm_shape);
            auto broadcast_lit =
                submod->add_instruction(migraphx::make_op("multibroadcast"), literal_ins, sm_input);
            auto add_ins0 =
                submod->add_instruction(migraphx::make_op("add"), sm_input, broadcast_lit);
            auto add_ins1 =
                submod->add_instruction(migraphx::make_op("add"), add_ins0, broadcast_lit);
            submod->add_return({add_ins0, add_ins1});
            return submod;
        };
        auto* batch1 = create_submodule(1, "batch_1");
        auto* batch2 = create_submodule(2, "batch_2");
        auto* batch3 = create_submodule(3, "batch_3");
        auto* batch4 = create_submodule(4, "batch_4");

        migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {4, 4}}};
        auto input                              = mm->add_parameter("data", s);
        std::vector<migraphx::shape> sub_shapes = {};
        sub_shapes.push_back(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {4, 4}}});
        sub_shapes.push_back(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {4, 4}}});
        migraphx::shape out_attr = migraphx::shape{sub_shapes};
        auto sm_ins              = mm->add_instruction(
            migraphx::make_op("select_module",
                              {{"output_dyn_shapes", migraphx::to_value(out_attr)}}),
            {input},
            {batch1, batch2, batch3, batch4});
        auto ret0 =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), sm_ins);
        auto ret1 =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), sm_ins);
        mm->add_return({ret0, ret1});

        return p;
    }
};
