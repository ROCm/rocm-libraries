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

#include <migraphx/split_single_dyn_dim.hpp>
#include <migraphx/program.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/builtin.hpp>
#include <test.hpp>

static void run_pass(migraphx::program& p)
{
    migraphx::run_passes(p, {migraphx::split_single_dyn_dim{}, migraphx::dead_code_elimination{}});
}

TEST_CASE(dynamic_batch)
{
    // Slightly different from ref_ops_test in that the literal is copied over the submodules.
    // A different compiler pass will pull the literals from the submodules to the main module.
    migraphx::program p0;
    {
        auto* mm0 = p0.get_main_module();

        // create batch submodules
        auto create_submodule = [&](std::size_t batch_size, const std::string& module_name) {
            auto* submod = p0.create_module(module_name);
            migraphx::shape sm_shape{migraphx::shape::float_type, {batch_size, 4}};
            auto sm_input = submod->add_parameter("data", sm_shape);
            migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {1}}};
            auto literal_ins   = submod->add_literal(migraphx::literal{lit_s, {6}});
            auto broadcast_lit =
                submod->add_instruction(migraphx::make_op("multibroadcast"), literal_ins, sm_input);
            auto add_ins =
                submod->add_instruction(migraphx::make_op("add"), sm_input, broadcast_lit);
            submod->add_return({add_ins});
            return submod;
        };
        auto* dim1 = create_submodule(1, "dim_1");
        auto* dim2 = create_submodule(2, "dim_2");
        auto* dim3 = create_submodule(3, "dim_3");
        auto* dim4 = create_submodule(4, "dim_4");

        migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {4, 4}}};
        auto input0                             = mm0->add_parameter("data", s);
        std::vector<migraphx::shape> sub_shapes = {};
        sub_shapes.push_back(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {4, 4}}});
        migraphx::shape out_attr = migraphx::shape{sub_shapes};
        auto sm_ins              = mm0->add_instruction(
            migraphx::make_op("select_module",
                              {{"output_dyn_shapes", migraphx::to_value(out_attr)}}),
            {input0},
            {dim1, dim2, dim3, dim4});
        auto ret =
            mm0->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), sm_ins);
        mm0->add_return({ret});
    }

    migraphx::program p1;
    {
        auto* mm1 = p1.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {4, 4}}};
        auto input1 = mm1->add_parameter("data", s);
        migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {1}}};
        auto literal_ins = mm1->add_literal(migraphx::literal{lit_s, {6}});
        auto broadcast_lit =
            mm1->add_instruction(migraphx::make_op("multibroadcast"), literal_ins, input1);
        auto add_ins = mm1->add_instruction(migraphx::make_op("add"), input1, broadcast_lit);
        mm1->add_return({add_ins});
    }
    run_pass(p1);

    EXPECT(p0 == p1);
}

TEST_CASE(dynamic_batch_multiple_input)
{
    migraphx::program p0;
    {
        auto* mm0 = p0.get_main_module();

        // create batch submodules
        auto create_submodule = [&](std::size_t batch_size, const std::string& module_name) {
            auto* submod = p0.create_module(module_name);
            migraphx::shape sm_shape{migraphx::shape::float_type, {batch_size, 4}};
            auto sm_input0 = submod->add_parameter("data0", sm_shape);
            auto sm_input1 = submod->add_parameter("data1", sm_shape);
            auto sm_input2 = submod->add_parameter("data2", sm_shape);
            migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {1}}};
            auto literal_ins   = submod->add_literal(migraphx::literal{lit_s, {6}});
            auto broadcast_lit = submod->add_instruction(
                migraphx::make_op("multibroadcast"), literal_ins, sm_input0);
            auto add_ins0 =
                submod->add_instruction(migraphx::make_op("add"), sm_input0, broadcast_lit);
            auto add_ins1 = submod->add_instruction(migraphx::make_op("add"), add_ins0, sm_input1);
            auto add_ins2 = submod->add_instruction(migraphx::make_op("add"), add_ins1, sm_input2);
            submod->add_return({add_ins2});
            return submod;
        };
        auto* dim1 = create_submodule(1, "dim_1");
        auto* dim2 = create_submodule(2, "dim_2");
        auto* dim3 = create_submodule(3, "dim_3");
        auto* dim4 = create_submodule(4, "dim_4");

        migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {4, 4}}};
        auto input0                             = mm0->add_parameter("data0", s);
        auto input1                             = mm0->add_parameter("data1", s);
        auto input2                             = mm0->add_parameter("data2", s);
        std::vector<migraphx::shape> sub_shapes = {};
        sub_shapes.push_back(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {4, 4}}});
        migraphx::shape out_attr = migraphx::shape{sub_shapes};
        auto sm_ins              = mm0->add_instruction(
            migraphx::make_op("select_module",
                                           {{"output_dyn_shapes", migraphx::to_value(out_attr)}}),
            {input0, input1, input2},
            {dim1, dim2, dim3, dim4});
        auto ret =
            mm0->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), sm_ins);
        mm0->add_return({ret});
    }

    migraphx::program p1;
    {
        auto* mm1 = p1.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {4, 4}}};
        auto input0 = mm1->add_parameter("data0", s);
        auto input1 = mm1->add_parameter("data1", s);
        auto input2 = mm1->add_parameter("data2", s);
        migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {1}}};
        auto literal_ins = mm1->add_literal(migraphx::literal{lit_s, {6}});
        auto broadcast_lit =
            mm1->add_instruction(migraphx::make_op("multibroadcast"), literal_ins, input0);
        auto add_ins0 = mm1->add_instruction(migraphx::make_op("add"), input0, broadcast_lit);
        auto add_ins1 = mm1->add_instruction(migraphx::make_op("add"), add_ins0, input1);
        auto add_ins2 = mm1->add_instruction(migraphx::make_op("add"), add_ins1, input2);
        mm1->add_return({add_ins2});
    }
    run_pass(p1);

    EXPECT(p0.sort() == p1.sort());
}

TEST_CASE(multiple_outputs)
{
    migraphx::program p0;
    {
        auto* mm0 = p0.get_main_module();

        // create batch submodules
        auto create_submodule = [&](std::size_t batch_size, const std::string& module_name) {
            auto* submod = p0.create_module(module_name);
            migraphx::shape sm_shape{migraphx::shape::float_type, {batch_size, 4}};
            auto sm_input = submod->add_parameter("data", sm_shape);
            migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {1}}};
            auto literal_ins   = submod->add_literal(migraphx::literal{lit_s, {6}});
            auto broadcast_lit =
                submod->add_instruction(migraphx::make_op("multibroadcast"), literal_ins, sm_input);
            auto add0_ins =
                submod->add_instruction(migraphx::make_op("add"), sm_input, broadcast_lit);
            auto add1_ins = submod->add_instruction(migraphx::make_op("add"), sm_input, sm_input);
            submod->add_return({add0_ins, add1_ins});
            return submod;
        };
        auto* dim1 = create_submodule(1, "dim_1");
        auto* dim2 = create_submodule(2, "dim_2");
        auto* dim3 = create_submodule(3, "dim_3");
        auto* dim4 = create_submodule(4, "dim_4");

        migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {4, 4}}};
        auto input0                             = mm0->add_parameter("data", s);
        std::vector<migraphx::shape> sub_shapes = {};
        migraphx::shape tmp_s{migraphx::shape::float_type, {{1, 4}, {4, 4}}};
        sub_shapes.push_back(tmp_s);
        sub_shapes.push_back(tmp_s);
        migraphx::shape out_attr = migraphx::shape{sub_shapes};
        auto sm_ins              = mm0->add_instruction(
            migraphx::make_op("select_module",
                              {{"output_dyn_shapes", migraphx::to_value(out_attr)}}),
            {input0},
            {dim1, dim2, dim3, dim4});
        auto ret0 =
            mm0->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), sm_ins);
        auto ret1 =
            mm0->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), sm_ins);
        mm0->add_return({ret0, ret1});
    }

    migraphx::program p1;
    {
        auto* mm1 = p1.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {4, 4}}};
        auto input1 = mm1->add_parameter("data", s);
        migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {1}}};
        auto literal_ins = mm1->add_literal(migraphx::literal{lit_s, {6}});
        auto broadcast_lit =
            mm1->add_instruction(migraphx::make_op("multibroadcast"), literal_ins, input1);
        auto add0_ins = mm1->add_instruction(migraphx::make_op("add"), input1, broadcast_lit);
        auto add1_ins = mm1->add_instruction(migraphx::make_op("add"), input1, input1);
        mm1->add_return({add0_ins, add1_ins});
    }
    run_pass(p1);

    EXPECT(p0 == p1);
}

// code coverage, does nothing
TEST_CASE(empty_param_shapes)
{
    migraphx::program p0;
    {
        auto* mm0 = p0.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {1, 4}};
        auto input0 = mm0->add_literal(migraphx::literal{s, {0, 1, 2, 3}});
        migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {1}}};
        auto literal_ins = mm0->add_literal(migraphx::literal{lit_s, {6}});
        auto broadcast_lit =
            mm0->add_instruction(migraphx::make_op("multibroadcast"), literal_ins, input0);
        auto add0_ins = mm0->add_instruction(migraphx::make_op("add"), input0, broadcast_lit);
        mm0->add_return({add0_ins});
    }
    migraphx::program p1 = p0;
    run_pass(p0);
    EXPECT(p0 == p1);
};

// code coverage, does nothing
TEST_CASE(multiple_non_fixed_dd_in_a_param)
{
    migraphx::program p0;
    {
        auto* mm0 = p0.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {4, 20}}};
        auto input0 = mm0->add_parameter("data", s);
        migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {1}}};
        auto literal_ins = mm0->add_literal(migraphx::literal{lit_s, {6}});
        auto broadcast_lit =
            mm0->add_instruction(migraphx::make_op("multibroadcast"), literal_ins, input0);
        auto add0_ins = mm0->add_instruction(migraphx::make_op("add"), input0, broadcast_lit);
        mm0->add_return({add0_ins});
    }
    migraphx::program p1 = p0;
    run_pass(p0);
    EXPECT(p0 == p1);
}

// code coverage, does nothing
TEST_CASE(different_non_fixed_dd)
{
    migraphx::program p0;
    {
        auto* mm1 = p0.get_main_module();
        migraphx::shape s0{migraphx::shape::float_type, {{1, 4}, {4, 4}}};
        migraphx::shape s1{migraphx::shape::float_type, {{3, 6}, {1, 1}, {4, 4}}};
        auto input0 = mm1->add_parameter("data0", s0);
        auto input1 = mm1->add_parameter("data1", s1);
        auto broadcast_in0 =
            mm1->add_instruction(migraphx::make_op("multibroadcast"), input0, input1);
        auto broadcast_in1 =
            mm1->add_instruction(migraphx::make_op("multibroadcast"), input1, input0);
        auto add0_ins =
            mm1->add_instruction(migraphx::make_op("add"), broadcast_in0, broadcast_in1);
        mm1->add_return({add0_ins});
    }
    migraphx::program p1 = p0;
    run_pass(p0);
    EXPECT(p0 == p1);
}

// check that the parameter inputs into select_module are lexiographically ordered
TEST_CASE(ordered_inputs_to_select_module)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {4, 4}}};
    auto input0 = mm->add_parameter("breadfruit", s);
    auto input1 = mm->add_parameter("Apricot", s);
    auto input2 = mm->add_parameter("pineapple", s);
    migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {1}}};
    auto literal_ins = mm->add_literal(migraphx::literal{lit_s, {6}});
    auto broadcast_lit =
        mm->add_instruction(migraphx::make_op("multibroadcast"), literal_ins, input0);
    auto add_ins0 = mm->add_instruction(migraphx::make_op("add"), input0, broadcast_lit);
    auto add_ins1 = mm->add_instruction(migraphx::make_op("add"), add_ins0, input1);
    auto add_ins2 = mm->add_instruction(migraphx::make_op("add"), add_ins1, input2);
    mm->add_return({add_ins2});
    run_pass(p);

    auto sm_ins = std::find_if(
        mm->begin(), mm->end(), [&](auto&& ins) { return ins.name() == "select_module"; });
    std::vector<std::string> sm_param_names;
    for(auto&& ins : sm_ins->inputs())
    {
        if(ins->name() == "@param")
        {
            auto&& param = migraphx::any_cast<migraphx::builtin::param>(ins->get_operator());
            sm_param_names.push_back(param.parameter);
        }
    }
    EXPECT(std::is_sorted(sm_param_names.begin(), sm_param_names.end()));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
