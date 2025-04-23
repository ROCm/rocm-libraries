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
#include <migraphx/simplify_dyn_ops.hpp>
#include <migraphx/split_single_dyn_dim.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/common.hpp>
#include <migraphx/program.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/make_op.hpp>
#include <test.hpp>

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::simplify_dyn_ops{}, migraphx::dead_code_elimination{}});
}

TEST_CASE(broadcast_with_dims)
{
    migraphx::module m0;
    {
        // the X input
        migraphx::shape sx{migraphx::shape::float_type, {3, 1, 1}};
        auto inx = m0.add_parameter("x", sx);

        // the shape input.  Broadcast to this
        migraphx::shape dims_s{migraphx::shape::int64_type, {4}};
        std::vector<size_t> dims = {2, 3, 4, 5};
        auto out_dims            = m0.add_literal(migraphx::literal{dims_s, dims});

        auto r = m0.add_instruction(migraphx::make_op("broadcast_with_dims"), inx, out_dims);
        m0.add_return({r});
    }
    run_pass(m0);

    migraphx::module m1;
    {
        migraphx::shape sx{migraphx::shape::float_type, {3, 1, 1}};
        auto inx = m1.add_parameter("x", sx);

        auto r = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 3, 4, 5}}}), inx);
        m1.add_return({r});
    }
    EXPECT(m0 == m1);
}

TEST_CASE(broadcast_with_dims_invalid)
{
    migraphx::module m0;
    {
        // X input shape is not broadcastable to given shape
        migraphx::shape sx{migraphx::shape::float_type, {3, 1, 2}};
        auto inx = m0.add_parameter("x", sx);

        // the shape input.  Broadcast to this
        migraphx::shape dims_s{migraphx::shape::int64_type, {4}};
        std::vector<size_t> dims = {2, 3, 4, 5};
        auto out_dims            = m0.add_literal(migraphx::literal{dims_s, dims});

        auto r = m0.add_instruction(migraphx::make_op("broadcast_with_dims"), inx, out_dims);
        m0.add_return({r});
    }
    // replacement will be rejected by multibroadcast operation
    EXPECT(test::throws([&] { run_pass(m0); }));
}

TEST_CASE(resize)
{
    migraphx::module m0;
    {
        std::vector<size_t> ds = {1, 1, 4, 6};
        migraphx::shape ss{migraphx::shape::int64_type, {4}};

        auto li = m0.add_literal(migraphx::literal{ss, ds});
        m0.add_instruction(migraphx::make_op("undefined"));

        migraphx::shape sx{migraphx::shape::int64_type, {1, 1, 2, 2}};
        auto inx = m0.add_parameter("X", sx);

        auto r = m0.add_instruction(
            migraphx::make_op("resize",
                              {{"mode", "nearest"},
                               {"nearest_mode", "floor"},
                               // scales attr. should be ignored when there are 2 inputs
                               {"scales", {1., 2.1, 3.1, 4.1}},
                               {"coordinate_transformation_mode", "asymmetric"}}),
            inx,
            li);

        m0.add_return({r});
    }
    run_pass(m0);

    migraphx::module m1;
    {
        migraphx::shape sx{migraphx::shape::int64_type, {1, 1, 2, 2}};
        auto inx = m1.add_parameter("X", sx);

        std::vector<float> indices = {0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1,
                                      2, 2, 2, 3, 3, 3, 2, 2, 2, 3, 3, 3};
        migraphx::shape ss{migraphx::shape::int32_type, {1, 1, 4, 6}};
        auto li = m1.insert_literal(inx, migraphx::literal{ss, indices});

        auto reshape_ins = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {4}}}), inx);
        auto gather_ins =
            m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), reshape_ins, li);
        m1.add_return({gather_ins});
    }
    EXPECT(m0 == m1);
}

TEST_CASE(resize_scales)
{
    migraphx::module m0;
    {
        // resize op. with scales (float_type) rather than sizes as an input
        std::vector<float> ds = {1., 1., 2., 3.};
        migraphx::shape ss{migraphx::shape::float_type, {4}};

        auto li = m0.add_literal(migraphx::literal{ss, ds});
        m0.add_instruction(migraphx::make_op("undefined"));

        migraphx::shape sx{migraphx::shape::int64_type, {1, 1, 2, 2}};
        auto inx = m0.add_parameter("X", sx);

        auto r = m0.add_instruction(
            migraphx::make_op("resize",
                              {{"mode", "nearest"},
                               {"nearest_mode", "floor"},
                               // scales attr. should be ignored when there are 2 inputs
                               {"scales", {1., 2.1, 3.1, 4.1}},
                               {"coordinate_transformation_mode", "asymmetric"}}),
            inx,
            li);

        m0.add_return({r});
    }
    run_pass(m0);

    migraphx::module m1;
    {
        migraphx::shape sx{migraphx::shape::int64_type, {1, 1, 2, 2}};
        auto inx = m1.add_parameter("X", sx);

        std::vector<float> indices = {0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1,
                                      2, 2, 2, 3, 3, 3, 2, 2, 2, 3, 3, 3};
        migraphx::shape ss{migraphx::shape::int32_type, {1, 1, 4, 6}};
        auto li = m1.insert_literal(inx, migraphx::literal{ss, indices});

        auto reshape_ins = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {4}}}), inx);
        auto gather_ins =
            m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), reshape_ins, li);
        m1.add_return({gather_ins});
    }
    EXPECT(m0 == m1);
}

TEST_CASE(static_broadcast)
{
    migraphx::module m0;
    {
        migraphx::shape s{migraphx::shape::float_type, {2, 4}};
        auto input = m0.add_parameter("data", s);
        migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {4}}};
        auto literal_ins = m0.add_literal(migraphx::literal{lit_s, {6, 5, 4, 3}});
        auto broadcast_lit =
            m0.add_instruction(migraphx::make_op("broadcast", {{"axis", 1}}), literal_ins, input);
        auto add_ins = m0.add_instruction(migraphx::make_op("add"), input, broadcast_lit);
        m0.add_return({add_ins});
    }
    run_pass(m0);

    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {2, 4}};
        auto input = m1.add_parameter("data", s);
        migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {4}}};
        auto literal_ins   = m1.add_literal(migraphx::literal{lit_s, {6, 5, 4, 3}});
        auto broadcast_lit = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", s.lens()}}), literal_ins);
        auto add_ins = m1.add_instruction(migraphx::make_op("add"), input, broadcast_lit);
        m1.add_return({add_ins});
    }
    EXPECT(m0 == m1);
}

TEST_CASE(static_multibroadcast)
{

    migraphx::module m0;
    {
        migraphx::shape s{migraphx::shape::float_type, {2, 4}};
        auto input = m0.add_parameter("data", s);
        migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {1}, {0}}};
        auto literal_ins = m0.add_literal(migraphx::literal{lit_s, {6}});
        auto broadcast_lit =
            m0.add_instruction(migraphx::make_op("multibroadcast"), literal_ins, input);
        auto add_ins = m0.add_instruction(migraphx::make_op("add"), input, broadcast_lit);
        m0.add_return({add_ins});
    }
    run_pass(m0);

    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {2, 4}};
        auto input = m1.add_parameter("data", s);
        migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {1}, {0}}};
        auto literal_ins   = m1.add_literal(migraphx::literal{lit_s, {6}});
        auto broadcast_lit = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), literal_ins);
        auto add_ins = m1.add_instruction(migraphx::make_op("add"), input, broadcast_lit);
        m1.add_return({add_ins});
    }
    EXPECT(m0 == m1);
}

TEST_CASE(after_split_dyn_broadcast_match)
{
    migraphx::program p0;
    {
        auto* mm1 = p0.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {4, 4}}};
        auto input1 = mm1->add_parameter("data", s);
        migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {4}}};
        auto literal_ins   = mm1->add_literal(migraphx::literal{lit_s, {6, 5, 4, 3}});
        auto broadcast_lit = mm1->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}}), literal_ins, input1);
        auto add_ins = mm1->add_instruction(migraphx::make_op("add"), input1, broadcast_lit);
        mm1->add_return({add_ins});
    }
    migraphx::run_passes(p0,
                         {migraphx::split_single_dyn_dim{},
                          migraphx::dead_code_elimination{},
                          migraphx::simplify_dyn_ops{}});

    migraphx::program p1;
    {
        auto* mm0 = p1.get_main_module();

        // create batch submodules
        auto create_submodule = [&](std::size_t batch_size, const std::string& module_name) {
            auto* submod = p1.create_module(module_name);
            migraphx::shape sm_shape{migraphx::shape::float_type, {batch_size, 4}};
            auto sm_input = submod->add_parameter("data", sm_shape);
            migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {4}}};
            auto literal_ins   = submod->add_literal(migraphx::literal{lit_s, {6, 5, 4, 3}});
            auto broadcast_lit = submod->add_instruction(
                migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", sm_shape.lens()}}),
                literal_ins);
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
    EXPECT(p0 == p1);
}

TEST_CASE(const_slice_2input_ends_axes)
{
    migraphx::module m0;
    {
        migraphx::shape s{migraphx::shape::float_type, {6, 4, 4}};
        auto input = m0.add_parameter("data", s);
        migraphx::shape s1{migraphx::shape::int32_type, {1}};
        auto input_starts = m0.add_literal(migraphx::literal{s1, {0}});
        auto slice_ins    = m0.add_instruction(
            migraphx::make_op("slice", {{"ends", {3}}, {"axes", {0}}}), input, input_starts);
        m0.add_return({slice_ins});
    }
    run_pass(m0);

    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {6, 4, 4}};
        auto input     = m1.add_parameter("data", s);
        auto slice_ins = m1.add_instruction(
            migraphx::make_op("slice", {{"starts", {0}}, {"ends", {3}}, {"axes", {0}}}), input);
        m1.add_return({slice_ins});
    }
    EXPECT(m0 == m1);
}

TEST_CASE(const_slice_2input_starts_axes)
{
    migraphx::module m0;
    {
        migraphx::shape s{migraphx::shape::float_type, {6, 4, 4}};
        auto input = m0.add_parameter("data", s);
        migraphx::shape s1{migraphx::shape::int32_type, {1}};
        auto input_ends = m0.add_literal(migraphx::literal{s1, {3}});
        auto slice_ins  = m0.add_instruction(
            migraphx::make_op("slice", {{"starts", {0}}, {"axes", {0}}}), input, input_ends);
        m0.add_return({slice_ins});
    }
    run_pass(m0);

    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {6, 4, 4}};
        auto input     = m1.add_parameter("data", s);
        auto slice_ins = m1.add_instruction(
            migraphx::make_op("slice", {{"starts", {0}}, {"ends", {3}}, {"axes", {0}}}), input);
        m1.add_return({slice_ins});
    }
    EXPECT(m0 == m1);
}

TEST_CASE(const_slice_2input_starts_ends)
{
    migraphx::module m0;
    {
        migraphx::shape s{migraphx::shape::float_type, {6, 4, 4}};
        auto input = m0.add_parameter("data", s);
        migraphx::shape s1{migraphx::shape::int32_type, {1}};
        auto input_axes = m0.add_literal(migraphx::literal{s1, {0}});
        auto slice_ins  = m0.add_instruction(
            migraphx::make_op("slice", {{"starts", {0}}, {"ends", {3}}}), input, input_axes);
        m0.add_return({slice_ins});
    }
    run_pass(m0);

    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {6, 4, 4}};
        auto input     = m1.add_parameter("data", s);
        auto slice_ins = m1.add_instruction(
            migraphx::make_op("slice", {{"starts", {0}}, {"ends", {3}}, {"axes", {0}}}), input);
        m1.add_return({slice_ins});
    }
    EXPECT(m0 == m1);
}

TEST_CASE(const_slice_3input_axes_only)
{
    migraphx::module m0;
    {
        migraphx::shape s{migraphx::shape::float_type, {6, 4, 4}};
        auto input = m0.add_parameter("data", s);
        migraphx::shape s1{migraphx::shape::int32_type, {1}};
        auto input_starts = m0.add_literal(migraphx::literal{s1, {0}});
        auto input_ends   = m0.add_literal(migraphx::literal{s1, {3}});
        auto slice_ins    = m0.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}}), input, input_starts, input_ends);
        m0.add_return({slice_ins});
    }
    run_pass(m0);

    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {6, 4, 4}};
        auto input     = m1.add_parameter("data", s);
        auto slice_ins = m1.add_instruction(
            migraphx::make_op("slice", {{"starts", {0}}, {"ends", {3}}, {"axes", {0}}}), input);
        m1.add_return({slice_ins});
    }
    EXPECT(m0 == m1);
}

TEST_CASE(const_slice_3input_ends_only)
{
    migraphx::module m0;
    {
        migraphx::shape s{migraphx::shape::float_type, {6, 4, 4}};
        auto input = m0.add_parameter("data", s);
        migraphx::shape s1{migraphx::shape::int32_type, {1}};
        auto input_starts = m0.add_literal(migraphx::literal{s1, {0}});
        auto input_axes   = m0.add_literal(migraphx::literal{s1, {0}});
        auto slice_ins    = m0.add_instruction(
            migraphx::make_op("slice", {{"ends", {3}}}), input, input_starts, input_axes);
        m0.add_return({slice_ins});
    }
    run_pass(m0);

    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {6, 4, 4}};
        auto input     = m1.add_parameter("data", s);
        auto slice_ins = m1.add_instruction(
            migraphx::make_op("slice", {{"starts", {0}}, {"ends", {3}}, {"axes", {0}}}), input);
        m1.add_return({slice_ins});
    }
    EXPECT(m0 == m1);
}

TEST_CASE(const_slice_3inputs_starts_only)
{
    migraphx::module m0;
    {
        migraphx::shape s{migraphx::shape::float_type, {6, 4, 4}};
        auto input = m0.add_parameter("data", s);
        migraphx::shape s1{migraphx::shape::int32_type, {1}};
        auto input_ends = m0.add_literal(migraphx::literal{s1, {3}});
        auto input_axes = m0.add_literal(migraphx::literal{s1, {0}});
        auto slice_ins  = m0.add_instruction(
            migraphx::make_op("slice", {{"starts", {0}}}), input, input_ends, input_axes);
        m0.add_return({slice_ins});
    }
    run_pass(m0);

    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {6, 4, 4}};
        auto input     = m1.add_parameter("data", s);
        auto slice_ins = m1.add_instruction(
            migraphx::make_op("slice", {{"starts", {0}}, {"ends", {3}}, {"axes", {0}}}), input);
        m1.add_return({slice_ins});
    }
    EXPECT(m0 == m1);
}

TEST_CASE(const_slice_2input_ends_axes_dyn)
{
    migraphx::module m0;
    {
        migraphx::shape s{migraphx::shape::float_type, {{6, 6}, {2, 4, {2, 4}}, {2, 4, {2, 4}}}};
        auto input = m0.add_parameter("data", s);
        migraphx::shape s1{migraphx::shape::int32_type, {1}};
        auto input_starts = m0.add_literal(migraphx::literal{s1, {0}});
        auto slice_ins    = m0.add_instruction(
            migraphx::make_op("slice", {{"ends", {3}}, {"axes", {0}}}), input, input_starts);
        m0.add_return({slice_ins});
    }
    run_pass(m0);

    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {{6, 6}, {2, 4, {2, 4}}, {2, 4, {2, 4}}}};
        auto input     = m1.add_parameter("data", s);
        auto slice_ins = m1.add_instruction(
            migraphx::make_op("slice", {{"starts", {0}}, {"ends", {3}}, {"axes", {0}}}), input);
        m1.add_return({slice_ins});
    }
    EXPECT(m0 == m1);
}

TEST_CASE(const_slice_3input_dyn)
{

    migraphx::module m0;
    {
        migraphx::shape s{migraphx::shape::float_type, {{6, 6}, {2, 4, {2, 4}}, {2, 4, {2, 4}}}};
        auto input = m0.add_parameter("data", s);
        migraphx::shape s1{migraphx::shape::int32_type, {1}};
        auto input_starts = m0.add_literal(migraphx::literal{s1, {0}});
        auto input_ends   = m0.add_literal(migraphx::literal{s1, {3}});
        auto slice_ins    = m0.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}}), input, input_starts, input_ends);
        m0.add_return({slice_ins});
    }
    run_pass(m0);

    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {{6, 6}, {2, 4, {2, 4}}, {2, 4, {2, 4}}}};
        auto input     = m1.add_parameter("data", s);
        auto slice_ins = m1.add_instruction(
            migraphx::make_op("slice", {{"starts", {0}}, {"ends", {3}}, {"axes", {0}}}), input);
        m1.add_return({slice_ins});
    }
    EXPECT(m0 == m1);
}

TEST_CASE(const_slice_4input)
{
    migraphx::module m0;
    {
        migraphx::shape s{migraphx::shape::float_type, {6, 4, 4}};
        auto input = m0.add_parameter("data", s);
        migraphx::shape s1{migraphx::shape::int32_type, {1}};
        auto input_starts = m0.add_literal(migraphx::literal{s1, {0}});
        auto input_ends   = m0.add_literal(migraphx::literal{s1, {3}});
        auto input_axes   = m0.add_literal(migraphx::literal{s1, {0}});
        auto slice_ins    = m0.add_instruction(
            migraphx::make_op("slice"), input, input_starts, input_ends, input_axes);
        m0.add_return({slice_ins});
    }
    run_pass(m0);

    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {6, 4, 4}};
        auto input     = m1.add_parameter("data", s);
        auto slice_ins = m1.add_instruction(
            migraphx::make_op("slice", {{"starts", {0}}, {"ends", {3}}, {"axes", {0}}}), input);
        m1.add_return({slice_ins});
    }
    EXPECT(m0 == m1);
}

TEST_CASE(static_dimensions_of0)
{
    // dead_code_elimination will get rid of atan
    migraphx::module m0;
    {
        migraphx::shape s{migraphx::shape::float_type, {2, 4, 4}};
        auto input    = m0.add_parameter("data", s);
        auto atan_ins = m0.add_instruction(migraphx::make_op("atan"), input);
        auto dimensions_of_ins =
            m0.add_instruction(migraphx::make_op("dimensions_of", {{"end", 3}}), atan_ins);
        m0.add_return({dimensions_of_ins});
    }
    run_pass(m0);

    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {2, 4, 4}};
        m1.add_parameter("data", s);
        migraphx::shape lit_shape{migraphx::shape::int64_type, {3}};
        std::vector<int64_t> lit_data = {2, 4, 4};
        auto lit_ins                  = m1.add_literal(migraphx::literal{lit_shape, lit_data});
        m1.add_return({lit_ins});
    }
    EXPECT(m0 == m1);
}

TEST_CASE(static_dimensions_of1)
{
    // dead_code_elimination will get rid of atan
    migraphx::module m0;
    {
        migraphx::shape s{migraphx::shape::float_type, {{2, 4, {2, 4}}, {4, 4}, {4, 4}}};
        auto input             = m0.add_parameter("data", s);
        auto atan_ins          = m0.add_instruction(migraphx::make_op("atan"), input);
        auto dimensions_of_ins = m0.add_instruction(
            migraphx::make_op("dimensions_of", {{"start", 1}, {"end", 3}}), atan_ins);
        m0.add_return({dimensions_of_ins});
    }
    run_pass(m0);

    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {{2, 4, {2, 4}}, {4, 4}, {4, 4}}};
        m1.add_parameter("data", s);
        migraphx::shape lit_shape{migraphx::shape::int64_type, {2}};
        std::vector<int64_t> lit_data = {4, 4};
        auto lit_ins                  = m1.add_literal(migraphx::literal{lit_shape, lit_data});
        m1.add_return({lit_ins});
    }
    EXPECT(m0 == m1);
}

// Does nothing because the dynamic_dimensions from start to end
// are not all fixed
TEST_CASE(static_dimensions_of_nonfixed)
{
    // dead_code_elimination will get rid of atan
    migraphx::module m0;
    {
        migraphx::shape s{migraphx::shape::float_type, {{2, 4, {2, 4}}, {4, 8}, {4, 8}}};
        auto input             = m0.add_parameter("data", s);
        auto atan_ins          = m0.add_instruction(migraphx::make_op("atan"), input);
        auto dimensions_of_ins = m0.add_instruction(
            migraphx::make_op("dimensions_of", {{"start", 1}, {"end", 3}}), atan_ins);
        m0.add_return({dimensions_of_ins});
    }
    run_pass(m0);

    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {{2, 4, {2, 4}}, {4, 8}, {4, 8}}};
        auto input             = m1.add_parameter("data", s);
        auto atan_ins          = m1.add_instruction(migraphx::make_op("atan"), input);
        auto dimensions_of_ins = m1.add_instruction(
            migraphx::make_op("dimensions_of", {{"start", 1}, {"end", 3}}), atan_ins);
        m1.add_return({dimensions_of_ins});
    }
    EXPECT(m0 == m1);
}

TEST_CASE(constant_alloc_reshape)
{
    migraphx::module m0;
    {
        migraphx::shape s{migraphx::shape::float_type, {3, 32}};
        auto input = m0.add_parameter("data", s);
        migraphx::shape lit_s{migraphx::shape::int64_type, {3}};
        auto literal_ins = m0.add_literal(migraphx::literal{lit_s, {3, 4, 8}});
        auto alloc_ins   = m0.add_instruction(
            migraphx::make_op("allocate", {{"buf_type", migraphx::shape::float_type}}),
            literal_ins);
        auto reshape_ins = m0.add_instruction(migraphx::make_op("reshape"), input, alloc_ins);
        m0.add_return({reshape_ins});
    }
    run_pass(m0);

    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {3, 32}};
        auto input = m1.add_parameter("data", s);
        auto reshape_ins =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {3, 4, 8}}}), input);
        m1.add_return({reshape_ins});
    }
    EXPECT(m0 == m1);
}

// A more contrived example to test static dimensions_of and constant reshape
TEST_CASE(static_dimensions_of_to_constant_alloc_reshape)
{
    migraphx::module m0;
    {
        migraphx::shape input_shape{migraphx::shape::float_type, {3, 4, 8}};
        auto x_param = m0.add_parameter("x", input_shape);
        auto dimensions_of_ins =
            m0.add_instruction(migraphx::make_op("dimensions_of", {{"end", 3}}), x_param);
        migraphx::shape lit_shape{migraphx::shape::int64_type, {1}};
        auto lit0 = m0.add_literal(migraphx::literal{lit_shape, {0}});
        auto gather_ins =
            m0.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), dimensions_of_ins, lit0);
        auto slice_ins = m0.add_instruction(
            migraphx::make_op("slice", {{"starts", {1}}, {"ends", {3}}, {"axes", {0}}}),
            dimensions_of_ins);
        auto reduce_ins =
            m0.add_instruction(migraphx::make_op("reduce_prod", {{"axes", {0}}}), slice_ins);
        auto concat_ins =
            m0.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), gather_ins, reduce_ins);
        auto alloc_ins = m0.add_instruction(
            migraphx::make_op("allocate", {{"buf_type", migraphx::shape::float_type}}), concat_ins);
        auto reshape_ins = m0.add_instruction(migraphx::make_op("reshape"), x_param, alloc_ins);
        m0.add_return({reshape_ins});
    }
    run_pass(m0);

    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {3, 4, 8}};
        auto x_param = m1.add_parameter("x", s);
        auto reshape_ins =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {3, 32}}}), x_param);
        m1.add_return({reshape_ins});
    }
    EXPECT(m0 == m1);
}

TEST_CASE(const_alloc_fill)
{
    migraphx::module m0;
    {
        migraphx::shape val_shape{migraphx::shape::int64_type, {1}, {0}};
        std::vector<int64_t> lit_data = {3};
        auto value_lit                = m0.add_literal(migraphx::literal{val_shape, lit_data});
        migraphx::shape lit_s{migraphx::shape::int64_type, {3}};
        auto output_dim_lit = m0.add_literal(migraphx::literal{lit_s, {3, 4, 4}});
        auto alloc_ins      = m0.add_instruction(
            migraphx::make_op("allocate", {{"buf_type", migraphx::shape::int64_type}}),
            output_dim_lit);
        auto ret = m0.add_instruction(migraphx::make_op("fill"), value_lit, alloc_ins);
        m0.add_return({ret});
    }
    run_pass(m0);

    migraphx::module m1;
    {
        migraphx::shape lit_shape{migraphx::shape::int64_type, {3, 4, 4}};
        std::vector<int64_t> lit_data(3 * 4 * 4, 3);
        auto ret = m1.add_literal(migraphx::literal{lit_shape, lit_data});
        m1.add_return({ret});
    }
    EXPECT(m0 == m1);
}

TEST_CASE(static_broadcast_for_dot)
{
    migraphx::module m0;
    {
        migraphx::shape s{migraphx::shape::float_type, {2, 4, 6, 8}};
        auto input = m0.add_parameter("data", s);
        migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {8, 10}}};
        std::vector<float> lit_vec(80, 2.f);
        auto literal_ins = m0.add_literal(migraphx::literal{lit_s, lit_vec});
        auto broadcast_for_dot_ins =
            m0.add_instruction(migraphx::make_op("broadcast_for_dot"), literal_ins, input);
        auto dot_ins = m0.add_instruction(migraphx::make_op("dot"), input, broadcast_for_dot_ins);
        m0.add_return({dot_ins});
    }
    run_pass(m0);

    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {2, 4, 6, 8}};
        auto input = m1.add_parameter("data", s);
        migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {8, 10}}};
        std::vector<float> lit_vec(80, 2.f);
        auto literal_ins        = m1.add_literal(migraphx::literal{lit_s, lit_vec});
        auto multibroadcast_ins = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 8, 10}}}), literal_ins);
        auto dot_ins = m1.add_instruction(migraphx::make_op("dot"), input, multibroadcast_ins);
        m1.add_return({dot_ins});
    }
    EXPECT(m0 == m1);
}

TEST_CASE(static_onehot)
{
    // depth as a literal
    migraphx::module m0;
    {
        migraphx::shape inds_s{migraphx::shape::int64_type, {4}};
        migraphx::shape depth_s{migraphx::shape::int64_type, {1}};
        migraphx::shape values_s{migraphx::shape::float_type, {2}};
        auto inds_param   = m0.add_parameter("indices", inds_s);
        auto depth_lit    = m0.add_literal(migraphx::literal{depth_s, {3}});
        auto values_param = m0.add_parameter("values", values_s);
        auto onehot_ins   = m0.add_instruction(
            migraphx::make_op("onehot", {{"axis", -1}}), inds_param, depth_lit, values_param);
        m0.add_return({onehot_ins});
    }
    run_pass(m0);

    migraphx::module m1;
    {
        migraphx::shape inds_s{migraphx::shape::int64_type, {4}};
        migraphx::shape depth_s{migraphx::shape::int64_type, {1}};
        migraphx::shape values_s{migraphx::shape::float_type, {2}};
        auto inds_param   = m1.add_parameter("indices", inds_s);
        auto values_param = m1.add_parameter("values", values_s);
        migraphx::shape output_shape{migraphx::shape::float_type, {4, 3}};
        std::vector<float> zeros(output_shape.elements(), 0);
        auto zeros_lit = m1.add_literal(migraphx::literal(output_shape, zeros));
        auto unsqueeze_inds =
            m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), inds_param);
        auto ones_lit = m1.add_literal(
            migraphx::literal(migraphx::shape{migraphx::shape::float_type, {1}, {0}}, {1}));
        auto mb_ones = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {4, 1}}}), ones_lit);
        auto mask = m1.add_instruction(
            migraphx::make_op("scatter_none", {{"axis", 1}, {"skip_out_of_bounds", 1}}),
            zeros_lit,
            unsqueeze_inds,
            mb_ones);
        auto off_val = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}),
            values_param);
        auto on_val = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}),
            values_param);
        auto diff_val      = m1.add_instruction(migraphx::make_op("sub"), on_val, off_val);
        auto mul_diff_mask = add_common_op(m1, migraphx::make_op("mul"), {diff_val, mask});
        auto ret           = add_common_op(m1, migraphx::make_op("add"), {off_val, mul_diff_mask});
        m1.add_return({ret});
    }
    EXPECT(m0 == m1);

    // depth as an attribute
    migraphx::module m2;
    {
        migraphx::shape inds_s{migraphx::shape::int64_type, {4}};
        migraphx::shape values_s{migraphx::shape::float_type, {2}};
        auto inds_param   = m2.add_parameter("indices", inds_s);
        auto values_param = m2.add_parameter("values", values_s);
        auto onehot_ins   = m2.add_instruction(
            migraphx::make_op("onehot", {{"axis", -1}, {"depth", 3}}), inds_param, values_param);
        m2.add_return({onehot_ins});
    }
    run_pass(m2);
    EXPECT(m2 == m1);
}

TEST_CASE(onehot_cannot_simplify)
{
    migraphx::module m0;
    {
        migraphx::shape inds_s{migraphx::shape::int64_type, {4}};
        migraphx::shape depth_s{migraphx::shape::int64_type, {1}};
        migraphx::shape values_s{migraphx::shape::float_type, {2}};
        auto inds_param   = m0.add_parameter("indices", inds_s);
        auto depth_param  = m0.add_parameter("depth", depth_s);
        auto values_param = m0.add_parameter("values", values_s);
        auto onehot_ins   = m0.add_instruction(
            migraphx::make_op("onehot", {{"axis", -1}}), inds_param, depth_param, values_param);
        m0.add_return({onehot_ins});
    }
    migraphx::module m1 = m0;
    run_pass(m0);
    EXPECT(m0 == m1);
}

// Test case with static output shape in the submodules (look at `sm_shape`)
TEST_CASE(select_module_update0)
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
            auto literal_ins = submod->add_literal(migraphx::literal{lit_s, {6}});
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
        auto max_int                            = std::numeric_limits<std::size_t>::max();
        sub_shapes.push_back(migraphx::shape{migraphx::shape::float_type, {{0, max_int}, {4, 4}}});
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
    migraphx::run_passes(p0, {migraphx::simplify_dyn_ops{}, migraphx::dead_code_elimination{}});

    // difference is `output_dyn_shapes` attribute in `select_module`
    // multibroadcast also simplified
    migraphx::program p1;
    {
        auto* mm1 = p1.get_main_module();

        // create batch submodules
        auto create_submodule = [&](std::size_t batch_size, const std::string& module_name) {
            auto* submod = p1.create_module(module_name);
            migraphx::shape sm_shape{migraphx::shape::float_type, {batch_size, 4}};
            auto sm_input = submod->add_parameter("data", sm_shape);
            migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {1}}};
            auto literal_ins   = submod->add_literal(migraphx::literal{lit_s, {6}});
            auto broadcast_lit = submod->add_instruction(
                migraphx::make_op("multibroadcast", {{"out_lens", sm_shape.lens()}}), literal_ins);
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
        auto input0                             = mm1->add_parameter("data", s);
        std::vector<migraphx::shape> sub_shapes = {};
        sub_shapes.push_back(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {4, 4}}});
        migraphx::shape out_attr = migraphx::shape{sub_shapes};
        auto sm_ins              = mm1->add_instruction(
            migraphx::make_op("select_module",
                                           {{"output_dyn_shapes", migraphx::to_value(out_attr)}}),
            {input0},
            {dim1, dim2, dim3, dim4});
        auto ret =
            mm1->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), sm_ins);
        mm1->add_return({ret});
    }

    EXPECT(p0 == p1);
}

// Test case with dynamic output shape in the submodules (look at `sm_shape`)
TEST_CASE(select_module_update1)
{
    migraphx::program p0;
    {
        auto* mm0 = p0.get_main_module();

        // create batch submodules
        auto create_submodule = [&](std::size_t batch_size, const std::string& module_name) {
            auto* submod = p0.create_module(module_name);
            migraphx::shape sm_shape{migraphx::shape::float_type,
                                     {{batch_size, batch_size}, {4, 20}}};
            auto sm_input = submod->add_parameter("data", sm_shape);
            migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {1}}};
            auto literal_ins = submod->add_literal(migraphx::literal{lit_s, {6}});
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

        migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {4, 20}}};
        auto input0                             = mm0->add_parameter("data", s);
        std::vector<migraphx::shape> sub_shapes = {};
        auto max_int                            = std::numeric_limits<std::size_t>::max();
        sub_shapes.push_back(migraphx::shape{migraphx::shape::float_type, {{0, max_int}, {4, 20}}});
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
    migraphx::run_passes(p0, {migraphx::simplify_dyn_ops{}, migraphx::dead_code_elimination{}});

    // difference is `output_dyn_shapes` attribute in `select_module`
    // note that multibroadcast is not simplify-able for this case
    migraphx::program p1;
    {
        auto* mm1 = p1.get_main_module();

        // create batch submodules
        auto create_submodule = [&](std::size_t batch_size, const std::string& module_name) {
            auto* submod = p1.create_module(module_name);
            migraphx::shape sm_shape{migraphx::shape::float_type,
                                     {{batch_size, batch_size}, {4, 20}}};
            auto sm_input = submod->add_parameter("data", sm_shape);
            migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {1}}};
            auto literal_ins = submod->add_literal(migraphx::literal{lit_s, {6}});
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

        migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {4, 20}}};
        auto input0                             = mm1->add_parameter("data", s);
        std::vector<migraphx::shape> sub_shapes = {};
        sub_shapes.push_back(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {4, 20}}});
        migraphx::shape out_attr = migraphx::shape{sub_shapes};
        auto sm_ins              = mm1->add_instruction(
            migraphx::make_op("select_module",
                                           {{"output_dyn_shapes", migraphx::to_value(out_attr)}}),
            {input0},
            {dim1, dim2, dim3, dim4});
        auto ret =
            mm1->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), sm_ins);
        mm1->add_return({ret});
    }

    EXPECT(p0 == p1);
}

// contrived example where each submodule to select_module outputs the same static shape.
TEST_CASE(select_module_update2)
{
    migraphx::program p0;
    {
        auto* mm0 = p0.get_main_module();

        // create batch submodules
        auto create_submodule = [&](std::size_t batch_size, const std::string& module_name) {
            auto* submod = p0.create_module(module_name);
            migraphx::shape sm_shape{migraphx::shape::float_type, {batch_size, 4}};
            auto sm_input   = submod->add_parameter("data", sm_shape);
            auto slice_data = submod->add_instruction(
                migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}),
                sm_input);
            submod->add_return({slice_data});
            return submod;
            // output shape is static shape with lens={1, 4}
        };
        auto* dim1 = create_submodule(1, "dim_1");
        auto* dim2 = create_submodule(2, "dim_2");
        auto* dim3 = create_submodule(3, "dim_3");
        auto* dim4 = create_submodule(4, "dim_4");

        migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {4, 4}}};
        auto input0                             = mm0->add_parameter("data", s);
        std::vector<migraphx::shape> sub_shapes = {};
        auto max_int                            = std::numeric_limits<std::size_t>::max();
        sub_shapes.push_back(migraphx::shape{migraphx::shape::float_type, {{0, max_int}, {4, 4}}});
        // {0, max_int} dimension for `output_dyn_shapes` attribute will be simplified to
        // a fixed shape of {1, 4}
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
    migraphx::run_passes(p0, {migraphx::simplify_dyn_ops{}, migraphx::dead_code_elimination{}});

    migraphx::program p1;
    {
        auto* mm1 = p1.get_main_module();

        // create batch submodules
        auto create_submodule = [&](std::size_t batch_size, const std::string& module_name) {
            auto* submod = p1.create_module(module_name);
            migraphx::shape sm_shape{migraphx::shape::float_type, {batch_size, 4}};
            auto sm_input   = submod->add_parameter("data", sm_shape);
            auto slice_data = submod->add_instruction(
                migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}),
                sm_input);
            submod->add_return({slice_data});
            return submod;
            // output shape is static shape with lens={1, 4}
        };
        auto* dim1 = create_submodule(1, "dim_1");
        auto* dim2 = create_submodule(2, "dim_2");
        auto* dim3 = create_submodule(3, "dim_3");
        auto* dim4 = create_submodule(4, "dim_4");

        migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {4, 4}}};
        auto input0                             = mm1->add_parameter("data", s);
        std::vector<migraphx::shape> sub_shapes = {};
        // note single static shape output
        sub_shapes.push_back(migraphx::shape{migraphx::shape::float_type, {1, 4}});
        migraphx::shape out_attr = migraphx::shape{sub_shapes};
        auto sm_ins              = mm1->add_instruction(
            migraphx::make_op("select_module",
                                           {{"output_dyn_shapes", migraphx::to_value(out_attr)}}),
            {input0},
            {dim1, dim2, dim3, dim4});
        auto ret =
            mm1->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), sm_ins);
        mm1->add_return({ret});
    }

    EXPECT(p0 == p1);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
