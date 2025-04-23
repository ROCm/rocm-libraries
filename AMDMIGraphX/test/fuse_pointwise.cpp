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
#include <migraphx/fuse_pointwise.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/eliminate_contiguous.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>

#include <test.hpp>
#include <pointwise.hpp>

static void run_pass(migraphx::program& p, migraphx::fuse_pointwise pass = {})
{
    migraphx::run_passes(p, {pass, migraphx::dead_code_elimination{}});
}

TEST_CASE(single)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto z    = mm->add_parameter("z", s);
        auto add1 = mm->add_instruction(migraphx::make_op("add"), x, y);
        auto pass = mm->add_instruction(pass_op{}, add1);
        auto add2 = mm->add_instruction(migraphx::make_op("add"), pass, z);
        mm->add_return({add2});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto z    = mm->add_parameter("z", s);
        auto add1 = add_pointwise(p2, "main:pointwise0", {x, y}, single_pointwise("add"));
        auto pass = mm->add_instruction(pass_op{}, add1);
        auto add2 = add_pointwise(p2, "main:pointwise1", {pass, z}, single_pointwise("add"));
        mm->add_return({add2});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(double_add)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto z    = mm->add_parameter("z", s);
        auto add1 = mm->add_instruction(migraphx::make_op("add"), x, y);
        auto add2 = mm->add_instruction(migraphx::make_op("add"), add1, z);
        mm->add_return({add2});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        auto z   = mm->add_parameter("z", s);
        auto fadd =
            add_pointwise(p2, "main:pointwise0", {x, y, z}, [=](auto* pm, const auto& inputs) {
                auto add1 = pm->add_instruction(migraphx::make_op("add"), inputs[0], inputs[1]);
                return pm->add_instruction(migraphx::make_op("add"), add1, inputs[2]);
            });
        mm->add_return({fadd});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(double_add_without_return)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto z    = mm->add_parameter("z", s);
        auto add1 = mm->add_instruction(migraphx::make_op("add"), x, y);
        mm->add_instruction(migraphx::make_op("add"), add1, z);
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        auto z   = mm->add_parameter("z", s);
        add_pointwise(p2, "main:pointwise0", {x, y, z}, [=](auto* pm, const auto& inputs) {
            auto add1 = pm->add_instruction(migraphx::make_op("add"), inputs[0], inputs[1]);
            return pm->add_instruction(migraphx::make_op("add"), add1, inputs[2]);
        });
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(used_twice_not_fused)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto add1 = mm->add_instruction(migraphx::make_op("add"), x, y);
        auto pass = mm->add_instruction(pass_op{}, add1);
        auto add2 = mm->add_instruction(migraphx::make_op("add"), add1, y);
        auto add3 = mm->add_instruction(migraphx::make_op("add"), pass, add2);
        mm->add_return({add3});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto add1 = add_pointwise(p2, "main:pointwise0", {x, y}, single_pointwise("add"));
        auto pass = mm->add_instruction(pass_op{}, add1);
        auto fadd = add_pointwise(
            p2, "main:pointwise1", {add1, y, pass}, [=](auto* pm, const auto& inputs) {
                auto add2 = pm->add_instruction(migraphx::make_op("add"), inputs[0], inputs[1]);
                return pm->add_instruction(migraphx::make_op("add"), inputs[2], add2);
            });
        mm->add_return({fadd});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(used_twice_fused)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto add1 = mm->add_instruction(migraphx::make_op("add"), x, y);
        auto add2 = mm->add_instruction(migraphx::make_op("add"), add1, x);
        auto add3 = mm->add_instruction(migraphx::make_op("add"), add1, y);
        auto add4 = mm->add_instruction(migraphx::make_op("add"), add2, add3);
        mm->add_return({add4});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto fadd = add_pointwise(p2, "main:pointwise0", {x, y}, [=](auto* pm, const auto& inputs) {
            auto add1 = pm->add_instruction(migraphx::make_op("add"), inputs[0], inputs[1]);
            auto add2 = pm->add_instruction(migraphx::make_op("add"), add1, inputs[0]);
            auto add3 = pm->add_instruction(migraphx::make_op("add"), add1, inputs[1]);
            return pm->add_instruction(migraphx::make_op("add"), add2, add3);
        });
        mm->add_return({fadd});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(duplicate_inputs)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto add1 = mm->add_instruction(migraphx::make_op("add"), x, x);
        auto pass = mm->add_instruction(pass_op{}, add1);
        auto add2 = mm->add_instruction(migraphx::make_op("add"), pass, y);
        mm->add_return({add2});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto add1 = add_pointwise(p2, "main:pointwise0", {x}, [=](auto* pm, const auto& inputs) {
            return pm->add_instruction(migraphx::make_op("add"), inputs[0], inputs[0]);
        });
        auto pass = mm->add_instruction(pass_op{}, add1);
        auto add2 = add_pointwise(p2, "main:pointwise1", {pass, y}, single_pointwise("add"));
        mm->add_return({add2});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(scalar_input)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto one = mm->add_literal(1.0f);
        auto y =
            mm->add_instruction(migraphx::make_op("scalar", {{"scalar_bcst_dims", s.lens()}}), one);
        auto add1 = mm->add_instruction(migraphx::make_op("add"), x, y);
        mm->add_return({add1});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto add1 = add_pointwise(p2, "main:pointwise0", {x}, [=](auto* pm, const auto& inputs) {
            auto y = pm->add_literal(1.0f);
            return pm->add_instruction(migraphx::make_op("add"), inputs[0], y);
        });
        mm->add_return({add1});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(scalar_like_input)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 1}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto one = mm->add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1}}, {1.0f}});
        auto y =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), one);
        auto add1 = mm->add_instruction(migraphx::make_op("add"), x, y);
        mm->add_return({add1});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto add1 = add_pointwise(p2, "main:pointwise0", {x}, [=](auto* pm, const auto& inputs) {
            auto y = pm->add_literal(1.0f);
            return pm->add_instruction(migraphx::make_op("add"), inputs[0], y);
        });
        mm->add_return({add1});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(contiguous_input)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto one = mm->add_literal(1.0f);
        auto yb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), one);
        auto y    = mm->add_instruction(migraphx::make_op("contiguous"), yb);
        auto add1 = mm->add_instruction(migraphx::make_op("add"), x, y);
        mm->add_return({add1});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto add1 = add_pointwise(p2, "main:pointwise0", {x}, [=](auto* pm, const auto& inputs) {
            auto y = pm->add_literal(1.0f);
            return pm->add_instruction(migraphx::make_op("add"), inputs[0], y);
        });
        mm->add_return({add1});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(contiguous_boolean_input)
{

    migraphx::shape s{migraphx::shape::bool_type, {2, 3}};
    migraphx::shape s_lit{migraphx::shape::bool_type, {1}, {0}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto one = mm->add_literal(migraphx::literal(s_lit, {1.0}));
        auto yb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), one);
        auto y    = mm->add_instruction(migraphx::make_op("contiguous"), yb);
        auto xor1 = mm->add_instruction(migraphx::make_op("logical_xor"), x, y);
        mm->add_return({xor1});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto xor1 = add_pointwise(p2, "main:pointwise0", {x}, [=](auto* pm, const auto& inputs) {
            auto y = pm->add_literal(migraphx::literal(s_lit, {1}));
            return pm->add_instruction(migraphx::make_op("logical_xor"), inputs[0], y);
        });
        mm->add_return({xor1});
    }
}

TEST_CASE(all_scalar_input)
{
    migraphx::shape s{migraphx::shape::float_type};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto add1 = mm->add_instruction(migraphx::make_op("add"), x, y);
        mm->add_return({add1});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto add1 = add_pointwise(p2, "main:pointwise0", {x, y}, [=](auto* pm, const auto& inputs) {
            return pm->add_instruction(migraphx::make_op("add"), inputs[0], inputs[1]);
        });
        mm->add_return({add1});
    }
    EXPECT(p1.get_output_shapes().size() == 1);
    EXPECT(p1.get_output_shapes().front().scalar());
    EXPECT(p1.get_output_shapes() == p2.get_output_shapes());
    EXPECT(p1 == p2);
}

TEST_CASE(no_input)
{
    migraphx::program p;
    {
        auto* mm = p.get_main_module();
        migraphx::shape g_shape{migraphx::shape::int64_type, {1}, {0}};
        migraphx::shape s_indices{migraphx::shape::int32_type, {3}};
        std::vector<int> indices{3, 800, 800};
        auto a0  = mm->add_literal(migraphx::literal{s_indices, indices});
        auto a1  = mm->add_literal(migraphx::literal{g_shape, {1}});
        int axis = 0;
        auto out = mm->add_instruction(migraphx::make_op("gather", {{"axis", axis}}), a0, a1);
        mm->add_return({out});
    }
    run_pass(p);

    // This should NOT create a pointwise module if there are no inputs here.
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        migraphx::shape g_shape{migraphx::shape::int64_type, {1}, {0}};
        migraphx::shape s_indices{migraphx::shape::int32_type, {3}};
        std::vector<int> indices{3, 800, 800};
        auto a0  = mm->add_literal(migraphx::literal{s_indices, indices});
        auto a1  = mm->add_literal(migraphx::literal{g_shape, {1}});
        int axis = 0;
        auto out = mm->add_instruction(migraphx::make_op("gather", {{"axis", axis}}), a0, a1);
        mm->add_return({out});
    }
    EXPECT(p == p2);
}

TEST_CASE(add_reshape_add)
{
    migraphx::shape s1{migraphx::shape::float_type, {3, 10, 16}};
    migraphx::shape s2{migraphx::shape::float_type, {3, 40, 2, 2}};
    migraphx::shape s3{migraphx::shape::float_type, {3, 10, 4, 2, 2}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s1);
        auto y    = mm->add_parameter("y", s1);
        auto z    = mm->add_parameter("z", s2);
        auto add1 = mm->add_instruction(migraphx::make_op("add"), x, y);
        auto reshape =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", s2.lens()}}), add1);
        auto add2 = mm->add_instruction(migraphx::make_op("add"), reshape, z);
        mm->add_return({add2});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s1);
        auto y   = mm->add_parameter("y", s1);
        auto z   = mm->add_parameter("z", s2);
        auto x2  = mm->add_instruction(migraphx::make_op("reshape", {{"dims", s3.lens()}}), x);
        auto y2  = mm->add_instruction(migraphx::make_op("reshape", {{"dims", s3.lens()}}), y);
        auto z2  = mm->add_instruction(migraphx::make_op("reshape", {{"dims", s3.lens()}}), z);
        auto fadd =
            add_pointwise(p2, "main:pointwise0", {x2, y2, z2}, [=](auto* pm, const auto& inputs) {
                auto add1 = pm->add_instruction(migraphx::make_op("add"), inputs[0], inputs[1]);
                return pm->add_instruction(migraphx::make_op("add"), add1, inputs[2]);
            });
        auto reshape =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", s2.lens()}}), fadd);
        mm->add_return({reshape});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(add_transpose_reshape_add)
{
    migraphx::shape s1{migraphx::shape::float_type, {3, 16, 10}};
    migraphx::shape s2{migraphx::shape::float_type, {3, 40, 2, 2}};
    migraphx::shape s3{migraphx::shape::float_type, {3, 10, 4, 2, 2}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s1);
        auto y    = mm->add_parameter("y", s1);
        auto z    = mm->add_parameter("z", s2);
        auto add1 = mm->add_instruction(migraphx::make_op("add"), x, y);
        auto transpose =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), add1);
        auto reshape =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", s2.lens()}}), transpose);
        auto add2 = mm->add_instruction(migraphx::make_op("add"), reshape, z);
        mm->add_return({add2});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s1);
        auto y   = mm->add_parameter("y", s1);
        auto z   = mm->add_parameter("z", s2);
        auto x2 =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {3, 4, 2, 2, 10}}}), x);
        auto x3 = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 4, 1, 2, 3}}}), x2);
        auto y2 =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {3, 4, 2, 2, 10}}}), y);
        auto y3 = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 4, 1, 2, 3}}}), y2);
        auto z2 = mm->add_instruction(migraphx::make_op("reshape", {{"dims", s3.lens()}}), z);
        auto fadd =
            add_pointwise(p2, "main:pointwise0", {x3, y3, z2}, [=](auto* pm, const auto& inputs) {
                auto add1 = pm->add_instruction(migraphx::make_op("add"), inputs[0], inputs[1]);
                return pm->add_instruction(migraphx::make_op("add"), add1, inputs[2]);
            });
        auto reshape =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", s2.lens()}}), fadd);
        mm->add_return({reshape});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(add_contiguous_reshape_add)
{
    auto s1 =
        migraphx::shape::from_permutation(migraphx::shape::float_type, {3, 10, 16}, {0, 2, 1});
    auto s2 = migraphx::shape{migraphx::shape::float_type, {3, 40, 2, 2}};
    auto s3 = migraphx::shape{migraphx::shape::float_type, {3, 10, 4, 2, 2}};
    migraphx::program p1;
    {
        auto* mm        = p1.get_main_module();
        auto x          = mm->add_parameter("x", s1);
        auto y          = mm->add_parameter("y", s1);
        auto z          = mm->add_parameter("z", s2);
        auto add1       = mm->add_instruction(migraphx::make_op("add"), x, y);
        auto contiguous = mm->add_instruction(migraphx::make_op("contiguous"), add1);
        auto reshape =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", s2.lens()}}), contiguous);
        auto add2 = mm->add_instruction(migraphx::make_op("add"), reshape, z);
        mm->add_return({add2});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s1);
        auto y   = mm->add_parameter("y", s1);
        auto z   = mm->add_parameter("z", s2);
        auto x2  = mm->add_instruction(migraphx::make_op("reshape", {{"dims", s3.lens()}}), x);
        auto y2  = mm->add_instruction(migraphx::make_op("reshape", {{"dims", s3.lens()}}), y);
        auto z2  = mm->add_instruction(migraphx::make_op("reshape", {{"dims", s3.lens()}}), z);
        auto fadd =
            add_pointwise(p2, "main:pointwise0", {x2, y2, z2}, [=](auto* pm, const auto& inputs) {
                auto add1 = pm->add_instruction(migraphx::make_op("add"), inputs[0], inputs[1]);
                return pm->add_instruction(migraphx::make_op("add"), add1, inputs[2]);
            });
        auto reshape =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", s2.lens()}}), fadd);
        mm->add_return({reshape});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(add_reshape_add_nonstandard)
{
    migraphx::shape s1 =
        migraphx::shape::from_permutation(migraphx::shape::float_type, {3, 10, 16}, {2, 0, 1});
    migraphx::shape s2{migraphx::shape::float_type, {3, 40, 2, 2}};
    migraphx::shape s3{migraphx::shape::float_type, {3, 10, 4, 2, 2}};
    migraphx::program p1;
    {
        auto* mm     = p1.get_main_module();
        auto x       = mm->add_parameter("x", s1);
        auto y       = mm->add_parameter("y", s1);
        auto z       = mm->add_parameter("z", s2);
        auto add1    = mm->add_instruction(migraphx::make_op("add"), x, y);
        auto reshape =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", s2.lens()}}), add1);
        auto add2    = mm->add_instruction(migraphx::make_op("add"), reshape, z);
        mm->add_return({add2});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s1);
        auto y   = mm->add_parameter("y", s1);
        auto z   = mm->add_parameter("z", s2);
        auto x2  = mm->add_instruction(migraphx::make_op("reshape", {{"dims", s3.lens()}}), x);
        auto y2  = mm->add_instruction(migraphx::make_op("reshape", {{"dims", s3.lens()}}), y);
        auto z2  = mm->add_instruction(migraphx::make_op("reshape", {{"dims", s3.lens()}}), z);
        auto fadd =
            add_pointwise(p2, "main:pointwise0", {x2, y2, z2}, [=](auto* pm, const auto& inputs) {
                auto add1 = pm->add_instruction(migraphx::make_op("add"), inputs[0], inputs[1]);
                return pm->add_instruction(migraphx::make_op("add"), add1, inputs[2]);
            });
        auto reshape =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", s2.lens()}}), fadd);
        mm->add_return({reshape});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(add_unsqueeze_add_nonstandard)
{
    migraphx::shape s1 =
        migraphx::shape::from_permutation(migraphx::shape::float_type, {3, 10, 16}, {2, 0, 1});
    migraphx::shape s2{migraphx::shape::float_type, {3, 10, 1, 16}};
    migraphx::program p1;
    {
        auto* mm       = p1.get_main_module();
        auto x         = mm->add_parameter("x", s1);
        auto y         = mm->add_parameter("y", s1);
        auto z         = mm->add_parameter("z", s2);
        auto add1      = mm->add_instruction(migraphx::make_op("add"), x, y);
        auto unsqueeze = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), add1);
        auto add2      = mm->add_instruction(migraphx::make_op("add"), unsqueeze, z);
        mm->add_return({add2});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s1);
        auto y   = mm->add_parameter("y", s1);
        auto z   = mm->add_parameter("z", s2);
        auto x2  = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), x);
        auto y2  = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), y);
        auto fadd =
            add_pointwise(p2, "main:pointwise0", {x2, y2, z}, [=](auto* pm, const auto& inputs) {
                auto add1 = pm->add_instruction(migraphx::make_op("add"), inputs[0], inputs[1]);
                return pm->add_instruction(migraphx::make_op("add"), add1, inputs[2]);
            });
        mm->add_return({fadd});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(add_reshape_add_error)
{
    migraphx::shape s1{migraphx::shape::float_type, {6, 35}};
    migraphx::shape s2{migraphx::shape::float_type, {3, 7, 2, 5}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s1);
        auto y    = mm->add_parameter("y", s1);
        auto z    = mm->add_parameter("z", s2);
        auto add1 = mm->add_instruction(migraphx::make_op("add"), x, y);
        auto reshape =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", s2.lens()}}), add1);
        auto add2 = mm->add_instruction(migraphx::make_op("add"), reshape, z);
        mm->add_return({add2});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto x     = mm->add_parameter("x", s1);
        auto y     = mm->add_parameter("y", s1);
        auto z     = mm->add_parameter("z", s2);
        auto fadd1 = add_pointwise(p2, "main:pointwise0", {x, y}, single_pointwise("add"));
        auto reshape =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", s2.lens()}}), fadd1);
        auto fadd2 = add_pointwise(p2, "main:pointwise1", {reshape, z}, single_pointwise("add"));
        mm->add_return({fadd2});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(add_broadcast_add)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 1}};
    migraphx::shape s2{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s1);
        auto y     = mm->add_parameter("y", s1);
        auto z     = mm->add_parameter("z", s2);
        auto add1  = mm->add_instruction(migraphx::make_op("add"), x, y);
        auto badd1 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s2.lens()}}), add1);
        auto add2 = mm->add_instruction(migraphx::make_op("add"), badd1, z);
        mm->add_return({add2});
    }
    run_pass(p1, {.enable_rewrite_broadcasts = true});
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s1);
        auto y   = mm->add_parameter("y", s1);
        auto z   = mm->add_parameter("z", s2);
        auto bx =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s2.lens()}}), x);
        auto by =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s2.lens()}}), y);
        auto fadd =
            add_pointwise(p2, "main:pointwise0", {bx, by, z}, [=](auto* pm, const auto& inputs) {
                auto add1 = pm->add_instruction(migraphx::make_op("add"), inputs[0], inputs[1]);
                return pm->add_instruction(migraphx::make_op("add"), add1, inputs[2]);
            });
        mm->add_return({fadd});
    }
    EXPECT(p1.sort() == p2.sort());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
