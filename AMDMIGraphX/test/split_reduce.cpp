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
#include <migraphx/split_reduce.hpp>
#include <migraphx/fuse_pointwise.hpp>
#include <migraphx/fuse_reduce.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>

#include <test.hpp>
#include <pointwise.hpp>
#include <reduce.hpp>

static void run_pass(migraphx::program& p)
{
    migraphx::run_passes(p,
                         {migraphx::fuse_pointwise{},
                          migraphx::fuse_reduce{},
                          migraphx::split_reduce{.split_size = 8192},
                          migraphx::fuse_pointwise{.enable_rewrite_broadcasts = true},
                          migraphx::dead_code_elimination{}});
}

static void run_fuse_pass(migraphx::program& p)
{
    migraphx::run_passes(
        p,
        {migraphx::fuse_pointwise{}, migraphx::fuse_reduce{}, migraphx::dead_code_elimination{}});
}

TEST_CASE(single)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 327680}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), x);
        mm->add_return({rsum});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto rsum = add_reduce(
            p2, "main:reduce_sum0_split", {x}, {2}, "assign_add", single_reduce("reduce_sum"));
        mm->add_return({rsum});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(fused)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 327680}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto rsum  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), x);
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum);
        auto add = mm->add_instruction(migraphx::make_op("add"), x, rsumb);
        mm->add_return({add});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto rsum  = add_reduce(p2,
                               "main:reduce_sum0:main:pointwise0_split",
                                {x},
                                {2},
                               "assign_add",
                               single_reduce("reduce_sum"));
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum);
        auto add = add_pointwise(p2, mm, "main:pointwise0", {x, rsumb}, single_pointwise("add"));
        mm->add_return({add});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(small)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 1024}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto rsum  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), x);
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum);
        auto add = mm->add_instruction(migraphx::make_op("add"), x, rsumb);
        mm->add_return({add});
    }
    migraphx::program p2 = p1;
    run_fuse_pass(p2);
    run_pass(p1);

    EXPECT(p1 == p2);
}

TEST_CASE(split_pointwise)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 327680}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto sqrt  = mm->add_instruction(migraphx::make_op("sqrt"), x);
        auto rsum  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), sqrt);
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum);
        auto add = mm->add_instruction(migraphx::make_op("add"), sqrt, rsumb);
        mm->add_return({add});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto sqrt  = add_pointwise(p2, mm, "main:pointwise0", {x}, single_pointwise("sqrt"));
        auto rsum  = add_reduce(p2,
                               "main:pointwise0:main:reduce_sum0:main:pointwise1_split",
                                {sqrt},
                                {2},
                               "assign_add",
                               single_reduce("reduce_sum"));
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum);
        auto add = add_pointwise(p2, mm, "main:pointwise1", {sqrt, rsumb}, single_pointwise("add"));
        mm->add_return({add});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(sequence_reduces)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 327680}};
    migraphx::program p1;
    {
        auto* mm    = p1.get_main_module();
        auto x      = mm->add_parameter("x", s);
        auto rsum1  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), x);
        auto rsum1b = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum1);
        auto sub    = mm->add_instruction(migraphx::make_op("sub"), x, rsum1b);
        auto rsum2  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), sub);
        auto rsum2b = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum2);
        auto add = mm->add_instruction(migraphx::make_op("add"), rsum2b, x);
        mm->add_return({add});
    }
    migraphx::program p2 = p1;
    run_fuse_pass(p2);
    run_pass(p1);

    EXPECT(p1 == p2);
}

TEST_CASE(parallel_reduce)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 327680}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto xx    = mm->add_instruction(migraphx::make_op("mul"), x, x);
        auto rsum1 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), x);
        auto rsum2 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), xx);
        auto mul   = mm->add_instruction(migraphx::make_op("mul"), rsum1, rsum2);
        mm->add_return({mul});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto rsum = add_reduce(
            p2,
            "main:reduce_sum0:main:pointwise1:main:pointwise0:main:reduce_sum1_split",
            {x},
            {2},
            "assign_add",
            [&](auto* rm,
                const auto& inputs,
                const auto& axes) -> std::vector<migraphx::instruction_ref> {
                auto xx    = add_pointwise(p2, rm, "main:pointwise0", {inputs[0]}, squared());
                auto rsum1 = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                 inputs[0]);
                auto rsum2 =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), xx);
                return {rsum2, rsum1};
            });
        auto rsum2 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), rsum);
        auto rsum1 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), rsum);
        auto mul =
            add_pointwise(p2, mm, "main:pointwise1", {rsum1, rsum2}, single_pointwise("mul"));
        mm->add_return({mul});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(double_split_live)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 327680}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto rsum = add_reduce(
            p1, "fuse_reduce0", {x}, {2}, [&](auto* rm, const auto& inputs, const auto& axes) {
                auto rsum1 = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                 inputs[0]);
                auto sqrt =
                    add_pointwise(p1, rm, "main:pointwise0", {rsum1}, single_pointwise("sqrt"));
                auto sqrtb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), sqrt);
                auto mul = add_pointwise(p1, rm, "main:pointwise1", {inputs[0]}, squared());
                auto rsum2 =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), mul);
                auto add = add_pointwise(
                    p1, rm, "main:pointwise2", {rsum2, sqrt}, single_pointwise("add"));
                auto addb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), add);
                return add_pointwise(
                    p1, rm, "main:pointwise3", {addb, sqrtb}, single_pointwise("mul"));
            });
        mm->add_return({rsum});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto rsums = add_reduce(
            p2,
            "fuse_reduce0_split",
            {x},
            {2},
            "assign_add",
            [&](auto* rm,
                const auto& inputs,
                const auto& axes) -> std::vector<migraphx::instruction_ref> {
                auto rsum1 = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                 inputs[0]);
                auto mul   = add_pointwise(p2, rm, "main:pointwise1", {inputs[0]}, squared());
                auto rsum2 =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), mul);
                return {rsum1, rsum2};
            });
        auto rsum1 =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), rsums);
        auto rsum2 =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), rsums);
        auto rsum1b = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum1);
        auto rsum2b = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum2);
        auto sqrt_add_mul = add_pointwise(
            p2, "main:pointwise0", {rsum1b, rsum2b}, [](auto* pm, const auto& inputs) {
                auto sqrt = pm->add_instruction(migraphx::make_op("sqrt"), inputs[0]);
                auto add  = pm->add_instruction(migraphx::make_op("add"), inputs[1], sqrt);
                return pm->add_instruction(migraphx::make_op("mul"), add, sqrt);
            });
        mm->add_return({sqrt_add_mul});
    }
    EXPECT(p1.sort() == p2.sort());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
