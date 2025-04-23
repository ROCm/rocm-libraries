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
#include <migraphx/fuse_reduce.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>

#include <test.hpp>
#include <reduce.hpp>
#include <pointwise.hpp>

static void run_pass(migraphx::program& p)
{
    migraphx::run_passes(p, {migraphx::fuse_reduce{}, migraphx::dead_code_elimination{}});
}

TEST_CASE(single)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto y     = mm->add_parameter("y", s);
        auto rsum1 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto rsum2 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), y);
        mm->add_return({rsum1, rsum2});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto y     = mm->add_parameter("y", s);
        auto rsum1 = add_reduce(p2, "main:reduce_sum0", {x}, {1}, single_reduce("reduce_sum"));
        auto rsum2 = add_reduce(p2, "main:reduce_sum1", {y}, {1}, single_reduce("reduce_sum"));
        mm->add_return({rsum1, rsum2});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(pointwise_reduce)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto add  = add_pointwise(p1, "main:pointwise0", {x, y}, single_pointwise("add"));
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), add);
        mm->add_return({rsum});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto rsum = add_reduce(
            p2,
            "main:pointwise0:main:reduce_sum0",
            {x, y},
            {1},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto add =
                    add_pointwise(p2, rm, "main:pointwise0", inputs, single_pointwise("add"));
                return rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), add);
            });
        mm->add_return({rsum});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(pointwise_reduce_unfusable_broadcast)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 1, 3}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto add  = add_pointwise(p1, "main:pointwise0", {x, y}, single_pointwise("add"));
        auto addb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 3}}}), add);
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), addb);
        mm->add_return({rsum});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto add  = add_pointwise(p2, "main:pointwise0", {x, y}, single_pointwise("add"));
        auto addb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 3}}}), add);
        auto rsum =
            add_reduce(p2,
                       "main:reduce_sum0",
                       {addb},
                       {2},
                       [&](auto* rm, const auto& inputs, const auto& axes) {
                           return rm->add_instruction(
                               migraphx::make_op("reduce_sum", {{"axes", axes}}), inputs[0]);
                       });
        mm->add_return({rsum});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(scalar_multibroadcast)
{
    // Matches the find_pointwise_reduce matcher, but input x has a (scalar) shape
    // incompatible with the multibroadcast instruction; therefore it
    // creates a fused_reduce module but does not add a submodule for the
    // multibroadcast instruction.
    migraphx::shape sdot{migraphx::shape::double_type, {80, 204, 204}};
    migraphx::shape sdot_double{migraphx::shape::double_type, {80, 204, 204}};
    migraphx::shape scalar{migraphx::shape::double_type, {1}, {0}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", scalar);
        auto zap = add_pointwise(p1, "main:pointwise0", {x}, single_pointwise("sqrt"));
        auto pow = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sdot.lens()}}), zap);
        auto bip = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1, 2}}}), pow);

        mm->add_return({bip});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", scalar);
        auto zap = add_pointwise(p2, mm, "main:pointwise0", {x}, single_pointwise("sqrt"));

        auto pow = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sdot.lens()}}), zap);

        // Add a reduce module.  These are created by fuse_reduce::apply() for any reduce
        // instruction whether the individual matchers do anything or not.
        auto* reduce_mod = p2.create_module("main:reduce_sum0");
        auto x0          = reduce_mod->add_parameter("x0", sdot_double);
        auto sqrtbc =
            reduce_mod->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1, 2}}}), x0);
        reduce_mod->add_return({sqrtbc});

        EXPECT(test::throws([&] {
            mm->add_instruction(
                migraphx::make_op("fused_reduce", {{"axes", {1, 2}}}), {pow}, {reduce_mod});
        }));
        // reduce modules must be flagged for bypass when running subsequent passes
        reduce_mod->set_bypass();
        auto bip = mm->add_instruction(
            migraphx::make_op("fused_reduce", {{"axes", {1, 2}}}), {pow}, {reduce_mod});
        mm->add_return({bip});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(scalar_multibroadcast_contiguous)
{
    // Contains a contiguous op which is not passed through.
    migraphx::shape sdot{migraphx::shape::double_type, {80, 204, 204}};
    migraphx::shape scalar{migraphx::shape::double_type, {1}, {0}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", scalar);
        auto zap = add_pointwise(p1, "main:pointwise0", {x}, single_pointwise("sqrt"));
        auto pow = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sdot.lens()}}), zap);
        auto bip    = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1, 2}}}), pow);
        auto sqrtbc = mm->add_instruction(migraphx::make_op("contiguous"), bip);

        mm->add_return({sqrtbc});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", scalar);
        auto zap = add_pointwise(p2, mm, "main:pointwise0", {x}, single_pointwise("sqrt"));

        auto pow = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sdot.lens()}}), zap);

        // Add a reduce module.  These are created by fuse_reduce::apply() for any reduce
        // instruction whether the individual matchers do anything or not.
        auto* reduce_mod = p2.create_module("main:reduce_sum0");

        auto x0 = reduce_mod->add_parameter("x0", sdot);
        auto sqrtbc =
            reduce_mod->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1, 2}}}), x0);
        reduce_mod->add_return({sqrtbc});

        EXPECT(test::throws([&] {
            mm->add_instruction(
                migraphx::make_op("fused_reduce", {{"axes", {1, 2}}}), {pow}, {reduce_mod});
        }));
        // reduce modules must be flagged for bypass when running subsequent passes
        reduce_mod->set_bypass();
        auto bip = mm->add_instruction(
            migraphx::make_op("fused_reduce", {{"axes", {1, 2}}}), {pow}, {reduce_mod});
        mm->add_return({bip});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(pointwise_broadcast_reduce_reshape)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::shape rs{migraphx::shape::float_type, {2, 1}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", rs);
        auto sqrt  = add_pointwise(p1, "main:pointwise0", {x}, single_pointwise("sqrt"));
        auto sqrtb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), sqrt);
        auto rsum  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), sqrtb);
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum);
        auto add = add_pointwise(p1, "main:pointwise1", {sqrtb, rsumb}, single_pointwise("add"));
        auto reshape = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {6}}}), add);
        mm->add_return({reshape});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", rs);
        auto add = add_reduce(
            p2,
            "main:pointwise0:main:reduce_sum0:main:pointwise1",
            {x},
            {1},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto sqrt =
                    add_pointwise(p2, rm, "main:pointwise0", inputs, single_pointwise("sqrt"));
                auto sqrtb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), sqrt);
                auto rsum =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), sqrtb);
                auto rsumb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum);
                return add_pointwise(
                    p2, rm, "main:pointwise1", {sqrtb, rsumb}, single_pointwise("add"));
            });
        auto reshape = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {6}}}), add);
        mm->add_return({reshape});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(reduce_pointwise)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto y     = mm->add_parameter("y", s);
        auto rsum  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum);
        auto add = add_pointwise(p1, "main:pointwise0", {rsumb, y}, single_pointwise("add"));
        mm->add_return({add});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        auto add = add_reduce(
            p2,
            "main:reduce_sum0:main:pointwise0",
            {x, y},
            {1},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto rsum  = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                inputs[0]);
                auto rsumb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum);
                return add_pointwise(
                    p2, rm, "main:pointwise0", {rsumb, inputs[1]}, single_pointwise("add"));
            });
        mm->add_return({add});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(reduce_pointwise_unfusable_broadcast)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 1, 3}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto y     = mm->add_parameter("y", s);
        auto rsum  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), x);
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 3}}}), rsum);
        auto yb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 3}}}), y);
        auto add = add_pointwise(p1, "main:pointwise0", {rsumb, yb}, single_pointwise("add"));
        mm->add_return({add});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto rsum = add_reduce(
            p2, "main:reduce_sum0", {x}, {2}, [&](auto* rm, const auto& inputs, const auto& axes) {
                return rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                           inputs[0]);
            });
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 3}}}), rsum);
        auto yb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 3}}}), y);
        auto add = add_pointwise(p2, "main:pointwise0", {rsumb, yb}, single_pointwise("add"));
        mm->add_return({add});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(reduce_reduce)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto rsum  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum);
        auto rsumdiff = add_pointwise(p1, "main:pointwise0", {rsumb, x}, single_pointwise("sub"));
        auto rsum2 =
            mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), rsumdiff);
        auto sqrt = add_pointwise(p1, "main:pointwise1", {rsum2}, single_pointwise("sqrt"));
        mm->add_return({sqrt});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto sqrt = add_reduce(
            p2,
            "main:reduce_sum1:main:reduce_sum0:main:pointwise0:main:pointwise1",
            {x},
            {1},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto rsum  = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                inputs[0]);
                auto rsumb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum);
                auto rsumdiff = add_pointwise(
                    p2, rm, "main:pointwise0", {rsumb, inputs[0]}, single_pointwise("sub"));
                auto rsum2 = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                 rsumdiff);
                return add_pointwise(p2, rm, "main:pointwise1", {rsum2}, single_pointwise("sqrt"));
            });
        mm->add_return({sqrt});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(reduce_reduce_unfusable_broadcast)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 1, 3}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto rsum  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), x);
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 3}}}), rsum);
        auto xb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 3}}}), x);
        auto rsumdiff = add_pointwise(p1, "main:pointwise0", {rsumb, xb}, single_pointwise("sub"));
        auto rsum2 =
            mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), rsumdiff);
        auto sqrt = add_pointwise(p1, "main:pointwise1", {rsum2}, single_pointwise("sqrt"));
        mm->add_return({sqrt});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto rsum = add_reduce(
            p2, "main:reduce_sum0", {x}, {2}, [&](auto* rm, const auto& inputs, const auto& axes) {
                return rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                           inputs[0]);
            });

        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 3}}}), rsum);
        auto xb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 3}}}), x);

        auto sqrt = add_reduce(
            p2,
            "main:pointwise0:main:reduce_sum1:main:pointwise1",
            {rsumb, xb},
            {2},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto rsumdiff = add_pointwise(
                    p2, rm, "main:pointwise0", {inputs[0], inputs[1]}, single_pointwise("sub"));
                auto rsum2 = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                 rsumdiff);
                return add_pointwise(p2, rm, "main:pointwise1", {rsum2}, single_pointwise("sqrt"));
            });
        mm->add_return({sqrt});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(parallel_reduce_reduce)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm    = p1.get_main_module();
        auto x      = mm->add_parameter("x", s);
        auto xx     = add_pointwise(p1, "main:pointwise0", {x, x}, single_pointwise("mul"));
        auto rsumx  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto rsumxx = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), xx);
        auto add = add_pointwise(p1, "main:pointwise1", {rsumx, rsumxx}, single_pointwise("add"));
        mm->add_return({add});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto add = add_reduce(
            p2,
            "main:reduce_sum0:main:pointwise1:main:pointwise0:main:reduce_sum1",
            {x},
            {1},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto xx = add_pointwise(
                    p2, rm, "main:pointwise0", {inputs[0], inputs[0]}, single_pointwise("mul"));
                auto rsumx = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                 inputs[0]);
                auto rsumxx =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), xx);
                return add_pointwise(
                    p2, rm, "main:pointwise1", {rsumx, rsumxx}, single_pointwise("add"));
            });
        mm->add_return({add});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(parallel_reduce_reduce_broadcast)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm    = p1.get_main_module();
        auto x      = mm->add_parameter("x", s);
        auto sqrt   = add_pointwise(p1, "main:pointwise0", {x}, single_pointwise("sqrt"));
        auto rsum1  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), sqrt);
        auto rsum1b = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum1);
        auto relu  = add_pointwise(p1, "main:pointwise1", {x}, single_pointwise("relu"));
        auto rsum2 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), relu);
        auto add   = add_pointwise(p1, "main:pointwise2", {rsum1, rsum2}, single_pointwise("add"));
        auto addb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), add);
        auto clip =
            add_pointwise(p1, "main:pointwise3", {x, rsum1b, addb}, single_pointwise("clip"));
        mm->add_return({clip});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto clip = add_reduce(
            p2,
            "main:pointwise1:main:reduce_sum1:main:pointwise2:main:pointwise3:main:pointwise0:main:"
            "reduce_sum0",
            {x},
            {1},
            [&](auto* rm, const auto& inputs, const auto&) {
                auto sqrt =
                    add_pointwise(p2, rm, "main:pointwise0", {inputs[0]}, single_pointwise("sqrt"));
                auto rsum1 =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), sqrt);
                auto rsum1b = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum1);
                auto relu =
                    add_pointwise(p2, rm, "main:pointwise1", {inputs[0]}, single_pointwise("relu"));
                auto rsum2 =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), relu);
                auto add = add_pointwise(
                    p2, rm, "main:pointwise2", {rsum1, rsum2}, single_pointwise("add"));
                auto addb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), add);
                return add_pointwise(
                    p2, rm, "main:pointwise3", {inputs[0], rsum1b, addb}, single_pointwise("clip"));
            });
        mm->add_return({clip});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(parallel_reduce_reduce_broadcast_contiguous)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm    = p1.get_main_module();
        auto x      = mm->add_parameter("x", s);
        auto sqrt   = add_pointwise(p1, "main:pointwise0", {x}, single_pointwise("sqrt"));
        auto rsum1  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), sqrt);
        auto rsum1b = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum1);
        auto rsum1bc = mm->add_instruction(migraphx::make_op("contiguous"), rsum1b);
        auto relu    = add_pointwise(p1, "main:pointwise1", {x}, single_pointwise("relu"));
        auto rsum2   = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), relu);
        auto add = add_pointwise(p1, "main:pointwise2", {rsum1, rsum2}, single_pointwise("add"));
        auto addb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), add);
        auto clip =
            add_pointwise(p1, "main:pointwise3", {x, rsum1bc, addb}, single_pointwise("clip"));
        mm->add_return({clip});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto clip = add_reduce(
            p2,
            "main:pointwise1:main:reduce_sum1:main:pointwise2:main:pointwise3:main:pointwise0:main:"
            "reduce_sum0",
            {x},
            {1},
            [&](auto* rm, const auto& inputs, const auto&) {
                auto sqrt =
                    add_pointwise(p2, rm, "main:pointwise0", {inputs[0]}, single_pointwise("sqrt"));
                auto rsum1 =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), sqrt);
                auto rsum1b = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum1);
                auto relu =
                    add_pointwise(p2, rm, "main:pointwise1", {inputs[0]}, single_pointwise("relu"));
                auto rsum2 =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), relu);
                auto add = add_pointwise(
                    p2, rm, "main:pointwise2", {rsum1, rsum2}, single_pointwise("add"));
                auto addb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), add);
                return add_pointwise(
                    p2, rm, "main:pointwise3", {inputs[0], rsum1b, addb}, single_pointwise("clip"));
            });
        mm->add_return({clip});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(reduce_reduce_mismatch_axis)
{
    migraphx::shape s{migraphx::shape::float_type, {4, 2, 3}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto rsum1 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto rsum2 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), rsum1);
        mm->add_return({rsum2});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto rsum1 = add_reduce(p2, "main:reduce_sum0", {x}, {1}, single_reduce("reduce_sum"));
        auto rsum2 = add_reduce(p2, "main:reduce_sum1", {rsum1}, {2}, single_reduce("reduce_sum"));
        mm->add_return({rsum2});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(pointwise_reduce_broadcast)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto rsum1 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto sqrt  = add_pointwise(p1, "main:pointwise0", {rsum1}, single_pointwise("sqrt"));
        auto sqrtb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), sqrt);
        auto add1  = add_pointwise(p1, "main:pointwise1", {sqrtb, x}, single_pointwise("add"));
        auto rsum2 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), add1);
        auto add2  = add_pointwise(p1, "main:pointwise2", {rsum2, rsum1}, single_pointwise("add"));
        mm->add_return({add2});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto add2 = add_reduce(
            p2,
            "main:pointwise0:main:pointwise1:main:reduce_sum1:main:pointwise2:main:reduce_sum0",
            {x},
            {1},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto rsum1 = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                 inputs[0]);
                auto sqrt =
                    add_pointwise(p2, rm, "main:pointwise0", {rsum1}, single_pointwise("sqrt"));
                auto sqrtb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), sqrt);
                auto add1 = add_pointwise(
                    p2, rm, "main:pointwise1", {sqrtb, inputs[0]}, single_pointwise("add"));
                auto rsum2 =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), add1);
                return add_pointwise(
                    p2, rm, "main:pointwise2", {rsum2, rsum1}, single_pointwise("add"));
            });
        mm->add_return({add2});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(pointwise_reduce_broadcast_contiguous)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto rsum1 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto sqrt  = add_pointwise(p1, "main:pointwise0", {rsum1}, single_pointwise("sqrt"));
        auto sqrtb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), sqrt);
        auto sqrtbc = mm->add_instruction(migraphx::make_op("contiguous"), sqrtb);
        auto add1   = add_pointwise(p1, "main:pointwise1", {sqrtbc, x}, single_pointwise("add"));
        auto rsum2  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), add1);
        auto add2   = add_pointwise(p1, "main:pointwise2", {rsum2, rsum1}, single_pointwise("add"));
        mm->add_return({add2});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto add2 = add_reduce(
            p2,
            "main:pointwise0:main:pointwise1:main:reduce_sum1:main:pointwise2:main:reduce_sum0",
            {x},
            {1},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto rsum1 = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                 inputs[0]);
                auto sqrt =
                    add_pointwise(p2, rm, "main:pointwise0", {rsum1}, single_pointwise("sqrt"));
                auto sqrtb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), sqrt);
                auto add1 = add_pointwise(
                    p2, rm, "main:pointwise1", {sqrtb, inputs[0]}, single_pointwise("add"));
                auto rsum2 =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), add1);
                return add_pointwise(
                    p2, rm, "main:pointwise2", {rsum2, rsum1}, single_pointwise("add"));
            });
        mm->add_return({add2});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(reduce_reduce_broadcast)
{
    migraphx::shape s{migraphx::shape::float_type, {4, 2, 3}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto rsum1 = add_reduce(p1, "test:reduce_sum0", {x}, {1}, single_reduce("reduce_sum"));
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum1);
        auto add = add_reduce(
            p1,
            "test:reduce_sum1",
            {rsumb, x},
            {1},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto add2 =
                    add_pointwise(p1, rm, "test:pointwise0", inputs, single_pointwise("add"));
                return rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), add2);
            });
        mm->add_return({add});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto rsum = add_reduce(
            p2,
            "test:reduce_sum1:test:reduce_sum0",
            {x},
            {1},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto rsum1 = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                 inputs[0]);
                auto rsumb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum1);
                auto add = add_pointwise(
                    p2, rm, "test:pointwise0", {rsumb, inputs[0]}, single_pointwise("add"));
                return rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), add);
            });
        mm->add_return({rsum});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(reduce_reduce_broadcast_contiguous)
{
    migraphx::shape s{migraphx::shape::float_type, {4, 2, 3}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto rsum1 = add_reduce(p1, "test:reduce_sum0", {x}, {1}, single_reduce("reduce_sum"));
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum1);
        auto rsumbc = mm->add_instruction(migraphx::make_op("contiguous"), rsumb);
        auto add    = add_reduce(
            p1,
            "test:reduce_sum1",
            {rsumbc, x},
            {1},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto add2 =
                    add_pointwise(p1, rm, "test:pointwise0", inputs, single_pointwise("add"));
                return rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), add2);
            });
        mm->add_return({add});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto rsum = add_reduce(
            p2,
            "test:reduce_sum1:test:reduce_sum0",
            {x},
            {1},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto rsum1 = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                 inputs[0]);
                auto rsumb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum1);
                auto add = add_pointwise(
                    p2, rm, "test:pointwise0", {rsumb, inputs[0]}, single_pointwise("add"));
                return rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), add);
            });
        mm->add_return({rsum});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(reduce_reshape_pointwise1)
{
    migraphx::shape s1{migraphx::shape::float_type, {64, 4}};
    migraphx::shape s2{migraphx::shape::float_type, {8, 8, 2, 2}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s1);
        auto y     = mm->add_parameter("y", s2);
        auto rsum  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rsum);
        auto rsumr =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", s2.lens()}}), rsumb);
        auto add = add_pointwise(p1, "main:pointwise0", {rsumr, y}, single_pointwise("add"));
        mm->add_return({add});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s1);
        auto y   = mm->add_parameter("y", s2);
        auto xr  = mm->add_instruction(migraphx::make_op("reshape", {{"dims", s2.lens()}}), x);
        auto add = add_reduce(
            p2,
            "main:reduce_sum0_reshape:main:pointwise0",
            {xr, y},
            {2, 3},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto rsum  = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                inputs[0]);
                auto rsumb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s2.lens()}}), rsum);
                return add_pointwise(
                    p2, rm, "main:pointwise0", {rsumb, inputs[1]}, single_pointwise("add"));
            });
        mm->add_return({add});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(reduce_reshape_pointwise2)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 32, 40960}};
    migraphx::shape s2{migraphx::shape::float_type, {2, 320, 64, 64}};
    migraphx::shape s3{migraphx::shape::float_type, {2, 32, 10, 64, 64}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s1);
        auto y     = mm->add_parameter("y", s2);
        auto rsum  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), x);
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rsum);
        auto rsumr =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", s2.lens()}}), rsumb);
        auto add = add_pointwise(p1, "main:pointwise0", {rsumr, y}, single_pointwise("add"));
        mm->add_return({add});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s1);
        auto y   = mm->add_parameter("y", s2);
        auto xr  = mm->add_instruction(migraphx::make_op("reshape", {{"dims", s3.lens()}}), x);
        auto yr  = mm->add_instruction(migraphx::make_op("reshape", {{"dims", s3.lens()}}), y);
        auto add = add_reduce(
            p2,
            "main:reduce_sum0_reshape:main:pointwise0",
            {xr, yr},
            {2, 3, 4},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto rsum  = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                inputs[0]);
                auto rsumb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s3.lens()}}), rsum);
                return add_pointwise(
                    p2, rm, "main:pointwise0", {rsumb, inputs[1]}, single_pointwise("add"));
            });
        auto addr = mm->add_instruction(migraphx::make_op("reshape", {{"dims", s2.lens()}}), add);
        mm->add_return({addr});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(reduce_contiguous_reshape_pointwise)
{
    migraphx::shape s1 =
        migraphx::shape::from_permutation(migraphx::shape::float_type, {2, 32, 40960}, {1, 0, 2});
    auto s2 = migraphx::shape{migraphx::shape::float_type, {2, 320, 64, 64}};
    auto s3 = migraphx::shape{migraphx::shape::float_type, {2, 32, 10, 64, 64}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s1);
        auto y     = mm->add_parameter("y", s2);
        auto rsum  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), x);
        auto rsumc = mm->add_instruction(migraphx::make_op("contiguous"), rsum);
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rsumc);
        auto rsumr =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", s2.lens()}}), rsumb);
        auto add = add_pointwise(p1, "main:pointwise0", {rsumr, y}, single_pointwise("add"));
        mm->add_return({add});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s1);
        auto y   = mm->add_parameter("y", s2);
        auto xr  = mm->add_instruction(migraphx::make_op("reshape", {{"dims", s3.lens()}}), x);
        auto yr  = mm->add_instruction(migraphx::make_op("reshape", {{"dims", s3.lens()}}), y);
        auto add = add_reduce(
            p2,
            "main:reduce_sum0_reshape:main:pointwise0",
            {xr, yr},
            {2, 3, 4},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto rsum  = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                inputs[0]);
                auto rsumb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s3.lens()}}), rsum);
                return add_pointwise(
                    p2, rm, "main:pointwise0", {rsumb, inputs[1]}, single_pointwise("add"));
            });
        auto addr = mm->add_instruction(migraphx::make_op("reshape", {{"dims", s2.lens()}}), add);
        mm->add_return({addr});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(reduce_reshape_reduce)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 32, 4096}};
    migraphx::shape s1r{migraphx::shape::float_type, {2, 32, 1}};
    migraphx::shape s2{migraphx::shape::float_type, {4, 16, 64, 64}};
    migraphx::shape s2r{migraphx::shape::float_type, {4, 16, 1, 1}};
    migraphx::shape s3{migraphx::shape::float_type, {2, 2, 16, 64, 64}};
    migraphx::shape s3r{migraphx::shape::float_type, {2, 2, 16, 1, 1}};
    migraphx::program p1;
    {
        auto* mm       = p1.get_main_module();
        auto x1        = mm->add_parameter("x1", s1);
        auto x2        = mm->add_parameter("x2", s1r);
        auto y         = mm->add_parameter("y", s2);
        auto rsum1     = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), x1);
        auto rsum1_add = add_pointwise(p1, "main:pointwise0", {rsum1, x2}, single_pointwise("add"));

        auto rsum1_addb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rsum1_add);
        auto rsum1_sub =
            add_pointwise(p1, "main:pointwise1", {rsum1_addb, x1}, single_pointwise("sub"));
        auto rsum2 =
            mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), rsum1_sub);
        auto rsum2b = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rsum2);
        auto rsum2_sub =
            add_pointwise(p1, "main:pointwise2", {rsum2b, x1}, single_pointwise("sub"));
        auto rsum2_subr =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", s2.lens()}}), rsum2_sub);
        auto rsum3 =
            mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2, 3}}}), rsum2_subr);
        auto rsum3b = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s2.lens()}}), rsum3);
        auto rsum3_add = add_pointwise(p1, "main:pointwise3", {rsum3b, y}, single_pointwise("add"));
        mm->add_return({rsum3_add});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x1  = mm->add_parameter("x1", s1);
        auto x2  = mm->add_parameter("x2", s1r);
        auto y   = mm->add_parameter("y", s2);
        auto x1r = mm->add_instruction(migraphx::make_op("reshape", {{"dims", s3.lens()}}), x1);
        auto x2r = mm->add_instruction(migraphx::make_op("reshape", {{"dims", s3r.lens()}}), x2);
        auto yr      = mm->add_instruction(migraphx::make_op("reshape", {{"dims", s3.lens()}}), y);
        auto freduce = add_reduce(
            p2,
            "main:pointwise2:main:reduce_sum2_reshape_reshape:main:pointwise3_reshape:main:reduce_"
            "sum1:main:reduce_sum0:main:pointwise0:main:pointwise1_reshape",
            {x1r, x2r, yr},
            {3, 4},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto rsum1 = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                 inputs[0]);
                auto add   = add_pointwise(
                    p2, rm, "main:pointwise0", {rsum1, inputs[1]}, single_pointwise("add"));
                auto addb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s3.lens()}}), add);
                auto sub1 = add_pointwise(
                    p2, rm, "main:pointwise1", {addb, inputs[0]}, single_pointwise("sub"));
                auto rsum2 =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), sub1);
                auto rsum2b = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s3.lens()}}), rsum2);
                auto sub2 = add_pointwise(
                    p2, rm, "main:pointwise2", {rsum2b, inputs[0]}, single_pointwise("sub"));
                auto rsum3 =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), sub2);
                auto rsum3b = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s3.lens()}}), rsum3);
                return add_pointwise(
                    p2, rm, "main:pointwise3", {rsum3b, inputs[2]}, single_pointwise("add"));
            });
        auto freducer =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", s2.lens()}}), freduce);
        mm->add_return({freducer});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(reshape_reduce_reduce_reduce_diff_axes)
{
    migraphx::shape s1{migraphx::shape::float_type, {128, 196, 256}};
    migraphx::shape s2{migraphx::shape::float_type, {1}};
    migraphx::shape s3{migraphx::shape::float_type, {25088, 256}};

    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x1  = mm->add_parameter("x1", s1);
        auto x2  = mm->add_parameter("x2", s1);
        auto l1  = mm->add_literal(migraphx::literal(s2, {1.0}));
        auto l2  = mm->add_literal(migraphx::literal(s2, {2.0}));

        auto x1_rsp = mm->add_instruction(migraphx::make_op("reshape", {{"dims", s3.lens()}}), x1);
        auto l2_mb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s3.lens()}}), l2);
        auto l2_ct = mm->add_instruction(migraphx::make_op("contiguous"), l2_mb);
        auto pw0   = add_pointwise(p1, "main:pointwise0", {l2_ct, x1_rsp}, single_pointwise("add"));
        auto pw0_rsp =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", s1.lens()}}), pw0);
        auto pw1   = add_pointwise(p1, "main:pointwise1", {x2, pw0_rsp}, single_pointwise("mul"));
        auto pw2   = add_pointwise(p1, "main:pointwise2", {pw1}, single_pointwise("log"));
        auto rsum0 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), pw2);
        auto rsum0_mb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rsum0);
        auto rsum0_ct = mm->add_instruction(migraphx::make_op("contiguous"), rsum0_mb);
        auto pw3 =
            add_pointwise(p1, "main:pointwise3", {pw0_rsp, rsum0_ct}, single_pointwise("add"));
        auto pw4    = add_pointwise(p1, "main:pointwise4", {pw3}, single_pointwise("log"));
        auto rsum1  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), pw4);
        auto pw5    = add_pointwise(p1, "main:pointwise5", {rsum1}, single_pointwise("log"));
        auto pw5_mb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), pw5);
        auto pw5_ct = mm->add_instruction(migraphx::make_op("contiguous"), pw5_mb);

        auto l1_mb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), l1);
        auto pw6 = add_pointwise(p1, "main:pointwise6", {pw5_ct, l1_mb}, single_pointwise("mul"));

        auto rsum2 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), pw6);
        mm->add_return({rsum2});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x1  = mm->add_parameter("x1", s1);
        auto x2  = mm->add_parameter("x2", s1);
        auto l1  = mm->add_literal(migraphx::literal(s2, {1.0}));
        auto l2  = mm->add_literal(migraphx::literal(s2, {2.0}));

        auto l1_mb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), l1);
        auto l2_mb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), l2);

        auto reduce0 = add_reduce(
            p2,
            "main:pointwise0:main:pointwise1:main:reduce_sum1:main:pointwise2:main:reduce_sum0:"
            "main:pointwise3:main:pointwise4:main:pointwise5:main:pointwise6_reshape",
            {l2_mb, x1, x2, l1_mb},
            {2},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto pw0 = add_pointwise(
                    p2, rm, "main:pointwise0", {inputs[0], inputs[1]}, single_pointwise("add"));
                auto pw1 = add_pointwise(
                    p2, rm, "main:pointwise1", {inputs[2], pw0}, single_pointwise("mul"));
                auto pw2 = add_pointwise(p2, rm, "main:pointwise2", {pw1}, single_pointwise("log"));
                auto rsum0 =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), pw2);
                auto rsum0_mb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rsum0);
                auto pw3 = add_pointwise(
                    p2, rm, "main:pointwise3", {pw0, rsum0_mb}, single_pointwise("add"));
                auto pw4 = add_pointwise(p2, rm, "main:pointwise4", {pw3}, single_pointwise("log"));
                auto rsum1 =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), pw4);
                auto pw5 =
                    add_pointwise(p2, rm, "main:pointwise5", {rsum1}, single_pointwise("log"));
                auto pw5_mb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), pw5);
                return add_pointwise(
                    p2, rm, "main:pointwise6", {pw5_mb, inputs[3]}, single_pointwise("mul"));
            });

        auto reduce1 =
            add_reduce(p2,
                       "main:reduce_sum2",
                       {reduce0},
                       {1},
                       [&](auto* rm, const auto& inputs, const auto& axes) {
                           return rm->add_instruction(
                               migraphx::make_op("reduce_sum", {{"axes", axes}}), inputs[0]);
                       });

        mm->add_return({reduce1});
    }

    EXPECT(p1.sort() == p2.sort());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
