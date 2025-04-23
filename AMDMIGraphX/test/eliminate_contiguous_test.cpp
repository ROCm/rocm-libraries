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
#include <migraphx/eliminate_contiguous.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/instruction.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>

#include <pointwise.hpp>
#include <test.hpp>

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(
        m, {migraphx::eliminate_contiguous{"contiguous"}, migraphx::dead_code_elimination{}});
}

TEST_CASE(standard_op)
{
    migraphx::module m;

    auto l = m.add_parameter("x", {migraphx::shape::float_type, {2, 2}});
    auto t = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l);
    auto c = m.add_instruction(migraphx::make_op("contiguous"), t);
    m.add_instruction(pass_standard_op{}, c);
    auto count = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(std::distance(m.begin(), m.end()) == count);
}

TEST_CASE(standard_op_const)
{
    migraphx::module m;

    auto l = m.add_literal(get_2x2());
    auto t = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l);
    auto c = m.add_instruction(migraphx::make_op("contiguous"), t);
    m.add_instruction(pass_standard_op{}, c);
    run_pass(m);
    EXPECT(std::distance(m.begin(), m.end()) == 2);
}

TEST_CASE(non_standard_op)
{
    migraphx::module m;

    auto l = m.add_parameter("x", {migraphx::shape::float_type, {2, 2}});
    auto t = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l);
    auto c = m.add_instruction(migraphx::make_op("contiguous"), t);
    m.add_instruction(pass_op{}, c);
    auto count = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(std::distance(m.begin(), m.end()) == count);
}

TEST_CASE(non_standard_op_const)
{
    migraphx::module m;

    auto l = m.add_literal(get_2x2());
    auto t = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l);
    auto c = m.add_instruction(migraphx::make_op("contiguous"), t);
    m.add_instruction(pass_op{}, c);
    run_pass(m);
    EXPECT(std::distance(m.begin(), m.end()) == 2);
}

TEST_CASE(transpose_gem)
{
    migraphx::module m;

    auto l  = m.add_literal(get_2x2());
    auto t  = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l);
    auto c  = m.add_instruction(migraphx::make_op("contiguous"), t);
    auto ic = m.add_instruction(migraphx::make_op("identity"), c);
    m.add_instruction(migraphx::make_op("dot"), ic, l);
    auto count = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(std::distance(m.begin(), m.end()) == (count - 1));
}

TEST_CASE(transpose_standard_op)
{
    migraphx::module m;

    auto l  = m.add_parameter("x", {migraphx::shape::float_type, {2, 2}});
    auto t  = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l);
    auto c  = m.add_instruction(migraphx::make_op("contiguous"), t);
    auto sn = m.add_instruction(migraphx::make_op("sin"), c);
    m.add_instruction(pass_standard_op{}, sn);
    auto count = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(std::distance(m.begin(), m.end()) == count);
}

TEST_CASE(transpose_standard_op_const)
{
    migraphx::module m;

    auto l  = m.add_literal(get_2x2());
    auto t  = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l);
    auto c  = m.add_instruction(migraphx::make_op("contiguous"), t);
    auto sn = m.add_instruction(migraphx::make_op("sin"), c);
    m.add_instruction(pass_standard_op{}, sn);
    run_pass(m);
    EXPECT(std::distance(m.begin(), m.end()) == 3);
}

TEST_CASE(no_packed_unary_op)
{
    migraphx::module m;

    auto l = m.add_literal(get_2x2());
    auto t = m.add_instruction(
        migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), l);
    auto c  = m.add_instruction(migraphx::make_op("contiguous"), t);
    auto sn = m.add_instruction(migraphx::make_op("sin"), c);
    m.add_instruction(pass_standard_op{}, sn);
    auto count = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(std::distance(m.begin(), m.end()) == count - 1);
}

TEST_CASE(non_standard_return_input)
{
    migraphx::module m;

    auto l  = m.add_literal(get_2x2());
    auto tl = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l);
    auto c  = m.add_instruction(migraphx::make_op("contiguous"), tl);
    m.add_return({c});
    auto count = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(std::distance(m.begin(), m.end()) == count);
}

TEST_CASE(non_standard_flatten_op)
{
    migraphx::module m;

    auto l = m.add_parameter("x", {migraphx::shape::float_type, {2, 6, 6, 6}});
    auto t = m.add_instruction(
        migraphx::make_op("slice", {{"axes", {2, 3}}, {"starts", {1, 1}}, {"ends", {6, 6}}}), l);
    auto c = m.add_instruction(migraphx::make_op("contiguous"), t);
    m.add_instruction(migraphx::make_op("flatten"), c);
    auto count = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(std::distance(m.begin(), m.end()) == (count - 1));
}

TEST_CASE(standard_flatten_op)
{
    migraphx::module m;

    auto l = m.add_parameter("x", {migraphx::shape::float_type, {2, 6, 6, 6}});
    auto t = m.add_instruction(
        migraphx::make_op("slice", {{"axes", {0, 1}}, {"starts", {1, 1}}, {"ends", {6, 6}}}), l);
    auto c = m.add_instruction(migraphx::make_op("contiguous"), t);
    m.add_instruction(migraphx::make_op("flatten"), c);
    auto count = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(std::distance(m.begin(), m.end()) == (count - 1));
}

TEST_CASE(transpose_contiguous_reshape_add)
{

    migraphx::shape s1{migraphx::shape::float_type, {2, 10}};
    migraphx::shape s2{migraphx::shape::float_type, {5, 4}};
    migraphx::program p1;
    auto* m1 = p1.get_main_module();
    {
        auto x  = m1->add_parameter("x", s1);
        auto y  = m1->add_parameter("y", s2);
        auto xt = m1->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), x);
        auto xc = m1->add_instruction(migraphx::make_op("contiguous"), xt);
        auto br = m1->add_instruction(migraphx::make_op("reshape_lazy", {{"dims", {10, 2}}}), y);
        auto bc = m1->add_instruction(migraphx::make_op("contiguous"), br);
        m1->add_instruction(sum_std_op{}, xc, bc);
    }
    auto count = std::distance(m1->begin(), m1->end());
    run_pass(*m1);
    EXPECT(std::distance(m1->begin(), m1->end()) == (count - 1));
    migraphx::program p2;
    auto* m2 = p2.get_main_module();
    {
        auto x  = m2->add_parameter("x", s1);
        auto y  = m2->add_parameter("y", s2);
        auto xt = m2->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), x);
        auto xc = m2->add_instruction(migraphx::make_op("contiguous"), xt);
        auto br = m2->add_instruction(migraphx::make_op("reshape_lazy", {{"dims", {10, 2}}}), y);
        m2->add_instruction(sum_std_op{}, xc, br);
    }
    EXPECT(p1 == p2);
}

TEST_CASE(contiguous_pointwise)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 8, 8}};
    migraphx::program p;
    auto* mm = p.get_main_module();
    {
        auto x  = mm->add_parameter("x", s);
        auto y  = mm->add_parameter("y", migraphx::shape{migraphx::shape::float_type, {3}});
        auto yb = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {2, 3, 8, 8}}}), y);
        auto yc  = mm->add_instruction(migraphx::make_op("contiguous"), yb);
        auto add = add_pointwise(p, "main:pointwise0", {x, yc}, single_pointwise("add"));
        auto cadd = mm->add_instruction(migraphx::make_op("contiguous"), add);
        mm->add_instruction(pass_op{}, cadd);
    }
    auto count = std::distance(mm->begin(), mm->end());
    run_pass(*mm);
    EXPECT(std::distance(mm->begin(), mm->end()) == (count - 2));
    EXPECT(std::none_of(
        mm->begin(), mm->end(), [](auto&& ins) { return ins.name() == "contiguous"; }));
}

TEST_CASE(contiguous_nhwc_pointwise)
{
    auto s =
        migraphx::shape::from_permutation(migraphx::shape::float_type, {2, 3, 8, 8}, {0, 2, 3, 1});
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", migraphx::shape{migraphx::shape::float_type, {3}});
        auto yb  = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {2, 3, 8, 8}}}), y);
        auto yc   = mm->add_instruction(migraphx::make_op("contiguous"), yb);
        auto add  = add_pointwise(p1, "main:pointwise0", {x, yc}, single_pointwise("add"));
        auto cadd = mm->add_instruction(migraphx::make_op("contiguous"), add);
        mm->add_instruction(pass_op{}, cadd);
    }
    run_pass(*p1.get_main_module());
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", migraphx::shape{migraphx::shape::float_type, {3}});
        auto yb  = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {2, 3, 8, 8}}}), y);
        auto add  = add_pointwise(p2, "main:pointwise0", {x, yb}, single_pointwise("add"));
        auto cadd = mm->add_instruction(migraphx::make_op("contiguous"), add);
        mm->add_instruction(pass_op{}, cadd);
    }
    EXPECT(p1 == p2);
}

TEST_CASE(slice_contiguous)
{
    migraphx::module m;

    migraphx::shape s{migraphx::shape::float_type, {4, 2}};
    auto x  = m.add_parameter("x", s);
    auto t  = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), x);
    auto c  = m.add_instruction(migraphx::make_op("contiguous"), t);
    auto s1 = m.add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), c);
    auto s2 = m.add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), c);
    auto c1 = m.add_instruction(migraphx::make_op("contiguous"), s1);
    auto c2 = m.add_instruction(migraphx::make_op("contiguous"), s2);
    m.add_instruction(pass_standard_op{}, c1, c2);
    run_pass(m);
    EXPECT(std::count_if(
               m.begin(), m.end(), [](auto&& ins) { return ins.name() == "contiguous"; }) == 1);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
