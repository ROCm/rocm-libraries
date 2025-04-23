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
#include <migraphx/simplify_reshapes.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/generate.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>

#include <migraphx/serialize.hpp>

#include <test.hpp>

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::simplify_reshapes{}, migraphx::dead_code_elimination{}});
}

inline static std::vector<std::vector<std::size_t>>
to_lens(const std::vector<migraphx::shape>& shapes)
{
    std::vector<std::vector<std::size_t>> result;
    std::transform(shapes.begin(), shapes.end(), std::back_inserter(result), [&](const auto& s) {
        return s.lens();
    });
    return result;
}

static migraphx::module make_concat_multibroadcast(const std::vector<size_t>& in_lens,
                                                   const std::vector<size_t>& mbcast_lens,
                                                   const int axis)
{
    migraphx::module m;
    auto s = migraphx::shape{migraphx::shape::float_type, in_lens};
    auto x = m.add_parameter("x", s);
    auto y = m.add_parameter("y", s);
    auto z = m.add_parameter("z", s);
    auto xm =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", mbcast_lens}}), x);
    auto ym =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", mbcast_lens}}), y);
    auto zm =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", mbcast_lens}}), z);
    auto concat = m.add_instruction(migraphx::make_op("concat", {{"axis", axis}}), xm, ym, zm);
    m.add_return({concat});
    return m;
}

TEST_CASE(broadcast_transpose)
{
    migraphx::module m1;
    {
        auto l = m1.add_parameter("x", {migraphx::shape::float_type, {5}});
        auto mb =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 3, 5}}}), l);
        auto t1 =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {2, 0, 1}}}), mb);
        m1.add_return({t1});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto l  = m2.add_parameter("x", {migraphx::shape::float_type, {5}});
        auto b  = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 0}, {"out_lens", {5, 2, 3}}}), l);
        m2.add_return({b});
    }

    EXPECT(m1 == m2);
}

TEST_CASE(broadcast_transpose_opt)
{
    // extra transpose from transformation will be optimized out
    migraphx::module m1;
    {
        auto l = m1.add_parameter("x", {migraphx::shape::float_type, {5}});
        auto mb =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 3, 5}}}), l);
        auto t1 =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0, 2}}}), mb);
        m1.add_return({t1});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto l  = m2.add_parameter("x", {migraphx::shape::float_type, {5}});
        auto b  = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 2}, {"out_lens", {3, 2, 5}}}), l);
        m2.add_return({b});
    }

    EXPECT(m1 == m2);
}

TEST_CASE(broadcast_transpose_scalar)
{
    migraphx::module m1;
    {
        auto l = m1.add_parameter("x", {migraphx::shape::float_type, {1}, {0}});
        auto mb =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 3}}}), l);
        auto t1 = m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), mb);
        m1.add_return({t1});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto l = m2.add_parameter("x", {migraphx::shape::float_type, {1}, {0}});
        auto mb =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3, 2}}}), l);
        m2.add_return({mb});
    }

    EXPECT(m1 == m2);
}

TEST_CASE(broadcast_transpose_scalar_multi_use)
{
    // multibroadcast used more than once
    migraphx::module m1;
    {
        auto l = m1.add_parameter("x", {migraphx::shape::float_type, {1}, {0}});
        auto mb =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 3}}}), l);
        auto t1 = m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), mb);
        auto id = m1.add_instruction(migraphx::make_op("identity"), mb);
        m1.add_return({t1, id});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto l = m2.add_parameter("x", {migraphx::shape::float_type, {1}, {0}});
        auto mb =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3, 2}}}), l);
        auto mb2 =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 3}}}), l);
        auto id = m2.add_instruction(migraphx::make_op("identity"), mb2);
        m2.add_return({mb, id});
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(double_contig)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto l  = mm->add_literal(get_2x2());
    auto t1 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l);
    auto c1 = mm->add_instruction(migraphx::make_op("contiguous"), t1);
    auto c2 = mm->add_instruction(migraphx::make_op("contiguous"), c1);
    mm->add_return({c2});
    EXPECT(mm->get_output_shapes().back().standard());
    EXPECT(not mm->get_output_shapes().back().transposed());
    run_pass(*mm);
    EXPECT(mm->get_output_shapes().back().standard());
    EXPECT(not mm->get_output_shapes().back().transposed());
    EXPECT(std::distance(mm->begin(), mm->end()) == 4);
    auto result = p.eval({}).back();
    EXPECT(result != get_2x2());
}

TEST_CASE(double_transpose)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto l  = mm->add_literal(get_2x2());
    auto t1 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l);
    auto t2 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), t1);
    mm->add_return({t2});
    EXPECT(mm->get_output_shapes().back().standard());
    EXPECT(not mm->get_output_shapes().back().transposed());
    run_pass(*mm);
    EXPECT(mm->get_output_shapes().back().standard());
    EXPECT(not mm->get_output_shapes().back().transposed());
    EXPECT(std::distance(mm->begin(), mm->end()) == 2);
    auto result = p.eval({}).back();
    EXPECT(result == get_2x2());
}

TEST_CASE(double_transpose_contig)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto l  = mm->add_literal(get_2x2());
    auto t1 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l);
    auto c1 = mm->add_instruction(migraphx::make_op("contiguous"), t1);
    auto t2 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), c1);
    auto c2 = mm->add_instruction(migraphx::make_op("contiguous"), t2);
    mm->add_return({c2});
    EXPECT(mm->get_output_shapes().back().standard());
    EXPECT(not mm->get_output_shapes().back().transposed());
    run_pass(*mm);
    EXPECT(mm->get_output_shapes().back().standard());
    EXPECT(not mm->get_output_shapes().back().transposed());
    EXPECT(std::distance(mm->begin(), mm->end()) == 2);
    auto result = p.eval({}).back();
    EXPECT(result == get_2x2());
}

TEST_CASE(single_transpose)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto l  = mm->add_literal(get_2x2());
    auto t1 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l);
    mm->add_return({t1});
    EXPECT(not mm->get_output_shapes().back().standard());
    EXPECT(mm->get_output_shapes().back().transposed());
    run_pass(*mm);
    EXPECT(not mm->get_output_shapes().back().standard());
    EXPECT(mm->get_output_shapes().back().transposed());
    EXPECT(std::distance(mm->begin(), mm->end()) == 3);
    auto result = p.eval({}).back();
    EXPECT(result != get_2x2());
}

TEST_CASE(double_transpose_sin_pass)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto l  = mm->add_literal(get_2x2());
    auto t1 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l);
    mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), t1);
    EXPECT(mm->get_output_shapes().back().standard());
    EXPECT(not mm->get_output_shapes().back().transposed());
    run_pass(*mm);
    EXPECT(mm->get_output_shapes().back().standard());
    EXPECT(not mm->get_output_shapes().back().transposed());
    // TODO: Fix this
    // EXPECT(std::distance(mm->begin(), mm->end()) == 1);
    auto result = p.eval({}).back();
    EXPECT(result == get_2x2());
}

TEST_CASE(single_transpose_sin_pass)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto l = mm->add_literal(get_2x2());
    mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l);
    EXPECT(not mm->get_output_shapes().back().standard());
    EXPECT(mm->get_output_shapes().back().transposed());
    run_pass(*mm);
    EXPECT(not mm->get_output_shapes().back().standard());
    EXPECT(mm->get_output_shapes().back().transposed());
    EXPECT(std::distance(mm->begin(), mm->end()) == 2);
    auto result = p.eval({}).back();
    EXPECT(result != get_2x2());
}

TEST_CASE(reshape_transpose)
{
    migraphx::module m;

    auto s  = migraphx::shape{migraphx::shape::float_type, {1, 112, 56, 56}};
    auto x  = m.add_parameter("x", s);
    auto r1 = m.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 4, 28, 56, 56}}}), x);
    auto t =
        m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3, 4}}}), r1);
    auto r2 = m.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 112, 56, 56}}}), t);
    m.add_return({r2});
    EXPECT(m.get_output_shapes().back() == s);
    auto n = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back() == s);
    EXPECT(std::distance(m.begin(), m.end()) == n);
}

TEST_CASE(transpose_contiguous)
{
    migraphx::module m;

    auto s  = migraphx::shape{migraphx::shape::float_type, {4, 4}};
    auto x  = m.add_parameter("x", s);
    auto t  = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), x);
    auto c1 = m.add_instruction(migraphx::make_op("contiguous"), t);
    m.add_return({c1});
    auto out_shape = m.get_output_shapes().back();
    auto n         = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back() == out_shape);
    EXPECT(std::distance(m.begin(), m.end()) == n);
}

TEST_CASE(transpose_double_contiguous)
{
    migraphx::module m;

    auto s  = migraphx::shape{migraphx::shape::float_type, {4, 4}};
    auto x  = m.add_parameter("x", s);
    auto t  = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), x);
    auto c1 = m.add_instruction(migraphx::make_op("contiguous"), t);
    auto c2 = m.add_instruction(migraphx::make_op("contiguous"), c1);
    m.add_return({c2});
    auto out_shape = m.get_output_shapes().back();
    auto n         = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back() == out_shape);
    EXPECT(std::distance(m.begin(), m.end()) == n - 1);
    EXPECT(m.has_instruction(t));
}

TEST_CASE(transpose_partial1)
{
    migraphx::module m;

    auto s  = migraphx::shape{migraphx::shape::float_type, {1, 2, 3}};
    auto x  = m.add_parameter("x", s);
    auto t1 = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0, 2}}}), x);
    auto t2 = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 2, 0}}}), t1);
    m.add_return({t2});
    auto out_shape = m.get_output_shapes().back();
    auto n         = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back() == out_shape);
    EXPECT(std::distance(m.begin(), m.end()) == n - 1);
}

TEST_CASE(transpose_partial2)
{
    migraphx::module m;

    auto s  = migraphx::shape{migraphx::shape::float_type, {1, 2, 3}};
    auto x  = m.add_parameter("x", s);
    auto t1 = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0, 2}}}), x);
    auto t2 = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 2, 0}}}), t1);
    auto t3 = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0, 2}}}), t2);
    m.add_return({t3});
    auto out_shape = m.get_output_shapes().back();
    auto n         = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back() == out_shape);
    EXPECT(std::distance(m.begin(), m.end()) == n - 2);
}

TEST_CASE(transpose_partial3)
{
    migraphx::module m;

    auto s  = migraphx::shape{migraphx::shape::float_type, {1, 2, 3}};
    auto x  = m.add_parameter("x", s);
    auto t1 = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0, 2}}}), x);
    auto t2 = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 2, 0}}}), t1);
    auto t3 = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0, 2}}}), t2);
    auto t4 = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0, 2}}}), t3);
    m.add_return({t4});
    auto out_shape = m.get_output_shapes().back();
    auto n         = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back() == out_shape);
    EXPECT(std::distance(m.begin(), m.end()) == n - 3);
}

TEST_CASE(nop_transpose1)
{
    migraphx::module m;

    auto s = migraphx::shape{migraphx::shape::float_type, {1, 2, 3}};
    auto x = m.add_parameter("x", s);
    auto t = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2}}}), x);
    m.add_return({t});
    auto out_shape = m.get_output_shapes().back();
    auto n         = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back() == out_shape);
    EXPECT(std::distance(m.begin(), m.end()) == n - 1);
}

TEST_CASE(nop_transpose2)
{
    migraphx::module m;

    auto s  = migraphx::shape{migraphx::shape::float_type, {1, 2, 3}};
    auto x  = m.add_parameter("x", s);
    auto t1 = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2}}}), x);
    auto t2 = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2}}}), t1);
    auto t3 = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2}}}), t2);
    auto t4 = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2}}}), t3);
    m.add_instruction(pass_op{}, t4);
    auto out_shape = m.get_output_shapes().back();
    auto n         = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back() == out_shape);
    EXPECT(std::distance(m.begin(), m.end()) == n - 4);
}

TEST_CASE(nop_transpose3)
{
    migraphx::module m;

    auto s      = migraphx::shape{migraphx::shape::float_type, {1, 2, 3, 4}};
    auto x      = m.add_parameter("x", s);
    auto y      = m.add_parameter("y", s);
    auto concat = m.add_instruction(migraphx::make_op("concat", {{"axis", 3}}), x, y);
    auto t1 =
        m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2, 3}}}), concat);
    auto t2 =
        m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), t1);
    m.add_return({t2});
    auto out_shape = m.get_output_shapes().back();
    auto n         = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back() == out_shape);
    EXPECT(std::distance(m.begin(), m.end()) == n - 1);
}

TEST_CASE(nop_convert)
{
    migraphx::module m;

    auto s = migraphx::shape{migraphx::shape::float_type, {1, 2, 3}};
    auto x = m.add_parameter("x", s);
    auto t = m.add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
        x);
    m.add_return({t});
    auto out_shape = m.get_output_shapes().back();
    auto n         = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back() == out_shape);
    EXPECT(std::distance(m.begin(), m.end()) == n - 1);
}

TEST_CASE(nested_reshape)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {1, 2, 3, 4, 5, 6, 7}};
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", s);
        auto rshp1 =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 3, 4, 5, 42}}}), x);
        auto rshp2 =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 12, 5, 42}}}), rshp1);
        auto rshp3 =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 12, 5, 42}}}), rshp2);
        auto rshp4 =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 60, 42}}}), rshp3);
        auto rshp5 = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {120, 42}}}), rshp4);
        auto rshp6 = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {5040}}}), rshp5);
        m1.add_return({rshp6});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", s);
        auto rshp = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {5040}}}), x);
        m2.add_return({rshp});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(nested_reshape_contiguous)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {1, 2, 3, 4, 5, 6, 7}};
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", s);
        auto rshp1 =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 3, 4, 5, 42}}}), x);
        auto c1 = m1.add_instruction(migraphx::make_op("contiguous"), rshp1);
        auto rshp2 =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 12, 5, 42}}}), c1);
        auto c2 = m1.add_instruction(migraphx::make_op("contiguous"), rshp2);
        auto rshp3 =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 12, 5, 42}}}), c2);
        auto c3    = m1.add_instruction(migraphx::make_op("contiguous"), rshp3);
        auto rshp4 = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 60, 42}}}), c3);
        auto c4    = m1.add_instruction(migraphx::make_op("contiguous"), rshp4);
        auto rshp5 = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {120, 42}}}), c4);
        auto c5    = m1.add_instruction(migraphx::make_op("contiguous"), rshp5);
        auto rshp6 = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {5040}}}), c5);
        m1.add_return({rshp6});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", s);
        auto rshp = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {5040}}}), x);
        m2.add_return({rshp});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(nested_reshape_squeeze)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {1, 2, 3, 4}};
    migraphx::module m1;
    {
        auto x       = m1.add_parameter("x", s);
        auto rshp    = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 12}}}), x);
        auto squeeze = m1.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), rshp);
        m1.add_return({squeeze});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", s);
        auto rshp = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 12}}}), x);
        m2.add_return({rshp});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(nested_squeeze_reshape)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {1, 2, 3, 4}};
    migraphx::module m1;
    {
        auto x       = m1.add_parameter("x", s);
        auto squeeze = m1.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), x);
        auto rshp = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 12}}}), squeeze);
        m1.add_return({rshp});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", s);
        auto rshp = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 12}}}), x);
        m2.add_return({rshp});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(squeeze_unsqueeze_scalar)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {1}};
    migraphx::module m1;
    {
        auto x       = m1.add_parameter("x", s);
        auto squeeze = m1.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), x);
        auto unsqueeze =
            m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), squeeze);
        m1.add_return({unsqueeze});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", s);
        m2.add_return({x});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(unsqueeze_broadcast_single)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {1}};
    migraphx::module m1;
    {
        auto x         = m1.add_parameter("x", s);
        auto unsqueeze = m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), x);
        auto broadcast = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 8}}}), unsqueeze);
        m1.add_return({broadcast});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x         = m2.add_parameter("x", s);
        auto broadcast = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 0}, {"out_lens", {1, 8}}}), x);
        m2.add_return({broadcast});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(concat_slice_different_axis_1)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {2, 160}};
    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", s);
        auto y      = m1.add_parameter("y", s);
        auto concat = m1.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), x, y);
        auto slice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), concat);
        auto slice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), concat);
        auto add = m1.add_instruction(migraphx::make_op("add"), slice1, slice2);
        m1.add_return({add});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(concat_slice_different_axis_2)
{
    // two slices, one with same axis but other with different
    auto s = migraphx::shape{migraphx::shape::float_type, {2, 160}};
    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", s);
        auto y      = m1.add_parameter("y", s);
        auto concat = m1.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), x, y);
        auto slice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), concat);
        auto slice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {160}}, {"ends", {320}}}),
            concat);
        auto add = m1.add_instruction(migraphx::make_op("add"), x, slice2);
        m1.add_return({slice1, add});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x      = m2.add_parameter("x", s);
        auto y      = m2.add_parameter("y", s);
        auto concat = m2.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), x, y);
        auto slice1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), concat);
        auto add = m2.add_instruction(migraphx::make_op("add"), x, y);
        m2.add_return({slice1, add});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(concat_slice_in_same_order)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {2, 160}};
    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", s);
        auto y      = m1.add_parameter("y", s);
        auto concat = m1.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), x, y);
        auto slice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {160}}}), concat);
        auto slice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {160}}, {"ends", {320}}}),
            concat);
        auto add = m1.add_instruction(migraphx::make_op("add"), slice1, slice2);
        m1.add_return({add});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x   = m2.add_parameter("x", s);
        auto y   = m2.add_parameter("y", s);
        auto add = m2.add_instruction(migraphx::make_op("add"), x, y);
        m2.add_return({add});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(concat_slice_in_reverse_order)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {2, 160}};
    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", s);
        auto y      = m1.add_parameter("y", s);
        auto concat = m1.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), x, y);
        auto slice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {160}}, {"ends", {320}}}),
            concat);
        auto slice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {160}}}), concat);
        auto add = m1.add_instruction(migraphx::make_op("add"), slice1, slice2);
        m1.add_return({add});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x   = m2.add_parameter("x", s);
        auto y   = m2.add_parameter("y", s);
        auto add = m2.add_instruction(migraphx::make_op("add"), y, x);
        m2.add_return({add});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(concat_slice_inorder_with_empty_slice)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {2, 160}};
    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", s);
        auto y      = m1.add_parameter("y", s);
        auto concat = m1.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), x, y);
        auto slice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {160}}}), concat);
        auto slice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {160}}, {"ends", {320}}}),
            concat);
        auto slice3 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {320}}, {"ends", {360}}}),
            concat);
        auto add = m1.add_instruction(migraphx::make_op("add"), slice1, slice2);
        m1.add_return({add, slice3});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x      = m2.add_parameter("x", s);
        auto y      = m2.add_parameter("y", s);
        auto concat = m2.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), x, y);
        auto slice3 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {320}}, {"ends", {360}}}),
            concat);
        auto add = m2.add_instruction(migraphx::make_op("add"), x, y);
        m2.add_return({add, slice3});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(concat_slice_uneven_len_1)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {2, 160}};
    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", s);
        auto y      = m1.add_parameter("y", s);
        auto z      = m1.add_parameter("z", s);
        auto concat = m1.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), x, y, z);
        auto slice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {100}}}), concat);
        auto slice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {100}}, {"ends", {160}}}),
            concat);
        auto slice3 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {160}}, {"ends", {320}}}),
            concat);
        auto slice4 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {320}}, {"ends", {420}}}),
            concat);
        auto slice5 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {420}}, {"ends", {480}}}),
            concat);
        auto add1 = m1.add_instruction(migraphx::make_op("add"), slice1, slice4);
        auto add2 = m1.add_instruction(migraphx::make_op("add"), slice2, slice5);
        auto add3 = m1.add_instruction(migraphx::make_op("add"), slice3, z);
        m1.add_return({add1, add2, add3});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x      = m2.add_parameter("x", s);
        auto y      = m2.add_parameter("y", s);
        auto z      = m2.add_parameter("z", s);
        auto concat = m2.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), x, y, z);
        auto slice1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {100}}}), concat);
        auto slice2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {100}}, {"ends", {160}}}),
            concat);
        auto slice4 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {320}}, {"ends", {420}}}),
            concat);
        auto slice5 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {420}}, {"ends", {480}}}),
            concat);
        auto add1 = m2.add_instruction(migraphx::make_op("add"), slice1, slice4);
        auto add2 = m2.add_instruction(migraphx::make_op("add"), slice2, slice5);
        auto add3 = m2.add_instruction(migraphx::make_op("add"), y, z);
        m2.add_return({add1, add2, add3});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(concat_slice_uneven_len_2)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {2, 160}};
    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", s);
        auto y      = m1.add_parameter("y", s);
        auto concat = m1.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), x, y);
        auto slice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {150}}}), concat);
        auto slice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {150}}, {"ends", {300}}}),
            concat);
        auto slice3 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {300}}, {"ends", {320}}}),
            concat);
        auto add = m1.add_instruction(migraphx::make_op("add"), slice1, slice2);
        m1.add_return({add, slice3});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(concat_slice_multiple_slice_use)
{
    // multiple use for slice1 and slice3, single use for slice2
    auto s = migraphx::shape{migraphx::shape::float_type, {2, 160}};
    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", s);
        auto y      = m1.add_parameter("y", s);
        auto z      = m1.add_parameter("z", s);
        auto concat = m1.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), x, y, z);
        auto slice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {160}}}), concat);
        auto slice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {160}}, {"ends", {320}}}),
            concat);
        auto slice3 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {320}}, {"ends", {480}}}),
            concat);
        auto add1 = m1.add_instruction(migraphx::make_op("add"), slice1, z);
        auto add2 = m1.add_instruction(migraphx::make_op("add"), slice3, x);
        auto sub1 = m1.add_instruction(migraphx::make_op("sub"), slice1, z);
        auto sub2 = m1.add_instruction(migraphx::make_op("sub"), slice3, x);
        auto add3 = m1.add_instruction(migraphx::make_op("add"), sub1, sub2);
        auto sub3 = m1.add_instruction(migraphx::make_op("sub"), add3, slice2);
        m1.add_return({add1, add2, sub3});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", s);
        auto y    = m2.add_parameter("y", s);
        auto z    = m2.add_parameter("z", s);
        auto add1 = m2.add_instruction(migraphx::make_op("add"), x, z);
        auto add2 = m2.add_instruction(migraphx::make_op("add"), z, x);
        auto sub1 = m2.add_instruction(migraphx::make_op("sub"), x, z);
        auto sub2 = m2.add_instruction(migraphx::make_op("sub"), z, x);
        auto add3 = m2.add_instruction(migraphx::make_op("add"), sub1, sub2);
        auto sub3 = m2.add_instruction(migraphx::make_op("sub"), add3, y);
        m2.add_return({add1, add2, sub3});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(concat_slice_with_multiple_concat_outs)
{
    auto s1 = migraphx::shape{migraphx::shape::float_type, {2, 160}};
    auto s2 = migraphx::shape{migraphx::shape::float_type, {2, 480}};
    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", s1);
        auto y      = m1.add_parameter("y", s1);
        auto z      = m1.add_parameter("z", s1);
        auto w      = m1.add_parameter("w", s2);
        auto concat = m1.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), x, y, z);
        auto slice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {160}}, {"ends", {320}}}),
            concat);
        auto add1 = m1.add_instruction(migraphx::make_op("add"), concat, w);
        auto add2 = m1.add_instruction(migraphx::make_op("add"), slice1, z);
        m1.add_return({add1, add2});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x      = m2.add_parameter("x", s1);
        auto y      = m2.add_parameter("y", s1);
        auto z      = m2.add_parameter("z", s1);
        auto w      = m2.add_parameter("w", s2);
        auto concat = m2.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), x, y, z);
        auto add1   = m2.add_instruction(migraphx::make_op("add"), concat, w);
        auto add2   = m2.add_instruction(migraphx::make_op("add"), y, z);
        m2.add_return({add1, add2});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(concat_multibroadcasts1)
{
    // Broadcasted batch dim, new axis < old axis
    std::vector<std::size_t> in_lens     = {3, 4};
    std::vector<std::size_t> mbcast_lens = {2, 3, 4};
    const int axis                       = 2;
    auto m                               = make_concat_multibroadcast(in_lens, mbcast_lens, axis);
    auto out_shape                       = m.get_output_shapes().back();
    auto n                               = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back().lens() == out_shape.lens());
    EXPECT(std::distance(m.begin(), m.end()) == n - 2);
    auto new_concat =
        std::find_if(m.begin(), m.end(), [](const auto& ins) { return ins.name() == "concat"; });
    EXPECT(bool{new_concat != m.end()});
    auto cd = std::distance(m.begin(), new_concat);
    auto new_mb = std::find_if(
        m.begin(), m.end(), [](const auto& ins) { return ins.name() == "multibroadcast"; });
    auto md = std::distance(m.begin(), new_mb);
    EXPECT(cd == md - 1);
    EXPECT(new_concat->get_operator().to_value()["axis"].to<int>() == 1);
}

TEST_CASE(concat_multibroadcasts2)
{
    // Broadcasted middle dim, new axis == old axis
    std::vector<std::size_t> in_lens     = {3, 1, 4};
    std::vector<std::size_t> mbcast_lens = {3, 2, 4};
    const int axis                       = 0;
    auto m                               = make_concat_multibroadcast(in_lens, mbcast_lens, axis);
    auto out_shape                       = m.get_output_shapes().back();
    auto n                               = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back().lens() == out_shape.lens());
    EXPECT(std::distance(m.begin(), m.end()) == n - 2);
    auto new_concat =
        std::find_if(m.begin(), m.end(), [](const auto& ins) { return ins.name() == "concat"; });
    EXPECT(bool{new_concat != m.end()});
    auto cd = std::distance(m.begin(), new_concat);
    auto new_mb = std::find_if(
        m.begin(), m.end(), [](const auto& ins) { return ins.name() == "multibroadcast"; });
    auto md = std::distance(m.begin(), new_mb);
    EXPECT(cd == md - 1);
    EXPECT(new_concat->get_operator().to_value()["axis"].to<int>() == 0);
}

TEST_CASE(concat_multibroadcasts3)
{
    // Broadcasted middle dim, new axis == old axis
    std::vector<std::size_t> in_lens     = {3, 1, 4};
    std::vector<std::size_t> mbcast_lens = {3, 2, 4};
    const int axis                       = 2;
    auto m                               = make_concat_multibroadcast(in_lens, mbcast_lens, axis);
    auto out_shape                       = m.get_output_shapes().back();
    auto n                               = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back().lens() == out_shape.lens());
    EXPECT(std::distance(m.begin(), m.end()) == n - 2);
    auto new_concat =
        std::find_if(m.begin(), m.end(), [](const auto& ins) { return ins.name() == "concat"; });
    EXPECT(bool{new_concat != m.end()});
    auto cd = std::distance(m.begin(), new_concat);
    auto new_mb = std::find_if(
        m.begin(), m.end(), [](const auto& ins) { return ins.name() == "multibroadcast"; });
    auto md = std::distance(m.begin(), new_mb);
    EXPECT(cd == md - 1);
    EXPECT(new_concat->get_operator().to_value()["axis"].to<int>() == 2);
}

// Broadcasted batch dim, axis is broadcasted dim
// matched by find_concat_multibroadcasts but it skips this case
TEST_CASE(concat_multibroadcasts4)
{
    std::vector<std::size_t> in_lens     = {3, 4};
    std::vector<std::size_t> mbcast_lens = {2, 3, 4};
    const int axis                       = 0;
    auto m                               = make_concat_multibroadcast(in_lens, mbcast_lens, axis);
    auto m1                              = m;
    run_pass(m);
    EXPECT(m1 == m);
}

// Matched by find_concat_multibroadcasts but skipped because dimensions other than concat axis do
// not match
TEST_CASE(concat_multibroadcasts5)
{
    migraphx::module m;
    auto s0 = migraphx::shape{migraphx::shape::float_type, {1, 1, 1, 1, 64}};
    auto s1 = migraphx::shape{migraphx::shape::float_type, {1, 1, 60, 64, 192}};
    auto x  = m.add_parameter("x", s0);
    auto y  = m.add_parameter("y", s1);
    std::vector<std::size_t> mb_lens0 = {1, 12, 60, 64, 64};
    std::vector<std::size_t> mb_lens1 = {1, 12, 60, 64, 192};
    auto mb_x = m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", mb_lens0}}), x);
    auto mb_y = m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", mb_lens1}}), y);
    auto concat_xy = m.add_instruction(migraphx::make_op("concat", {{"axis", 4}}), mb_x, mb_y);
    m.add_return({concat_xy});
    auto m_original = m;
    run_pass(m);
    EXPECT(m == m_original);
}

// Matched by find_concat_multibroadcasts but skipped because parameter inputs are not the same
// rank.
TEST_CASE(concat_multibroadcasts6)
{
    migraphx::module m;
    auto s0                           = migraphx::shape{migraphx::shape::float_type, {64}};
    auto s1                           = migraphx::shape{migraphx::shape::float_type, {60, 64, 192}};
    auto x                            = m.add_parameter("x", s0);
    auto y                            = m.add_parameter("y", s1);
    std::vector<std::size_t> mb_lens0 = {12, 60, 64, 64};
    std::vector<std::size_t> mb_lens1 = {12, 60, 64, 192};
    auto mb_x = m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", mb_lens0}}), x);
    auto mb_y = m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", mb_lens1}}), y);
    auto concat_xy = m.add_instruction(migraphx::make_op("concat", {{"axis", 3}}), mb_x, mb_y);
    m.add_return({concat_xy});
    auto m_original = m;
    run_pass(m);
    EXPECT(m == m_original);
}

// Concat axis moved to 2 because rank(in_dims) < rank(out_dims)
// Matched by find_concat_multibroadcasts but skipped because the dimensions
// other than the concat axis are not the same.
// TODO: has common broadcast axes, so can be simplified by moving multibroadcast up to have a
// smaller concat.
TEST_CASE(concat_multibroadcasts7)
{
    migraphx::module m;
    auto s0                           = migraphx::shape{migraphx::shape::float_type, {1, 1, 64}};
    auto s1                           = migraphx::shape{migraphx::shape::float_type, {60, 64, 192}};
    auto x                            = m.add_parameter("x", s0);
    auto y                            = m.add_parameter("y", s1);
    std::vector<std::size_t> mb_lens0 = {1, 12, 60, 64, 64};
    std::vector<std::size_t> mb_lens1 = {1, 12, 60, 64, 192};
    auto mb_x = m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", mb_lens0}}), x);
    auto mb_y = m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", mb_lens1}}), y);
    auto concat_xy = m.add_instruction(migraphx::make_op("concat", {{"axis", 4}}), mb_x, mb_y);
    m.add_return({concat_xy});
    auto m_original = m;
    run_pass(m);
    EXPECT(m == m_original);
}

// Shape of inputs to multibroadcasts do not have the same rank.
// Matched by find_concat_multibroadcasts but skipped.
// TODO: has a common broadcast axis, so can be simplified by moving multibroadcast up to have a
// smaller concat.
TEST_CASE(concat_multibroadcasts8)
{
    migraphx::module m;
    auto s0                           = migraphx::shape{migraphx::shape::float_type, {64, 64}};
    auto s1                           = migraphx::shape{migraphx::shape::float_type, {60, 1, 192}};
    auto x                            = m.add_parameter("x", s0);
    auto y                            = m.add_parameter("y", s1);
    std::vector<std::size_t> mb_lens0 = {60, 64, 64};
    std::vector<std::size_t> mb_lens1 = {60, 64, 192};
    auto mb_x = m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", mb_lens0}}), x);
    auto mb_y = m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", mb_lens1}}), y);
    auto concat_xy = m.add_instruction(migraphx::make_op("concat", {{"axis", 2}}), mb_x, mb_y);
    m.add_return({concat_xy});
    auto m_original = m;
    run_pass(m);
    EXPECT(m == m_original);
}

// Shape of inputs to multibroadcasts do not have a common broadcast axis.
// Matched by find_concat_multibroadcasts, but skipped because the dimensions other than
// the concat axis are not the same.
TEST_CASE(concat_multibroadcasts9)
{
    migraphx::module m;
    auto s0                           = migraphx::shape{migraphx::shape::float_type, {1, 64, 64}};
    auto s1                           = migraphx::shape{migraphx::shape::float_type, {60, 1, 192}};
    auto x                            = m.add_parameter("x", s0);
    auto y                            = m.add_parameter("y", s1);
    std::vector<std::size_t> mb_lens0 = {60, 64, 64};
    std::vector<std::size_t> mb_lens1 = {60, 64, 192};
    auto mb_x = m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", mb_lens0}}), x);
    auto mb_y = m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", mb_lens1}}), y);
    auto concat_xy = m.add_instruction(migraphx::make_op("concat", {{"axis", 2}}), mb_x, mb_y);
    m.add_return({concat_xy});
    auto m_original = m;
    run_pass(m);
    EXPECT(m == m_original);
}

TEST_CASE(concat_broadcast1)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {1024, 1024}};
    migraphx::module m1;
    {
        auto x  = m1.add_parameter("x", s);
        auto y  = m1.add_parameter("y", s);
        auto xb = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {8, 1024, 1024}}}), x);
        auto yb = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {8, 1024, 1024}}}), y);
        auto concat = m1.add_instruction(migraphx::make_op("concat", {{"axis", 2}}), xb, yb);
        m1.add_return({concat});
    }
    migraphx::module m2;
    {
        auto x      = m2.add_parameter("x", s);
        auto y      = m2.add_parameter("y", s);
        auto concat = m2.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), x, y);
        auto b      = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {8, 1024, 2048}}}), concat);
        m2.add_return({b});
    }
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(concat_transpose1)
{
    migraphx::module m;

    auto s  = migraphx::shape{migraphx::shape::float_type, {1, 2, 3, 4}};
    auto x  = m.add_parameter("x", s);
    auto y  = m.add_parameter("y", s);
    auto xt = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), x);
    auto yt = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), y);
    auto concat = m.add_instruction(migraphx::make_op("concat", {{"axis", 2}}), xt, yt);
    auto t =
        m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), concat);
    m.add_return({t});
    auto out_shape = m.get_output_shapes().back();
    auto n         = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back().lens() == out_shape.lens());
    EXPECT(std::distance(m.begin(), m.end()) == n - 3);
    auto new_concat =
        std::find_if(m.begin(), m.end(), [](const auto& ins) { return ins.name() == "concat"; });
    EXPECT(bool{new_concat != m.end()});
    EXPECT(new_concat->get_operator().to_value()["axis"].to<int>() == 3);
}

TEST_CASE(concat_transpose2)
{
    migraphx::module m;

    auto s  = migraphx::shape{migraphx::shape::float_type, {1, 2, 3, 4}};
    auto x  = m.add_parameter("x", s);
    auto y  = m.add_parameter("y", s);
    auto xt = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), x);
    auto yt = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), y);
    auto concat = m.add_instruction(migraphx::make_op("concat", {{"axis", -1}}), xt, yt);
    auto t =
        m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), concat);
    m.add_return({t});
    auto out_shape = m.get_output_shapes().back();
    auto n         = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back().lens() == out_shape.lens());
    EXPECT(std::distance(m.begin(), m.end()) == n - 2);
    auto new_concat =
        std::find_if(m.begin(), m.end(), [](const auto& ins) { return ins.name() == "concat"; });
    EXPECT(bool{new_concat != m.end()});
    EXPECT(new_concat->get_operator().to_value()["axis"].to<int>() == 1);
}

TEST_CASE(concat_transpose3)
{
    migraphx::module m;

    auto s  = migraphx::shape{migraphx::shape::float_type, {1, 2, 3, 4}};
    auto x  = m.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 2, 3, 4}});
    auto y  = m.add_parameter("y", migraphx::shape{migraphx::shape::float_type, {1, 5, 3, 4}});
    auto xt = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), x);
    auto yt = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), y);
    auto concat = m.add_instruction(migraphx::make_op("concat", {{"axis", 3}}), xt, yt);
    auto t =
        m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), concat);
    m.add_return({t});
    auto out_shape = m.get_output_shapes().back();
    auto n         = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back().lens() == out_shape.lens());
    EXPECT(std::distance(m.begin(), m.end()) == n - 2);
    auto new_concat =
        std::find_if(m.begin(), m.end(), [](const auto& ins) { return ins.name() == "concat"; });
    EXPECT(bool{new_concat != m.end()});
    EXPECT(new_concat->get_operator().to_value()["axis"].to<int>() == 1);
}

TEST_CASE(concat_transpose4)
{
    migraphx::module m;
    auto sx = migraphx::shape{migraphx::shape::float_type, {1, 1, 12, 64}};
    auto sy = migraphx::shape{migraphx::shape::float_type, {1, 12, 1, 64}};
    auto x  = m.add_parameter("x", sx);
    auto y  = m.add_parameter("y", sy);
    auto xt = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), x);
    auto yt = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), y);
    auto concat = m.add_instruction(migraphx::make_op("concat", {{"axis", 3}}), xt, yt);
    auto t =
        m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), concat);
    m.add_return({t});

    migraphx::module m1 = m;
    run_pass(m);

    EXPECT(m1 == m);
}

TEST_CASE(concat_unsqueeze)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {11008, 4096}};
    migraphx::module m1;
    {
        auto x          = m1.add_parameter("x", s);
        auto y          = m1.add_parameter("y", s);
        auto xunsqueeze = m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), x);
        auto yunsqueeze = m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), y);
        auto concat =
            m1.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), xunsqueeze, yunsqueeze);
        m1.add_return({concat});
    }
    migraphx::module m2;
    {
        auto x      = m2.add_parameter("x", s);
        auto y      = m2.add_parameter("y", s);
        auto concat = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), x, y);
        auto unsqueeze =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 22016, 4096}}}), concat);
        m2.add_return({unsqueeze});
    }
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(concat_reshape)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {11008, 32, 128}};
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", s);
        auto y = m1.add_parameter("y", s);
        auto xreshape =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {11008, 4096}}}), x);
        auto yreshape =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {11008, 4096}}}), y);
        auto concat =
            m1.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), xreshape, yreshape);
        m1.add_return({concat});
    }
    migraphx::module m2;
    {
        auto x      = m2.add_parameter("x", s);
        auto y      = m2.add_parameter("y", s);
        auto concat = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), x, y);
        auto reshape =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {22016, 4096}}}), concat);
        m2.add_return({reshape});
    }
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(concat_reshape_change_axis)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {2, 256, 1280}};
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", s);
        auto y = m1.add_parameter("y", s);
        auto xreshape =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 16, 16, 1280}}}), x);
        auto yreshape =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 16, 16, 1280}}}), y);
        auto concat =
            m1.add_instruction(migraphx::make_op("concat", {{"axis", 3}}), xreshape, yreshape);
        m1.add_return({concat});
    }
    migraphx::module m2;
    {
        auto x      = m2.add_parameter("x", s);
        auto y      = m2.add_parameter("y", s);
        auto concat = m2.add_instruction(migraphx::make_op("concat", {{"axis", 2}}), x, y);
        auto reshape =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 16, 16, 2560}}}), concat);
        m2.add_return({reshape});
    }
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(concat_reshape_broadcast)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {11008, 32, 1}};
    migraphx::module m1;
    {
        auto x  = m1.add_parameter("x", s);
        auto y  = m1.add_parameter("y", s);
        auto xb = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {11008, 32, 128}}}), x);
        auto yb = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {11008, 32, 128}}}), y);
        auto xreshape =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {11008, 4096}}}), xb);
        auto yreshape =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {11008, 4096}}}), yb);
        auto concat =
            m1.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), xreshape, yreshape);
        m1.add_return({concat});
    }
    migraphx::module m2;
    {
        auto x         = m2.add_parameter("x", s);
        auto y         = m2.add_parameter("y", s);
        auto concat    = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), x, y);
        auto broadcast = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {22016, 32, 128}}}), concat);
        auto reshape =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {22016, 4096}}}), broadcast);
        m2.add_return({reshape});
    }
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(nested_concat)
{
    migraphx::module m;

    auto s       = migraphx::shape{migraphx::shape::float_type, {1, 2, 3, 4}};
    auto x       = m.add_parameter("x", s);
    auto y       = m.add_parameter("y", s);
    auto concat1 = m.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), x, y);
    auto concat2 = m.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), y, x);
    auto concat3 = m.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), concat1, concat2);
    m.add_return({concat3});
    auto out_shape = m.get_output_shapes().back();
    auto n         = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back().lens() == out_shape.lens());
    EXPECT(std::distance(m.begin(), m.end()) == n - 2);
    EXPECT(std::count_if(m.begin(), m.end(), [](auto ins) { return ins.name() == "concat"; }) == 1);
}

TEST_CASE(nested_concat_partial)
{
    migraphx::module m;

    auto s = migraphx::shape{migraphx::shape::float_type, {1, 2, 3, 4}};
    auto x = m.add_parameter("x", s);
    auto y = m.add_parameter("y", s);
    auto l = m.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1, 4, 3, 4}}));
    auto concat1 = m.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), x, y);
    auto concat2 = m.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), y, x);
    auto concat3 =
        m.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), concat1, concat2, l);
    m.add_return({concat3});
    auto out_shape = m.get_output_shapes().back();
    auto n         = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back().lens() == out_shape.lens());
    EXPECT(std::distance(m.begin(), m.end()) == n - 2);
    EXPECT(std::count_if(m.begin(), m.end(), [](auto ins) { return ins.name() == "concat"; }) == 1);
}

TEST_CASE(multibroadcast_simplify)
{
    migraphx::module m;

    std::vector<size_t> s_lens{1, 2, 3, 4};
    auto s = migraphx::shape{migraphx::shape::float_type, s_lens};
    auto x = m.add_parameter("x", s);
    auto y = m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s_lens}}), x);
    m.add_instruction(migraphx::make_op("mul"), y, y);
    auto n = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(std::distance(m.begin(), m.end()) == n - 1);
}

TEST_CASE(multibroadcast_unsqueeze_scalar)
{
    migraphx::module m1;
    {
        auto l   = m1.add_parameter("x", {migraphx::shape::float_type, {1}, {0}});
        auto mb  = m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2}}}), l);
        auto t1  = m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), mb);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), t1, t1);
        m1.add_return({mul});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto l = m2.add_parameter("x", {migraphx::shape::float_type, {1}, {0}});
        auto mb =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 1}}}), l);
        auto mul = m2.add_instruction(migraphx::make_op("mul"), mb, mb);
        m2.add_return({mul});
    }

    EXPECT(m1 == m2);
}

TEST_CASE(multibroadcast_unsqueeze_cont_scalar)
{
    migraphx::module m1;
    {
        auto l   = m1.add_parameter("x", {migraphx::shape::float_type, {1}, {0}});
        auto mb  = m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2}}}), l);
        auto cnt = m1.add_instruction(migraphx::make_op("contiguous"), mb);
        auto t1  = m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), cnt);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), t1, t1);
        m1.add_return({mul});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto l = m2.add_parameter("x", {migraphx::shape::float_type, {1}, {0}});
        auto mb =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 1}}}), l);
        auto mul = m2.add_instruction(migraphx::make_op("mul"), mb, mb);
        m2.add_return({mul});
    }

    EXPECT(m1 == m2);
}

TEST_CASE(double_slice1)
{
    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", {migraphx::shape::int32_type, {256}});
        auto slice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {32}}, {"ends", {256}}}), x);
        auto slice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {32}}, {"ends", {64}}}), slice1);
        m1.add_return({slice2});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x     = m2.add_parameter("x", {migraphx::shape::int32_type, {256}});
        auto slice = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {64}}, {"ends", {96}}}), x);
        m2.add_return({slice});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(double_slice2)
{
    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", {migraphx::shape::int32_type, {256}});
        auto slice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {32}}, {"ends", {128}}}), x);
        auto slice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {32}}}), slice1);
        m1.add_return({slice2});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x     = m2.add_parameter("x", {migraphx::shape::int32_type, {256}});
        auto slice = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {32}}, {"ends", {64}}}), x);
        m2.add_return({slice});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(double_slice_multi_axes)
{
    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", {migraphx::shape::int32_type, {256, 128}});
        auto slice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {32}}, {"ends", {128}}}), x);
        auto slice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {32}}}), slice1);
        m1.add_return({slice2});
    }
    run_pass(m1);

    migraphx::module m2;

    {
        auto x     = m2.add_parameter("x", {migraphx::shape::int32_type, {256, 128}});
        auto slice = m2.add_instruction(
            migraphx::make_op("slice",
                              {{"axes", {0, 1}}, {"starts", {32, 0}}, {"ends", {128, 32}}}),
            x);
        m2.add_return({slice});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(optimize_resize)
{
    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 2}};
    auto create_resize_module = [&] {
        migraphx::module m;
        auto inx = m.add_parameter("X", sx);

        migraphx::shape si{migraphx::shape::int32_type, {1, 2, 4, 6}};
        std::vector<int> ind = {0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3,
                                3, 3, 2, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1, 0, 0,
                                0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 3, 3, 3};
        auto li              = m.add_literal(migraphx::literal(si, ind));

        auto lrsp = m.add_instruction(migraphx::make_op("reshape", {{"dims", {4}}}), inx);
        auto gr   = m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), lrsp, li);
        auto r    = m.add_instruction(migraphx::make_op("softmax", {{"axis", 1}}), gr);
        m.add_return({r});

        return m;
    };

    auto m1 = create_resize_module();
    run_pass(m1);

    auto create_optimized_module = [&] {
        migraphx::module m;
        auto inx                  = m.add_parameter("X", sx);
        std::vector<int64_t> dims = {1, 1, 2, 1, 2, 1};
        auto rspx = m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {3, 5}}}), inx);
        auto mbx  = m.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 1, 2, 2, 2, 3}}}), rspx);
        std::vector<int64_t> orig_dims = {1, 2, 4, 6};
        auto rmb  = m.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 4, 6}}}), mbx);
        auto rmbb = m.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 2, 4, 6}}}), rmb);
        auto r = m.add_instruction(migraphx::make_op("softmax", {{"axis", 1}}), rmbb);
        m.add_return({r});

        return m;
    };

    EXPECT(m1 == create_optimized_module());
}

TEST_CASE(optimize_resize_ind_not_apply)
{
    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 2}};
    auto create_resize_module = [&] {
        migraphx::module m;
        auto inx = m.add_parameter("X", sx);

        migraphx::shape si{migraphx::shape::int32_type, {1, 2, 4, 6}};
        std::vector<int> ind = {0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 2, 2, 2, 3,
                                3, 3, 2, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1, 0, 0,
                                0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 3, 3, 3};
        auto li              = m.add_literal(migraphx::literal(si, ind));

        auto lrsp = m.add_instruction(migraphx::make_op("reshape", {{"dims", {4}}}), inx);
        auto gr   = m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), lrsp, li);
        auto r    = m.add_instruction(migraphx::make_op("softmax", {{"axis", 1}}), gr);
        m.add_return({r});

        return m;
    };

    auto m1 = create_resize_module();
    run_pass(m1);
    EXPECT(m1 == create_resize_module());
}

TEST_CASE(optimize_resize_rsp_dim_1)
{
    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 2}};
    auto create_resize_module = [&] {
        migraphx::module m;
        auto inx = m.add_parameter("X", sx);

        migraphx::shape si{migraphx::shape::int32_type, {1, 1, 4, 3, 2}};
        std::vector<int> ind = {0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1,
                                2, 2, 2, 3, 3, 3, 2, 2, 2, 3, 3, 3};
        auto li              = m.add_literal(migraphx::literal(si, ind));

        auto lrsp = m.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2}}}), inx);
        auto r    = m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), lrsp, li);
        m.add_return({r});

        return m;
    };

    auto m = create_resize_module();
    run_pass(m);
    EXPECT(m == create_resize_module());
}

TEST_CASE(optimize_resize_ndims_unequal)
{
    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 2}};
    migraphx::shape sy{migraphx::shape::float_type, {1, 1, 4, 3, 2}};
    auto create_resize_module = [&] {
        migraphx::module m;
        auto inx = m.add_parameter("X", sx);
        auto iny = m.add_parameter("Y", sy);

        migraphx::shape si{migraphx::shape::int32_type, {1, 1, 4, 3, 2}};
        std::vector<int> ind = {0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1,
                                2, 2, 2, 3, 3, 3, 2, 2, 2, 3, 3, 3};
        auto li              = m.add_literal(migraphx::literal(si, ind));

        auto lrsp = m.add_instruction(migraphx::make_op("reshape", {{"dims", {4}}}), inx);
        auto gr   = m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), lrsp, li);
        auto r    = m.add_instruction(migraphx::make_op("sub"), iny, gr);
        m.add_return({r});

        return m;
    };

    auto m = create_resize_module();
    run_pass(m);
    EXPECT(m == create_resize_module());
}

TEST_CASE(optimize_resize_ind_non_brcst)
{
    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 3, 2}};
    migraphx::shape sy{migraphx::shape::float_type, {1, 1, 4, 6}};
    auto create_resize_module = [&] {
        migraphx::module m;
        auto inx = m.add_parameter("X", sx);
        auto iny = m.add_parameter("Y", sy);

        migraphx::shape si{migraphx::shape::int32_type, {1, 1, 4, 6}};
        std::vector<int> ind = {0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1,
                                2, 2, 2, 3, 3, 3, 2, 2, 2, 3, 3, 3};
        auto li              = m.add_literal(migraphx::literal(si, ind));

        auto lrsp = m.add_instruction(migraphx::make_op("reshape", {{"dims", {6}}}), inx);
        auto gr   = m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), lrsp, li);
        auto r    = m.add_instruction(migraphx::make_op("sub"), iny, gr);
        m.add_return({r});

        return m;
    };

    auto m = create_resize_module();
    run_pass(m);
    EXPECT(m == create_resize_module());
}

TEST_CASE(optimize_resize_ind_non_const)
{
    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 3, 2}};
    migraphx::shape sy{migraphx::shape::float_type, {1, 1, 4, 6}};
    auto create_resize_module = [&] {
        migraphx::module m;
        auto inx = m.add_parameter("X", sx);
        auto iny = m.add_parameter("Y", sy);

        migraphx::shape si{migraphx::shape::int32_type, {1, 1, 4, 6}};
        auto li   = m.add_parameter("ind", si);
        auto lrsp = m.add_instruction(migraphx::make_op("reshape", {{"dims", {6}}}), inx);
        auto gr   = m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), lrsp, li);
        auto r    = m.add_instruction(migraphx::make_op("sub"), iny, gr);
        m.add_return({r});

        return m;
    };

    auto m = create_resize_module();
    run_pass(m);
    EXPECT(m == create_resize_module());
}

TEST_CASE(optimize_where_true)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 1, 3, 2}};
    auto create_where_module = [&](bool cond) {
        migraphx::module m;
        auto inx = m.add_parameter("X", s);
        auto iny = m.add_parameter("Y", s);

        migraphx::shape si{migraphx::shape::bool_type, {1, 1, 3, 2}};
        std::vector<char> idata(si.elements(), static_cast<char>(cond));
        auto li     = m.add_literal(migraphx::literal(si, idata));
        auto data   = m.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), inx, iny);
        auto data_1 = m.add_instruction(migraphx::make_op("reshape", {{"dims", {12}}}), data);
        auto r      = m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), data_1, li);
        m.add_return({r});
        return m;
    };

    auto return_xy = [&](bool cond) {
        migraphx::module m;
        auto x = m.add_parameter("X", s);
        auto y = m.add_parameter("Y", s);
        cond ? m.add_return({x}) : m.add_return({y});
        return m;
    };

    auto m = create_where_module(true);
    run_pass(m);
    EXPECT(m == return_xy(true));

    auto m1 = create_where_module(false);
    run_pass(m1);
    EXPECT(m1 == return_xy(false));
}

TEST_CASE(where_different_cond_values)
{
    auto create_where_module = [] {
        migraphx::module m;
        migraphx::shape s{migraphx::shape::float_type, {1, 1, 3, 2}};
        auto inx = m.add_parameter("X", s);
        auto iny = m.add_parameter("Y", s);

        migraphx::shape si{migraphx::shape::bool_type, {1, 1, 3, 2}};
        std::vector<char> idata = {1, 1, 0, 1, 0, 1};
        auto li                 = m.add_literal(migraphx::literal(si, idata));
        auto data   = m.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), inx, iny);
        auto data_1 = m.add_instruction(migraphx::make_op("reshape", {{"dims", {12}}}), data);
        auto r      = m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), data_1, li);
        m.add_return({r});
        return m;
    };

    auto m = create_where_module();
    run_pass(m);
    EXPECT(m == create_where_module());
}

TEST_CASE(where_axis_nonzero)
{
    auto create_where_module = [] {
        migraphx::module m;
        migraphx::shape s{migraphx::shape::float_type, {1, 1, 3, 2}};
        auto inx = m.add_parameter("X", s);
        auto iny = m.add_parameter("Y", s);

        migraphx::shape si{migraphx::shape::bool_type, {1, 1, 3, 2}};
        std::vector<char> idata(6, 1);
        auto li     = m.add_literal(migraphx::literal(si, idata));
        auto data   = m.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), inx, iny);
        auto data_1 = m.add_instruction(migraphx::make_op("reshape", {{"dims", {12}}}), data);
        auto r      = m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), data_1, li);
        m.add_return({r});
        return m;
    };

    auto m = create_where_module();
    run_pass(m);
    EXPECT(m == create_where_module());
}

TEST_CASE(where_three_concat_inputs)
{
    auto create_where_module = [] {
        migraphx::module m;
        migraphx::shape s{migraphx::shape::float_type, {1, 1, 3, 2}};
        auto inx = m.add_parameter("X", s);
        auto iny = m.add_parameter("Y", s);

        migraphx::shape si{migraphx::shape::bool_type, {1, 1, 3, 2}};
        std::vector<char> idata(6, 1);
        auto li     = m.add_literal(migraphx::literal(si, idata));
        auto data   = m.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), inx, iny, inx);
        auto data_1 = m.add_instruction(migraphx::make_op("reshape", {{"dims", {18}}}), data);
        auto r      = m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), data_1, li);
        m.add_return({r});
        return m;
    };

    auto m = create_where_module();
    run_pass(m);
    EXPECT(m == create_where_module());
}

TEST_CASE(where_three_inputs_diff_shapes)
{
    auto create_where_module = [] {
        migraphx::module m;
        migraphx::shape sx{migraphx::shape::float_type, {1, 1, 3, 2}};
        migraphx::shape sy{migraphx::shape::float_type, {2, 1, 3, 2}};
        auto inx = m.add_parameter("X", sx);
        auto iny = m.add_parameter("Y", sy);

        migraphx::shape si{migraphx::shape::bool_type, {1, 1, 3, 2}};
        std::vector<char> idata(6, 1);
        auto li     = m.add_literal(migraphx::literal(si, idata));
        auto data   = m.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), inx, iny);
        auto data_1 = m.add_instruction(migraphx::make_op("reshape", {{"dims", {18}}}), data);
        auto r      = m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), data_1, li);
        m.add_return({r});
        return m;
    };

    auto m = create_where_module();
    run_pass(m);
    EXPECT(m == create_where_module());
}

TEST_CASE(where_three_lens_diff)
{
    auto create_where_module = [] {
        migraphx::module m;
        migraphx::shape sx{migraphx::shape::float_type, {1, 1, 3, 2}};
        migraphx::shape sy{migraphx::shape::float_type, {1, 1, 3, 2}};
        auto inx = m.add_parameter("X", sx);
        auto iny = m.add_parameter("Y", sy);

        migraphx::shape si{migraphx::shape::bool_type, {1, 1, 6}};
        std::vector<char> idata(6, 1);
        auto li     = m.add_literal(migraphx::literal(si, idata));
        auto data   = m.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), inx, iny);
        auto data_1 = m.add_instruction(migraphx::make_op("reshape", {{"dims", {12}}}), data);
        auto r      = m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), data_1, li);
        m.add_return({r});
        return m;
    };

    auto m = create_where_module();
    run_pass(m);
    EXPECT(m == create_where_module());
}

TEST_CASE(reshape_cont)
{
    auto create_module = [] {
        migraphx::module m;
        migraphx::shape sx{migraphx::shape::float_type, {1, 4, 1}};
        migraphx::shape sy{migraphx::shape::float_type, {2, 2, 2, 6}};

        auto inx = m.add_parameter("x", sx);
        auto iny = m.add_parameter("y", sy);
        auto mb_inx =
            m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 6}}}), inx);
        auto std_inx = m.add_instruction(migraphx::make_op("contiguous"), mb_inx);
        auto rsp =
            m.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 6}}}), std_inx);
        auto r = m.add_instruction(migraphx::make_op("add"), rsp, iny);
        m.add_return({r});

        return m;
    };

    auto m1 = create_module();
    run_pass(m1);

    auto create_opt_module = [] {
        migraphx::module m;
        migraphx::shape sx{migraphx::shape::float_type, {1, 4, 1}};
        migraphx::shape sy{migraphx::shape::float_type, {2, 2, 2, 6}};

        auto inx = m.add_parameter("x", sx);
        auto iny = m.add_parameter("y", sy);
        auto rsp = m.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 2, 1}}}), inx);
        auto mb_inx = m.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 2, 6}}}), rsp);
        auto r = m.add_instruction(migraphx::make_op("add"), mb_inx, iny);
        m.add_return({r});

        return m;
    };
    auto m2 = create_opt_module();

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reshape_input_non_std)
{
    migraphx::shape sx{migraphx::shape::float_type, {1, 4, 1}};
    migraphx::shape sy{migraphx::shape::float_type, {2, 6, 2, 2}};
    migraphx::module m1;
    {
        auto inx = m1.add_parameter("x", sx);
        auto iny = m1.add_parameter("y", sy);
        auto mb_inx =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 6}}}), inx);
        auto std_inx = m1.add_instruction(migraphx::make_op("contiguous"), mb_inx);
        auto rsp =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 6}}}), std_inx);
        auto ty = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), iny);
        auto r = m1.add_instruction(migraphx::make_op("add"), rsp, ty);
        m1.add_return({r});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto inx = m2.add_parameter("x", sx);
        auto iny = m2.add_parameter("y", sy);
        auto rsp = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 2, 1}}}), inx);
        auto mb_inx = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 2, 6}}}), rsp);
        auto ty = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), iny);
        auto r = m2.add_instruction(migraphx::make_op("add"), mb_inx, ty);
        m2.add_return({r});
    }

    EXPECT(m1 == m2);
}

TEST_CASE(reshape_cont_nonpw)
{
    migraphx::module m1;
    {
        migraphx::shape sx{migraphx::shape::float_type, {1, 4, 1}};
        migraphx::shape sy{migraphx::shape::float_type, {2, 2, 2, 6}};

        auto inx = m1.add_parameter("x", sx);
        auto iny = m1.add_parameter("y", sy);
        auto mb_inx =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 6}}}), inx);
        auto std_inx = m1.add_instruction(migraphx::make_op("contiguous"), mb_inx);
        auto rsp =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 6}}}), std_inx);
        auto r = m1.add_instruction(migraphx::make_op("convolution"), rsp, iny);
        m1.add_return({r});
    }

    run_pass(m1);

    migraphx::module m2;
    {
        migraphx::shape sx{migraphx::shape::float_type, {1, 4, 1}};
        migraphx::shape sy{migraphx::shape::float_type, {2, 2, 2, 6}};

        auto inx = m2.add_parameter("x", sx);
        auto iny = m2.add_parameter("y", sy);
        auto rsp = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 2, 1}}}), inx);
        auto mb_inx = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 2, 6}}}), rsp);
        auto r = m2.add_instruction(migraphx::make_op("convolution"), mb_inx, iny);
        m2.add_return({r});
    }

    EXPECT(m1 == m2);
}

TEST_CASE(reshape_unary_transpose)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {2, 8, 5, 5}};
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", s);
        auto reshape_ins =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 2, 5, 5}}}), x);
        auto relu      = m1.add_instruction(migraphx::make_op("relu"), reshape_ins);
        auto transpose = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 3, 4, 1, 5, 2}}}), relu);
        m1.add_instruction(pass_op{}, transpose);
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", s);
        auto relu = m2.add_instruction(migraphx::make_op("relu"), x);
        auto reshape_ins =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 2, 5, 5}}}), relu);
        auto transpose = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 3, 4, 1, 5, 2}}}), reshape_ins);
        m2.add_instruction(pass_op{}, transpose);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(reshape_unary_last)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {2, 8, 5, 5}};
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", s);
        auto reshape_ins =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 2, 5, 5}}}), x);
        m1.add_instruction(migraphx::make_op("relu"), reshape_ins);
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(pointwise_reshape_unary_pointwise)
{
    auto s1 = migraphx::shape{migraphx::shape::float_type, {2, 8, 5, 5}};
    auto s2 = migraphx::shape{migraphx::shape::float_type, {2, 2, 2, 2, 5, 5}};
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", s1);
        auto y   = m1.add_parameter("y", s1);
        auto z   = m1.add_parameter("z", s2);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), x, y);
        auto reshape_ins =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 2, 5, 5}}}), mul);
        auto relu = m1.add_instruction(migraphx::make_op("relu"), reshape_ins);
        auto pw   = m1.add_instruction(migraphx::make_op("add"), z, relu);
        m1.add_instruction(pass_op{}, pw);
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(literal_reshape_unary_transpose_pointwise)
{
    auto s1 = migraphx::shape{migraphx::shape::float_type, {2, 8, 5, 5}};
    auto s2 = migraphx::shape{migraphx::shape::float_type, {2, 2, 5, 2, 5, 2}};
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", s2);
        auto one = m1.add_literal(migraphx::generate_literal(s1));
        auto reshape_ins =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 2, 5, 5}}}), one);
        auto relu      = m1.add_instruction(migraphx::make_op("relu"), reshape_ins);
        auto transpose = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 3, 4, 1, 5, 2}}}), relu);
        auto pw = m1.add_instruction(migraphx::make_op("add"), x, transpose);
        m1.add_instruction(pass_op{}, pw);
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", s2);
        auto one  = m2.add_literal(migraphx::generate_literal(s1));
        auto relu = m2.add_instruction(migraphx::make_op("relu"), one);
        auto reshape_ins =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 2, 5, 5}}}), relu);
        auto transpose = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 3, 4, 1, 5, 2}}}), reshape_ins);
        auto pw = m2.add_instruction(migraphx::make_op("add"), x, transpose);
        m2.add_instruction(pass_op{}, pw);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(reshape_unary_transpose_pointwise)
{
    auto s1 = migraphx::shape{migraphx::shape::float_type, {2, 8, 5, 5}};
    auto s2 = migraphx::shape{migraphx::shape::float_type, {2, 2, 5, 2, 5, 2}};
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", s1);
        auto y = m1.add_parameter("y", s2);
        auto reshape_ins =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 2, 5, 5}}}), x);
        auto relu      = m1.add_instruction(migraphx::make_op("relu"), reshape_ins);
        auto transpose = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 3, 4, 1, 5, 2}}}), relu);
        auto add = m1.add_instruction(migraphx::make_op("add"), transpose, y);
        m1.add_instruction(pass_op{}, add);
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", s1);
        auto y = m2.add_parameter("y", s2);
        auto reshape_ins =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 2, 5, 5}}}), x);
        auto transpose = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 3, 4, 1, 5, 2}}}), reshape_ins);
        auto relu = m2.add_instruction(migraphx::make_op("relu"), transpose);
        auto add  = m2.add_instruction(migraphx::make_op("add"), relu, y);
        m2.add_instruction(pass_op{}, add);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(pointwise_reshape_unary)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {2, 8, 5, 5}};
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", s);
        auto y   = m1.add_parameter("y", s);
        auto add = m1.add_instruction(migraphx::make_op("add"), x, y);
        auto reshape_ins =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 2, 5, 5}}}), add);
        auto relu = m1.add_instruction(migraphx::make_op("relu"), reshape_ins);
        m1.add_instruction(pass_op{}, relu);
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", s);
        auto y    = m2.add_parameter("y", s);
        auto add  = m2.add_instruction(migraphx::make_op("add"), x, y);
        auto relu = m2.add_instruction(migraphx::make_op("relu"), add);
        auto reshape_ins =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 2, 5, 5}}}), relu);
        m2.add_instruction(pass_op{}, reshape_ins);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(pointwise_reshape_layout_convolution)
{
    auto s1 = migraphx::shape{migraphx::shape::float_type, {2, 32, 10, 64, 64}};
    auto s2 = migraphx::shape{migraphx::shape::float_type, {640, 320, 1, 1}};
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", s1);
        auto y   = m1.add_parameter("y", s1);
        auto w   = m1.add_parameter("w", s2);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), x, y);
        auto reshape_ins =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 320, 64, 64}}}), mul);
        auto layout = m1.add_instruction(
            migraphx::make_op("layout", {{"permutation", {0, 2, 3, 1}}}), reshape_ins);
        auto conv = m1.add_instruction(migraphx::make_op("convolution"), layout, w);
        m1.add_instruction(pass_op{}, conv);
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(transpose_contiguous_reshape_binary_packed)
{
    migraphx::module m1;
    {
        auto x  = m1.add_parameter("x", {migraphx::shape::float_type, {2, 128, 28, 28}});
        auto w1 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {256, 128, 1, 1}}));
        auto conv1 = m1.add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {0, 0}}, {"stride", {1, 1}}, {"dilation", {1, 1}}}),
            x,
            w1); // (2, 256, 28, 28)
        auto w2 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {512, 256, 1, 1}}));
        auto conv2 = m1.add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {0, 0}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            conv1,
            w2); // (2, 512, 14, 14)

        auto conv2_rsp1 = m1.add_instruction(
            migraphx::make_op("reshape", {{"dims", {2, 2, 2, 128, 14, 14}}}), conv2);
        auto conv2_trans = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 3, 4, 1, 5, 2}}}), conv2_rsp1);
        auto conv2_cont = m1.add_instruction(migraphx::make_op("contiguous"), conv2_trans);
        auto conv2_rsp2 = m1.add_instruction(
            migraphx::make_op("reshape", {{"dims", {2, 128, 28, 28}}}), conv2_cont);
        auto add_ins = m1.add_instruction(migraphx::make_op("add"), conv2_rsp2, x);
        m1.add_instruction(pass_op{}, add_ins);
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x  = m2.add_parameter("x", {migraphx::shape::float_type, {2, 128, 28, 28}});
        auto w1 = m2.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {256, 128, 1, 1}}));
        auto conv1 = m2.add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {0, 0}}, {"stride", {1, 1}}, {"dilation", {1, 1}}}),
            x,
            w1); // (2, 256, 28, 28)
        auto w2 = m2.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {512, 256, 1, 1}}));
        auto conv2 = m2.add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {0, 0}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            conv1,
            w2); // (2, 512, 14, 14)

        auto conv2_rsp = m2.add_instruction(
            migraphx::make_op("reshape", {{"dims", {2, 2, 2, 128, 14, 14}}}), conv2);
        auto conv2_trans = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 3, 4, 1, 5, 2}}}), conv2_rsp);
        auto x_rsp =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 128, 14, 2, 14, 2}}}), x);
        auto add_ins = m2.add_instruction(migraphx::make_op("add"), conv2_trans, x_rsp);
        auto add_rsp =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 128, 28, 28}}}), add_ins);
        m2.add_instruction(pass_op{}, add_rsp);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(transpose_contiguous_reshape_binary_broadcast)
{
    migraphx::module m1;
    {
        migraphx::shape sx{migraphx::shape::float_type, {4}};
        migraphx::shape sy{migraphx::shape::float_type, {2, 6, 2, 2}};

        auto x       = m1.add_parameter("x", sx);
        auto y       = m1.add_parameter("y", sy);
        auto x_brcst = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {2, 4, 6}}}), x);
        auto y_trans =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), y);
        auto y_rsp =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 4, 6}}}), y_trans);
        auto r = m1.add_instruction(migraphx::make_op("add"), y_rsp, x_brcst);
        m1.add_return({r});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(transpose_unsqueeze_concat)
{
    migraphx::module m1;
    {
        auto l0 = m1.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 2, 1, 1}});
        auto lt0 =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), l0);
        auto l1 = m1.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {1, 2, 1, 1}});
        auto lt1 =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), l1);
        auto l2 = m1.add_parameter("2", migraphx::shape{migraphx::shape::float_type, {1, 2, 1, 1}});
        auto lt2 =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), l2);
        std::vector<migraphx::instruction_ref> args{lt0, lt1, lt2};
        std::vector<migraphx::instruction_ref> unsqueezed_args;
        int64_t axis = 3;

        std::transform(
            args.begin(),
            args.end(),
            std::back_inserter(unsqueezed_args),
            [&](migraphx::instruction_ref arg) {
                return m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {axis}}}), arg);
            });
        auto concat =
            m1.add_instruction(migraphx::make_op("concat", {{"axis", axis}}), unsqueezed_args);
        m1.add_return({concat});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto l0 = m2.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 2, 1, 1}});
        auto l1 = m2.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {1, 2, 1, 1}});
        auto l2 = m2.add_parameter("2", migraphx::shape{migraphx::shape::float_type, {1, 2, 1, 1}});
        std::vector<migraphx::instruction_ref> args{l0, l1, l2};
        std::vector<migraphx::instruction_ref> unsqueezed_args;
        int64_t axis = 1;

        std::transform(
            args.begin(),
            args.end(),
            std::back_inserter(unsqueezed_args),
            [&](migraphx::instruction_ref arg) {
                return m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {axis}}}), arg);
            });
        auto concat =
            m2.add_instruction(migraphx::make_op("concat", {{"axis", axis}}), unsqueezed_args);
        auto transpose = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 3, 4, 1, 2}}}), concat);
        m2.add_return({transpose});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(transpose_slice)
{
    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", {migraphx::shape::float_type, {1, 384, 36, 64}});
        auto slice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {12}}}), x);
        auto transpose1 = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), slice1);
        auto slice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {12}}, {"ends", {24}}}), x);
        auto transpose2 = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), slice2);
        auto slice3 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {24}}, {"ends", {36}}}), x);
        auto transpose3 = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), slice3);
        m1.add_return({transpose1, transpose2, transpose3});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", {migraphx::shape::float_type, {1, 384, 36, 64}});
        auto transpose =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), x);
        auto slice1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {12}}}),
            transpose);
        auto slice2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {12}}, {"ends", {24}}}),
            transpose);
        auto slice3 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {24}}, {"ends", {36}}}),
            transpose);
        m2.add_return({slice1, slice2, slice3});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(transpose_slice_unsqueeze)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {4, 1024, 96, 64}});
        auto transpose1 =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), x);
        auto slice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {8}}}),
            transpose1);
        auto slice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {16}}, {"ends", {24}}}),
            transpose1);
        auto slice3 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {32}}, {"ends", {40}}}),
            transpose1);
        m1.add_return({slice1, slice2, slice3});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", {migraphx::shape::float_type, {4, 1024, 96, 64}});
        auto unsq =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {4, 1024, 12, 8, 64}}}), x);
        auto transpose = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {2, 0, 3, 4, 1}}}), unsq);
        auto slice1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), transpose);
        auto sq1    = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), slice1);
        auto slice2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {2}}, {"ends", {3}}}), transpose);
        auto sq2    = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), slice2);
        auto slice3 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {4}}, {"ends", {5}}}), transpose);
        auto sq3 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), slice3);
        m2.add_return({sq1, sq2, sq3});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(transpose_slice_diff_perm)
{
    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", {migraphx::shape::float_type, {1, 384, 36, 64}});
        auto slice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {12}}}), x);
        auto transpose1 = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), slice1);
        auto slice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {12}}, {"ends", {24}}}), x);
        auto transpose2 = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), slice2);
        auto slice3 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {24}}, {"ends", {36}}}), x);
        auto transpose3 = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), slice3);
        m1.add_return({transpose1, transpose2, transpose3});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", {migraphx::shape::float_type, {1, 384, 36, 64}});
        auto transpose =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), x);
        auto slice1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {12}}}),
            transpose);
        auto slice2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {12}}, {"ends", {24}}}),
            transpose);
        auto transpose2 = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), slice2);
        auto slice3 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {24}}, {"ends", {36}}}),
            transpose);
        m2.add_return({slice1, transpose2, slice3});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(transpose_slice_single_transpose)
{
    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", {migraphx::shape::float_type, {1, 384, 36, 64}});
        auto slice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {12}}}), x);
        auto sqrt1  = m1.add_instruction(migraphx::make_op("sqrt"), slice1);
        auto slice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {12}}, {"ends", {24}}}), x);
        auto transpose = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), slice2);
        auto slice3 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {24}}, {"ends", {36}}}), x);
        auto sqrt3 = m1.add_instruction(migraphx::make_op("sqrt"), slice3);
        m1.add_return({sqrt1, transpose, sqrt3});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(transpose_slice_non_packed_axis)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {2, 384, 36, 64}});
        auto transpose =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), x);
        auto slice = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {12}}}),
            transpose);
        auto sqrt = m1.add_instruction(migraphx::make_op("sqrt"), slice);
        m1.add_return({sqrt});
    }
    auto output_shapes = m1.get_output_shapes();
    run_pass(m1);
    EXPECT(m1.get_output_shapes() == output_shapes);
    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", {migraphx::shape::float_type, {2, 384, 36, 64}});
        auto unsqueeze =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 384, 3, 12, 64}}}), x);
        auto transpose = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {2, 0, 3, 1, 4}}}), unsqueeze);
        auto slice = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), transpose);
        auto squeeze = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), slice);
        auto sqrt    = m2.add_instruction(migraphx::make_op("sqrt"), squeeze);
        m2.add_return({sqrt});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(transpose_slice_non_packed_multi_axis)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {2, 384, 36, 64}});
        auto transpose =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), x);
        auto slice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {12}}}),
            transpose);
        auto slice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {12}}, {"ends", {24}}}),
            transpose);
        auto transpose2 = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), slice2);
        auto slice3 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {24}}, {"ends", {36}}}),
            transpose);
        m1.add_return({slice1, transpose2, slice3});
    }
    auto output_shapes = m1.get_output_shapes();
    run_pass(m1);
    EXPECT(to_lens(m1.get_output_shapes()) == to_lens(output_shapes));
    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", {migraphx::shape::float_type, {2, 384, 36, 64}});
        auto unsqueeze =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 384, 3, 12, 64}}}), x);
        auto transpose = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {2, 0, 3, 1, 4}}}), unsqueeze);
        auto slice1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), transpose);
        auto squeeze1 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), slice1);
        auto slice2   = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), transpose);
        auto transpose2 = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 1, 2, 4, 3}}}), slice2);
        auto squeeze2 =
            m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), transpose2);
        auto slice3 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {2}}, {"ends", {3}}}), transpose);
        auto squeeze3 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), slice3);
        m2.add_return({squeeze1, squeeze2, squeeze3});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reshape_reshape_dot)
{
    migraphx::shape as{migraphx::shape::float_type, {2, 10, 32, 16}};
    migraphx::shape bs{migraphx::shape::float_type, {2, 10, 16, 32}};
    migraphx::module m1;
    {
        auto a     = m1.add_literal(migraphx::generate_literal(as));
        auto b     = m1.add_parameter("input", bs);
        auto a_rsp = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {20, 32, 16}}}), a);
        auto b_rsp = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {20, 16, 32}}}), b);

        auto dot = m1.add_instruction(migraphx::make_op("dot"), a_rsp, b_rsp);
        auto dot_rsp =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 10, 32, 32}}}), dot);
        m1.add_return({dot_rsp});
    };
    run_pass(m1);

    migraphx::module m2;
    {
        auto a   = m2.add_literal(migraphx::generate_literal(as));
        auto b   = m2.add_parameter("input", bs);
        auto dot = m2.add_instruction(migraphx::make_op("dot"), a, b);
        m2.add_return({dot});
    };

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reshape_reshape_dot_gemm_axis)
{
    migraphx::shape as{migraphx::shape::float_type, {2, 10, 512}};
    migraphx::shape bs{migraphx::shape::float_type, {2, 10, 512}};
    migraphx::module m1;
    {
        auto a     = m1.add_literal(migraphx::generate_literal(as));
        auto b     = m1.add_parameter("input", bs);
        auto a_rsp = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {20, 32, 16}}}), a);
        auto b_rsp = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {20, 16, 32}}}), b);

        auto dot = m1.add_instruction(migraphx::make_op("dot"), a_rsp, b_rsp);
        auto dot_rsp =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 10, 1024}}}), dot);
        m1.add_return({dot_rsp});
    };
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reshape_dot)
{
    migraphx::shape s_inp{migraphx::shape::float_type, {2, 8, 8, 32}};
    migraphx::shape s_w{migraphx::shape::float_type, {32, 32}};

    migraphx::module m1;
    {
        auto inp = m1.add_parameter("inp", s_inp);
        auto rsp = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 64, 32}}}), inp);
        auto w   = m1.add_literal(migraphx::generate_literal(s_w));
        auto w_bc =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 32, 32}}}), w);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), rsp, w_bc);
        m1.add_return({dot});
    };
    run_pass(m1);

    migraphx::module m2;
    {
        auto inp  = m2.add_parameter("inp", s_inp);
        auto w    = m2.add_literal(migraphx::generate_literal(s_w));
        auto w_bc = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 8, 32, 32}}}), w);
        auto dot = m2.add_instruction(migraphx::make_op("dot"), inp, w_bc);
        auto rsp = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 64, 32}}}), dot);
        m2.add_return({rsp});
    };

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reshape_dot_flipped)
{
    migraphx::shape s_inp{migraphx::shape::float_type, {2, 8, 8, 32}};
    migraphx::shape s_w{migraphx::shape::float_type, {16, 8}};

    migraphx::module m1;
    {
        auto inp = m1.add_parameter("inp", s_inp);
        auto rsp = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {16, 8, 32}}}), inp);
        auto w   = m1.add_literal(migraphx::generate_literal(s_w));
        auto w_bc =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {16, 16, 8}}}), w);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), w_bc, rsp);
        m1.add_return({dot});
    };
    run_pass(m1);

    migraphx::module m2;
    {
        auto inp  = m2.add_parameter("inp", s_inp);
        auto w    = m2.add_literal(migraphx::generate_literal(s_w));
        auto w_bc = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 8, 16, 8}}}), w);
        auto dot = m2.add_instruction(migraphx::make_op("dot"), w_bc, inp);
        auto rsp = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {16, 16, 32}}}), dot);
        m2.add_return({rsp});
    };

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reshape_dot_dot_axis)
{
    migraphx::shape s_inp{migraphx::shape::float_type, {2, 8, 8, 4}};
    migraphx::shape s_w{migraphx::shape::float_type, {32, 32}};

    migraphx::module m1;
    {
        auto inp = m1.add_parameter("inp", s_inp);
        auto rsp = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 8, 32}}}), inp);
        auto w   = m1.add_literal(migraphx::generate_literal(s_w));
        auto w_bc =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 32, 32}}}), w);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), rsp, w_bc);
        m1.add_return({dot});
    };

    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reshape_dot_flipped_dot_axis)
{
    migraphx::shape s_inp{migraphx::shape::float_type, {2, 8, 8, 32}};
    migraphx::shape s_w{migraphx::shape::float_type, {8, 64}};

    migraphx::module m1;
    {
        auto inp = m1.add_parameter("inp", s_inp);
        auto rsp = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 64, 32}}}), inp);
        auto w   = m1.add_literal(migraphx::generate_literal(s_w));
        auto w_bc =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 8, 64}}}), w);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), w_bc, rsp);
        m1.add_return({dot});
    };

    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reshape_dot_broadcast)
{
    migraphx::shape s_inp{migraphx::shape::float_type, {2, 8, 8, 32}};
    migraphx::shape s_w{migraphx::shape::float_type, {32}};

    migraphx::module m1;
    {
        auto inp  = m1.add_parameter("inp", s_inp);
        auto rsp  = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 64, 32}}}), inp);
        auto w    = m1.add_literal(migraphx::generate_literal(s_w));
        auto w_bc = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {2, 32, 32}}}), w);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), rsp, w_bc);
        m1.add_return({dot});
    };
    run_pass(m1);

    migraphx::module m2;
    {
        auto inp  = m2.add_parameter("inp", s_inp);
        auto w    = m2.add_literal(migraphx::generate_literal(s_w));
        auto w_bc = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 2}, {"out_lens", {2, 8, 32, 32}}}), w);
        auto dot = m2.add_instruction(migraphx::make_op("dot"), inp, w_bc);
        auto rsp = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 64, 32}}}), dot);
        m2.add_return({rsp});
    };

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reshape_dot_broadcast_2)
{
    migraphx::shape s_inp{migraphx::shape::float_type, {2, 8, 8, 32}};
    migraphx::shape s_w{migraphx::shape::float_type, {32}};

    migraphx::module m1;
    {
        auto inp  = m1.add_parameter("inp", s_inp);
        auto rsp  = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {128, 32}}}), inp);
        auto w    = m1.add_literal(migraphx::generate_literal(s_w));
        auto w_bc = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {32, 32}}}), w);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), rsp, w_bc);
        m1.add_return({dot});
    };
    run_pass(m1);

    migraphx::module m2;
    {
        auto inp  = m2.add_parameter("inp", s_inp);
        auto w    = m2.add_literal(migraphx::generate_literal(s_w));
        auto w_bc = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 3}, {"out_lens", {2, 8, 32, 32}}}), w);
        auto dot = m2.add_instruction(migraphx::make_op("dot"), inp, w_bc);
        auto rsp = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {128, 32}}}), dot);
        m2.add_return({rsp});
    };

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(mul_transpose)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 32, 64, 64}};
    migraphx::shape s2{migraphx::shape::float_type, {2, 64, 32, 32}};
    migraphx::module m1;
    {
        auto inp   = m1.add_parameter("input", s);
        auto c1    = m1.add_literal(migraphx::generate_literal(s));
        auto mul   = m1.add_instruction(migraphx::make_op("mul"), inp, c1);
        auto trans = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), mul);

        auto c3  = m1.add_literal(migraphx::generate_literal(s2));
        auto dot = m1.add_instruction(migraphx::make_op("dot"), trans, c3);
        m1.add_return({dot});
    };
    run_pass(m1);

    migraphx::module m2;
    {
        auto inp       = m2.add_parameter("input", s);
        auto inp_trans = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), inp);
        auto c1 = m2.add_literal(migraphx::generate_literal(s));
        auto c1_trans =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), c1);
        auto mul = m2.add_instruction(migraphx::make_op("mul"), inp_trans, c1_trans);
        auto c3  = m2.add_literal(migraphx::generate_literal(s2));
        auto dot = m2.add_instruction(migraphx::make_op("dot"), mul, c3);
        m2.add_return({dot});
    };

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(add_transpose)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 32, 64, 64}};
    migraphx::shape s2{migraphx::shape::float_type, {2, 64, 32, 32}};
    migraphx::module m1;
    {
        auto inp   = m1.add_parameter("input", s);
        auto c1    = m1.add_literal(migraphx::generate_literal(s));
        auto mul   = m1.add_instruction(migraphx::make_op("add"), inp, c1);
        auto trans = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), mul);

        auto c3  = m1.add_literal(migraphx::generate_literal(s2));
        auto dot = m1.add_instruction(migraphx::make_op("dot"), trans, c3);
        m1.add_return({dot});
    };
    run_pass(m1);

    migraphx::module m2;
    {
        auto inp       = m2.add_parameter("input", s);
        auto inp_trans = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), inp);
        auto c1 = m2.add_literal(migraphx::generate_literal(s));
        auto c1_trans =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), c1);
        auto mul = m2.add_instruction(migraphx::make_op("add"), inp_trans, c1_trans);
        auto c3  = m2.add_literal(migraphx::generate_literal(s2));
        auto dot = m2.add_instruction(migraphx::make_op("dot"), mul, c3);
        m2.add_return({dot});
    };

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(flatten)
{
    migraphx::shape s{migraphx::shape::float_type, {4608, 8, 2}};

    migraphx::module m1;
    {
        auto inp  = m1.add_parameter("input", s);
        auto flat = m1.add_instruction(migraphx::make_op("flatten", {{"axis", 1}}), inp);
        m1.add_return({flat});
    };
    run_pass(m1);

    migraphx::module m2;
    {
        auto inp  = m2.add_parameter("input", s);
        auto flat = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {4608, 16}}}), inp);
        m2.add_return({flat});
    };

    EXPECT(m1.sort() == m2.sort());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
