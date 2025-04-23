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
#include <migraphx/propagate_constant.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>
#include <basic_ops.hpp>

#include <test.hpp>

static void run_pass(migraphx::module& m, const std::unordered_set<std::string>& skip_ops = {})
{
    migraphx::run_passes(
        m, {migraphx::propagate_constant{skip_ops}, migraphx::dead_code_elimination{}});
}

TEST_CASE(const_add)
{
    migraphx::module m1;
    auto one = m1.add_literal(1);
    auto two = m1.add_literal(2);
    auto sum = m1.add_instruction(migraphx::make_op("add"), one, two);
    m1.add_instruction(non_const_pass_op{}, sum);
    run_pass(m1);

    migraphx::module m2;
    auto total = m2.add_literal(3);
    m2.add_instruction(non_const_pass_op{}, total);
    EXPECT(m1 == m2);
}

TEST_CASE(const_add_parameter)
{
    migraphx::module m1;
    auto one = m1.add_parameter("one", {migraphx::shape::int32_type, {1}});
    auto two = m1.add_literal(2);
    auto sum = m1.add_instruction(migraphx::make_op("add"), one, two);
    m1.add_instruction(non_const_pass_op{}, sum);
    run_pass(m1);

    migraphx::module m2;
    auto total = m2.add_literal(3);
    m2.add_instruction(non_const_pass_op{}, total);
    EXPECT(m1 != m2);
}

TEST_CASE(const_multiadd)
{
    migraphx::module m1;
    auto one  = m1.add_literal(1);
    auto two  = m1.add_literal(2);
    auto sum1 = m1.add_instruction(migraphx::make_op("add"), one, two);
    auto sum2 = m1.add_instruction(migraphx::make_op("add"), sum1, two);
    m1.add_instruction(non_const_pass_op{}, sum2);
    run_pass(m1);

    migraphx::module m2;
    auto total = m2.add_literal(5);
    m2.add_instruction(non_const_pass_op{}, total);
    EXPECT(m1 == m2);
}

TEST_CASE(const_add_mul)
{
    migraphx::module m1;
    auto one  = m1.add_literal(1);
    auto two  = m1.add_literal(2);
    auto mul  = m1.add_instruction(migraphx::make_op("mul"), two, two);
    auto sum1 = m1.add_instruction(migraphx::make_op("add"), one, mul);
    auto sum2 = m1.add_instruction(migraphx::make_op("add"), sum1, two);
    m1.add_instruction(non_const_pass_op{}, sum2);
    run_pass(m1);

    migraphx::module m2;
    auto total = m2.add_literal(7);
    m2.add_instruction(non_const_pass_op{}, total);
    EXPECT(m1 == m2);
}

TEST_CASE(const_add_scalar)
{
    migraphx::module m1;
    auto one = m1.add_instruction(migraphx::make_op("scalar", {{"scalar_bcst_dims", {2, 2}}}),
                                  m1.add_literal(1));
    auto two = m1.add_instruction(migraphx::make_op("scalar", {{"scalar_bcst_dims", {2, 2}}}),
                                  m1.add_literal(2));
    auto sum = m1.add_instruction(migraphx::make_op("add"), one, two);
    m1.add_instruction(non_const_pass_op{}, sum);
    run_pass(m1);

    migraphx::module m2;
    auto total =
        m2.add_literal(migraphx::literal{{migraphx::shape::int32_type, {2, 2}}, {3, 3, 3, 3}});
    m2.add_instruction(non_const_pass_op{}, total);
    EXPECT(m1 == m2);
}

TEST_CASE(const_scalar)
{
    migraphx::module m1;
    {
        auto one = m1.add_instruction(migraphx::make_op("scalar", {{"scalar_bcst_dims", {2, 2}}}),
                                      m1.add_literal(1));
        m1.add_instruction(non_const_pass_op{}, one);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto one = m2.add_instruction(migraphx::make_op("scalar", {{"scalar_bcst_dims", {2, 2}}}),
                                      m2.add_literal(1));
        m2.add_instruction(non_const_pass_op{}, one);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(const_dot)
{
    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {2, 2}};
        std::vector<float> vec = {1.0f, 2.0f, 1.0f, 2.0f};

        auto l  = m1.add_literal(migraphx::literal(s, vec));
        auto dl = m1.add_instruction(migraphx::make_op("dot"), l, l);
        auto x  = m1.add_parameter("x", s);
        auto r  = m1.add_instruction(migraphx::make_op("add"), dl, x);
        m1.add_return({r});
    }

    run_pass(m1);

    migraphx::module m2;
    {
        migraphx::shape s{migraphx::shape::float_type, {2, 2}};
        std::vector<float> vec = {3.0f, 6.0f, 3.0f, 6.0f};

        auto x = m2.add_parameter("x", s);
        auto l = m2.add_literal(migraphx::literal(s, vec));
        auto r = m2.add_instruction(migraphx::make_op("add"), l, x);
        m2.add_return({r});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(last_const)
{
    const std::vector<float> vec = {1.0f, 2.0f, 1.0f, 2.0f};
    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {2, 2}};
        auto l = m1.add_literal(migraphx::literal(s, vec));
        m1.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), l);
    }

    run_pass(m1);

    migraphx::module m2;
    {
        migraphx::shape s{migraphx::shape::half_type, {2, 2}};
        auto l = m2.add_literal(migraphx::literal(s, vec));
        m2.add_instruction(migraphx::make_op("identity"), l);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(skip_broadcast)
{
    migraphx::module m1;
    {
        auto one = m1.add_literal(1);
        auto oneb =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 2}}}), one);
        m1.add_return({oneb});
    }

    migraphx::module m2 = m1;
    run_pass(m1);

    EXPECT(m1 == m2);
}

TEST_CASE(skip_broadcast_transpose)
{
    const std::vector<float> vec = {1.0f, 2.0f};
    migraphx::shape s{migraphx::shape::float_type, {1, 2}};
    migraphx::module m1;
    {
        auto one = m1.add_literal(migraphx::literal(s, vec));
        auto oneb =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 2}}}), one);
        auto transpose =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), oneb);
        m1.add_return({transpose});
    }

    migraphx::module m2 = m1;
    run_pass(m1);

    EXPECT(m1 == m2);
}

TEST_CASE(fold_broadcast)
{
    const std::vector<float> vec = {1.0f, 2.0f, 1.0f, 2.0f};
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    migraphx::module m1;
    {
        auto one = m1.add_literal(1.0f);
        auto oneb =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 2}}}), one);
        auto l   = m1.add_literal(migraphx::literal(s, vec));
        auto mul = m1.add_instruction(migraphx::make_op("mul"), oneb, l);
        m1.add_return({mul});
    }

    run_pass(m1);

    migraphx::module m2;
    {
        auto l = m2.add_literal(migraphx::literal(s, vec));
        m2.add_return({l});
    }

    EXPECT(m1 == m2);
}

TEST_CASE(fold_broadcast_non_overlapping_broadcast)
{
    const std::vector<float> vec = {1.0f, 2.0f};
    migraphx::shape s1{migraphx::shape::float_type, {1, 2}};
    migraphx::shape s2{migraphx::shape::float_type, {2, 1}};
    migraphx::module m1;
    {
        auto l1 = m1.add_literal(migraphx::literal(s1, vec));
        auto l1b =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 2}}}), l1);
        auto l2 = m1.add_literal(migraphx::literal(s2, vec));
        auto l2b =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 2}}}), l2);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), l1b, l2b);
        m1.add_return({mul});
    }

    run_pass(m1);

    migraphx::module m2;
    {
        migraphx::shape s{migraphx::shape::float_type, {2, 2}};
        auto l = m2.add_literal(migraphx::literal(s, {1.0f, 2.0f, 2.0f, 4.0f}));
        m2.add_return({l});
    }

    EXPECT(m1 == m2);
}

TEST_CASE(fold_packed_broadcast)
{
    const std::vector<float> vec = {0.0f, 1.0f};
    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {2}};
        auto l = m1.add_literal(migraphx::literal(s, vec));
        auto lb =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {1, 2}}}), l);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), lb, lb);
        m1.add_return({mul});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        migraphx::shape s{migraphx::shape::float_type, {1, 2}, {0, 1}};
        auto l = m2.add_literal(migraphx::literal(s, vec));
        m2.add_return({l});
    }

    EXPECT(m1 == m2);
}

TEST_CASE(fold_slice)
{
    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {2, 2}};
        const std::vector<float> vec = {1.0f, 2.0f, 1.0f, 2.0f};
        auto l                       = m1.add_literal(migraphx::literal(s, vec));
        auto slice                   = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), l);
        m1.add_return({slice});
    }

    run_pass(m1);

    migraphx::module m2;
    {
        migraphx::shape s{migraphx::shape::float_type, {1, 2}};
        const std::vector<float> vec = {1.0f, 2.0f};
        auto l                       = m2.add_literal(migraphx::literal(s, vec));
        m2.add_return({l});
    }

    EXPECT(m1 == m2);
}

TEST_CASE(fold_slice_nonpack)
{
    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {2, 2}};
        const std::vector<float> vec = {1.0f, 1.0f, 2.0f, 2.0f};
        auto l                       = m1.add_literal(migraphx::literal(s, vec));
        auto slice                   = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), l);
        m1.add_return({slice});
    }

    run_pass(m1);

    migraphx::module m2;
    {
        migraphx::shape s{migraphx::shape::float_type, {2, 1}};
        const std::vector<float> vec = {1.0f, 2.0f};
        auto l                       = m2.add_literal(migraphx::literal(s, vec));
        m2.add_return({l});
    }

    EXPECT(m1 == m2);
}

TEST_CASE(pack_unpack_int4)
{
    migraphx::shape s1{migraphx::shape::int8_type, {4}};
    migraphx::shape s2{migraphx::shape::int8_type, {2}};
    migraphx::module m1;
    {
        const std::vector<int8_t> vec = {1, 0, 2, 0};
        auto l                        = m1.add_literal(migraphx::literal(s1, vec));
        auto pack                     = m1.add_instruction(migraphx::make_op("pack_int4"), l);
        auto unpack                   = m1.add_instruction(migraphx::make_op("unpack_int4"), pack);
        m1.add_return({unpack});
    }

    run_pass(m1);

    migraphx::module m2;
    {
        const std::vector<int8_t> vec = {1, 2};
        auto l                        = m2.add_literal(migraphx::literal(s2, vec));
        auto unpack                   = m2.add_instruction(migraphx::make_op("unpack_int4"), l);
        m2.add_return({unpack});
    }

    EXPECT(m1 == m2);
}

TEST_CASE(skip_ops)
{
    const std::vector<float> vec = {1.0f, 2.0f, 1.0f, 2.0f};
    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {2, 2}};
        auto l     = m1.add_literal(migraphx::literal(s, vec));
        auto scale = m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 2}}}),
                                        m1.add_literal(0.5f));
        auto zp    = m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 2}}}),
                                     m1.add_literal(0));
        auto q     = m1.add_instruction(migraphx::make_op("quantizelinear"), l, scale, zp);
        m1.add_instruction(migraphx::make_op("dequantizelinear"), q, scale, zp);
    }

    migraphx::module m2 = m1;

    run_pass(m1, {"quantizelinear", "dequantizelinear"});
    EXPECT(m1 == m2);
}

TEST_CASE(block_dequantize)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", migraphx::shape{migraphx::shape::int8_type, {2, 5, 2}});
        auto scalelit =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {2, 2, 2}}));
        auto zplit =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::int8_type, {2, 2, 2}}));

        auto unsqueeze1 =
            m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), scalelit);
        auto broadcast1 = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 3, 2}}}), unsqueeze1);
        auto reshape1 =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 6, 2}}}), broadcast1);
        auto scale = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {5}}}), reshape1);

        auto unsqueeze2 =
            m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), zplit);
        auto broadcast2 = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 3, 2}}}), unsqueeze2);
        auto reshape2 =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 6, 2}}}), broadcast2);
        auto zp = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {5}}}), reshape2);

        auto dq = m1.add_instruction(migraphx::make_op("dequantizelinear"), x, scale, zp);
        m1.add_return({dq});
    }

    run_pass(m1);
    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", migraphx::shape{migraphx::shape::int8_type, {2, 5, 2}});
        auto scalelit =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {2, 2, 1, 2}}));
        auto zplit =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::int8_type, {2, 2, 1, 2}}));

        auto broadcast1 = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 3, 2}}}), scalelit);
        auto reshape1 =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 6, 2}}}), broadcast1);
        auto scale = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {5}}}), reshape1);

        auto broadcast2 = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 3, 2}}}), zplit);
        auto reshape2 =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 6, 2}}}), broadcast2);
        auto zp = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {5}}}), reshape2);

        auto dq = m2.add_instruction(migraphx::make_op("dequantizelinear"), x, scale, zp);
        m2.add_return({dq});
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(block_dequantize_int4)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 5, 2}});
        auto w =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::int8_type, {2, 5, 1}}));
        auto wunpack = m1.add_instruction(migraphx::make_op("unpack_int4"), w);
        auto scalelit =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {2, 2, 2}}));
        auto zplit =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::int8_type, {2, 2, 2}}));

        auto unsqueeze1 =
            m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), scalelit);
        auto broadcast1 = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 3, 2}}}), unsqueeze1);
        auto reshape1 =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 6, 2}}}), broadcast1);
        auto scale = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {5}}}), reshape1);

        auto unsqueeze2 =
            m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), zplit);
        auto broadcast2 = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 3, 2}}}), unsqueeze2);
        auto reshape2 =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 6, 2}}}), broadcast2);
        auto zp = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {5}}}), reshape2);

        auto dq = m1.add_instruction(migraphx::make_op("dequantizelinear"), wunpack, scale, zp);
        auto transpose =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), dq);

        auto dot = m1.add_instruction(migraphx::make_op("dot"), x, transpose);
        m1.add_return({dot});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 5, 2}});
        auto w =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::int8_type, {2, 5, 1}}));
        auto wunpack = m2.add_instruction(migraphx::make_op("unpack_int4"), w);
        auto scalelit =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {2, 2, 1, 2}}));
        auto zplit =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::int8_type, {2, 2, 1, 2}}));

        auto broadcast1 = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 3, 2}}}), scalelit);
        auto reshape1 =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 6, 2}}}), broadcast1);
        auto scale = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {5}}}), reshape1);

        auto broadcast2 = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 3, 2}}}), zplit);
        auto reshape2 =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 6, 2}}}), broadcast2);
        auto zp = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {5}}}), reshape2);

        auto dq = m2.add_instruction(migraphx::make_op("dequantizelinear"), wunpack, scale, zp);
        auto transpose =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), dq);

        auto dot = m2.add_instruction(migraphx::make_op("dot"), x, transpose);
        m2.add_return({dot});
    }
    EXPECT(m1.sort() == m2.sort());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
