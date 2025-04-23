/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/register_op.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/op/convolution.hpp>
#include <migraphx/op/rnn_variable_seq_lens.hpp>
#include <migraphx/module.hpp>
#include <sstream>
#include <string>

#include <migraphx/serialize.hpp>

#include "test.hpp"

TEST_CASE(load_op)
{
    for(const auto& name : migraphx::get_operators())
    {
        auto op = migraphx::load_op(name);
        CHECK(op.name() == name);
    }
}

TEST_CASE(make_op)
{
    for(const auto& name : migraphx::get_operators())
    {
        auto op = migraphx::load_op(name);
        CHECK(op == migraphx::make_op(name));
    }
}

TEST_CASE(save_op)
{
    for(const auto& name : migraphx::get_operators())
    {
        auto op1 = migraphx::load_op(name);
        auto v   = migraphx::to_value(op1);
        auto op2 = migraphx::from_value<migraphx::operation>(v);
        CHECK(op1 == op2);
    }
}

TEST_CASE(make_op_from_value1)
{
    migraphx::operation x = migraphx::make_op(
        "convolution", {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {2, 2}}});
    migraphx::operation y = migraphx::make_op(
        "convolution", {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {2, 2}}});
    EXPECT(x == y);
}

TEST_CASE(make_op_from_value2)
{
    migraphx::operation x = migraphx::make_op("convolution", {{"padding", {1, 1}}});
    migraphx::operation y = migraphx::make_op("convolution", {{"padding", {1, 1}}});
    EXPECT(x == y);
}

TEST_CASE(make_rnn_op_from_value)
{
    migraphx::op::rnn_direction dirct = migraphx::op::rnn_direction::reverse;
    migraphx::operation x             = migraphx::make_op(
        "rnn_var_sl_shift_output", {{"output_name", "hidden_states"}, {"direction", dirct}});
    migraphx::operation y = migraphx::make_op(
        "rnn_var_sl_shift_output",
        {{"output_name", "hidden_states"}, {"direction", migraphx::to_value(dirct)}});
    EXPECT(x == y);
}

TEST_CASE(make_op_invalid_key)
{
    EXPECT(test::throws([] { migraphx::make_op("convolution", {{"paddings", {1, 1}}}); }));
}

TEST_CASE(load_offset)
{
    migraphx::shape s{migraphx::shape::float_type, {4}};
    migraphx::shape bs{migraphx::shape::int8_type, {32}};
    auto op = migraphx::make_op("load", {{"offset", 4}, {"shape", migraphx::to_value(s)}});
    EXPECT(op.compute_shape({bs}) == s);

    migraphx::argument a{bs};
    EXPECT(op.compute(bs, {a}).data() == a.data() + 4);
}

TEST_CASE(load_out_of_bounds)
{
    migraphx::shape s{migraphx::shape::float_type, {4}};
    migraphx::shape bs{migraphx::shape::int8_type, {16}};
    auto op = migraphx::make_op("load", {{"offset", 4}, {"shape", migraphx::to_value(s)}});

    migraphx::argument a{bs};
    EXPECT(test::throws([&] { op.compute(bs, {a}); }));
}

TEST_CASE(load_tuple)
{
    migraphx::shape s{{migraphx::shape{migraphx::shape::int8_type, {3}},
                       migraphx::shape{migraphx::shape::float_type, {4}}}};
    migraphx::shape bs{migraphx::shape::int8_type, {32}};
    auto op = migraphx::make_op("load", {{"offset", 4}, {"shape", migraphx::to_value(s)}});
    EXPECT(op.compute_shape({bs}) == s);

    migraphx::argument a{bs};
    auto r = op.compute(bs, {a});
    EXPECT(r.get_sub_objects().size() == 2);
    auto* start = a.data() + 4;
    EXPECT(r.get_sub_objects()[0].data() == start + 16);
    EXPECT(r.get_sub_objects()[1].data() == start);
}

TEST_CASE(ops)
{
    auto names = migraphx::get_operators();
    EXPECT(names.size() > 1);
}

TEST_CASE(rnn)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 1}};
    std::vector<float> data1(2, 2.0f);
    std::vector<float> data2(2, 3.0f);
    migraphx::argument a1(s, data1.data());
    migraphx::argument a2(s, data2.data());

    auto op = migraphx::make_op("rnn");

    EXPECT(test::throws([&] { op.compute(s, {a1, a2}); }));
}

TEST_CASE(if_op)
{
    migraphx::shape s{migraphx::shape::bool_type, {1}};
    std::vector<char> data = {1};
    migraphx::argument cond(s, data.data());
    migraphx::shape sd{migraphx::shape::float_type, {2, 1}};
    std::vector<float> data1(2, 2.0f);
    std::vector<float> data2(2, 3.0f);
    migraphx::argument a1(sd, data1.data());
    migraphx::argument a2(sd, data2.data());

    migraphx::module m("name");
    auto l = m.add_literal(migraphx::literal(sd, data1));
    m.add_return({l});

    auto op = migraphx::make_op("add");
    EXPECT(test::throws([&] { op.compute(s, {cond, a1, a2}, {&m, &m}, {}); }));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
