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
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <sstream>
#include "test.hpp"
#include <basic_ops.hpp>

struct sum_cf_op
{
    std::string name() const { return "sum_cf"; }
    migraphx::argument compute(const migraphx::shape&, std::vector<migraphx::argument> args) const
    {
        migraphx::argument result;
        if(args.size() != 2)
            MIGRAPHX_THROW("Wrong args");
        if(args[0].get_shape() != args[1].get_shape())
            MIGRAPHX_THROW("Wrong args");
        if(args[0].get_shape().lens().size() != 1)
            MIGRAPHX_THROW("Wrong args");
        if(args[0].get_shape().lens().front() != 1)
            MIGRAPHX_THROW("Wrong args");

        args[0].visit_at([&](auto x) {
            args[1].visit_at([&](auto y) { result = migraphx::literal{x + y}.get_argument(); });
        });
        return result;
    }

    migraphx::shape compute_shape(std::vector<migraphx::shape> inputs) const
    {
        if(inputs.size() != 2)
            MIGRAPHX_THROW("Wrong inputs");
        return inputs.front();
    }
};

struct non_computable_cf
{
    std::string name() const { return "non_computable"; }
    migraphx::shape compute_shape(std::vector<migraphx::shape> inputs) const
    {
        if(inputs.empty())
            return {};
        return inputs.front();
    }
};

struct test_context
{
    void finish() const {}
};

TEST_CASE(literal_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto lit = mm->add_literal(1);
    CHECK(lit->eval() == migraphx::literal{1});
}

TEST_CASE(param_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto lit = mm->add_parameter("param", migraphx::shape{migraphx::shape::float_type, {1}});
    CHECK(lit->eval().empty());
}

TEST_CASE(op_test1)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    auto sum = mm->add_instruction(sum_cf_op{}, one, two);
    CHECK(sum->eval() == migraphx::literal{3});
}

TEST_CASE(op_test2)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("param", migraphx::shape{migraphx::shape::float_type, {1}});
    auto two = mm->add_literal(2);
    auto sum = mm->add_instruction(sum_cf_op{}, x, two);
    CHECK(sum->eval().empty());
}

TEST_CASE(op_test3)
{
    migraphx::program p;

    auto* mm  = p.get_main_module();
    auto one  = mm->add_literal(1);
    auto two  = mm->add_literal(2);
    auto sum1 = mm->add_instruction(sum_op{}, one, two);
    auto sum2 = mm->add_instruction(sum_cf_op{}, sum1, two);
    CHECK(sum2->eval().empty());
}

TEST_CASE(compute_op_c)
{
    migraphx::operation op = sum_op{};
    auto one               = migraphx::literal{1}.get_argument();
    auto two               = migraphx::literal{2}.get_argument();
    EXPECT(test::throws([&] {
        op.compute(migraphx::shape{migraphx::shape::float_type, {1}}, {one, two});
    }));
}

TEST_CASE(compute_nop_c)
{
    migraphx::operation op = non_computable_cf{};
    auto one               = migraphx::literal{1}.get_argument();
    auto two               = migraphx::literal{2}.get_argument();
    EXPECT(test::throws([&] {
        op.compute(migraphx::shape{migraphx::shape::float_type, {1}}, {one, two});
    }));
}

TEST_CASE(compute_nop_context)
{
    migraphx::operation op = non_computable_cf{};
    auto one               = migraphx::literal{1}.get_argument();
    auto two               = migraphx::literal{2}.get_argument();
    migraphx::context ctx  = test_context{};
    EXPECT(test::throws([&] {
        op.compute(ctx, migraphx::shape{migraphx::shape::float_type, {1}}, {one, two});
    }));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
