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

#include <migraphx/operation.hpp>
#include <migraphx/context.hpp>
#include <sstream>
#include <string>
#include "test.hpp"

struct simple_operation
{
    template <class T, class F>
    static auto reflect(T& x, F f)
    {
        return migraphx::pack(f(x.data, "data"));
    }
    int data = 1;
    std::string name() const { return "simple"; }
    migraphx::shape compute_shape(const std::vector<migraphx::shape>&) const
    {
        MIGRAPHX_THROW("not computable");
    }
    migraphx::argument compute(migraphx::context&,
                               const migraphx::shape&,
                               const std::vector<migraphx::argument>&) const
    {
        MIGRAPHX_THROW("not computable");
    }
    friend std::ostream& operator<<(std::ostream& os, const simple_operation& op)
    {
        os << op.name() << "[" << op.data << "]";
        return os;
    }
};

struct simple_operation_no_print
{
    std::string name() const { return "simple"; }
    migraphx::shape compute_shape(const std::vector<migraphx::shape>&) const
    {
        MIGRAPHX_THROW("not computable");
    }
    migraphx::argument compute(migraphx::context&,
                               const migraphx::shape&,
                               const std::vector<migraphx::argument>&) const
    {
        MIGRAPHX_THROW("not computable");
    }
};

struct compilable_op
{
    std::string name() const { return "compilable"; }
    migraphx::argument
    compute(migraphx::context&, const migraphx::shape&, std::vector<migraphx::argument> args) const
    {
        if(args.empty())
            return {};
        return args.front();
    }

    migraphx::shape compute_shape(std::vector<migraphx::shape> inputs) const
    {
        if(inputs.empty())
            return {};
        return inputs.front();
    }

    int output_alias(const std::vector<migraphx::shape>&) const { return 0; }

    migraphx::value
    compile(migraphx::context&, const migraphx::shape&, const std::vector<migraphx::shape>&)
    {
        return {{"compiled", true}};
    }
};

TEST_CASE(operation_copy_test)
{
    simple_operation s{};
    migraphx::operation op1 = s;   // NOLINT
    migraphx::operation op2 = op1; // NOLINT
    // cppcheck-suppress duplicateExpression
    EXPECT(s == op1);
    // cppcheck-suppress duplicateExpression
    EXPECT(op2 == op1);
}

TEST_CASE(operation_copy_assign_test)
{
    simple_operation s{};
    migraphx::operation op;
    op = s;
    // cppcheck-suppress duplicateExpression
    EXPECT(s == op);
}

TEST_CASE(operation_equal_test)
{
    simple_operation s{};
    migraphx::operation op1 = s;
    s.data                  = 2;
    migraphx::operation op2 = op1; // NOLINT
    migraphx::operation op3 = s;   // NOLINT

    EXPECT(s != op1);
    EXPECT(op2 == op1);
    EXPECT(op3 != op2);
    EXPECT(op3 != op1);
}

struct not_operation
{
};

TEST_CASE(operation_any_cast)
{
    migraphx::operation op1 = simple_operation{};
    EXPECT(migraphx::any_cast<simple_operation>(op1).data == 1);
    EXPECT(migraphx::any_cast<not_operation*>(&op1) == nullptr);
    EXPECT(test::throws([&] { migraphx::any_cast<not_operation&>(op1); }));
    migraphx::operation op2 = simple_operation{2};
    EXPECT(migraphx::any_cast<simple_operation>(op2).data == 2);
    EXPECT(migraphx::any_cast<not_operation*>(&op2) == nullptr);
}

TEST_CASE(operation_print)
{
    migraphx::operation op = simple_operation{};
    std::stringstream ss;
    ss << op;
    std::string s = ss.str();
    EXPECT(s == "simple[1]");
}

TEST_CASE(operation_default_print)
{
    migraphx::operation op = simple_operation_no_print{};
    std::stringstream ss;
    ss << op;
    std::string s = ss.str();
    EXPECT(s == "simple");
}

struct final_operation
{
    std::string name() const { return "final"; }
    migraphx::shape compute_shape(const std::vector<migraphx::shape>&) const
    {
        MIGRAPHX_THROW("not computable");
    }
    void
    finalize(migraphx::context&, const migraphx::shape&, const std::vector<migraphx::shape>&) const
    {
    }
};

struct final_operation_throw
{
    std::string name() const { return "final"; }
    migraphx::shape compute_shape(const std::vector<migraphx::shape>&) const
    {
        MIGRAPHX_THROW("not computable");
    }
    [[gnu::noreturn]] void
    finalize(migraphx::context&, const migraphx::shape&, const std::vector<migraphx::shape>&) const
    {
        MIGRAPHX_THROW("finalize");
    }
};

TEST_CASE(check_has_finalize_simple)
{
    migraphx::operation op = simple_operation{};
    EXPECT(not migraphx::has_finalize(op));
}

TEST_CASE(check_has_finalize)
{
    migraphx::operation op = final_operation{};
    EXPECT(migraphx::has_finalize(op));
}

TEST_CASE(check_run_finalize)
{
    migraphx::operation op = final_operation{};
    migraphx::context ctx{};
    op.finalize(ctx, {}, {});
}

TEST_CASE(check_run_finalize_simple)
{
    migraphx::operation op = simple_operation{};
    migraphx::context ctx{};
    op.finalize(ctx, {}, {});
}

TEST_CASE(check_run_finalize_throw)
{
    migraphx::operation op = final_operation_throw{};
    migraphx::context ctx{};
    EXPECT(test::throws([&] { op.finalize(ctx, {}, {}); }));
}

TEST_CASE(check_to_value1)
{
    migraphx::operation op = simple_operation{};
    auto v                 = op.to_value();
    EXPECT(v == migraphx::value{{"data", 1}});
}

TEST_CASE(check_to_value2)
{
    migraphx::operation op = simple_operation{};
    auto v                 = migraphx::to_value(op);
    EXPECT(v == migraphx::value{{"name", "simple"}, {"operator", {{"data", 1}}}});
}

TEST_CASE(check_from_value1)
{
    migraphx::operation op1 = simple_operation{};
    migraphx::operation op2 = simple_operation{3};

    op1.from_value({{"data", 3}});
    EXPECT(op1 == op2);
}

TEST_CASE(check_from_value2)
{
    migraphx::operation op1 = migraphx::from_value<simple_operation>({{"data", 3}});
    migraphx::operation op2 = simple_operation{3};

    EXPECT(op1 == op2);
}

TEST_CASE(compile)
{
    migraphx::operation op = compilable_op{};
    migraphx::context ctx{};
    auto v = op.compile(ctx, {}, {});
    EXPECT(v.at("compiled").to<bool>() == true);
}

TEST_CASE(compile_non_compilable)
{
    migraphx::operation op = simple_operation{};
    migraphx::context ctx{};
    auto v = op.compile(ctx, {}, {});
    EXPECT(v.empty());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
