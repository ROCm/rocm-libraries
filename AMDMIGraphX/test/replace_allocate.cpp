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
#include <migraphx/allocation_model.hpp>
#include <migraphx/replace_allocate.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/register_op.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

struct test_copy : migraphx::auto_register_op<test_copy>
{
    std::string name() const { return "test_copy"; }
    migraphx::shape compute_shape(const std::vector<migraphx::shape>& inputs) const
    {
        migraphx::check_shapes{inputs, *this}.has(2);
        return inputs.back();
    }
    migraphx::argument compute(migraphx::context&,
                               const migraphx::shape&,
                               const std::vector<migraphx::argument>& inputs) const
    {
        inputs.at(0).visit([&](auto x) {
            inputs.at(1).visit([&](auto y) { std::copy(x.begin(), x.end(), y.begin()); });
        });
        return inputs.back();
    }

    std::ptrdiff_t output_alias(const std::vector<migraphx::shape>&) const { return 1; }
};

struct allocate_no_out : migraphx::auto_register_op<allocate_no_out>
{
    migraphx::shape s{};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::pack(f(self.s, "shape"));
    }

    std::string name() const { return "allocate_no_out"; }
    migraphx::shape compute_shape(const std::vector<migraphx::shape>& inputs) const
    {
        migraphx::check_shapes{inputs, *this}.has(0);
        return s;
    }
    migraphx::argument compute(migraphx::context&,
                               const migraphx::shape& output_shape,
                               const std::vector<migraphx::argument>&) const
    {
        return migraphx::argument{output_shape};
    }
};

struct allocate_with_out : migraphx::auto_register_op<allocate_with_out>
{
    migraphx::shape s{};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::pack(f(self.s, "shape"));
    }

    std::string name() const { return "allocate_with_out"; }
    migraphx::shape compute_shape(const std::vector<migraphx::shape>& inputs) const
    {
        migraphx::check_shapes{inputs, *this}.has(0);
        return s;
    }
    migraphx::argument compute(migraphx::context&,
                               const migraphx::shape& output_shape,
                               const std::vector<migraphx::argument>&) const
    {
        return migraphx::argument{output_shape};
    }
};

// allocation model that has no out params
struct allocation_no_out_model
{
    std::string name() const { return "allocate_no_out"; }
    migraphx::operation allocate(const migraphx::shape& s) const
    {
        return migraphx::make_op(name(), {{"shape", to_value(s)}});
    }
    migraphx::operation preallocate(const migraphx::shape&, const std::string&) const { return {}; }
    std::string copy() const { return "test_copy"; }
    bool needs_out_params() const { return false; }
};

// allocation model with out params
struct allocation_with_out_model
{
    std::string name() const { return "allocate_with_out"; }
    migraphx::operation allocate(const migraphx::shape& s) const
    {
        return migraphx::make_op(name(), {{"shape", to_value(s)}});
    }
    migraphx::operation preallocate(const migraphx::shape&, const std::string&) const { return {}; }
    std::string copy() const { return "test_copy"; }
    bool needs_out_params() const { return true; }
};

static void
run_pass(migraphx::module& m, migraphx::allocation_model model, bool offload_copy = false)
{
    migraphx::run_passes(m,
                         {migraphx::replace_allocate{std::move(model), offload_copy},
                          migraphx::dead_code_elimination{}});
}

static void
run_pass(migraphx::program& p, migraphx::allocation_model model, bool offload_copy = false)
{
    migraphx::run_passes(p,
                         {migraphx::replace_allocate{std::move(model), offload_copy},
                          migraphx::dead_code_elimination{}});
}

static migraphx::module create_simple_program()
{
    migraphx::module m;
    migraphx::shape s{migraphx::shape::float_type, {5}};
    auto x = m.add_parameter("x", s);
    auto y = m.add_parameter("y", s);
    auto alloc =
        m.add_instruction(migraphx::make_op("allocate", {{"shape", migraphx::to_value(s)}}));
    m.add_instruction(pass_op{}, alloc, x, y);
    return m;
}

TEST_CASE(allocate_no_out)
{
    migraphx::module m = create_simple_program();
    run_pass(m, allocation_no_out_model{});

    EXPECT(std::any_of(m.begin(), m.end(), [](const migraphx::instruction& ins) {
        return migraphx::contains(ins.name(), "allocate_no_out");
    }));
}

TEST_CASE(allocate_with_out_param)
{
    migraphx::module m = create_simple_program();
    run_pass(m, allocation_with_out_model{});

    EXPECT(std::none_of(m.begin(), m.end(), [](const migraphx::instruction& ins) {
        return migraphx::contains(ins.name(), "allocate");
    }));
}

TEST_CASE(allocate_with_out_return)
{
    migraphx::module m = create_simple_program();
    m.add_return({std::prev(m.end())});
    run_pass(m, allocation_with_out_model{});

    EXPECT(std::none_of(m.begin(), m.end(), [](const migraphx::instruction& ins) {
        return migraphx::contains(ins.name(), "allocate");
    }));
}

TEST_CASE(allocate_with_out_no_params)
{
    migraphx::module m;
    migraphx::shape s{migraphx::shape::float_type, {5}};
    auto x = m.add_parameter("x", s);
    auto y = m.add_parameter("y", s);
    auto z = m.add_parameter("z", s);
    auto alloc =
        m.add_instruction(migraphx::make_op("allocate", {{"shape", migraphx::to_value(s)}}));
    auto pass1 = m.add_instruction(pass_op{}, alloc, x, y);
    auto alloc2 =
        m.add_instruction(migraphx::make_op("allocate", {{"shape", migraphx::to_value(s)}}));
    m.add_instruction(pass_op{}, alloc2, z, pass1);
    run_pass(m, allocation_with_out_model{});

    EXPECT(std::any_of(m.begin(), m.end(), [](const migraphx::instruction& ins) {
        return migraphx::contains(ins.name(), "allocate_with_out");
    }));
}

TEST_CASE(if_allocate)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape cond_s{migraphx::shape::bool_type};
    auto cond = mm->add_parameter("cond", cond_s);
    migraphx::shape s{migraphx::shape::float_type, {5}};
    auto x = mm->add_parameter("x", s);
    auto y = mm->add_parameter("y", s);

    auto* then_mod = p.create_module("If_0_if");
    auto alloc     = then_mod->add_instruction(
        migraphx::make_op("allocate", {{"shape", migraphx::to_value(s)}}));
    auto a1 = then_mod->add_instruction(pass_op{}, alloc, x);
    then_mod->add_return({a1});

    auto* else_mod = p.create_module("If_0_else");
    auto alloc1    = else_mod->add_instruction(
        migraphx::make_op("allocate", {{"shape", migraphx::to_value(s)}}));
    auto a2 = else_mod->add_instruction(pass_op{}, alloc1, y);
    else_mod->add_return({a2});

    mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});

    run_pass(p, allocation_with_out_model{});
    EXPECT(std::any_of(mm->begin(), mm->end(), [](const migraphx::instruction& ins) {
        return migraphx::contains(ins.name(), "allocate_with_out");
    }));
}

TEST_CASE(allocate_copy_with_out)
{
    migraphx::shape s{migraphx::shape::float_type, {5}};
    migraphx::module m1;
    {
        auto x     = m1.add_parameter("x", s);
        auto y     = m1.add_parameter("y", s);
        auto pass  = m1.add_instruction(tuple_op{}, x, y);
        auto elem1 = m1.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), pass);
        m1.add_return({elem1});
    }
    run_pass(m1, allocation_with_out_model{});
    migraphx::module m2;
    {
        auto x      = m2.add_parameter("x", s);
        auto y      = m2.add_parameter("y", s);
        auto output = m2.add_parameter("output", s);
        auto pass   = m2.add_instruction(tuple_op{}, x, y);
        auto elem1  = m2.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), pass);
        auto copy   = m2.add_instruction(migraphx::make_op("test_copy"), elem1, output);
        m2.add_return({copy});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(allocate_copy_with_no_out)
{
    migraphx::shape s{migraphx::shape::float_type, {5}};
    migraphx::module m1;
    {
        auto x     = m1.add_parameter("x", s);
        auto y     = m1.add_parameter("y", s);
        auto pass  = m1.add_instruction(tuple_op{}, x, y);
        auto elem1 = m1.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), pass);
        m1.add_return({elem1});
    }
    migraphx::module m2 = m1;
    run_pass(m1, allocation_no_out_model{});
    EXPECT(m1.sort() == m2.sort());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
