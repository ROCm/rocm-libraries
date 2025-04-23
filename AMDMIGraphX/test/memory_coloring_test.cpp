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
#include <migraphx/memory_coloring.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::memory_coloring{"allocate", true}});
}

struct allocate
{
    migraphx::shape s{};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::pack(f(self.s, "shape"));
    }

    std::string name() const { return "allocate"; }
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

static migraphx::instruction_ref add_alloc(migraphx::module& m, const migraphx::shape& s)
{
    return m.add_instruction(allocate{s});
}

static bool no_allocate(const migraphx::module& m)
{
    return std::none_of(m.begin(), m.end(), [](auto&& ins) { return ins.name() == "allocate"; });
}

static bool is_overlap(std::pair<std::size_t, std::size_t> x, std::pair<std::size_t, std::size_t> y)
{
    return std::max(x.first, y.first) < std::min(x.second, y.second);
}

static std::pair<std::size_t, std::size_t> get_load_interval(migraphx::instruction_ref a)
{
    auto v      = a->get_operator().to_value();
    auto offset = v.at("offset").to<std::size_t>();
    auto s      = migraphx::from_value<migraphx::shape>(v.at("shape"));
    return {offset, offset + s.bytes()};
}

static bool is_overlap_load(migraphx::instruction_ref a, migraphx::instruction_ref b)
{
    return is_overlap(get_load_interval(a), get_load_interval(b));
}

static bool is_disjoint(const std::vector<migraphx::instruction_ref>& inss)
{
    return std::none_of(inss.begin(), inss.end(), [&](auto ins1) {
        return std::none_of(inss.begin(), inss.end(), [&](auto ins2) {
            if(ins1 == ins2)
                return true;
            return is_overlap_load(ins1, ins2);
        });
    });
}

TEST_CASE(test1)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto m1 = m.add_instruction(pass_op{}, a1);
    auto a2 = add_alloc(m, {migraphx::shape::float_type, {40}});
    m.add_instruction(pass_op{}, a2, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 192);
    CHECK(no_allocate(m));
    CHECK(is_disjoint({a1, a2}));
}

TEST_CASE(test2)
{
    migraphx::module m;

    auto input = m.add_parameter("input", migraphx::shape{migraphx::shape::float_type, {16}});

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {128}});
    auto m1 = m.add_instruction(pass_op{}, a1, input);
    auto m2 = add_alloc(m, {migraphx::shape::float_type, {40}});
    m.add_instruction(pass_op{}, m2, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 672);
    CHECK(no_allocate(m));
}

TEST_CASE(test3)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto m2 = add_alloc(m, {migraphx::shape::float_type, {128}});
    auto m1 = m.add_instruction(pass_op{}, m2, a1);
    auto p3 = add_alloc(m, {migraphx::shape::float_type, {40}});
    m.add_instruction(pass_op{}, p3, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 672);
    CHECK(no_allocate(m));
}

TEST_CASE(test4)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {0}});
    auto m2 = add_alloc(m, {migraphx::shape::float_type, {128}});
    auto m1 = m.add_instruction(pass_op{}, m2, a1);
    auto p3 = add_alloc(m, {migraphx::shape::float_type, {40}});
    m.add_instruction(pass_op{}, p3, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 672);
    CHECK(no_allocate(m));
}

TEST_CASE(test5)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto m1 = m.add_instruction(pass_op{}, a1);
    auto m2 = add_alloc(m, {migraphx::shape::float_type, {8}});
    m.add_instruction(pass_op{}, m2, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 192);
    CHECK(no_allocate(m));
}

TEST_CASE(test6)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto m1 = m.add_instruction(pass_op{}, a1);
    auto m2 = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto p3 = add_alloc(m, {migraphx::shape::float_type, {40}});
    m.add_instruction(pass_op{}, p3, m2, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 352);
    CHECK(no_allocate(m));
}

TEST_CASE(test7)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto m1 = m.add_instruction(pass_op{}, a1);
    auto m2 = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto p3 = add_alloc(m, {migraphx::shape::float_type, {8}});
    m.add_instruction(pass_op{}, p3, m2, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 224);
    CHECK(no_allocate(m));
}

TEST_CASE(test8)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto m1 = m.add_instruction(pass_op{}, a1);
    auto m2 = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto p3 = add_alloc(m, {migraphx::shape::float_type, {192}});
    m.add_instruction(pass_op{}, p3, m2, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 960);
    CHECK(no_allocate(m));
}

TEST_CASE(test9)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto m1 = m.add_instruction(pass_op{}, a1);
    auto m2 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto p3 = add_alloc(m, {migraphx::shape::float_type, {8}});
    m.add_instruction(pass_op{}, p3, m2, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 96);
    CHECK(no_allocate(m));
}

TEST_CASE(test10)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {8}});
    m.add_instruction(pass_op{}, a1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 32);
    CHECK(no_allocate(m));
}

TEST_CASE(test11)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto m1 = m.add_instruction(pass_op{}, a1);
    auto a2 = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto a3 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto m2 = m.add_instruction(pass_op{}, a2, m1);
    m.add_instruction(pass_op{}, a3, m2);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 224);
    CHECK(no_allocate(m));
}

TEST_CASE(test12)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto m1 = m.add_instruction(pass_op{}, a1);
    auto a2 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto a3 = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto m2 = m.add_instruction(pass_op{}, a2, m1);
    m.add_instruction(pass_op{}, a3, m2);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 352);
    CHECK(no_allocate(m));
}

TEST_CASE(test13)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto a3 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto m1 = m.add_instruction(pass_op{}, a1);
    auto a2 = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto m2 = m.add_instruction(pass_op{}, a2, m1);
    m.add_instruction(pass_op{}, a3, m2);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 224);
    CHECK(no_allocate(m));
}

TEST_CASE(test14)
{
    migraphx::module m;

    auto a3 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto a2 = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto a1 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto m1 = m.add_instruction(pass_op{}, a1);
    auto m2 = m.add_instruction(pass_op{}, a2, m1);
    m.add_instruction(pass_op{}, a3, m2);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 224);
    CHECK(no_allocate(m));
}

TEST_CASE(test15)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto m1 = m.add_instruction(pass_op{}, a1);
    auto a2 = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto m2 = m.add_instruction(pass_op{}, a2);
    auto a3 = add_alloc(m, {migraphx::shape::float_type, {40}});
    m.add_instruction(pass_op{}, a3, m1, m2);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 352);
    CHECK(no_allocate(m));
}

TEST_CASE(test16)
{
    migraphx::module m;

    auto a1 = m.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8}}));
    auto m1 = m.add_instruction(pass_op{}, a1);
    auto a2 = m.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {40}}));
    auto m2 = m.add_instruction(pass_op{}, a2);
    auto a3 = add_alloc(m, {migraphx::shape::float_type, {40}});
    m.add_instruction(pass_op{}, a3, m1, m2);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 160);
    CHECK(no_allocate(m));
}

TEST_CASE(test17)
{
    migraphx::module m;

    auto a3 = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto a1 = m.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8}}));
    auto m1 = m.add_instruction(pass_op{}, a1);
    auto a2 = m.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {40}}));
    auto m2 = m.add_instruction(pass_op{}, a2);
    m.add_instruction(pass_op{}, a3, m1, m2);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 160);
    CHECK(no_allocate(m));
}

TEST_CASE(test18)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto m1 = m.add_instruction(pass_op{}, a1);
    auto m2 = m.add_instruction(pass_op{}, a1, m1);
    auto p3 = m.add_instruction(pass_op{}, m2, m1);
    auto a2 = add_alloc(m, {migraphx::shape::float_type, {40}});
    m.add_instruction(pass_op{}, a2, m1, m2, p3);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 192);
    CHECK(no_allocate(m));
}

TEST_CASE(test19)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto m1 = m.add_instruction(pass_op{}, a1);
    auto a2 = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto m2 = m.add_instruction(pass_op{}, a2, m1);
    auto a3 = add_alloc(m, {migraphx::shape::float_type, {40}});
    m.add_instruction(pass_op{}, a3, m2, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 352);
    CHECK(no_allocate(m));
}

TEST_CASE(test20)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {32}});
    auto a2 = add_alloc(m, {migraphx::shape::float_type, {32}});
    auto a3 = add_alloc(m, {migraphx::shape::float_type, {32}});
    auto m1 = m.add_instruction(pass_op{}, a1, a2, a3);
    auto a4 = add_alloc(m, {migraphx::shape::float_type, {32}});
    m.add_instruction(pass_op{}, a4, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 384);
    CHECK(no_allocate(m));
}

TEST_CASE(test21)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {32}});
    auto a2 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto a3 = add_alloc(m, {migraphx::shape::float_type, {32}});
    auto m1 = m.add_instruction(pass_op{}, a1, a2, a3);
    auto a4 = add_alloc(m, {migraphx::shape::float_type, {8}});
    m.add_instruction(pass_op{}, a4, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 288);
    CHECK(no_allocate(m));
}

TEST_CASE(test22)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {32}});
    auto a2 = add_alloc(m, {migraphx::shape::float_type, {32}});
    auto a3 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto m1 = m.add_instruction(pass_op{}, a1, a2, a3);
    auto a4 = add_alloc(m, {migraphx::shape::float_type, {8}});
    m.add_instruction(pass_op{}, a4, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 288);
    CHECK(no_allocate(m));
}

TEST_CASE(test23)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto a2 = add_alloc(m, {migraphx::shape::float_type, {32}});
    auto a3 = add_alloc(m, {migraphx::shape::float_type, {32}});
    auto m1 = m.add_instruction(pass_op{}, a1, a2, a3);
    auto a4 = add_alloc(m, {migraphx::shape::float_type, {8}});
    m.add_instruction(pass_op{}, a4, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 288);
    CHECK(no_allocate(m));
}

TEST_CASE(test24)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {32}});
    auto a2 = add_alloc(m, {migraphx::shape::float_type, {32}});
    auto a3 = add_alloc(m, {migraphx::shape::float_type, {32}});
    auto m1 = m.add_instruction(pass_op{}, a1, a2, a3);
    auto a4 = add_alloc(m, {migraphx::shape::float_type, {8}});
    m.add_instruction(pass_op{}, a4, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 384);
    CHECK(no_allocate(m));
}

TEST_CASE(test25)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {8}});
    m.add_instruction(nop{});
    auto m1 = m.add_instruction(pass_op{}, a1);
    m.add_instruction(nop{});
    auto a2 = add_alloc(m, {migraphx::shape::float_type, {40}});
    m.add_instruction(pass_op{}, a2, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 192);
    CHECK(no_allocate(m));
}

TEST_CASE(test26)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {8}});
    m.add_instruction(nop{}, a1);
    auto m1 = m.add_instruction(pass_op{}, a1);
    m.add_instruction(nop{}, a1, m1);
    auto a2 = add_alloc(m, {migraphx::shape::float_type, {40}});
    m.add_instruction(pass_op{}, a2, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 192);
    CHECK(no_allocate(m));
}

TEST_CASE(test27)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto m1 = m.add_instruction(pass_op{}, a1);
    auto a2 = add_alloc(m, {migraphx::shape::float_type, {40}});
    m.add_instruction(nop{}, a2, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 192);
    CHECK(no_allocate(m));
}

TEST_CASE(test28)
{
    migraphx::module m;

    auto output = m.add_parameter("output", {migraphx::shape::float_type, {8}});
    auto a1     = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto m1     = m.add_instruction(pass_op{}, a1);
    auto a2     = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto m2     = m.add_instruction(pass_op{}, a2, m1);
    m.add_instruction(pass_op{}, m2, output);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 192);
    CHECK(no_allocate(m));
}

TEST_CASE(test29)
{
    migraphx::module m;
    auto output = m.add_parameter("output", {migraphx::shape::float_type, {8}});
    auto a1     = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto m1     = m.add_instruction(pass_op{}, a1);
    auto a2     = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto m2     = m.add_instruction(pass_op{}, a2, m1);
    m.move_instruction(output, m2);
    m.add_instruction(pass_op{}, m2, output);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 192);
    CHECK(no_allocate(m));
}

TEST_CASE(test30)
{
    migraphx::module m;

    auto output = m.add_parameter("x", {migraphx::shape::float_type, {8}});
    auto a1     = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto m1     = m.add_instruction(pass_op{}, a1);
    auto a2     = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto m2     = m.add_instruction(pass_op{}, a2, m1);
    m.move_instruction(output, m2);
    m.add_instruction(pass_op{}, m2, output);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 192);
    CHECK(no_allocate(m));
}

TEST_CASE(test31)
{
    migraphx::module m;

    auto output = m.add_parameter("output", {migraphx::shape::float_type, {8}});
    auto a1     = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto m1     = m.add_instruction(pass_op{}, a1);
    auto a2     = add_alloc(m, {migraphx::shape::float_type, {40}});
    m.move_instruction(output, a2);
    m.add_instruction(pass_op{}, a2, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 192);
    CHECK(no_allocate(m));
}

TEST_CASE(test32)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto a2 = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto a3 = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto m1 = m.add_instruction(pass_op{}, a2, a1, a3);
    auto a5 = add_alloc(m, {migraphx::shape::float_type, {40}});
    m.add_instruction(pass_op{}, a5, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 352);
    CHECK(no_allocate(m));
}

TEST_CASE(test33)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto a2 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto a3 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto m1 = m.add_instruction(pass_op{}, a2, a1, a3);
    auto a5 = add_alloc(m, {migraphx::shape::float_type, {40}});
    m.add_instruction(pass_op{}, a5, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 192);
    CHECK(no_allocate(m));
}

TEST_CASE(test34)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto a2 = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto a3 = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto m1 = m.add_instruction(pass_op{}, a2, a1, a3);
    auto a5 = add_alloc(m, {migraphx::shape::float_type, {8}});
    m.add_instruction(pass_op{}, a5, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 480);
    CHECK(no_allocate(m));
}

TEST_CASE(test35)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto a2 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto a3 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto m1 = m.add_instruction(pass_op{}, a2, a1, a3);
    auto a5 = add_alloc(m, {migraphx::shape::float_type, {8}});
    m.add_instruction(pass_op{}, a5, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 224);
    CHECK(no_allocate(m));
}

TEST_CASE(test36)
{
    migraphx::module m;

    auto output = m.add_parameter("output", {migraphx::shape::float_type, {20}});
    auto a1     = add_alloc(m, {migraphx::shape::float_type, {0}});
    auto a2     = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto m1     = m.add_instruction(pass_op{}, a2, a1);
    auto a3     = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto m2     = m.add_instruction(pass_op{}, a3, m1);
    auto a4     = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto p3     = m.add_instruction(pass_op{}, a4, m2);
    m.add_instruction(pass_op{}, output, p3);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 320);
    CHECK(no_allocate(m));
}

TEST_CASE(test37)
{
    migraphx::module m;

    auto output = m.add_parameter("output", {migraphx::shape::float_type, {20}});
    auto a1     = add_alloc(m, {migraphx::shape::float_type, {4}});
    auto a2     = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto m1     = m.add_instruction(pass_op{}, a2, a1);
    auto a3     = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto m2     = m.add_instruction(pass_op{}, a3, m1);
    auto a4     = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto p3     = m.add_instruction(pass_op{}, a4, m2);
    m.add_instruction(pass_op{}, output, p3);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 320);
    CHECK(no_allocate(m));
}

TEST_CASE(test38)
{
    migraphx::module m;

    auto output = m.add_parameter("output", {migraphx::shape::float_type, {1, 64, 56, 56}});
    auto m29    = add_alloc(m, {migraphx::shape::float_type, {0}});
    auto p30    = add_alloc(m, {migraphx::shape::float_type, {1, 64, 112, 112}});
    auto p31    = m.add_instruction(pass_op{}, p30, m29);
    auto p32    = add_alloc(m, {migraphx::shape::float_type, {1, 64, 112, 112}});
    auto p37    = m.add_instruction(pass_op{}, p32, p31);
    auto p38    = add_alloc(m, {migraphx::shape::float_type, {1, 64, 112, 112}});
    auto p39    = m.add_instruction(pass_op{}, p38, p37);
    auto p40    = add_alloc(m, {migraphx::shape::float_type, {1, 64, 56, 56}});
    auto p41    = m.add_instruction(pass_op{}, p40, p39);
    auto p42    = add_alloc(m, {migraphx::shape::float_type, {0}});
    auto p43    = add_alloc(m, {migraphx::shape::float_type, {1, 64, 56, 56}});
    auto p44    = m.add_instruction(pass_op{}, p43, p41, p42);
    auto p45    = add_alloc(m, {migraphx::shape::float_type, {1, 64, 56, 56}});
    auto p50    = m.add_instruction(pass_op{}, p45, p44);
    auto p51    = add_alloc(m, {migraphx::shape::float_type, {1, 64, 56, 56}});
    auto p52    = m.add_instruction(pass_op{}, p51, p50);
    auto p53    = add_alloc(m, {migraphx::shape::float_type, {0}});
    auto p54    = add_alloc(m, {migraphx::shape::float_type, {1, 64, 56, 56}});
    auto p55    = m.add_instruction(pass_op{}, p54, p52, p53);
    auto p56    = add_alloc(m, {migraphx::shape::float_type, {1, 64, 56, 56}});
    auto p61    = m.add_instruction(pass_op{}, p56, p55);
    auto p62    = add_alloc(m, {migraphx::shape::float_type, {1, 64, 56, 56}});
    auto p63    = m.add_instruction(pass_op{}, p62, p61, p41);
    auto p64    = add_alloc(m, {migraphx::shape::float_type, {0}});
    auto p65    = add_alloc(m, {migraphx::shape::float_type, {1, 64, 56, 56}});
    auto p66    = m.add_instruction(pass_op{}, p65, p63, p64);
    auto p67    = add_alloc(m, {migraphx::shape::float_type, {1, 64, 56, 56}});
    auto p72    = m.add_instruction(pass_op{}, p67, p66);
    auto p73    = add_alloc(m, {migraphx::shape::float_type, {1, 64, 56, 56}});
    auto p74    = m.add_instruction(pass_op{}, p73, p72);
    auto p75    = add_alloc(m, {migraphx::shape::float_type, {0}});
    auto p76    = add_alloc(m, {migraphx::shape::float_type, {1, 64, 56, 56}});
    auto p77    = m.add_instruction(pass_op{}, p76, p74, p75);
    auto p78    = add_alloc(m, {migraphx::shape::float_type, {1, 64, 56, 56}});
    auto p83    = m.add_instruction(pass_op{}, p78, p77);
    m.add_instruction(pass_op{}, output, p83, p63);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 6422528);
    CHECK(no_allocate(m));
}

TEST_CASE(test39)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape cond_s{migraphx::shape::bool_type};
    auto cond   = add_alloc(*mm, cond_s);
    auto output = mm->add_parameter("output", {migraphx::shape::float_type, {20}});

    migraphx::shape ds{migraphx::shape::float_type, {2, 3}};
    std::vector<float> data1 = {0.384804, -1.77948, -0.453775, 0.477438, -1.06333, -1.12893};
    auto l1                  = mm->add_literal(migraphx::literal(ds, data1));
    std::vector<float> data2 = {-0.258047, 0.360394, 0.536804, -0.577762, 1.0217, 1.02442};
    auto l2                  = mm->add_literal(migraphx::literal(ds, data2));

    auto* then_mod = p.create_module("If_0_if");
    auto i1        = add_alloc(*then_mod, ds);
    auto a1        = then_mod->add_instruction(pass_op{}, i1, l1);
    then_mod->add_return({a1, output});

    auto* else_mod = p.create_module("If_0_else");
    auto i2        = add_alloc(*else_mod, ds);
    auto a2        = else_mod->add_instruction(pass_op{}, i2, l2);
    else_mod->add_return({a2, output});

    auto ret = mm->add_instruction(mod_pass_op{}, {cond}, {then_mod, else_mod});
    mm->add_return({ret, output});

    auto sub_modules = p.get_modules();
    std::reverse(sub_modules.begin(), sub_modules.end());
    for(const auto& smod : sub_modules)
    {
        run_pass(*smod);
    }

    CHECK(mm->get_parameter_shape("scratch").bytes() == 1);
    CHECK(then_mod->get_parameter_shape("scratch").bytes() == 24);
    CHECK(else_mod->get_parameter_shape("scratch").bytes() == 24);
    CHECK(no_allocate(*mm));
    CHECK(no_allocate(*then_mod));
    CHECK(no_allocate(*else_mod));
}

// NOLINTNEXTLINE
TEST_CASE(rnn_dom)
{
    migraphx::module m;

    auto mx0 = add_alloc(m, migraphx::shape{migraphx::shape::float_type, {2, 10}});
    auto mx1 = m.add_instruction(pass_op{});
    auto mr  = m.add_parameter("r", migraphx::shape{migraphx::shape::float_type, {1, 15, 5}});
    auto mx2 = m.add_instruction(pass_op{}, mr);
    auto mx3 = m.add_instruction(pass_op{}, mx2);
    auto mx4 = m.add_instruction(pass_op{}, mx3);
    m.add_instruction(pass_op{});
    auto mx6 = m.add_instruction(pass_op{}, mx0, mx1, mx4);
    m.add_instruction(pass_op{});
    auto mx8 = add_alloc(m, migraphx::shape{migraphx::shape::float_type, {2, 15}});
    m.add_instruction(pass_op{}, mx8, mx1, mx0, mx6);
    auto mseq = m.add_parameter("seq", migraphx::shape{migraphx::shape::float_type, {3, 2, 8}});
    auto mx10 = m.add_instruction(pass_op{}, mseq);
    auto mx11 = m.add_instruction(pass_op{}, mx10);
    auto mw   = m.add_parameter("w", migraphx::shape{migraphx::shape::float_type, {1, 15, 8}});
    auto mx12 = m.add_instruction(pass_op{}, mw);
    auto mx13 = m.add_instruction(pass_op{}, mx12);
    m.add_instruction(pass_op{});
    auto mx15 = m.add_instruction(pass_op{}, mx8, mx11, mx13);
    m.add_instruction(pass_op{}, mx15, mx1, mx0, mx6);
    m.add_instruction(pass_op{});
    auto mx18 = add_alloc(m, migraphx::shape{migraphx::shape::float_type, {2, 5}});
    m.add_instruction(pass_op{}, mx18, mx6, mx15, mx0, mx1, mx8);
    auto mx20 = m.add_instruction(pass_op{}, mx6);
    m.add_instruction(pass_op{}, mx20, mx8, mx15, mx18);
    auto mx22 = m.add_instruction(pass_op{}, mx15);
    m.add_instruction(pass_op{}, mx22, mx1, mx0, mx20, mx6, mx18);
    m.add_instruction(pass_op{});
    m.add_instruction(pass_op{});
    m.add_instruction(pass_op{});
    auto mx27 = m.add_instruction(pass_op{}, mx18, mx22, mx20);
    m.add_instruction(pass_op{}, mx27, mx15, mx8, mx6, mx20, mx1, mx22, mx0);
    m.add_instruction(pass_op{});
    auto mx30 = add_alloc(m, migraphx::shape{migraphx::shape::float_type, {2, 5}});
    m.add_instruction(pass_op{}, mx30, mx20, mx22, mx1, mx15, mx8, mx6, mx27, mx0, mx18);
    auto mx32 = m.add_instruction(pass_op{}, mx15);
    m.add_instruction(pass_op{}, mx32, mx20, mx30, mx0, mx18, mx1, mx27, mx6);
    m.add_instruction(pass_op{});
    m.add_instruction(pass_op{});
    auto mx36 = m.add_instruction(pass_op{}, mx30, mx32);
    m.add_instruction(pass_op{}, mx36, mx32, mx0, mx27, mx8, mx1, mx15, mx6, mx20, mx22, mx18);
    auto mx38 = add_alloc(m, migraphx::shape{migraphx::shape::float_type, {2, 5}});
    m.add_instruction(pass_op{}, mx38, mx32, mx0, mx27, mx8, mx1, mx15, mx6, mx20, mx22, mx18);
    auto mx40 = m.add_instruction(pass_op{}, mx38, mx36);
    m.add_instruction(pass_op{}, mx40, mx32, mx0, mx27, mx8, mx1, mx15, mx6, mx20, mx22, mx18);
    m.add_instruction(pass_op{});
    auto mx43 = add_alloc(m, migraphx::shape{migraphx::shape::float_type, {2, 5}});
    m.add_instruction(pass_op{}, mx43, mx15, mx32, mx27, mx30, mx18, mx8, mx40, mx36, mx22, mx38);
    auto mx45 = m.add_instruction(pass_op{}, mx6);
    m.add_instruction(pass_op{}, mx45, mx32, mx27, mx30, mx18, mx40, mx36, mx22, mx8, mx15, mx38);
    auto mx47 = m.add_instruction(pass_op{}, mx15);
    m.add_instruction(
        pass_op{}, mx47, mx30, mx18, mx43, mx6, mx1, mx45, mx0, mx27, mx36, mx20, mx40, mx38);
    m.add_instruction(pass_op{});
    m.add_instruction(pass_op{});
    auto mx51 = m.add_instruction(pass_op{}, mx43, mx47, mx45);
    m.add_instruction(
        pass_op{}, mx51, mx15, mx47, mx32, mx27, mx30, mx18, mx8, mx36, mx22, mx40, mx38);
    auto mx53 = add_alloc(m, migraphx::shape{migraphx::shape::float_type, {2, 5}});
    m.add_instruction(
        pass_op{}, mx53, mx15, mx47, mx32, mx27, mx30, mx18, mx8, mx36, mx22, mx40, mx38);
    auto mx55 = m.add_instruction(pass_op{}, mx53, mx51, mx1);
    m.add_instruction(
        pass_op{}, mx55, mx15, mx47, mx32, mx27, mx30, mx18, mx8, mx36, mx22, mx40, mx38);
    auto mx57 = m.add_instruction(pass_op{}, mx3);
    m.add_instruction(pass_op{});
    auto mx59 = m.add_instruction(pass_op{}, mx40, mx55, mx57, mx40);
    m.add_instruction(
        pass_op{}, mx59, mx15, mx8, mx38, mx18, mx30, mx27, mx47, mx32, mx40, mx36, mx22);
    auto mx61 = add_alloc(m, migraphx::shape{migraphx::shape::float_type, {2, 5}});
    m.add_instruction(pass_op{},
                      mx61,
                      mx30,
                      mx15,
                      mx1,
                      mx51,
                      mx20,
                      mx59,
                      mx32,
                      mx45,
                      mx22,
                      mx8,
                      mx47,
                      mx40,
                      mx53,
                      mx6,
                      mx55,
                      mx0,
                      mx43,
                      mx38,
                      mx36);
    m.add_instruction(pass_op{});
    auto mx64 = m.add_instruction(pass_op{}, mx61, mx27, mx1);
    m.add_instruction(pass_op{},
                      mx64,
                      mx30,
                      mx15,
                      mx1,
                      mx51,
                      mx20,
                      mx59,
                      mx32,
                      mx45,
                      mx22,
                      mx8,
                      mx47,
                      mx40,
                      mx53,
                      mx6,
                      mx55,
                      mx0,
                      mx43,
                      mx38,
                      mx36);
    m.add_instruction(pass_op{});
    auto mx67 = add_alloc(m, migraphx::shape{migraphx::shape::float_type, {2, 5}});
    m.add_instruction(pass_op{},
                      mx67,
                      mx18,
                      mx6,
                      mx1,
                      mx51,
                      mx20,
                      mx59,
                      mx27,
                      mx55,
                      mx43,
                      mx38,
                      mx0,
                      mx61,
                      mx45,
                      mx36,
                      mx40,
                      mx53,
                      mx64,
                      mx30);
    auto mx69 = m.add_instruction(pass_op{});
    m.add_instruction(pass_op{},
                      mx69,
                      mx18,
                      mx6,
                      mx1,
                      mx51,
                      mx20,
                      mx59,
                      mx27,
                      mx55,
                      mx43,
                      mx38,
                      mx0,
                      mx61,
                      mx45,
                      mx36,
                      mx40,
                      mx53,
                      mx64,
                      mx30);
    m.add_instruction(pass_op{});
    m.add_instruction(pass_op{});
    auto mx73 = m.add_instruction(pass_op{}, mx67, mx69, mx27);
    m.add_instruction(pass_op{},
                      mx73,
                      mx18,
                      mx6,
                      mx1,
                      mx51,
                      mx20,
                      mx59,
                      mx27,
                      mx55,
                      mx43,
                      mx38,
                      mx0,
                      mx61,
                      mx45,
                      mx36,
                      mx40,
                      mx53,
                      mx64,
                      mx30);
    m.add_instruction(pass_op{});
    auto mx76 = add_alloc(m, migraphx::shape{migraphx::shape::float_type, {2, 5}});
    m.add_instruction(pass_op{},
                      mx76,
                      mx64,
                      mx30,
                      mx18,
                      mx40,
                      mx8,
                      mx61,
                      mx38,
                      mx69,
                      mx67,
                      mx73,
                      mx27,
                      mx47,
                      mx32,
                      mx36,
                      mx15,
                      mx22);
    m.add_instruction(pass_op{});
    auto mx79 = m.add_instruction(pass_op{}, mx76, mx59);
    m.add_instruction(pass_op{},
                      mx79,
                      mx64,
                      mx30,
                      mx18,
                      mx40,
                      mx8,
                      mx61,
                      mx38,
                      mx69,
                      mx67,
                      mx73,
                      mx27,
                      mx47,
                      mx32,
                      mx36,
                      mx15,
                      mx22);
    auto mx81 = add_alloc(m, migraphx::shape{migraphx::shape::float_type, {2, 5}});
    m.add_instruction(pass_op{},
                      mx81,
                      mx36,
                      mx32,
                      mx27,
                      mx47,
                      mx18,
                      mx30,
                      mx73,
                      mx67,
                      mx22,
                      mx15,
                      mx61,
                      mx8,
                      mx64,
                      mx40,
                      mx69,
                      mx38);
    m.add_instruction(pass_op{});
    m.add_instruction(pass_op{});
    auto mx85 = m.add_instruction(pass_op{}, mx81, mx73, mx79, mx64);
    m.add_instruction(pass_op{},
                      mx85,
                      mx36,
                      mx32,
                      mx27,
                      mx47,
                      mx18,
                      mx30,
                      mx73,
                      mx67,
                      mx22,
                      mx15,
                      mx61,
                      mx8,
                      mx64,
                      mx40,
                      mx69,
                      mx38);
    m.add_instruction(pass_op{});
    auto mx88 = add_alloc(m, migraphx::shape{migraphx::shape::float_type, {2, 10}});
    m.add_instruction(pass_op{},
                      mx88,
                      mx36,
                      mx32,
                      mx27,
                      mx47,
                      mx18,
                      mx30,
                      mx73,
                      mx67,
                      mx22,
                      mx15,
                      mx61,
                      mx8,
                      mx64,
                      mx40,
                      mx69,
                      mx38);
    auto mx90 = m.add_instruction(pass_op{}, mx88, mx85, mx4);
    m.add_instruction(pass_op{},
                      mx90,
                      mx36,
                      mx32,
                      mx27,
                      mx47,
                      mx18,
                      mx30,
                      mx73,
                      mx67,
                      mx22,
                      mx15,
                      mx61,
                      mx8,
                      mx64,
                      mx40,
                      mx69,
                      mx38);
    m.add_instruction(pass_op{});
    auto mx93 = add_alloc(m, migraphx::shape{migraphx::shape::float_type, {2, 15}});
    m.add_instruction(pass_op{},
                      mx93,
                      mx51,
                      mx88,
                      mx20,
                      mx64,
                      mx43,
                      mx61,
                      mx53,
                      mx81,
                      mx47,
                      mx6,
                      mx45,
                      mx0,
                      mx55,
                      mx18,
                      mx76,
                      mx1,
                      mx79,
                      mx85,
                      mx90,
                      mx8,
                      mx69,
                      mx67,
                      mx73,
                      mx32,
                      mx59,
                      mx22,
                      mx15,
                      mx27);
    auto mx95 = m.add_instruction(pass_op{}, mseq);
    auto mx96 = m.add_instruction(pass_op{}, mx95);
    m.add_instruction(pass_op{});
    auto mx98 = m.add_instruction(pass_op{}, mx93, mx96, mx13);
    m.add_instruction(pass_op{},
                      mx98,
                      mx51,
                      mx88,
                      mx20,
                      mx64,
                      mx43,
                      mx61,
                      mx53,
                      mx81,
                      mx47,
                      mx6,
                      mx45,
                      mx0,
                      mx55,
                      mx18,
                      mx76,
                      mx1,
                      mx79,
                      mx85,
                      mx90,
                      mx8,
                      mx69,
                      mx67,
                      mx73,
                      mx32,
                      mx59,
                      mx22,
                      mx15,
                      mx27);
    m.add_instruction(pass_op{});
    auto mx101 = add_alloc(m, migraphx::shape{migraphx::shape::float_type, {2, 5}});
    m.add_instruction(pass_op{},
                      mx101,
                      mx43,
                      mx40,
                      mx53,
                      mx59,
                      mx51,
                      mx6,
                      mx61,
                      mx81,
                      mx38,
                      mx45,
                      mx20,
                      mx0,
                      mx76,
                      mx55,
                      mx18,
                      mx85,
                      mx1,
                      mx93,
                      mx79,
                      mx90,
                      mx27,
                      mx88,
                      mx64,
                      mx30,
                      mx98,
                      mx36);
    auto mx103 = m.add_instruction(pass_op{}, mx90);
    m.add_instruction(pass_op{},
                      mx103,
                      mx64,
                      mx101,
                      mx15,
                      mx67,
                      mx73,
                      mx18,
                      mx40,
                      mx8,
                      mx47,
                      mx98,
                      mx27,
                      mx32,
                      mx61,
                      mx22,
                      mx93,
                      mx69,
                      mx36,
                      mx38,
                      mx30);
    auto mx105 = m.add_instruction(pass_op{}, mx98);
    m.add_instruction(pass_op{},
                      mx105,
                      mx43,
                      mx88,
                      mx53,
                      mx64,
                      mx59,
                      mx6,
                      mx76,
                      mx61,
                      mx81,
                      mx47,
                      mx103,
                      mx22,
                      mx45,
                      mx0,
                      mx55,
                      mx18,
                      mx85,
                      mx51,
                      mx20,
                      mx1,
                      mx79,
                      mx90,
                      mx8,
                      mx101,
                      mx15,
                      mx69,
                      mx67,
                      mx73,
                      mx32,
                      mx27);
    m.add_instruction(pass_op{});
    m.add_instruction(pass_op{});
    m.add_instruction(pass_op{});
    auto mx110 = m.add_instruction(pass_op{}, mx101, mx105, mx103);
    m.add_instruction(pass_op{},
                      mx110,
                      mx88,
                      mx40,
                      mx93,
                      mx59,
                      mx43,
                      mx61,
                      mx53,
                      mx81,
                      mx103,
                      mx6,
                      mx45,
                      mx0,
                      mx55,
                      mx18,
                      mx64,
                      mx20,
                      mx76,
                      mx1,
                      mx79,
                      mx38,
                      mx85,
                      mx90,
                      mx27,
                      mx30,
                      mx105,
                      mx98,
                      mx51,
                      mx36);
    m.add_instruction(pass_op{});
    auto mx113 = add_alloc(m, migraphx::shape{migraphx::shape::float_type, {2, 5}});
    m.add_instruction(pass_op{},
                      mx113,
                      mx59,
                      mx20,
                      mx51,
                      mx1,
                      mx79,
                      mx90,
                      mx55,
                      mx85,
                      mx76,
                      mx81,
                      mx47,
                      mx6,
                      mx38,
                      mx88,
                      mx43,
                      mx40,
                      mx0,
                      mx45,
                      mx53,
                      mx93,
                      mx8,
                      mx101,
                      mx15,
                      mx69,
                      mx67,
                      mx73,
                      mx32,
                      mx110,
                      mx22,
                      mx103,
                      mx30,
                      mx36,
                      mx98,
                      mx105);
    auto mx115 = m.add_instruction(pass_op{}, mx98);
    m.add_instruction(pass_op{},
                      mx115,
                      mx59,
                      mx20,
                      mx51,
                      mx1,
                      mx79,
                      mx90,
                      mx55,
                      mx18,
                      mx85,
                      mx76,
                      mx61,
                      mx81,
                      mx47,
                      mx6,
                      mx88,
                      mx43,
                      mx0,
                      mx45,
                      mx53,
                      mx64,
                      mx8,
                      mx101,
                      mx15,
                      mx69,
                      mx67,
                      mx73,
                      mx113,
                      mx32,
                      mx110,
                      mx22,
                      mx103,
                      mx27);
    m.add_instruction(pass_op{});
    m.add_instruction(pass_op{});
    auto mx119 = m.add_instruction(pass_op{}, mx113, mx115);
    m.add_instruction(pass_op{},
                      mx119,
                      mx59,
                      mx20,
                      mx51,
                      mx1,
                      mx79,
                      mx90,
                      mx55,
                      mx85,
                      mx76,
                      mx81,
                      mx47,
                      mx6,
                      mx38,
                      mx88,
                      mx43,
                      mx40,
                      mx0,
                      mx45,
                      mx53,
                      mx93,
                      mx8,
                      mx101,
                      mx15,
                      mx69,
                      mx67,
                      mx73,
                      mx32,
                      mx110,
                      mx22,
                      mx103,
                      mx30,
                      mx36,
                      mx115,
                      mx98,
                      mx105);
    auto mx121 = add_alloc(m, migraphx::shape{migraphx::shape::float_type, {2, 5}});
    m.add_instruction(pass_op{},
                      mx121,
                      mx59,
                      mx20,
                      mx51,
                      mx1,
                      mx79,
                      mx90,
                      mx55,
                      mx85,
                      mx76,
                      mx81,
                      mx47,
                      mx6,
                      mx38,
                      mx88,
                      mx43,
                      mx40,
                      mx0,
                      mx45,
                      mx53,
                      mx93,
                      mx8,
                      mx101,
                      mx15,
                      mx69,
                      mx67,
                      mx73,
                      mx32,
                      mx110,
                      mx22,
                      mx103,
                      mx30,
                      mx36,
                      mx115,
                      mx98,
                      mx105);
    auto mx123 = m.add_instruction(pass_op{}, mx121, mx119);
    m.add_instruction(pass_op{},
                      mx123,
                      mx59,
                      mx20,
                      mx51,
                      mx1,
                      mx79,
                      mx90,
                      mx55,
                      mx85,
                      mx76,
                      mx81,
                      mx47,
                      mx6,
                      mx38,
                      mx88,
                      mx43,
                      mx40,
                      mx0,
                      mx45,
                      mx53,
                      mx93,
                      mx8,
                      mx101,
                      mx15,
                      mx69,
                      mx67,
                      mx73,
                      mx32,
                      mx110,
                      mx22,
                      mx103,
                      mx30,
                      mx36,
                      mx115,
                      mx98,
                      mx105);
    m.add_instruction(pass_op{});
    auto mx126 = add_alloc(m, migraphx::shape{migraphx::shape::float_type, {2, 5}});
    m.add_instruction(pass_op{},
                      mx126,
                      mx115,
                      mx113,
                      mx8,
                      mx67,
                      mx61,
                      mx73,
                      mx18,
                      mx123,
                      mx119,
                      mx32,
                      mx15,
                      mx36,
                      mx110,
                      mx27,
                      mx101,
                      mx22,
                      mx98,
                      mx47,
                      mx40,
                      mx93,
                      mx38,
                      mx69,
                      mx121,
                      mx64,
                      mx30,
                      mx105);
    auto mx128 = m.add_instruction(pass_op{}, mx90);
    m.add_instruction(pass_op{},
                      mx128,
                      mx93,
                      mx98,
                      mx8,
                      mx67,
                      mx73,
                      mx18,
                      mx123,
                      mx61,
                      mx40,
                      mx47,
                      mx27,
                      mx32,
                      mx101,
                      mx22,
                      mx15,
                      mx110,
                      mx36,
                      mx119,
                      mx38,
                      mx64,
                      mx30,
                      mx69,
                      mx121,
                      mx113,
                      mx115,
                      mx105);
    auto mx130 = m.add_instruction(pass_op{}, mx98);
    m.add_instruction(pass_op{},
                      mx130,
                      mx119,
                      mx64,
                      mx22,
                      mx110,
                      mx126,
                      mx128,
                      mx121,
                      mx113,
                      mx67,
                      mx90,
                      mx69,
                      mx15,
                      mx20,
                      mx8,
                      mx27,
                      mx51,
                      mx85,
                      mx79,
                      mx123,
                      mx103,
                      mx18,
                      mx55,
                      mx32,
                      mx0,
                      mx45,
                      mx61,
                      mx53,
                      mx76,
                      mx6,
                      mx47,
                      mx59,
                      mx73,
                      mx81,
                      mx88,
                      mx1,
                      mx43,
                      mx101);
    m.add_instruction(pass_op{});
    m.add_instruction(pass_op{});
    auto mx134 = m.add_instruction(pass_op{}, mx126, mx130, mx128);
    m.add_instruction(pass_op{},
                      mx134,
                      mx130,
                      mx8,
                      mx67,
                      mx61,
                      mx73,
                      mx18,
                      mx123,
                      mx119,
                      mx32,
                      mx15,
                      mx36,
                      mx110,
                      mx27,
                      mx101,
                      mx22,
                      mx113,
                      mx115,
                      mx98,
                      mx47,
                      mx40,
                      mx93,
                      mx38,
                      mx69,
                      mx121,
                      mx64,
                      mx30,
                      mx105);
    auto mx136 = add_alloc(m, migraphx::shape{migraphx::shape::float_type, {2, 5}});
    m.add_instruction(pass_op{},
                      mx136,
                      mx130,
                      mx8,
                      mx67,
                      mx61,
                      mx73,
                      mx18,
                      mx123,
                      mx119,
                      mx32,
                      mx15,
                      mx36,
                      mx110,
                      mx27,
                      mx101,
                      mx22,
                      mx113,
                      mx115,
                      mx98,
                      mx47,
                      mx40,
                      mx93,
                      mx38,
                      mx69,
                      mx121,
                      mx64,
                      mx30,
                      mx105);
    auto mx138 = m.add_instruction(pass_op{}, mx136, mx134, mx85);
    m.add_instruction(pass_op{},
                      mx138,
                      mx130,
                      mx8,
                      mx67,
                      mx61,
                      mx73,
                      mx18,
                      mx123,
                      mx119,
                      mx32,
                      mx15,
                      mx36,
                      mx110,
                      mx27,
                      mx101,
                      mx22,
                      mx113,
                      mx115,
                      mx98,
                      mx47,
                      mx40,
                      mx93,
                      mx38,
                      mx69,
                      mx121,
                      mx64,
                      mx30,
                      mx105);
    m.add_instruction(pass_op{});
    auto mx141 = m.add_instruction(pass_op{}, mx123, mx138, mx57, mx123);
    m.add_instruction(pass_op{},
                      mx141,
                      mx113,
                      mx115,
                      mx130,
                      mx105,
                      mx38,
                      mx93,
                      mx61,
                      mx98,
                      mx27,
                      mx64,
                      mx30,
                      mx119,
                      mx121,
                      mx69,
                      mx8,
                      mx67,
                      mx40,
                      mx47,
                      mx32,
                      mx101,
                      mx22,
                      mx36,
                      mx110,
                      mx15,
                      mx73,
                      mx18,
                      mx123);
    auto mx143 = add_alloc(m, migraphx::shape{migraphx::shape::float_type, {2, 5}});
    m.add_instruction(pass_op{},
                      mx143,
                      mx8,
                      mx73,
                      mx121,
                      mx67,
                      mx101,
                      mx110,
                      mx69,
                      mx15,
                      mx138,
                      mx88,
                      mx43,
                      mx79,
                      mx53,
                      mx61,
                      mx45,
                      mx18,
                      mx0,
                      mx6,
                      mx27,
                      mx22,
                      mx134,
                      mx32,
                      mx1,
                      mx119,
                      mx59,
                      mx85,
                      mx103,
                      mx126,
                      mx64,
                      mx128,
                      mx55,
                      mx76,
                      mx47,
                      mx81,
                      mx90,
                      mx136,
                      mx51,
                      mx141,
                      mx20,
                      mx113,
                      mx123);
    m.add_instruction(pass_op{});
    m.add_instruction(pass_op{});
    auto mx147 = m.add_instruction(pass_op{}, mx143, mx69, mx110);
    m.add_instruction(pass_op{},
                      mx147,
                      mx8,
                      mx73,
                      mx121,
                      mx67,
                      mx101,
                      mx110,
                      mx69,
                      mx15,
                      mx138,
                      mx88,
                      mx43,
                      mx79,
                      mx53,
                      mx61,
                      mx45,
                      mx18,
                      mx0,
                      mx6,
                      mx27,
                      mx22,
                      mx134,
                      mx32,
                      mx1,
                      mx119,
                      mx59,
                      mx85,
                      mx103,
                      mx126,
                      mx64,
                      mx128,
                      mx55,
                      mx76,
                      mx47,
                      mx81,
                      mx90,
                      mx136,
                      mx51,
                      mx141,
                      mx20,
                      mx113,
                      mx123);
    m.add_instruction(pass_op{});
    auto mx150 = add_alloc(m, migraphx::shape{migraphx::shape::float_type, {2, 5}});
    m.add_instruction(pass_op{},
                      mx150,
                      mx30,
                      mx121,
                      mx115,
                      mx98,
                      mx130,
                      mx85,
                      mx88,
                      mx90,
                      mx79,
                      mx1,
                      mx93,
                      mx64,
                      mx18,
                      mx53,
                      mx61,
                      mx38,
                      mx27,
                      mx147,
                      mx0,
                      mx6,
                      mx51,
                      mx40,
                      mx134,
                      mx43,
                      mx119,
                      mx59,
                      mx45,
                      mx76,
                      mx128,
                      mx81,
                      mx136,
                      mx55,
                      mx138,
                      mx123,
                      mx126,
                      mx141,
                      mx103,
                      mx20,
                      mx105,
                      mx113,
                      mx143,
                      mx36);
    m.add_instruction(pass_op{});
    m.add_instruction(pass_op{});
    auto mx154 = m.add_instruction(pass_op{}, mx150, mx110, mx85);
    m.add_instruction(pass_op{},
                      mx154,
                      mx30,
                      mx121,
                      mx115,
                      mx98,
                      mx130,
                      mx85,
                      mx88,
                      mx90,
                      mx79,
                      mx1,
                      mx93,
                      mx64,
                      mx18,
                      mx53,
                      mx61,
                      mx38,
                      mx27,
                      mx147,
                      mx0,
                      mx6,
                      mx51,
                      mx40,
                      mx134,
                      mx43,
                      mx119,
                      mx59,
                      mx45,
                      mx76,
                      mx128,
                      mx81,
                      mx136,
                      mx55,
                      mx138,
                      mx123,
                      mx126,
                      mx141,
                      mx103,
                      mx20,
                      mx105,
                      mx113,
                      mx143,
                      mx36);
    m.add_instruction(pass_op{});
    auto mx157 = add_alloc(m, migraphx::shape{migraphx::shape::float_type, {2, 5}});
    m.add_instruction(pass_op{},
                      mx157,
                      mx101,
                      mx8,
                      mx115,
                      mx130,
                      mx105,
                      mx38,
                      mx147,
                      mx93,
                      mx64,
                      mx61,
                      mx98,
                      mx40,
                      mx27,
                      mx121,
                      mx30,
                      mx154,
                      mx113,
                      mx73,
                      mx119,
                      mx36,
                      mx150,
                      mx69,
                      mx67,
                      mx47,
                      mx110,
                      mx32,
                      mx22,
                      mx15,
                      mx18,
                      mx123,
                      mx143);
    m.add_instruction(pass_op{});
    auto mx160 = m.add_instruction(pass_op{}, mx157, mx141);
    m.add_instruction(pass_op{},
                      mx160,
                      mx101,
                      mx8,
                      mx115,
                      mx130,
                      mx105,
                      mx38,
                      mx147,
                      mx93,
                      mx64,
                      mx61,
                      mx98,
                      mx40,
                      mx27,
                      mx121,
                      mx30,
                      mx154,
                      mx113,
                      mx73,
                      mx119,
                      mx36,
                      mx150,
                      mx69,
                      mx67,
                      mx47,
                      mx110,
                      mx32,
                      mx22,
                      mx15,
                      mx18,
                      mx123,
                      mx143);
    auto mx162 = add_alloc(m, migraphx::shape{migraphx::shape::float_type, {2, 5}});
    m.add_instruction(pass_op{},
                      mx162,
                      mx101,
                      mx8,
                      mx115,
                      mx130,
                      mx105,
                      mx38,
                      mx147,
                      mx93,
                      mx64,
                      mx61,
                      mx98,
                      mx40,
                      mx27,
                      mx121,
                      mx30,
                      mx154,
                      mx113,
                      mx73,
                      mx119,
                      mx36,
                      mx150,
                      mx69,
                      mx67,
                      mx47,
                      mx110,
                      mx32,
                      mx22,
                      mx15,
                      mx18,
                      mx123,
                      mx143);
    m.add_instruction(pass_op{});
    m.add_instruction(pass_op{});
    auto mx166 = m.add_instruction(pass_op{}, mx162, mx147, mx160, mx154);
    m.add_instruction(pass_op{},
                      mx166,
                      mx101,
                      mx8,
                      mx115,
                      mx130,
                      mx105,
                      mx38,
                      mx147,
                      mx93,
                      mx64,
                      mx61,
                      mx98,
                      mx40,
                      mx27,
                      mx121,
                      mx30,
                      mx154,
                      mx113,
                      mx73,
                      mx119,
                      mx36,
                      mx150,
                      mx69,
                      mx67,
                      mx47,
                      mx110,
                      mx32,
                      mx22,
                      mx15,
                      mx18,
                      mx123,
                      mx143);
    m.add_instruction(pass_op{});
    auto mx169 = add_alloc(m, migraphx::shape{migraphx::shape::float_type, {2, 15}});
    m.add_instruction(pass_op{},
                      mx169,
                      mx154,
                      mx90,
                      mx88,
                      mx79,
                      mx126,
                      mx15,
                      mx103,
                      mx22,
                      mx134,
                      mx166,
                      mx30,
                      mx73,
                      mx20,
                      mx128,
                      mx160,
                      mx8,
                      mx45,
                      mx0,
                      mx6,
                      mx157,
                      mx53,
                      mx136,
                      mx93,
                      mx47,
                      mx81,
                      mx141,
                      mx85,
                      mx110,
                      mx59,
                      mx1,
                      mx162,
                      mx101,
                      mx36,
                      mx38,
                      mx76,
                      mx143,
                      mx67,
                      mx147,
                      mx150,
                      mx138,
                      mx115,
                      mx105,
                      mx51,
                      mx69,
                      mx40,
                      mx32,
                      mx43,
                      mx55,
                      mx130,
                      mx98);
    auto mx171 = m.add_instruction(pass_op{}, mseq);
    auto mx172 = m.add_instruction(pass_op{}, mx171);
    m.add_instruction(pass_op{});
    auto mx174 = m.add_instruction(pass_op{}, mx169, mx172, mx13);
    m.add_instruction(pass_op{},
                      mx174,
                      mx154,
                      mx90,
                      mx88,
                      mx79,
                      mx126,
                      mx15,
                      mx103,
                      mx22,
                      mx134,
                      mx166,
                      mx30,
                      mx73,
                      mx20,
                      mx128,
                      mx160,
                      mx8,
                      mx45,
                      mx0,
                      mx6,
                      mx157,
                      mx53,
                      mx136,
                      mx93,
                      mx47,
                      mx81,
                      mx141,
                      mx85,
                      mx110,
                      mx59,
                      mx1,
                      mx162,
                      mx101,
                      mx36,
                      mx38,
                      mx76,
                      mx143,
                      mx67,
                      mx147,
                      mx150,
                      mx138,
                      mx115,
                      mx105,
                      mx51,
                      mx69,
                      mx40,
                      mx32,
                      mx43,
                      mx55,
                      mx130,
                      mx98);
    m.add_instruction(pass_op{});
    auto mx177 = add_alloc(m, migraphx::shape{migraphx::shape::float_type, {2, 10}});
    m.add_instruction(pass_op{},
                      mx177,
                      mx101,
                      mx8,
                      mx115,
                      mx130,
                      mx105,
                      mx38,
                      mx147,
                      mx93,
                      mx64,
                      mx154,
                      mx61,
                      mx98,
                      mx40,
                      mx27,
                      mx174,
                      mx121,
                      mx30,
                      mx113,
                      mx73,
                      mx119,
                      mx36,
                      mx150,
                      mx69,
                      mx67,
                      mx47,
                      mx110,
                      mx32,
                      mx22,
                      mx169,
                      mx15,
                      mx18,
                      mx123,
                      mx143);
    m.add_instruction(pass_op{});
    auto mx180 = m.add_instruction(pass_op{}, mx177, mx166, mx4);
    m.add_instruction(pass_op{},
                      mx180,
                      mx101,
                      mx8,
                      mx115,
                      mx130,
                      mx105,
                      mx38,
                      mx147,
                      mx93,
                      mx64,
                      mx154,
                      mx61,
                      mx98,
                      mx40,
                      mx27,
                      mx174,
                      mx121,
                      mx30,
                      mx113,
                      mx73,
                      mx119,
                      mx36,
                      mx150,
                      mx69,
                      mx67,
                      mx47,
                      mx110,
                      mx32,
                      mx22,
                      mx169,
                      mx15,
                      mx18,
                      mx123,
                      mx143);
    m.add_instruction(pass_op{});
    auto mx183 = add_alloc(m, migraphx::shape{migraphx::shape::float_type, {2, 5}});
    m.add_instruction(pass_op{},
                      mx183,
                      mx67,
                      mx90,
                      mx150,
                      mx138,
                      mx88,
                      mx79,
                      mx126,
                      mx15,
                      mx103,
                      mx22,
                      mx134,
                      mx180,
                      mx166,
                      mx174,
                      mx73,
                      mx20,
                      mx154,
                      mx32,
                      mx43,
                      mx55,
                      mx157,
                      mx18,
                      mx0,
                      mx113,
                      mx6,
                      mx76,
                      mx53,
                      mx61,
                      mx177,
                      mx136,
                      mx81,
                      mx141,
                      mx85,
                      mx110,
                      mx64,
                      mx45,
                      mx8,
                      mx169,
                      mx59,
                      mx1,
                      mx162,
                      mx101,
                      mx119,
                      mx51,
                      mx69,
                      mx128,
                      mx160,
                      mx27,
                      mx47,
                      mx123,
                      mx121);
    auto mx185 = m.add_instruction(pass_op{}, mx180);
    m.add_instruction(pass_op{},
                      mx185,
                      mx101,
                      mx8,
                      mx115,
                      mx130,
                      mx105,
                      mx38,
                      mx147,
                      mx93,
                      mx64,
                      mx154,
                      mx61,
                      mx98,
                      mx40,
                      mx27,
                      mx183,
                      mx174,
                      mx121,
                      mx30,
                      mx113,
                      mx73,
                      mx119,
                      mx36,
                      mx150,
                      mx69,
                      mx67,
                      mx47,
                      mx110,
                      mx32,
                      mx22,
                      mx169,
                      mx15,
                      mx18,
                      mx123,
                      mx143);
    auto mx187 = m.add_instruction(pass_op{}, mx174);
    m.add_instruction(pass_op{},
                      mx187,
                      mx150,
                      mx128,
                      mx67,
                      mx15,
                      mx88,
                      mx43,
                      mx79,
                      mx126,
                      mx103,
                      mx22,
                      mx90,
                      mx180,
                      mx183,
                      mx166,
                      mx141,
                      mx30,
                      mx20,
                      mx59,
                      mx55,
                      mx38,
                      mx160,
                      mx0,
                      mx32,
                      mx85,
                      mx6,
                      mx76,
                      mx157,
                      mx45,
                      mx162,
                      mx138,
                      mx154,
                      mx53,
                      mx177,
                      mx136,
                      mx51,
                      mx47,
                      mx81,
                      mx93,
                      mx73,
                      mx8,
                      mx110,
                      mx101,
                      mx69,
                      mx185,
                      mx36,
                      mx143,
                      mx147,
                      mx134,
                      mx1,
                      mx130,
                      mx115,
                      mx105,
                      mx40,
                      mx98);
    m.add_instruction(pass_op{});
    m.add_instruction(pass_op{});
    m.add_instruction(pass_op{});
    auto mx192 = m.add_instruction(pass_op{}, mx183, mx187, mx185);
    m.add_instruction(pass_op{},
                      mx192,
                      mx150,
                      mx128,
                      mx67,
                      mx187,
                      mx15,
                      mx88,
                      mx43,
                      mx79,
                      mx126,
                      mx103,
                      mx64,
                      mx22,
                      mx90,
                      mx180,
                      mx141,
                      mx20,
                      mx59,
                      mx134,
                      mx1,
                      mx55,
                      mx113,
                      mx160,
                      mx0,
                      mx32,
                      mx85,
                      mx6,
                      mx76,
                      mx157,
                      mx45,
                      mx162,
                      mx138,
                      mx154,
                      mx53,
                      mx61,
                      mx177,
                      mx174,
                      mx136,
                      mx119,
                      mx185,
                      mx51,
                      mx47,
                      mx81,
                      mx73,
                      mx8,
                      mx110,
                      mx18,
                      mx169,
                      mx101,
                      mx69,
                      mx27,
                      mx123,
                      mx166,
                      mx121);
    m.add_instruction(pass_op{});
    auto mx195 = add_alloc(m, migraphx::shape{migraphx::shape::float_type, {2, 5}});
    m.add_instruction(pass_op{},
                      mx195,
                      mx115,
                      mx105,
                      mx98,
                      mx123,
                      mx27,
                      mx126,
                      mx103,
                      mx64,
                      mx183,
                      mx174,
                      mx136,
                      mx177,
                      mx141,
                      mx51,
                      mx93,
                      mx113,
                      mx38,
                      mx160,
                      mx55,
                      mx30,
                      mx61,
                      mx138,
                      mx53,
                      mx76,
                      mx85,
                      mx6,
                      mx20,
                      mx59,
                      mx0,
                      mx40,
                      mx43,
                      mx88,
                      mx79,
                      mx180,
                      mx90,
                      mx187,
                      mx81,
                      mx128,
                      mx157,
                      mx45,
                      mx162,
                      mx134,
                      mx1,
                      mx130,
                      mx147,
                      mx166,
                      mx121,
                      mx18,
                      mx169,
                      mx143,
                      mx119,
                      mx36,
                      mx185,
                      mx192);
    auto mx197 = m.add_instruction(pass_op{}, mx174);
    m.add_instruction(pass_op{},
                      mx197,
                      mx128,
                      mx150,
                      mx101,
                      mx69,
                      mx126,
                      mx103,
                      mx22,
                      mx166,
                      mx183,
                      mx136,
                      mx177,
                      mx141,
                      mx30,
                      mx73,
                      mx93,
                      mx38,
                      mx160,
                      mx55,
                      mx76,
                      mx32,
                      mx85,
                      mx6,
                      mx20,
                      mx59,
                      mx0,
                      mx43,
                      mx15,
                      mx88,
                      mx79,
                      mx180,
                      mx90,
                      mx67,
                      mx81,
                      mx138,
                      mx154,
                      mx53,
                      mx157,
                      mx45,
                      mx162,
                      mx51,
                      mx47,
                      mx195,
                      mx110,
                      mx8,
                      mx143,
                      mx147,
                      mx134,
                      mx1,
                      mx130,
                      mx115,
                      mx105,
                      mx40,
                      mx98,
                      mx36,
                      mx185,
                      mx192);
    m.add_instruction(pass_op{});
    m.add_instruction(pass_op{});
    auto mx201 = m.add_instruction(pass_op{}, mx195, mx197);
    m.add_instruction(pass_op{},
                      mx201,
                      mx115,
                      mx105,
                      mx98,
                      mx123,
                      mx27,
                      mx126,
                      mx103,
                      mx64,
                      mx183,
                      mx174,
                      mx136,
                      mx177,
                      mx141,
                      mx51,
                      mx93,
                      mx113,
                      mx38,
                      mx160,
                      mx55,
                      mx30,
                      mx61,
                      mx138,
                      mx53,
                      mx76,
                      mx85,
                      mx6,
                      mx20,
                      mx59,
                      mx0,
                      mx40,
                      mx43,
                      mx197,
                      mx88,
                      mx79,
                      mx180,
                      mx90,
                      mx187,
                      mx81,
                      mx128,
                      mx157,
                      mx45,
                      mx162,
                      mx134,
                      mx1,
                      mx130,
                      mx147,
                      mx166,
                      mx121,
                      mx18,
                      mx169,
                      mx143,
                      mx119,
                      mx36,
                      mx185,
                      mx192);
    auto mx203 = add_alloc(m, migraphx::shape{migraphx::shape::float_type, {2, 5}});
    m.add_instruction(pass_op{},
                      mx203,
                      mx115,
                      mx105,
                      mx98,
                      mx123,
                      mx27,
                      mx126,
                      mx103,
                      mx64,
                      mx183,
                      mx174,
                      mx136,
                      mx177,
                      mx141,
                      mx51,
                      mx93,
                      mx113,
                      mx38,
                      mx160,
                      mx55,
                      mx30,
                      mx61,
                      mx138,
                      mx53,
                      mx76,
                      mx85,
                      mx6,
                      mx20,
                      mx59,
                      mx0,
                      mx40,
                      mx43,
                      mx197,
                      mx88,
                      mx79,
                      mx180,
                      mx90,
                      mx187,
                      mx81,
                      mx128,
                      mx157,
                      mx45,
                      mx162,
                      mx134,
                      mx1,
                      mx130,
                      mx147,
                      mx166,
                      mx121,
                      mx18,
                      mx169,
                      mx143,
                      mx119,
                      mx36,
                      mx185,
                      mx192);
    auto mx205 = m.add_instruction(pass_op{}, mx203, mx201);
    m.add_instruction(pass_op{},
                      mx205,
                      mx115,
                      mx105,
                      mx98,
                      mx123,
                      mx27,
                      mx126,
                      mx103,
                      mx64,
                      mx183,
                      mx174,
                      mx136,
                      mx177,
                      mx141,
                      mx51,
                      mx93,
                      mx113,
                      mx38,
                      mx160,
                      mx55,
                      mx30,
                      mx61,
                      mx138,
                      mx53,
                      mx76,
                      mx85,
                      mx6,
                      mx20,
                      mx59,
                      mx0,
                      mx40,
                      mx43,
                      mx197,
                      mx88,
                      mx79,
                      mx180,
                      mx90,
                      mx187,
                      mx81,
                      mx128,
                      mx157,
                      mx45,
                      mx162,
                      mx134,
                      mx1,
                      mx130,
                      mx147,
                      mx166,
                      mx121,
                      mx18,
                      mx169,
                      mx143,
                      mx119,
                      mx36,
                      mx185,
                      mx192);
    m.add_instruction(pass_op{});
    auto mx208 = add_alloc(m, migraphx::shape{migraphx::shape::float_type, {2, 5}});
    m.add_instruction(pass_op{},
                      mx208,
                      mx30,
                      mx40,
                      mx64,
                      mx93,
                      mx18,
                      mx98,
                      mx115,
                      mx143,
                      mx38,
                      mx147,
                      mx183,
                      mx197,
                      mx150,
                      mx119,
                      mx32,
                      mx8,
                      mx105,
                      mx101,
                      mx110,
                      mx195,
                      mx47,
                      mx27,
                      mx22,
                      mx205,
                      mx121,
                      mx67,
                      mx187,
                      mx113,
                      mx73,
                      mx201,
                      mx130,
                      mx203,
                      mx169,
                      mx69,
                      mx15,
                      mx154,
                      mx61,
                      mx174,
                      mx123,
                      mx36,
                      mx192);
    auto mx210 = m.add_instruction(pass_op{}, mx180);
    m.add_instruction(pass_op{},
                      mx210,
                      mx143,
                      mx115,
                      mx18,
                      mx93,
                      mx150,
                      mx47,
                      mx187,
                      mx15,
                      mx169,
                      mx69,
                      mx205,
                      mx32,
                      mx119,
                      mx113,
                      mx73,
                      mx201,
                      mx30,
                      mx67,
                      mx121,
                      mx22,
                      mx27,
                      mx40,
                      mx98,
                      mx174,
                      mx61,
                      mx154,
                      mx64,
                      mx147,
                      mx38,
                      mx203,
                      mx130,
                      mx8,
                      mx110,
                      mx105,
                      mx101,
                      mx195,
                      mx183,
                      mx197,
                      mx123,
                      mx36,
                      mx192);
    auto mx212 = m.add_instruction(pass_op{}, mx174);
    m.add_instruction(pass_op{},
                      mx212,
                      mx32,
                      mx67,
                      mx90,
                      mx15,
                      mx138,
                      mx126,
                      mx103,
                      mx38,
                      mx136,
                      mx180,
                      mx141,
                      mx51,
                      mx30,
                      mx22,
                      mx201,
                      mx59,
                      mx134,
                      mx154,
                      mx150,
                      mx1,
                      mx160,
                      mx45,
                      mx6,
                      mx76,
                      mx88,
                      mx53,
                      mx47,
                      mx183,
                      mx81,
                      mx157,
                      mx93,
                      mx79,
                      mx85,
                      mx0,
                      mx210,
                      mx73,
                      mx8,
                      mx110,
                      mx20,
                      mx69,
                      mx177,
                      mx36,
                      mx143,
                      mx162,
                      mx147,
                      mx130,
                      mx115,
                      mx55,
                      mx105,
                      mx40,
                      mx98,
                      mx208,
                      mx203,
                      mx128,
                      mx205,
                      mx195,
                      mx101,
                      mx185,
                      mx43,
                      mx166,
                      mx192);
    m.add_instruction(pass_op{});
    m.add_instruction(pass_op{});
    auto mx216 = m.add_instruction(pass_op{}, mx208, mx212, mx210);
    m.add_instruction(pass_op{},
                      mx216,
                      mx121,
                      mx30,
                      mx64,
                      mx93,
                      mx123,
                      mx143,
                      mx119,
                      mx36,
                      mx150,
                      mx8,
                      mx101,
                      mx169,
                      mx147,
                      mx110,
                      mx27,
                      mx61,
                      mx40,
                      mx205,
                      mx115,
                      mx32,
                      mx69,
                      mx67,
                      mx98,
                      mx187,
                      mx195,
                      mx73,
                      mx105,
                      mx183,
                      mx197,
                      mx22,
                      mx113,
                      mx201,
                      mx47,
                      mx130,
                      mx154,
                      mx15,
                      mx212,
                      mx18,
                      mx174,
                      mx38,
                      mx203,
                      mx192);
    auto mx218 = add_alloc(m, migraphx::shape{migraphx::shape::float_type, {2, 5}});
    m.add_instruction(pass_op{},
                      mx218,
                      mx121,
                      mx30,
                      mx64,
                      mx93,
                      mx123,
                      mx143,
                      mx119,
                      mx36,
                      mx150,
                      mx8,
                      mx101,
                      mx169,
                      mx147,
                      mx110,
                      mx27,
                      mx61,
                      mx40,
                      mx205,
                      mx115,
                      mx32,
                      mx69,
                      mx67,
                      mx98,
                      mx187,
                      mx195,
                      mx73,
                      mx105,
                      mx183,
                      mx197,
                      mx22,
                      mx113,
                      mx201,
                      mx47,
                      mx130,
                      mx154,
                      mx15,
                      mx212,
                      mx18,
                      mx174,
                      mx38,
                      mx203,
                      mx192);
    auto mx220 = m.add_instruction(pass_op{}, mx218, mx216, mx166);
    m.add_instruction(pass_op{},
                      mx220,
                      mx121,
                      mx30,
                      mx64,
                      mx93,
                      mx123,
                      mx143,
                      mx119,
                      mx36,
                      mx150,
                      mx8,
                      mx101,
                      mx169,
                      mx147,
                      mx110,
                      mx27,
                      mx61,
                      mx40,
                      mx205,
                      mx115,
                      mx32,
                      mx69,
                      mx67,
                      mx98,
                      mx187,
                      mx195,
                      mx73,
                      mx105,
                      mx183,
                      mx197,
                      mx22,
                      mx113,
                      mx201,
                      mx47,
                      mx130,
                      mx154,
                      mx15,
                      mx212,
                      mx18,
                      mx174,
                      mx38,
                      mx203,
                      mx192);
    m.add_instruction(pass_op{});
    auto mx223 = m.add_instruction(pass_op{}, mx205, mx220, mx57, mx205);
    m.add_instruction(pass_op{},
                      mx223,
                      mx38,
                      mx192,
                      mx203,
                      mx130,
                      mx47,
                      mx143,
                      mx123,
                      mx169,
                      mx121,
                      mx147,
                      mx110,
                      mx27,
                      mx36,
                      mx150,
                      mx119,
                      mx101,
                      mx8,
                      mx64,
                      mx61,
                      mx115,
                      mx32,
                      mx69,
                      mx67,
                      mx98,
                      mx187,
                      mx195,
                      mx73,
                      mx105,
                      mx183,
                      mx197,
                      mx22,
                      mx113,
                      mx201,
                      mx174,
                      mx18,
                      mx93,
                      mx205,
                      mx40,
                      mx30,
                      mx154,
                      mx15,
                      mx212);
    auto mx225 = add_alloc(m, migraphx::shape{migraphx::shape::float_type, {2, 5}});
    m.add_instruction(pass_op{},
                      mx225,
                      mx45,
                      mx59,
                      mx76,
                      mx90,
                      mx218,
                      mx67,
                      mx126,
                      mx103,
                      mx136,
                      mx138,
                      mx15,
                      mx32,
                      mx1,
                      mx160,
                      mx150,
                      mx110,
                      mx51,
                      mx30,
                      mx6,
                      mx157,
                      mx93,
                      mx79,
                      mx85,
                      mx88,
                      mx53,
                      mx154,
                      mx134,
                      mx141,
                      mx180,
                      mx38,
                      mx81,
                      mx223,
                      mx183,
                      mx220,
                      mx210,
                      mx0,
                      mx208,
                      mx20,
                      mx69,
                      mx73,
                      mx185,
                      mx101,
                      mx201,
                      mx22,
                      mx203,
                      mx47,
                      mx128,
                      mx205,
                      mx195,
                      mx8,
                      mx177,
                      mx36,
                      mx55,
                      mx216,
                      mx105,
                      mx115,
                      mx130,
                      mx40,
                      mx98,
                      mx43,
                      mx166,
                      mx192,
                      mx162,
                      mx147,
                      mx143);
    m.add_instruction(pass_op{});
    m.add_instruction(pass_op{});
    auto mx229 = m.add_instruction(pass_op{}, mx225, mx69, mx192);
    m.add_instruction(pass_op{},
                      mx229,
                      mx45,
                      mx59,
                      mx76,
                      mx90,
                      mx218,
                      mx67,
                      mx126,
                      mx103,
                      mx136,
                      mx138,
                      mx15,
                      mx32,
                      mx1,
                      mx160,
                      mx150,
                      mx110,
                      mx51,
                      mx30,
                      mx6,
                      mx157,
                      mx93,
                      mx79,
                      mx85,
                      mx88,
                      mx53,
                      mx154,
                      mx134,
                      mx141,
                      mx180,
                      mx38,
                      mx81,
                      mx223,
                      mx183,
                      mx220,
                      mx210,
                      mx0,
                      mx208,
                      mx20,
                      mx69,
                      mx73,
                      mx185,
                      mx101,
                      mx201,
                      mx22,
                      mx203,
                      mx47,
                      mx128,
                      mx205,
                      mx195,
                      mx8,
                      mx177,
                      mx36,
                      mx55,
                      mx216,
                      mx105,
                      mx115,
                      mx130,
                      mx40,
                      mx98,
                      mx43,
                      mx166,
                      mx192,
                      mx162,
                      mx147,
                      mx143);
    m.add_instruction(pass_op{});
    auto mx232 = add_alloc(m, migraphx::shape{migraphx::shape::float_type, {2, 5}});
    m.add_instruction(pass_op{},
                      mx232,
                      mx160,
                      mx154,
                      mx76,
                      mx43,
                      mx67,
                      mx55,
                      mx187,
                      mx88,
                      mx126,
                      mx197,
                      mx225,
                      mx136,
                      mx59,
                      mx64,
                      mx15,
                      mx212,
                      mx128,
                      mx32,
                      mx218,
                      mx150,
                      mx216,
                      mx110,
                      mx169,
                      mx103,
                      mx113,
                      mx141,
                      mx79,
                      mx223,
                      mx90,
                      mx6,
                      mx18,
                      mx138,
                      mx210,
                      mx85,
                      mx53,
                      mx61,
                      mx45,
                      mx134,
                      mx119,
                      mx180,
                      mx166,
                      mx20,
                      mx0,
                      mx177,
                      mx81,
                      mx208,
                      mx157,
                      mx185,
                      mx1,
                      mx69,
                      mx201,
                      mx174,
                      mx101,
                      mx51,
                      mx22,
                      mx162,
                      mx220,
                      mx203,
                      mx47,
                      mx195,
                      mx73,
                      mx27,
                      mx205,
                      mx229,
                      mx8,
                      mx123,
                      mx121);
    m.add_instruction(pass_op{});
    m.add_instruction(pass_op{});
    auto mx236 = m.add_instruction(pass_op{}, mx232, mx192, mx166);
    m.add_instruction(pass_op{},
                      mx236,
                      mx160,
                      mx154,
                      mx76,
                      mx43,
                      mx67,
                      mx55,
                      mx187,
                      mx88,
                      mx126,
                      mx197,
                      mx225,
                      mx136,
                      mx59,
                      mx64,
                      mx15,
                      mx212,
                      mx128,
                      mx32,
                      mx218,
                      mx150,
                      mx216,
                      mx110,
                      mx169,
                      mx103,
                      mx113,
                      mx141,
                      mx79,
                      mx223,
                      mx90,
                      mx6,
                      mx18,
                      mx138,
                      mx210,
                      mx85,
                      mx53,
                      mx61,
                      mx45,
                      mx134,
                      mx119,
                      mx180,
                      mx166,
                      mx20,
                      mx0,
                      mx177,
                      mx81,
                      mx208,
                      mx157,
                      mx185,
                      mx1,
                      mx69,
                      mx201,
                      mx174,
                      mx101,
                      mx51,
                      mx22,
                      mx162,
                      mx220,
                      mx203,
                      mx47,
                      mx195,
                      mx73,
                      mx27,
                      mx205,
                      mx229,
                      mx8,
                      mx123,
                      mx121);
    m.add_instruction(pass_op{});
    auto mx239 = add_alloc(m, migraphx::shape{migraphx::shape::float_type, {2, 5}});
    m.add_instruction(pass_op{},
                      mx239,
                      mx38,
                      mx192,
                      mx232,
                      mx203,
                      mx229,
                      mx183,
                      mx154,
                      mx201,
                      mx113,
                      mx174,
                      mx110,
                      mx197,
                      mx36,
                      mx115,
                      mx150,
                      mx98,
                      mx130,
                      mx32,
                      mx101,
                      mx169,
                      mx8,
                      mx64,
                      mx27,
                      mx225,
                      mx22,
                      mx147,
                      mx67,
                      mx205,
                      mx73,
                      mx61,
                      mx105,
                      mx18,
                      mx47,
                      mx123,
                      mx93,
                      mx195,
                      mx119,
                      mx69,
                      mx40,
                      mx187,
                      mx30,
                      mx15,
                      mx143,
                      mx236,
                      mx121,
                      mx212);
    m.add_instruction(pass_op{});
    auto mx242 = m.add_instruction(pass_op{}, mx239, mx223);
    m.add_instruction(pass_op{},
                      mx242,
                      mx38,
                      mx192,
                      mx232,
                      mx203,
                      mx229,
                      mx183,
                      mx154,
                      mx201,
                      mx113,
                      mx174,
                      mx110,
                      mx197,
                      mx36,
                      mx115,
                      mx150,
                      mx98,
                      mx130,
                      mx32,
                      mx101,
                      mx169,
                      mx8,
                      mx64,
                      mx27,
                      mx225,
                      mx22,
                      mx147,
                      mx67,
                      mx205,
                      mx73,
                      mx61,
                      mx105,
                      mx18,
                      mx47,
                      mx123,
                      mx93,
                      mx195,
                      mx119,
                      mx69,
                      mx40,
                      mx187,
                      mx30,
                      mx15,
                      mx143,
                      mx236,
                      mx121,
                      mx212);
    auto mx244 = add_alloc(m, migraphx::shape{migraphx::shape::float_type, {2, 5}});
    m.add_instruction(pass_op{});
    m.add_instruction(pass_op{});
    auto mx247 = m.add_instruction(pass_op{}, mx244, mx229, mx242, mx236);
    auto moutput =
        m.add_parameter("output", migraphx::shape{migraphx::shape::float_type, {3, 1, 2, 5}});
    auto mx248 = m.add_instruction(pass_op{}, mx247);
    auto mx249 = m.add_instruction(pass_op{}, mx166);
    auto mx250 = m.add_instruction(pass_op{}, mx85);
    m.add_instruction(pass_op{}, moutput, mx250, mx249, mx248);

    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 1824); // Optimal is 1600
    CHECK(no_allocate(m));
    CHECK(is_disjoint({mx0, mx8}));
    CHECK(is_disjoint({mx0, mx8}));
    CHECK(is_disjoint({mx0, mx18, mx8}));
    CHECK(is_disjoint({mx0, mx18, mx8}));
    CHECK(is_disjoint({mx0, mx18, mx8}));
    CHECK(is_disjoint({mx0, mx18, mx8}));
    CHECK(is_disjoint({mx0, mx18, mx8}));
    CHECK(is_disjoint({mx0, mx18, mx30, mx8}));
    CHECK(is_disjoint({mx0, mx18, mx30, mx8}));
    CHECK(is_disjoint({mx30, mx8}));
    CHECK(is_disjoint({mx0, mx18, mx30, mx8}));
    CHECK(is_disjoint({mx0, mx18, mx38, mx8}));
    CHECK(is_disjoint({mx30, mx38}));
    CHECK(is_disjoint({mx0, mx18, mx38, mx8}));
    CHECK(is_disjoint({mx18, mx30, mx38, mx43, mx8}));
    CHECK(is_disjoint({mx0, mx18, mx30, mx38, mx8}));
    CHECK(is_disjoint({mx0, mx18, mx30, mx38, mx43, mx8}));
    CHECK(is_disjoint({mx0, mx43, mx8}));
    CHECK(is_disjoint({mx18, mx30, mx38, mx43, mx8}));
    CHECK(is_disjoint({mx18, mx30, mx38, mx53, mx8}));
    CHECK(is_disjoint({mx43, mx53}));
    CHECK(is_disjoint({mx18, mx30, mx38, mx53, mx8}));
    CHECK(is_disjoint({mx38, mx53}));
    CHECK(is_disjoint({mx18, mx30, mx38, mx8}));
    CHECK(is_disjoint({mx0, mx30, mx38, mx43, mx53, mx61, mx8}));
    CHECK(is_disjoint({mx18, mx61}));
    CHECK(is_disjoint({mx0, mx30, mx38, mx43, mx53, mx61, mx8}));
    CHECK(is_disjoint({mx0, mx18, mx30, mx38, mx43, mx53, mx61, mx67}));
    CHECK(is_disjoint({mx0, mx18, mx30, mx38, mx43, mx53, mx61}));
    CHECK(is_disjoint({mx18, mx67}));
    CHECK(is_disjoint({mx0, mx18, mx30, mx38, mx43, mx53, mx61, mx67}));
    CHECK(is_disjoint({mx18, mx30, mx38, mx61, mx67, mx76, mx8}));
    CHECK(is_disjoint({mx38, mx76}));
    CHECK(is_disjoint({mx18, mx30, mx38, mx61, mx67, mx76, mx8}));
    CHECK(is_disjoint({mx18, mx30, mx38, mx61, mx67, mx8, mx81}));
    CHECK(is_disjoint({mx61, mx67, mx76, mx81}));
    CHECK(is_disjoint({mx18, mx30, mx38, mx61, mx67, mx8, mx81}));
    CHECK(is_disjoint({mx18, mx30, mx38, mx61, mx67, mx8, mx88}));
    CHECK(is_disjoint({mx81, mx88}));
    CHECK(is_disjoint({mx18, mx30, mx38, mx61, mx67, mx8, mx88}));
    CHECK(is_disjoint({mx0, mx18, mx38, mx43, mx53, mx61, mx67, mx76, mx8, mx81, mx88, mx93}));
    CHECK(is_disjoint({mx0, mx18, mx38, mx43, mx53, mx61, mx67, mx76, mx8, mx81, mx88, mx93}));
    CHECK(is_disjoint({mx0, mx101, mx18, mx30, mx38, mx43, mx53, mx61, mx76, mx81, mx88, mx93}));
    CHECK(is_disjoint({mx101, mx18, mx30, mx38, mx61, mx67, mx8, mx88, mx93}));
    CHECK(
        is_disjoint({mx0, mx101, mx18, mx38, mx43, mx53, mx61, mx67, mx76, mx8, mx81, mx88, mx93}));
    CHECK(is_disjoint({mx101, mx88, mx93}));
    CHECK(is_disjoint({mx0, mx101, mx18, mx30, mx38, mx43, mx53, mx61, mx76, mx81, mx88, mx93}));
    CHECK(is_disjoint(
        {mx0, mx101, mx113, mx30, mx38, mx43, mx53, mx67, mx76, mx8, mx81, mx88, mx93}));
    CHECK(is_disjoint(
        {mx0, mx101, mx113, mx18, mx38, mx43, mx53, mx61, mx67, mx76, mx8, mx81, mx88, mx93}));
    CHECK(is_disjoint({mx113, mx93}));
    CHECK(is_disjoint(
        {mx0, mx101, mx113, mx30, mx38, mx43, mx53, mx67, mx76, mx8, mx81, mx88, mx93}));
    CHECK(is_disjoint(
        {mx0, mx101, mx121, mx30, mx38, mx43, mx53, mx67, mx76, mx8, mx81, mx88, mx93}));
    CHECK(is_disjoint({mx113, mx121}));
    CHECK(is_disjoint(
        {mx0, mx101, mx121, mx30, mx38, mx43, mx53, mx67, mx76, mx8, mx81, mx88, mx93}));
    CHECK(is_disjoint({mx101, mx113, mx121, mx126, mx18, mx30, mx38, mx61, mx67, mx8, mx93}));
    CHECK(is_disjoint({mx101, mx113, mx121, mx18, mx30, mx38, mx61, mx67, mx8, mx88, mx93}));
    CHECK(is_disjoint({mx0,
                       mx101,
                       mx113,
                       mx121,
                       mx126,
                       mx18,
                       mx38,
                       mx43,
                       mx53,
                       mx61,
                       mx67,
                       mx76,
                       mx8,
                       mx81,
                       mx88,
                       mx93}));
    CHECK(is_disjoint({mx126, mx88, mx93}));
    CHECK(is_disjoint({mx101, mx113, mx121, mx126, mx18, mx30, mx38, mx61, mx67, mx8, mx93}));
    CHECK(is_disjoint({mx101, mx113, mx121, mx136, mx18, mx30, mx38, mx61, mx67, mx8, mx93}));
    CHECK(is_disjoint({mx126, mx136, mx81}));
    CHECK(is_disjoint({mx101, mx113, mx121, mx136, mx18, mx30, mx38, mx61, mx67, mx8, mx93}));
    CHECK(is_disjoint({mx121, mx136}));
    CHECK(is_disjoint({mx101, mx113, mx121, mx18, mx30, mx38, mx61, mx67, mx8, mx93}));
    CHECK(is_disjoint({mx0,
                       mx101,
                       mx113,
                       mx121,
                       mx126,
                       mx136,
                       mx143,
                       mx18,
                       mx38,
                       mx43,
                       mx53,
                       mx61,
                       mx67,
                       mx76,
                       mx8,
                       mx81,
                       mx88}));
    CHECK(is_disjoint({mx101, mx143}));
    CHECK(is_disjoint({mx0,
                       mx101,
                       mx113,
                       mx121,
                       mx126,
                       mx136,
                       mx143,
                       mx18,
                       mx38,
                       mx43,
                       mx53,
                       mx61,
                       mx67,
                       mx76,
                       mx8,
                       mx81,
                       mx88}));
    CHECK(is_disjoint({mx0,
                       mx113,
                       mx121,
                       mx126,
                       mx136,
                       mx143,
                       mx150,
                       mx18,
                       mx30,
                       mx38,
                       mx43,
                       mx53,
                       mx61,
                       mx76,
                       mx81,
                       mx88,
                       mx93}));
    CHECK(is_disjoint({mx101, mx150, mx81}));
    CHECK(is_disjoint({mx0,
                       mx113,
                       mx121,
                       mx126,
                       mx136,
                       mx143,
                       mx150,
                       mx18,
                       mx30,
                       mx38,
                       mx43,
                       mx53,
                       mx61,
                       mx76,
                       mx81,
                       mx88,
                       mx93}));
    CHECK(is_disjoint(
        {mx101, mx113, mx121, mx143, mx150, mx157, mx18, mx30, mx38, mx61, mx67, mx8, mx93}));
    CHECK(is_disjoint({mx121, mx157}));
    CHECK(is_disjoint(
        {mx101, mx113, mx121, mx143, mx150, mx157, mx18, mx30, mx38, mx61, mx67, mx8, mx93}));
    CHECK(is_disjoint(
        {mx101, mx113, mx121, mx143, mx150, mx162, mx18, mx30, mx38, mx61, mx67, mx8, mx93}));
    CHECK(is_disjoint({mx143, mx150, mx157, mx162}));
    CHECK(is_disjoint(
        {mx101, mx113, mx121, mx143, mx150, mx162, mx18, mx30, mx38, mx61, mx67, mx8, mx93}));
    CHECK(is_disjoint({mx0,  mx101, mx121, mx126, mx136, mx143, mx150, mx157, mx162, mx169,
                       mx30, mx38,  mx43,  mx53,  mx67,  mx76,  mx8,   mx81,  mx88,  mx93}));
    CHECK(is_disjoint({mx0,  mx101, mx121, mx126, mx136, mx143, mx150, mx157, mx162, mx169,
                       mx30, mx38,  mx43,  mx53,  mx67,  mx76,  mx8,   mx81,  mx88,  mx93}));
    CHECK(is_disjoint({mx101,
                       mx113,
                       mx121,
                       mx143,
                       mx150,
                       mx169,
                       mx177,
                       mx18,
                       mx30,
                       mx38,
                       mx61,
                       mx67,
                       mx8,
                       mx93}));
    CHECK(is_disjoint({mx162, mx177}));
    CHECK(is_disjoint({mx101,
                       mx113,
                       mx121,
                       mx143,
                       mx150,
                       mx169,
                       mx177,
                       mx18,
                       mx30,
                       mx38,
                       mx61,
                       mx67,
                       mx8,
                       mx93}));
    CHECK(is_disjoint({mx0,  mx101, mx113, mx121, mx126, mx136, mx150, mx157, mx162, mx169, mx177,
                       mx18, mx183, mx38,  mx43,  mx53,  mx61,  mx67,  mx76,  mx8,   mx81,  mx88}));
    CHECK(is_disjoint({mx101,
                       mx113,
                       mx121,
                       mx143,
                       mx150,
                       mx169,
                       mx177,
                       mx18,
                       mx183,
                       mx30,
                       mx38,
                       mx61,
                       mx67,
                       mx8,
                       mx93}));
    CHECK(
        is_disjoint({mx0,   mx101, mx121, mx126, mx136, mx143, mx150, mx157, mx162, mx169, mx177,
                     mx183, mx30,  mx38,  mx43,  mx53,  mx67,  mx76,  mx8,   mx81,  mx88,  mx93}));
    CHECK(is_disjoint({mx169, mx177, mx183}));
    CHECK(is_disjoint({mx0,  mx101, mx113, mx121, mx126, mx136, mx150, mx157, mx162, mx169, mx177,
                       mx18, mx183, mx38,  mx43,  mx53,  mx61,  mx67,  mx76,  mx8,   mx81,  mx88}));
    CHECK(
        is_disjoint({mx0,   mx113, mx121, mx126, mx136, mx143, mx157, mx162, mx169, mx177, mx18,
                     mx183, mx195, mx30,  mx38,  mx43,  mx53,  mx61,  mx76,  mx81,  mx88,  mx93}));
    CHECK(is_disjoint({mx0,   mx101, mx121, mx126, mx136, mx143, mx150, mx157,
                       mx162, mx169, mx177, mx183, mx195, mx30,  mx38,  mx43,
                       mx53,  mx67,  mx76,  mx8,   mx81,  mx88,  mx93}));
    CHECK(is_disjoint({mx169, mx195}));
    CHECK(
        is_disjoint({mx0,   mx113, mx121, mx126, mx136, mx143, mx157, mx162, mx169, mx177, mx18,
                     mx183, mx195, mx30,  mx38,  mx43,  mx53,  mx61,  mx76,  mx81,  mx88,  mx93}));
    CHECK(
        is_disjoint({mx0,   mx113, mx121, mx126, mx136, mx143, mx157, mx162, mx169, mx177, mx18,
                     mx183, mx203, mx30,  mx38,  mx43,  mx53,  mx61,  mx76,  mx81,  mx88,  mx93}));
    CHECK(is_disjoint({mx195, mx203}));
    CHECK(
        is_disjoint({mx0,   mx113, mx121, mx126, mx136, mx143, mx157, mx162, mx169, mx177, mx18,
                     mx183, mx203, mx30,  mx38,  mx43,  mx53,  mx61,  mx76,  mx81,  mx88,  mx93}));
    CHECK(is_disjoint({mx101,
                       mx113,
                       mx121,
                       mx143,
                       mx150,
                       mx169,
                       mx18,
                       mx183,
                       mx195,
                       mx203,
                       mx208,
                       mx30,
                       mx38,
                       mx61,
                       mx67,
                       mx8,
                       mx93}));
    CHECK(is_disjoint({mx101,
                       mx113,
                       mx121,
                       mx143,
                       mx150,
                       mx169,
                       mx177,
                       mx18,
                       mx183,
                       mx195,
                       mx203,
                       mx30,
                       mx38,
                       mx61,
                       mx67,
                       mx8,
                       mx93}));
    CHECK(is_disjoint({mx0,   mx101, mx121, mx126, mx136, mx143, mx150, mx157, mx162,
                       mx169, mx177, mx183, mx195, mx203, mx208, mx30,  mx38,  mx43,
                       mx53,  mx67,  mx76,  mx8,   mx81,  mx88,  mx93}));
    CHECK(is_disjoint({mx169, mx177, mx208}));
    CHECK(is_disjoint({mx101,
                       mx113,
                       mx121,
                       mx143,
                       mx150,
                       mx169,
                       mx18,
                       mx183,
                       mx195,
                       mx203,
                       mx208,
                       mx30,
                       mx38,
                       mx61,
                       mx67,
                       mx8,
                       mx93}));
    CHECK(is_disjoint({mx101,
                       mx113,
                       mx121,
                       mx143,
                       mx150,
                       mx169,
                       mx18,
                       mx183,
                       mx195,
                       mx203,
                       mx218,
                       mx30,
                       mx38,
                       mx61,
                       mx67,
                       mx8,
                       mx93}));
    CHECK(is_disjoint({mx162, mx208, mx218}));
    CHECK(is_disjoint({mx101,
                       mx113,
                       mx121,
                       mx143,
                       mx150,
                       mx169,
                       mx18,
                       mx183,
                       mx195,
                       mx203,
                       mx218,
                       mx30,
                       mx38,
                       mx61,
                       mx67,
                       mx8,
                       mx93}));
    CHECK(is_disjoint({mx203, mx218}));
    CHECK(is_disjoint({mx101,
                       mx113,
                       mx121,
                       mx143,
                       mx150,
                       mx169,
                       mx18,
                       mx183,
                       mx195,
                       mx203,
                       mx30,
                       mx38,
                       mx61,
                       mx67,
                       mx8,
                       mx93}));
    CHECK(is_disjoint({mx0,   mx101, mx121, mx126, mx136, mx143, mx150, mx157, mx162,
                       mx177, mx183, mx195, mx203, mx208, mx218, mx225, mx30,  mx38,
                       mx43,  mx53,  mx67,  mx76,  mx8,   mx81,  mx88,  mx93}));
    CHECK(is_disjoint({mx183, mx225}));
    CHECK(is_disjoint({mx0,   mx101, mx121, mx126, mx136, mx143, mx150, mx157, mx162,
                       mx177, mx183, mx195, mx203, mx208, mx218, mx225, mx30,  mx38,
                       mx43,  mx53,  mx67,  mx76,  mx8,   mx81,  mx88,  mx93}));
    CHECK(is_disjoint({mx0,   mx101, mx113, mx121, mx126, mx136, mx150, mx157, mx162,
                       mx169, mx177, mx18,  mx195, mx203, mx208, mx218, mx225, mx232,
                       mx38,  mx43,  mx53,  mx61,  mx67,  mx76,  mx8,   mx81,  mx88}));
    CHECK(is_disjoint({mx162, mx183, mx232}));
    CHECK(is_disjoint({mx0,   mx101, mx113, mx121, mx126, mx136, mx150, mx157, mx162,
                       mx169, mx177, mx18,  mx195, mx203, mx208, mx218, mx225, mx232,
                       mx38,  mx43,  mx53,  mx61,  mx67,  mx76,  mx8,   mx81,  mx88}));
    CHECK(is_disjoint({mx101,
                       mx113,
                       mx121,
                       mx143,
                       mx150,
                       mx169,
                       mx18,
                       mx183,
                       mx195,
                       mx203,
                       mx225,
                       mx232,
                       mx239,
                       mx30,
                       mx38,
                       mx61,
                       mx67,
                       mx8,
                       mx93}));
    CHECK(is_disjoint({mx203, mx239}));
    CHECK(is_disjoint({mx101,
                       mx113,
                       mx121,
                       mx143,
                       mx150,
                       mx169,
                       mx18,
                       mx183,
                       mx195,
                       mx203,
                       mx225,
                       mx232,
                       mx239,
                       mx30,
                       mx38,
                       mx61,
                       mx67,
                       mx8,
                       mx93}));
    CHECK(is_disjoint({mx225, mx232, mx239, mx244}));
    CHECK(is_disjoint({mx162, mx244, mx81}));
}

TEST_CASE(literal_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto lit = generate_literal(migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
    mm->add_literal(lit);
    run_pass(*mm);
    auto result = p.eval({}).back();
    CHECK(lit == result);
}

TEST_CASE(test_tuple)
{
    migraphx::module m;

    auto s1 = migraphx::shape{migraphx::shape::float_type, {8}};
    auto s2 = migraphx::shape{migraphx::shape::half_type, {10}};

    auto s = migraphx::shape{{s1, s2}};

    auto a1 = add_alloc(m, s);
    auto m1 = m.add_instruction(pass_op{}, a1);
    auto a2 = add_alloc(m, {migraphx::shape::float_type, {4}});
    m.add_instruction(pass_op{}, a2, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 68);
    CHECK(no_allocate(m));
    CHECK(is_disjoint({a1, a2}));
}

TEST_CASE(test_large_offsets)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {10000000000}});
    auto m1 = m.add_instruction(pass_op{}, a1);
    auto a2 = add_alloc(m, {migraphx::shape::float_type, {10000000000}});
    m.add_instruction(pass_op{}, a2, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 80000000000);
    CHECK(no_allocate(m));
    CHECK(is_disjoint({a1, a2}));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
